"""Pure-DuckDB/Arrow local parity runner for Bronze->Silver->Gold when Docker/JVM Spark is unavailable on Windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from intel_core.schemas import (
    canonicalize_brand_id,
    compute_asset_id,
    compute_media_hash_key,
    convert_to_usd,
    iso_week_start,
    normalize_country_iso,
)
from intel_core.storage import (
    HumanOverrideStore,
    is_rejection_value,
    read_parquet_table,
    resolve_brand_override_decision,
    write_parquet_table,
)

from intel_etl_spark.jobs.bronze_to_silver import _sha256_file, load_manifest, save_manifest


def run_local_bronze_to_silver(
    *,
    bronze_occurrences_path: str | Path,
    silver_dir: str | Path,
    manifest_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Local DuckDB/Arrow equivalent of Spark Job 1 (`bronze_to_silver`) for Windows dev without Docker."""
    bronze_path = Path(bronze_occurrences_path)
    silver_root = Path(silver_dir)
    silver_root.mkdir(parents=True, exist_ok=True)
    m_path = Path(manifest_path) if manifest_path else (silver_root / "_bronze_manifest.json")
    prev_manifest = load_manifest(m_path)

    files = [bronze_path] if bronze_path.is_file() else sorted(bronze_path.rglob("*.parquet"))
    checksums = {p.name: _sha256_file(p) for p in files}
    if not force and all(prev_manifest.get(k) == v for k, v in checksums.items()):
        return {
            "status": "SKIPPED_UNCHANGED",
            "processed_files": 0,
            "skipped_files": len(files),
            "silver_rows": 0,
            "asset_rows": 0,
        }

    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df["week_start"] = df["event_date"].map(iso_week_start)
    df["country_iso"] = df["country_raw"].map(normalize_country_iso)
    df["brand_id_canon"] = df["brand_raw"].map(lambda b: canonicalize_brand_id(b) or str(b))
    df["asset_id"] = [
        str(a) if (pd.notna(a) and str(a).strip()) else compute_asset_id(str(u))
        for a, u in zip(df["asset_id"], df["media_uri"], strict=True)
    ]
    df["media_hash_key"] = [
        compute_media_hash_key(None if pd.isna(p) else str(p), aid)
        for p, aid in zip(df["phash"], df["asset_id"], strict=True)
    ]
    df["raw_spend_usd"] = [
        convert_to_usd(float(amt), str(cur), wk)
        for amt, cur, wk in zip(df["raw_spend_local"], df["currency"], df["week_start"], strict=True)
    ]
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["event_date_parsed"] = pd.to_datetime(df["event_date"]).dt.date

    # Deduplicate by occurrence_id keeping latest snapshot_ts
    df = df.sort_values("snapshot_ts", ascending=False).drop_duplicates(subset=["occurrence_id"]).copy()

    # Compute eu_total_reach weekly deltas per (asset_id, country_iso)
    df = df.sort_values(["asset_id", "country_iso", "week_start", "snapshot_ts"]).reset_index(drop=True)
    prev_reach = df.groupby(["asset_id", "country_iso"])["raw_eu_total_reach"].shift(1)
    reach_est = []
    for raw_r, prev_r, orig_est in zip(
        df["raw_eu_total_reach"], prev_reach, df["raw_reach_est"], strict=True
    ):
        if pd.notna(raw_r) and pd.notna(prev_r):
            reach_est.append(max(0, int(raw_r) - int(prev_r)))
        elif pd.notna(raw_r) and pd.isna(orig_est):
            reach_est.append(int(raw_r))
        else:
            reach_est.append(int(orig_est))
    df["raw_reach_est"] = reach_est

    silver_raw = pd.DataFrame(
        {
            "occurrence_id": df["occurrence_id"].astype(str),
            "week_start": df["week_start"],
            "country_iso": df["country_iso"].astype(str),
            "asset_id": df["asset_id"].astype(str),
            "media_hash_key": df["media_hash_key"].astype(str),
            "channel_group": df["channel_group"].astype(str),
            "publisher_name": df["publisher_name"].astype(str),
            "campaign_id": df["campaign_id"].astype(str),
            "candidate_brand_ids": [
                list(x) if isinstance(x, (list, tuple)) or hasattr(x, "tolist") else [b]
                for x, b in zip(df["candidate_brand_ids"], df["brand_id_canon"], strict=True)
            ],
            "raw_impressions": df["raw_impressions"].astype("int64"),
            "raw_reach_est": df["raw_reach_est"].astype("int64"),
            "raw_eu_total_reach": df["raw_eu_total_reach"].astype("Int64"),
            "snapshot_ts": df["snapshot_ts"],
            "is_weekly_interpolated": df["is_weekly_interpolated"].astype(bool),
            "raw_spend_local": df["raw_spend_local"].astype(float),
            "currency": df["currency"].astype(str),
            "raw_spend_usd": df["raw_spend_usd"].astype(float),
            "is_spend_modeled": df["is_spend_modeled"].astype(bool),
        }
    )

    dim_asset = (
        df.groupby("asset_id", as_index=False)
        .agg(
            phash=("phash", "first"),
            media_hash_key=("media_hash_key", "first"),
            brand_id_raw=("brand_id_canon", "first"),
            source_connector=("source_connector", "first"),
            media_type=("media_type", "first"),
            media_uri=("media_uri", "first"),
            aspect_ratio=("aspect_ratio", "first"),
            duration_sec=("duration_sec", "first"),
            first_seen_date=("event_date_parsed", "min"),
            last_seen_date=("event_date_parsed", "max"),
        )
    )
    dim_asset["is_active"] = True

    silver_raw_path = silver_root / "silver_ad_delivery_raw"
    dim_asset_path = silver_root / "dim_creative_asset"
    write_parquet_table(silver_raw, silver_raw_path, table_name="silver_ad_delivery_raw", partition_cols=["country_iso"])
    write_parquet_table(dim_asset, dim_asset_path, table_name="dim_creative_asset")

    save_manifest(m_path, {**prev_manifest, **checksums})
    return {
        "status": "PROCESSED",
        "processed_files": len(files),
        "skipped_files": 0,
        "silver_rows": int(len(silver_raw)),
        "asset_rows": int(len(dim_asset)),
        "silver_raw_path": str(silver_raw_path),
        "dim_asset_path": str(dim_asset_path),
    }


def run_local_silver_to_gold(
    *,
    silver_dir: str | Path,
    gold_dir: str | Path,
    outcomes_path: str | Path,
    overrides_dir: str | Path | None = None,
    dissection_path: str | Path | None = None,
    themes_path: str | Path | None = None,
) -> dict[str, Any]:
    """Local DuckDB/Arrow equivalent of Spark Job 2 (`silver_to_gold`) preserving 1/K & conservation invariants."""
    silver_root = Path(silver_dir)
    gold_root = Path(gold_dir)
    gold_root.mkdir(parents=True, exist_ok=True)

    raw_df = read_parquet_table(silver_root / "silver_ad_delivery_raw")
    assets_df = read_parquet_table(silver_root / "dim_creative_asset")
    outcomes_df = read_parquet_table(outcomes_path)

    diss_p = Path(dissection_path) if dissection_path else (silver_root / "dim_creative_dissection")
    diss_df = read_parquet_table(diss_p) if diss_p.exists() else pd.DataFrame()
    thm_p = Path(themes_path) if themes_path else (silver_root / "dim_creative_themes")
    themes_df = read_parquet_table(thm_p) if thm_p.exists() else pd.DataFrame()

    # Build dissection lookup by media_hash_key (is_active_version == True)
    diss_map: dict[str, dict[str, Any]] = {}
    if not diss_df.empty:
        act = diss_df[diss_df["is_active_version"].astype(bool)].sort_values(
            ["confidence", "prompt_version"], ascending=[False, False]
        )
        for row in act.drop_duplicates(subset=["media_hash_key"]).to_dict(orient="records"):
            diss_map[str(row["media_hash_key"])] = row

    # Build override precedence lookup: asset_id (1) > media_hash_key (2) > theme_label (3), latest created_at wins
    overrides = HumanOverrideStore(overrides_dir).list_overrides() if overrides_dir else []
    asset_ov: dict[tuple[str, str], tuple[str, int, Any]] = {}
    hash_ov: dict[tuple[str, str], tuple[str, int, Any]] = {}
    theme_ov: dict[tuple[str, str], tuple[str, int, Any]] = {}
    # list_overrides() returns created_at DESC, so first seen per key wins!
    for ov in overrides:
        k = (ov.target_key_value, ov.field_name)
        if ov.target_key_type == "asset_id" and k not in asset_ov:
            asset_ov[k] = (ov.human_value, 1, ov.created_at)
        elif ov.target_key_type == "media_hash_key" and k not in hash_ov:
            hash_ov[k] = (ov.human_value, 2, ov.created_at)
        elif ov.target_key_type == "theme_label":
            tk = (ov.target_key_value.strip().lower(), ov.field_name)
            if tk not in theme_ov:
                theme_ov[tk] = (ov.human_value, 3, ov.created_at)

    fct_rows: list[dict[str, Any]] = []
    raw_sum = float(raw_df["raw_spend_usd"].sum())
    rej_sum = 0.0
    quality_by_occ: dict[str, float] = {}

    for row in raw_df.to_dict(orient="records"):
        aid = str(row["asset_id"])
        mkey = str(row["media_hash_key"])
        ai = diss_map.get(mkey)

        ov_bel = asset_ov.get((aid, "belongs_to_brand")) or hash_ov.get((mkey, "belongs_to_brand"))
        ov_mat = asset_ov.get((aid, "matched_brand_id")) or hash_ov.get((mkey, "matched_brand_id"))
        ai_belongs = bool(ai["belongs_to_brand"]) if ai is not None else None
        ai_matched = ai.get("matched_brand_id") if ai is not None else None

        belongs, matched_raw, has_ov = resolve_brand_override_decision(
            ai_belongs=ai_belongs,
            ai_matched=ai_matched,
            ov_belongs=ov_bel,
            ov_matched=ov_mat,
        )

        if belongs is False:
            audit_status = "rejected"
        elif has_ov:
            audit_status = "human_override"
        elif belongs is True:
            audit_status = "vision_verified"
        else:
            audit_status = "unverified_tail"

        if audit_status == "rejected":
            rej_sum += float(row["raw_spend_usd"])
            continue

        if matched_raw and str(matched_raw).strip() and not is_rejection_value(str(matched_raw)):
            brands = [b.strip() for b in str(matched_raw).split(",") if b.strip()]
        else:
            cand = row.get("candidate_brand_ids")
            brands = list(cand) if cand is not None and len(cand) > 0 else []

        k = len(brands)
        if k == 0:
            rej_sum += float(row["raw_spend_usd"])
            continue
        q_score = float(ai.get("format_fit_score", 65.0)) if ai else 65.0
        quality_by_occ[str(row["occurrence_id"])] = q_score
        for brand in brands:
            fct_rows.append(
                {
                    "week_start": row["week_start"].date() if hasattr(row["week_start"], "date") else row["week_start"],
                    "country_iso": row["country_iso"],
                    "brand_id": brand,
                    "occurrence_id": row["occurrence_id"],
                    "asset_id": aid,
                    "media_hash_key": mkey,
                    "channel_group": row["channel_group"],
                    "publisher_name": row["publisher_name"],
                    "campaign_id": row["campaign_id"],
                    "attribution_k": k,
                    "spend_usd": float(row["raw_spend_usd"]) / k,
                    "impressions": float(row["raw_impressions"]) / k,
                    "reach_est": float(row["raw_reach_est"]) / k,
                    "is_spend_modeled": bool(row["is_spend_modeled"]),
                    "is_weekly_interpolated": bool(row["is_weekly_interpolated"]),
                    "audit_status": audit_status,
                }
            )

    fct_df = pd.DataFrame(fct_rows)
    attr_sum = float(fct_df["spend_usd"].sum()) if not fct_df.empty else 0.0
    conservation_diff = abs((attr_sum + rej_sum) - raw_sum)
    if conservation_diff >= 1e-6:
        raise ValueError(f"Spend conservation invariant violated: diff={conservation_diff}")

    con = duckdb.connect(":memory:")
    try:
        con.register("fct_delivery", fct_df)
        con.register("assets_tbl", assets_df)
        con.register("outcomes_tbl", outcomes_df)

        monthly_df = con.execute(
            """
            WITH agg AS (
                SELECT
                    DATE_TRUNC('month', CAST(week_start AS DATE))::DATE AS month_start,
                    country_iso,
                    brand_id,
                    SUM(spend_usd) AS spend_usd,
                    SUM(impressions) AS impressions,
                    SUM(CASE WHEN audit_status IN ('domain_verified', 'vision_verified', 'human_override') THEN spend_usd ELSE 0.0 END) AS verified_spend,
                    COUNT(DISTINCT asset_id)::INT AS creative_count
                FROM fct_delivery
                GROUP BY 1, 2, 3
            ),
            cat AS (
                SELECT
                    *,
                    SUM(spend_usd) OVER (PARTITION BY month_start, country_iso) AS cat_spend,
                    SUM(impressions) OVER (PARTITION BY month_start, country_iso) AS cat_impr
                FROM agg
            )
            SELECT
                month_start, country_iso, brand_id, spend_usd, impressions,
                CASE WHEN cat_spend > 0 THEN (spend_usd / cat_spend) * 100.0 ELSE 0.0 END AS sos_pct,
                CASE WHEN cat_impr > 0 THEN (impressions / cat_impr) * 100.0 ELSE 0.0 END AS sov_pct,
                CASE WHEN spend_usd > 0 THEN verified_spend / spend_usd ELSE 0.0 END AS audit_coverage_pct,
                creative_count
            FROM cat
            """
        ).fetchdf()

        if not themes_df.empty:
            active_themes = themes_df[themes_df["is_active_run"].astype(bool)].drop_duplicates("media_hash_key").copy()
            con.register("themes_tbl", active_themes)
            joined_themes_df = con.execute(
                """
                SELECT
                    CAST(f.week_start AS DATE) AS week_start,
                    f.country_iso,
                    f.brand_id,
                    f.asset_id,
                    f.media_hash_key,
                    t.theme_id,
                    t.theme_label,
                    f.spend_usd,
                    f.impressions
                FROM fct_delivery f
                INNER JOIN themes_tbl t ON f.media_hash_key = t.media_hash_key
                """
            ).fetchdf()
            if not joined_themes_df.empty and (asset_ov or hash_ov or theme_ov):
                resolved_labels: list[str] = []
                for aid_v, mkey_v, orig_lbl in zip(
                    joined_themes_df["asset_id"],
                    joined_themes_df["media_hash_key"],
                    joined_themes_df["theme_label"],
                    strict=True,
                ):
                    lbl_ov = (
                        asset_ov.get((str(aid_v), "theme_label"))
                        or hash_ov.get((str(mkey_v), "theme_label"))
                        or theme_ov.get((str(orig_lbl).strip().lower(), "theme_label"))
                    )
                    resolved_labels.append(str(lbl_ov[0]).strip() if lbl_ov and str(lbl_ov[0]).strip() else str(orig_lbl))
                joined_themes_df["theme_label"] = resolved_labels
            if not joined_themes_df.empty:
                theme_spend_df = (
                    joined_themes_df.groupby(
                        ["week_start", "country_iso", "brand_id", "theme_id", "theme_label"],
                        as_index=False,
                    )
                    .agg(
                        theme_spend_usd=("spend_usd", "sum"),
                        theme_impressions=("impressions", "sum"),
                    )
                )
            else:
                theme_spend_df = pd.DataFrame()
        else:
            theme_spend_df = pd.DataFrame()

        # Quality + creative age enrichment for gold_mmm_weekly_panel
        fct_df["quality_score"] = fct_df["occurrence_id"].map(quality_by_occ).fillna(65.0)
        con.register("fct_enriched", fct_df)
        brand_week_df = con.execute(
            """
            WITH joined AS (
                SELECT
                    f.*,
                    GREATEST(0.0, DATEDIFF('day', CAST(a.first_seen_date AS DATE), CAST(f.week_start AS DATE)) / 7.0) AS creative_age_weeks
                FROM fct_enriched f
                LEFT JOIN assets_tbl a ON f.asset_id = a.asset_id
            ),
            bw AS (
                SELECT
                    CAST(week_start AS DATE) AS week_start,
                    country_iso,
                    brand_id,
                    SUM(CASE WHEN channel_group = 'meta_paid' THEN spend_usd ELSE 0.0 END) AS own_spend_meta_usd,
                    SUM(CASE WHEN channel_group = 'youtube_paid' THEN spend_usd ELSE 0.0 END) AS own_spend_youtube_usd,
                    SUM(CASE WHEN channel_group = 'tiktok_paid' THEN spend_usd ELSE 0.0 END) AS own_spend_tiktok_usd,
                    SUM(CASE WHEN channel_group = 'display' THEN spend_usd ELSE 0.0 END) AS own_spend_display_usd,
                    SUM(spend_usd) AS own_spend_total_usd,
                    SUM(impressions) AS raw_own_impr,
                    BOOL_OR(is_spend_modeled) AS is_spend_modeled,
                    SUM(CASE WHEN is_weekly_interpolated THEN spend_usd ELSE 0.0 END) AS interp_spend,
                    SUM(quality_score * spend_usd) AS w_quality,
                    SUM(creative_age_weeks * spend_usd) AS w_age,
                    SUM(CASE WHEN audit_status IN ('domain_verified', 'vision_verified', 'human_override') THEN spend_usd ELSE 0.0 END) AS verified_spend
                FROM joined
                GROUP BY 1, 2, 3
            ),
            cw AS (
                SELECT
                    *,
                    SUM(own_spend_total_usd) OVER (PARTITION BY week_start, country_iso) AS cw_spend,
                    SUM(raw_own_impr) OVER (PARTITION BY week_start, country_iso) AS cw_impr
                FROM bw
            ),
            shares AS (
                SELECT
                    *,
                    POWER(CASE WHEN cw_spend > 0 THEN (own_spend_total_usd / cw_spend) * 100.0 ELSE 0.0 END, 2.0) AS share_sq
                FROM cw
            )
            SELECT
                week_start,
                country_iso,
                brand_id,
                own_spend_meta_usd,
                own_spend_youtube_usd,
                own_spend_tiktok_usd,
                own_spend_display_usd,
                own_spend_total_usd,
                CASE WHEN is_spend_modeled THEN NULL ELSE raw_own_impr END AS own_impressions_total,
                GREATEST(0.0, cw_spend - own_spend_total_usd) AS competitor_spend_usd,
                GREATEST(0.0, cw_impr - raw_own_impr) AS competitor_impressions,
                SUM(share_sq) OVER (PARTITION BY week_start, country_iso) AS category_hhi,
                CASE WHEN own_spend_total_usd > 0 THEN w_quality / own_spend_total_usd ELSE 0.0 END AS avg_creative_quality_score,
                CASE WHEN own_spend_total_usd > 0 THEN w_age / own_spend_total_usd ELSE 0.0 END AS avg_creative_age_weeks,
                CASE WHEN own_spend_total_usd > 0 THEN verified_spend / own_spend_total_usd ELSE 0.0 END AS audit_coverage_pct,
                is_spend_modeled,
                CASE WHEN own_spend_total_usd > 0 THEN (interp_spend / own_spend_total_usd) > 0.5 ELSE FALSE END AS is_weekly_interpolated
            FROM shares
            """
        ).fetchdf()
        con.register("bw_full", brand_week_df)

        has_conv = (outcomes_df["outcome_metric_name"] == "conversions").any()
        primary_metric = "conversions" if has_conv else "google_trends_index"
        mmm_panel_df = con.execute(
            f"""
            SELECT
                b.*,
                o.outcome_metric_name,
                o.outcome_value,
                o.outcome_source,
                o.promo_flag,
                o.holiday_flag,
                o.price_index,
                o.macro_seasonality
            FROM bw_full b
            INNER JOIN outcomes_tbl o
                ON b.week_start = CAST(o.week_start AS DATE)
                AND b.country_iso = o.country_iso
                AND b.brand_id = o.brand_id
            WHERE o.outcome_metric_name = '{primary_metric}'
            """
        ).fetchdf()

        mmm_proxy_df = con.execute(
            """
            SELECT
                b.*,
                o.outcome_metric_name,
                o.outcome_value,
                o.outcome_source,
                o.promo_flag,
                o.holiday_flag,
                o.price_index,
                o.macro_seasonality
            FROM bw_full b
            INNER JOIN outcomes_tbl o
                ON b.week_start = CAST(o.week_start AS DATE)
                AND b.country_iso = o.country_iso
                AND b.brand_id = o.brand_id
            WHERE o.outcome_metric_name = 'google_trends_index'
            """
        ).fetchdf()
    finally:
        con.close()

    for col in ("week_start",):
        fct_df[col] = pd.to_datetime(fct_df[col]).dt.date
        if not theme_spend_df.empty:
            theme_spend_df[col] = pd.to_datetime(theme_spend_df[col]).dt.date
        mmm_panel_df[col] = pd.to_datetime(mmm_panel_df[col]).dt.date
        if not mmm_proxy_df.empty:
            mmm_proxy_df[col] = pd.to_datetime(mmm_proxy_df[col]).dt.date
    monthly_df["month_start"] = pd.to_datetime(monthly_df["month_start"]).dt.date

    write_parquet_table(fct_df, gold_root / "fct_ad_delivery_weekly", table_name="fct_ad_delivery_weekly", partition_cols=["country_iso"])
    write_parquet_table(monthly_df, gold_root / "gold_brand_monthly", table_name="gold_brand_monthly", partition_cols=["country_iso"])
    write_parquet_table(theme_spend_df, gold_root / "gold_theme_spend_weekly", table_name="gold_theme_spend_weekly", partition_cols=["country_iso"])
    write_parquet_table(mmm_panel_df, gold_root / "gold_mmm_weekly_panel", table_name="gold_mmm_weekly_panel", partition_cols=["country_iso"])
    if not mmm_proxy_df.empty:
        write_parquet_table(mmm_proxy_df, gold_root / "gold_mmm_proxy_weekly_panel", table_name="gold_mmm_weekly_panel", partition_cols=["country_iso"])

    return {
        "status": "SUCCESS",
        "spend_conservation_diff": conservation_diff,
        "raw_spend_usd": round(raw_sum, 4),
        "attributed_spend_usd": round(attr_sum, 4),
        "rejected_spend_usd": round(rej_sum, 4),
        "fct_delivery_rows": int(len(fct_df)),
        "gold_mmm_rows": int(len(mmm_panel_df)),
        "gold_theme_rows": int(len(theme_spend_df)),
    }
