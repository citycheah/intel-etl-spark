"""Deterministic 156-week EU Automotive (BMW, Mercedes-Benz, Audi, Porsche in DE/FR/GB) DGP generator."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from intel_core.media import compute_format_fit_score
from intel_core.optimize import hill_saturation
from intel_core.schemas import (
    compute_asset_id,
    compute_media_hash_key,
    convert_to_usd,
)
from intel_core.storage import write_parquet_table

BRANDS = ["BMW", "Mercedes-Benz", "Audi", "Porsche"]
COUNTRIES = ["DE", "FR", "GB"]
CHANNELS = ["meta_paid", "youtube_paid", "tiktok_paid", "display"]

TRUE_DGP_PARAMS: dict[str, Any] = {
    "adstock_lambda": {
        "meta_paid": 0.35,
        "youtube_paid": 0.50,
        "tiktok_paid": 0.25,
        "display": 0.15,
    },
    # Structural true Hill response parameters in USD weekly spend space
    "channels": {
        "own_spend_meta_usd": {"beta": 240.0, "ec50_usd": 22000.0, "slope": 1.35, "adstock_lambda": 0.35},
        "own_spend_youtube_usd": {"beta": 310.0, "ec50_usd": 24000.0, "slope": 1.30, "adstock_lambda": 0.50},
        "own_spend_tiktok_usd": {"beta": 180.0, "ec50_usd": 14000.0, "slope": 1.40, "adstock_lambda": 0.25},
        "own_spend_display_usd": {"beta": 110.0, "ec50_usd": 16000.0, "slope": 1.20, "adstock_lambda": 0.15},
    },
    "competitor_cannibalization_coef": -18.0,
    "base_intercept": 420.0,
    "promo_lift": 55.0,
    "holiday_lift": -28.0,
    "price_elasticity": -90.0,
    "noise_std": 8.0,
}

THEMES_BY_SCOPE = [
    ("thm_ev_range", "EV Range & Charging Infrastructure", "shared_battleground"),
    ("thm_lease_apr", "0% APR Flexible Lease Offer", "shared_battleground"),
    ("thm_safety_adas", "Autonomous & ADAS Safety Tech", "shared_battleground"),
    ("thm_track_heritage", "Motorsport & Track Heritage", "brand_exclusive_whitespace"),
    ("thm_craft_interior", "Bespoke Cabin Craftsmanship", "brand_exclusive_whitespace"),
]


def _geometric_adstock(series: np.ndarray, lam: float) -> np.ndarray:
    """Normalized geometric adstock (steady-state sum of weights == 1.0)."""
    out = np.zeros_like(series, dtype=np.float64)
    prev = 0.0
    for idx, val in enumerate(series):
        cur = float(val) + lam * prev
        out[idx] = cur
        prev = cur
    return out * (1.0 - lam)


def generate_eu_auto_dgp(
    output_dir: str | Path,
    *,
    num_weeks: int = 156,
    start_monday: str = "2024-01-01",
    seed: int = 42,
) -> dict[str, Path]:
    """Generates Bronze ad occurrences, seed dissections/themes, weekly outcomes, and DGP truth JSON."""
    rng = np.random.default_rng(seed)
    out_root = Path(output_dir)
    bronze_dir = out_root / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)

    start_date = dt.date.fromisoformat(start_monday)
    if start_date.weekday() != 0:
        raise ValueError(f"start_monday must be an ISO Monday, got {start_date}")
    weeks = [start_date + dt.timedelta(weeks=w) for w in range(num_weeks)]

    # Build a stable creative library per brand (12 creatives per brand = 48 + 4 off-brand/comparative = 52)
    creative_catalog: list[dict[str, Any]] = []
    for b_idx, brand in enumerate(BRANDS):
        for c_idx in range(12):
            ch = CHANNELS[(b_idx + c_idx) % len(CHANNELS)]
            ar = "9:16" if ch == "tiktok_paid" else ("16:9" if ch == "youtube_paid" else ("4:5" if ch == "meta_paid" else "1:1"))
            m_type = "video" if ch in ("tiktok_paid", "youtube_paid") else "image"
            dur = 15.0 if ch == "tiktok_paid" else (30.0 if ch == "youtube_paid" else None)
            uri = f"https://cdn.eu-auto.example/{brand.lower()}/creative_{c_idx + 1:02d}.{'mp4' if m_type == 'video' else 'jpg'}"
            asset_id = compute_asset_id(uri)
            # Deliberately leave 1 creative without phash to exercise COALESCE(NULLIF(phash,''), asset_id)
            phash = None if c_idx == 11 else f"{(b_idx + 1):02x}{c_idx:02x}a1b2c3d4e5f6"
            theme_id, theme_label, theme_scope = THEMES_BY_SCOPE[c_idx % len(THEMES_BY_SCOPE)]
            if theme_scope == "brand_exclusive_whitespace":
                theme_id = f"{theme_id}_{brand.lower().replace('-', '_')}"
                theme_label = f"{brand} {theme_label}"
            creative_catalog.append(
                {
                    "asset_id": asset_id,
                    "phash": phash,
                    "media_hash_key": compute_media_hash_key(phash, asset_id),
                    "brand_id": brand,
                    "candidate_brand_ids": [brand],
                    "belongs_to_brand": True,
                    "matched_brand_id": brand,
                    "channel_group": ch,
                    "media_type": m_type,
                    "media_uri": uri,
                    "aspect_ratio": ar,
                    "duration_sec": dur,
                    "theme_id": theme_id,
                    "theme_label": theme_label,
                    "theme_scope": theme_scope,
                    "hook_type": ["offer_first", "product_demo", "brand_story", "ugc_testimonial"][c_idx % 4],
                    "launch_week_idx": (c_idx * 10) % max(1, num_weeks - 20),
                }
            )

    # Add 2 multi-brand comparative ads (attribution K=2) & 2 off-brand/rejected ads
    comp_uri = "https://cdn.eu-auto.example/comparative/bmw_vs_audi_2025.mp4"
    comp_id = compute_asset_id(comp_uri)
    creative_catalog.append(
        {
            "asset_id": comp_id,
            "phash": "ff11223344556677",
            "media_hash_key": "ff11223344556677",
            "brand_id": "BMW",
            "candidate_brand_ids": ["BMW", "Audi"],
            "belongs_to_brand": True,
            "matched_brand_id": "BMW,Audi",
            "channel_group": "youtube_paid",
            "media_type": "video",
            "media_uri": comp_uri,
            "aspect_ratio": "16:9",
            "duration_sec": 30.0,
            "theme_id": "thm_ev_range",
            "theme_label": "EV Range & Charging Infrastructure",
            "theme_scope": "shared_battleground",
            "hook_type": "problem_agitation",
            "launch_week_idx": 40,
        }
    )
    rej_uri = "https://cdn.eu-auto.example/reseller/unofficial_mats.jpg"
    rej_id = compute_asset_id(rej_uri)
    creative_catalog.append(
        {
            "asset_id": rej_id,
            "phash": "9988776655443322",
            "media_hash_key": "9988776655443322",
            "brand_id": "BMW",
            "candidate_brand_ids": ["BMW"],
            "belongs_to_brand": False,
            "matched_brand_id": None,
            "channel_group": "display",
            "media_type": "image",
            "media_uri": rej_uri,
            "aspect_ratio": "1:1",
            "duration_sec": None,
            "theme_id": "thm_lease_apr",
            "theme_label": "0% APR Flexible Lease Offer",
            "theme_scope": "shared_battleground",
            "hook_type": "offer_first",
            "launch_week_idx": 20,
        }
    )

    country_scale = {"DE": 1.0, "FR": 0.72, "GB": 0.80}
    brand_scale = {"BMW": 1.05, "Mercedes-Benz": 1.00, "Audi": 0.88, "Porsche": 0.65}
    currency_map = {"DE": "EUR", "FR": "EUR", "GB": "GBP"}
    publishers = {
        "meta_paid": "Instagram",
        "youtube_paid": "YouTube",
        "tiktok_paid": "TikTok",
        "display": "Programmatic Display",
    }

    bronze_rows: list[dict[str, Any]] = []
    # Keep track of cumulative EU reach per (asset_id, country_iso) for Meta DSA delta check
    cum_eu_reach: dict[tuple[str, str], int] = {}

    # Pre-generate independent weekly channel budgets per (country, brand, week) so MMM identifiability is crisp
    weekly_channel_budgets: dict[tuple[str, str, int, str], float] = {}
    for country in COUNTRIES:
        c_mult = country_scale[country]
        for brand in BRANDS:
            b_mult = brand_scale[brand]
            for w_idx, w_date in enumerate(weeks):
                season = 1.0 + 0.22 * math.sin(2.0 * math.pi * w_idx / 52.0)
                # Independent per-channel pulse + AR(1) variation so channels are NOT collinear
                for ch_idx, ch in enumerate(CHANNELS):
                    base_ch = {"meta_paid": 18000.0, "youtube_paid": 20000.0, "tiktok_paid": 10000.0, "display": 12000.0}[ch]
                    pulse = 1.35 if ((w_idx + ch_idx * 5) % 13 == 0) else 1.0
                    noise = float(rng.lognormal(mean=0.0, sigma=0.28))
                    spend_local = base_ch * c_mult * b_mult * season * pulse * noise
                    weekly_channel_budgets[(country, brand, w_idx, ch)] = spend_local

                # Pick active creatives for each channel and emit raw occurrences
                for ch in CHANNELS:
                    budget_local = weekly_channel_budgets[(country, brand, w_idx, ch)]
                    eligible = [
                        c for c in creative_catalog
                        if c["brand_id"] == brand and c["channel_group"] == ch
                    ]
                    if not eligible:
                        continue
                    # Rotate 2-3 active creatives per week
                    selected = [eligible[(w_idx + k) % len(eligible)] for k in range(min(2, len(eligible)))]
                    share = 1.0 / len(selected)
                    cur = currency_map[country]
                    for sel in selected:
                        occ_spend_local = round(budget_local * share, 2)
                        spend_usd_est = convert_to_usd(occ_spend_local, cur, w_date)
                        cpm = {"meta_paid": 8.5, "youtube_paid": 11.0, "tiktok_paid": 6.5, "display": 4.5}[ch]
                        impr = int(round((spend_usd_est / cpm) * 1000.0))
                        reach_delta = int(round(impr * float(rng.uniform(0.62, 0.78))))
                        key = (sel["asset_id"], country)
                        prev_cum = cum_eu_reach.get(key, 0)
                        new_cum = prev_cum + reach_delta
                        cum_eu_reach[key] = new_cum
                        occ_id = f"occ_{w_date.isoformat()}_{country}_{sel['asset_id'][:10]}_{ch}"
                        bronze_rows.append(
                            {
                                "occurrence_id": occ_id,
                                "event_date": w_date.isoformat(),
                                "country_raw": "Germany" if country == "DE" else ("France" if country == "FR" else "United Kingdom"),
                                "brand_raw": sel["brand_id"],
                                "candidate_brand_ids": sel["candidate_brand_ids"],
                                "asset_id": sel["asset_id"],
                                "phash": sel["phash"],
                                "media_type": sel["media_type"],
                                "media_uri": sel["media_uri"],
                                "aspect_ratio": sel["aspect_ratio"],
                                "duration_sec": sel["duration_sec"],
                                "source_connector": "meta_ad_lib_eu" if ch == "meta_paid" else "synthetic_eu_auto",
                                "channel_group": ch,
                                "publisher_name": publishers[ch],
                                "campaign_id": f"cmp_{brand[:3].lower()}_{country.lower()}_{w_date.year}",
                                "raw_impressions": impr,
                                "raw_reach_est": reach_delta,
                                "raw_eu_total_reach": new_cum if ch == "meta_paid" else None,
                                "snapshot_ts": f"{w_date.isoformat()}T12:00:00+00:00",
                                "is_weekly_interpolated": False,
                                "raw_spend_local": occ_spend_local,
                                "currency": cur,
                                "is_spend_modeled": False,
                            }
                        )

    # Seed dissections & themes tables (so full Spark Job 2 can run 100% offline cleanly)
    dissection_rows = []
    theme_rows = []
    for item in creative_catalog:
        fit = compute_format_fit_score(
            aspect_ratio=item["aspect_ratio"],
            duration_sec=item["duration_sec"],
            channel_group=item["channel_group"],
            media_type=item["media_type"],
        )
        dissection_rows.append(
            {
                "media_hash_key": item["media_hash_key"],
                "evaluated_brand_id": item["brand_id"],
                "prompt_version": "creative_dissection_v2",
                "is_active_version": True,
                "model_name": "gemini-2.5-flash",
                "belongs_to_brand": item["belongs_to_brand"],
                "matched_brand_id": item["matched_brand_id"],
                "confidence": 0.94 if item["belongs_to_brand"] else 0.91,
                "category": "EU Automotive",
                "core_concept_summary": item["theme_label"],
                "messaging_angle": "Electric performance & lease value",
                "visual_style_tags": ["automotive", "studio"],
                "hook_type": item["hook_type"],
                "hook_transcript": f"{item['brand_id']} - {item['theme_label']}",
                "time_to_brand_sec": 1.1 if item["belongs_to_brand"] else None,
                "has_voiceover": item["media_type"] == "video",
                "has_captions": True,
                "cta_text": "Configure now",
                "cta_clarity_score": 4 if item["belongs_to_brand"] else 2,
                "format_fit_score": fit,
            }
        )
        theme_rows.append(
            {
                "media_hash_key": item["media_hash_key"],
                "run_id": "dgp_seed_run_001",
                "is_active_run": True,
                "theme_id": item["theme_id"],
                "theme_label": item["theme_label"],
                "theme_scope": item["theme_scope"],
                "umap_x": float(rng.normal(0.0, 2.5)),
                "umap_y": float(rng.normal(0.0, 2.5)),
                "centroid_similarity": 0.93,
            }
        )

    # Compute true weekly DGP outcomes (conversions & Google Trends proxy) per (country, brand)
    bronze_df = pd.DataFrame(bronze_rows)
    outcome_rows: list[dict[str, Any]] = []
    truth_summary: dict[str, Any] = {
        "vertical": "EU_AUTOMOTIVE",
        "num_weeks": num_weeks,
        "start_monday": start_monday,
        "parameters": TRUE_DGP_PARAMS,
        "true_marginal_roas_at_mean": {},
    }

    # Build spend series in USD aligned per (country, brand, week)
    for country in COUNTRIES:
        cur = currency_map[country]
        for brand in BRANDS:
            # Extract valid verified spend (exclude rejected reseller asset)
            sub_valid = bronze_df[
                (bronze_df["country_raw"] == ("Germany" if country == "DE" else ("France" if country == "FR" else "United Kingdom")))
                & (bronze_df["brand_raw"] == brand)
                & (bronze_df["asset_id"] != rej_id)
            ]
            # Map week -> channel spend_usd
            ch_series: dict[str, np.ndarray] = {}
            for ch in CHANNELS:
                arr = np.zeros(num_weeks, dtype=np.float64)
                for w_idx, w_date in enumerate(weeks):
                    local_amt = sub_valid.loc[
                        (sub_valid["event_date"] == w_date.isoformat()) & (sub_valid["channel_group"] == ch),
                        "raw_spend_local",
                    ].sum()
                    arr[w_idx] = convert_to_usd(float(local_amt), cur, w_date)
                ch_series[ch] = arr

            # Exogenous competitor spend in same country
            comp_arr = np.zeros(num_weeks, dtype=np.float64)
            sub_comp = bronze_df[
                (bronze_df["country_raw"] == ("Germany" if country == "DE" else ("France" if country == "FR" else "United Kingdom")))
                & (bronze_df["brand_raw"] != brand)
                & (bronze_df["asset_id"] != rej_id)
            ]
            for w_idx, w_date in enumerate(weeks):
                comp_local = sub_comp.loc[sub_comp["event_date"] == w_date.isoformat(), "raw_spend_local"].sum()
                comp_arr[w_idx] = convert_to_usd(float(comp_local), cur, w_date)

            # Adstock + Hill + controls -> true KPI conversions
            promo_arr = np.array([1 if (w % 12 == 2) else 0 for w in range(num_weeks)], dtype=np.int32)
            holiday_arr = np.array([1 if (weeks[w].month == 12 and weeks[w].day >= 20) or (weeks[w].month == 8 and weeks[w].day <= 14) else 0 for w in range(num_weeks)], dtype=np.int32)
            price_arr = 1.0 + 0.04 * np.sin(2.0 * math.pi * np.arange(num_weeks) / 26.0)
            season_arr = np.sin(2.0 * math.pi * np.arange(num_weeks) / 52.0)

            y_conv = (
                TRUE_DGP_PARAMS["base_intercept"] * country_scale[country] * brand_scale[brand]
                + 35.0 * season_arr
                + TRUE_DGP_PARAMS["promo_lift"] * promo_arr
                + TRUE_DGP_PARAMS["holiday_lift"] * holiday_arr
                + TRUE_DGP_PARAMS["price_elasticity"] * (price_arr - 1.0)
                + TRUE_DGP_PARAMS["competitor_cannibalization_coef"] * np.log1p(comp_arr / 50000.0)
                + rng.normal(0.0, TRUE_DGP_PARAMS["noise_std"], size=num_weeks)
            )

            col_map = {
                "meta_paid": "own_spend_meta_usd",
                "youtube_paid": "own_spend_youtube_usd",
                "tiktok_paid": "own_spend_tiktok_usd",
                "display": "own_spend_display_usd",
            }
            for ch in CHANNELS:
                col_key = col_map[ch]
                p = TRUE_DGP_PARAMS["channels"][col_key]
                adstocked = _geometric_adstock(ch_series[ch], p["adstock_lambda"])
                contrib = np.array(
                    [p["beta"] * hill_saturation(x, ec50_usd=p["ec50_usd"], slope=p["slope"]) for x in adstocked]
                )
                y_conv += contrib

                if country == "DE" and brand == "BMW":
                    mean_spend = float(np.mean(ch_series[ch]))
                    total_contrib = float(np.sum(contrib))
                    total_spend = float(np.sum(ch_series[ch]))
                    # Conversions per $1000 spend (or scaled units) + total share
                    truth_summary["true_marginal_roas_at_mean"][col_key] = {
                        "mean_weekly_spend_usd": round(mean_spend, 2),
                        "true_avg_roas_per_1k_usd": round((total_contrib / max(1.0, total_spend)) * 1000.0, 4),
                        "true_total_contribution": round(total_contrib, 2),
                    }

            y_conv = np.maximum(20.0, y_conv)
            trends_idx = np.clip((y_conv / np.max(y_conv)) * 100.0 + rng.normal(0.0, 2.0, size=num_weeks), 5.0, 100.0)

            for w_idx, w_date in enumerate(weeks):
                outcome_rows.append(
                    {
                        "week_start": w_date,
                        "country_iso": country,
                        "brand_id": brand,
                        "outcome_metric_name": "conversions",
                        "outcome_value": round(float(y_conv[w_idx]), 4),
                        "outcome_source": "synthetic_dgp",
                        "promo_flag": int(promo_arr[w_idx]),
                        "holiday_flag": int(holiday_arr[w_idx]),
                        "price_index": round(float(price_arr[w_idx]), 4),
                        "macro_seasonality": round(float(season_arr[w_idx]), 4),
                    }
                )
                outcome_rows.append(
                    {
                        "week_start": w_date,
                        "country_iso": country,
                        "brand_id": brand,
                        "outcome_metric_name": "google_trends_index",
                        "outcome_value": round(float(trends_idx[w_idx]), 2),
                        "outcome_source": "google_trends_proxy",
                        "promo_flag": int(promo_arr[w_idx]),
                        "holiday_flag": int(holiday_arr[w_idx]),
                        "price_index": round(float(price_arr[w_idx]), 4),
                        "macro_seasonality": round(float(season_arr[w_idx]), 4),
                    }
                )

    occ_path = bronze_dir / "bronze_ad_occurrences.parquet"
    write_parquet_table(bronze_df, occ_path)

    outcomes_path = bronze_dir / "fct_business_outcomes_weekly"
    write_parquet_table(outcome_rows, outcomes_path, table_name="fct_business_outcomes_weekly")

    dissect_path = bronze_dir / "seed_creative_dissection"
    write_parquet_table(dissection_rows, dissect_path, table_name="dim_creative_dissection")

    themes_path = bronze_dir / "seed_creative_themes"
    write_parquet_table(theme_rows, themes_path, table_name="dim_creative_themes")

    truth_path = out_root / "eu_auto_dgp_truth.json"
    truth_path.write_text(json.dumps(truth_summary, indent=2), encoding="utf-8")

    return {
        "bronze_occurrences": occ_path,
        "outcomes": outcomes_path,
        "seed_dissection": dissect_path,
        "seed_themes": themes_path,
        "truth_json": truth_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 156-week EU Automotive DGP fixtures.")
    parser.add_argument("--output-dir", default="data/eu_auto_dgp", help="Destination lakehouse base dir")
    parser.add_argument("--weeks", type=int, default=156)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    outputs = generate_eu_auto_dgp(args.output_dir, num_weeks=args.weeks, seed=args.seed)
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
