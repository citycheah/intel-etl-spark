"""PySpark Stage 2 ETL: Silver + Dissection + Themes + HITL Overrides -> Gold Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from intel_core.storage import REJECTION_SENTINELS, HumanOverrideStore, is_rejection_value
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from intel_etl_spark.jobs.bronze_to_silver import (
    get_spark_session,
    write_spark_parquet_atomic,
)


def _read_optional_parquet(spark: SparkSession, path: Path) -> DataFrame | None:
    if not path.exists():
        return None
    if path.is_file() or list(path.rglob("*.parquet")):
        return spark.read.parquet(str(path))
    return None


def _parse_brand_list(raw_matched: str | None, fallback_list: list[str] | None) -> list[str]:
    if raw_matched and str(raw_matched).strip() and not is_rejection_value(str(raw_matched)):
        parts = [p.strip() for p in str(raw_matched).split(",") if p.strip()]
        if parts:
            return parts
    return list(fallback_list or [])


def run_silver_to_gold(
    spark: SparkSession,
    *,
    silver_dir: str | Path,
    gold_dir: str | Path,
    outcomes_path: str | Path,
    overrides_dir: str | Path | None = None,
    dissection_path: str | Path | None = None,
    themes_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolves HITL overrides & verified brands BEFORE 1/K splitting, verifies spend conservation, and writes Gold tables."""
    silver_root = Path(silver_dir)
    gold_root = Path(gold_dir)
    gold_root.mkdir(parents=True, exist_ok=True)

    raw_df = spark.read.parquet(str(silver_root / "silver_ad_delivery_raw"))
    assets_df = spark.read.parquet(str(silver_root / "dim_creative_asset"))

    diss_p = Path(dissection_path) if dissection_path else (silver_root / "dim_creative_dissection")
    diss_df = _read_optional_parquet(spark, diss_p)

    thm_p = Path(themes_path) if themes_path else (silver_root / "dim_creative_themes")
    themes_df = _read_optional_parquet(spark, thm_p)

    outcomes_df = spark.read.parquet(str(outcomes_path))

    # Load HITL overrides (`asset_id` > `media_hash_key`, latest `created_at` wins)
    ov_records = HumanOverrideStore(overrides_dir).list_overrides() if overrides_dir else []

    if diss_df is not None:
        active_diss = (
            diss_df.filter(F.col("is_active_version") == F.lit(True))
            .withColumn(
                "_rn",
                F.row_number().over(
                    Window.partitionBy("media_hash_key").orderBy(F.col("confidence").desc_nulls_last(), F.col("prompt_version").desc())
                ),
            )
            .filter(F.col("_rn") == 1)
            .drop("_rn")
        )
    else:
        active_diss = None

    # Join Silver occurrences to active creative dissections (1:1 on media_hash_key)
    if active_diss is not None:
        joined = raw_df.join(
            active_diss.select(
                "media_hash_key",
                F.col("belongs_to_brand").alias("ai_belongs_to_brand"),
                F.col("matched_brand_id").alias("ai_matched_brand_id"),
                F.col("format_fit_score").alias("ai_format_fit_score"),
            ),
            on="media_hash_key",
            how="left",
        )
    else:
        joined = (
            raw_df.withColumn("ai_belongs_to_brand", F.lit(None).cast("boolean"))
            .withColumn("ai_matched_brand_id", F.lit(None).cast("string"))
            .withColumn("ai_format_fit_score", F.lit(65.0).cast("double"))
        )

    # Apply overrides in Spark with strict precedence: asset_id > media_hash_key > theme_label, latest created_at wins
    ov_theme_sdf = None
    if ov_records:
        ov_pdf = HumanOverrideStore(overrides_dir).to_dataframe()
        ov_sdf = spark.createDataFrame(ov_pdf)
        ov_asset = (
            ov_sdf.filter(F.col("target_key_type") == "asset_id")
            .withColumn(
                "_rn",
                F.row_number().over(Window.partitionBy("target_key_value", "field_name").orderBy(F.col("created_at").desc())),
            )
            .filter(F.col("_rn") == 1)
            .groupBy("target_key_value")
            .agg(
                F.max(F.when(F.col("field_name") == "belongs_to_brand", F.col("human_value"))).alias("ov_a_belongs"),
                F.max(F.when(F.col("field_name") == "belongs_to_brand", F.col("created_at"))).alias("ov_a_belongs_ts"),
                F.max(F.when(F.col("field_name") == "matched_brand_id", F.col("human_value"))).alias("ov_a_matched"),
                F.max(F.when(F.col("field_name") == "matched_brand_id", F.col("created_at"))).alias("ov_a_matched_ts"),
                F.max(F.when(F.col("field_name") == "theme_label", F.col("human_value"))).alias("ov_a_theme"),
            )
            .withColumnRenamed("target_key_value", "asset_id")
        )
        ov_hash = (
            ov_sdf.filter(F.col("target_key_type") == "media_hash_key")
            .withColumn(
                "_rn",
                F.row_number().over(Window.partitionBy("target_key_value", "field_name").orderBy(F.col("created_at").desc())),
            )
            .filter(F.col("_rn") == 1)
            .groupBy("target_key_value")
            .agg(
                F.max(F.when(F.col("field_name") == "belongs_to_brand", F.col("human_value"))).alias("ov_h_belongs"),
                F.max(F.when(F.col("field_name") == "belongs_to_brand", F.col("created_at"))).alias("ov_h_belongs_ts"),
                F.max(F.when(F.col("field_name") == "matched_brand_id", F.col("human_value"))).alias("ov_h_matched"),
                F.max(F.when(F.col("field_name") == "matched_brand_id", F.col("created_at"))).alias("ov_h_matched_ts"),
                F.max(F.when(F.col("field_name") == "theme_label", F.col("human_value"))).alias("ov_h_theme"),
            )
            .withColumnRenamed("target_key_value", "media_hash_key")
        )
        ov_theme_sdf = (
            ov_sdf.filter((F.col("target_key_type") == "theme_label") & (F.col("field_name") == "theme_label"))
            .withColumn("_tkey", F.lower(F.trim(F.col("target_key_value"))))
            .withColumn(
                "_rn",
                F.row_number().over(Window.partitionBy("_tkey").orderBy(F.col("created_at").desc())),
            )
            .filter(F.col("_rn") == 1)
            .select(F.col("_tkey"), F.col("human_value").alias("ov_t_theme"))
        )
        joined = joined.join(ov_asset, on="asset_id", how="left").join(ov_hash, on="media_hash_key", how="left")
    else:
        joined = (
            joined.withColumn("ov_a_belongs", F.lit(None).cast("string"))
            .withColumn("ov_a_belongs_ts", F.lit(None).cast("timestamp"))
            .withColumn("ov_a_matched", F.lit(None).cast("string"))
            .withColumn("ov_a_matched_ts", F.lit(None).cast("timestamp"))
            .withColumn("ov_a_theme", F.lit(None).cast("string"))
            .withColumn("ov_h_belongs", F.lit(None).cast("string"))
            .withColumn("ov_h_belongs_ts", F.lit(None).cast("timestamp"))
            .withColumn("ov_h_matched", F.lit(None).cast("string"))
            .withColumn("ov_h_matched_ts", F.lit(None).cast("timestamp"))
            .withColumn("ov_h_theme", F.lit(None).cast("string"))
        )

    ov_belongs_str = F.coalesce(F.col("ov_a_belongs"), F.col("ov_h_belongs"))
    ov_belongs_prio = F.when(F.col("ov_a_belongs").isNotNull(), F.lit(1)).when(F.col("ov_h_belongs").isNotNull(), F.lit(2)).otherwise(F.lit(99))
    ov_belongs_ts = F.coalesce(F.col("ov_a_belongs_ts"), F.col("ov_h_belongs_ts"))
    ov_matched_str = F.coalesce(F.col("ov_a_matched"), F.col("ov_h_matched"))
    ov_matched_prio = F.when(F.col("ov_a_matched").isNotNull(), F.lit(1)).when(F.col("ov_h_matched").isNotNull(), F.lit(2)).otherwise(F.lit(99))
    ov_matched_ts = F.coalesce(F.col("ov_a_matched_ts"), F.col("ov_h_matched_ts"))
    has_override = ov_belongs_str.isNotNull() | ov_matched_str.isNotNull()
    sentinels = list(REJECTION_SENTINELS)
    matched_wins = ov_matched_str.isNotNull() & (
        ov_belongs_str.isNull()
        | (ov_matched_prio < ov_belongs_prio)
        | ((ov_matched_prio == ov_belongs_prio) & (ov_matched_ts >= ov_belongs_ts))
    )

    resolved = (
        joined.withColumn(
            "final_belongs",
            F.when(
                matched_wins,
                F.when(F.lower(F.trim(ov_matched_str)).isin(sentinels), F.lit(False))
                .when(F.length(F.trim(ov_matched_str)) > 0, F.lit(True))
                .otherwise(F.col("ai_belongs_to_brand")),
            )
            .when(ov_belongs_str.isNotNull(), F.lower(F.trim(ov_belongs_str)).isin("true", "1", "yes", "y"))
            .when(F.lower(F.trim(F.coalesce(F.col("ai_matched_brand_id"), F.lit("")))).isin(sentinels), F.lit(False))
            .otherwise(F.col("ai_belongs_to_brand")),
        )
        .withColumn(
            "final_matched_str",
            F.coalesce(ov_matched_str, F.col("ai_matched_brand_id")),
        )
        .withColumn(
            "audit_status",
            F.when(F.col("final_belongs") == F.lit(False), F.lit("rejected"))
            .when(has_override, F.lit("human_override"))
            .when(F.col("final_belongs") == F.lit(True), F.lit("vision_verified"))
            .otherwise(F.lit("unverified_tail")),
        )
    )

    udf_parse_brands = F.udf(_parse_brand_list, T.ArrayType(T.StringType()))
    with_brands = resolved.withColumn(
        "verified_brands",
        F.when(F.col("audit_status") == F.lit("rejected"), F.array().cast(T.ArrayType(T.StringType())))
        .otherwise(udf_parse_brands(F.col("final_matched_str"), F.col("candidate_brand_ids"))),
    )

    # Spend Conservation Check BEFORE/AFTER 1/K splitting
    raw_totals = with_brands.agg(
        F.coalesce(F.sum("raw_spend_usd"), F.lit(0.0)).alias("raw_sum"),
        F.coalesce(
            F.sum(
                F.when(
                    (F.col("audit_status") == F.lit("rejected")) | (F.size("verified_brands") == 0),
                    F.col("raw_spend_usd"),
                ).otherwise(F.lit(0.0))
            ),
            F.lit(0.0),
        ).alias("rej_sum"),
    ).collect()[0]

    # Explode verified_brands AFTER override + verification resolution -> 1/K splitting!
    split_df = (
        with_brands.filter((F.col("audit_status") != F.lit("rejected")) & (F.size("verified_brands") > 0))
        .withColumn("attribution_k", F.size("verified_brands").cast("int"))
        .withColumn("brand_id", F.explode("verified_brands"))
        .withColumn("spend_usd", F.col("raw_spend_usd") / F.col("attribution_k"))
        .withColumn("impressions", F.col("raw_impressions").cast("double") / F.col("attribution_k"))
        .withColumn("reach_est", F.col("raw_reach_est").cast("double") / F.col("attribution_k"))
        .withColumn("quality_score", F.coalesce(F.col("ai_format_fit_score"), F.lit(65.0)))
    )

    fct_delivery = split_df.select(
        "week_start",
        "country_iso",
        "brand_id",
        "occurrence_id",
        "asset_id",
        "media_hash_key",
        "channel_group",
        "publisher_name",
        "campaign_id",
        "attribution_k",
        "spend_usd",
        "impressions",
        "reach_est",
        "is_spend_modeled",
        "is_weekly_interpolated",
        "audit_status",
    )
    fct_delivery.cache()
    attr_sum_row = fct_delivery.agg(F.coalesce(F.sum("spend_usd"), F.lit(0.0)).alias("attr_sum")).collect()[0]

    raw_sum = float(raw_totals["raw_sum"])
    rej_sum = float(raw_totals["rej_sum"])
    attr_sum = float(attr_sum_row["attr_sum"])
    conservation_diff = abs((attr_sum + rej_sum) - raw_sum)
    if conservation_diff >= 1e-6:
        raise ValueError(
            f"Spend conservation invariant violated: attributed({attr_sum}) + rejected({rej_sum}) != raw({raw_sum}) (diff={conservation_diff})"
        )

    # 2. gold_brand_monthly
    country_month_win = Window.partitionBy("month_start", "country_iso")
    monthly_brand = (
        fct_delivery.withColumn("month_start", F.trunc(F.col("week_start"), "month"))
        .groupBy("month_start", "country_iso", "brand_id")
        .agg(
            F.sum("spend_usd").alias("spend_usd"),
            F.sum("impressions").alias("impressions"),
            F.sum(
                F.when(
                    F.col("audit_status").isin("domain_verified", "vision_verified", "human_override"),
                    F.col("spend_usd"),
                ).otherwise(F.lit(0.0))
            ).alias("_verified_spend"),
            F.countDistinct("asset_id").cast("int").alias("creative_count"),
        )
        .withColumn("_cat_spend", F.sum("spend_usd").over(country_month_win))
        .withColumn("_cat_impr", F.sum("impressions").over(country_month_win))
        .withColumn("sos_pct", F.when(F.col("_cat_spend") > 0, (F.col("spend_usd") / F.col("_cat_spend")) * 100.0).otherwise(F.lit(0.0)))
        .withColumn("sov_pct", F.when(F.col("_cat_impr") > 0, (F.col("impressions") / F.col("_cat_impr")) * 100.0).otherwise(F.lit(0.0)))
        .withColumn("audit_coverage_pct", F.when(F.col("spend_usd") > 0, F.col("_verified_spend") / F.col("spend_usd")).otherwise(F.lit(0.0)))
        .drop("_verified_spend", "_cat_spend", "_cat_impr")
    )

    # 3. gold_theme_spend_weekly (structurally separate from channel MMM panel!)
    if themes_df is not None:
        active_themes = (
            themes_df.filter(F.col("is_active_run") == F.lit(True))
            .select("media_hash_key", "theme_id", "theme_label")
            .dropDuplicates(["media_hash_key"])
        )
        theme_joined = fct_delivery.join(active_themes, on="media_hash_key", how="inner")
        if ov_records:
            theme_joined = (
                theme_joined.join(
                    ov_asset.select("asset_id", "ov_a_theme"),
                    on="asset_id",
                    how="left",
                )
                .join(
                    ov_hash.select("media_hash_key", "ov_h_theme"),
                    on="media_hash_key",
                    how="left",
                )
                .withColumn("_tkey", F.lower(F.trim(F.col("theme_label"))))
                .join(ov_theme_sdf, on="_tkey", how="left")
                .withColumn(
                    "theme_label",
                    F.coalesce(
                        F.when(F.length(F.trim(F.col("ov_a_theme"))) > 0, F.trim(F.col("ov_a_theme"))),
                        F.when(F.length(F.trim(F.col("ov_h_theme"))) > 0, F.trim(F.col("ov_h_theme"))),
                        F.when(F.length(F.trim(F.col("ov_t_theme"))) > 0, F.trim(F.col("ov_t_theme"))),
                        F.col("theme_label"),
                    ),
                )
                .drop("_tkey", "ov_a_theme", "ov_h_theme", "ov_t_theme")
            )
        theme_spend_weekly = theme_joined.groupBy("week_start", "country_iso", "brand_id", "theme_id", "theme_label").agg(
            F.sum("spend_usd").alias("theme_spend_usd"),
            F.sum("impressions").alias("theme_impressions"),
        )
    else:
        theme_spend_weekly = spark.createDataFrame(
            [],
            schema=T.StructType(
                [
                    T.StructField("week_start", T.DateType(), False),
                    T.StructField("country_iso", T.StringType(), False),
                    T.StructField("brand_id", T.StringType(), False),
                    T.StructField("theme_id", T.StringType(), False),
                    T.StructField("theme_label", T.StringType(), False),
                    T.StructField("theme_spend_usd", T.DoubleType(), False),
                    T.StructField("theme_impressions", T.DoubleType(), False),
                ]
            ),
        )

    # 4. gold_mmm_weekly_panel (with exogenous competitor_spend + HHI + creative age & quality; NO competitor_sov_pct!)
    with_age = (
        split_df.join(assets_df.select("asset_id", "first_seen_date"), on="asset_id", how="left")
        .withColumn(
            "creative_age_weeks",
            F.greatest(F.lit(0.0), F.datediff(F.col("week_start"), F.col("first_seen_date")).cast("double") / 7.0),
        )
    )

    brand_week = with_age.groupBy("week_start", "country_iso", "brand_id").agg(
        F.sum(F.when(F.col("channel_group") == "meta_paid", F.col("spend_usd")).otherwise(0.0)).alias("own_spend_meta_usd"),
        F.sum(F.when(F.col("channel_group") == "youtube_paid", F.col("spend_usd")).otherwise(0.0)).alias("own_spend_youtube_usd"),
        F.sum(F.when(F.col("channel_group") == "tiktok_paid", F.col("spend_usd")).otherwise(0.0)).alias("own_spend_tiktok_usd"),
        F.sum(F.when(F.col("channel_group") == "display", F.col("spend_usd")).otherwise(0.0)).alias("own_spend_display_usd"),
        F.sum("spend_usd").alias("own_spend_total_usd"),
        F.sum("impressions").alias("_raw_own_impr"),
        F.max(F.col("is_spend_modeled").cast("int")).cast("boolean").alias("is_spend_modeled"),
        F.sum(F.when(F.col("is_weekly_interpolated"), F.col("spend_usd")).otherwise(0.0)).alias("_interp_spend"),
        F.sum(F.col("quality_score") * F.col("spend_usd")).alias("_w_quality"),
        F.sum(F.col("creative_age_weeks") * F.col("spend_usd")).alias("_w_age"),
        F.sum(
            F.when(
                F.col("audit_status").isin("domain_verified", "vision_verified", "human_override"),
                F.col("spend_usd"),
            ).otherwise(0.0)
        ).alias("_verified_spend"),
    )

    cw_win = Window.partitionBy("week_start", "country_iso")
    brand_week_full = (
        brand_week
        .withColumn("_cw_spend", F.sum("own_spend_total_usd").over(cw_win))
        .withColumn("_cw_impr", F.sum("_raw_own_impr").over(cw_win))
        .withColumn("competitor_spend_usd", F.greatest(F.lit(0.0), F.col("_cw_spend") - F.col("own_spend_total_usd")))
        .withColumn("competitor_impressions", F.greatest(F.lit(0.0), F.col("_cw_impr") - F.col("_raw_own_impr")))
        .withColumn("_brand_share_sq", F.pow(F.when(F.col("_cw_spend") > 0, (F.col("own_spend_total_usd") / F.col("_cw_spend")) * 100.0).otherwise(0.0), 2.0))
        .withColumn("category_hhi", F.sum("_brand_share_sq").over(cw_win))
        .withColumn(
            "own_impressions_total",
            F.when(F.col("is_spend_modeled") == F.lit(True), F.lit(None).cast("double")).otherwise(F.col("_raw_own_impr")),
        )
        .withColumn("avg_creative_quality_score", F.when(F.col("own_spend_total_usd") > 0, F.col("_w_quality") / F.col("own_spend_total_usd")).otherwise(0.0))
        .withColumn("avg_creative_age_weeks", F.when(F.col("own_spend_total_usd") > 0, F.col("_w_age") / F.col("own_spend_total_usd")).otherwise(0.0))
        .withColumn("audit_coverage_pct", F.when(F.col("own_spend_total_usd") > 0, F.col("_verified_spend") / F.col("own_spend_total_usd")).otherwise(0.0))
        .withColumn("is_weekly_interpolated", F.when(F.col("own_spend_total_usd") > 0, (F.col("_interp_spend") / F.col("own_spend_total_usd")) > 0.5).otherwise(False))
        .drop("_raw_own_impr", "_interp_spend", "_w_quality", "_w_age", "_verified_spend", "_cw_spend", "_cw_impr", "_brand_share_sq")
    )

    conv_outcomes = outcomes_df.filter(F.col("outcome_metric_name") == "conversions")
    proxy_outcomes = outcomes_df.filter(F.col("outcome_metric_name") == "google_trends_index")
    primary_outcomes = conv_outcomes if conv_outcomes.head(1) else proxy_outcomes

    mmm_panel = brand_week_full.join(
        primary_outcomes.select(
            "week_start",
            "country_iso",
            "brand_id",
            "outcome_metric_name",
            "outcome_value",
            "outcome_source",
            "promo_flag",
            "holiday_flag",
            "price_index",
            "macro_seasonality",
        ),
        on=["week_start", "country_iso", "brand_id"],
        how="inner",
    )

    mmm_proxy_panel = brand_week_full.join(
        proxy_outcomes.select(
            "week_start",
            "country_iso",
            "brand_id",
            "outcome_metric_name",
            "outcome_value",
            "outcome_source",
            "promo_flag",
            "holiday_flag",
            "price_index",
            "macro_seasonality",
        ),
        on=["week_start", "country_iso", "brand_id"],
        how="inner",
    )

    write_spark_parquet_atomic(fct_delivery, gold_root / "fct_ad_delivery_weekly", partition_cols=["country_iso"])
    write_spark_parquet_atomic(monthly_brand, gold_root / "gold_brand_monthly", partition_cols=["country_iso"])
    write_spark_parquet_atomic(theme_spend_weekly, gold_root / "gold_theme_spend_weekly", partition_cols=["country_iso"])
    write_spark_parquet_atomic(mmm_panel, gold_root / "gold_mmm_weekly_panel", partition_cols=["country_iso"])
    write_spark_parquet_atomic(mmm_proxy_panel, gold_root / "gold_mmm_proxy_weekly_panel", partition_cols=["country_iso"])

    fct_count = fct_delivery.count()
    mmm_count = mmm_panel.count()
    theme_count = theme_spend_weekly.count()
    fct_delivery.unpersist()

    return {
        "status": "SUCCESS",
        "spend_conservation_diff": conservation_diff,
        "raw_spend_usd": round(raw_sum, 4),
        "attributed_spend_usd": round(attr_sum, 4),
        "rejected_spend_usd": round(rej_sum, 4),
        "fct_delivery_rows": int(fct_count),
        "gold_mmm_rows": int(mmm_count),
        "gold_theme_rows": int(theme_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark Job 2: Silver + HITL Overrides -> Gold")
    parser.add_argument("--silver-dir", required=True)
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--overrides-dir", default=None)
    parser.add_argument("--dissection", default=None)
    parser.add_argument("--themes", default=None)
    args = parser.parse_args()

    spark = get_spark_session("intel-etl-silver-to-gold")
    try:
        res = run_silver_to_gold(
            spark,
            silver_dir=args.silver_dir,
            gold_dir=args.gold_dir,
            outcomes_path=args.outcomes,
            overrides_dir=args.overrides_dir,
            dissection_path=args.dissection,
            themes_path=args.themes,
        )
        print(json.dumps(res, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
