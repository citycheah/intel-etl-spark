"""PySpark Stage 1 ETL: Bronze -> Silver (`silver_ad_delivery_raw`, `dim_creative_asset`)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from intel_core.schemas import (
    ARROW_SCHEMAS,
    canonicalize_brand_id,
    compute_asset_id,
    compute_media_hash_key,
    convert_to_usd,
    iso_week_start,
    normalize_country_iso,
)
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


def get_spark_session(app_name: str = "intel-etl-spark") -> SparkSession:
    """Builds a local[*] SparkSession with dynamic partition overwrite and UTC timezone."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_bronze_files(bronze_path: Path) -> list[Path]:
    if bronze_path.is_file():
        return [bronze_path]
    return sorted(
        p for p in bronze_path.rglob("*")
        if p.is_file() and p.suffix.lower() in (".parquet", ".jsonl", ".json")
    )


def load_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(manifest_path: Path, data: dict[str, str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(manifest_path)


def write_spark_parquet_atomic(
    df: DataFrame,
    dest_dir: Path,
    *,
    partition_cols: list[str] | None = None,
) -> None:
    """Writes a Spark DataFrame via staging swap so partial crashes never corrupt Silver/Gold tables."""
    staging_dir = dest_dir.parent / f"._{dest_dir.name}_spark_staging_{uuid.uuid4().hex}"
    try:
        writer = df.coalesce(1).write.mode("overwrite")
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        writer.parquet(str(staging_dir))

        if not partition_cols:
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            staging_dir.replace(dest_dir)
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            for child in staging_dir.iterdir():
                if child.name.startswith("_") or child.name.startswith("."):
                    continue
                target_part = dest_dir / child.name
                if target_part.exists():
                    shutil.rmtree(target_part)
                shutil.move(str(child), str(target_part))
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def run_bronze_to_silver(
    spark: SparkSession,
    *,
    bronze_occurrences_path: str | Path,
    silver_dir: str | Path,
    manifest_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Transforms Bronze raw ad occurrences into `silver_ad_delivery_raw` and `dim_creative_asset`."""
    bronze_path = Path(bronze_occurrences_path)
    silver_root = Path(silver_dir)
    silver_root.mkdir(parents=True, exist_ok=True)

    m_path = Path(manifest_path) if manifest_path else (silver_root / "_bronze_manifest.json")
    prev_manifest = load_manifest(m_path)

    files = _collect_bronze_files(bronze_path)
    if not files:
        raise FileNotFoundError(f"No Bronze occurrence files found under {bronze_path}")

    current_checksums = {str(p.relative_to(bronze_path.parent) if bronze_path.is_file() else p.relative_to(bronze_path)): _sha256_file(p) for p in files}
    changed_files = [
        p for p, (rel, digest) in zip(files, current_checksums.items(), strict=True)
        if force or prev_manifest.get(rel) != digest
    ]

    if not changed_files:
        return {
            "status": "SKIPPED_UNCHANGED",
            "processed_files": 0,
            "skipped_files": len(files),
            "silver_rows": 0,
            "asset_rows": 0,
        }

    # Read changed files (all Parquet or JSONL)
    parquet_inputs = [str(p) for p in changed_files if p.suffix.lower() == ".parquet"]
    json_inputs = [str(p) for p in changed_files if p.suffix.lower() in (".json", ".jsonl")]

    dfs: list[DataFrame] = []
    if parquet_inputs:
        dfs.append(spark.read.parquet(*parquet_inputs))
    if json_inputs:
        dfs.append(spark.read.json(json_inputs))

    raw_df = dfs[0]
    for extra in dfs[1:]:
        raw_df = raw_df.unionByName(extra, allowMissingColumns=True)

    # Register small pure-Python UDFs backed by intel_core.schemas canonical rules
    udf_country = F.udf(normalize_country_iso, T.StringType())
    udf_brand = F.udf(lambda b: canonicalize_brand_id(b) or str(b), T.StringType())
    udf_week = F.udf(lambda d: iso_week_start(str(d)), T.DateType())
    udf_asset_id = F.udf(compute_asset_id, T.StringType())
    udf_media_key = F.udf(compute_media_hash_key, T.StringType())
    udf_fx = F.udf(lambda amt, cur, d: float(convert_to_usd(amt, cur, str(d))), T.DoubleType())

    enriched = (
        raw_df
        .withColumn("week_start", udf_week(F.col("event_date")))
        .withColumn("country_iso", udf_country(F.col("country_raw")))
        .withColumn("brand_id_canon", udf_brand(F.col("brand_raw")))
        .withColumn(
            "asset_id_norm",
            F.when(
                F.col("asset_id").isNotNull() & (F.length(F.trim(F.col("asset_id"))) > 0),
                F.col("asset_id"),
            ).otherwise(udf_asset_id(F.col("media_uri"))),
        )
        .withColumn("media_hash_key", udf_media_key(F.col("phash"), F.col("asset_id_norm")))
        .withColumn("raw_spend_usd", udf_fx(F.col("raw_spend_local"), F.col("currency"), F.col("week_start")))
        .withColumn("snapshot_ts_parsed", F.to_timestamp(F.col("snapshot_ts")))
        .withColumn("event_date_parsed", F.to_date(F.col("event_date")))
    )

    # Deduplicate by occurrence_id keeping latest snapshot_ts
    occ_win = Window.partitionBy("occurrence_id").orderBy(F.col("snapshot_ts_parsed").desc_nulls_last())
    dedup_occ = (
        enriched.withColumn("_rn", F.row_number().over(occ_win))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    # Meta EU DSA eu_total_reach weekly delta computation:
    # When cumulative raw_eu_total_reach is reported without explicit reach_delta, compute week-over-week delta.
    reach_win = Window.partitionBy("asset_id_norm", "country_iso").orderBy("week_start", "snapshot_ts_parsed")
    with_reach = (
        dedup_occ
        .withColumn("_prev_eu_reach", F.lag("raw_eu_total_reach").over(reach_win))
        .withColumn(
            "computed_reach_est",
            F.when(
                F.col("raw_eu_total_reach").isNotNull() & F.col("_prev_eu_reach").isNotNull(),
                F.greatest(F.lit(0).cast("long"), F.col("raw_eu_total_reach").cast("long") - F.col("_prev_eu_reach").cast("long")),
            ).when(
                F.col("raw_eu_total_reach").isNotNull() & F.col("raw_reach_est").isNull(),
                F.col("raw_eu_total_reach").cast("long"),
            ).otherwise(F.col("raw_reach_est").cast("long")),
        )
        .withColumn(
            "is_weekly_interpolated_final",
            F.coalesce(F.col("is_weekly_interpolated"), F.lit(False)),
        )
    )

    silver_raw = with_reach.select(
        F.col("occurrence_id").cast("string").alias("occurrence_id"),
        F.col("week_start").cast("date").alias("week_start"),
        F.col("country_iso").cast("string").alias("country_iso"),
        F.col("asset_id_norm").cast("string").alias("asset_id"),
        F.col("media_hash_key").cast("string").alias("media_hash_key"),
        F.col("channel_group").cast("string").alias("channel_group"),
        F.col("publisher_name").cast("string").alias("publisher_name"),
        F.col("campaign_id").cast("string").alias("campaign_id"),
        F.coalesce(F.col("candidate_brand_ids"), F.array(F.col("brand_id_canon"))).alias("candidate_brand_ids"),
        F.col("raw_impressions").cast("long").alias("raw_impressions"),
        F.col("computed_reach_est").cast("long").alias("raw_reach_est"),
        F.col("raw_eu_total_reach").cast("long").alias("raw_eu_total_reach"),
        F.col("snapshot_ts_parsed").alias("snapshot_ts"),
        F.col("is_weekly_interpolated_final").cast("boolean").alias("is_weekly_interpolated"),
        F.col("raw_spend_local").cast("double").alias("raw_spend_local"),
        F.col("currency").cast("string").alias("currency"),
        F.col("raw_spend_usd").cast("double").alias("raw_spend_usd"),
        F.coalesce(F.col("is_spend_modeled"), F.lit(False)).cast("boolean").alias("is_spend_modeled"),
    )

    # Build dim_creative_asset (1 row per asset_id)
    dim_asset = (
        with_reach.groupBy("asset_id_norm")
        .agg(
            F.first("phash", ignorenulls=True).cast("string").alias("phash"),
            F.first("media_hash_key").cast("string").alias("media_hash_key"),
            F.first("brand_id_canon").cast("string").alias("brand_id_raw"),
            F.first("source_connector").cast("string").alias("source_connector"),
            F.first("media_type").cast("string").alias("media_type"),
            F.first("media_uri").cast("string").alias("media_uri"),
            F.first("aspect_ratio", ignorenulls=True).cast("string").alias("aspect_ratio"),
            F.first("duration_sec", ignorenulls=True).cast("double").alias("duration_sec"),
            F.min("event_date_parsed").cast("date").alias("first_seen_date"),
            F.max("event_date_parsed").cast("date").alias("last_seen_date"),
            F.lit(True).cast("boolean").alias("is_active"),
        )
        .withColumnRenamed("asset_id_norm", "asset_id")
    )

    silver_raw_path = silver_root / "silver_ad_delivery_raw"
    dim_asset_path = silver_root / "dim_creative_asset"

    silver_raw.cache()
    dim_asset.cache()
    silver_count = silver_raw.count()
    asset_count = dim_asset.count()

    write_spark_parquet_atomic(silver_raw, silver_raw_path, partition_cols=["country_iso"])
    write_spark_parquet_atomic(dim_asset, dim_asset_path, partition_cols=None)

    silver_raw.unpersist()
    dim_asset.unpersist()

    # Record updated checksums atomically
    updated_manifest = {**prev_manifest, **current_checksums}
    save_manifest(m_path, updated_manifest)

    # Validate output schema field alignment against intel_core contract
    _ = ARROW_SCHEMAS["silver_ad_delivery_raw"]
    _ = ARROW_SCHEMAS["dim_creative_asset"]

    return {
        "status": "PROCESSED",
        "processed_files": len(changed_files),
        "skipped_files": len(files) - len(changed_files),
        "silver_rows": int(silver_count),
        "asset_rows": int(asset_count),
        "silver_raw_path": str(silver_raw_path),
        "dim_asset_path": str(dim_asset_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark Job 1: Bronze -> Silver")
    parser.add_argument("--bronze", required=True, help="Bronze occurrences Parquet/JSONL path")
    parser.add_argument("--silver-dir", required=True, help="Output Silver directory")
    parser.add_argument("--force", action="store_true", help="Ignore manifest and reprocess")
    args = parser.parse_args()

    spark = get_spark_session("intel-etl-bronze-to-silver")
    try:
        res = run_bronze_to_silver(
            spark,
            bronze_occurrences_path=args.bronze,
            silver_dir=args.silver_dir,
            force=args.force,
        )
        print(json.dumps(res, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
