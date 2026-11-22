# `intel-etl-spark`

Batch PySpark 2-stage Medallion ETL (Bronze $\rightarrow$ Silver $\rightarrow$ Gold) and deterministic 156-week **EU Automotive** (`BMW`, `Mercedes-Benz`, `Audi`, `Porsche` in `DE`, `FR`, `GB`) synthetic DGP fixture generator.

> **Windows Execution Note:** PySpark requires OpenJDK + Hadoop native binaries. On Windows hosts, run Spark jobs and tests inside the included `Dockerfile` (`docker build -t intel-etl-spark . && docker run --rm intel-etl-spark`), via the workspace root Compose service (`docker compose --profile batch run etl`), or natively on Linux / WSL2 / GitHub Actions.

## Pipeline Stages

1. `scripts/generate_eu_auto_dgp.py` — Emits 156 weeks of raw ad occurrences (including `1/K` comparative ads, off-brand rejected ads, cumulative Meta EU DSA `eu_total_reach` snapshots), seed dissections/themes, `conversions` + `google_trends_index` outcomes, and `eu_auto_dgp_truth.json`.
2. `intel_etl_spark.jobs.bronze_to_silver` — Bronze SHA-256 manifest watermarking, canonical `asset_id` / `media_hash_key` / `country_iso` / monthly FX -> USD, cumulative `eu_total_reach` delta extraction, dynamic partition swap (`silver_ad_delivery_raw`, `dim_creative_asset`).
3. `intel_etl_spark.jobs.silver_to_gold` — Applies `dim_human_overrides` (`asset_id` > `media_hash_key`, latest `created_at` wins) + verified `matched_brands` **before** `1/K` splitting, enforces spend conservation (`< 1e-6`), and outputs `fct_ad_delivery_weekly`, `gold_brand_monthly`, `gold_mmm_weekly_panel` (exogenous `competitor_spend_usd`, no collider `competitor_sov_pct`), and structurally separate `gold_theme_spend_weekly`.
