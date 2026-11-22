# `intel-etl-spark`

Batch PySpark 2-stage Medallion ETL (Bronze $\rightarrow$ Silver $\rightarrow$ Gold) and deterministic 156-week **EU Automotive** (`BMW`, `Mercedes-Benz`, `Audi`, `Porsche` in `DE`, `FR`, `GB`) synthetic DGP fixture generator.

> **Windows Execution Note:** PySpark requires OpenJDK + Hadoop native binaries. On Windows hosts, run Spark jobs and tests inside the included `Dockerfile` (`docker build -t intel-etl-spark . && docker run --rm intel-etl-spark`), via the workspace root Compose service (`docker compose --profile batch run etl`), or natively on Linux / WSL2 / GitHub Actions.

## Pipeline Stages

1. `scripts/generate_eu_auto_dgp.py` — Emits 156 weeks of raw ad occurrences (including `1/K` comparative ads, off-brand rejected ads, cumulative Meta EU DSA `eu_total_reach` snapshots), seed dissections/themes, `conversions` + `google_trends_index` outcomes, and `eu_auto_dgp_truth.json`.
2. `intel_etl_spark.jobs.bronze_to_silver` — Bronze SHA-256 manifest watermarking, canonical `asset_id` / `media_hash_key` / `country_iso` / monthly FX -> USD, cumulative `eu_total_reach` delta extraction, dynamic partition swap (`silver_ad_delivery_raw`, `dim_creative_asset`).
3. `intel_etl_spark.jobs.silver_to_gold` — Applies `dim_human_overrides` (`asset_id` > `media_hash_key`, latest `created_at` wins) + verified `matched_brands` **before** `1/K` splitting, enforces spend conservation (`< 1e-6`), and outputs `fct_ad_delivery_weekly`, `gold_brand_monthly`, `gold_mmm_weekly_panel` (exogenous `competitor_spend_usd`, no collider `competitor_sov_pct`), and structurally separate `gold_theme_spend_weekly`.

## Creative & Marketing Intelligence Suite (`v0.1.0`) — Module Map

Part of the 7-repository **`creative-intel-suite`** (flagship UI & copilot in [`brand-analytics-agent`](https://github.com/citycheah/brand-analytics-agent)):

| Component | Role in `creative-intel-suite` (`v0.1.0`) | Stack |
| :--- | :--- | :--- |
| **[`brand-analytics-agent`](https://github.com/citycheah/brand-analytics-agent)** | **Flagship Product Shell** — 5-page Next.js 15 Portal, FastAPI REST/SSE & Google ADK Copilot | Next.js 15, FastAPI, Google ADK, DuckDB |
| **[`intel-core`](https://github.com/citycheah/intel-core)** | **Core SDK** — Canonical contracts, DuckDB HITL views, FinOps, Hill optimizer, Golden Vision evals | Pydantic v2, PyArrow, DuckDB, Gemini |
| **[`intel-etl-spark`](https://github.com/citycheah/intel-etl-spark)** | **Medallion Batch ETL** — Bronze->Silver->Gold Spark jobs, EU Auto DGP & DuckDB local parity | PySpark 3.5, DuckDB, PyArrow |
| **[`intel-mmm`](https://github.com/citycheah/intel-mmm)** | **Causal Incrementality** — 3-tier Bayesian MMM ladder, separate theme attribution & ROAS guardrails | PyMC-Marketing, `nutpie`, Ridge |
| **[`creative-intelligence-toolkit`](https://github.com/citycheah/creative-intelligence-toolkit)** | **Ingestion Connectors** — Meta EU DSA Ad Library, Google Trends, YouTube/TikTok/IG/FB & alerts | Gemini, Playwright, `yt-dlp`, `intel-core` |
| **[`creative-analytics-grouping`](https://github.com/citycheah/creative-analytics-grouping)** | **Theme & Fatigue Engine** — Anti-join unseen dissection, 30-D UMAP+HDBSCAN & wearout tracking | Vertex/Gemini, HDBSCAN, UMAP |
| **[`competitive-velocity-ml-engine`](https://github.com/citycheah/competitive-velocity-ml-engine)** | **Velocity & Drift Engine** — YouTube $t_0$ breakout prediction, dbt PIT features & PSI drift | dbt-duckdb, LightGBM, SHAP |
