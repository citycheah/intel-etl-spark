import datetime as dt
import os
import shutil

import pandas as pd
import pytest
from intel_core.schemas import validate_records
from intel_core.storage import HumanOverrideStore, read_parquet_table

from intel_etl_spark.jobs.bronze_to_silver import get_spark_session, run_bronze_to_silver
from intel_etl_spark.jobs.silver_to_gold import run_silver_to_gold
from intel_etl_spark.local_runner import run_local_bronze_to_silver, run_local_silver_to_gold
from scripts.generate_eu_auto_dgp import generate_eu_auto_dgp


def _spark_runtime_available() -> bool:
    if os.name == "nt" and not os.environ.get("HADOOP_HOME") and not os.environ.get("SPARK_DOCKER_TEST"):
        # Check if java exists; Windows PySpark without winutils.exe cannot write Hadoop Parquet files directly.
        return False
    return shutil.which("java") is not None


def test_eu_auto_dgp_generator_offline(tmp_path):
    outputs = generate_eu_auto_dgp(tmp_path / "dgp", num_weeks=12, seed=42)
    assert outputs["bronze_occurrences"].exists()
    assert outputs["truth_json"].exists()
    occ_df = pd.read_parquet(outputs["bronze_occurrences"])
    assert len(occ_df) > 100
    assert occ_df["phash"].isna().any()  # Null phash edge case included


def test_local_medallion_parity_and_conservation(tmp_path):
    dgp_dir = tmp_path / "dgp"
    outputs = generate_eu_auto_dgp(dgp_dir, num_weeks=10, seed=7)
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    overrides_dir = tmp_path / "overrides"

    res1 = run_local_bronze_to_silver(
        bronze_occurrences_path=outputs["bronze_occurrences"],
        silver_dir=silver_dir,
    )
    assert res1["status"] == "PROCESSED"
    assert res1["silver_rows"] > 0

    # Manifest skip on unchanged Bronze file
    res_skip = run_local_bronze_to_silver(
        bronze_occurrences_path=outputs["bronze_occurrences"],
        silver_dir=silver_dir,
    )
    assert res_skip["status"] == "SKIPPED_UNCHANGED"

    # Human override on rejected reseller creative (media_hash_key 9988776655443322) -> Porsche
    ov_store = HumanOverrideStore(overrides_dir)
    ov_store.write_override(
        target_key_type="media_hash_key",
        target_key_value="9988776655443322",
        field_name="matched_brand_id",
        human_value="Porsche",
        reviewer="qa_engineer",
        created_at=dt.datetime(2026, 2, 1, 12, 5, tzinfo=dt.UTC),
    )

    res2 = run_local_silver_to_gold(
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        outcomes_path=outputs["outcomes"],
        overrides_dir=overrides_dir,
        dissection_path=outputs["seed_dissection"],
        themes_path=outputs["seed_themes"],
    )
    assert res2["status"] == "SUCCESS"
    assert res2["spend_conservation_diff"] < 1e-6
    assert res2["gold_mmm_rows"] > 0
    assert res2["gold_theme_rows"] > 0

    mmm_df = read_parquet_table(gold_dir / "gold_mmm_weekly_panel")
    assert "competitor_sov_pct" not in mmm_df.columns
    assert "competitor_spend_usd" in mmm_df.columns
    validate_records("gold_mmm_weekly_panel", mmm_df.head(10).to_dict(orient="records"))

    fct_df = read_parquet_table(gold_dir / "fct_ad_delivery_weekly")
    assert (fct_df["audit_status"] == "human_override").any()
    assert (fct_df["attribution_k"] == 2).any()
    validate_records("fct_ad_delivery_weekly", fct_df.head(10).to_dict(orient="records"))

    # Now verify "Unverified / Other" rejection + theme_label override preserve spend conservation
    target_row = fct_df[fct_df["brand_id"] == "BMW"].iloc[0]
    target_aid = str(target_row["asset_id"])
    t_base = dt.datetime(2026, 2, 2, 10, 0, tzinfo=dt.UTC)
    ov_store.write_override(
        target_key_type="asset_id",
        target_key_value=target_aid,
        field_name="belongs_to_brand",
        human_value="true",
        reviewer="qa",
        created_at=t_base,
    )
    ov_store.write_override(
        target_key_type="asset_id",
        target_key_value=target_aid,
        field_name="matched_brand_id",
        human_value="Unverified / Other",
        reviewer="qa",
        created_at=t_base + dt.timedelta(minutes=1),
    )
    seed_themes = pd.read_parquet(outputs["seed_themes"])
    orig_theme = str(seed_themes.loc[seed_themes["is_active_run"].astype(bool), "theme_label"].iloc[0])
    ov_store.write_override(
        target_key_type="theme_label",
        target_key_value=orig_theme,
        field_name="theme_label",
        human_value="Renamed EV Theme",
        reviewer="qa",
        created_at=t_base,
    )

    gold_dir2 = tmp_path / "gold_after_reject"
    res3 = run_local_silver_to_gold(
        silver_dir=silver_dir,
        gold_dir=gold_dir2,
        outcomes_path=outputs["outcomes"],
        overrides_dir=overrides_dir,
        dissection_path=outputs["seed_dissection"],
        themes_path=outputs["seed_themes"],
    )
    assert res3["status"] == "SUCCESS"
    assert res3["spend_conservation_diff"] < 1e-6
    fct2 = read_parquet_table(gold_dir2 / "fct_ad_delivery_weekly")
    assert target_aid not in set(fct2["asset_id"])
    thm2 = read_parquet_table(gold_dir2 / "gold_theme_spend_weekly")
    assert "Renamed EV Theme" in set(thm2["theme_label"])


@pytest.mark.skipif(
    not _spark_runtime_available(),
    reason="PySpark Parquet writes on Windows require Docker (Dockerfile) or Linux/HADOOP_HOME.",
)
def test_medallion_spark_etl_end_to_end(tmp_path):
    dgp_dir = tmp_path / "dgp"
    outputs = generate_eu_auto_dgp(dgp_dir, num_weeks=10, seed=7)
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    overrides_dir = tmp_path / "overrides"

    spark = get_spark_session("test-intel-etl-spark")
    try:
        res1 = run_bronze_to_silver(
            spark,
            bronze_occurrences_path=outputs["bronze_occurrences"],
            silver_dir=silver_dir,
        )
        assert res1["status"] == "PROCESSED"
        assert res1["silver_rows"] > 0

        # Idempotent manifest skip on unchanged Bronze files
        res_skip = run_bronze_to_silver(
            spark,
            bronze_occurrences_path=outputs["bronze_occurrences"],
            silver_dir=silver_dir,
        )
        assert res_skip["status"] == "SKIPPED_UNCHANGED"

        # Add an HITL override on a rejected reseller creative (media_hash_key 9988776655443322)
        ov_store = HumanOverrideStore(overrides_dir)
        ov_store.write_override(
            target_key_type="media_hash_key",
            target_key_value="9988776655443322",
            field_name="belongs_to_brand",
            human_value="true",
            reviewer="qa_engineer",
            created_at=dt.datetime(2026, 2, 1, 12, 0, tzinfo=dt.UTC),
        )
        ov_store.write_override(
            target_key_type="media_hash_key",
            target_key_value="9988776655443322",
            field_name="matched_brand_id",
            human_value="Porsche",
            reviewer="qa_engineer",
            created_at=dt.datetime(2026, 2, 1, 12, 5, tzinfo=dt.UTC),
        )

        res2 = run_silver_to_gold(
            spark,
            silver_dir=silver_dir,
            gold_dir=gold_dir,
            outcomes_path=outputs["outcomes"],
            overrides_dir=overrides_dir,
            dissection_path=outputs["seed_dissection"],
            themes_path=outputs["seed_themes"],
        )
        assert res2["status"] == "SUCCESS"
        assert res2["spend_conservation_diff"] < 1e-6
        assert res2["gold_mmm_rows"] > 0
        assert res2["gold_theme_rows"] > 0

        # Contract validation against intel-core schemas
        mmm_df = read_parquet_table(gold_dir / "gold_mmm_weekly_panel")
        assert "competitor_sov_pct" not in mmm_df.columns
        assert "competitor_spend_usd" in mmm_df.columns
        validate_records("gold_mmm_weekly_panel", mmm_df.head(10).to_dict(orient="records"))

        fct_df = read_parquet_table(gold_dir / "fct_ad_delivery_weekly")
        assert (fct_df["audit_status"] == "human_override").any()
        assert (fct_df["attribution_k"] == 2).any()
        validate_records("fct_ad_delivery_weekly", fct_df.head(10).to_dict(orient="records"))
    finally:
        spark.stop()
