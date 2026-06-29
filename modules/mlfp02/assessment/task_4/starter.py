# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 - Assessment Task 4: Feature Engineering & Feature Store

Complete the `solve()` function. Read problem.md for the full specification.
You join five raw ICU tables into one admission-level feature table. The event
tables are messy: lab values contain junk strings, doses are like "34.8MG", and
many admissions have no vitals/labs at all. Your output is auto-graded
column-by-column against an independent re-derivation.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader

DT_FMT = "%Y-%m-%d %H:%M:%S"

FEATURE_COLUMNS = [
    "admission_id",
    "feature_timestamp",
    "age",
    "gender",
    "bmi",
    "diagnosis",
    "icu_type",
    "mean_heart_rate",
    "mean_systolic_bp",
    "min_spo2",
    "max_temperature",
    "n_vitals",
    "n_labs",
    "n_abnormal_labs",
    "mean_creatinine",
    "n_distinct_drugs",
    "n_iv_meds",
    "total_dose_mg",
    "los_days",
]


def solve() -> pl.DataFrame:
    """Build the 19-column admission-level feature-store table.

    See problem.md for the exact columns, aggregations, parsing rules, and the
    imputation policy. Return the table sorted ascending by admission_id.
    """
    loader = MLFPDataLoader()
    adm = loader.load("mlfp02", "icu_admissions.parquet")
    pat = loader.load("mlfp02", "icu_patients.parquet")
    vit = loader.load("mlfp02", "icu_vitals.parquet")
    labs = loader.load("mlfp02", "icu_labs.parquet")
    meds = loader.load("mlfp02", "icu_medications.parquet")

    base = (
        adm.select(
            "admission_id",
            "patient_id",
            "diagnosis",
            "icu_type",
            "los_days",
            pl.col("admit_time").str.strptime(pl.Datetime, format=DT_FMT).alias(
                "feature_timestamp"
            ),
        )
        .join(pat.select("patient_id", "age", "gender", "bmi"), on="patient_id", how="left")
    )

    vitals = vit.group_by("admission_id").agg(
        pl.col("heart_rate").mean().alias("mean_heart_rate"),
        pl.col("systolic_bp").mean().alias("mean_systolic_bp"),
        pl.col("spo2").min().alias("min_spo2"),
        pl.col("temperature").max().alias("max_temperature"),
        pl.len().alias("n_vitals"),
    )

    lab_features = (
        labs.with_columns(
            pl.col("value").cast(pl.Float64, strict=False).alias("value_num"),
            pl.col("flag").str.to_lowercase().alias("flag_lower"),
        )
        .group_by("admission_id")
        .agg(
            pl.len().alias("n_labs"),
            (pl.col("flag_lower") == "abnormal").sum().alias("n_abnormal_labs"),
            pl.when(pl.col("test_name") == "Creatinine")
            .then(pl.col("value_num"))
            .otherwise(None)
            .mean()
            .alias("mean_creatinine"),
        )
    )

    med_features = (
        meds.with_columns(
            pl.col("dose")
            .str.extract(r"([0-9]+\.?[0-9]*)", group_index=1)
            .cast(pl.Float64, strict=False)
            .alias("dose_mg")
        )
        .group_by("admission_id")
        .agg(
            pl.col("drug_name").n_unique().alias("n_distinct_drugs"),
            (pl.col("route") == "IV").sum().alias("n_iv_meds"),
            pl.col("dose_mg").sum().alias("total_dose_mg"),
        )
    )

    result = (
        base.join(vitals, on="admission_id", how="left")
        .join(lab_features, on="admission_id", how="left")
        .join(med_features, on="admission_id", how="left")
    )

    median_cols = [
        "age",
        "bmi",
        "mean_heart_rate",
        "mean_systolic_bp",
        "min_spo2",
        "max_temperature",
        "mean_creatinine",
    ]
    medians = result.select(
        [pl.col(col).cast(pl.Float64).median().alias(col) for col in median_cols]
    ).row(0, named=True)

    count_cols = [
        "n_vitals",
        "n_labs",
        "n_abnormal_labs",
        "n_distinct_drugs",
        "n_iv_meds",
    ]

    result = result.with_columns(
        pl.col("gender").fill_null("Unknown"),
        pl.col("total_dose_mg").fill_null(0.0).cast(pl.Float64),
        *[pl.col(col).fill_null(0).cast(pl.Int64) for col in count_cols],
        *[
            pl.col(col).cast(pl.Float64).fill_null(float(medians[col]))
            for col in median_cols
        ],
    )

    return result.select(FEATURE_COLUMNS).sort("admission_id")


if __name__ == "__main__":
    print(solve().head())
