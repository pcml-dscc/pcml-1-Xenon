# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 3: Window Functions & Price Trends

Complete the `solve()` function. Read problem.md for the full specification.
This task is about correct window partitioning and ordering.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


OUTPUT_COLUMNS = [
    "town",
    "sale_year",
    "n_sales",
    "median_price",
    "yoy_pct",
    "rolling_3yr_avg",
    "price_rank_in_year",
]


def solve() -> pl.DataFrame:
    """Per-town, per-year HDB price-trend table (7 columns)."""
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet")

    df = (
        df.with_columns(pl.col("month").str.slice(0, 4).cast(pl.Int64).alias("sale_year"))
        .group_by(["town", "sale_year"])
        .agg(
            pl.median("resale_price").alias("median_price"),
            pl.len().cast(pl.Int64).alias("n_sales"),
        )
        .sort(["town", "sale_year"])
        .with_columns(
            pl.col("median_price").shift(1).over("town").alias("prev_year_median"),
            pl.col("median_price")
            .rolling_mean(window_size=3, min_samples=1)
            .over("town")
            .alias("rolling_3yr_avg"),
            pl.col("median_price")
            .rank(method="min", descending=True)
            .over("sale_year")
            .cast(pl.Int64)
            .alias("price_rank_in_year"),
        )
        .with_columns(
            (
                100
                * (pl.col("median_price") - pl.col("prev_year_median"))
                / pl.col("prev_year_median")
            ).alias("yoy_pct")
        )
        .select(OUTPUT_COLUMNS)
        .sort(["town", "sale_year"])
    )

    return df


if __name__ == "__main__":
    print(solve().head())
