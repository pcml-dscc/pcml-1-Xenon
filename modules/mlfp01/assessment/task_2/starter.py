# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 2: HDB Feature Engineering

Complete the `solve()` function. Read problem.md for the full specification.
The raw strings are deliberately messy — read the data before you parse it.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


OUTPUT_COLUMNS = [
    "town",
    "flat_type",
    "flat_type_rooms",
    "sale_year",
    "storey_midpoint",
    "floor_area_sqm",
    "flat_age_years",
    "remaining_lease_years",
    "resale_price",
    "price_per_sqm",
]


def solve() -> pl.DataFrame:
    """Engineer the 10-column feature table from raw HDB resale data."""
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet")

    storey_low = (
        pl.col("storey_range")
        .str.extract(r"^([0-9O]+) TO ([0-9O]+)$", 1)
        .str.replace_all("O", "0")
        .cast(pl.Float64)
    )
    storey_high = (
        pl.col("storey_range")
        .str.extract(r"^([0-9O]+) TO ([0-9O]+)$", 2)
        .str.replace_all("O", "0")
        .cast(pl.Float64)
    )

    lease_years = pl.col("remaining_lease").str.extract(r"^(\d+)", 1).cast(pl.Float64)
    lease_months = (
        pl.col("remaining_lease")
        .str.extract(r"years\s+(\d+)\s+months", 1)
        .cast(pl.Float64)
        .fill_null(0.0)
    )
    parsed_remaining_lease = lease_years + (lease_months / 12)

    df = (
        df.with_columns(
            pl.col("month").str.slice(0, 4).cast(pl.Int64).alias("sale_year"),
            ((storey_low + storey_high) / 2).alias("storey_midpoint"),
        )
        .with_columns(
            (pl.col("sale_year") - pl.col("lease_commence_date")).alias(
                "flat_age_years"
            ),
            (pl.col("resale_price") / pl.col("floor_area_sqm")).alias(
                "price_per_sqm"
            ),
            pl.when(pl.col("flat_type") == "2 ROOM")
            .then(pl.lit(2))
            .when(pl.col("flat_type") == "3 ROOM")
            .then(pl.lit(3))
            .when(pl.col("flat_type") == "4 ROOM")
            .then(pl.lit(4))
            .when(pl.col("flat_type") == "5 ROOM")
            .then(pl.lit(5))
            .when(pl.col("flat_type") == "EXECUTIVE")
            .then(pl.lit(6))
            .when(pl.col("flat_type") == "MULTI-GENERATION")
            .then(pl.lit(7))
            .cast(pl.Int64)
            .alias("flat_type_rooms"),
        )
        .with_columns(
            parsed_remaining_lease.fill_null(
                (99 - pl.col("flat_age_years")).cast(pl.Float64)
            ).alias("remaining_lease_years")
        )
        .select(OUTPUT_COLUMNS)
        .sort(["sale_year", "town"])
    )

    return df


if __name__ == "__main__":
    print(solve().head())
