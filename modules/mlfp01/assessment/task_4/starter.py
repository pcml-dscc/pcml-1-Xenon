# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 - Assessment Task 4: Profile, Clean & Integrate with DataExplorer

Complete the `solve()` function. Read problem.md for the full specification.
This task uses the kailash-ml DataExplorer engine to PROVE your cleaning
improved data quality.

    python grader.py starter.py
"""
from __future__ import annotations

import asyncio

import polars as pl

from kailash_ml import DataExplorer
from shared import MLFPDataLoader


OUTPUT_COLUMNS = [
    "period_year",
    "period_quarter",
    "gdp_growth_pct",
    "unemployment_rate",
    "inflation_rate",
    "trade_balance_sgd_bn",
    "property_price_index",
    "tourist_arrivals",
]


def solve() -> dict:
    """Return the cleaned quarterly table and DataExplorer alert counts."""
    raw = MLFPDataLoader().load("mlfp01", "economic_indicators.csv")
    return asyncio.run(_solve_async(raw))


async def _solve_async(raw: pl.DataFrame) -> dict:
    quarterly = raw.filter(pl.col("period_type") == "quarterly")

    period_year = (
        pl.coalesce(
            pl.col("period").str.extract(r"^Q[1-4]\s+(\d{4})$", 1),
            pl.col("period").str.extract(r"^(\d{4})-Q[1-4]$", 1),
            pl.col("period").str.extract(r"^(\d{4})-[1-4]$", 1),
        )
        .cast(pl.Int64)
        .alias("period_year")
    )
    period_quarter = (
        pl.coalesce(
            pl.col("period").str.extract(r"^Q([1-4])\s+\d{4}$", 1),
            pl.col("period").str.extract(r"^\d{4}-Q([1-4])$", 1),
            pl.col("period").str.extract(r"^\d{4}-([1-4])$", 1),
        )
        .cast(pl.Int64)
        .alias("period_quarter")
    )

    cleaned = (
        quarterly.with_columns(
            period_year,
            period_quarter,
            pl.col("tourist_arrivals")
            .str.replace_all(",", "")
            .cast(pl.Int64)
            .alias("tourist_arrivals"),
        )
        .with_columns(
            pl.col("inflation_rate")
            .fill_null(pl.col("inflation_rate").median())
            .alias("inflation_rate"),
            pl.col("trade_balance_sgd_bn")
            .fill_null(pl.col("trade_balance_sgd_bn").median())
            .alias("trade_balance_sgd_bn"),
        )
        .select(OUTPUT_COLUMNS)
        .sort(["period_year", "period_quarter"])
    )

    explorer = DataExplorer()
    raw_profile = await explorer.profile(quarterly)
    clean_profile = await explorer.profile(cleaned)

    return {
        "cleaned": cleaned,
        "raw_alert_count": len(raw_profile.alerts),
        "clean_alert_count": len(clean_profile.alerts),
    }


if __name__ == "__main__":
    print(solve())
