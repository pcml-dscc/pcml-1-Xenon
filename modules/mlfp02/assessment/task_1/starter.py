# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 - Assessment Task 1: Probability, Bayes & Experiment Validation

Complete the `solve()` function. Read problem.md for the full specification.
Your submission is auto-graded: every probability, the SRM chi-square, the
base-rate Bayes scalar, and the Beta posterior must match the independently
re-derived reference within tight tolerances.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
COHORT = ["control", "treatment_a"]
CONVERT_THRESHOLD = 50.0          # converted := metric_value >= 50.0
FRAUD_BASE_RATE = 0.02            # P(fraud)
FRAUD_SENSITIVITY = 0.95          # P(flagged | fraud)
FRAUD_FPR = 0.03                  # P(flagged | not fraud)
BETA_PRIOR_ALPHA = 2.0
BETA_PRIOR_BETA = 20.0


def solve() -> dict:
    """Return the probability / Bayes / experiment-validation answer dict.

    See problem.md for the exact 13 keys and how each is defined.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")

    cohort = (
        df.filter(pl.col("experiment_group").is_in(COHORT))
        .with_columns(
            (pl.col("metric_value") >= CONVERT_THRESHOLD).alias("converted")
        )
    )

    overall = cohort.select(
        pl.len().alias("n_total"),
        pl.col("converted").mean().alias("p_convert_overall"),
        (pl.col("experiment_group") == "treatment_a").mean().alias("p_treatment"),
    ).row(0, named=True)

    by_group = {
        row["experiment_group"]: row
        for row in cohort.group_by("experiment_group").agg(
            pl.len().alias("n"),
            pl.col("converted").sum().alias("successes"),
            pl.col("converted").mean().alias("p_convert"),
        ).iter_rows(named=True)
    }

    n_control = by_group["control"]["n"]
    n_treatment = by_group["treatment_a"]["n"]
    n_total = overall["n_total"]
    expected = n_total / 2.0
    srm_chi2 = ((n_control - expected) ** 2 + (n_treatment - expected) ** 2) / expected
    srm_p_value = float(stats.chi2.sf(srm_chi2, df=1))

    p_convert_treatment = by_group["treatment_a"]["p_convert"]
    p_convert_overall = overall["p_convert_overall"]
    p_treatment_given_convert = (
        p_convert_treatment * overall["p_treatment"] / p_convert_overall
    )

    p_fraud_given_flagged = (
        FRAUD_SENSITIVITY
        * FRAUD_BASE_RATE
        / (
            FRAUD_SENSITIVITY * FRAUD_BASE_RATE
            + FRAUD_FPR * (1.0 - FRAUD_BASE_RATE)
        )
    )

    successes = by_group["treatment_a"]["successes"]
    failures = n_treatment - successes
    beta_post_alpha = BETA_PRIOR_ALPHA + successes
    beta_post_beta = BETA_PRIOR_BETA + failures
    posterior_mean = beta_post_alpha / (beta_post_alpha + beta_post_beta)
    cred_int_low = float(stats.beta.ppf(0.025, beta_post_alpha, beta_post_beta))
    cred_int_high = float(stats.beta.ppf(0.975, beta_post_alpha, beta_post_beta))

    return {
        "p_convert_overall": p_convert_overall,
        "p_convert_control": by_group["control"]["p_convert"],
        "p_convert_treatment": p_convert_treatment,
        "p_treatment_given_convert": p_treatment_given_convert,
        "srm_chi2": srm_chi2,
        "srm_p_value": srm_p_value,
        "srm_flag": bool(srm_p_value < 1e-3),
        "p_fraud_given_flagged": p_fraud_given_flagged,
        "beta_post_alpha": beta_post_alpha,
        "beta_post_beta": beta_post_beta,
        "posterior_mean": posterior_mean,
        "cred_int_low": cred_int_low,
        "cred_int_high": cred_int_high,
    }


if __name__ == "__main__":
    print(solve())
