# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 - Assessment Task 2: Hypothesis Testing, Bootstrap & CUPED

Complete the `solve()` function. Read problem.md for the full specification.
The bootstrap is auto-graded against a bit-reproducible reference: you MUST
follow the seed / resample protocol exactly (treatment resampled before
control, every iteration).

    python grader.py starter.py
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
COHORT = ["control", "treatment_a"]
BOOT_SEED = 2024            # np.random.default_rng(BOOT_SEED)
BOOT_B = 2000              # number of bootstrap resamples
MT_P_VALUES = [0.03, 0.012, 0.04, 0.65, 0.009]   # five simultaneous tests
MT_ALPHA = 0.05


def solve() -> dict:
    """Return the hypothesis-testing / bootstrap / CUPED answer dict.

    See problem.md for the exact 13 keys and how each is defined.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")
    co = df.filter(pl.col("experiment_group").is_in(COHORT))

    treatment = co.filter(pl.col("experiment_group") == "treatment_a")
    control = co.filter(pl.col("experiment_group") == "control")
    t = treatment.select("metric_value").to_numpy().ravel().astype(float)
    c = control.select("metric_value").to_numpy().ravel().astype(float)

    welch_t, welch_p = stats.ttest_ind(t, c, equal_var=False)
    mean_diff = t.mean() - c.mean()

    rng = np.random.default_rng(BOOT_SEED)
    diffs = np.empty(BOOT_B)
    for b in range(BOOT_B):
        bt = rng.choice(t, size=t.size, replace=True)
        bc = rng.choice(c, size=c.size, replace=True)
        diffs[b] = bt.mean() - bc.mean()
    boot_ci_low, boot_ci_high = np.percentile(diffs, [2.5, 97.5])

    metric = co.select("metric_value").to_numpy().ravel().astype(float)
    pre = co.select("pre_metric_value").to_numpy().ravel().astype(float)
    cuped_theta = np.cov(metric, pre, ddof=1)[0, 1] / np.var(pre, ddof=1)
    metric_adj = metric - cuped_theta * (pre - pre.mean())
    var_metric = np.var(metric, ddof=1)
    var_adj = np.var(metric_adj, ddof=1)
    cuped_var_reduction = 1.0 - var_adj / var_metric

    groups = co.select("experiment_group").to_numpy().ravel()
    t_adj = metric_adj[groups == "treatment_a"]
    c_adj = metric_adj[groups == "control"]
    welch_t_cuped, welch_p_cuped = stats.ttest_ind(t_adj, c_adj, equal_var=False)

    p_values = np.array(MT_P_VALUES, dtype=float)
    m = p_values.size
    bonferroni_n_sig = int(np.sum(p_values < MT_ALPHA / m))

    sorted_p = np.sort(p_values)
    thresholds = MT_ALPHA * np.arange(1, m + 1) / m
    rejected_ranks = np.flatnonzero(sorted_p <= thresholds)
    bh_n_sig = int(rejected_ranks[-1] + 1) if rejected_ranks.size else 0

    return {
        "welch_t": float(welch_t),
        "welch_p": float(welch_p),
        "mean_diff": float(mean_diff),
        "boot_ci_low": float(boot_ci_low),
        "boot_ci_high": float(boot_ci_high),
        "cuped_theta": float(cuped_theta),
        "var_metric": float(var_metric),
        "var_adj": float(var_adj),
        "cuped_var_reduction": float(cuped_var_reduction),
        "welch_t_cuped": float(welch_t_cuped),
        "welch_p_cuped": float(welch_p_cuped),
        "bonferroni_n_sig": bonferroni_n_sig,
        "bh_n_sig": bh_n_sig,
    }


if __name__ == "__main__":
    print(solve())
