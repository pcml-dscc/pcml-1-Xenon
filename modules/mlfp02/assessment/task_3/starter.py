# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 - Assessment Task 3: Regression Modelling & Interpretation

Complete the `solve()` function. Read problem.md for the full specification.
Every regression is solved in closed form (OLS via least squares; logistic via
Newton-Raphson to the unique MLE) so your numbers must match the independently
re-derived reference. Standardise predictors (z-score) before fitting.

    python grader.py starter.py
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
OLS_FEATURES = [
    "income_imp",
    "age",
    "employment_years",
    "debt_to_income",
    "credit_age_years",
    "num_dependents",
    "edu_ord",
]
LOGIT_FEATURES = [
    "credit_utilization",
    "num_late_payments",
    "previous_defaults",
    "debt_to_income",
    "num_hard_inquiries",
]
EDU_MAP = {
    "primary": 1.0,
    "secondary": 2.0,
    "diploma": 3.0,
    "degree": 4.0,
    "postgraduate": 5.0,
}
TARGET = "loan_amount_sgd"


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)


def _with_intercept(values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(values.shape[0]), values])


def _ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted
    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) @ (y - y.mean())))
    r_squared = 1.0 - rss / tss
    return beta, residuals, rss, r_squared


def _fit_logistic_irls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = np.clip(X @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-12, None)
        xtwx = X.T @ (X * w[:, None])
        score = X.T @ (y - p)
        try:
            step = np.linalg.solve(xtwx, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(xtwx, score, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def solve() -> dict:
    """Return the regression / interpretation answer dict.

    See problem.md for the exact 13 keys and how each is defined.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "sg_credit_scoring.parquet")

    income_median = df.select(pl.col("income_sgd").median()).item()
    df = df.with_columns(
        pl.col("income_sgd").fill_null(income_median).alias("income_imp"),
        pl.col("education").replace(EDU_MAP).cast(pl.Float64).alias("edu_ord"),
    )

    n_obs = df.height
    feature_names = ["intercept", *OLS_FEATURES]

    X_base_raw = df.select(OLS_FEATURES).to_numpy().astype(float)
    X_base_std = _standardize(X_base_raw)
    X = _with_intercept(X_base_std)
    y = df.select(TARGET).to_numpy().ravel().astype(float)

    beta, residuals, rss, r_squared = _ols_fit(X, y)
    n, p = X.shape
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - p)
    sigma2 = rss / (n - p)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))
    t_stats_arr = beta / se
    p_values_arr = 2.0 * stats.t.sf(np.abs(t_stats_arr), df=n - p)
    f_statistic = (r_squared / (p - 1)) / ((1.0 - r_squared) / (n - p))
    f_p_value = stats.f.sf(f_statistic, p - 1, n - p)

    income_std = X_base_std[:, OLS_FEATURES.index("income_imp")]
    age_std = X_base_std[:, OLS_FEATURES.index("age")]
    employment_std = X_base_std[:, OLS_FEATURES.index("employment_years")]
    X_full = np.column_stack(
        [X, income_std**2, age_std * employment_std]
    )
    _, _, rss_full, r_squared_full = _ols_fit(X_full, y)
    q = 2
    p_full = X_full.shape[1]
    partial_f = ((rss - rss_full) / q) / (rss_full / (n - p_full))
    partial_f_p_value = stats.f.sf(partial_f, q, n - p_full)
    delta_r_squared = r_squared_full - r_squared

    X_logit_raw = df.select(LOGIT_FEATURES).to_numpy().astype(float)
    X_logit = _with_intercept(_standardize(X_logit_raw))
    y_logit = df.select("default").to_numpy().ravel().astype(float)
    beta_logit = _fit_logistic_irls(X_logit, y_logit)
    logit_names = ["intercept", *LOGIT_FEATURES]
    odds_ratios = {
        name: float(value)
        for name, value in zip(logit_names, np.exp(beta_logit), strict=True)
    }
    strongest_logit_predictor = LOGIT_FEATURES[
        int(np.argmax(np.abs(beta_logit[1:])))
    ]

    return {
        "n_obs": int(n_obs),
        "coefficients": {
            name: float(value)
            for name, value in zip(feature_names, beta, strict=True)
        },
        "t_stats": {
            name: float(value)
            for name, value in zip(feature_names, t_stats_arr, strict=True)
        },
        "p_values": {
            name: float(value)
            for name, value in zip(feature_names, p_values_arr, strict=True)
        },
        "r_squared": float(r_squared),
        "adj_r_squared": float(adj_r_squared),
        "f_statistic": float(f_statistic),
        "f_p_value": float(f_p_value),
        "partial_f": float(partial_f),
        "partial_f_p_value": float(partial_f_p_value),
        "delta_r_squared": float(delta_r_squared),
        "odds_ratios": odds_ratios,
        "strongest_logit_predictor": strongest_logit_predictor,
    }


if __name__ == "__main__":
    print(solve())
