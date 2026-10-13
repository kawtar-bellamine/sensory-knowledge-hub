"""Questions a product team asks of the coffee cupping data, answered with honest statistics.

Comparisons between origins and processes are confounded (each country favours some
processes), so effects are estimated within country: a linear model with one dummy per
country. Lots from the same owner are not independent, so standard errors are clustered by
owner (CR1); without an owner column they fall back to heteroskedasticity-robust HC1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

GRADED = ["aroma", "flavor", "aftertaste", "acidity", "body", "balance", "cupper_points"]


def ols_hc1(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and HC1 robust standard errors."""
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    meat = X.T @ (X * resid[:, None] ** 2)
    cov = xtx_inv @ meat @ xtx_inv * n / (n - k)
    return beta, np.sqrt(np.diag(cov))


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and cluster-robust (CR1) standard errors."""
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    codes, uniq = pd.factorize(pd.Series(groups))
    g = len(uniq)
    scores = np.zeros((g, k))
    np.add.at(scores, codes, X * resid[:, None])
    cov = xtx_inv @ (scores.T @ scores) @ xtx_inv * g / (g - 1) * (n - 1) / (n - k)
    return beta, np.sqrt(np.diag(cov))


def _fit(y, X, d):
    if "owner" in d:
        owner = d.owner.astype(object).where(d.owner.notna(), [f"_lot{i}" for i in range(len(d))])
        return ols_cluster(y, X, owner.values)
    return ols_hc1(y, X)


def holm(p: pd.Series) -> pd.Series:
    """Holm adjustment for multiple tests."""
    order = p.sort_values()
    m = len(order)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(order.values)])
    return pd.Series(np.minimum(adj, 1), index=order.index).reindex(p.index)


def _country_dummies(country: pd.Series) -> np.ndarray:
    d = pd.get_dummies(country, drop_first=False).astype(float).values
    return d  # one column per country, no global intercept


def attribute_structure(scores: pd.DataFrame) -> dict:
    """How much the graded attributes say the same thing."""
    z = (scores - scores.mean()) / scores.std()
    eig = np.linalg.eigvalsh(np.corrcoef(z.T.values))[::-1]
    corr = scores.corr()
    off = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    return {
        "first_component_share": float(eig[0] / eig.sum()),
        "median_correlation": float(off.median()),
        "min_correlation": float(off.min()),
        "correlation": corr,
    }


def process_effect(df: pd.DataFrame, a: str = "Natural", b: str = "Washed",
                   min_per_country: int = 3) -> pd.DataFrame:
    """Difference a - b on each attribute, within country.

    Only countries with at least `min_per_country` lots of each process contribute.
    """
    all_naive = df.loc[df.process == a, "total_cup_points"].mean() - df.loc[df.process == b, "total_cup_points"].mean()
    d = df[df.process.isin([a, b])]
    counts = d.groupby(["country", "process"]).size().unstack(fill_value=0)
    keep = counts[(counts[a] >= min_per_country) & (counts[b] >= min_per_country)].index
    d = d[d.country.isin(keep)]
    X = np.column_stack([(d.process == a).astype(float), _country_dummies(d.country)])
    rows = []
    for attr in GRADED + ["total_cup_points"]:
        beta, se = _fit(d[attr].values.astype(float), X, d)
        t = beta[0] / se[0]
        naive = d.loc[d.process == a, attr].mean() - d.loc[d.process == b, attr].mean()
        rows.append({"attribute": attr, "naive_diff": naive, "within_country_diff": beta[0],
                     "ci_low": beta[0] - 1.96 * se[0], "ci_high": beta[0] + 1.96 * se[0],
                     "p": 2 * stats.norm.sf(abs(t))})
    out = pd.DataFrame(rows).set_index("attribute")
    out["p_holm"] = holm(out.p)
    out.attrs.update(countries=list(keep), n_lots=len(d), naive_total_all_lots=float(all_naive),
                     n_a=int((d.process == a).sum()), n_b=int((d.process == b).sum()))
    return out


def altitude_effect(df: pd.DataFrame, per: float = 500.0) -> pd.DataFrame:
    """Change of each attribute per `per` metres of altitude: across all lots, and within
    country and process (comparing farms of the same country)."""
    d = df.dropna(subset=["altitude_m"])
    d = d[d.groupby("country").country.transform("size") >= 5]
    alt = d.altitude_m.values / per
    proc = pd.get_dummies(d.process, drop_first=True).astype(float).values
    rows = []
    for attr in GRADED + ["total_cup_points"]:
        y = d[attr].values.astype(float)
        b0, se0 = ols_hc1(y, np.column_stack([np.ones(len(d)), alt]))
        b1, se1 = _fit(y, np.column_stack([alt, proc, _country_dummies(d.country)]), d)
        rows.append({"attribute": attr, "naive_slope": b0[1],
                     "within_country_slope": b1[0],
                     "ci_low": b1[0] - 1.96 * se1[0], "ci_high": b1[0] + 1.96 * se1[0],
                     "p": 2 * stats.norm.sf(abs(b1[0] / se1[0]))})
    out = pd.DataFrame(rows).set_index("attribute")
    out["p_holm"] = holm(out.p)
    out.attrs.update(n_lots=len(d), n_countries=int(d.country.nunique()), per_m=per)
    return out
