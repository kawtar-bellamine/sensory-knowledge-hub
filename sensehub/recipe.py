"""Recipe -> sensory profile -> liking, for a mixture design (the cocktail data).

The four juice volumes always add up to the same total, so the natural model is a Scheffé
mixture model on the proportions x_i (sum x_i = 1, no intercept):

linear     y = sum b_i x_i
quadratic  y = sum b_i x_i + sum_{i<j} b_ij x_i x_j

Each response gets the model with the lower leave-one-recipe-out error. Replicated recipes
are left out together, so a recipe is never predicted from its own replicate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import pandas as pd


def proportions(recipes: pd.DataFrame) -> pd.DataFrame:
    """Product x ingredient volume table -> proportions that sum to 1."""
    return recipes.div(recipes.sum(axis=1), axis=0)


def design_matrix(x: pd.DataFrame | np.ndarray, kind: str, names: list[str]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cols = [x[:, i] for i in range(x.shape[1])]
    if kind == "quadratic":
        cols += [x[:, i] * x[:, j] for i, j in combinations(range(x.shape[1]), 2)]
    elif kind != "linear":
        raise ValueError(kind)
    return np.column_stack(cols)


def term_names(names: list[str], kind: str) -> list[str]:
    out = list(names)
    if kind == "quadratic":
        out += [f"{a} x {b}" for a, b in combinations(names, 2)]
    return out


def _groups(x: pd.DataFrame) -> np.ndarray:
    """Same id for identical recipes."""
    keys = [tuple(r) for r in x.round(6).values]
    ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return np.array([ids[k] for k in keys])


def loo_predict(x: pd.DataFrame, y: pd.Series, kind: str, ridge: float = 1e-6) -> pd.Series:
    """Leave-one-recipe-out predictions (replicates are left out together)."""
    names = list(x.columns)
    groups = _groups(x)
    pred = pd.Series(np.nan, index=y.index)
    for g in np.unique(groups):
        test = groups == g
        a = design_matrix(x[~test], kind, names)
        coef = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ y[~test].values)
        pred[test] = design_matrix(x[test], kind, names) @ coef
    return pred


def loo_rmse(x: pd.DataFrame, y: pd.Series, kind: str, ridge: float = 1e-6) -> float:
    """Leave-one-recipe-out RMSE."""
    return float(np.sqrt(np.mean((loo_predict(x, y, kind, ridge) - y) ** 2)))


def replicate_sd(x: pd.DataFrame, y: pd.Series) -> float:
    """Pooled standard deviation between replicated recipes (the noise floor)."""
    groups = _groups(x)
    resid, dof = [], 0
    for g in np.unique(groups):
        v = y[groups == g].values
        if len(v) > 1:
            resid.extend(v - v.mean())
            dof += len(v) - 1
    return float(np.sqrt(np.sum(np.square(resid)) / dof)) if dof else float("nan")


@dataclass
class MixtureModel:
    kind: str
    names: list[str]
    coef: np.ndarray
    loo_rmse: float

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        return design_matrix(x, self.kind, self.names) @ self.coef

    def terms(self) -> pd.Series:
        return pd.Series(self.coef, index=term_names(self.names, self.kind))


def fit_mixture(x: pd.DataFrame, y: pd.Series, kinds=("linear", "quadratic"),
                ridge: float = 1e-6) -> MixtureModel:
    """Fit each candidate model and keep the one with the lowest leave-one-out error."""
    best = None
    for kind in kinds:
        err = loo_rmse(x, y, kind, ridge)
        if best is None or err < best[1]:
            best = (kind, err)
    kind, err = best
    a = design_matrix(x, kind, list(x.columns))
    coef = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ y.values)
    return MixtureModel(kind, list(x.columns), coef, err)


def fit_all(x: pd.DataFrame, responses: pd.DataFrame) -> dict[str, MixtureModel]:
    return {c: fit_mixture(x, responses[c]) for c in responses.columns}


def design_bounds(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"min": x.min(), "max": x.max()})


def candidate_grid(x: pd.DataFrame, step: float = 0.01) -> pd.DataFrame:
    """Recipes inside the tested range of every ingredient, proportions summing to 1."""
    b = design_bounds(x)
    names = list(x.columns)
    axes = [np.arange(b.loc[n, "min"], b.loc[n, "max"] + 1e-9, step) for n in names[:-1]]
    rows = []
    for combo in product(*axes):
        last = 1 - sum(combo)
        if b.loc[names[-1], "min"] - 1e-9 <= last <= b.loc[names[-1], "max"] + 1e-9:
            rows.append((*combo, last))
    return pd.DataFrame(rows, columns=names)


def best_recipe(model: MixtureModel, x: pd.DataFrame, step: float = 0.01) -> tuple[pd.Series, float]:
    grid = candidate_grid(x, step)
    pred = model.predict(grid)
    i = int(np.argmax(pred))
    return grid.iloc[i], float(pred[i])
