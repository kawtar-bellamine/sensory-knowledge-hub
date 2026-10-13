"""Sensory maps, consumer segments, preference maps and drivers of liking.

The classic chain of a sensory and consumer study:

1. a principal component analysis (PCA) of the panel profiles gives a sensory map;
2. consumers are grouped by the shape of their preferences (segments);
3. each consumer's liking is regressed on the map (external preference mapping), which shows
   where on the map most consumers would like a product to be;
4. attributes are related to liking (drivers of liking).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score


@dataclass
class SensoryMap:
    scores: pd.DataFrame       # product coordinates on the components
    loadings: pd.DataFrame     # correlation of each attribute with each component
    explained: np.ndarray      # share of variance of each component


def sensory_map(profiles: pd.DataFrame, n_components: int = 2) -> SensoryMap:
    """Standardised PCA of a product x attribute table."""
    z = (profiles - profiles.mean()) / profiles.std(ddof=0)
    z = z.loc[:, profiles.std(ddof=0) > 0]
    u, s, vt = np.linalg.svd(z.values, full_matrices=False)
    k = min(n_components, len(s))
    explained = s ** 2 / (s ** 2).sum()
    scores = pd.DataFrame(u[:, :k] * s[:k], index=profiles.index,
                          columns=[f"PC{i + 1}" for i in range(k)])
    # sign convention: the attribute with the largest loading is positive
    for i, c in enumerate(scores.columns):
        if vt[i][np.argmax(np.abs(vt[i]))] < 0:
            scores[c] *= -1
            vt[i] *= -1
    loadings = pd.DataFrame(
        {c: [np.corrcoef(z[a], scores[c])[0, 1] for a in z.columns] for c in scores.columns},
        index=z.columns)
    return SensoryMap(scores, loadings, explained[:k])


def centre_consumers(liking: pd.DataFrame) -> pd.DataFrame:
    """Remove each consumer's own mean (some consumers score everything high)."""
    return liking - liking.mean(axis=0)


def segment_consumers(liking: pd.DataFrame, k_range=range(2, 5)) -> tuple[pd.Series, pd.DataFrame]:
    """Ward clustering of consumers on their centred liking; k chosen by silhouette.

    Returns the segment of each consumer and the silhouette of each k tried.
    Consumers who give the same score to every product carry no preference and are put
    in segment 0 ("flat").
    """
    c = centre_consumers(liking).T
    flat = c.abs().sum(axis=1) == 0
    x = c[~flat]
    tree = linkage(x.values, method="ward")
    scores = []
    for k in k_range:
        lab = fcluster(tree, k, criterion="maxclust")
        scores.append({"k": k, "silhouette": silhouette_score(x.values, lab)})
    scores = pd.DataFrame(scores)
    best = int(scores.loc[scores.silhouette.idxmax(), "k"])
    lab = pd.Series(fcluster(tree, best, criterion="maxclust"), index=x.index)
    # name segments by size, largest first
    order = lab.value_counts().index
    lab = lab.map({old: new + 1 for new, old in enumerate(order)})
    seg = pd.Series(0, index=c.index)
    seg.loc[lab.index] = lab
    return seg, scores


def segment_profiles(liking: pd.DataFrame, segments: pd.Series) -> pd.DataFrame:
    """Mean liking of each product in each segment (raw scale)."""
    out = {f"segment {s}": liking.loc[:, segments[segments == s].index].mean(axis=1)
           for s in sorted(segments.unique()) if s > 0}
    out["all consumers"] = liking.mean(axis=1)
    return pd.DataFrame(out)


def preference_map(scores: pd.DataFrame, liking: pd.DataFrame, model: str = "vector",
                   resolution: int = 60, margin: float = 0.15):
    """External preference mapping.

    Each consumer's centred liking is regressed on the two map coordinates:
    vector model     liking = a + b1 x + b2 y
    circular model   liking = a + b1 x + b2 y + c (x^2 + y^2)   (needs more products)

    Returns a grid (x, y) and, at every point, the share of consumers whose predicted liking is
    above their own average, plus the in-sample R^2 of each consumer's model.
    """
    xy = scores[["PC1", "PC2"]].loc[liking.index].values
    sd = xy.std(axis=0)
    u = xy / sd
    cols = [np.ones(len(u)), u[:, 0], u[:, 1]]
    if model == "circular":
        cols.append((u ** 2).sum(axis=1))
    design = np.column_stack(cols)
    if design.shape[0] <= design.shape[1]:
        raise ValueError("not enough products for this model")
    y = centre_consumers(liking).values
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    ss_tot = (y ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 0, 1 - ((y - fitted) ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, 1), np.nan)

    lo, hi = xy.min(axis=0), xy.max(axis=0)
    pad = (hi - lo) * margin
    gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], resolution)
    gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], resolution)
    X, Y = np.meshgrid(gx, gy)
    gu = np.column_stack([X.ravel() / sd[0], Y.ravel() / sd[1]])
    gcols = [np.ones(len(gu)), gu[:, 0], gu[:, 1]]
    if model == "circular":
        gcols.append((gu ** 2).sum(axis=1))
    pred = np.column_stack(gcols) @ coef
    share = (pred > 0).mean(axis=1).reshape(X.shape)
    return X, Y, share, pd.Series(r2, index=liking.columns)


def drivers_of_liking(profiles: pd.DataFrame, liking: pd.DataFrame, n_boot: int = 2000,
                      seed: int = 0) -> pd.DataFrame:
    """Correlation between each attribute and mean liking across products.

    The interval resamples consumers (how much the answer depends on who was asked); the
    p-value treats products as the sample (how much it depends on which products were tested).
    With few products, the p-value is the binding constraint.
    """
    rng = np.random.default_rng(seed)
    prof = profiles.loc[liking.index]
    mean_like = liking.mean(axis=1)
    n = liking.shape[1]
    boot = np.empty((n_boot, prof.shape[1]))
    for b in range(n_boot):
        m = liking.iloc[:, rng.integers(0, n, n)].mean(axis=1).values
        boot[b] = [np.corrcoef(prof[a].values, m)[0, 1] for a in prof.columns]
    rows = []
    for j, a in enumerate(prof.columns):
        r, p = stats.pearsonr(prof[a].values, mean_like.values)
        lo, hi = np.nanpercentile(boot[:, j], [2.5, 97.5])
        rows.append({"attribute": a, "r": r, "ci_low": lo, "ci_high": hi, "p_products": p})
    return pd.DataFrame(rows).set_index("attribute").sort_values("r", ascending=False)


def preference_structure_test(liking: pd.DataFrame, n_perm: int = 2000, seed: int = 0) -> dict:
    """Do consumers share preference directions, or is the spread just noise?

    Statistic: share of variance carried by the first principal component of the centred
    product x consumer liking table. Null: each consumer's scores are shuffled across products
    independently, which keeps every consumer's scale but destroys any shared structure.
    """
    rng = np.random.default_rng(seed)

    def first_share(m: np.ndarray) -> float:
        s = np.linalg.svd(m - m.mean(axis=0), compute_uv=False)
        return float(s[0] ** 2 / (s ** 2).sum())

    m = liking.values.astype(float)
    obs = first_share(m)
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = first_share(rng.permuted(m, axis=0))
    u, s, vt = np.linalg.svd(m - m.mean(axis=0), full_matrices=False)
    return {"first_share": obs, "null_median": float(np.median(null)),
            "null_95": float(np.quantile(null, 0.95)),
            "p": float((1 + (null >= obs).sum()) / (1 + n_perm)),
            "product_scores": pd.Series(u[:, 0] * s[0], index=liking.index)}
