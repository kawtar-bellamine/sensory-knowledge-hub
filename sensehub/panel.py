"""Sensory panel performance: is the panel able to tell the products apart, and do the
panelists agree with each other?

The design is balanced: every panelist tastes every product in every session. The product
effect is tested against the product x panelist interaction (panelists are a random sample
of possible assessors), the usual choice in sensory analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def descriptor_anova(panel: pd.DataFrame, attributes: list[str],
                     product: str = "Product", panelist: str = "Panelist") -> pd.DataFrame:
    """Two-way ANOVA with interaction for each descriptor.

    Returns, for each descriptor, the F statistic and p-value of the product effect (tested
    against the interaction) and of the product x panelist interaction (tested against the
    replicate error). A significant interaction means the panelists rank the products
    differently on that descriptor.
    """
    rows = []
    for a in attributes:
        y = panel[a].astype(float)
        grand = y.mean()
        n_prod = panel[product].nunique()
        n_pan = panel[panelist].nunique()
        cell = panel.groupby([panelist, product])[a]
        n_rep = int(cell.size().iloc[0])
        if (cell.size() != n_rep).any() or len(cell.size()) != n_prod * n_pan:
            raise ValueError("descriptor_anova needs a balanced design")
        m_p = panel.groupby(product)[a].mean()
        m_j = panel.groupby(panelist)[a].mean()
        m_pj = cell.mean()
        ss_p = n_pan * n_rep * ((m_p - grand) ** 2).sum()
        ss_j = n_prod * n_rep * ((m_j - grand) ** 2).sum()
        inter = m_pj - m_pj.index.get_level_values(0).map(m_j).values \
            - m_pj.index.get_level_values(1).map(m_p).values + grand
        ss_pj = n_rep * (inter ** 2).sum()
        fitted = panel.set_index([panelist, product]).index.map(m_pj)
        ss_e = ((y.values - np.asarray(fitted)) ** 2).sum()
        df_p, df_pj = n_prod - 1, (n_prod - 1) * (n_pan - 1)
        df_e = n_prod * n_pan * (n_rep - 1)
        ms_p, ms_pj, ms_e = ss_p / df_p, ss_pj / df_pj, ss_e / df_e
        f_p, f_pj = ms_p / ms_pj, ms_pj / ms_e
        rows.append({
            "descriptor": a,
            "F_product": f_p,
            "p_product": stats.f.sf(f_p, df_p, df_pj),
            "F_interaction": f_pj,
            "p_interaction": stats.f.sf(f_pj, df_pj, df_e),
            "panelist_share_of_variance": ss_j / (ss_p + ss_j + ss_pj + ss_e),
        })
    return pd.DataFrame(rows).set_index("descriptor")


def panelist_performance(panel: pd.DataFrame, attributes: list[str],
                         product: str = "Product", panelist: str = "Panelist") -> pd.DataFrame:
    """Per panelist and descriptor: discrimination, repeatability and agreement.

    discrimination  p-value of a one-way ANOVA of that panelist's scores on the products;
    repeatability   standard deviation between the two sessions' scores of the same product;
    agreement       correlation between the panelist's product means and the means of the
                    other panelists (leave-one-out, so a panelist is not compared with itself).
    """
    rows = []
    totals = panel.groupby(product)[attributes].sum()
    counts = panel.groupby(product)[attributes].count()
    for pid, g in panel.groupby(panelist):
        own_sum = g.groupby(product)[attributes].sum()
        own_cnt = g.groupby(product)[attributes].count()
        others = (totals - own_sum) / (counts - own_cnt)
        own = own_sum / own_cnt
        for a in attributes:
            groups = [v[a].values for _, v in g.groupby(product)]
            if np.ptp(g[a].values) == 0:
                p = 1.0
            else:
                p = stats.f_oneway(*groups).pvalue
                p = 1.0 if np.isnan(p) else p
            resid = g[a] - g.groupby(product)[a].transform("mean")
            dof = len(g) - g[product].nunique()
            sd = np.sqrt((resid ** 2).sum() / dof)
            x, y = own[a].values, others.loc[own.index, a].values
            r = np.corrcoef(x, y)[0, 1] if x.std() > 0 else np.nan
            rows.append({"panelist": pid, "descriptor": a, "p_discrimination": p,
                         "repeat_sd": sd, "agreement_r": r})
    return pd.DataFrame(rows)


def panelist_summary(perf: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """One row per panelist, sorted from weakest to strongest."""
    out = perf.groupby("panelist").agg(
        discriminates=("p_discrimination", lambda p: float((p < alpha).mean())),
        median_agreement=("agreement_r", "median"),
        median_repeat_sd=("repeat_sd", "median"),
    )
    return out.sort_values(["median_agreement", "discriminates"])
