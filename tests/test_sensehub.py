import sqlite3

import numpy as np
import pandas as pd
import pytest

from sensehub import coffee as cf
from sensehub.benchmark import evaluate, summarise
from sensehub.data import COFFEE_ATTRIBUTES, DATA, clean_coffee, load_knowledge_base
from sensehub.mapping import (drivers_of_liking, preference_map, preference_structure_test,
                              segment_consumers, sensory_map)
from sensehub.panel import descriptor_anova, panelist_performance, panelist_summary
from sensehub.recipe import (best_recipe, candidate_grid, fit_mixture, loo_predict, proportions,
                             replicate_sd)
from sensehub.search import KeywordBaseline, KnowledgeSearch, parse, product_cards


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


@pytest.fixture(scope="module")
def search(kb):
    wide = {c: kb.wide(c) for c in ["coffee", "chocolate", "cocktail"]}
    findings = pd.DataFrame({"title": ["Altitude and acidity"],
                             "text": ["Higher-grown coffee has more acidity within a country."]})
    return KnowledgeSearch(kb.products, wide, findings)


# ---------------------------------------------------------------- data
def test_coffee_cleaning_matches_audit():
    raw = pd.read_csv(DATA / "coffee_ratings.csv")
    clean, audit = clean_coffee(raw)
    assert audit["coffee_rows_all_zero"] == 1 and len(clean) == len(raw) - 1
    assert audit["coffee_altitude_implausible"] == 35
    assert clean.altitude_m.dropna().between(100, 3000).all()
    assert (clean.total_cup_points - clean[COFFEE_ATTRIBUTES].sum(axis=1)).abs().max() <= 0.5
    assert set(clean.process) <= {"Washed", "Natural", "Semi-washed", "Honey", "Other", "Unknown"}


def test_knowledge_base_is_consistent(kb):
    assert kb.products.product_id.is_unique
    assert set(kb.sensory.product_id) <= set(kb.products.product_id)
    assert set(kb.sensory.attribute_id) <= set(kb.attributes.attribute_id)
    assert set(kb.liking.product_id) <= set(kb.products.product_id)
    assert kb.liking_matrix("chocolate").shape == (6, 222)
    assert kb.liking_matrix("cocktail").shape == (16, 100)
    assert kb.wide("chocolate").shape == (6, 14)
    assert kb.audit["cocktail_replicated_recipes"] == [["cocktail_08", "cocktail_09", "cocktail_10"]]


def test_exports(kb, tmp_path):
    kb.to_sqlite(tmp_path / "kb.db")
    with sqlite3.connect(tmp_path / "kb.db") as con:
        n = con.execute("select count(*) from products").fetchone()[0]
        joined = con.execute("select count(*) from sensory s join attributes a using (attribute_id)").fetchone()[0]
    assert n == len(kb.products) and joined == len(kb.sensory)
    kb.to_powerbi(tmp_path / "pbi")
    assert {p.name for p in (tmp_path / "pbi").iterdir()} == {
        "dim_product.csv", "dim_attribute.csv", "fact_sensory.csv", "fact_liking.csv", "fact_recipe.csv"}


# ---------------------------------------------------------------- panel
def _synthetic_panel(effect, n_pan=12, n_prod=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_pan):
        bias = rng.normal(0, 1)
        for s in range(2):
            for p in range(n_prod):
                rows.append({"Panelist": j, "Product": p, "Session": s,
                             "a": 5 + effect * p + bias + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_descriptor_anova_detects_product_effect_only_when_present():
    assert descriptor_anova(_synthetic_panel(1.0), ["a"]).p_product.iloc[0] < 1e-6
    assert descriptor_anova(_synthetic_panel(0.0, seed=3), ["a"]).p_product.iloc[0] > 0.01
    with pytest.raises(ValueError):
        descriptor_anova(_synthetic_panel(1.0).iloc[1:], ["a"])


def test_random_panelist_has_lowest_agreement():
    df = _synthetic_panel(1.0)
    rng = np.random.default_rng(1)
    df.loc[df.Panelist == 0, "a"] = rng.uniform(0, 10, (df.Panelist == 0).sum())
    summ = panelist_summary(panelist_performance(df, ["a"]))
    assert summ.index[0] == 0


# ---------------------------------------------------------------- mapping
def test_sensory_map_properties(kb):
    m = sensory_map(kb.wide("chocolate"))
    assert 0 < m.explained.sum() <= 1
    assert m.loadings.abs().max().max() <= 1 + 1e-9
    assert abs(np.corrcoef(m.scores.PC1, m.scores.PC2)[0, 1]) < 1e-8


def _planted_liking(n=120, seed=0, structure=True):
    rng = np.random.default_rng(seed)
    prods = [f"p{i}" for i in range(8)]
    base = np.linspace(-2, 2, 8)
    cols = {}
    for c in range(n):
        sign = 1 if c < n // 2 else -1
        signal = sign * base if structure else 0
        cols[f"c{c}"] = 5 + signal + rng.normal(0, 0.7, 8)
    return pd.DataFrame(cols, index=prods), base


def test_segments_recover_planted_groups():
    lk, _ = _planted_liking()
    seg, _ = segment_consumers(lk)
    truth = np.array([1] * 60 + [2] * 60)
    agree = (seg.values == truth).mean()
    assert max(agree, 1 - agree) > 0.95


def test_preference_structure_test():
    lk, _ = _planted_liking()
    assert preference_structure_test(lk, n_perm=200)["p"] < 0.01
    lk0, _ = _planted_liking(structure=False, seed=4)
    assert preference_structure_test(lk0, n_perm=200)["p"] > 0.01


def test_preference_map_and_drivers():
    lk, base = _planted_liking()
    lk = lk.iloc[:, :60]                       # one segment: liking rises with `base`
    prof = pd.DataFrame({"x": base, "noise": np.random.default_rng(0).normal(size=8)}, index=lk.index)
    scores = pd.DataFrame({"PC1": base, "PC2": np.sin(base)}, index=lk.index)
    X, Y, share, r2 = preference_map(scores, lk, "vector")
    assert share.min() >= 0 and share.max() <= 1 and r2.median() > 0.5
    dr = drivers_of_liking(prof, lk, n_boot=200)
    assert dr.loc["x", "r"] > 0.95 and dr.loc["x", "ci_low"] > 0.9


# ---------------------------------------------------------------- recipe
def _mixture(kind="linear", seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.dirichlet([2, 2, 2], 20), columns=["a", "b", "c"])
    y = 3 * x.a + 6 * x.b + 1 * x.c
    if kind == "quadratic":
        y = y + 12 * x.a * x.b
    return x, pd.Series(y)


def test_mixture_recovers_coefficients_and_picks_model():
    x, y = _mixture()
    m = fit_mixture(x, y)
    assert m.kind == "linear" and np.allclose(m.coef, [3, 6, 1], atol=1e-3)
    xq, yq = _mixture("quadratic")
    assert fit_mixture(xq, yq).kind == "quadratic"


def test_leave_one_out_keeps_replicates_together():
    x, y = _mixture()
    x = pd.concat([x, x.iloc[[0, 0]]], ignore_index=True)
    y = pd.concat([y, pd.Series([y.iloc[0] + 5, y.iloc[0] - 5])], ignore_index=True)
    pred = loo_predict(x, y, "linear")
    assert pred.iloc[0] == pytest.approx(pred.iloc[20]) == pytest.approx(pred.iloc[21])
    assert replicate_sd(x, y) > 4


def test_candidate_grid_and_best_recipe():
    x, y = _mixture()
    grid = candidate_grid(x, step=0.02)
    assert np.allclose(grid.sum(axis=1), 1)
    assert (grid.min() >= x.min() - 1e-9).all() and (grid.max() <= x.max() + 1e-9).all()
    r, v = best_recipe(fit_mixture(x, y), x, step=0.02)
    assert r.b == pytest.approx(grid.b.max(), abs=0.03)


def test_cocktail_model_beats_spread(kb):
    rec = kb.recipes.pivot(index="product_id", columns="ingredient", values="dl")
    x = proportions(rec[["orange", "banana", "mango", "lemon"]])
    y = kb.liking_matrix("cocktail").mean(axis=1)
    m = fit_mixture(x, y)
    assert m.loo_rmse < 0.5 * y.std()


# ---------------------------------------------------------------- coffee
def test_holm_and_ols():
    p = pd.Series([0.01, 0.04, 0.03], index=list("abc"))
    assert cf.holm(p).round(4).tolist() == [0.03, 0.06, 0.06]
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = 1 + 2 * X[:, 1] + rng.normal(size=200)
    beta, se = cf.ols_hc1(y, X)
    assert np.allclose(beta, np.linalg.lstsq(X, y, rcond=None)[0]) and (se > 0).all()
    # with a shared shock per cluster, clustered errors must be wider than HC1
    groups = np.repeat(np.arange(20), 10)
    x = np.repeat(rng.normal(size=20), 10)
    y = 0.5 * x + np.repeat(rng.normal(size=20), 10) + rng.normal(0, 0.3, 200)
    X = np.column_stack([np.ones(200), x])
    assert cf.ols_cluster(y, X, groups)[1][1] > 1.5 * cf.ols_hc1(y, X)[1][1]


def test_process_effect_removes_country_confounding():
    rng = np.random.default_rng(0)
    rows = []
    for country, level, share_nat in [("A", 86, 0.8), ("B", 80, 0.2)]:
        for _ in range(300):
            proc = "Natural" if rng.random() < share_nat else "Washed"
            rows.append({"country": country, "process": proc,
                         **{a: level / 10 + rng.normal(0, 0.2) for a in cf.GRADED},
                         "total_cup_points": level + rng.normal(0, 1)})
    out = cf.process_effect(pd.DataFrame(rows)).loc["total_cup_points"]
    assert out.naive_diff > 3 and abs(out.within_country_diff) < 0.3


# ---------------------------------------------------------------- search
def test_parser(search):
    q = parse("bright, full-bodied Kenyan coffee above 1,500 m", search.countries)
    assert q.categories == ["coffee"] and q.countries == ["Kenya"] and q.altitude_min == 1500
    assert {(a, d) for _, a, d in q.attributes} == {("acidity", 1), ("body", 1)}
    q = parse("creamy chocolate that is not bitter", search.countries)
    assert {(a, d) for _, a, d in q.attributes} == {("MilkF", 1), ("Bitterness", -1)}
    assert parse("dark chocolate with lots of cocoa", search.countries).categories == ["chocolate"]
    assert parse("fruity coffee", search.countries).not_measured == ["fruity"]


def test_parser_clauses_and_filters(search):
    c = search.countries
    attrs = lambda t: {(a, d) for _, a, d in parse(t, c).attributes}  # noqa: E731
    assert attrs("not bitter, sweet chocolate") == {("Bitterness", -1), ("Sweetness", 1)}
    assert attrs("cocktail without lemon but sweet") == {("odor.lemon", -1), ("sweet", 1)}
    assert attrs("no-lemon cocktail") == {("odor.lemon", -1)}
    assert attrs("coffee with a long finish") == {("aftertaste", 1)}
    q = parse("washed or natural coffee not from Brazil above 6000 ft", c)
    assert q.processes == ["Washed", "Natural"] and q.exclude_countries == ["Brazil"] and not q.countries
    assert q.altitude_min == pytest.approx(1828.8)
    q = parse("coffee from 1,200 to 1,800 m", c)
    assert (q.altitude_min, q.altitude_max) == (1200, 1800)


def test_answers_respect_filters_and_abstain(search, kb):
    ans = search.ask("washed Ethiopian coffees above 2,000 m")
    got = kb.products.set_index("product_id").loc[ans.products.product_id]
    assert len(got) and (got.country == "Ethiopia").all() and (got.process == "Washed").all()
    assert (got.altitude_m >= 2000).all()
    assert search.ask("fruity coffee").products.empty
    assert "does not measure" in search.ask("fruity coffee").message
    assert not search.ask("does altitude affect acidity?").findings.empty
    smooth = search.ask("smooth coffee")
    assert smooth.products.empty and "could not match" in smooth.message
    brazil = search.ask("coffee not from Brazil")
    assert not brazil.products.empty and "Brazil" not in set(
        kb.products.set_index("product_id").loc[brazil.products.product_id, "country"])


def test_benchmark_regression(search, kb):
    wide = {c: kb.wide(c) for c in ["coffee", "chocolate", "cocktail"]}
    cards = product_cards(kb.products.set_index("product_id"), wide)
    s = summarise(evaluate(search, KeywordBaseline(cards)))
    assert s["understood"] >= 0.9
    assert s["assistant_precision"] > s["keyword_expanded_precision"] > s["keyword_precision"]
    assert s["assistant_abstains_correctly"] == 1.0
