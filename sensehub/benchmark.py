"""A small benchmark of plain-English questions, to measure the search assistant honestly.

Each question comes with what it should be understood as (category, attributes and their
direction, filters). The relevant products are then defined from that specification alone:
the products passing the filters whose score on the requested attributes is in the top
quarter (bottom quarter for "low"). Questions about things the data does not measure must be
answered with no product.

The score of a ranking is the precision of its top k, with k = min(5, number of relevant
products), so a category with six products is not penalised for having few relevant ones.

Read the numbers with this in mind: the assistant ranks with the same z-scores that define
relevance, so once a question is understood its precision is 1. Its precision therefore
measures understanding. The two keyword baselines show what a search box that matches words
(with or without the same synonym expansion) returns for the same questions.
"""
from __future__ import annotations

import pandas as pd

from .search import KeywordBaseline, KnowledgeSearch, parse

Q = list[dict]

QUERIES: Q = [
    {"q": "coffees with high acidity", "cat": "coffee", "attrs": [("acidity", 1)]},
    {"q": "bright coffee", "cat": "coffee", "attrs": [("acidity", 1)]},
    {"q": "full-bodied coffee", "cat": "coffee", "attrs": [("body", 1)]},
    {"q": "a sweet, balanced coffee", "cat": "coffee", "attrs": [("sweetness", 1), ("balance", 1)]},
    {"q": "most aromatic coffee", "cat": "coffee", "attrs": [("aroma", 1)]},
    {"q": "coffee with a long lingering finish", "cat": "coffee", "attrs": [("aftertaste", 1)]},
    {"q": "bright, full-bodied Kenyan coffee", "cat": "coffee", "attrs": [("acidity", 1), ("body", 1)],
     "countries": ["Kenya"]},
    {"q": "washed Ethiopian coffees above 2,000 m", "cat": "coffee", "countries": ["Ethiopia"],
     "process": "Washed", "alt_min": 2000},
    {"q": "best Colombian coffee", "cat": "coffee", "countries": ["Colombia"], "rank": 1},
    {"q": "natural coffees from Brazil with low acidity", "cat": "coffee", "attrs": [("acidity", -1)],
     "countries": ["Brazil"], "process": "Natural"},
    {"q": "high-grown Guatemalan coffee with good balance", "cat": "coffee", "attrs": [("balance", 1)],
     "countries": ["Guatemala"], "alt_min": 1600},
    {"q": "flavourful coffee grown below 1,000 m", "cat": "coffee", "attrs": [("flavor", 1)], "alt_max": 1000},
    {"q": "most aromatic robusta", "cat": "coffee", "attrs": [("aroma", 1)], "species": "Robusta"},
    {"q": "Mexican coffee that is not acidic", "cat": "coffee", "attrs": [("acidity", -1)], "countries": ["Mexico"]},
    {"q": "creamy chocolate that is not bitter", "cat": "chocolate",
     "attrs": [("MilkF", 1), ("Bitterness", -1)]},
    {"q": "the crunchiest chocolate", "cat": "chocolate", "attrs": [("Crunchy", 1)]},
    {"q": "dark chocolate with lots of cocoa", "cat": "chocolate", "attrs": [("CocoaF", 1)]},
    {"q": "chocolate with caramel and vanilla notes", "cat": "chocolate",
     "attrs": [("Caramel", 1), ("Vanilla", 1)]},
    {"q": "a chocolate that melts in the mouth", "cat": "chocolate", "attrs": [("Melting", 1)]},
    {"q": "least astringent chocolate", "cat": "chocolate", "attrs": [("Astringency", -1)]},
    {"q": "which chocolate do consumers like most", "cat": "chocolate", "rank": 1},
    {"q": "sweet cocktail without lemon", "cat": "cocktail", "attrs": [("sweet", 1), ("odor.lemon", -1)]},
    {"q": "which juice is least sour", "cat": "cocktail", "attrs": [("acidity", -1)]},
    {"q": "a thick, pulpy smoothie", "cat": "cocktail", "attrs": [("thickness", 1), ("pulp", 1)]},
    {"q": "cocktail that smells of banana", "cat": "cocktail", "attrs": [("odor.banana", 1)]},
    {"q": "most popular cocktail", "cat": "cocktail", "rank": 1},
    {"q": "a drink that is not too bitter", "cat": "cocktail", "attrs": [("bitterness", -1)]},
    {"q": "fruity coffee", "cat": "coffee", "abstain": True},
    {"q": "floral chocolate", "cat": "chocolate", "abstain": True},
    {"q": "coffee with low caffeine", "cat": "coffee", "abstain": True},
    {"q": "organic cocktails", "cat": "cocktail", "abstain": True},
    {"q": "smooth coffee", "cat": "coffee", "abstain": True},
]


def _filtered(search: KnowledgeSearch, spec: dict) -> pd.Index:
    p = search.products[search.products.category == spec["cat"]]
    if spec.get("countries"):
        p = p[p.country.isin(spec["countries"])]
    if spec.get("process"):
        p = p[p.process == spec["process"]]
    if spec.get("species"):
        p = p[p.species == spec["species"]]
    if spec.get("alt_min") is not None:
        p = p[p.altitude_m >= spec["alt_min"]]
    if spec.get("alt_max") is not None:
        p = p[p.altitude_m <= spec["alt_max"]]
    return p.index


def relevant(search: KnowledgeSearch, spec: dict) -> set[str]:
    ids = _filtered(search, spec)
    if spec.get("abstain") or not len(ids):
        return set()
    cat = spec["cat"]
    if spec.get("attrs"):
        z = search.z[cat].loc[ids]
        score = sum(d * z[a].fillna(0) for a, d in spec["attrs"]) / len(spec["attrs"])
    else:
        outcome = "total_score" if cat == "coffee" else "mean_liking"
        score = search.products.loc[ids, outcome] * (1 if spec.get("rank", 1) >= 0 else -1)
    return set(score[score >= score.quantile(0.75)].index)


def understood(search: KnowledgeSearch, spec: dict) -> bool:
    q = parse(spec["q"], search.countries)
    if spec.get("abstain"):
        ans = search.ask(spec["q"])
        return ans.products.empty and ("does not measure" in ans.message or "could not match" in ans.message)
    got_attrs = {(a, d) for c, a, d in q.attributes if c == spec["cat"]}
    ok = (q.categories == [spec["cat"]] and got_attrs == set(spec.get("attrs", []))
          and sorted(q.countries) == sorted(spec.get("countries", []))
          and q.processes == ([spec["process"]] if spec.get("process") else []) and not q.exclude_countries
          and q.species == spec.get("species")
          and q.altitude_min == spec.get("alt_min") and q.altitude_max == spec.get("alt_max"))
    if not spec.get("attrs") and spec.get("rank"):
        ok = ok and q.rank == spec["rank"]
    return ok


def precision(ranked: list[str], rel: set[str], k: int = 5) -> float:
    k = min(k, len(rel))
    return len(set(ranked[:k]) & rel) / k if k else float("nan")


def expanded(search: KnowledgeSearch, text: str) -> str:
    """The question plus the words the lexicon maps it to, in the vocabulary of the product cards."""
    q = parse(text, search.countries)
    extra = [f"{'high' if d > 0 else 'low'} {a}" for _, a, d in q.attributes]
    extra += q.countries + q.processes + ([q.species] if q.species else []) + q.categories
    return text + " " + " ".join(extra)


def evaluate(search: KnowledgeSearch, baseline: KeywordBaseline, queries: Q = QUERIES) -> pd.DataFrame:
    rows = []
    for spec in queries:
        ans = search.ask(spec["q"])
        ours = ans.products.product_id.tolist() if not ans.products.empty else []
        base = baseline.search(spec["q"])
        base_x = baseline.search(expanded(search, spec["q"]))
        rel = relevant(search, spec)
        row = {"query": spec["q"], "category": spec["cat"], "abstain": bool(spec.get("abstain")),
               "understood": understood(search, spec), "n_relevant": len(rel)}
        if spec.get("abstain"):
            row["assistant"] = float(row["understood"])
            row["keyword_search"] = float(not base)
            row["keyword_expanded"] = float(not base_x)
        else:
            row["assistant"] = precision(ours, rel)
            row["keyword_search"] = precision(base, rel)
            row["keyword_expanded"] = precision(base_x, rel)
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> dict:
    ranked = res[~res.abstain]
    out = {
        "n_queries": int(len(res)),
        "n_ranking": int(len(ranked)),
        "n_abstain": int(res.abstain.sum()),
        "understood": float(res.understood.mean()),
        "assistant_precision": float(ranked.assistant.mean()),
        "keyword_precision": float(ranked.keyword_search.mean()),
        "keyword_expanded_precision": float(ranked.keyword_expanded.mean()),
        "assistant_abstains_correctly": float(res[res.abstain].assistant.mean()),
    }
    return out
