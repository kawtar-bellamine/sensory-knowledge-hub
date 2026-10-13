"""Ask the knowledge base a question in plain English.

The assistant works in two steps, and every answer shows what it understood:

1. understand  the question is mapped to a product category, sensory attributes with a
               direction (high or low), filters (origin, process, altitude) and an intent
               (find products, or explain a finding), using a sensory lexicon;
2. retrieve    matching products are ranked on the requested attributes (z-scores within
               the category), and findings are retrieved by TF-IDF on the question expanded
               with the lexicon.

When a question asks for something the data does not contain (for example "fruity" coffee:
the cupping data has no fruit note), the assistant says so instead of guessing.
It runs locally, with no external service, so every answer can be traced to a table row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATEGORY_WORDS = {
    "coffee": ["coffee", "coffees", "arabica", "robusta", "bean", "beans", "cup", "cupping"],
    "chocolate": ["chocolate", "chocolates", "bar", "bars"],
    "cocktail": ["cocktail", "cocktails", "juice", "juices", "drink", "drinks", "smoothie", "smoothies"],
}

# phrase -> {category: attribute}
LEXICON: dict[str, dict[str, str]] = {}


def _add(category: str, attribute: str, *phrases: str) -> None:
    for p in phrases:
        LEXICON.setdefault(p, {})[category] = attribute


_add("coffee", "aroma", "aroma", "aromatic", "fragrant", "fragrance", "smell", "smells")
_add("coffee", "flavor", "flavor", "flavour", "flavorful", "flavourful", "tasty")
_add("coffee", "aftertaste", "aftertaste", "finish", "lingering")
_add("coffee", "acidity", "acidity", "acidic", "bright", "tangy", "sour", "tart")
_add("coffee", "body", "body", "bodied", "full-bodied", "full bodied", "mouthfeel", "heavy")
_add("coffee", "balance", "balance", "balanced", "harmonious")
_add("coffee", "sweetness", "sweet", "sweetness")
_add("coffee", "clean_cup", "clean", "clean cup")
_add("coffee", "uniformity", "uniform", "uniformity", "consistent")
_add("chocolate", "CocoaF", "cocoa", "chocolatey", "dark")
_add("chocolate", "MilkF", "milk", "milky", "creamy")
_add("chocolate", "Caramel", "caramel")
_add("chocolate", "Vanilla", "vanilla")
_add("chocolate", "Sweetness", "sweet", "sweetness")
_add("chocolate", "Acidity", "acidity", "acidic", "sour", "tangy")
_add("chocolate", "Bitterness", "bitter", "bitterness")
_add("chocolate", "Astringency", "astringent", "astringency", "drying")
_add("chocolate", "Crunchy", "crunchy", "crunch", "crispy", "snap")
_add("chocolate", "Melting", "melting", "melts", "melt")
_add("chocolate", "Sticky", "sticky")
_add("chocolate", "Granular", "granular", "grainy", "gritty")
_add("cocktail", "sweet", "sweet", "sweetness")
_add("cocktail", "acidity", "acidity", "acidic", "sour", "tangy", "tart")
_add("cocktail", "bitterness", "bitter", "bitterness")
_add("cocktail", "strongness", "strong", "strength", "intense")
_add("cocktail", "thickness", "thick", "thickness", "heavy", "full-bodied", "full bodied")
_add("cocktail", "pulp", "pulp", "pulpy")
_add("cocktail", "persistence", "lingering", "persistent", "persistence", "long finish")
_add("cocktail", "odor.banana", "banana")
_add("cocktail", "odor.mango", "mango")
_add("cocktail", "odor.orange", "orange")
_add("cocktail", "odor.lemon", "lemon")

# Notes people ask for that none of these data sets measure.
NOT_MEASURED = ["fruity", "floral", "nutty", "spicy", "earthy", "smoky", "herbal", "winey",
                "berry", "citrus", "caffeine", "price", "organic", "fair trade", "salty", "umami"]

LOW_WORDS = {"not", "no", "low", "less", "least", "without", "little", "mild", "non", "lower", "lowest", "minimal"}
HIGH_WORDS = {"high", "very", "more", "most", "strong", "strongly", "highly", "higher", "highest", "extra"}
BEST_WORDS = {"best", "top", "favourite", "favorite", "preferred", "highest-rated", "liked", "popular", "winning"}
WORST_WORDS = {"worst", "disliked", "least-liked", "lowest-rated"}
EXPLAIN_WORDS = {"why", "drive", "drives", "driver", "drivers", "effect", "effects", "impact",
                 "influence", "influences", "affect", "affects", "matter", "matters", "relationship",
                 "segment", "segments", "reliable", "reliability", "explain", "difference", "trust",
                 "agree", "consistent", "predict", "improve", "optimal", "optimise", "optimize"}

PROCESSES = {"washed": "Washed", "wet": "Washed", "natural": "Natural", "dry-processed": "Natural",
             "honey": "Honey", "semi-washed": "Semi-washed", "pulped": "Honey"}
DEMONYMS = {"ethiopian": "Ethiopia", "kenyan": "Kenya", "colombian": "Colombia", "brazilian": "Brazil",
            "guatemalan": "Guatemala", "mexican": "Mexico", "honduran": "Honduras", "costa rican": "Costa Rica",
            "tanzanian": "Tanzania", "ugandan": "Uganda", "rwandan": "Rwanda", "indonesian": "Indonesia",
            "panamanian": "Panama", "salvadoran": "El Salvador", "nicaraguan": "Nicaragua", "peruvian": "Peru",
            "taiwanese": "Taiwan", "chinese": "China", "thai": "Thailand", "indian": "India",
            "vietnamese": "Vietnam", "burundian": "Burundi", "malawian": "Malawi", "hawaiian": "United States",
            "hawaii": "United States", "usa": "United States"}


@dataclass
class Query:
    text: str
    categories: list[str] = field(default_factory=list)
    attributes: list[tuple[str, str, int]] = field(default_factory=list)  # (category, attribute, +1/-1)
    countries: list[str] = field(default_factory=list)
    exclude_countries: list[str] = field(default_factory=list)
    processes: list[str] = field(default_factory=list)
    species: str | None = None
    altitude_min: float | None = None
    altitude_max: float | None = None
    rank: int = 0                       # +1 best first, -1 worst first, 0 none
    explain: bool = False
    not_measured: list[str] = field(default_factory=list)

    @property
    def has_filters(self) -> bool:
        return bool(self.countries or self.exclude_countries or self.processes or self.species
                    or self.altitude_min is not None or self.altitude_max is not None)

    @property
    def recognised(self) -> bool:
        return bool(self.attributes or self.has_filters or self.rank or self.explain or self.not_measured)

    def describe(self) -> str:
        parts = []
        if self.categories:
            parts.append("category: " + ", ".join(self.categories))
        for c, a, d in self.attributes:
            parts.append(f"{'high' if d > 0 else 'low'} {a} ({c})")
        if self.countries:
            parts.append("origin: " + ", ".join(self.countries))
        if self.exclude_countries:
            parts.append("origin not: " + ", ".join(self.exclude_countries))
        if self.processes:
            parts.append("process: " + " or ".join(self.processes))
        if self.species:
            parts.append(f"species: {self.species}")
        if self.altitude_min is not None:
            parts.append(f"altitude >= {self.altitude_min:,.0f} m")
        if self.altitude_max is not None:
            parts.append(f"altitude <= {self.altitude_max:,.0f} m")
        if self.rank:
            parts.append("best rated first" if self.rank > 0 else "lowest rated first")
        if self.explain:
            parts.append("looking for findings")
        return "; ".join(parts) if parts else "nothing recognised"


TOKEN = r"[a-z0-9][a-z0-9\-']*|[,.;!?]"
BOUNDARY = {",", ".", ";", "!", "?", "but", "and", "with", "yet", "or", "while"}
UNIT = r"(m|metres|meters|metre|meter|ft|feet|foot)"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-']*", text.lower())


def _number(s: str, unit: str = "m") -> float:
    v = float(s.replace(",", "").replace("'", ""))
    return v * 0.3048 if unit.startswith(("ft", "fe", "fo")) else v


def _direction(before: list[str]) -> int:
    """Direction of an attribute from the (at most three) tokens before it, within its clause."""
    window = before[-3:]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in BOUNDARY:
            window = window[i + 1:]
            break
    return -1 if any(w in LOW_WORDS for w in window) else 1


def parse(text: str, countries: list[str]) -> Query:
    """Turn a question into a structured query."""
    q = Query(text)
    low = " " + text.lower().replace("’", "'") + " "
    low = re.sub(r"\b(no|non|low|less|not)-", r"\1 ", low)
    toks = _tokens(low)

    for cat, words in CATEGORY_WORDS.items():
        if any(w in toks for w in words):
            q.categories.append(cat)

    # Longest phrases first; a phrase is skipped if its span is inside one already matched.
    # When the category is known, only phrases that mean something in it are considered.
    spans: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []
    for phrase in sorted(LEXICON, key=len, reverse=True):
        if q.categories and not set(LEXICON[phrase]) & set(q.categories):
            continue
        for m in re.finditer(rf"(?<![a-z\-]){re.escape(phrase)}(?![a-z\-])", low):
            if any(a <= m.start() and m.end() <= b for a, b in spans):
                continue
            spans.append((m.start(), m.end()))
            found.append((m.start(), phrase))
            break
    for start, phrase in sorted(found):
        direction = _direction(re.findall(TOKEN, low[:start]))
        for cat, attr in LEXICON[phrase].items():
            if not q.categories or cat in q.categories:
                if (cat, attr, direction) not in q.attributes:
                    q.attributes.append((cat, attr, direction))

    q.not_measured = [w for w in NOT_MEASURED if re.search(rf"\b{re.escape(w)}\b", low)]

    names = {c.lower(): c for c in countries}
    names.update({d: c for d, c in DEMONYMS.items() if c in countries})
    for word, c in sorted(names.items(), key=lambda kv: -len(kv[0])):
        m = re.search(rf"\b{re.escape(word)}\b", low)
        if not m or c in q.countries or c in q.exclude_countries:
            continue
        negated = re.search(r"(not|except|outside|excluding|other than|apart from)\s+(from\s+|in\s+)?$",
                            low[: m.start()])
        (q.exclude_countries if negated else q.countries).append(c)
    for word, proc in PROCESSES.items():
        if re.search(rf"\b{re.escape(word)}\b", low) and proc not in q.processes:
            q.processes.append(proc)
    if "robusta" in toks:
        q.species = "Robusta"
    elif "arabica" in toks:
        q.species = "Arabica"

    num = r"([\d][\d,']*)"
    m = re.search(rf"\b(between|from)\s*{num}\s*{UNIT}?\s*(and|to|-)\s*{num}\s*{UNIT}\b", low)
    if m:
        unit = m.group(6)
        q.altitude_min, q.altitude_max = _number(m.group(2), m.group(3) or unit), _number(m.group(5), unit)
    else:
        m = re.search(rf"(above|over|higher than|more than|at least|>)\s*{num}\s*{UNIT}\b", low)
        if m:
            q.altitude_min = _number(m.group(2), m.group(3))
        m = re.search(rf"(below|under|lower than|less than|at most|<)\s*{num}\s*{UNIT}\b", low)
        if m:
            q.altitude_max = _number(m.group(2), m.group(3))
    if re.search(r"\bhigh[- ](altitude|grown|elevation)\b", low) and q.altitude_min is None:
        q.altitude_min = 1600.0
    if re.search(r"\blow[- ](altitude|grown|elevation)\b", low) and q.altitude_max is None:
        q.altitude_max = 1000.0
    coffee_hint = bool(re.search(r"\b(altitude|elevation|origin|origins|country|countries|process|processing)\b", low))
    if (q.has_filters or coffee_hint) and not q.categories:
        q.categories = ["coffee"]
        q.attributes = [t for t in q.attributes if t[0] == "coffee"]

    if any(w in BEST_WORDS for w in toks):
        q.rank = 1
    if any(w in WORST_WORDS for w in toks) or re.search(r"\bleast (liked|popular)\b", low):
        q.rank = -1
    q.explain = bool(set(toks) & EXPLAIN_WORDS)
    if not q.categories:
        q.categories = sorted({c for c, _, _ in q.attributes})
    return q


@dataclass
class Answer:
    query: Query
    message: str
    products: pd.DataFrame
    findings: pd.DataFrame


class KnowledgeSearch:
    """Search over products (structured) and findings (text)."""

    def __init__(self, products: pd.DataFrame, wide: dict[str, pd.DataFrame],
                 findings: pd.DataFrame | None = None):
        self.products = products.set_index("product_id")
        self.wide = wide
        self.countries = sorted(c for c in products.country.dropna().unique() if c != "Unknown")
        self.z = {c: (w - w.mean()) / w.std(ddof=0).replace(0, np.nan) for c, w in wide.items()}
        self.pct = {c: w.rank(pct=True) for c, w in wide.items()}
        self.findings = findings if findings is not None else pd.DataFrame(columns=["title", "text"])
        if len(self.findings):
            self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self.f_matrix = self.vec.fit_transform(self.findings.title + ". " + self.findings.text)

    # ------------------------------------------------------------------ products
    def _filter(self, q: Query, cat: str) -> pd.Index:
        p = self.products[self.products.category == cat]
        if cat == "coffee":
            if q.countries:
                p = p[p.country.isin(q.countries)]
            if q.exclude_countries:
                p = p[~p.country.isin(q.exclude_countries)]
            if q.processes:
                p = p[p.process.isin(q.processes)]
            if q.species:
                p = p[p.species == q.species]
            if q.altitude_min is not None:
                p = p[p.altitude_m >= q.altitude_min]
            if q.altitude_max is not None:
                p = p[p.altitude_m <= q.altitude_max]
        return p.index

    def rank_products(self, q: Query, k: int = 5) -> tuple[pd.DataFrame, dict[str, int]]:
        frames, counts = [], {}
        for cat in q.categories:
            ids = self._filter(q, cat)
            counts[cat] = len(ids)
            if not len(ids):
                continue
            attrs = [(a, d) for c, a, d in q.attributes if c == cat]
            outcome = "total_score" if cat == "coffee" else "mean_liking"
            p = self.products.loc[ids]
            if attrs:
                z = self.z[cat].loc[ids]
                score = sum(d * z[a].fillna(0) for a, d in attrs) / len(attrs)
                tie = p[outcome].fillna(0) * 1e-6
                order = (score + tie).sort_values(ascending=False)
            elif q.rank or q.has_filters:
                order = p[outcome].sort_values(ascending=q.rank < 0)
            else:
                continue
            top = order.index[:k]
            out = pd.DataFrame({"product_id": top, "category": cat,
                                "name": self.products.loc[top, "name"].values,
                                outcome.replace("_", " "): self.products.loc[top, outcome].round(2).values})
            for a, _ in attrs:
                out[a] = self.wide[cat].loc[top, a].round(2).values
                out[f"{a} percentile"] = (100 * self.pct[cat].loc[top, a]).round(0).values
            frames.append(out)
        return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), counts

    # ------------------------------------------------------------------ findings
    def search_findings(self, q: Query, k: int = 3, min_score: float = 0.08) -> pd.DataFrame:
        if not len(self.findings):
            return self.findings
        expanded = q.text + " " + " ".join(a for _, a, _ in q.attributes) + " " + " ".join(q.categories)
        sim = cosine_similarity(self.vec.transform([expanded]), self.f_matrix).ravel()
        order = np.argsort(-sim)[:k]
        out = self.findings.iloc[order].assign(score=sim[order])
        return out[out.score >= min_score]

    # ------------------------------------------------------------------ answer
    def ask(self, text: str, k: int = 5) -> Answer:
        q = parse(text, self.countries)
        empty = pd.DataFrame()
        if not q.recognised:
            return Answer(q, "I could not match this question to the data. Try naming an attribute "
                             "(acidity, body, bitterness...), an origin, a process or a ranking (best, most liked).",
                          empty, empty)
        products, counts = empty, {}
        if not q.explain:
            products, counts = self.rank_products(q, k)
        findings = self.search_findings(q, k=3 if q.explain else 2)
        notes = []
        if q.not_measured:
            cats = q.categories or ["coffee", "chocolate", "cocktail"]
            avail = "; ".join(f"{c}: " + ", ".join(self.wide[c].columns) for c in cats)
            notes.append(f"The data does not measure {', '.join(q.not_measured)}. Available attributes are {avail}.")
            if not q.explain:
                findings = findings.iloc[0:0]
            if not (q.attributes or q.has_filters or q.rank):
                products = empty
        if not products.empty:
            n = ", ".join(f"{v:,} {c}" + ("s" if v != 1 else "") for c, v in counts.items())
            verb = "matches" if sum(counts.values()) == 1 else "match"
            notes.append(f"{n} {verb} the question; showing the best {k} of each category at most.")
        elif counts and not sum(counts.values()):
            notes.append("No product matches all the filters.")
        return Answer(q, " ".join(notes), products, findings)


def product_cards(products: pd.DataFrame, wide: dict[str, pd.DataFrame]) -> pd.Series:
    """One short text per product, used by the keyword baseline."""
    cards = {}
    for cat, w in wide.items():
        pct = w.rank(pct=True)
        for pid in w.index:
            row = products.loc[pid]
            words = [row["name"], cat]
            for col in ["country", "process", "species", "variety", "region"]:
                if col in row and isinstance(row[col], str):
                    words.append(row[col])
            for a in w.columns:
                if w[a].quantile(0.75) < w[a].max() and pct.loc[pid, a] >= 0.75:
                    words.append(f"high {a}")
                elif w[a].quantile(0.25) > w[a].min() and pct.loc[pid, a] <= 0.25:
                    words.append(f"low {a}")
            cards[pid] = ". ".join(str(x) for x in words)
    return pd.Series(cards)


class KeywordBaseline:
    """Plain TF-IDF search over product cards: what a generic search box would do."""

    def __init__(self, cards: pd.Series):
        self.ids = cards.index
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.m = self.vec.fit_transform(cards.values)

    def search(self, text: str, k: int = 5) -> list[str]:
        sim = cosine_similarity(self.vec.transform([text]), self.m).ravel()
        order = np.argsort(-sim)[:k]
        return [self.ids[i] for i in order if sim[i] > 0]
