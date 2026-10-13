"""Load, clean and connect the three public sensory data sources.

Sources
-------
coffee     Coffee Quality Institute cupping scores (1,339 lots, 10 attributes, origin and process).
chocolate  Trained panel profiles (29 panelists x 6 chocolates x 2 sessions, 14 descriptors)
           and the liking scores of 222 consumers.
cocktail   Recipe (4 juices), panel means (13 descriptors) and the liking of 100 consumers
           for 16 cocktails.

Everything is connected in one model (`KnowledgeBase`): one product table, one attribute
table, and fact tables for sensory scores, consumer liking and recipes.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

COFFEE_ATTRIBUTES = [
    "aroma", "flavor", "aftertaste", "acidity", "body", "balance",
    "uniformity", "clean_cup", "sweetness", "cupper_points",
]
CHOCOLATE_ATTRIBUTES = [
    "CocoaA", "MilkA", "CocoaF", "MilkF", "Caramel", "Vanilla", "Sweetness", "Acidity",
    "Bitterness", "Astringency", "Crunchy", "Melting", "Sticky", "Granular",
]
INGREDIENTS = ["orange", "banana", "mango", "lemon"]

# Altitudes outside this range are recording errors (lost decimal points, feet typed as
# metres, "1.2" meaning 1,200 m). They are set to missing rather than guessed.
ALTITUDE_RANGE_M = (100.0, 3000.0)

COUNTRY_FIXES = {
    "Tanzania, United Republic Of": "Tanzania",
    "Cote d?Ivoire": "Côte d'Ivoire",
    "United States (Hawaii)": "United States",
    "United States (Puerto Rico)": "Puerto Rico",
}
PROCESS_FIXES = {
    "Washed / Wet": "Washed",
    "Natural / Dry": "Natural",
    "Semi-washed / Semi-pulped": "Semi-washed",
    "Pulped natural / honey": "Honey",
    "Other": "Other",
}


@dataclass
class KnowledgeBase:
    """The connected data model (a star schema)."""

    products: pd.DataFrame      # one row per product, all categories
    attributes: pd.DataFrame    # one row per (category, attribute)
    sensory: pd.DataFrame       # product_id, attribute_id, value
    liking: pd.DataFrame        # product_id, consumer_id, liking
    recipes: pd.DataFrame       # product_id, ingredient, dl
    panel: pd.DataFrame         # raw chocolate panel data (one row per tasting)
    audit: dict = field(default_factory=dict)

    def wide(self, category: str) -> pd.DataFrame:
        """Product x attribute table for one category."""
        attrs = self.attributes[self.attributes.category == category]
        s = self.sensory.merge(attrs[["attribute_id", "attribute"]], on="attribute_id")
        out = s.pivot(index="product_id", columns="attribute", values="value")
        return out[attrs.attribute.tolist()]

    def liking_matrix(self, category: str) -> pd.DataFrame:
        """Product x consumer liking table for one category."""
        ids = self.products.product_id[self.products.category == category]
        lk = self.liking[self.liking.product_id.isin(ids)]
        return lk.pivot(index="product_id", columns="consumer_id", values="liking")

    def to_sqlite(self, path: str | Path) -> None:
        with sqlite3.connect(path) as con:
            for name in ["products", "attributes", "sensory", "liking", "recipes"]:
                getattr(self, name).to_sql(name, con, if_exists="replace", index=False)

    def to_powerbi(self, folder: str | Path) -> None:
        """Write the star schema as CSV files for Power BI (or any BI tool)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        tables = {
            "dim_product": self.products,
            "dim_attribute": self.attributes,
            "fact_sensory": self.sensory,
            "fact_liking": self.liking,
            "fact_recipe": self.recipes,
        }
        for name, df in tables.items():
            df.to_csv(folder / f"{name}.csv", index=False)


# --------------------------------------------------------------------------- coffee
def _parse_year(text) -> float:
    years = re.findall(r"20\d\d", str(text))
    return float(years[0]) if years else np.nan


def clean_coffee(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the CQI table and report every change made."""
    df = raw.copy()
    audit = {"coffee_rows_raw": len(df)}

    blank = (df[COFFEE_ATTRIBUTES] == 0).all(axis=1)
    audit["coffee_rows_all_zero"] = int(blank.sum())
    df = df[~blank]

    total_gap = (df.total_cup_points - df[COFFEE_ATTRIBUTES].sum(axis=1)).abs()
    audit["coffee_total_minus_sum_max"] = round(float(total_gap.max()), 2)

    alt = df.altitude_mean_meters
    bad_alt = alt.notna() & ((alt < ALTITUDE_RANGE_M[0]) | (alt > ALTITUDE_RANGE_M[1]))
    audit["coffee_altitude_missing"] = int(alt.isna().sum())
    audit["coffee_altitude_implausible"] = int(bad_alt.sum())
    audit["coffee_altitude_max_raw"] = float(alt.max())
    df["altitude_m"] = alt.where(~bad_alt).round(0)

    df["country"] = df.country_of_origin.replace(COUNTRY_FIXES).fillna("Unknown")
    audit["coffee_process_missing"] = int(df.processing_method.isna().sum())
    df["process"] = df.processing_method.map(PROCESS_FIXES).fillna("Unknown")
    df["harvest_year"] = df.harvest_year.map(_parse_year)
    audit["coffee_harvest_year_entries"] = int(raw.harvest_year.nunique())

    ceiling = {a: round(float((df[a] == 10).mean()), 3) for a in ["uniformity", "clean_cup", "sweetness"]}
    audit["coffee_share_at_10"] = ceiling
    audit["coffee_rows_clean"] = len(df)
    df = df.reset_index(drop=True)
    df["product_id"] = [f"coffee_{i + 1:04d}" for i in range(len(df))]
    return df, audit


def _coffee_name(row) -> str:
    alt = f", {int(row.altitude_m):,} m" if pd.notna(row.altitude_m) else ""
    proc = f", {row.process.lower()}" if row.process not in ("Unknown", "Other") else ""
    return f"{row.country} {row.species.lower()}{proc}{alt} ({row.product_id[-4:]})"


# --------------------------------------------------------------------------- loader
def load_knowledge_base(data_dir: str | Path = DATA) -> KnowledgeBase:
    data_dir = Path(data_dir)
    coffee, audit = clean_coffee(pd.read_csv(data_dir / "coffee_ratings.csv"))

    # products ---------------------------------------------------------------
    cof = pd.DataFrame({
        "product_id": coffee.product_id,
        "category": "coffee",
        "name": coffee.apply(_coffee_name, axis=1),
        "country": coffee.country,
        "region": coffee.region.fillna(""),
        "process": coffee.process,
        "species": coffee.species,
        "variety": coffee.variety.fillna("Unknown"),
        "altitude_m": coffee.altitude_m,
        "harvest_year": coffee.harvest_year,
        "total_score": coffee.total_cup_points,
    })

    panel = pd.read_csv(data_dir / "sensochoc.csv").drop(columns="row")
    panel["Panelist"] = panel.Panelist.astype(int)
    hedo_choc = pd.read_csv(data_dir / "hedochoc.csv", index_col="product")
    choc_ids = {p: f"chocolate_{p[-1]}" for p in hedo_choc.index}
    panel["product_id"] = panel.Product.map(choc_ids)
    choc = pd.DataFrame({"product_id": list(choc_ids.values()), "category": "chocolate",
                         "name": [f"Chocolate {p[-1]}" for p in choc_ids]})

    senso_ck = pd.read_csv(data_dir / "senso_cocktail.csv", index_col="product")
    hedo_ck = pd.read_csv(data_dir / "hedo_cocktail.csv", index_col="product")
    compo = pd.read_csv(data_dir / "compo_cocktail.csv", index_col="product")
    ck_ids = {p: f"cocktail_{int(p):02d}" for p in senso_ck.index}
    ck = pd.DataFrame({"product_id": list(ck_ids.values()), "category": "cocktail",
                       "name": [f"Cocktail {int(p)}" for p in senso_ck.index]})
    audit["cocktail_replicated_recipes"] = _replicated_recipes(compo.rename(index=ck_ids))
    audit["cocktail_recipe_total_dl"] = sorted(set(compo.sum(axis=1).round(4)))

    products = pd.concat([cof, choc, ck], ignore_index=True)

    # attributes -------------------------------------------------------------
    attributes = pd.concat([
        pd.DataFrame({"category": "coffee", "attribute": COFFEE_ATTRIBUTES, "scale": "0-10 cupping score"}),
        pd.DataFrame({"category": "chocolate", "attribute": CHOCOLATE_ATTRIBUTES, "scale": "0-10 panel intensity"}),
        pd.DataFrame({"category": "cocktail", "attribute": list(senso_ck.columns), "scale": "0-10 panel intensity"}),
    ], ignore_index=True)
    attributes.insert(0, "attribute_id", [f"{c}:{a}" for c, a in zip(attributes.category, attributes.attribute)])

    # sensory facts ----------------------------------------------------------
    s_cof = coffee.melt(id_vars="product_id", value_vars=COFFEE_ATTRIBUTES, var_name="attribute")
    s_cof["n_assessors"] = np.nan
    s_choc = (panel.groupby("product_id")[CHOCOLATE_ATTRIBUTES].mean()
              .reset_index().melt(id_vars="product_id", var_name="attribute"))
    s_choc["n_assessors"] = panel.Panelist.nunique()
    s_ck = senso_ck.rename(index=ck_ids).rename_axis("product_id").reset_index().melt(
        id_vars="product_id", var_name="attribute")
    s_ck["n_assessors"] = 12
    sensory = []
    for cat, s in [("coffee", s_cof), ("chocolate", s_choc), ("cocktail", s_ck)]:
        s = s.assign(attribute_id=cat + ":" + s.attribute)
        sensory.append(s[["product_id", "attribute_id", "value", "n_assessors"]])
    sensory = pd.concat(sensory, ignore_index=True)

    # liking facts -----------------------------------------------------------
    def _liking(hedo, ids, prefix):
        out = hedo.copy().rename(index=ids).rename_axis("product_id").reset_index().melt(
            id_vars="product_id", var_name="consumer_id", value_name="liking")
        out["consumer_id"] = prefix + out.consumer_id.astype(str)
        return out

    liking = pd.concat([_liking(hedo_choc, choc_ids, "choc_c"),
                        _liking(hedo_ck, ck_ids, "ck_c")], ignore_index=True)
    mean_like = liking.groupby("product_id").liking.mean()
    products["mean_liking"] = products.product_id.map(mean_like)

    recipes = (compo.rename(index=ck_ids).rename_axis("product_id").reset_index()
               .melt(id_vars="product_id", var_name="ingredient", value_name="dl"))

    audit["n_products"] = products.category.value_counts().to_dict()
    audit["n_consumers"] = {"chocolate": hedo_choc.shape[1], "cocktail": hedo_ck.shape[1]}
    return KnowledgeBase(products, attributes, sensory, liking, recipes, panel, audit)


def _replicated_recipes(compo: pd.DataFrame) -> list[list[str]]:
    groups = compo.round(4).groupby(list(compo.columns)).groups
    return [sorted(v) for v in groups.values() if len(v) > 1]
