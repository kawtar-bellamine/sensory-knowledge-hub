"""Sensory knowledge hub: interactive dashboard.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sensehub.data import INGREDIENTS, load_knowledge_base  # noqa: E402
from sensehub.mapping import preference_map, sensory_map  # noqa: E402
from sensehub.recipe import design_bounds, fit_mixture, proportions  # noqa: E402
from sensehub.search import KnowledgeSearch  # noqa: E402

RES = ROOT / "results"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SEG_COLORS = alt.Scale(domain=["all consumers", "segment 1", "segment 2"], range=[MUTED, BLUE, ORANGE])

st.set_page_config(page_title="Sensory knowledge hub", layout="wide")


@st.cache_data
def kb_tables():
    kb = load_knowledge_base()
    return kb


@st.cache_resource
def search_engine():
    kb = kb_tables()
    findings = pd.read_csv(RES / "findings.csv")
    wide = {c: kb.wide(c) for c in ["coffee", "chocolate", "cocktail"]}
    return KnowledgeSearch(kb.products, wide, findings)


@st.cache_data
def read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RES / name, **kw)


def read_json(name: str) -> dict:
    return json.loads((RES / name).read_text())


kb = kb_tables()
products = kb.products.set_index("product_id")
PAGES = ["Ask the data", "Product explorer", "Consumers and segments", "Recipe simulator",
         "Panel quality", "Findings and data audit"]
page = st.sidebar.radio("View", PAGES)
st.sidebar.caption(
    f"{len(products):,} products from three public studies: "
    f"{(products.category == 'coffee').sum():,} coffees, 6 chocolates, 16 cocktails; "
    f"{kb.liking.consumer_id.nunique()} consumers and {kb.panel.Panelist.nunique()} trained panelists.")


# ======================================================================== ask
if page == "Ask the data":
    st.title("Ask the data")
    st.write("Ask in plain English. The assistant shows what it understood, the products that match, "
             "and the findings that answer the question, each traced to its source table.")
    examples = ["bright, full-bodied Kenyan coffee", "creamy chocolate that is not bitter",
                "sweet cocktail without lemon", "does altitude affect acidity?",
                "why do consumers disagree about chocolate?", "fruity coffee"]
    if "question" not in st.session_state:
        st.session_state.question = examples[0]
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        if c.button(ex, width="stretch"):
            st.session_state.question = ex
    question = st.text_input("Question", key="question")
    if question.strip():
        ans = search_engine().ask(question)
        st.markdown(f"**Understood:** {ans.query.describe()}")
        if ans.message:
            st.info(ans.message)
        if not ans.products.empty:
            st.subheader("Products")
            st.dataframe(ans.products, hide_index=True, width="stretch")
        if not ans.findings.empty:
            st.subheader("Related findings")
            for r in ans.findings.itertuples():
                with st.expander(f"{r.finding_id}  {r.title}", expanded=True):
                    st.write(r.text)
                    st.caption(f"Source: {r.source}")
    s = read_json("search_summary.json")
    with st.expander("How reliable is this search?"):
        st.write(
            f"On a benchmark of {s['n_queries']} plain-English questions written by the authors, the assistant "
            f"understood {s['understood']:.0%} of them. Once a question is understood, its ranking is exact, so its "
            f"precision on the {s['n_ranking']} product questions ({s['assistant_precision']:.0%}) measures "
            f"understanding. For the same questions, keyword search finds {s['keyword_precision']:.0%} of the relevant "
            f"products in its top results, and {s['keyword_expanded_precision']:.0%} with the same synonyms added. On the "
            f"{s['n_abstain']} questions the data cannot answer, the assistant says so "
            f"{s['assistant_abstains_correctly']:.0%} of the time.")
        st.dataframe(read("search_benchmark.csv").round(2), hide_index=True, width="stretch")


# ======================================================================== explorer
elif page == "Product explorer":
    st.title("Product explorer")
    cat = st.radio("Category", ["coffee", "chocolate", "cocktail"], horizontal=True)
    wide = kb.wide(cat)
    if cat == "cocktail":
        wide = wide.drop(columns="color.intensity")
        st.caption("Colour intensity is left out: it is not explained by the recipe (see the data audit).")
    if cat == "coffee":
        cof = products[products.category == "coffee"].join(wide)
        c1, c2, c3 = st.columns(3)
        countries = c1.multiselect("Origin", sorted(cof.country.unique()))
        processes = c2.multiselect("Process", sorted(cof.process.unique()))
        alt_rng = c3.slider("Altitude (m)", 0, 3000, (0, 3000), step=100)
        sel = cof
        if countries:
            sel = sel[sel.country.isin(countries)]
        if processes:
            sel = sel[sel.process.isin(processes)]
        if alt_rng != (0, 3000):
            sel = sel[sel.altitude_m.between(*alt_rng)]
        attr = st.selectbox("Attribute", list(wide.columns), index=list(wide.columns).index("acidity"))
        k1, k2, k3 = st.columns(3)
        k1.metric("Coffees", f"{len(sel):,}")
        k2.metric(f"Mean {attr}", f"{sel[attr].mean():.2f}" if len(sel) else "n/a",
                  f"{sel[attr].mean() - cof[attr].mean():+.2f} vs all" if len(sel) else None)
        k3.metric("Mean total score", f"{sel.total_score.mean():.2f}" if len(sel) else "n/a")
        if len(sel):
            top = sel.groupby("country")[attr].agg(["mean", "size"]).query("size >= 5")
            top = top.sort_values("mean", ascending=False).head(15).reset_index()
            ch = alt.Chart(top).mark_bar(color=BLUE).encode(
                x=alt.X("mean:Q", title=f"mean {attr}", scale=alt.Scale(zero=False)),
                y=alt.Y("country:N", sort="-x", title=None),
                tooltip=["country", alt.Tooltip("mean:Q", format=".2f"), alt.Tooltip("size:Q", title="lots")])
            sc = alt.Chart(sel.reset_index().dropna(subset=["altitude_m"])).mark_circle(size=28, opacity=0.5,
                                                                                        color=BLUE).encode(
                x=alt.X("altitude_m:Q", title="altitude (m)"),
                y=alt.Y(f"{attr}:Q", scale=alt.Scale(zero=False)),
                tooltip=["name", "process", alt.Tooltip(f"{attr}:Q", format=".2f")])
            a, b = st.columns(2)
            a.markdown(f"**{attr} by origin** (5 lots or more)")
            a.altair_chart(ch.properties(height=360), width="stretch")
            b.markdown(f"**{attr} against altitude**")
            b.altair_chart(sc.properties(height=360), width="stretch")
        pick = st.multiselect("Compare coffees (up to 4)", sel.sort_values("total_score", ascending=False).index[:200],
                              default=list(sel.sort_values("total_score", ascending=False).index[:2]),
                              max_selections=4, format_func=lambda p: products.loc[p, "name"])
        if pick:
            prof = wide.loc[pick].reset_index().melt(id_vars="product_id", var_name="attribute")
            prof["product"] = prof.product_id.map(products.name)
            st.altair_chart(alt.Chart(prof).mark_circle(size=90).encode(
                x=alt.X("value:Q", scale=alt.Scale(zero=False), title="score (0 to 10)"),
                y=alt.Y("attribute:N", sort=list(wide.columns), title=None),
                color=alt.Color("product:N", legend=alt.Legend(orient="bottom", columns=2)),
                tooltip=["product", "attribute", alt.Tooltip("value:Q", format=".2f")]).properties(height=320),
                width="stretch")
    else:
        z = ((wide - wide.mean()) / wide.std(ddof=0)).reset_index().melt(id_vars="product_id", var_name="attribute",
                                                                          value_name="z")
        z["product"] = z.product_id.map(products.name)
        raw = wide.reset_index().melt(id_vars="product_id", var_name="attribute", value_name="value")
        z = z.merge(raw, on=["product_id", "attribute"])
        st.markdown("**Sensory profiles** (colour: above or below the average product; hover for the raw score)")
        st.altair_chart(alt.Chart(z).mark_rect().encode(
            x=alt.X("attribute:N", sort=list(wide.columns), title=None),
            y=alt.Y("product:N", sort=list(products.loc[wide.index, "name"]), title=None),
            color=alt.Color("z:Q", scale=alt.Scale(scheme="blueorange", domain=[-2.5, 2.5]), title="vs average"),
            tooltip=["product", "attribute", alt.Tooltip("value:Q", format=".2f")]).properties(
            height=28 * len(wide) + 40), width="stretch")
        like = products.loc[wide.index, ["name", "mean_liking"]].sort_values("mean_liking", ascending=False)
        st.markdown("**Mean liking** (0 to 10)")
        st.dataframe(like.round(2), width="stretch")


# ======================================================================== consumers
elif page == "Consumers and segments":
    st.title("Consumers and segments")
    cat = st.radio("Study", ["chocolate", "cocktail"], horizontal=True)
    prof = kb.wide(cat)
    if cat == "cocktail":
        prof = prof.drop(columns="color.intensity")
    lk = kb.liking_matrix(cat)
    seg = read(f"{cat}_segments.csv", index_col=0).segment
    sp = read(f"{cat}_segment_profiles.csv", index_col=0)
    meta = read_json("consumer_meta.json")[cat]
    group = st.radio("Consumers", ["all consumers"] + [c for c in sp.columns if c.startswith("segment")],
                     horizontal=True, format_func=lambda g: g if g == "all consumers"
                     else f"{g} ({(seg == int(g[-1])).sum()} consumers)")
    who = lk.columns if group == "all consumers" else seg[seg == int(group[-1])].index
    m = sensory_map(prof)
    X, Y, share, _ = preference_map(m.scores, lk[who], meta["prefmap_model"], resolution=40)
    grid = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "share": share.ravel() * 100})
    dx, dy = (X[0, 1] - X[0, 0]), (Y[1, 0] - Y[0, 0])
    heat = alt.Chart(grid).mark_rect(opacity=0.75).encode(
        x=alt.X("x:Q", title=f"PC1 ({meta['explained'][0]:.0%})", scale=alt.Scale(domain=[X.min(), X.max()], nice=False)),
        x2=alt.X2("x2:Q"), y=alt.Y("y:Q", title=f"PC2 ({meta['explained'][1]:.0%})",
                                  scale=alt.Scale(domain=[Y.min(), Y.max()], nice=False)), y2="y2:Q",
        color=alt.Color("share:Q", scale=alt.Scale(scheme="blues", domain=[20, 80], clamp=True),
                        title="% would like it")).transform_calculate(x2=f"datum.x + {dx}", y2=f"datum.y + {dy}")
    pts = m.scores.join(products.name).assign(liking=lk[who].mean(axis=1).round(2)).reset_index()
    dots = alt.Chart(pts).mark_circle(size=110, color="#0b0b0b").encode(
        x="PC1:Q", y="PC2:Q", tooltip=["name", "liking"])
    labels = dots.mark_text(dx=8, dy=-8, align="left", fontSize=12).encode(text="name:N")
    lo = m.loadings.copy()
    strength = (lo.PC1 ** 2 + lo.PC2 ** 2)
    default = []
    for side in [lo.PC1.sort_values(ascending=False), lo.PC1.sort_values()]:
        picked = [side.index[0]]
        for a_ in side.index[1:4]:
            if all(abs(lo.loc[a_, "PC2"] - lo.loc[b_, "PC2"]) > 0.25 for b_ in picked):
                picked.append(a_)
                break
        default += picked
    default.append(lo.PC2.abs().idxmax())
    shown = st.multiselect("Attributes drawn on the map", list(lo.index), default=list(dict.fromkeys(default)))
    lo = lo.loc[shown] if shown else lo.loc[strength.nlargest(5).index]
    sx = 0.85 * min(-m.scores.PC1.min(), m.scores.PC1.max())
    sy = 0.85 * min(-m.scores.PC2.min(), m.scores.PC2.max())
    vec = pd.DataFrame({"attribute": lo.index, "x": lo.PC1 * sx, "y": lo.PC2 * sy, "x0": 0.0, "y0": 0.0})
    arrows = alt.Chart(vec).mark_rule(color=ORANGE).encode(x="x0:Q", y="y0:Q", x2="x:Q", y2="y:Q")
    alabels = alt.Chart(vec).mark_text(color=ORANGE, fontSize=11).encode(x="x:Q", y="y:Q", text="attribute:N")
    a, b = st.columns([1.3, 1])
    a.markdown("**Sensory map** (shading: share of these consumers predicted to like a product at that point)")
    a.altair_chart((heat + arrows + alabels + dots + labels).properties(height=430), width="stretch")
    long = sp.reset_index().melt(id_vars="product_id", var_name="group", value_name="liking")
    long["product"] = long.product_id.map(products.name)
    b.markdown("**Mean liking by segment**")
    b.altair_chart(alt.Chart(long).mark_circle(size=90).encode(
        x=alt.X("liking:Q", scale=alt.Scale(zero=False)), y=alt.Y("product:N", title=None,
                                                                    sort=list(sp.sort_values("segment 1").index.map(products.name))),
        color=alt.Color("group:N", scale=SEG_COLORS, legend=alt.Legend(orient="bottom")),
        tooltip=["product", "group", alt.Tooltip("liking:Q", format=".2f")]).properties(height=430),
        width="stretch")
    dfile = f"{cat}_drivers.csv" if group == "all consumers" else f"{cat}_drivers_segment{group[-1]}.csv"
    st.markdown(f"**Drivers of liking, {group}**: correlation of each attribute with mean liking across "
                f"products ({len(prof)} products)")
    drv = read(dfile).round(2)
    if group != "all consumers":
        drv = drv[["attribute", "r"]]
        st.caption("Descriptive only: the segments are formed from the same liking data, along the same sensory "
                   "axis, so per-segment correlations are not tests.")
    st.dataframe(drv, hide_index=True, width="stretch")


# ======================================================================== recipe
elif page == "Recipe simulator":
    st.title("Recipe simulator (cocktails)")
    st.write("Set a recipe of the four juices (total 3.9 dl, as in the study). The mixture models predict the "
             "sensory profile and the liking of each consumer segment, with their leave-one-recipe-out error.")
    rec = kb.recipes.pivot(index="product_id", columns="ingredient", values="dl")[INGREDIENTS]
    x = proportions(rec)
    total = 3.9
    b = design_bounds(x) * total
    c1, c2, c3 = st.columns(3)
    orange = c1.slider("Orange (dl)", 0.0, 3.9, 0.78, 0.05)
    banana = c2.slider("Banana (dl)", 0.0, 3.9, 1.17, 0.05)
    lemon = c3.slider("Lemon (dl)", 0.0, 1.0, 0.0, 0.05)
    mango = round(total - orange - banana - lemon, 3)
    st.markdown(f"Mango completes the recipe: **{mango:.2f} dl**")
    new = pd.Series({"orange": orange, "banana": banana, "mango": mango, "lemon": lemon})
    out = [k for k in INGREDIENTS if not (b.loc[k, "min"] - 1e-6 <= new[k] <= b.loc[k, "max"] + 1e-6)]
    if mango < 0:
        st.error("The other juices already exceed 3.9 dl.")
        st.stop()
    if out:
        st.warning("Outside the tested range for " + ", ".join(
            f"{k} ({b.loc[k, 'min']:.2f} to {b.loc[k, 'max']:.2f} dl)" for k in out)
            + ": the prediction is an extrapolation.")
    xn = (new / total).to_frame().T[INGREDIENTS]
    prof = kb.wide("cocktail").drop(columns="color.intensity")
    lk = kb.liking_matrix("cocktail")
    seg = read("cocktail_segments.csv", index_col=0).segment
    resp = {"all consumers": lk.mean(axis=1)}
    for s_ in sorted(set(seg) - {0}):
        resp[f"segment {s_}"] = lk.loc[:, seg[seg == s_].index].mean(axis=1)
    cols = st.columns(len(resp))
    for c, (name, y) in zip(cols, resp.items()):
        mdl = fit_mixture(x, y)
        c.metric(f"Predicted liking, {name}", f"{mdl.predict(xn)[0]:.1f} / 10")
        c.caption(f"model error ± {mdl.loo_rmse:.2f} (recipe left out)")
    rows = []
    for a in prof.columns:
        mdl = fit_mixture(x, prof[a])
        rows.append({"attribute": a, "predicted": float(mdl.predict(xn)[0]), "error": mdl.loo_rmse})
    pr = pd.DataFrame(rows)
    nearest = ((x - xn.iloc[0]) ** 2).sum(axis=1).idxmin()
    pr["nearest tested"] = prof.loc[nearest].values
    pr["low"], pr["high"] = pr.predicted - pr.error, pr.predicted + pr.error
    base = alt.Chart(pr).encode(y=alt.Y("attribute:N", title=None, sort=list(prof.columns)))
    st.markdown(f"**Predicted sensory profile** (blue, with model error) against the nearest tested recipe, "
                f"{products.loc[nearest, 'name']} (grey)")
    st.altair_chart((base.mark_rule(color=BLUE, strokeWidth=2).encode(x=alt.X("low:Q", title="intensity (0 to 10)"),
                                                                       x2="high:Q")
                     + base.mark_circle(size=90, color=BLUE).encode(x="predicted:Q")
                     + base.mark_tick(color=MUTED, thickness=3, size=18).encode(x="nearest tested:Q"))
                    .properties(height=340), width="stretch")
    best = read("cocktail_best_recipes.csv", index_col=0)
    st.caption("Best recipe inside the tested range, by group: " + "; ".join(
        f"{r.Index}: {r.orange:.2f} orange, {r.banana:.2f} banana, {r.mango:.2f} mango, {r.lemon:.2f} lemon "
        f"(predicted {r.predicted:.1f})" for r in best.itertuples()))


# ======================================================================== panel
elif page == "Panel quality":
    st.title("Panel quality (chocolate study)")
    da = read("chocolate_descriptors.csv", index_col=0)
    ps = read("chocolate_panelists.csv", index_col=0)
    k1, k2, k3 = st.columns(3)
    k1.metric("Descriptors that separate the products", f"{(da.p_product < 0.001).sum()} / {len(da)}")
    k2.metric("Descriptors with panel disagreement", f"{(da.p_interaction_holm < 0.05).sum()}",
              help="Product x panelist interaction, Holm-adjusted p < 0.05")
    k3.metric("Median panelist agreement", f"{ps.median_agreement.median():.2f}")
    a, b = st.columns(2)
    a.markdown("**Descriptors**")
    a.dataframe(da[["F_product", "p_product", "F_interaction", "p_interaction_holm"]].round(4),
                width="stretch")
    b.markdown("**Panelists, weakest first**")
    b.dataframe(ps.round(2), width="stretch")
    pid = st.selectbox("Check one panelist", ps.index.tolist())
    desc = st.selectbox("Descriptor", da.index.tolist())
    pan = kb.panel
    own = pan[pan.Panelist == pid].groupby("product_id")[desc].mean().rename("this panelist")
    rest = pan[pan.Panelist != pid].groupby("product_id")[desc].mean().rename("rest of the panel")
    d = pd.concat([own, rest], axis=1).reset_index().melt(id_vars="product_id", var_name="who", value_name="score")
    d["product"] = d.product_id.map(products.name)
    st.altair_chart(alt.Chart(d).mark_line(point=True).encode(
        x=alt.X("product:N", title=None), y=alt.Y("score:Q", title=f"{desc} (0 to 10)"),
        color=alt.Color("who:N", scale=alt.Scale(range=[ORANGE, MUTED]), legend=alt.Legend(orient="bottom")),
        tooltip=["product", "who", alt.Tooltip("score:Q", format=".2f")]).properties(height=300),
        width="stretch")


# ======================================================================== findings
else:
    st.title("Findings and data audit")
    f = read("findings.csv")
    cat = st.multiselect("Category", sorted(f.category.unique()), default=sorted(f.category.unique()))
    for r in f[f.category.isin(cat)].itertuples():
        with st.expander(f"{r.finding_id}  {r.title}"):
            st.write(r.text)
            st.caption(f"Source: {r.source}")
    st.subheader("Data audit")
    au = read_json("audit.json")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coffee lots kept", f"{au['coffee_rows_clean']:,}", f"-{au['coffee_rows_all_zero']} empty")
    c2.metric("Implausible altitudes", au["coffee_altitude_implausible"], "set to missing", delta_color="off")
    c3.metric("Missing process", au["coffee_process_missing"])
    c4.metric("Distinct harvest-year entries", au["coffee_harvest_year_entries"])
    st.markdown("**Connected data model** (also exported as CSV files for Power BI in `powerbi/`)")
    st.dataframe(pd.DataFrame({
        "table": ["dim_product", "dim_attribute", "fact_sensory", "fact_liking", "fact_recipe"],
        "rows": [len(kb.products), len(kb.attributes), len(kb.sensory), len(kb.liking), len(kb.recipes)],
        "key": ["product_id", "attribute_id", "product_id + attribute_id", "product_id + consumer_id",
                "product_id + ingredient"]}), hide_index=True, width="stretch")
