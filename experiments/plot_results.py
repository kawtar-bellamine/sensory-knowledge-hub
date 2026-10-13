"""Figures for the README (reads results/, writes results/figures/)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sensehub.data import INGREDIENTS, load_knowledge_base  # noqa: E402
from sensehub.mapping import preference_map, sensory_map  # noqa: E402
from sensehub.recipe import loo_predict, proportions  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEG = {"segment 1": BLUE, "segment 2": ORANGE, "all consumers": MUTED}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
    "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def short(pid: str) -> str:
    return pid.split("_")[1].lstrip("0")


def fig_chocolate_segments(kb):
    prof = kb.wide("chocolate")
    m = sensory_map(prof)
    sp = pd.read_csv(RES / "chocolate_segment_profiles.csv", index_col=0)
    meta = json.loads((RES / "consumer_meta.json").read_text())["chocolate"]
    seg = meta["segments"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    X, Y, share, _ = preference_map(m.scores, kb.liking_matrix("chocolate"), "vector")
    cs = ax.contourf(X, Y, share * 100, levels=np.arange(20, 85, 10), cmap="Blues", alpha=0.55, zorder=0)
    sc = m.scores
    ax.scatter(sc.PC1, sc.PC2, s=46, color=INK, zorder=3)
    offsets = {"chocolate_2": (6, -12), "chocolate_5": (-14, 8), "chocolate_6": (-20, 8),
               "chocolate_1": (-62, 8), "chocolate_3": (-20, -16)}
    for pid, r in sc.iterrows():
        ax.annotate(f"chocolate {short(pid)}", (r.PC1, r.PC2), xytext=offsets.get(pid, (6, 5)),
                    textcoords="offset points", fontsize=9, color=INK, zorder=4)
    lo = m.loadings
    sx = 0.85 * min(-sc.PC1.min(), sc.PC1.max())
    sy = 0.85 * min(-sc.PC2.min(), sc.PC2.max())
    for a in ["MilkF", "Melting", "CocoaF", "Crunchy"]:
        ax.annotate("", xy=(lo.loc[a, "PC1"] * sx, lo.loc[a, "PC2"] * sy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2), zorder=2)
        ax.text(lo.loc[a, "PC1"] * sx, lo.loc[a, "PC2"] * sy * 1.12 + (0.12 if lo.loc[a, "PC2"] >= 0 else -0.25),
                a, color=ORANGE, fontsize=8.5, ha="center", zorder=4)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel(f"PC1 ({meta['explained'][0]:.0%}): milky and sweet  <->  cocoa and bitter")
    ax.set_ylabel(f"PC2 ({meta['explained'][1]:.0%})")
    ax.set_title("Sensory map and share of consumers who would like it")
    cb = fig.colorbar(cs, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("% of consumers above their own average", color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)

    ax = axes[1]
    order = sp.sort_values("segment 1").index
    ypos = np.arange(len(order))
    for col, lab in [("all consumers", "all 222 consumers"), ("segment 1", f"segment 1 ({seg['1']})"),
                     ("segment 2", f"segment 2 ({seg['2']})")]:
        ax.scatter(sp.loc[order, col], ypos, s=60 if col != "all consumers" else 46, color=SEG[col], label=lab,
                   zorder=3, marker="o" if col != "all consumers" else "D")
    for i, p in enumerate(order):
        ax.plot(sp.loc[p, ["segment 1", "segment 2"]], [i, i], color=GRID, lw=3, zorder=1)
    ax.set_yticks(ypos, [f"chocolate {short(p)}" for p in order])
    ax.set_xlabel("mean liking (0 to 10)")
    ax.set_title("Mean liking is flat, the segments are not")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), fontsize=8.5, ncol=3)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Chocolate: one sensory axis, two opposite consumer segments", x=0.01, ha="left",
                 fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "fig1_chocolate_segments.png")


def fig_drivers():
    d1 = pd.read_csv(RES / "chocolate_drivers_segment1.csv", index_col=0)
    d2 = pd.read_csv(RES / "chocolate_drivers_segment2.csv", index_col=0)
    da = pd.read_csv(RES / "chocolate_drivers.csv", index_col=0)
    dc = pd.read_csv(RES / "cocktail_drivers.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    order = d1.sort_values("r").index
    y = np.arange(len(order))
    ax.barh(y - 0.27, d1.loc[order, "r"], height=0.27, color=BLUE, label="segment 1")
    ax.barh(y, d2.loc[order, "r"], height=0.27, color=ORANGE, label="segment 2")
    ax.barh(y + 0.27, da.loc[order, "r"], height=0.27, color=MUTED, label="all consumers")
    ax.set_yticks(y, order)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("correlation with mean liking (6 chocolates)")
    ax.set_title("Chocolate: the drivers flip between segments")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), fontsize=8.5, ncol=3)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    dc = dc.sort_values("r")
    y = np.arange(len(dc))
    colors = [AQUA if p < 0.01 else MUTED for p in dc.p_products]
    ax.barh(y, dc.r, color=colors, height=0.6)
    ax.errorbar(dc.r, y, xerr=[dc.r - dc.ci_low, dc.ci_high - dc.r], fmt="none", ecolor=INK2, lw=1, capsize=2)
    ax.set_yticks(y, dc.index)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("correlation with mean liking (16 cocktails)")
    ax.set_title("Cocktail: sweet up, acidity and strength down")
    ax.text(0.98, 0.04, "green: p < 0.01 over products\nbars: 95% interval over consumers",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig2_drivers_of_liking.png")


def fig_recipe(kb):
    rec = kb.recipes.pivot(index="product_id", columns="ingredient", values="dl")[INGREDIENTS]
    x = proportions(rec)
    y = kb.liking_matrix("cocktail").mean(axis=1)
    pred = loo_predict(x, y, "linear")
    models = pd.read_csv(RES / "cocktail_recipe_models.csv", index_col=0)
    terms = pd.read_csv(RES / "cocktail_recipe_terms.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    rep = ["cocktail_08", "cocktail_09", "cocktail_10"]
    other = [p for p in y.index if p not in rep]
    ax.plot([3, 7.6], [3, 7.6], color=GRID, lw=1.5, zorder=1)
    ax.scatter(pred[other], y[other], s=44, color=BLUE, zorder=3, label="recipe")
    ax.scatter(pred[rep], y[rep], s=44, color=ORANGE, zorder=3, label="same recipe served 3 times")
    for p in ["cocktail_02", "cocktail_15", "cocktail_11"]:
        ax.annotate(f"cocktail {short(p)}", (pred[p], y[p]), xytext=(6, -10), textcoords="offset points", fontsize=8.5)
    r = models.loc["liking"]
    ax.set_xlabel("predicted liking, recipe left out of the fit")
    ax.set_ylabel("observed mean liking (100 consumers)")
    ax.set_title(f"Predicting a new recipe: error {r.loo_rmse:.2f} points")
    ax.text(0.03, 0.97, f"spread between cocktails: {r.between_product_sd:.2f}\n"
            f"spread between servings of one recipe: {r.replicate_sd:.2f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes[1]
    b = terms["liking"].loc[INGREDIENTS]
    eff = {k: 0.05 * (b[k] - b["mango"]) for k in ["orange", "banana", "lemon"]}
    segs = [c for c in terms.columns if c.startswith("liking segment")]
    labels = ["orange", "banana", "lemon"]
    yy = np.arange(len(labels))
    ax.barh(yy - 0.25, [eff[k] for k in labels], height=0.25, color=MUTED, label="all consumers")
    for i, (c, col) in enumerate(zip(segs, [BLUE, ORANGE])):
        bb = terms[c].loc[INGREDIENTS]
        ax.barh(yy + 0.25 * i, [0.05 * (bb[k] - bb["mango"]) for k in labels], height=0.25, color=col,
                label=c.replace("liking ", ""))
    ax.set_yticks(yy, [f"0.2 dl mango -> {k}" for k in labels])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("change in predicted liking (points)")
    ax.set_title("Swapping 0.2 dl of mango for another juice")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig3_recipe_model.png")


def fig_coffee():
    pe = pd.read_csv(RES / "coffee_process.csv", index_col=0)
    ae = pd.read_csv(RES / "coffee_altitude.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, df, naive, within, title, xl in [
        (axes[0], pe, "naive_diff", "within_country_diff", "Natural minus washed",
         "difference in score (points)"),
        (axes[1], ae, "naive_slope", "within_country_slope", "Per 500 m of altitude",
         "change in score (points)")]:
        d = df.drop(index="total_cup_points")
        y = np.arange(len(d))
        ax.scatter(d[naive], y + 0.15, color=MUTED, s=40, label="all lots compared", zorder=3, marker="D")
        ax.errorbar(d[within], y - 0.15, xerr=[d[within] - d.ci_low, d.ci_high - d[within]], fmt="o",
                    color=BLUE, ms=6, lw=1.4, capsize=2, label="within the same country (95% CI clustered by owner, not adjusted)", zorder=3)
        ax.set_yticks(y, d.index.str.replace("_", " "))
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.17), fontsize=8.5, ncol=2)
    tp, ta = pe.loc["total_cup_points"], ae.loc["total_cup_points"]
    fig.suptitle(f"Coffee: within a country, the process gap shrinks ({tp.naive_diff:+.2f} -> "
                 f"{tp.within_country_diff:+.2f}, wide CI); altitude holds up ({ta.naive_slope:+.2f} -> "
                 f"{ta.within_country_slope:+.2f} per 500 m)", x=0.01, ha="left", fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "fig4_coffee_confounding.png")


def fig_panel():
    da = pd.read_csv(RES / "chocolate_descriptors.csv", index_col=0).sort_values("F_product")
    ps = pd.read_csv(RES / "chocolate_panelists.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    y = np.arange(len(da))
    colors = [ORANGE if p < 0.05 else BLUE for p in da.p_interaction_holm]
    ax.barh(y, da.F_product, color=colors, height=0.6)
    ax.set_yticks(y, da.index)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("F of the product effect (log scale)")
    ax.set_title("Every descriptor separates the chocolates")
    ax.text(0.98, 0.04, "orange: panelists disagree\n(interaction, Holm p < 0.05)", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ps = ps.sort_values("median_agreement")
    y = np.arange(len(ps))
    colors = [ORANGE if v < 0.4 else BLUE for v in ps.median_agreement]
    ax.scatter(ps.median_agreement, y, color=colors, s=26, zorder=3)
    for i, (pid, v) in enumerate(ps.median_agreement.items()):
        if v < 0.4:
            ax.annotate(f"panelist {pid}", (v, i), xytext=(8, -3), textcoords="offset points", fontsize=8.5)
    ax.set_yticks([])
    ax.set_ylabel(f"{len(ps)} panelists, sorted")
    ax.set_xlim(0, 1)
    ax.axvline(ps.median_agreement.median(), color=MUTED, lw=1, ls="--")
    ax.set_xlabel("median correlation with the rest of the panel")
    ax.set_title("Two panelists disagree with the others")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig5_panel_performance.png")


def fig_search():
    res = pd.read_csv(RES / "search_benchmark.csv")
    s = json.loads((RES / "search_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 3.4))
    groups = [("coffee", ~res.abstain & (res.category == "coffee")),
              ("chocolate", ~res.abstain & (res.category == "chocolate")),
              ("cocktail", ~res.abstain & (res.category == "cocktail"))]
    x = np.arange(len(groups))
    a = [res[m].assistant.mean() for _, m in groups]
    k = [res[m].keyword_search.mean() for _, m in groups]
    kx = [res[m].keyword_expanded.mean() for _, m in groups]
    ax.bar(x - 0.27, a, width=0.27, color=BLUE, label="knowledge assistant")
    ax.bar(x, kx, width=0.27, color=AQUA, label="keyword search + the same synonyms")
    ax.bar(x + 0.27, k, width=0.27, color=MUTED, label="keyword search (TF-IDF)")
    ax.set_xticks(x, [f"{g}\n{int(m.sum())} questions" for g, m in groups])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("precision of top results")
    ax.set_title(f"Search benchmark, {s['n_ranking']} product questions: precision of the top results "
                 "(the assistant's is 1 whenever it understands the question)")
    ax.legend(loc="upper center", fontsize=8.5, ncol=3, bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig6_search_benchmark.png")


def main():
    kb = load_knowledge_base()
    fig_chocolate_segments(kb)
    fig_drivers()
    fig_recipe(kb)
    fig_coffee()
    fig_panel()
    fig_search()


if __name__ == "__main__":
    main()
