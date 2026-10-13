"""Run every analysis and write the results used by the README, the figures and the dashboard.

    python experiments/run_all.py        (about 30 seconds)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sensehub import coffee as cf  # noqa: E402
from sensehub.benchmark import evaluate, summarise  # noqa: E402
from sensehub.data import CHOCOLATE_ATTRIBUTES, INGREDIENTS, clean_coffee, load_knowledge_base  # noqa: E402
from sensehub.mapping import (drivers_of_liking, preference_map, preference_structure_test,  # noqa: E402
                              segment_consumers, segment_profiles, sensory_map)
from sensehub.panel import descriptor_anova, panelist_performance, panelist_summary  # noqa: E402
from sensehub.recipe import best_recipe, fit_mixture, loo_rmse, proportions, replicate_sd  # noqa: E402
from sensehub.search import KeywordBaseline, KnowledgeSearch, product_cards  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def save_json(obj, name):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def main():
    kb = load_knowledge_base()
    findings = []

    def finding(title, text, category, source):
        findings.append({"title": title, "text": text, "category": category, "source": source})

    # ------------------------------------------------------------------ E1 data
    print("E1 data audit and knowledge base")
    a = kb.audit
    save_json(a, "audit.json")
    kb.to_sqlite(OUT / "knowledge.db")
    kb.to_powerbi(ROOT / "powerbi")
    raw = pd.read_csv(ROOT / "data" / "coffee_ratings.csv")
    bad = raw[(raw.altitude_mean_meters < 100) | (raw.altitude_mean_meters > 3000)]
    examples = (bad.drop_duplicates("altitude_mean_meters").sort_values("altitude_mean_meters", ascending=False)
                [["altitude", "altitude_mean_meters"]].head(3))
    ex = "; ".join(f'"{r.altitude}" recorded as {r.altitude_mean_meters:,.0f} m' for r in examples.itertuples())
    finding("Coffee altitudes: recording errors set to missing",
            f"{a['coffee_altitude_implausible']} coffee lots have an altitude below 100 m or above 3,000 m, "
            f"for example {ex}. Lost decimal points and feet typed as metres are the likely causes. "
            f"They are set to missing rather than guessed; {a['coffee_altitude_missing']} lots had no altitude at all. "
            f"One lot with every score at zero was removed, and the harvest year comes as "
            f"{a['coffee_harvest_year_entries']} distinct entries, such as '4T/10', 'Mayo a Julio' or '2017 / 2018'.",
            "coffee", "data audit")
    finding("Coffee total score is the sum of the ten attributes",
            f"The total cup score equals the sum of the ten attribute scores (largest gap "
            f"{a['coffee_total_minus_sum_max']} points). A model that predicts the total from the attributes "
            "is therefore circular, whatever its accuracy.", "coffee", "data audit")
    finding("The chocolate liking scale direction is ambiguous in the documentation",
            "The package documentation describes the chocolate liking scale as 0 = liking and 10 = disliking, "
            "the opposite of the cocktail scale. This project reads 10 as the most liked, like the cocktail data "
            "and the package's own preference-mapping functions. If the documentation is right, every chocolate "
            "liking conclusion flips (the most liked becomes the least liked, and each segment's favourites "
            "reverse); the segments themselves and the panel results do not change.", "chocolate", "data audit")

    # ------------------------------------------------------------------ E2 coffee
    print("E2 coffee insights")
    coffee, _ = clean_coffee(raw)
    arab = coffee[coffee.species == "Arabica"]
    st = cf.attribute_structure(arab[cf.GRADED])
    st["correlation"].round(3).to_csv(OUT / "coffee_attribute_correlation.csv")
    save_json({k: v for k, v in st.items() if k != "correlation"} | {"share_at_10": a["coffee_share_at_10"]},
              "coffee_structure.json")
    finding("The graded coffee attributes mostly measure one thing",
            f"On the {len(arab):,} arabica lots, the first principal component carries "
            f"{st['first_component_share']:.0%} of the variance of the seven graded attributes (median correlation "
            f"{st['median_correlation']:.2f}). Uniformity, clean cup and sweetness are scored 10 out of 10 for "
            f"{min(a['coffee_share_at_10'].values()):.0%} to {max(a['coffee_share_at_10'].values()):.0%} of lots. "
            "A dashboard showing all ten attributes side by side mostly repeats one overall quality signal.",
            "coffee", "coffee_structure.json")

    pe = cf.process_effect(arab)
    pe.round(4).to_csv(OUT / "coffee_process.csv")
    t = pe.loc["total_cup_points"]
    save_json(pe.attrs, "coffee_process_meta.json")
    finding("The natural-versus-washed gap cannot be separated from origin",
            f"Across all arabica lots, natural (dry) coffees score {pe.attrs['naive_total_all_lots']:+.2f} points above "
            f"washed ones, and {t.naive_diff:+.2f} in the {len(pe.attrs['countries'])} countries that use both "
            f"({pe.attrs['n_lots']} lots). Compared within the same country, the gap is {t.within_country_diff:+.2f} "
            f"points (95% CI {t.ci_low:+.2f} to {t.ci_high:+.2f}, standard errors clustered by owner). The interval is "
            "wide: the data can neither confirm nor rule out a process effect as large as the raw gap. What it does "
            "show is that the raw comparison mixes process with origin, so it should not be quoted as a process "
            "effect. No attribute differs after a Holm correction.",
            "coffee", "coffee_process.csv")

    ae = cf.altitude_effect(arab)
    ae.round(4).to_csv(OUT / "coffee_altitude.csv")
    t = ae.loc["total_cup_points"]
    acid = ae.loc["acidity"]
    sd_total = arab.total_cup_points.std()
    finding("Higher-grown coffee scores a little higher, even within a country",
            f"Each extra 500 m of altitude goes with {t.within_country_slope:+.2f} total points within the same "
            f"country and process (95% CI {t.ci_low:+.2f} to {t.ci_high:+.2f}, clustered by owner; "
            f"{ae.attrs['n_lots']:,} lots), about half "
            f"the raw slope of {t.naive_slope:+.2f}. Acidity rises the most ({acid.within_country_slope:+.3f} per 500 m). "
            f"The effect is real but small: one standard deviation of the total score is {sd_total:.1f} points.",
            "coffee", "coffee_altitude.csv")

    # ------------------------------------------------------------------ E3 panel
    print("E3 chocolate panel performance")
    da = descriptor_anova(kb.panel, CHOCOLATE_ATTRIBUTES)
    da["p_interaction_holm"] = cf.holm(da.p_interaction)
    da.to_csv(OUT / "chocolate_descriptors.csv")
    perf = panelist_performance(kb.panel, CHOCOLATE_ATTRIBUTES)
    summ = panelist_summary(perf)
    summ.to_csv(OUT / "chocolate_panelists.csv")
    perf.to_csv(OUT / "chocolate_panelist_detail.csv", index=False)
    n_sig = int((da.p_product < 0.001).sum())
    inter = da.index[da.p_interaction_holm < 0.05].tolist()
    weak = summ.head(2)
    finding("The chocolate panel separates the six chocolates on every descriptor",
            f"{n_sig} of 14 descriptors separate the chocolates at p < 0.001 (product effect tested against the "
            f"panelist-by-product interaction). After a Holm correction, the panelists rank the products "
            f"differently on {len(inter)} descriptor{'s' if len(inter) != 1 else ''} "
            f"({', '.join(inter) if inter else 'none'}), the first to recalibrate in panel training.",
            "chocolate", "chocolate_descriptors.csv")
    lazy = summ.discriminates.idxmin()
    finding("Three panelists to check before their scores weigh on decisions",
            f"Panelists {int(weak.index[0])} and {int(weak.index[1])} agree with the others with a median "
            f"correlation of {weak.median_agreement.iloc[0]:.2f} and {weak.median_agreement.iloc[1]:.2f}, against "
            f"{summ.median_agreement.median():.2f} for the median panelist. Panelist {int(lazy)} separates the "
            f"products on {round(summ.discriminates[lazy] * 14)} of the 14 descriptors on their own scores (median "
            f"panelist: {round(summ.discriminates.median() * 14)}).", "chocolate", "chocolate_panelists.csv")

    # ------------------------------------------------------------------ E4 consumers
    print("E4 sensory maps, segments, preference maps")
    meta = {}
    for cat, model in [("chocolate", "vector"), ("cocktail", "circular")]:
        prof = kb.wide(cat)
        if cat == "cocktail":
            prof = prof.drop(columns="color.intensity")
        lk = kb.liking_matrix(cat)
        m = sensory_map(prof)
        m.scores.to_csv(OUT / f"{cat}_map_scores.csv")
        m.loadings.to_csv(OUT / f"{cat}_map_loadings.csv")
        seg, sil = segment_consumers(lk)
        seg.rename("segment").to_csv(OUT / f"{cat}_segments.csv")
        sil.to_csv(OUT / f"{cat}_silhouette.csv", index=False)
        sp = segment_profiles(lk, seg)
        sp.to_csv(OUT / f"{cat}_segment_profiles.csv")
        dr = drivers_of_liking(prof, lk)
        dr.to_csv(OUT / f"{cat}_drivers.csv")
        for s in sorted(set(seg) - {0}):
            drivers_of_liking(prof, lk.loc[:, seg[seg == s].index]).to_csv(OUT / f"{cat}_drivers_segment{s}.csv")
        test = preference_structure_test(lk)
        r_axis = float(np.corrcoef(test["product_scores"], m.scores.loc[lk.index, "PC1"])[0, 1])
        _, _, share, r2 = preference_map(m.scores, lk, model)
        meta[cat] = {"explained": m.explained.tolist(), "segments": seg.value_counts().sort_index().to_dict(),
                     "silhouette": sil.set_index("k").silhouette.to_dict(),
                     "preference_first_share": test["first_share"], "preference_null_95": test["null_95"],
                     "preference_p": test["p"], "preference_axis_vs_sensory_pc1": r_axis,
                     "prefmap_model": model, "prefmap_max_share": float(share.max()),
                     "prefmap_median_r2": float(np.nanmedian(r2))}
    save_json(meta, "consumer_meta.json")

    cm = meta["chocolate"]
    sp = pd.read_csv(OUT / "chocolate_segment_profiles.csv", index_col=0)
    loved = (sp["segment 1"] - sp["segment 2"]).idxmax()
    finding("The six chocolates differ along one sensory axis",
            f"The first component of the panel profiles carries {cm['explained'][0]:.0%} of the variance: milky, "
            "sweet, caramel and melting on one side, cocoa, bitter, astringent and crunchy on the other. "
            f"{loved.replace('_', ' ').capitalize()} is the milky outlier.", "chocolate", "chocolate_map_loadings.csv")
    finding("Average liking hides two opposite chocolate segments",
            f"Mean liking ranges only from {sp['all consumers'].min():.1f} to {sp['all consumers'].max():.1f} out of 10, "
            f"as if no chocolate stood out. But consumers fall into two segments ({cm['segments'].get(1)} and "
            f"{cm['segments'].get(2)} consumers): {loved.replace('_', ' ')} scores {sp.loc[loved, 'segment 1']:.1f} in the "
            f"first and {sp.loc[loved, 'segment 2']:.1f} in the second. The shared preference axis is real (it carries "
            f"{cm['preference_first_share']:.0%} of the liking variance, against a 95th percentile of "
            f"{cm['preference_null_95']:.0%} for shuffled data, p < 0.001) and it is the sensory axis itself "
            f"(correlation {abs(cm['preference_axis_vs_sensory_pc1']):.2f}). The segment boundary is soft "
            f"(silhouette {max(cm['silhouette'].values()):.2f}): consumers spread along that axis rather than "
            "forming two separate clusters.", "chocolate", "chocolate_segment_profiles.csv")
    dch = pd.read_csv(OUT / "chocolate_drivers.csv", index_col=0)
    d1 = pd.read_csv(OUT / "chocolate_drivers_segment1.csv", index_col=0)
    d2 = pd.read_csv(OUT / "chocolate_drivers_segment2.csv", index_col=0)
    finding("No attribute drives chocolate liking on average, because the segments pull both ways",
            f"Across all consumers, the strongest correlation with mean liking is "
            f"{dch.r.loc[dch.r.abs().idxmax()]:+.2f} ({dch.r.abs().idxmax()}), and no attribute reaches p < 0.05 over "
            f"six products. Within segments the picture is sharp: melting correlates {d1.loc['Melting', 'r']:+.2f} with "
            f"liking in the first segment and {d2.loc['Melting', 'r']:+.2f} in the second, while crunchy goes "
            f"{d1.loc['Crunchy', 'r']:+.2f} and {d2.loc['Crunchy', 'r']:+.2f}. This flip partly follows from how the "
            "segments are formed (along the same sensory axis), so it describes the segments rather than testing "
            "them. A team reading only the averages would conclude that nothing matters.", "chocolate", "chocolate_drivers.csv")
    dck = pd.read_csv(OUT / "cocktail_drivers.csv", index_col=0)
    sig = dck[dck.p_products < 0.01]
    finding("Cocktail liking is driven by sweetness and hurt by acidity, strength and bitterness",
            "Across the 16 cocktails, mean liking correlates with " +
            ", ".join(f"{i} ({r.r:+.2f})" for i, r in sig.iterrows()) +
            " (each p < 0.01 over products).", "cocktail", "cocktail_drivers.csv")

    # ------------------------------------------------------------------ E5 recipe
    print("E5 recipe models (cocktails)")
    rec = kb.recipes.pivot(index="product_id", columns="ingredient", values="dl")[INGREDIENTS]
    x = proportions(rec)
    prof = kb.wide("cocktail")
    lk = kb.liking_matrix("cocktail")
    seg = pd.read_csv(OUT / "cocktail_segments.csv", index_col=0).segment
    resp = prof.copy()
    resp["liking"] = lk.mean(axis=1)
    for s in sorted(set(seg) - {0}):
        resp[f"liking segment {s}"] = lk.loc[:, seg[seg == s].index].mean(axis=1)
    rows, terms, best = [], {}, []
    for c in resp.columns:
        m = fit_mixture(x, resp[c])
        rows.append({"response": c, "model": m.kind, "loo_rmse": m.loo_rmse,
                     "loo_rmse_linear": loo_rmse(x, resp[c], "linear"),
                     "loo_rmse_quadratic": loo_rmse(x, resp[c], "quadratic"),
                     "replicate_sd": replicate_sd(x, resp[c]), "between_product_sd": resp[c].std()})
        terms[c] = m.terms()
        if c.startswith("liking"):
            r, v = best_recipe(m, x)
            nearest = ((x - r) ** 2).sum(axis=1).idxmin()
            best.append({"response": c, **(r * rec.sum(axis=1).median()).round(3).to_dict(), "predicted": v,
                         "loo_rmse": m.loo_rmse, "best_tested": resp[c].idxmax(),
                         "best_tested_value": resp[c].max(), "nearest_tested": nearest,
                         "distance_to_nearest": float(np.sqrt(((x.loc[nearest] - r) ** 2).sum()))})
    models = pd.DataFrame(rows).set_index("response")
    models.to_csv(OUT / "cocktail_recipe_models.csv")
    pd.DataFrame(terms).to_csv(OUT / "cocktail_recipe_terms.csv")
    best = pd.DataFrame(best).set_index("response")
    best.to_csv(OUT / "cocktail_best_recipes.csv")

    lik = models.loc["liking"]
    name = kb.products.set_index("product_id").name
    b = best.loc["liking"]
    tm = pd.DataFrame(terms)
    finding("Cocktail liking is predictable from the recipe",
            f"A mixture model of the four juice proportions predicts mean liking of a recipe it has not seen with an "
            f"error of {lik.loo_rmse:.2f} points (leave-one-recipe-out), against {lik.between_product_sd:.2f} points of "
            f"spread between cocktails and {lik.replicate_sd:.2f} between three servings of the same recipe. "
            f"Lemon costs the most: replacing 0.2 dl of mango with lemon lowers predicted liking by "
            f"{0.05 * (tm.loc['mango', 'liking'] - tm.loc['lemon', 'liking']):.1f} points.",
            "cocktail", "cocktail_recipe_models.csv")
    finding("No untested recipe is predicted to beat the best cocktail",
            f"Searching every recipe inside the tested range, the model's best is "
            f"{b.orange:.2f} dl orange, {b.banana:.2f} banana, {b.mango:.2f} mango and {b.lemon:.2f} lemon, which is "
            f"exactly {name[b.nearest_tested]}, already the most liked ({b.best_tested_value:.2f}). A linear mixture "
            "model always puts its optimum at a corner of the tested range, so its useful output is a direction "
            "for the next tasting: less orange and no lemon, beyond the tested range, to be confirmed by "
            "consumers.", "cocktail", "cocktail_best_recipes.csv")
    lemon_dl = rec.lemon
    gaps = {}
    for s_ in sorted(set(seg) - {0}):
        m_ = lk.loc[:, seg[seg == s_].index].mean(axis=1)
        gaps[s_] = (m_[lemon_dl == lemon_dl.max()].mean(), m_[lemon_dl == 0].mean())
    sp_ck = pd.read_csv(OUT / "cocktail_segment_profiles.csv", index_col=0)
    big = max(gaps, key=lambda k: gaps[k][1] - gaps[k][0])
    other = [k for k in gaps if k != big][0]
    n_seg = seg.value_counts()
    finding("A lemon-averse segment of cocktail drinkers",
            f"{n_seg[big]} of the 100 consumers rate the cocktails with the most lemon ({lemon_dl.max():.2f} dl) "
            f"{gaps[big][0]:.1f} on average, against {gaps[big][1]:.1f} for those without lemon. For the other "
            f"{n_seg[other]} consumers the gap is smaller ({gaps[other][0]:.1f} against {gaps[other][1]:.1f}). "
            f"Their favourite is {name[sp_ck['segment ' + str(big)].idxmax()]} "
            f"({sp_ck['segment ' + str(big)].max():.1f}), while the larger group prefers "
            f"{name[sp_ck['segment ' + str(other)].idxmax()]}.", "cocktail", "cocktail_segment_profiles.csv")
    ci = prof.loc[["cocktail_08", "cocktail_09", "cocktail_10"], "color.intensity"]
    finding("Cocktail colour intensity is not explained by the recipe",
            f"Three servings of the same recipe (cocktails 8, 9 and 10) score {ci.iloc[0]:.1f}, {ci.iloc[1]:.1f} and "
            f"{ci.iloc[2]:.1f} for colour intensity, and the scores split into two groups around 3 and 8 whatever the "
            "juices. A factor missing from the data (a colouring, or how samples were served) drives it, so colour "
            "is left out of the maps and the recipe model.", "cocktail", "senso_cocktail.csv")

    # ------------------------------------------------------------------ findings + E6 search
    print("E6 search benchmark")
    fdf = pd.DataFrame(findings)
    fdf.insert(0, "finding_id", [f"F{i + 1:02d}" for i in range(len(fdf))])
    fdf.to_csv(OUT / "findings.csv", index=False)
    wide = {c: kb.wide(c) for c in ["coffee", "chocolate", "cocktail"]}
    search = KnowledgeSearch(kb.products, wide, fdf)
    cards = product_cards(kb.products.set_index("product_id"), wide)
    res = evaluate(search, KeywordBaseline(cards))
    res.to_csv(OUT / "search_benchmark.csv", index=False)
    s = summarise(res)
    save_json(s, "search_summary.json")
    print(json.dumps(s, indent=1))
    print(f"{len(fdf)} findings written")


if __name__ == "__main__":
    main()
