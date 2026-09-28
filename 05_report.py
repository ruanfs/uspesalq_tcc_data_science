"""
05_report.py — Consolida resultados em tabelas LaTeX, figuras e resumo Markdown.

[v6] Correções de KeyError/AttributeError, DeLong, arquétipos com IC, permutação por bloco,
     interações por bloco, partidas por patch, recorte da rodada (run_config.json),
     waterfalls 1º vs 8º.

Entrada: data/processed/{features_log.json, feature_dict.csv}
         results/{metrics.csv, comparisons.csv, temporal.csv, oof_predictions.parquet,
                  run_config.json}
         results/shap/{model}_{sc}/{importance,grouped,hypotheses,stability,
                                    permutation_blocks,interactions_by_block,h1b_monotonic}.csv
         results/shap/logreg_vs_shap.csv
         report/archetypes_report.csv (gerado pelo 02b)
Saídas (--out): tables/*.tex, figures/*.png, report.md

Uso:
    python 05_report.py --res results --out report --model lgbm
"""
import argparse, json, logging, shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("report")
plt.rcParams.update({"font.size": 10, "figure.dpi": 200, "savefig.bbox": "tight"})

NAMES = {"baseline_level": "Baseline (nível)", "logreg": "Reg. Logística L1",
         "rf": "Random Forest", "lgbm": "LightGBM"}
BLOCKS = {"trait": "Traits", "unit": "Unidades", "item": "Itens", "unit_item": "Unidade×Item",
          "aggregate": "Agregados", "level": "Nível", "patch": "Patch", "archetype": "Arquétipos"}
HYP_TXT = {"H1_traits_vs_units": "H1 (compl.): traits > unidades",
           "H1b_high_tier": "H1: traits ouro/prismáticas > nº de traits ativas",
           "H2_quality_vs_quantity": "H2: 4-5 custos 2★ > custo total",
           "H3_concentration_vs_total": "H3: itens nos carregadores > total de itens completos"}
CI = "IC 95%"   # [v6] sem barra: o escape do pandas cuida do %
HYP_COL = f"Diferença [{CI}]"
SHAP_FIGS = ["beeswarm.png", "bar.png", "grouped_blocks.png", "interaction_heatmap.png",
             "dep_n_traits_gold_plus.png", "dep_n_hi_cost_2star.png", "dep_sum_cost.png",
             "dep_n_carries.png", "dep_n_full_items_on_hi_cost.png", "dep_board_value.png",
             "waterfall_confident_hit.png", "waterfall_false_positive.png",
             "waterfall_false_negative.png", "waterfall_uncertain.png",
             "waterfall_first_place.png", "waterfall_eighth_place.png"]   # [v6]


def label(k):
    if k in NAMES: return NAMES[k]
    m, sc = k.rsplit("_", 1)
    return f"{NAMES.get(m, m)} ({sc})"


def blk(b): return BLOCKS.get(b, b)


def ci(v, lo, hi, d=3):
    return f"{v:.{d}f} [{lo:.{d}f}; {hi:.{d}f}]"


def fmt_p(p):
    return "—" if pd.isna(p) else ("<0,001" if p < 1e-3 else f"{p:.3f}")


def fmt_n(v): return f"{int(v):,}".replace(",", ".")


def read(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def to_tex(df, path, caption, lab):
    tex = df.to_latex(index=False, escape=True, caption=caption, label=f"tab:{lab}",
                      position="htbp", column_format="l" + "c" * (df.shape[1] - 1))
    tex = tex.replace("\\begin{tabular}", "\\centering\n\\small\n\\begin{tabular}")
    path.write_text(tex, encoding="utf-8")


# ================= Tabelas =================
def tab_data(inp, T, cfg):
    fl = json.load(open(inp / "features_log.json"))
    fd = read(inp / "feature_dict.csv")
    by_block = (fd.drop_duplicates("feature").query("block != 'level_model_B'")["block"]
                .value_counts().to_dict() if fd is not None else fl["by_block"])
    fl["n_features"] = sum(by_block.values())
    rows = [("Partidas", fmt_n(fl["n_matches"])),
            ("Observações (jogador × partida)", fmt_n(fl["n_rows"]))]
    if cfg.get("min_level"):
        rows += [(f"Recorte level ≥ {cfg['min_level']}: observações", fmt_n(cfg["n_rows"])),
                 (f"Recorte level ≥ {cfg['min_level']}: taxa de Top 4", f"{cfg['top4_rate']:.3f}")]
    if cfg.get("drop_agg"):
        rows.append(("Agregados removidos", ", ".join(cfg["drop_agg"])))
    if cfg.get("drop_blocks"):
        rows.append(("Blocos removidos", ", ".join(blk(b) for b in cfg["drop_blocks"])))
    rows += [(f"  Partidas no patch {p}", fmt_n(c)) for p, c in sorted(fl.get("patches", {}).items())]
    rows += [("Atributos finais", fl["n_features"]),
             *[(f"  — {blk(k)}", v) for k, v in by_block.items()],
             ("Removidos por baixa frequência", fl["n_dropped_rare"]),
             ("Frequência mínima", f"{fl['min_freq']:.1%}"),
             ("Esparsidade (blocos esparsos)", f"{fl['sparsity_sparse_blocks']:.1%}")]   # [v6]
    to_tex(pd.DataFrame(rows, columns=["Item", "Valor"]), T / "dados.tex",
           "Caracterização da base de dados.", "dados")
    return fl


def tab_metrics(met, T):
    d = pd.DataFrame({
        "Modelo": met["model"].map(label),
        f"AUC [{CI}]": [ci(r.auc, r.auc_lo, r.auc_hi) for r in met.itertuples()],
        f"Log loss [{CI}]": [ci(r.logloss, r.logloss_lo, r.logloss_hi) for r in met.itertuples()],
        "Brier": met["brier"].round(3), "F1": met["f1"].round(3),
        "Acc. Top4/partida": met["match_top4_acc"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}"),
    })
    to_tex(d, T / "metricas.tex", "Desempenho out-of-fold (validação cruzada aninhada "
           "agrupada por partida). IC por bootstrap de partidas.", "metricas")


def tab_comparisons(comp, T, model):
    keys = {f"{model}_A", f"{model}_B", "baseline_level"}
    c = comp[(comp["metric"] == "auc") & (comp["m1"].isin(keys) | comp["m2"].isin(keys))].copy()
    if "delong_p" not in c: c["delong_p"] = np.nan
    d = pd.DataFrame({
        "Comparação": c["m1"].map(label) + " vs " + c["m2"].map(label),
        f"Dif. AUC [{CI}]": [ci(r.diff, r.lo, r.hi, 4) for r in c.itertuples()],
        "p (bootstrap, Holm)": c.get("p_holm", c["p_value"]).map(fmt_p),
        "p (DeLong, Holm)": c.get("delong_p_holm", c["delong_p"]).map(fmt_p),
    })
    to_tex(d, T / "comparacoes.tex", "Diferenças pareadas de AUC (bootstrap por partida e "
           "teste de DeLong).", "comparacoes")


def tab_hypotheses(S, T, model):
    rows = []
    for sc in ("A", "B"):
        h = read(S / f"{model}_{sc}" / "hypotheses.csv")
        if h is None: continue
        for r in h.itertuples():
            rows.append({"Hipótese": HYP_TXT.get(r.hypothesis, r.hypothesis), "Cenário": sc,
                         "|SHAP| quali": f"{r.qual_side:.3f}", "|SHAP| quanti": f"{r.quant_side:.3f}",
                         HYP_COL: ci(r.diff, r.lo, r.hi),
                         "Sustentada": "Sim" if r.supported else "Não"})
    hyp = pd.DataFrame(rows)
    if len(hyp):
        to_tex(hyp, T / "hipoteses.tex", "Teste das hipóteses via SHAP (log-odds). "
               "Sustentada quando o limite inferior do IC é maior que zero.", "hipoteses")
    return hyp


def tab_h1b(S, T, model):   # [v6] monotonicidade da H1
    parts = []
    for sc in ("A", "B"):
        m = read(S / f"{model}_{sc}" / "h1b_monotonic.csv")
        if m is not None:
            m.columns = ["n_traits_ouro", sc]
            parts.append(m.set_index("n_traits_ouro"))
    if not parts: return
    d = pd.concat(parts, axis=1).round(3).reset_index()
    d.columns = ["Nº traits ouro/prismáticas"] + [f"SHAP médio ({c})" for c in d.columns[1:]]
    to_tex(d, T / "h1_monotonicidade.tex",
           "SHAP médio por número de traits em tier ouro ou superior.", "h1mono")


def tab_grouped(S, T, model):
    parts = []
    for sc in ("A", "B"):
        g = read(S / f"{model}_{sc}" / "grouped.csv")
        if g is None: continue
        parts.append(g.assign(sc=sc, txt=[f"{r.share:.1%} [{r.share_lo:.1%}; {r.share_hi:.1%}]"
                                          for r in g.itertuples()]))
    if not parts: return None
    G = pd.concat(parts)
    W = G.pivot(index="block", columns="sc", values="txt").fillna("—").reset_index()
    W["block"] = W["block"].map(blk)
    W.columns = ["Bloco"] + [f"Cenário {c}" for c in W.columns[1:]]
    to_tex(W, T / "blocos.tex", "Participação de cada bloco na importância SHAP.", "blocos")
    return G


def tab_permutation(S, T, model):   # [v6]
    parts = []
    for sc in ("A", "B"):
        p = read(S / f"{model}_{sc}" / "permutation_blocks.csv")
        if p is not None:
            parts.append(p.assign(sc=sc, txt=[f"{r.auc_drop:.4f} ± {r.sd:.4f}"
                                              for r in p.itertuples()]))
    if not parts: return None
    P = pd.concat(parts)
    W = P.pivot(index="block", columns="sc", values="txt").fillna("—").reset_index()
    W["block"] = W["block"].map(blk)
    W.columns = ["Bloco"] + [f"Queda de AUC ({c})" for c in W.columns[1:]]
    to_tex(W, T / "permutacao.tex", "Importância por permutação conjunta de cada bloco "
           "(queda média de AUC ± desvio-padrão).", "permutacao")
    return P


def tab_interactions(S, T, model, k=10):   # [v6]
    for sc in ("A", "B"):
        ib = read(S / f"{model}_{sc}" / "interactions_by_block.csv")
        if ib is not None and len(ib):
            d = ib.head(k).copy()
            d["Par de blocos"] = d["b1"].map(blk) + " × " + d["b2"].map(blk)
            d = d[["Par de blocos", "sum", "count"]].round({"sum": 4})
            d.columns = ["Par de blocos", "Soma |interação|", "Nº de pares"]
            to_tex(d, T / f"interacoes_blocos_{sc}.tex",
                   f"Interações SHAP agregadas por par de blocos — cenário {sc}.", f"interb{sc}")
        it = read(S / f"{model}_{sc}" / "interactions.csv")
        if it is not None and len(it):
            it.columns = ["Atributo 1", "Atributo 2", "Média |interação|"]
            to_tex(it.head(15).round(4), T / f"interacoes_top_{sc}.tex",
                   f"Top 15 pares de interação SHAP — cenário {sc}.", f"intert{sc}")


def tab_top_features(S, T, model, k=15):
    for sc in ("A", "B"):
        imp = read(S / f"{model}_{sc}" / "importance.csv")
        if imp is None: continue
        d = imp.head(k).assign(block=lambda x: x["block"].map(blk))
        d = d[["rank", "feature", "block", "mean_abs_shap", "mean_shap"]].round(4)
        d.columns = ["#", "Atributo", "Bloco", "Média |SHAP|", "Média SHAP"]
        to_tex(d, T / f"top_features_{sc}.tex",
               f"Top {k} atributos por importância SHAP — cenário {sc}.", f"top{sc}")


def tab_archetypes(arch_path, T):   # [v6]
    a = read(arch_path)
    if a is None:
        log.warning(f"{arch_path} não encontrado: tabela de arquétipos pulada"); return None
    a = a.sort_values("top4_rate", ascending=False)
    d = pd.DataFrame({
        "Arquétipo": a["archetype_name"].str.replace("archetype_", "", regex=False),
        "n": a["n_players"].map(fmt_n),
        "Participação": a["share"].map(lambda v: f"{v:.1%}") if "share" in a else "—",
        f"Top 4 [{CI}]": [ci(r.top4_rate, r.top4_lo, r.top4_hi) for r in a.itertuples()]
                         if "top4_lo" in a else a["top4_rate"].round(3),
        "Difere de 50%": a["diff_vs_50"].map(lambda v: "Sim" if v else "Não")
                         if "diff_vs_50" in a else "—",
    })
    to_tex(d, T / "arquetipos.tex", "Arquétipos de composição (K-Means sobre traits e unidades "
           "binarizadas). IC por bootstrap de partidas.", "arquetipos")
    return a


def tab_robustness(S, R, T, model):
    rows = []
    for sc in ("A", "B"):
        st = read(S / f"{model}_{sc}" / "stability.csv")
        if st is not None:
            rows.append({"Análise": f"Estabilidade entre folds — modelos OOF ({sc})",
                         "Spearman": f"{st.spearman.mean():.3f} ± {st.spearman.std():.3f}",
                         "Jaccard@20": f"{st.jaccard_top20.mean():.3f}"})
    lg = read(S / "logreg_vs_shap.csv")
    if lg is not None:
        for r in lg.itertuples():
            rows.append({"Análise": f"SHAP vs coef. Logística ({r.scenario})",
                         "Spearman": f"{r.spearman_all:.3f}", "Jaccard@20": f"{r.jaccard_top20:.3f}"})
    tp = read(R / "temporal.csv")
    if tp is not None:
        for r in tp[tp["model"].str.startswith(model)].itertuples():
            rows.append({"Análise": f"Temporal: teste no patch {r.test_patch} ({r.model[-1]})",
                         "Spearman": f"AUC={r.auc:.3f}", "Jaccard@20": f"LogLoss={r.logloss:.3f}"})
    rob = pd.DataFrame(rows)
    if len(rob):
        to_tex(rob, T / "robustez.tex", "Análises de robustez.", "robustez")
    return rob


# ================= Figuras =================
def fig_roc_calib(oof, F):
    cols = [c for c in oof.columns if c.startswith("p_")]
    y = oof["top4"].values
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for c in cols:
        k = c[2:]; fpr, tpr, _ = roc_curve(y, oof[c])
        ax[0].plot(fpr, tpr, label=label(k), lw=1.2)
        pt, pp = calibration_curve(y, oof[c], n_bins=15, strategy="quantile")
        ax[1].plot(pp, pt, marker="o", ms=3, label=label(k), lw=1)
    for a_ in ax: a_.plot([0, 1], [0, 1], "k--", lw=.8)
    ax[0].set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos", title="Curva ROC")
    ax[1].set(xlabel="Probabilidade prevista", ylabel="Frequência observada", title="Calibração")
    ax[1].legend(fontsize=7)
    plt.savefig(F / "roc_calibracao.png"); plt.close()


def fig_auc_forest(met, F):
    m = met.sort_values("auc")
    plt.figure(figsize=(6, 0.4 * len(m) + 1))
    plt.errorbar(m["auc"], range(len(m)), xerr=[m["auc"] - m["auc_lo"], m["auc_hi"] - m["auc"]],
                 fmt="o", capsize=3)
    plt.yticks(range(len(m)), m["model"].map(label)); plt.xlabel("AUC (IC 95%)")
    plt.grid(axis="x", alpha=.3); plt.savefig(F / "auc_ic.png"); plt.close()


def fig_placement(oof, F, model):
    c = f"p_{model}_A"
    if c not in oof: return
    ks = sorted(oof["placement"].unique())
    plt.figure(figsize=(6, 4))
    plt.boxplot([oof.loc[oof.placement == k, c] for k in ks], tick_labels=ks, showfliers=False)

    plt.axvline(4.5, color="r", ls="--", lw=.8)
    plt.xlabel("Colocação final"); plt.ylabel("P(Top 4) prevista — cenário A")
    plt.savefig(F / "prob_por_colocacao.png"); plt.close()


def fig_blocks_AB(G, F):
    if G is None: return
    W = G.pivot(index="block", columns="sc", values="share").fillna(0).sort_values(G.sc.iloc[0])
    W.index = W.index.map(blk)
    W.plot.barh(figsize=(7, 4)); plt.xlabel("Participação na importância SHAP")
    plt.legend(title="Cenário"); plt.savefig(F / "blocos_A_vs_B.png"); plt.close()


def fig_permutation(P, F):   # [v6]
    if P is None: return
    W = P.pivot(index="block", columns="sc", values="auc_drop").fillna(0).sort_values(P.sc.iloc[0])
    W.index = W.index.map(blk)
    W.plot.barh(figsize=(7, 4)); plt.xlabel("Queda de AUC ao permutar o bloco")
    plt.legend(title="Cenário"); plt.savefig(F / "permutacao_blocos.png"); plt.close()


def fig_archetypes(a, F):   # [v6]
    if a is None or "top4_lo" not in a: return
    a = a.sort_values("top4_rate")
    nm = a["archetype_name"].str.replace("archetype_", "", regex=False)
    plt.figure(figsize=(7, 0.4 * len(a) + 1))
    plt.errorbar(a["top4_rate"], range(len(a)),
                 xerr=[a["top4_rate"] - a["top4_lo"], a["top4_hi"] - a["top4_rate"]],
                 fmt="o", capsize=3)
    plt.axvline(0.5, color="r", ls="--", lw=.8)
    plt.yticks(range(len(a)), nm, fontsize=7); plt.xlabel("Taxa de Top 4 (IC 95%)")
    plt.grid(axis="x", alpha=.3); plt.savefig(F / "arquetipos_top4.png"); plt.close()


def copy_shap_figs(S, F, model):
    for sc in ("A", "B"):
        d = S / f"{model}_{sc}"
        if not d.exists(): continue
        for f in SHAP_FIGS + [p.name for p in d.glob("dep_trait_*.png")]:   # [v6] traits por tier
            if (d / f).exists():
                shutil.copy(d / f, F / f"shap_{sc}_{f}")


# ================= Markdown =================
def write_md(out, fl, cfg, met, hyp, rob, arch, model):
    best = met.iloc[0]
    g = lambda k: met.set_index("model").loc[k] if k in set(met.model) else None
    A, B, bl = g(f"{model}_A"), g(f"{model}_B"), g("baseline_level")
    L = ["# Resultados — Previsão de Top 4 em TFT\n",
         f"**Base:** {fmt_n(fl['n_matches'])} partidas, {fmt_n(fl['n_rows'])} observações, "
         f"{fl['n_features']} atributos.\n"]
    if cfg.get("min_level"):
        L.append(f"**Recorte:** level ≥ {cfg['min_level']} → {fmt_n(cfg['n_rows'])} observações "
                 f"(Top 4 = {cfg['top4_rate']:.3f}).\n")
    if cfg.get("drop_agg") or cfg.get("drop_blocks"):
        L.append(f"**Removidos:** features {cfg.get('drop_agg') or '—'} | "
                 f"blocos {cfg.get('drop_blocks') or '—'}.\n")
    L += ["## Desempenho\n",
          f"- Melhor modelo: **{label(best.model)}**, AUC {ci(best.auc, best.auc_lo, best.auc_hi)}."]
    if A is not None and bl is not None:
        L.append(f"- {label(model + '_A')} (só composição): AUC {A.auc:.3f} vs baseline de nível {bl.auc:.3f}.")
    if A is not None and B is not None:
        L.append(f"- Incluir o nível (B) altera a AUC em {B.auc - A.auc:+.3f}.")
    L.append("\n## Hipóteses\n")
    for r in hyp.to_dict("records"):   # [v6] corrige AttributeError
        L.append(f"- {r['Hipótese']} [{r['Cenário']}]: diferença {r[HYP_COL]} → **{r['Sustentada']}**")
    if arch is not None:
        top = arch.sort_values("top4_rate", ascending=False)
        L += ["\n## Arquétipos\n",
              f"- Mais forte: **{top.iloc[0].archetype_name}** (Top 4 = {top.iloc[0].top4_rate:.3f})",
              f"- Mais fraco: **{top.iloc[-1].archetype_name}** (Top 4 = {top.iloc[-1].top4_rate:.3f})"]
    if len(rob):
        L += ["\n## Robustez\n", rob.to_markdown(index=False)]
    L += ["\n## Arquivos\n", "- Tabelas LaTeX: `tables/` (use `\\input{tables/metricas.tex}`)",
          "- Figuras: `figures/`"]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


def main(a):
    inp, R = Path(a.inp), Path(a.res)
    S, out = R / "shap", Path(a.out)
    T, F = out / "tables", out / "figures"
    T.mkdir(parents=True, exist_ok=True); F.mkdir(parents=True, exist_ok=True)

    cfg_p = R / "run_config.json"   # [v6]
    cfg = json.load(open(cfg_p)) if cfg_p.exists() else {}
    met = pd.read_csv(R / "metrics.csv").sort_values("auc", ascending=False)
    comp = pd.read_csv(R / "comparisons.csv")
    oof = pd.read_parquet(R / "oof_predictions.parquet")

    fl = tab_data(inp, T, cfg)
    tab_metrics(met, T)
    tab_comparisons(comp, T, a.model)
    hyp = tab_hypotheses(S, T, a.model)
    tab_h1b(S, T, a.model)
    G = tab_grouped(S, T, a.model)
    P = tab_permutation(S, T, a.model)
    tab_interactions(S, T, a.model)
    tab_top_features(S, T, a.model)
    arch = tab_archetypes(Path(a.arch), T)
    rob = tab_robustness(S, R, T, a.model)

    fig_roc_calib(oof, F); fig_auc_forest(met, F)
    fig_placement(oof, F, a.model); fig_blocks_AB(G, F)
    fig_permutation(P, F); fig_archetypes(arch, F)
    copy_shap_figs(S, F, a.model)

    write_md(out, fl, cfg, met, hyp, rob, arch, a.model)
    log.info(f"Relatório gerado em {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed")
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="report")
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--arch", default="report/archetypes_report.csv")   # [v6] saída do 02b
    main(ap.parse_args())
