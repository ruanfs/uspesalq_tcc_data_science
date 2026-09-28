"""
07_compare_AB.py — Comparação sistemática Cenário A (só composição) vs Cenário B (+level).
Não retreina: lê as saídas do 03/04 das três rodadas.
Convenção: diferenças sempre B − A.
Uso: python 07_compare_AB.py --model lgbm --out report
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RUNS = {"Principal": "results", "Level ≥ 8": "results_lvl8", "Sem unidades": "results_no_units"}
FAM = {"logreg": "Reg. Logística L1", "rf": "Random Forest", "lgbm": "LightGBM"}
BLOCKS = {"trait": "Traits", "unit": "Unidades", "item": "Itens", "unit_item": "Unidade×Item",
          "aggregate": "Agregados", "level": "Nível", "patch": "Patch", "archetype": "Arquétipos"}
HYP = {"H1_traits_vs_units": "H1 compl.: traits > unidades",
       "H1b_high_tier": "H1: traits ouro+ > nº traits ativas",
       "H2_quality_vs_quantity": "H2: 4-5 custos 2★ > custo total",
       "H3_concentration_vs_total": "H3: itens em carregadores > total"}
CI = "IC 95%"


def read(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def fmt_p(p): return "—" if pd.isna(p) else ("<0,001" if p < 1e-3 else f"{p:.3f}")


def ci(v, lo, hi, d=4): return f"{v:+.{d}f} [{lo:+.{d}f}; {hi:+.{d}f}]"


def to_tex(df, path, cap, lab):
    tex = df.to_latex(index=False, escape=True, caption=cap, label=f"tab:{lab}",
                      position="htbp", column_format="l" + "c" * (df.shape[1] - 1))
    path.write_text(tex.replace("\\begin{tabular}", "\\centering\n\\small\n\\begin{tabular}"),
                    encoding="utf-8")


def pair(comp, a, b, metric):
    """Diferença B − A a partir do comparisons.csv (lá diff = m1 − m2)."""
    c = comp[(comp.metric == metric) & (((comp.m1 == a) & (comp.m2 == b)) |
                                        ((comp.m1 == b) & (comp.m2 == a)))]
    if c.empty: return None
    r = c.iloc[0]; s = 1 if r.m1 == b else -1
    lo, hi = sorted([s * r.lo, s * r.hi])
    return {"diff": s * r["diff"], "lo": lo, "hi": hi,
            "p": r.get("p_holm", r.p_value), "p_dl": r.get("delong_p_holm", r.get("delong_p", np.nan))}


# ---------------- 1) Desempenho ----------------
def perf(runs):
    rows = []
    for run, d in runs.items():
        met, comp = read(d / "metrics.csv"), read(d / "comparisons.csv")
        if met is None or comp is None: continue
        met = met.set_index("model")
        for f in FAM:
            a, b = f"{f}_A", f"{f}_B"
            if a not in met.index or b not in met.index: continue
            au, ll = pair(comp, a, b, "auc"), pair(comp, a, b, "logloss")
            if au is None or ll is None: continue
            rows.append({"run": run, "fam": f, "auc_A": met.loc[a, "auc"], "auc_B": met.loc[b, "auc"],
                         "d_auc": au["diff"], "d_auc_lo": au["lo"], "d_auc_hi": au["hi"],
                         "p_auc": au["p"], "p_delong": au["p_dl"],
                         "d_ll": ll["diff"], "d_ll_lo": ll["lo"], "d_ll_hi": ll["hi"], "p_ll": ll["p"],
                         "d_brier": met.loc[b, "brier"] - met.loc[a, "brier"],
                         "d_acc": met.loc[b, "match_top4_acc"] - met.loc[a, "match_top4_acc"]})
    return pd.DataFrame(rows)


def temporal(runs):
    rows = []
    for run, d in runs.items():
        tp = read(d / "temporal.csv")
        if tp is None: continue
        tp = tp.set_index("model")
        for f in FAM:
            a, b = f"{f}_A", f"{f}_B"
            if a in tp.index and b in tp.index:
                rows.append({"Rodada": run, "Modelo": FAM[f], "Patch teste": tp.loc[a, "test_patch"],
                             "AUC A": f"{tp.loc[a, 'auc']:.3f}", "AUC B": f"{tp.loc[b, 'auc']:.3f}",
                             "ΔAUC (B−A)": f"{tp.loc[b, 'auc'] - tp.loc[a, 'auc']:+.4f}",
                             "ΔLog loss (B−A)": f"{tp.loc[b, 'logloss'] - tp.loc[a, 'logloss']:+.4f}"})
    return pd.DataFrame(rows)


# ---------------- 2) SHAP ----------------
def shap_ab(runs, model):
    rows, rank_tab, blk_tab, sc_data = [], None, None, None
    for run, d in runs.items():
        S = d / "shap"
        ia, ib = read(S / f"{model}_A" / "importance.csv"), read(S / f"{model}_B" / "importance.csv")
        if ia is None or ib is None: continue
        a = ia.set_index("feature")["mean_abs_shap"]
        bfull = ib.set_index("feature")["mean_abs_shap"]
        b = bfull.drop("level", errors="ignore")
        com = a.index.intersection(b.index)
        ta, tb = set(a[com].nlargest(20).index), set(b[com].nlargest(20).index)
        rows.append({"Rodada": run,
                     "Spearman (ranking A vs B)": f"{spearmanr(a[com], b[com]).statistic:.3f}",
                     "Jaccard@20": f"{len(ta & tb) / len(ta | tb):.3f}",
                     "Participação do level em B": f"{bfull.get('level', 0) / bfull.sum():.1%}",
                     "Importância da composição retida em B": f"{b[com].sum() / a[com].sum():.1%}"})
        if run == "Principal":
            ra = a.rank(ascending=False).astype(int); rb = b.rank(ascending=False).astype(int)
            top = a.nlargest(15).index
            rank_tab = pd.DataFrame({"Atributo": top, "Rank A": ra[top].values,
                                     "Rank B": rb.reindex(top).values,
                                     "|SHAP| A": a[top].round(4).values,
                                     "|SHAP| B": b.reindex(top).round(4).values})
            rank_tab["Δ rank"] = rank_tab["Rank A"] - rank_tab["Rank B"]
            sc_data = (a[com], b[com])
            ga, gb = read(S / f"{model}_A" / "grouped.csv"), read(S / f"{model}_B" / "grouped.csv")
            if ga is not None and gb is not None:
                m = ga.set_index("block")[["share"]].join(gb.set_index("block")[["share"]],
                                                          how="outer", lsuffix="_A", rsuffix="_B").fillna(0)
                m["delta"] = m.share_B - m.share_A
                blk_tab = pd.DataFrame({"Bloco": m.index.map(lambda x: BLOCKS.get(x, x)),
                                        "Participação A": m.share_A.map("{:.1%}".format),
                                        "Participação B": m.share_B.map("{:.1%}".format),
                                        "Δ (p.p.)": (m.delta * 100).map("{:+.1f}".format)}
                                       ).sort_values("Participação A", ascending=False)
    return pd.DataFrame(rows), rank_tab, blk_tab, sc_data


def hyp_ab(runs, model):
    rows = []
    for run, d in runs.items():
        ha = read(d / "shap" / f"{model}_A" / "hypotheses.csv")
        hb = read(d / "shap" / f"{model}_B" / "hypotheses.csv")
        if ha is None or hb is None: continue
        m = ha.set_index("hypothesis").join(hb.set_index("hypothesis"), lsuffix="_A", rsuffix="_B")
        for h, r in m.iterrows():
            sa, sb = bool(r.supported_A), bool(r.supported_B)
            rows.append({"Rodada": run, "Hipótese": HYP.get(h, h),
                         f"Dif. A [{CI}]": ci(r.diff_A, r.lo_A, r.hi_A, 3),
                         f"Dif. B [{CI}]": ci(r.diff_B, r.lo_B, r.hi_B, 3),
                         "Conclusão": "Sustentada em A e B" if sa and sb else
                                      "Só em A" if sa else "Só em B" if sb else "Não sustentada"})
    return pd.DataFrame(rows)


# ---------------- 3) Figura ----------------
def fig(P, sc_data, F):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    if len(P):
        P = P.iloc[::-1].reset_index(drop=True)
        ax[0].errorbar(P.d_auc, range(len(P)), xerr=[P.d_auc - P.d_auc_lo, P.d_auc_hi - P.d_auc],
                       fmt="o", capsize=3)
        ax[0].set_yticks(range(len(P)), [f"{FAM[f]} | {r}" for f, r in zip(P.fam, P.run)], fontsize=7)
        ax[0].axvline(0, color="r", ls="--", lw=.8)
        ax[0].set(xlabel="ΔAUC (B − A) [IC 95%]", title="Ganho de AUC ao incluir o level")
    if sc_data is not None:
        a, b = sc_data
        ax[1].loglog(a + 1e-6, b + 1e-6, ".", ms=3, alpha=.5)
        lim = [min(a.min(), b.min()) + 1e-6, max(a.max(), b.max()) * 1.2]
        ax[1].plot(lim, lim, "k--", lw=.8)
        for f in a.nlargest(6).index:
            ax[1].annotate(f, (a[f], b[f]), fontsize=6)
        ax[1].set(xlabel="Média |SHAP| — A", ylabel="Média |SHAP| — B",
                  title="Importância da composição: A vs B")
    plt.tight_layout(); plt.savefig(F / "ab_resumo.png", dpi=200, bbox_inches="tight"); plt.close()


def main(a):
    out = Path(a.out); T, F = out / "tables", out / "figures"
    T.mkdir(parents=True, exist_ok=True); F.mkdir(parents=True, exist_ok=True)
    runs = {k: Path(v) for k, v in RUNS.items() if Path(v).exists()}

    P = perf(runs)
    P.to_csv(out / "ab_desempenho.csv", index=False)
    if len(P):
        to_tex(pd.DataFrame({
            "Rodada": P.run, "Modelo": P.fam.map(FAM),
            "AUC A": P.auc_A.round(3), "AUC B": P.auc_B.round(3),
            f"ΔAUC [{CI}]": [ci(r.d_auc, r.d_auc_lo, r.d_auc_hi) for r in P.itertuples()],
            "p DeLong (Holm)": P.p_delong.map(fmt_p),
            f"ΔLog loss [{CI}]": [ci(r.d_ll, r.d_ll_lo, r.d_ll_hi) for r in P.itertuples()],
            "ΔAcc. Top4": P.d_acc.map(lambda v: "—" if pd.isna(v) else f"{v:+.3f}")}),
            T / "ab_desempenho.tex", "Cenário B (composição + level) vs A (só composição). "
            "Diferenças B − A; IC por bootstrap de partidas.", "abperf")

    tp = temporal(runs)
    if len(tp): to_tex(tp, T / "ab_temporal.tex", "Validação temporal: A vs B.", "abtemp")

    St, rk, bk, sc = shap_ab(runs, a.model)
    if len(St): to_tex(St, T / "ab_shap.tex", "Estabilidade da explicação SHAP entre A e B.", "abshap")
    if rk is not None: to_tex(rk, T / "ab_ranking.tex", "Top 15 atributos de A e posição em B.", "abrank")
    if bk is not None: to_tex(bk, T / "ab_blocos.tex", "Participação dos blocos na importância SHAP: A vs B.", "abblk")

    H = hyp_ab(runs, a.model)
    if len(H): to_tex(H, T / "ab_hipoteses.tex", "Hipóteses nos cenários A e B, por rodada.", "abhyp")

    fig(P, sc, F)

    L = ["# Comparação Cenário A vs B\n", "## Desempenho (B − A)\n"]
    for r in P.itertuples():
        L.append(f"- {FAM[r.fam]} | {r.run}: ΔAUC {ci(r.d_auc, r.d_auc_lo, r.d_auc_hi)} "
                 f"(DeLong Holm p={fmt_p(r.p_delong)})")
    if len(St): L += ["\n## SHAP\n", St.to_markdown(index=False)]
    if len(H): L += ["\n## Hipóteses\n", H.to_markdown(index=False)]
    (out / "compare_AB.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Comparação A vs B salva em {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--out", default="report")
    main(ap.parse_args())
