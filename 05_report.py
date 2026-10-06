"""
05_report.py v10 — Tabelas LaTeX, figuras e resumo. Compatível com 03 v9 e 04 v9.
Substitui 05/06/07 antigos.

Entradas:
  {res}/metrics.csv, comparisons.csv, folds.csv, run_config.json, test_predictions.parquet
  {res}/shap/{spec}/importance.csv, grouped.csv, hypotheses_descriptive.csv,
        permutation_blocks.csv, interactions_by_block.csv, stability.csv, errors_top100.csv
  --sens results_lvl8 (opcional) | --logreg results_logreg (opcional)
Uso:
  python 05_report.py --res results --sens results_lvl8 --out report
"""
import argparse, importlib, json, logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("report")
train = importlib.import_module("03_train")
plt.rcParams.update({"font.size": 10, "figure.dpi": 200, "savefig.bbox": "tight"})
CI = "IC 95%"

NAMES = {"base_level": "Baseline: nível", "base_board_value": "Baseline: valor do tabuleiro",
         "agg_lgbm": "LGBM: agregados", "agg+sparse_lgbm": "LGBM: agregados + esparsos",
         "lgbm_A": "LGBM: Cenário A", "lgbm_B": "LGBM: Cenário B", "lgbm_C": "LGBM: Cenário C (sem volume)",
         "rf_A": "Random Forest: Cenário A",
         "logreg_A": "Logística L1: Cenário A",
         "abl_agg_trait": "Agregados + traits", "abl_agg_unit": "Agregados + unidades",
         "abl_agg_item": "Agregados + itens"}
BLOCKS = {"trait": "Traits", "unit": "Unidades", "item": "Itens", "unit_item": "Unidade×Item",
          "aggregate": "Agregados", "level": "Nível", "archetype": "Arquétipos",
          "intensity": "Intensidade", "other": "Outros"}
HYP = {"H1": "H1 (compl.): traits vs unidades",
       "H1b": "H1: traits ouro/prismáticas vs nº de traits ativas",
       "H2": "H2: 4-5 custos 2★ vs custo total",
       "H3": "H3: itens nos carregadores vs total de itens"}
INCR = ["abl_agg_trait", "abl_agg_unit", "abl_agg_item"]


# ---------------- utilitários ----------------
def lab(k): return NAMES.get(k, k)
def blk(b): return BLOCKS.get(b, b)
def ci(v, lo, hi, d=4, s=False):
    f = f"{{:{'+' if s else ''}.{d}f}}"
    return f"{f.format(v)} [{f.format(lo)}; {f.format(hi)}]"
def fmt_p(p): return "—" if p is None or pd.isna(p) else ("<0,001" if p < 1e-3 else f"{p:.3f}")
def read(p):
    p = Path(p)
    if not p.exists() or p.stat().st_size == 0: return None
    try:
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None


def to_tex(df, path, cap, label):
    tex = df.to_latex(index=False, escape=True, caption=cap, label=f"tab:{label}",
                      position="htbp", column_format="l" + "c" * (df.shape[1] - 1))
    path.write_text(tex.replace("\\begin{tabular}", "\\centering\n\\small\n\\begin{tabular}"),
                    encoding="utf-8")


def pcol(c, *names):
    for n in names:
        if n in c and not pd.isna(c[n]): return c[n]
    return None


def contrast(comp, a, b):
    """Linha do comparisons.csv para (a − b); inverte o sinal se estiver gravada como (b − a)."""
    r = comp[(comp.m1 == a) & (comp.m2 == b)]
    s = 1
    if r.empty:
        r = comp[(comp.m1 == b) & (comp.m2 == a)]; s = -1
    if r.empty: return None
    r = r.iloc[0]
    out = {}
    for k in ("cv", "test"):
        lo, hi = sorted([s * r[f"{k}_lo"], s * r[f"{k}_hi"]])
        out[k] = (s * r[f"{k}_diff"], lo, hi)
    out["p_cv"] = pcol(r, "cv_p_holm", "cv_p", "p_holm", "p_value")
    out["p_test"] = pcol(r, "delong_p_holm", "test_p_holm", "delong_p", "test_p")
    return out


# ---------------- tabelas de desempenho ----------------
def tab_metrics(met, T):
    m = met[~met.model.str.startswith("abl_H")].sort_values("auc", ascending=False)
    d = pd.DataFrame({
        "Modelo": m.model.map(lab),
        f"AUC [{CI}]": [ci(r.auc, r.auc_lo, r.auc_hi) for r in m.itertuples()],
        f"Log loss [{CI}]": [ci(r.logloss, r.logloss_lo, r.logloss_hi) for r in m.itertuples()],
        "Brier": m.brier.round(4), "Acc. Top 4/partida": m.match_top4_acc.round(4),
        "Spearman/partida": m.spearman_match.round(4), "NDCG@4": m.ndcg4_match.round(4)})
    to_tex(d, T / "metricas.tex", "Desempenho no patch 16.5 (out-of-time). "
           "IC por bootstrap de partidas.", "metricas")
    return m


def tab_folds(folds, T):
    if folds is None: return
    f = folds[~folds.spec.str.startswith("abl_H")]
    g = f.groupby("spec").agg(auc_m=("auc", "mean"), auc_sd=("auc", "std"),
                              n=("auc", "size"), it=("n_iter", "mean")).reset_index()
    d = pd.DataFrame({"Modelo": g.spec.map(lab), "Folds": g.n,
                      "AUC OOF (média ± dp)": [f"{r.auc_m:.4f} ± {r.auc_sd:.4f}" for r in g.itertuples()],
                      "Árvores (média)": g.it.map(lambda v: "—" if pd.isna(v) else f"{v:.0f}")})
    to_tex(d.sort_values("Modelo"), T / "folds.tex", "Validação cruzada agrupada por partida.", "folds")


def tab_comparisons(comp, T, extra=()):
    pairs = [("base_board_value", "base_level"), ("agg_lgbm", "base_board_value"),
             ("agg+sparse_lgbm", "agg_lgbm"), ("lgbm_A", "agg+sparse_lgbm"),
             ("lgbm_A", "lgbm_C"), ("lgbm_B", "lgbm_A"), ("lgbm_A", "rf_A"), ("lgbm_A", "logreg_A")]
    pairs += [(i, "agg_lgbm") for i in INCR]
    rows = []
    for a, b in pairs:
        c = contrast(comp, a, b)
        if c is None: continue
        rows.append({"Comparação": f"{lab(a)} − {lab(b)}",
                     f"ΔAUC CV [{CI}]": ci(*c["cv"], s=True),
                     f"ΔAUC 16.5 [{CI}]": ci(*c["test"], s=True),
                     "p (Holm)": fmt_p(c["p_test"] if c["p_test"] is not None else c["p_cv"])})
    rows += list(extra)
    if rows:
        to_tex(pd.DataFrame(rows), T / "comparacoes.tex",
               "Diferenças pareadas de AUC: validação cruzada (Nadeau-Bengio) e 16.5 (bootstrap).",
               "comparacoes")


def tab_hypotheses(comp, S, T, spec="lgbm_A", tag="", cap=""):
    hd = read(S / spec / "hypotheses_descriptive.csv") if S else None
    rows, md = [], []
    for h, txt in HYP.items():
        c = contrast(comp, f"abl_{h}_noR", f"abl_{h}_noQ")
        if c is None:
            log.warning(f"{h}: contraste de ablação ausente"); continue
        d, lo, hi = c["test"]
        concl = ("Q > R" if lo > 0 else "R > Q" if hi < 0 else "Sem diferença")
        r = {"Hipótese": txt, f"ΔAUC CV [{CI}]": ci(*c["cv"], s=True),
             f"ΔAUC 16.5 [{CI}]": ci(d, lo, hi, s=True), "Conclusão": concl}
        if hd is not None:
            x = hd[hd.hypothesis.str.startswith(h + "_")]
            if len(x):
                x = x.iloc[0]
                r["|SHAP| Q / R"] = f"{x.abs_Q_test:.3f} / {x.abs_R_test:.3f}"
                r["Nº feats Q/R"] = f"{int(x.n_feat_Q_test)}/{int(x.n_feat_R_test)}"
        rows.append(r); md.append(f"- {txt}: ΔAUC 16.5 {ci(d, lo, hi, s=True)} → **{concl}**")
    if rows:
        to_tex(pd.DataFrame(rows), T / f"hipoteses{tag}.tex",
               f"Hipóteses via ablação{cap}: ΔAUC = AUC(sem R) − AUC(sem Q). "
               "Valor positivo indica que o conjunto Q carrega mais informação única. "
               "|SHAP| é apenas descritivo.", f"hipoteses{tag}")
    return md


# ---------------- tabelas SHAP ----------------
def tab_shap(S, T, specs):
    G = {}
    for sp in specs:
        d = S / sp
        imp = read(d / "importance.csv")
        if imp is None: continue
        t = imp.head(15).copy()
        t = pd.DataFrame({"#": t["rank"], "Atributo": t.feature, "Bloco": t.block.map(blk),
                          "Média |SHAP|": t.mean_abs_shap.round(4),
                          "Direção (corr)": t.direction_corr.round(2),
                          "Freq. ≠ 0": t.freq_nonzero.map("{:.1%}".format)})
        to_tex(t, T / f"top_features_{sp}.tex", f"Top 15 atributos (SHAP, 16.5) — {lab(sp)}.", f"top{sp}")
        g = read(d / "grouped.csv")
        if g is not None:
            G[sp] = g
            has = "lo" in g
            txt = [f"{r.share_test:.1%}" + (f" [{r.lo:.1%}; {r.hi:.1%}]" if has and not pd.isna(r.lo) else "")
                   for r in g.itertuples()]
            to_tex(pd.DataFrame({"Bloco": g.block.map(blk), "Participação 16.5 [IC entre folds]": txt}),
                   T / f"blocos_{sp}.tex", f"Participação dos blocos em |SHAP| — {lab(sp)}.", f"blk{sp}")
        p = read(d / "permutation_blocks.csv")
        if p is not None:
            to_tex(pd.DataFrame({"Bloco": p.block.map(blk), "Nº feats": p.n_features,
                                 "Queda de AUC": [f"{r.auc_drop:.4f} ± {r.sd:.4f}" for r in p.itertuples()]}),
                   T / f"permutacao_{sp}.tex", f"Permutação conjunta por bloco — {lab(sp)}.", f"perm{sp}")
        ib = read(d / "interactions_by_block.csv")
        if ib is not None:
            ib = ib.head(10)
            to_tex(pd.DataFrame({"Par de blocos": ib.iloc[:, 0], "Soma": ib["sum"].round(4),
                                 "Nº pares": ib["count"]}),
                   T / f"interacoes_{sp}.tex", f"Interações SHAP por par de blocos — {lab(sp)}.", f"int{sp}")
    return G


def tab_robust(S, R, sens, T, specs, lr, mid=None, extra=()):
    rows = []
    for sp in specs:
        st = read(S / sp / "stability.csv")
        if st is not None:
            rows.append({"Análise": f"Estabilidade SHAP entre folds ({lab(sp)})",
                         "Resultado": f"Spearman {st.spearman.mean():.3f}; Jaccard@20 {st.jaccard_top20.mean():.3f}"})
    for name, d in (("Level ≥ 8", sens), ("Logística", lr), ("Colocações 3–6", mid)):
        if d is None: continue
        m = read(Path(d) / "metrics.csv")
        if m is None: continue
        for r in m.itertuples():
            if r.model.startswith("abl_"): continue
            rows.append({"Análise": f"{name}: {lab(r.model)}",
                         "Resultado": f"AUC {ci(r.auc, r.auc_lo, r.auc_hi)}"})
    rows += list(extra)
    if rows:
        to_tex(pd.DataFrame(rows), T / "robustez.tex", "Robustez e sensibilidade.", "robustez")
    return rows


# ---------------- análise de erros ----------------
def load_preds(R):
    p = read(R / "test_predictions.parquet")
    if p is None: return None
    if "spec" in p.columns and "p" in p.columns:   # formato longo
        idx = [c for c in ("match_id", "puuid", "placement", "top4", "level") if c in p.columns]
        p = p.pivot_table(index=idx, columns="spec", values="p").add_prefix("p_").reset_index()
    return p


def errors(P, T, F, spec="lgbm_A"):
    c = f"p_{spec}"
    if P is None or c not in P or "placement" not in P:
        log.warning("test_predictions sem colunas esperadas: análise de erros pulada"); return []
    y, p = P.top4.values, P[c].values
    g = P.assign(pred=(p > .5).astype(int)).groupby("placement").agg(
        n=("top4", "size"), p_mean=(c, "mean"), acerto=("pred", lambda s: 0))
    g["acerto"] = [((P.placement == k) & ((P[c] > .5) == (P.top4 == 1))).sum() / n
                   for k, n in zip(g.index, g.n)]
    to_tex(g.reset_index().rename(columns={"placement": "Colocação", "n": "n",
                                           "p_mean": "P(Top 4) média", "acerto": "Acerto"}).round(3),
           T / "erros_colocacao.tex", f"Previsão por colocação final — {lab(spec)} (16.5).", "errcol")
    mid = P.placement.between(3, 6).values
    md = [f"- AUC geral {roc_auc_score(y, p):.4f} | só colocações 3–6: "
          f"{roc_auc_score(y[mid], p[mid]):.4f} (fronteira do Top 4)"]
    if "level" in P:
        rows = [{"Nível": int(lv), "n": int(m.sum()), "AUC": roc_auc_score(y[m], p[m]),
                 "P média": p[m].mean(), "Top 4 obs.": y[m].mean()}
                for lv in sorted(P.level.unique()) if (m := (P.level == lv).values).sum() > 500
                and 0 < y[m].mean() < 1]
        to_tex(pd.DataFrame(rows).round(4), T / "erros_nivel.tex",
               "Discriminação e calibração por nível do jogador (16.5).", "errlvl")
    plt.figure(figsize=(6, 4))
    ks = sorted(P.placement.unique())
    plt.boxplot([P.loc[P.placement == k, c] for k in ks], tick_labels=ks, showfliers=False)
    plt.axvline(4.5, color="r", ls="--", lw=.8)
    plt.xlabel("Colocação final"); plt.ylabel("P(Top 4) prevista")
    plt.savefig(F / "prob_por_colocacao.png"); plt.close()
    return md


# ---------------- lacunas do parecer ----------------
def ece(y, p, b=10):
    q = np.minimum((p * b).astype(int), b - 1)
    return float(sum(abs(y[q == i].mean() - p[q == i].mean()) * (q == i).mean()
                     for i in range(b) if (q == i).any()))


def logreg_vs_lgbm(R, lr, n_boot=500):
    """ΔAUC lgbm_A − logreg_A no 16.5 (DeLong + bootstrap de partidas), entre execuções."""
    if lr is None: return None
    a, b = read(R / "test_predictions.parquet"), read(Path(lr) / "test_predictions.parquet")
    if a is None or b is None or "p_lgbm_A" not in a or "p_logreg_A" not in b: return None
    k = ["match_id", "placement"]
    d = a[k + ["top4", "p_lgbm_A"]].merge(b[k + ["p_logreg_A"]], on=k)
    y, g = d.top4.to_numpy(), d.match_id.to_numpy()
    p1, p2 = train.clip(d.p_lgbm_A), train.clip(d.p_logreg_A)
    _, pv = train.delong_test(train.delong_components(y, p1), train.delong_components(y, p2))
    bt = train.bootstrap(y, {"a": p1, "b": p2}, g, n_boot, train.SEED)
    db = bt["a"]["auc"] - bt["b"]["auc"]
    return {"Comparação": f"{lab('lgbm_A')} − {lab('logreg_A')}", f"ΔAUC CV [{CI}]": "—",
            f"ΔAUC 16.5 [{CI}]": ci(db.mean(), db.quantile(.025), db.quantile(.975), s=True),
            "p (Holm)": fmt_p(pv) + " (DeLong)"}


def tab_convergence(lr, T):
    f = read(Path(lr) / "folds.csv") if lr else None
    if f is None or "converged" not in f: return []
    bpp = Path(lr) / "best_params.json"
    bp = json.load(open(bpp)) if bpp.exists() else {}
    rows = [{"Ajuste": f"Fold {int(r.fold)}", "C": json.loads(r.params).get("C"),
             "n_iter": int(r.n_iter), "Convergiu": "sim" if r.converged else "não",
             "Coef. ≠ 0": f"{int(r.nnz_coef)}/{int(r.n_coef)}"} for r in f.itertuples()]
    for k, v in bp.items():
        if "converged" in v:
            rows.append({"Ajuste": f"Final ({k})", "C": v["params"].get("C"), "n_iter": v["n_iter"],
                         "Convergiu": "sim" if v["converged"] else "não",
                         "Coef. ≠ 0": f"{v['nnz_coef']}/{v['n_coef']}"})
    d = pd.DataFrame(rows); d["C"] = d["C"].map(lambda c: f"{c:.2e}")
    to_tex(d, T / "logreg_convergencia.tex", "Convergência da logística L1 (SAGA).", "logconv")
    return [f"- Logística: {(d.Convergiu == 'sim').sum()}/{len(d)} ajustes convergiram "
            f"(n_iter máx. {d.n_iter.max()})"]


def tab_hparams(dirs, T):
    rows = []
    for name, d in dirs:
        p = Path(d) / "best_params.json" if d else None
        if p is None or not p.exists(): continue
        for k, v in json.load(open(p)).items():
            if k.startswith("abl_") or not v.get("params"): continue
            pr = "; ".join(f"{x}={y:.4g}" if isinstance(y, float) else f"{x}={y}"
                           for x, y in v["params"].items())
            rows.append({"Execução": name, "Modelo": lab(k), "Hiperparâmetros": pr,
                         "Árvores/iter.": v.get("n_iter", v.get("n_trees", "—")),
                         "Nº feats": v.get("n_features")})
    if rows:
        to_tex(pd.DataFrame(rows), T / "hiperparametros.tex",
               "Hiperparâmetros finais (tuning só no treino 16.1–16.4). "
               "Ablações herdam os da base; baselines logísticos não têm tuning.", "hparams")


def errors_groups(R, inp, T):
    """AUC e calibração (ECE) por patch e por arquétipo: OOF 16.1–16.4 e 16.5."""
    import pyarrow.parquet as pq
    fp = Path(inp) / "features.parquet"
    if not fp.exists(): return
    cols = [c for c in pq.ParquetFile(fp).schema.names if c.startswith("archetype_")]
    if not cols: return
    Fe = pd.read_parquet(fp, columns=["split"] + cols)
    arch = np.array([c.replace("archetype_", "") for c in cols])[Fe[cols].to_numpy().argmax(1)]
    out = []
    for split, f in (("train", "oof_predictions.parquet"), ("test", "test_predictions.parquet")):
        P, m = read(R / f), (Fe.split.astype(str) == split).to_numpy()
        if P is None or "p_lgbm_A" not in P or len(P) != m.sum(): continue
        P = P.assign(archetype=arch[m], patch=P.patch.astype(str))
        for by in ("patch", "archetype"):
            for k, s in P.groupby(by):
                yv, pv = s.top4.to_numpy(), s.p_lgbm_A.to_numpy()
                if len(s) < 500 or yv.min() == yv.max(): continue
                out.append({"by": by, "Origem": "OOF 16.1–16.4" if split == "train" else "16.5",
                            "Grupo": k, "n": len(s), "AUC": roc_auc_score(yv, pv),
                            "P média": pv.mean(), "Top 4 obs.": yv.mean(), "ECE": ece(yv, pv)})
    D = pd.DataFrame(out)
    for by, fn, txt in (("patch", "erros_patch", "patch"), ("archetype", "erros_arquetipo", "arquétipo")):
        x = D[D.by == by].drop(columns="by").round(4) if len(D) else D
        if len(x):
            to_tex(x, T / f"{fn}.tex", f"Discriminação e calibração por {txt} — LGBM: Cenário A.", fn)


def audit_suspects(interim, T):
    """Parecer 1.1: EmptyBag/ThiefsGloves × colocação × última rodada (leak_last_round)."""
    it, pp = Path(interim) / "items.parquet", Path(interim) / "participants.parquet"
    if not it.exists() or not pp.exists():
        log.warning("dados interim ausentes: auditoria pulada"); return []
    k = ["match_id", "pidx"]
    I = pd.read_parquet(it, columns=k + ["item_category"], filters=[("is_suspect", "==", 1)])
    P = pd.read_parquet(pp, columns=k + ["placement", "top4", "leak_last_round"])
    idx, tags = P.set_index(k).index, {"empty_bag": "EmptyBag", "thiefs_gloves": "ThiefsGloves"}
    for t in tags:
        P[t] = idx.isin(I.loc[I.item_category.astype(str) == t].set_index(k).index).astype(int)
    g = P.groupby("placement")
    d = pd.DataFrame({"Colocação": g.size().index, "Última rodada (média)": g.leak_last_round.mean().values})
    for t, n in tags.items():
        d[f"% {n}"] = (g[t].mean() * 100).values
        d[f"corr({n}, rodada)"] = g.apply(lambda s: s[t].corr(s.leak_last_round)).values
    to_tex(d.round(3), T / "auditoria_colocacao.tex",
           "Itens suspeitos por colocação e correlação com a última rodada dentro da colocação.", "audcol")
    P["Quintil da última rodada"] = pd.qcut(P.leak_last_round, 5, duplicates="drop").astype(str)
    q = P.groupby("Quintil da última rodada")[list(tags)].mean().mul(100).rename(
        columns={t: f"% {n}" for t, n in tags.items()}).reset_index()
    to_tex(q.round(3), T / "auditoria_rodada.tex", "Itens suspeitos por quintil da última rodada.", "audrod")
    md = []
    for t, n in tags.items():
        r = g[t].mean()
        md.append(f"- {n}: {r.loc[1]:.2%} no 1º contra {P.loc[P.placement >= 5, t].mean():.2%} "
                  f"no 5º–8º | AUC isolada {roc_auc_score(P.top4, P[t]):.3f} | "
                  f"corr com a última rodada {P[t].corr(P.leak_last_round):.3f}")
    return md


def lambdamart_rows(R):
    p = R / "lambdamart.json"
    if not p.exists(): return []
    j = json.load(open(p))
    return [{"Análise": "LambdaMART (Cenário A) − LGBM: Cenário A, 16.5",
             "Resultado": f"AUC {j['auc']:.4f} (Δ {j['delta_auc_vs_lgbm_A']:+.4f}, DeLong p = "
                          f"{fmt_p(j['delong_p'])}); Spearman {j['spearman_match']:.4f} vs "
                          f"{j['lgbm_A_spearman']:.4f}; NDCG@4 {j['ndcg4_match']:.4f} vs "
                          f"{j['lgbm_A_ndcg4']:.4f}"}]


# ---------------- figuras ----------------
def figs(met, P, G, F):
    m = met.sort_values("auc")
    plt.figure(figsize=(6, .4 * len(m) + 1))
    plt.errorbar(m.auc, range(len(m)), xerr=[m.auc - m.auc_lo, m.auc_hi - m.auc], fmt="o", capsize=3)
    plt.yticks(range(len(m)), m.model.map(lab)); plt.xlabel("AUC no 16.5 (IC 95%)")
    plt.grid(axis="x", alpha=.3); plt.savefig(F / "auc_ic.png"); plt.close()
    if P is not None:
        cols = [c for c in P.columns if c.startswith("p_") and not c.startswith("p_abl_H")]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        for c in cols:
            fpr, tpr, _ = roc_curve(P.top4, P[c]); ax[0].plot(fpr, tpr, lw=1.1, label=lab(c[2:]))
            pt, pp = calibration_curve(P.top4, P[c], n_bins=15, strategy="quantile")
            ax[1].plot(pp, pt, marker="o", ms=3, lw=1, label=lab(c[2:]))
        for a in ax: a.plot([0, 1], [0, 1], "k--", lw=.8)
        ax[0].set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos", title="ROC")
        ax[1].set(xlabel="Probabilidade prevista", ylabel="Frequência observada", title="Calibração")
        ax[1].legend(fontsize=6); plt.savefig(F / "roc_calibracao.png"); plt.close()
    if G:
        W = pd.concat({sp: g.set_index("block").share_test for sp, g in G.items()}, axis=1).fillna(0)
        W.index = W.index.map(blk); W.columns = [lab(c) for c in W.columns]
        W.sort_values(W.columns[0]).plot.barh(figsize=(7, 4))
        plt.xlabel("Participação em |SHAP| (16.5)"); plt.savefig(F / "blocos.png"); plt.close()


# ---------------- main ----------------
def main(a):
    R, out = Path(a.res), Path(a.out)
    S, T, F = R / "shap", out / "tables", out / "figures"
    T.mkdir(parents=True, exist_ok=True); F.mkdir(parents=True, exist_ok=True)
    met, comp = read(R / "metrics.csv"), read(R / "comparisons.csv")
    folds, cfg = read(R / "folds.csv"), json.load(open(R / "run_config.json"))
    if a.logreg:
        for f in ("metrics.csv", "comparisons.csv"):
            x = read(Path(a.logreg) / f)
            if x is not None:
                if f == "metrics.csv": met = pd.concat([met, x]).drop_duplicates("model", keep="last")
                else: comp = pd.concat([comp, x])
    specs = [s for s in a.specs.split(",") if (S / s).exists()]

    lv = logreg_vs_lgbm(R, a.logreg)
    m = tab_metrics(met, T); tab_folds(folds, T); tab_comparisons(comp, T, [lv] if lv else [])
    hyp_md = tab_hypotheses(comp, S, T)
    c8 = read(Path(a.sens) / "comparisons.csv") if a.sens else None
    if c8 is not None:
        hyp_md += ["\n**Level ≥ 8:**", *tab_hypotheses(c8, None, T, tag="_lvl8", cap=" (level ≥ 8)")]
    G = tab_shap(S, T, specs)
    conv_md = tab_convergence(a.logreg, T)
    tab_hparams([("Principal", R), ("Logística", a.logreg), ("Level ≥ 8", a.sens),
                 ("Colocações 3–6", a.mid)], T)
    rob = tab_robust(S, R, a.sens, T, specs, a.logreg, a.mid, lambdamart_rows(R))
    P = load_preds(R); err_md = errors(P, T, F); errors_groups(R, a.inp, T)
    aud_md = audit_suspects(a.interim, T)
    figs(m, P, G, F)

    best = m.iloc[0]
    L = ["# Resultados — Previsão de Top 4 em TFT (16.5 out-of-time)\n",
         f"**Configuração:** {json.dumps(cfg.get('args', {}), ensure_ascii=False)}\n",
         "## Desempenho\n",
         f"- Melhor: **{lab(best.model)}**, AUC {ci(best.auc, best.auc_lo, best.auc_hi)}."]
    for x, y in (("agg+sparse_lgbm", "agg_lgbm"), ("lgbm_A", "lgbm_C"), ("lgbm_B", "lgbm_A"),
                 *[(i, "agg_lgbm") for i in INCR]):
        c = contrast(comp, x, y)
        if c: L.append(f"- {lab(x)} − {lab(y)}: ΔAUC {ci(*c['test'], s=True)}")
    L += ["\n## Hipóteses (ablação)\n", *hyp_md, "\n## Erros\n", *err_md,
          "\n## Auditoria EmptyBag/ThiefsGloves\n", *aud_md, "\n## Convergência\n", *conv_md]
    if rob: L += ["\n## Robustez\n", *[f"- {r['Análise']}: {r['Resultado']}" for r in rob]]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    log.info(f"Relatório em {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="report")
    ap.add_argument("--specs", default="lgbm_A,lgbm_C")
    ap.add_argument("--sens", default="results_lvl8")
    ap.add_argument("--logreg", default="")
    ap.add_argument("--mid", default="results_mid")
    ap.add_argument("--inp", default="data/processed")
    ap.add_argument("--interim", default="data/interim")
    a = ap.parse_args()
    a.mid = a.mid if a.mid and Path(a.mid).exists() else None
    a.sens = a.sens if a.sens and Path(a.sens).exists() else None
    a.logreg = a.logreg if a.logreg and Path(a.logreg).exists() else None
    main(a)
