"""
04_shap.py — Interpretabilidade (SHAP) dos modelos finais.

[v6] Hipóteses H1b/H2/H3, estabilidade com n_iter do fold, recorte de level, permutação
     por bloco, dependência por tier, interações por par de blocos, waterfall 1º vs 8º.
[v7] Carga esparsa via 03_train.load (mesmas colunas/ordem do treino), SHAP nativo do
     LightGBM (pred_contrib), bootstrap vetorizado, estabilidade em esparsa com --stab-frac,
     checkpoint (--force), --scenario, --threads.
[v8] Estabilidade usa os modelos por fold salvos pelo 03 (fold_{model}_{sc}_{k}.joblib);
     fallback para retreino (CSC + rng por fold) se não existirem; checkpoint por fold.

Uso:
    python 04_shap.py --res results --models lgbm --top-inter 40
    start python 04_shap.py --scenario A --threads 8
    start python 04_shap.py --scenario B --threads 8
"""
import argparse, gc, importlib, json, logging, time, warnings
from joblib import Parallel, delayed
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("shap")
SEED = 42
rng = np.random.default_rng(SEED)
train = importlib.import_module("03_train")
THREADS = 8

HYP = {
    "H1_traits_vs_units": (["trait"], ["unit"]),
    "H1b_high_tier": (["n_traits_gold_plus", "n_traits_prismatic"], ["n_active_traits"]),
    "H2_quality_vs_quantity": (["n_hi_cost_2star"], ["sum_cost"]),
    "H3_concentration_vs_total": (["n_carries", "n_full_items_on_hi_cost"], ["n_full_items"]),
}
DEP_FEATURES = ["n_traits_gold_plus", "n_hi_cost_2star", "sum_cost", "board_value", "n_carries",
                "n_full_items_on_hi_cost", "n_full_items", "level"]


def block_map(cols, fd):
    m = dict(zip(fd["feature"], fd["block"]))
    return {c: ("patch" if c.startswith("patch_") else
                "level" if c == "level" else m.get(c, "aggregate")) for c in cols}


def tree_shap(model, X):
    """TreeSHAP nativo do LightGBM (idêntico ao shap.TreeExplainer, em log-odds)."""
    arr = X.to_numpy(np.float64) if isinstance(X, pd.DataFrame) else \
        (X.toarray() if hasattr(X, "toarray") else np.asarray(X, np.float64))
    if hasattr(model, "booster_"):
        c = np.asarray(model.booster_.predict(arr, pred_contrib=True, num_threads=THREADS))
        return c[:, :-1], float(c[0, -1])
    ex = shap.TreeExplainer(model); sv = ex.shap_values(arr); ev = ex.expected_value
    if isinstance(sv, list): sv, ev = sv[1], ev[1]
    elif np.ndim(sv) == 3: sv, ev = sv[:, :, 1], np.atleast_1d(ev)[-1]
    return np.asarray(sv), float(np.atleast_1d(ev)[-1])


def proba(model, X):
    arr = X.to_numpy(np.float32) if isinstance(X, pd.DataFrame) else X
    return model.predict_proba(arr)[:, 1]


def sample_by_match(g, n):
    mids = g.unique()
    k = min(len(mids), max(1, n // 8))
    pick = rng.choice(mids, k, replace=False)
    return np.flatnonzero(g.isin(pick).values)


def boot_ci(values_fn, gidx, n_boot=500):
    out = [values_fn(np.concatenate([gidx[i] for i in rng.integers(0, len(gidx), len(gidx))]))
           for _ in range(n_boot)]
    return np.percentile(out, [2.5, 97.5], axis=0)


def savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close("all")


def perm_block(model, X, y, bmap, n_rep=5):
    Xa = X.to_numpy(np.float32)
    base = roc_auc_score(y, proba(model, Xa)); out = []
    cols = list(X.columns)
    for b in sorted(set(bmap.values())):
        ix = [i for i, c in enumerate(cols) if bmap[c] == b]
        drops = []
        for _ in range(n_rep):
            Xp = Xa.copy()
            Xp[:, ix] = Xp[rng.permutation(len(Xp))][:, ix]
            drops.append(base - roc_auc_score(y, proba(model, Xp)))
        out.append({"block": b, "n_features": len(ix), "auc_drop": np.mean(drops),
                    "sd": np.std(drops)})
    return pd.DataFrame(out).sort_values("auc_drop", ascending=False)


def _inter_chunk(model, Xc, pos):
    """Interações SHAP de um bloco de linhas, já recortadas no top-k (roda em 1 núcleo)."""
    ex = shap.TreeExplainer(model)
    iv = ex.shap_interaction_values(Xc)
    iv = iv[1] if isinstance(iv, list) else iv
    sub = np.abs(iv[:, pos][:, :, pos]).sum(0)
    del iv; gc.collect()
    return sub, len(Xc)


def fold_model(name, sc, k, n_f, X, y, gk, tr, fold_params, a, rk):   # [v8]
    """Carrega o modelo do fold salvo pelo 03; se não existir, retreina (fallback v7)."""
    fpath = Path(a.res) / "models" / f"fold_{name}_{sc}_{k}.joblib"
    if fpath.exists():
        art = joblib.load(fpath)
        assert art["n_folds"] == n_f, f"nº de folds diferente do 03 ({art['n_folds']} vs {n_f})"
        m = art["model"]
        if name == "lgbm": m.set_params(n_jobs=THREADS)
        return m, {"src": "03", "n_train": None, "n_iter": art.get("n_iter"), "t_fit": 0.0}
    if a.stab_frac < 1:
        mt = np.unique(gk[tr])
        keep = rk.choice(mt, int(len(mt) * a.stab_frac), replace=False)
        tr = tr[np.isin(gk[tr], keep)]
    fp = fold_params[k]; prm = dict(fp["params"])
    if name == "lgbm":
        prm["n_estimators"] = fp.get("n_iter") or 500
    m = train.build(name, prm, n_rows=len(tr))
    if name == "lgbm": m.set_params(n_jobs=THREADS)
    Xtr = X[tr].tocsc() if name == "lgbm" else X[tr]    # CSC: binning por coluna mais rápido
    tb = time.time(); m.fit(Xtr, y[tr]); t_fit = time.time() - tb
    del Xtr; gc.collect()
    return m, {"src": "retreino", "n_train": len(tr), "n_iter": prm.get("n_estimators"),
               "t_fit": t_fit}


# ================= Análises =================
def analyze(name, sc, meta, scen, fd, params, a, root):
    t0 = time.time()
    out = root / f"{name}_{sc}"; out.mkdir(parents=True, exist_ok=True)
    done = lambda f: (out / f).exists() and not a.force
    art = joblib.load(Path(a.res) / "models" / f"final_{name}_{sc}.joblib")
    model, cols = art["model"], list(art["features"])
    X, names = scen[sc]
    assert list(names) == cols, f"[{name}_{sc}] colunas diferentes das do treino"
    g, y = meta["match_id"], meta["top4"].to_numpy()

    idx = sample_by_match(g, a.n_shap)
    Xs = pd.DataFrame(X[idx].toarray(), columns=cols)
    gs = g.iloc[idx].reset_index(drop=True)
    sv, ev = tree_shap(model, Xs)
    bmap = block_map(cols, fd)
    gidx = [np.flatnonzero(gs.values == m) for m in gs.unique()]
    log.info(f"[{name}_{sc}] SHAP em {len(Xs)} linhas × {len(cols)} features | {time.time()-t0:.0f}s")

    # 1) Global
    imp = pd.DataFrame({"feature": cols, "block": [bmap[c] for c in cols],
                        "mean_abs_shap": np.abs(sv).mean(0),
                        "mean_shap": sv.mean(0)}).sort_values("mean_abs_shap", ascending=False)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp.to_csv(out / "importance.csv", index=False)
    if not done("beeswarm.png"):
        shap.summary_plot(sv, Xs, max_display=25, show=False); savefig(out / "beeswarm.png")
        shap.summary_plot(sv, Xs, plot_type="bar", max_display=25, show=False); savefig(out / "bar.png")

    # 2) Agrupado por bloco
    B = pd.DataFrame(sv, columns=cols).T.groupby(pd.Series(bmap)).sum().T
    if not done("grouped.csv"):
        Babs = B.abs().values
        share = lambda ix: Babs[ix].mean(0) / Babs[ix].mean(0).sum()
        lo, hi = boot_ci(share, gidx, a.n_boot)
        gr = pd.DataFrame({"block": B.columns, "mean_abs_shap": Babs.mean(0),
                           "share": share(np.arange(len(B))), "share_lo": lo, "share_hi": hi}
                          ).sort_values("share", ascending=False)
        gr.to_csv(out / "grouped.csv", index=False)
        plt.figure(figsize=(7, 4))
        plt.barh(gr["block"], gr["share"],
                 xerr=[gr["share"] - gr["share_lo"], gr["share_hi"] - gr["share"]])
        plt.gca().invert_yaxis(); plt.xlabel("Participação na importância SHAP (IC 95%)")
        savefig(out / "grouped_blocks.png")
        log.info(f"  [2] agrupado ok | {time.time()-t0:.0f}s")

    # 2b) Permutação por bloco
    if not done("permutation_blocks.csv"):
        perm_block(model, Xs, y[idx], bmap, a.n_perm).to_csv(out / "permutation_blocks.csv", index=False)
        log.info(f"  [2b] permutação ok | {time.time()-t0:.0f}s")

    # 3) Hipóteses (bootstrap vetorizado: mesma estatística)
    S = pd.DataFrame(sv, columns=cols)
    if not done("hypotheses.csv"):
        hyp = []
        for h, (qual, quant) in HYP.items():
            if h.startswith("H1_"):
                if not all(q in B.columns for q in qual + quant):
                    log.warning(f"[{name}_{sc}] {h} pulada: blocos ausentes"); continue
                va = np.abs(B[qual].sum(1).to_numpy()); vb = np.abs(B[quant].sum(1).to_numpy())
            else:
                qa, qb = [c for c in qual if c in S], [c for c in quant if c in S]
                if not qa or not qb:
                    log.warning(f"[{name}_{sc}] {h} pulada: features ausentes {set(qual + quant) - set(S)}")
                    continue
                va = np.abs(S[qa].sum(1).to_numpy()); vb = np.abs(S[qb].sum(1).to_numpy())
            d = va - vb
            lo, hi = boot_ci(lambda ix, d=d: d[ix].mean(), gidx, a.n_boot)
            hyp.append({"hypothesis": h, "qual_side": va.mean(), "quant_side": vb.mean(),
                        "diff": d.mean(), "lo": lo, "hi": hi, "supported": bool(lo > 0)})
        if "n_traits_gold_plus" in S:
            mono = (pd.DataFrame({"v": Xs["n_traits_gold_plus"].values, "s": S["n_traits_gold_plus"]})
                    .groupby("v")["s"].mean())
            mono.to_csv(out / "h1b_monotonic.csv")
            log.info(f"  H1b SHAP médio por nº de traits ouro+: {mono.round(3).to_dict()}")
        pd.DataFrame(hyp).to_csv(out / "hypotheses.csv", index=False)
        log.info(f"  [3] hipóteses ok | {time.time()-t0:.0f}s")

    # 4) Dependência
    top_traits = imp.query("block == 'trait'")["feature"].head(5).tolist()
    dep = list(dict.fromkeys([f for f in DEP_FEATURES if f in cols] + top_traits
                             + imp["feature"].head(5).tolist()))
    for f in dep:
        if done(f"dep_{f}.png"): continue
        shap.dependence_plot(f, sv, Xs, interaction_index="auto" if a.dep_color == "auto" else None,
                             show=False)
        savefig(out / f"dep_{f}.png")
    log.info(f"  [4] dependência ok | {time.time()-t0:.0f}s")

    # 5) Interações (paralelo por blocos de linhas; resultado idêntico ao serial)
    if name == "lgbm" and not done("interactions.csv"):
        ii = rng.choice(len(Xs), min(a.n_inter, len(Xs)), replace=False)
        top = imp["feature"].head(a.top_inter).tolist()
        pos = [cols.index(c) for c in top]
        Xi = Xs.iloc[ii].reset_index(drop=True)
        n_jobs = max(1, min(THREADS, len(Xi)))
        chunks = [c for c in np.array_split(np.arange(len(Xi)), n_jobs * 2) if len(c)]
        if hasattr(model, "set_params"):
            model.set_params(n_jobs=1)                      # evita oversubscription nos filhos
        res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_inter_chunk)(model, Xi.iloc[c], pos) for c in chunks)
        if hasattr(model, "set_params"):
            model.set_params(n_jobs=THREADS)
        M = sum(r[0] for r in res) / sum(r[1] for r in res)
        del res; gc.collect()
        np.fill_diagonal(M, 0)
        Mdf = pd.DataFrame(M * 2, index=top, columns=top)
        pairs = (Mdf.where(np.triu(np.ones(M.shape), 1).astype(bool)).stack()
                 .sort_values(ascending=False).rename("mean_abs_interaction"))
        pairs.to_csv(out / "interactions.csv")
        pr = pairs.reset_index(); pr.columns = ["f1", "f2", "mean_abs_interaction"]
        b1, b2 = pr["f1"].map(bmap), pr["f2"].map(bmap)
        pr["b1"], pr["b2"] = np.minimum(b1, b2), np.maximum(b1, b2)
        (pr.groupby(["b1", "b2"])["mean_abs_interaction"].agg(["sum", "count"])
           .sort_values("sum", ascending=False).to_csv(out / "interactions_by_block.csv"))
        plt.figure(figsize=(10, 8)); plt.imshow(Mdf, cmap="viridis")
        plt.xticks(range(len(top)), top, rotation=90, fontsize=6)
        plt.yticks(range(len(top)), top, fontsize=6); plt.colorbar()
        savefig(out / "interaction_heatmap.png")
        log.info(f"  [5] interações ok ({len(ii)} linhas) | {time.time()-t0:.0f}s")

    # 6) Waterfalls
    if not done("waterfall_confident_hit.png"):
        p = proba(model, Xs); ys = y[idx]
        pl = meta["placement"].to_numpy()[idx]
        cases = {
            "confident_hit": np.argmax(np.where(ys == 1, p, -1)),
            "false_positive": np.argmax(np.where(ys == 0, p, -1)),
            "false_negative": np.argmin(np.where(ys == 1, p, 2)),
            "uncertain": np.argmin(np.abs(p - 0.5)),
        }
        if (pl == 1).any(): cases["first_place"] = np.argmax(np.where(pl == 1, p, -1))
        if (pl == 8).any(): cases["eighth_place"] = np.argmin(np.where(pl == 8, p, 2))
        for c, i in cases.items():
            e = shap.Explanation(sv[i], ev, Xs.iloc[i].values, feature_names=cols)
            shap.plots.waterfall(e, max_display=15, show=False)
            plt.title(f"{c} | p={p[i]:.2f} | colocação={pl[i]}"); savefig(out / f"waterfall_{c}.png")

    # 7) Estabilidade [v8] (mesmos folds do 03: GroupKFold + factorize na base filtrada)
    fold_params = params.get(f"{name}_{sc}", {}).get("folds", [])
    if fold_params and not done("stability.csv"):
        gk = pd.factorize(g)[0]
        n_f = len(fold_params)
        ranks, srcs = [], []
        for k, (tr, te) in enumerate(GroupKFold(n_f).split(np.zeros(len(gk)), y, gk)):
            ck = out / f"stab_fold{k}.csv"                      # checkpoint por fold
            if ck.exists() and not a.force:
                ranks.append(pd.read_csv(ck, index_col=0).iloc[:, 0].reindex(cols))
                log.info(f"  estabilidade: fold {k+1}/{n_f} carregado do checkpoint")
                continue
            tf = time.time()
            rk = np.random.default_rng(SEED + k)                # reprodutível ao retomar
            m, info = fold_model(name, sc, k, n_f, X, y, gk, tr, fold_params, a, rk)
            te_s = np.sort(rk.choice(te, min(a.n_stab, len(te)), replace=False))
            svk, _ = tree_shap(m, X[te_s])
            r = pd.Series(np.abs(svk).mean(0), index=cols, name="mean_abs_shap")
            r.to_csv(ck); ranks.append(r); srcs.append(info["src"])
            log.info(f"  estabilidade: fold {k+1}/{n_f} | origem={info['src']} | "
                     f"treino={info['n_train'] or 'completo (03)'} | árvores={info['n_iter']} | "
                     f"fit {info['t_fit']:.0f}s | total {time.time()-tf:.0f}s")
            del m, svk; gc.collect()

        stab = []
        for (i, r1), (j, r2) in combinations(enumerate(ranks), 2):
            t1, t2 = set(r1.nlargest(20).index), set(r2.nlargest(20).index)
            stab.append({"fold_i": i, "fold_j": j, "spearman": spearmanr(r1, r2).statistic,
                         "jaccard_top20": len(t1 & t2) / len(t1 | t2)})
        st = pd.DataFrame(stab); st.to_csv(out / "stability.csv", index=False)
        if srcs and len(set(srcs)) > 1:
            log.warning("  estabilidade: folds com origens mistas (03 e retreino)")
        log.info(f"  estabilidade: Spearman={st.spearman.mean():.3f} | "
                 f"Jaccard@20={st.jaccard_top20.mean():.3f}")
    log.info(f"[{name}_{sc}] concluído em {(time.time()-t0)/60:.1f} min")
    return imp


def logreg_agreement(root, a):
    rows = []
    for sc in ("A", "B"):
        f = Path(a.res) / "models" / f"final_logreg_{sc}.joblib"
        fi = root / f"lgbm_{sc}" / "importance.csv"
        if not f.exists() or not fi.exists():
            continue
        art = joblib.load(f)
        coef = pd.Series(np.abs(art["model"][-1].coef_[0]), index=art["features"])
        s = pd.read_csv(fi).set_index("feature")["mean_abs_shap"]
        common = s.index.intersection(coef.index)
        t1, t2 = set(s.nlargest(20).index), set(coef.nlargest(20).index)
        rows.append({"scenario": sc, "spearman_all": spearmanr(s[common], coef[common]).statistic,
                     "jaccard_top20": len(t1 & t2) / len(t1 | t2),
                     "n_zero_coef_L1": int((coef == 0).sum())})
    if rows:
        pd.DataFrame(rows).to_csv(root / "logreg_vs_shap.csv", index=False)


def main(a):
    global THREADS
    THREADS = a.threads
    root = Path(a.res) / "shap"; root.mkdir(parents=True, exist_ok=True)
    params = json.load(open(Path(a.res) / "best_params.json"))
    cfg_p = Path(a.res) / "run_config.json"
    cfg = json.load(open(cfg_p)) if cfg_p.exists() else {}
    drop_blocks = cfg.get("drop_blocks", []); drop_agg = cfg.get("drop_agg", [])
    min_level = a.min_level if a.min_level >= 0 else cfg.get("min_level", 0)

    # mesma carga do 03 (esparsa, mesmas colunas e ordem)
    meta, scen = train.load(a.inp, drop_blocks, drop_agg)
    fd = pd.read_csv(Path(a.inp) / "feature_dict.csv").drop_duplicates("feature")
    if min_level:
        keep = (meta["level"] >= min_level).to_numpy()
        meta = meta[keep].reset_index(drop=True)
        scen = {k: (X[keep], n) for k, (X, n) in scen.items()}
        log.info(f"[main] level >= {min_level}: {len(meta):,} linhas")
    scs = [s.strip() for s in a.scenario.split(",") if s.strip()]
    scen = {k: v for k, v in scen.items() if k in scs}; gc.collect()

    for name in a.models.split(","):
        for sc in scs:
            analyze(name, sc, meta, scen, fd, params, a, root)
    logreg_agreement(root, a)
    log.info(f"Concluído -> {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed")
    ap.add_argument("--res", default="results")
    ap.add_argument("--models", default="lgbm")
    ap.add_argument("--scenario", default="A,B")                 # [v7]
    ap.add_argument("--threads", type=int, default=8)           # [v7] use 8 se rodar A e B juntos
    ap.add_argument("--n-shap", type=int, default=5000)
    ap.add_argument("--n-inter", type=int, default=200)          # [v7] era 400
    ap.add_argument("--top-inter", type=int, default=40)
    ap.add_argument("--n-stab", type=int, default=2000)
    ap.add_argument("--stab-frac", type=float, default=0.5)      # [v8] só vale no fallback (retreino)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--dep-color", default="auto", choices=["auto", "none"])  # [v7]
    ap.add_argument("--min-level", type=int, default=-1)
    ap.add_argument("--force", action="store_true")              # [v7] refaz etapas já salvas
    main(ap.parse_args())
