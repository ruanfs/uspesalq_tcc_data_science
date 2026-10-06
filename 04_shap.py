"""
04_shap.py — Interpretabilidade SHAP. v9, compatível com o 03_train v9.

Princípios (Parecer 1.5, 2.2 e Etapa 2):
  * O SHAP é DESCRITIVO. A inferência sobre H1–H3 vem das ablações do 03 (comparisons.csv).
  * Modelo final explicado no 16.5 (out-of-time). Os modelos de fold são explicados nas suas
    próprias linhas OOF -> IC das participações por bloco ENTRE folds (inclui a variância de
    reajuste), com a correção de Nadeau & Bengio.
  * 50k linhas no global; 2k nas interações (com IC por bootstrap de partidas).
  * Importância com sinal: corr(x, φ) e φ médio.
  * Casos: par 4º×5º com tabuleiros parecidos e previsões opostas, mais os top 100 FP/FN.

Uso:
  python 04_shap.py --res results_smoke --specs lgbm_A,lgbm_C
  python 04_shap.py --res results_smoke --specs lgbm_A --n-inter 200   # estimativa de tempo
"""
import argparse, gc, importlib, json, logging, time, warnings
from itertools import combinations
from pathlib import Path

import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, shap
from joblib import Parallel, delayed
from scipy.stats import spearmanr, t as student_t
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
SCRIPT_VERSION, SEED, THREADS = "04_shap v9", 42, 8
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("shap")
train = importlib.import_module("03_train")

# Mesmas definições do ablations.json (Q = qualidade, R = referência)
HYP = {
    "H1_traits_vs_units": ({"blocks": ["trait"]}, {"blocks": ["unit"]}),
    "H1b_high_tier": ({"feats": ["n_traits_gold_plus", "n_traits_prismatic", "gold_plus_share"]},
                      {"feats": ["n_active_traits"]}),
    "H2_quality_vs_quantity": ({"feats": ["n_hi_cost_2star", "share_hi_cost_2star"]},
                               {"feats": ["sum_cost"]}),
    "H3_concentration_vs_total": ({"feats": ["n_carries", "n_full_items_on_hi_cost", "carries_per_unit"]},
                                  {"feats": ["n_full_items", "full_items_per_unit"]}),
}
DEP_FEATURES = ["board_value", "n_traits_gold_plus", "n_hi_cost_2star", "sum_cost",
                "n_carries", "n_full_items", "board_value_per_unit", "share_2star"]


# ================= Utilitários =================
def block_map(cols, fd):
    m = dict(zip(fd["feature"], fd["block"]))
    return {c: m.get(c, "level" if c == "level" else "other") for c in cols}


def group_idx(spec, cols, bmap):
    f, b = set(spec.get("feats", [])), set(spec.get("blocks", []))
    return [i for i, c in enumerate(cols) if c in f or bmap[c] in b]


def sample_matches(pool, g, n, rng):
    mids = np.unique(g[pool])
    k = min(len(mids), max(1, n // 8))
    return pool[np.isin(g[pool], rng.choice(mids, k, replace=False))]


def tree_shap(m, X, chunk=5000):
    """TreeSHAP nativo do LightGBM (log-odds), em blocos."""
    out = [np.asarray(m.booster_.predict(X[i:i + chunk].toarray(), pred_contrib=True,
                                         num_threads=THREADS), np.float32)
           for i in range(0, X.shape[0], chunk)]
    c = np.vstack(out)
    return c[:, :-1], float(c[0, -1])


def proba(m, X): return m.predict_proba(X)[:, 1]


def cv_ci(v):
    """Média ± t·SE com a correção de Nadeau & Bengio (folds compartilham o treino)."""
    v = np.asarray(v, float); k = len(v)
    if k < 2:
        return float(v.mean()), np.nan, np.nan
    se = v.std(ddof=1) * np.sqrt(1 / k + 1 / (k - 1))
    q = student_t.ppf(0.975, k - 1)
    return float(v.mean()), float(v.mean() - q * se), float(v.mean() + q * se)


def block_shares(sv, cols, bmap):
    blocks = sorted(set(bmap.values()))
    a = np.array([np.abs(sv[:, [i for i, c in enumerate(cols) if bmap[c] == b]].sum(1)).mean()
                  for b in blocks])
    return pd.Series(a / a.sum(), index=blocks)


def hyp_stats(sv, cols, bmap):
    r = {}
    for h, (Q, R) in HYP.items():
        iq, ir = group_idx(Q, cols, bmap), group_idx(R, cols, bmap)
        if not iq or not ir:
            continue
        q, s = sv[:, iq].sum(1), sv[:, ir].sum(1)
        r[h] = {"abs_Q": float(np.abs(q).mean()), "abs_R": float(np.abs(s).mean()),
                "mean_Q": float(q.mean()), "mean_R": float(s.mean()),
                "n_feat_Q": len(iq), "n_feat_R": len(ir)}
    return r


def direction(Xd, sv):
    """corr(x_j, φ_j): >0 = valores altos empurram para o Top 4."""
    xc, sc = Xd - Xd.mean(0), sv - sv.mean(0)
    den = np.sqrt((xc ** 2).sum(0) * (sc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (xc * sc).sum(0) / den, np.nan)


def savefig(p):
    plt.tight_layout(); plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close("all")


def boot_match(vals, g, n_boot, seed):
    """IC 95% da média por bootstrap de partidas. vals: (n, P)."""
    codes = pd.factorize(g)[0]
    S = np.zeros((codes.max() + 1, vals.shape[1])); np.add.at(S, codes, vals)
    c = np.bincount(codes).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(c), (n_boot, len(c)))
    b = S[idx].sum(1) / c[idx].sum(1)[:, None]
    return np.percentile(b, [2.5, 97.5], axis=0)


# ================= Etapas =================
def fold_analysis(spec, cols, X, bmap, a, out):
    files = sorted((Path(a.res) / "models").glob(f"fold_{spec}_*.joblib"),
                   key=lambda f: int(f.stem.rsplit("_", 1)[1]))
    if not files:
        log.warning(f"[{spec}] sem modelos de fold: IC entre folds indisponível"); return None
    imps, shares, hyps = [], [], []
    for f in files:
        k = int(f.stem.rsplit("_", 1)[1]); ck = out / f"fold{k}.joblib"
        if ck.exists() and not a.force:
            r = joblib.load(ck)
        else:
            art = joblib.load(f)
            assert list(art["features"]) == cols, f"{f.name}: colunas divergentes"
            m = art["model"]; m.set_params(n_jobs=THREADS)
            rk = np.random.default_rng(SEED + k)
            pool = art["test_rows_global"]
            rows = np.sort(rk.choice(pool, min(a.n_stab, len(pool)), replace=False))
            sv, _ = tree_shap(m, X[rows])
            r = {"imp": pd.Series(np.abs(sv).mean(0), index=cols),
                 "share": block_shares(sv, cols, bmap), "hyp": hyp_stats(sv, cols, bmap)}
            joblib.dump(r, ck); del m, sv; gc.collect()
        imps.append(r["imp"]); shares.append(r["share"]); hyps.append(r["hyp"])
        log.info(f"  [{spec}] fold {k}: SHAP OOF ok")
    st = [{"fold_i": i, "fold_j": j, "spearman": spearmanr(r1, r2).statistic,
           "jaccard_top20": len(set(r1.nlargest(20).index) & set(r2.nlargest(20).index)) /
                            len(set(r1.nlargest(20).index) | set(r2.nlargest(20).index))}
          for (i, r1), (j, r2) in combinations(enumerate(imps), 2)]
    st = pd.DataFrame(st); st.to_csv(out / "stability.csv", index=False)
    log.info(f"  [{spec}] estabilidade: Spearman={st.spearman.mean():.3f} | "
             f"Jaccard@20={st.jaccard_top20.mean():.3f}")
    return {"share": pd.DataFrame(shares), "hyp": hyps}


def perm_block(m, Xd, y, cols, bmap, n_rep, rng):
    base = roc_auc_score(y, proba(m, Xd)); out = []
    for b in sorted(set(bmap.values())):
        ix = [i for i, c in enumerate(cols) if bmap[c] == b]
        d = []
        for _ in range(n_rep):
            Xp = Xd.copy(); Xp[:, ix] = Xd[rng.permutation(len(Xd))][:, ix]
            d.append(base - roc_auc_score(y, proba(m, Xp))); del Xp
        out.append({"block": b, "n_features": len(ix), "auc_drop": np.mean(d), "sd": np.std(d)})
    return pd.DataFrame(out).sort_values("auc_drop", ascending=False)


def _inter_chunk(model, Xc, pos):
    iv = shap.TreeExplainer(model).shap_interaction_values(Xc)
    iv = iv[1] if isinstance(iv, list) else iv
    return np.abs(iv[:, pos][:, :, pos]).astype(np.float32)


def interactions(m, Xd, g, cols, imp, bmap, a, out, rng):
    ii = np.sort(rng.choice(len(Xd), min(a.n_inter, len(Xd)), replace=False))
    top = imp["feature"].head(a.top_inter).tolist(); pos = [cols.index(c) for c in top]
    chunks = [ii[i:i + a.inter_chunk] for i in range(0, len(ii), a.inter_chunk)]
    t0 = time.time(); m.set_params(n_jobs=1)
    R = np.concatenate(Parallel(n_jobs=THREADS, backend="loky", verbose=5)(
        delayed(_inter_chunk)(m, Xd[c], pos) for c in chunks))
    m.set_params(n_jobs=THREADS)
    iu = np.triu_indices(len(top), 1)
    V = R[:, iu[0], iu[1]] * 2                     # efeito total do par (i,j)+(j,i)
    mean = V.mean(0); order = np.argsort(-mean)[:a.top_pairs_ci]
    lo, hi = boot_match(V[:, order], g[ii], a.n_boot, SEED)
    pr = pd.DataFrame({"f1": np.array(top)[iu[0]], "f2": np.array(top)[iu[1]],
                       "mean_abs_interaction": mean})
    pr["lo"], pr["hi"] = np.nan, np.nan
    pr.loc[pr.index[order], "lo"], pr.loc[pr.index[order], "hi"] = lo, hi
    pr["b1"], pr["b2"] = pr.f1.map(bmap), pr.f2.map(bmap)
    pr.sort_values("mean_abs_interaction", ascending=False).to_csv(out / "interactions.csv", index=False)
    (pr.assign(bb=pr[["b1", "b2"]].apply(lambda r: " × ".join(sorted(r)), axis=1))
       .groupby("bb")["mean_abs_interaction"].agg(["sum", "count", "mean"])
       .sort_values("sum", ascending=False).to_csv(out / "interactions_by_block.csv"))
    M = np.zeros((len(top),) * 2); M[iu] = mean; M += M.T
    plt.figure(figsize=(10, 8)); plt.imshow(M, cmap="viridis"); plt.colorbar()
    plt.xticks(range(len(top)), top, rotation=90, fontsize=6)
    plt.yticks(range(len(top)), top, fontsize=6); savefig(out / "interaction_heatmap.png")
    log.info(f"  interações: {len(ii)} linhas | {(time.time() - t0) / 60:.1f} min")


def boundary_pair(Xs, p, pl, cols, bmap, rng, cap=5000):
    """4º com p<0,5 × 5º com p>0,5, maximizando Jaccard(traits+unidades) × |Δp|."""
    jb = [i for i, c in enumerate(cols) if bmap[c] in ("trait", "unit")]
    i4 = np.flatnonzero((pl == 4) & (p < .5)); i5 = np.flatnonzero((pl == 5) & (p > .5))
    if not len(i4) or not len(i5):
        return None
    i4 = rng.choice(i4, min(cap, len(i4)), replace=False)
    i5 = rng.choice(i5, min(cap, len(i5)), replace=False)
    A = (Xs[i4][:, jb] > 0).astype(np.float32); B = (Xs[i5][:, jb] > 0).astype(np.float32)
    inter = (A @ B.T).toarray(); na, nb = A.sum(1).A1, B.sum(1).A1
    jac = inter / (na[:, None] + nb[None, :] - inter + 1e-9)
    sc = jac * (p[i5][None, :] - p[i4][:, None])
    a_, b_ = np.unravel_index(np.argmax(sc), sc.shape)
    return int(i4[a_]), int(i5[b_]), float(jac[a_, b_])


def top_contrib(sv_row, cols, k=5):
    o = np.argsort(-np.abs(sv_row))[:k]
    return "; ".join(f"{cols[i]}={sv_row[i]:+.2f}" for i in o)


def cases(m, Xs, Xd, sv, ev, ys, pl, gs, cols, bmap, out, rng):
    p = proba(m, Xs)
    lgt = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    err = np.abs(sv[:200].sum(1) + ev - lgt[:200]).max()
    assert err < 1e-3, f"aditividade SHAP falhou ({err:.2e})"
    sel = {"false_positive": np.argmax(np.where(ys == 0, p, -1)),
           "false_negative": np.argmin(np.where(ys == 1, p, 2)),
           "uncertain": np.argmin(np.abs(p - .5))}
    bp = boundary_pair(Xs, p, pl, cols, bmap, rng)
    if bp:
        sel["boundary_4th_low_p"], sel["boundary_5th_high_p"] = bp[0], bp[1]
        json.dump({"jaccard_traits_units": bp[2], "p_4th": float(p[bp[0]]),
                   "p_5th": float(p[bp[1]])}, open(out / "boundary_pair.json", "w"), indent=2)
    for c, i in sel.items():
        shap.plots.waterfall(shap.Explanation(sv[i], ev, Xd[i], feature_names=cols),
                             max_display=15, show=False)
        plt.title(f"{c} | p={p[i]:.2f} | colocação={pl[i]}"); savefig(out / f"waterfall_{c}.png")
    ctx = [c for c in ("level", "n_units", "board_value", "n_full_items") if c in cols]
    rows = []
    for kind, mask, asc in (("FP", ys == 0, False), ("FN", ys == 1, True)):
        ix = np.flatnonzero(mask); ix = ix[np.argsort(p[ix] if asc else -p[ix])][:100]
        for i in ix:
            rows.append({"type": kind, "match_id": gs[i], "placement": int(pl[i]),
                         "p": float(p[i]), **{c: float(Xd[i, cols.index(c)]) for c in ctx},
                         "top_contrib": top_contrib(sv[i], cols)})
    pd.DataFrame(rows).to_csv(out / "errors_top100.csv", index=False)


# ================= Análise por spec =================
def analyze(spec, a, fd, meta, Xall, colidx, root):
    t0 = time.time(); out = root / spec; out.mkdir(parents=True, exist_ok=True)
    art = joblib.load(Path(a.res) / "models" / f"final_{spec}.joblib")
    assert art.get("fit_scope") == "train_only", "modelo final não é train_only"
    m, cols = art["model"], list(art["features"]); m.set_params(n_jobs=THREADS)
    X = Xall[:, [colidx[c] for c in cols]].tocsr()
    bmap = block_map(cols, fd)
    rng = np.random.default_rng(SEED)
    g = meta["match_id"].to_numpy(); y = meta["top4"].to_numpy(); pl = meta["placement"].to_numpy()
    te = np.flatnonzero(meta["split"].astype(str).to_numpy() == "test")

    # 1) Global no 16.5 (out-of-time)
    idx = np.sort(sample_matches(te, g, a.n_shap, rng))
    Xs = X[idx]; Xd = Xs.toarray().astype(np.float32)
    sv, ev = tree_shap(m, Xs)
    log.info(f"[{spec}] SHAP 16.5: {sv.shape} | {time.time() - t0:.0f}s")
    imp = pd.DataFrame({"feature": cols, "block": [bmap[c] for c in cols],
                        "mean_abs_shap": np.abs(sv).mean(0), "mean_shap": sv.mean(0),
                        "direction_corr": direction(Xd, sv),
                        "freq_nonzero": (Xd != 0).mean(0)}
                       ).sort_values("mean_abs_shap", ascending=False)
    imp["rank"] = np.arange(1, len(imp) + 1); imp.to_csv(out / "importance.csv", index=False)
    npl = min(10_000, len(Xd))
    shap.summary_plot(sv[:npl], Xd[:npl], feature_names=cols, max_display=25, show=False)
    savefig(out / "beeswarm.png")

    # 2) Folds: participação por bloco e hipóteses com IC entre folds
    fa = fold_analysis(spec, cols, X, bmap, a, out)
    sh_te = block_shares(sv, cols, bmap)
    gr = pd.DataFrame({"block": sh_te.index, "share_test": sh_te.values})
    if fa is not None:
        ci = {b: cv_ci(fa["share"][b]) for b in fa["share"].columns}
        gr["share_folds"], gr["lo"], gr["hi"] = zip(*[ci.get(b, (np.nan,) * 3) for b in gr.block])
    gr = gr.sort_values("share_test", ascending=False); gr.to_csv(out / "grouped.csv", index=False)
    plt.figure(figsize=(7, 4))
    xerr = None if fa is None else [gr.share_folds - gr.lo, gr.hi - gr.share_folds]
    plt.barh(gr.block, gr.get("share_folds", gr.share_test), xerr=xerr)
    plt.gca().invert_yaxis(); plt.xlabel("Participação em |SHAP| (IC 95% entre folds)")
    savefig(out / "grouped_blocks.png")

    # 3) Hipóteses — SOMENTE DESCRITIVO (inferência: ablações em comparisons.csv)
    ht = hyp_stats(sv, cols, bmap); rows = []
    for h, s in ht.items():
        r = {"hypothesis": h, **{f"{k}_test": v for k, v in s.items()},
             "diff_abs_test": s["abs_Q"] - s["abs_R"],
             "inference": f"ver comparisons.csv: abl_{h.split('_')[0]}_noR vs abl_{h.split('_')[0]}_noQ"}
        if fa is not None:
            d = [fh[h]["abs_Q"] - fh[h]["abs_R"] for fh in fa["hyp"] if h in fh]
            r["diff_abs_folds"], r["diff_lo"], r["diff_hi"] = cv_ci(d)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "hypotheses_descriptive.csv", index=False)
    if "n_traits_gold_plus" in cols:
        j = cols.index("n_traits_gold_plus")
        (pd.DataFrame({"v": Xd[:, j], "s": sv[:, j]}).groupby("v")["s"].agg(["mean", "count"])
           .to_csv(out / "h1b_monotonic.csv"))

    # 4) Permutação por bloco
    if a.n_perm and not (out / "permutation_blocks.csv").exists() or a.force:
        pi = rng.choice(len(Xd), min(a.n_perm_rows, len(Xd)), replace=False)
        perm_block(m, Xd[pi], y[idx][pi], cols, bmap, a.n_perm, rng).to_csv(
            out / "permutation_blocks.csv", index=False)

    # 5) Dependência
    dep = list(dict.fromkeys([f for f in DEP_FEATURES if f in cols] +
                             imp.query("block == 'trait'").feature.head(5).tolist() +
                             imp.feature.head(5).tolist()))
    for f in dep:
        shap.dependence_plot(f, sv[:npl], Xd[:npl], feature_names=cols, show=False)
        savefig(out / f"dep_{f}.png")

    # 6) Interações
    if a.n_inter and (not (out / "interactions.csv").exists() or a.force):
        interactions(m, Xd, g[idx], cols, imp, bmap, a, out, rng)

    # 7) Casos e erros
    cases(m, Xs, Xd, sv, ev, y[idx], pl[idx], g[idx], cols, bmap, out, rng)
    log.info(f"[{spec}] concluído em {(time.time() - t0) / 60:.1f} min")
    del Xs, Xd, sv, X; gc.collect()


def main(a):
    global THREADS; THREADS = a.threads
    res = Path(a.res); root = res / "shap"; root.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(res / "run_config.json"))
    min_level = cfg["args"].get("min_level", 0)
    fd = pd.read_csv(Path(a.inp) / "feature_dict.csv").drop_duplicates("feature")
    specs = train.split_arg(a.specs)
    needed = set()
    for s in specs:
        needed |= set(joblib.load(res / "models" / f"final_{s}.joblib")["features"])
    meta, Xall, colidx = train.load(a.inp, sorted(needed | {"level"}))
    if min_level:
        keep = (meta["level"] >= min_level).to_numpy()
        meta, Xall = meta[keep].reset_index(drop=True), Xall[keep]
    Xall = Xall.tocsr()
    json.dump({"script": SCRIPT_VERSION, "args": vars(a), "min_level": min_level,
               "shap": shap.__version__}, open(root / "shap_config.json", "w"), indent=2)
    for s in specs:
        analyze(s, a, fd, meta, Xall, colidx, root)
    log.info(f"Concluído -> {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed")
    ap.add_argument("--res", default="results")
    ap.add_argument("--specs", default="lgbm_A,lgbm_C")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-shap", type=int, default=50_000)
    ap.add_argument("--n-stab", type=int, default=20_000)
    ap.add_argument("--n-inter", type=int, default=2_000)
    ap.add_argument("--inter-chunk", type=int, default=25, help="linhas por tarefa (~130 MB cada)")
    ap.add_argument("--top-inter", type=int, default=40)
    ap.add_argument("--top-pairs-ci", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--n-perm-rows", type=int, default=20_000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
