"""
03_train.py — Treino e avaliação dos modelos supervisionados (Top 4).

[v6] --min-level (recorte de sobrevivência), --drop-agg (remove features),
     --drop-blocks (remove blocos inteiros: modelos só-traits / só-unidades),
     match_top4_acc protegido para partidas incompletas, run_config.json.
[v7] LGBM: teto 10000 árvores, paciência 200, lr 0.03–0.20, log do ganho marginal;
     correção de Holm nos p-valores; run_config com args, constantes e versões.

Uso:
    python 03_train.py --budget-min 480 --trials 50 --outer 5 --n-boot 2000
    python 03_train.py --min-level 8 --drop-agg n_units,board_value,n_items,n_components --out results_lvl8
    python 03_train.py --drop-blocks unit,unit_item --out results_no_units
"""
import argparse, gc, json, logging, platform, time, warnings
from itertools import combinations
from pathlib import Path

import joblib, lightgbm as lgb, numpy as np, optuna, pandas as pd
import pyarrow.parquet as pq
import scipy, sklearn
from scipy import sparse
from scipy.stats import norm, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train")
SEED, N_JOBS = 42, 8
RF_N_JOBS = -1
RF_DENSE_MAX_GB = 12
RF_MAX_SAMPLES = 30_000
RF_N_TREES = 200
rng = np.random.default_rng(SEED)
T0 = time.time()
WEIGHT = {"logreg": 0.10, "rf": 0.35, "lgbm": 0.55}
TUNE_FRAC_OF_BUDGET = 0.30
LGBM_MAX_TREES = 10000          # [v7] era 5000
LGBM_ES_FRAC = 0.10
LGBM_ES_ROUNDS = 200            # [v7] era 50
LGBM_LR_RANGE = (0.03, 0.20)    # [v7] era (0.05, 0.20)
LGBM_GAIN_WINDOW = 1000         # [v7] janela do ganho marginal de logloss
LOGREG_MAX_ITER = 100
DEADLINE_S = None
SAVE_FOLD_MODELS = ("lgbm",)    # [v8] modelos cujo ajuste por fold é salvo (RF ocupa muitos GB)

def elapsed(): return (time.time() - T0) / 60


def remaining_s(): return DEADLINE_S - (time.time() - T0)


def mb(X):
    if sparse.issparse(X):
        return (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
    return getattr(X, "nbytes", 0) / 1e6


def split_arg(s): return [x.strip() for x in s.split(",") if x.strip()]   # [v6]


def holm(p):   # [v7] correção de Holm-Bonferroni (ignora NaN)
    p = np.asarray(p, np.float64)
    adj = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    n, run = len(ok), 0.0
    for r, i in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, (n - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ================= Carga esparsa =================
def load(inp, drop_blocks=(), drop_feats=()):   # [v6]
    t = time.time()
    p = Path(inp) / "features.parquet"
    log.info(f"[load] lendo {p}")
    fd = pd.read_csv(Path(inp) / "feature_dict.csv").drop_duplicates("feature")
    miss = set(drop_feats) - set(fd["feature"])
    if miss: log.warning(f"[load] --drop-agg: features não encontradas: {sorted(miss)}")
    miss_b = set(drop_blocks) - set(fd["block"])
    if miss_b: log.warning(f"[load] --drop-blocks: blocos não encontrados: {sorted(miss_b)}")
    n0 = len(fd)
    fd = fd[~fd["block"].isin(drop_blocks) & ~fd["feature"].isin(drop_feats)]
    if len(fd) < n0:
        log.info(f"[load] removidas {n0 - len(fd)} features (blocos={list(drop_blocks)}, "
                 f"features={sorted(set(drop_feats) - miss)})")
    feats = fd.query("block != 'level_model_B'")["feature"].tolist()
    log.info(f"[load] {len(fd)} features | {len(feats)} no cenário A")
    meta = pd.read_parquet(p, columns=["match_id", "puuid", "placement", "top4", "patch", "level"])
    log.info(f"[load] meta: {len(meta):,} linhas | {meta['match_id'].nunique():,} partidas | "
             f"top4={meta['top4'].mean():.3f} | patches={sorted(meta['patch'].astype(str).unique())}")
    pf, blocks = pq.ParquetFile(p), []
    n_blocks = (len(feats) + 99) // 100
    for i in range(0, len(feats), 100):
        tb = time.time()
        chunk = pf.read(columns=feats[i:i + 100]).to_pandas().to_numpy(np.float32)
        blocks.append(sparse.csr_matrix(chunk)); del chunk; gc.collect()
        log.info(f"[load] bloco {i // 100 + 1}/{n_blocks}: nnz={blocks[-1].nnz:,} | "
                 f"{mb(blocks[-1]):.0f} MB | {time.time() - tb:.1f}s")
    patch_d = pd.get_dummies(meta["patch"].astype(str), prefix="patch", dtype=np.float32)
    blocks.append(sparse.csr_matrix(patch_d.to_numpy()))
    names = feats + patch_d.columns.tolist()
    assert len(set(names)) == len(names), "features duplicadas"
    XA = sparse.hstack(blocks, format="csr", dtype=np.float32); del blocks; gc.collect()
    XB = sparse.hstack([XA, sparse.csr_matrix(meta[["level"]].to_numpy(np.float32))],
                       format="csr", dtype=np.float32)
    dens = XA.nnz / (XA.shape[0] * XA.shape[1])
    log.info(f"[load] XA: {XA.shape}, densidade={dens:.4f}, {mb(XA):.0f} MB | "
             f"XB: {XB.shape} | tempo {time.time() - t:.1f}s")
    return meta, {"A": (XA, names), "B": (XB, names + ["level"])}


# ================= Modelos =================
def build(name, p, n_rows=None):
    if name == "logreg":
        return make_pipeline(MaxAbsScaler(), LogisticRegression(
            penalty="l1", solver="saga", max_iter=LOGREG_MAX_ITER, tol=1e-3,
            random_state=SEED, **p))
    if name == "rf":
        ms = min(RF_MAX_SAMPLES, n_rows) if n_rows else RF_MAX_SAMPLES
        return RandomForestClassifier(n_estimators=RF_N_TREES, bootstrap=True, max_samples=ms,
                                      n_jobs=RF_N_JOBS, random_state=SEED,
                                      verbose=1 if (n_rows or 0) > 500_000 else 0, **p)
    if name == "lgbm":
        return lgb.LGBMClassifier(random_state=SEED, n_jobs=N_JOBS, verbose=-1,
                                  subsample_freq=1, max_bin=63, force_col_wise=True, **p)
    raise ValueError(name)


def space(name, t):
    if name == "logreg":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True)}
    if name == "rf":
        return {"max_depth": t.suggest_int("max_depth", 8, 24),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 50, log=True),
                "max_features": t.suggest_categorical("max_features", ["sqrt", 0.3, 0.5])}
    return {"n_estimators": LGBM_MAX_TREES,
            "learning_rate": t.suggest_float("learning_rate", *LGBM_LR_RANGE, log=True),  # [v7]
            "num_leaves": t.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 20, 300, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}


def dense_if_fits(X, max_gb=RF_DENSE_MAX_GB):
    if not sparse.issparse(X):
        return np.ascontiguousarray(X, dtype=np.float32)
    gb = X.shape[0] * X.shape[1] * 4 / 1e9
    if gb <= max_gb:
        log.info(f"[dense] {X.shape} -> denso float32 ({gb:.1f} GB)")
        return X.astype(np.float32).toarray()
    log.warning(f"[dense] {gb:.1f} GB > limite, mantendo esparso")
    return X


def lgbm_marginal_gain(m):   # [v7] queda de logloss nas últimas LGBM_GAIN_WINDOW árvores
    try:
        ll = m.evals_result_["valid_0"]["binary_logloss"]
    except (AttributeError, KeyError):
        return None
    bi = m.best_iteration_ or len(ll)
    if bi <= LGBM_GAIN_WINDOW:
        return None
    return float(ll[bi - 1 - LGBM_GAIN_WINDOW] - ll[bi - 1])


def fit(name, p, X, y, Xv=None, yv=None, g=None):
    outer_es = name == "lgbm" and Xv is None and g is not None
    if outer_es:
        tr, va = next(GroupShuffleSplit(1, test_size=LGBM_ES_FRAC,
                                        random_state=SEED).split(X, y, g))
        X, Xv, y, yv = X[tr], X[va], y[tr], y[va]
        p = {**p, "n_estimators": LGBM_MAX_TREES}
    es = name == "lgbm" and Xv is not None
    if name == "rf":
        X = dense_if_fits(X)
    m = build(name, p, n_rows=X.shape[0])
    if es:
        m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(LGBM_ES_ROUNDS, verbose=False)])
        if outer_es:
            bi = m.best_iteration_ or LGBM_MAX_TREES
            hit = bi >= LGBM_MAX_TREES
            gain = lgbm_marginal_gain(m)                      # [v7]
            m.marginal_gain_ = gain
            log.info(f"[fit] lgbm early stopping: {bi} árvores"
                     + (" (ATINGIU O TETO)" if hit else "")
                     + (f" | ganho logloss últimas {LGBM_GAIN_WINDOW}: {gain:.5f}"
                        if gain is not None else ""))
            if hit and gain is not None and gain > 5e-4:
                log.warning(f"[fit] ganho marginal {gain:.5f} > 0.0005: considere aumentar "
                            f"LGBM_MAX_TREES ou o limite inferior do learning_rate")
    else:
        m.fit(X, y)
    if name == "rf":
        m.set_params(verbose=0)
    if name == "logreg" and Xv is None:
        lr = m[-1]
        log.info(f"[fit] logreg: {int((lr.coef_ != 0).sum())}/{lr.coef_.size} coef. não nulos | "
                 f"n_iter={int(lr.n_iter_[0])}"
                 + (" (NÃO CONVERGIU)" if lr.n_iter_[0] >= LOGREG_MAX_ITER else ""))
    return m


def n_iter_of(m):
    return int(m.best_iteration_) if getattr(m, "best_iteration_", None) else None


def gain_of(m): return getattr(m, "marginal_gain_", None)   # [v7]


def proba(m, X, chunk=100_000):
    rf = isinstance(m, RandomForestClassifier)
    if not (rf and sparse.issparse(X)):
        return m.predict_proba(X)[:, 1].astype(np.float64)
    return np.concatenate([m.predict_proba(X[i:i + chunk].astype(np.float32).toarray())[:, 1]
                           for i in range(0, X.shape[0], chunk)]).astype(np.float64)


def group_sample(g, frac):
    if frac >= 1:
        return np.arange(len(g))
    u = np.unique(g)
    keep = rng.choice(u, int(len(u) * frac), replace=False)
    idx = np.flatnonzero(np.isin(g, keep))
    log.info(f"[group_sample] {len(keep):,}/{len(u):,} partidas ({frac:.0%}) -> {len(idx):,} linhas")
    return idx


def inner_splits(name, X, y, g, n_inner):
    if name == "lgbm":
        return list(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, g))
    return list(GroupKFold(n_inner).split(X, y, g))


def tune(name, X, y, g, a, n_trials, timeout_s, warm=None):
    t0 = time.time()
    reserve = remaining_s() * (1 - TUNE_FRAC_OF_BUDGET)
    if warm and remaining_s() < 0.25 * DEADLINE_S:
        n_trials = 1
        log.warning(f"[tune] pouco tempo restante ({remaining_s()/60:.0f} min) -> só warm start")
    timeout_s = max(min(timeout_s, remaining_s() - reserve), 30)
    log.info(f"[tune] {name} | n_trials={n_trials} | timeout={timeout_s/60:.1f} min")
    s = group_sample(g, a.tune_frac)
    X, y, g = X[s], y[s], g[s]
    if name == "rf":
        X = dense_if_fits(X)
    cv = inner_splits(name, X, y, g, a.inner)

    def obj(t):
        tt = time.time()
        p, losses, iters = space(name, t), [], []
        for i, (tr, va) in enumerate(cv):
            m = fit(name, p, X[tr], y[tr], X[va], y[va])
            losses.append(log_loss(y[va], np.clip(proba(m, X[va]), 1e-15, 1 - 1e-15)))
            if name == "lgbm": iters.append(m.best_iteration_ or p["n_estimators"])
            t.report(np.mean(losses), i)
            if t.should_prune():
                raise optuna.TrialPruned()
        if iters: t.set_user_attr("n_iter", int(np.mean(iters)))
        res = float(np.mean(losses))
        log.info(f"[obj] trial {t.number}: loss={res:.5f} | {time.time() - tt:.1f}s"
                 + (f" | árvores={int(np.mean(iters))}" if iters else ""))
        return res

    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    if warm:
        wp = {k: v for k, v in warm.items() if k != "n_estimators"}
        if "learning_rate" in wp:   # [v7] garante warm start dentro do novo intervalo
            wp["learning_rate"] = float(np.clip(wp["learning_rate"], *LGBM_LR_RANGE))
        st.enqueue_trial(wp)
    st.optimize(obj, n_trials=n_trials, timeout=timeout_s, gc_after_trial=True)
    bp = dict(st.best_params)
    if name == "lgbm":
        bp["n_estimators"] = LGBM_MAX_TREES
    n_pr = sum(tr.state == optuna.trial.TrialState.PRUNED for tr in st.trials)
    log.info(f"    tuning {name}: {len(st.trials)} trials ({n_pr} podados), "
             f"loss={st.best_value:.5f} | {(time.time() - t0)/60:.1f} min | best={bp}")
    if name == "logreg" and st.best_value > 0.6:
        log.warning(f"[tune] logreg loss={st.best_value:.3f} > 0.6: modelo mal calibrado")
    del X; gc.collect()
    return bp, st.best_value


# ================= Métricas =================
def match_top4_acc(y, p, g):
    d = pd.DataFrame({"y": y, "p": p, "g": g})
    if (d.groupby("g").size() != 8).any():   # [v6] partidas incompletas após filtro
        return np.nan
    d["pred"] = d.groupby("g")["p"].rank(ascending=False, method="first") <= 4
    return (d["pred"] == d["y"].astype(bool)).mean()


def metrics(y, p, g):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)
    r = {"auc": roc_auc_score(y, p), "logloss": log_loss(y, p),
         "brier": brier_score_loss(y, p), "f1": f1_score(y, p >= 0.5),
         "match_top4_acc": match_top4_acc(y, p, g)}
    log.info(f"[metrics] n={len(y):,} | " + " ".join(f"{k}={v:.4f}" for k, v in r.items()))
    return r


def delong_components(y, p):
    pos = y == 1
    px, py = p[pos], p[~pos]
    m, n = len(px), len(py)
    tz = rankdata(np.concatenate([px, py]))
    v01 = (tz[:m] - rankdata(px)) / n
    v10 = 1.0 - (tz[m:] - rankdata(py)) / m
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    return auc, v01, v10


def delong_test(c1, c2):
    (a1, x1, y1), (a2, x2, y2) = c1, c2
    sx, sy = np.cov(np.vstack([x1, x2])), np.cov(np.vstack([y1, y2]))
    S = sx / len(x1) + sy / len(y1)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return float(z), float(2 * norm.sf(abs(z)))


def bootstrap(y, preds, g_codes, n_boot):
    t = time.time()
    n_m = g_codes.max() + 1
    y = y.astype(np.float64)
    pre = {}
    for k, p in preds.items():
        pc = np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)
        pre[k] = (np.argsort(pc, kind="stable"),
                  -(y * np.log(pc) + (1 - y) * np.log(1 - pc)), (pc - y) ** 2)
    out = {k: {"auc": [], "logloss": [], "brier": []} for k in preds}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_m, n_m), minlength=n_m)[g_codes].astype(np.float64)
        W = w.sum()
        for k, (o, ll, br) in pre.items():
            wo, yo = w[o], y[o]
            wneg = wo * (1 - yo)
            cneg = np.cumsum(wneg) - wneg
            out[k]["auc"].append((wo * yo * cneg).sum() / ((wo * yo).sum() * wneg.sum()))
            out[k]["logloss"].append((w * ll).sum() / W)
            out[k]["brier"].append((w * br).sum() / W)
        if (b + 1) % 100 == 0 or b + 1 == n_boot:
            log.info(f"[bootstrap] {b + 1}/{n_boot} | {time.time() - t:.0f}s")
    return {k: pd.DataFrame(v) for k, v in out.items()}


# ================= Pipeline =================
def run_nested(name, X, y, g, a, budget_s, warm=None, save_tag=None):   # [v8] save_tag
    t0 = time.time()
    oof, fps = np.zeros(len(y), np.float64), []
    share = [0.4] + [0.6 / (a.outer - 1)] * (a.outer - 1)
    for k, (tr, te) in enumerate(GroupKFold(a.outer).split(X, y, g)):
        tf = time.time()
        n_tr = a.trials if (k == 0 and warm is None) else max(4, a.trials // 3)
        log.info(f"  [fold {k+1}/{a.outer}] treino={len(tr):,} teste={len(te):,} | trials={n_tr}")
        bp, loss = tune(name, X[tr], y[tr], g[tr], a, n_tr, budget_s * share[k], warm)
        warm = bp
        m = fit(name, bp, X[tr], y[tr], g=g[tr])
        oof[te] = proba(m, X[te])
        if save_tag and name in SAVE_FOLD_MODELS:                          # [v8]
            fpath = Path(a.out) / "models" / f"fold_{save_tag}_{k}.joblib"
            joblib.dump({"model": m, "fold": k, "n_folds": a.outer,
                         "n_iter": n_iter_of(m)}, fpath, compress=3)
            log.info(f"  [fold {k+1}] modelo salvo {fpath.name} ({fpath.stat().st_size/1e6:.1f} MB)")
        fps.append({"params": bp, "inner_loss": loss, "n_iter": n_iter_of(m),
                    "marginal_gain": gain_of(m)})   # [v7]
        log.info(f"  fold {k+1}: AUC={roc_auc_score(y[te], oof[te]):.4f} | "
                 f"{(time.time() - tf)/60:.1f} min | decorrido {elapsed():.0f} min")
        del m; gc.collect()
    log.info(f"[run_nested] {name}: {(time.time() - t0)/60:.1f} min | AUC OOF={roc_auc_score(y, oof):.4f}")
    return oof, fps



def main(a):
    global DEADLINE_S
    DEADLINE_S = a.budget_min * 60
    log.info(f"[main] args: {vars(a)}")
    out = Path(a.out); (out / "models").mkdir(parents=True, exist_ok=True)
    meta, scen = load(a.inp, split_arg(a.drop_blocks), split_arg(a.drop_agg))   # [v6]

    if a.min_level:   # [v6] recorte de sobrevivência
        keep = (meta["level"] >= a.min_level).to_numpy()
        meta = meta[keep].reset_index(drop=True)
        scen = {k: (X[keep], n) for k, (X, n) in scen.items()}
        log.info(f"[main] level >= {a.min_level}: {keep.sum():,} linhas | "
                 f"{meta['match_id'].nunique():,} partidas | top4={meta['top4'].mean():.3f}")

    json.dump({"args": vars(a),                                                  # [v7]
               "min_level": a.min_level, "drop_agg": split_arg(a.drop_agg),
               "drop_blocks": split_arg(a.drop_blocks), "n_rows": len(meta),
               "n_matches": int(meta["match_id"].nunique()),
               "top4_rate": float(meta["top4"].mean()),
               "constants": {"SEED": SEED, "LGBM_MAX_TREES": LGBM_MAX_TREES,
                             "LGBM_ES_ROUNDS": LGBM_ES_ROUNDS, "LGBM_ES_FRAC": LGBM_ES_FRAC,
                             "LGBM_LR_RANGE": LGBM_LR_RANGE, "RF_N_TREES": RF_N_TREES,
                             "RF_MAX_SAMPLES": RF_MAX_SAMPLES, "LOGREG_MAX_ITER": LOGREG_MAX_ITER,
                             "TUNE_FRAC_OF_BUDGET": TUNE_FRAC_OF_BUDGET,
                             "SAVE_FOLD_MODELS": list(SAVE_FOLD_MODELS)},
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__,
                            "sklearn": sklearn.__version__, "lightgbm": lgb.__version__,
                            "optuna": optuna.__version__}},
              open(out / "run_config.json", "w"), indent=2, default=str)

    y = meta["top4"].to_numpy(np.int8)
    g = pd.factorize(meta["match_id"])[0]
    models = a.models.split(",")

    tune_budget = DEADLINE_S * TUNE_FRAC_OF_BUDGET
    wsum = sum(WEIGHT[m] for m in models)
    sc_share = {"A": 0.60, "B": 0.40}

    oof_all, params = {}, {}
    lv = meta[["level"]].to_numpy(np.float32)
    oof_all["baseline_level"] = np.zeros(len(y), np.float64)
    for tr, te in GroupKFold(a.outer).split(lv, y, g):
        oof_all["baseline_level"][te] = proba(LogisticRegression().fit(lv[tr], y[tr]), lv[te])
    log.info(f"[main] baseline AUC OOF={roc_auc_score(y, oof_all['baseline_level']):.4f}")

    for sc, (X, names) in scen.items():
        for name in models:
            b = tune_budget * WEIGHT[name] / wsum * sc_share[sc]
            log.info(f"== {name} | {sc} | {X.shape[1]} feats | tuning {b/60:.1f} min "
                     f"| decorrido {elapsed():.0f} min")
            warm = params.get(f"{name}_A", {}).get("final") if sc == "B" else None
            oof, fps = run_nested(name, X, y, g, a, b, warm, save_tag=f"{name}_{sc}")   # [v8]
            oof_all[f"{name}_{sc}"] = oof
            bp = min(fps, key=lambda d: d["inner_loss"])["params"]
            m = fit(name, bp, X, y, g=g)
            path = out / "models" / f"final_{name}_{sc}.joblib"
            joblib.dump({"model": m, "features": names}, path, compress=3)
            log.info(f"[main] salvo {path} ({path.stat().st_size/1e6:.1f} MB)")
            params[f"{name}_{sc}"] = {"folds": fps, "final": bp, "final_n_iter": n_iter_of(m),
                                      "final_marginal_gain": gain_of(m)}   # [v7]
            del m; gc.collect()

    boots = bootstrap(y, oof_all, g, a.n_boot)
    rows = []
    for k, p in oof_all.items():
        r = {"model": k, **metrics(y, p, g)}
        for mt in ("auc", "logloss", "brier"):
            r[f"{mt}_lo"], r[f"{mt}_hi"] = np.percentile(boots[k][mt], [2.5, 97.5])
        rows.append(r)
    met = pd.DataFrame(rows).sort_values("auc", ascending=False)
    met.to_csv(out / "metrics.csv", index=False)
    log.info("\n" + met.round(4).to_string(index=False))

    dl = {k: delong_components(y, np.asarray(p, np.float64)) for k, p in oof_all.items()}
    comp = []
    for m1, m2 in combinations(oof_all, 2):
        for mn in ("auc", "logloss"):
            d = boots[m1][mn] - boots[m2][mn]
            row = {"m1": m1, "m2": m2, "metric": mn, "diff": d.mean(),
                   "lo": d.quantile(.025), "hi": d.quantile(.975),
                   "p_value": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())),
                   "delong_z": np.nan, "delong_p": np.nan}
            if mn == "auc":
                row["delong_z"], row["delong_p"] = delong_test(dl[m1], dl[m2])
                log.info(f"[delong] {m1} vs {m2}: z={row['delong_z']:.2f} | p={row['delong_p']:.2e}")
            comp.append(row)
    cdf = pd.DataFrame(comp)
    cdf["p_holm"], cdf["delong_p_holm"] = np.nan, np.nan   # [v7] Holm por métrica
    for mn, idx in cdf.groupby("metric").groups.items():
        cdf.loc[idx, "p_holm"] = holm(cdf.loc[idx, "p_value"])
        if mn == "auc":
            cdf.loc[idx, "delong_p_holm"] = holm(cdf.loc[idx, "delong_p"])
    cdf.to_csv(out / "comparisons.csv", index=False)
    log.info(f"[main] Holm aplicado: {len(cdf)} comparações "
             f"({(cdf['p_holm'] < 0.05).sum()} significativas a 5%)")
    del dl; gc.collect()

    pat = meta["patch"].astype(str)
    patches = sorted(pat.unique(), key=lambda s: tuple(map(int, s.split("."))))
    if len(patches) > 1:
        te = (pat == patches[-1]).to_numpy(); tr = ~te
        log.info(f"[main] temporal: teste=patch {patches[-1]} ({te.sum():,}) | treino={tr.sum():,}")
        trows = []
        for sc, (X, names) in scen.items():
            keep = [i for i, n in enumerate(names) if not n.startswith("patch_")]
            Xt = X[:, keep]
            Xtr, Xte = Xt[tr], Xt[te]; del Xt; gc.collect()
            for name in models:
                m = fit(name, params[f"{name}_{sc}"]["final"], Xtr, y[tr], g=g[tr])
                trows.append({"model": f"{name}_{sc}", "test_patch": patches[-1],
                              "n_iter": n_iter_of(m), "marginal_gain": gain_of(m),   # [v7]
                              **metrics(y[te], proba(m, Xte), g[te])})
                del m; gc.collect()
            del Xtr, Xte; gc.collect()
        pd.DataFrame(trows).to_csv(out / "temporal.csv", index=False)

    oof_df = meta[["match_id", "puuid", "placement", "top4"]].copy()
    for k, p in oof_all.items(): oof_df[f"p_{k}"] = p.astype(np.float32)
    oof_df.to_parquet(out / "oof_predictions.parquet", index=False)
    json.dump(params, open(out / "best_params.json", "w"), indent=2, default=str)
    log.info(f"[main] saídas em {out.resolve()} | Total: {elapsed():.0f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="logreg,rf,lgbm")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--outer", type=int, default=3)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--tune-frac", type=float, default=0.25)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--budget-min", type=float, default=240)
    ap.add_argument("--min-level", type=int, default=0)      # [v6]
    ap.add_argument("--drop-agg", default="")                # [v6] features separadas por vírgula
    ap.add_argument("--drop-blocks", default="")             # [v6] ex.: unit,unit_item
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args()
    log.setLevel(args.log_level)
    main(args)
