"""
03_train.py — Treino e avaliação dos modelos supervisionados (Top 4). v9.1

Desenho de validação (Parecer 1.4 e 2.2):
  * TREINO = split=='train' (16.1–16.4); TESTE = split=='test' (16.5), puramente out-of-time.
    Nenhum tuning, escala, early stopping ou vocabulário vê o 16.5.
  * CV aninhada SÓ no treino: outer GroupKFold por partida (OOF, IC, ablações pareadas);
    inner (Optuna) dentro de cada fold externo.
  * Modelo final: novo tuning no treino inteiro, ajuste no treino e UMA avaliação no 16.5.
  * Sem dummies de patch (o 16.5 seria uma categoria não vista).

Especificações (Parecer 1.2 e 1.3):
  base_level / base_board_value : logística univariada
  agg_lgbm                      : LightGBM com os 22 agregados (bloco 'aggregate')
  agg+{trait,unit,item,sparse}  : ganho marginal dos blocos esparsos sobre agg_lgbm
  {model}_A : agregados + intensidade + esparsos + arquétipos (sem level)
  {model}_B : A + level
  {model}_C : Cenário C (in_scenario_c, sem volume)
  abl_*     : ablações (--ablations JSON) -> Parecer 1.5

[v9.1] --lgbm-lr-min e --es-rounds; ablações herdam os hiperparâmetros da base fold a fold
       (sem tuning; --abl-retune para re-tunar); modelos de fold não são salvos para abl_*.

Comparações pareadas: Δ fold a fold com t corrigido de Nadeau & Bengio (2003);
no 16.5, bootstrap de partidas e DeLong; Holm por métrica.

Uso:
  python 03_train.py --outer 5 --trials 30 --n-boot 2000
  python 03_train.py --only agg_lgbm,lgbm_A,lgbm_C --out results_quick
  python 03_train.py --min-level 8 --out results_lvl8
"""
import argparse, gc, inspect, json, logging, platform, time, warnings
from pathlib import Path

import joblib, lightgbm as lgb, numpy as np, optuna, pandas as pd
import pyarrow.parquet as pq
import scipy, sklearn
from scipy import sparse
from scipy.stats import norm, rankdata, t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

# LightGBM novo: eval_X/eval_y; antigo: eval_set
_LGBM_NEW_EVAL = "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters


def _lgbm_eval_kwargs(Xv, yv):
    return ({"eval_X": (Xv,), "eval_y": (yv,)} if _LGBM_NEW_EVAL
            else {"eval_set": [(Xv, yv)]})


SCRIPT_VERSION = "03_train v9.2"
SEED, N_JOBS = 42, 8
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("train")

CFG = {"LGBM_MAX_TREES": 10_000, "LGBM_ES_FRAC": 0.10, "LGBM_ES_ROUNDS": 200,
       "LGBM_LR_RANGE": (0.03, 0.20), "LOGREG_MAX_ITER": 5_000, "LOGREG_TOL": 1e-4,
       "RF_N_TREES": 300, "RF_MAX_SAMPLES": 0.20, "RF_DENSE_MAX_GB": 12}
SPARSE_BLOCKS = ["trait", "unit", "item", "unit_item"]
SAVE_FOLD_MODELS = ("lgbm",)
SLOW_MODELS = ("rf", "logreg")
BASE_PAIRS = [("agg_lgbm", "base_board_value"), ("base_board_value", "base_level"),
              ("agg+trait_lgbm", "agg_lgbm"), ("agg+unit_lgbm", "agg_lgbm"),
              ("agg+item_lgbm", "agg_lgbm"), ("agg+sparse_lgbm", "agg_lgbm"),
              ("lgbm_A", "agg_lgbm"), ("lgbm_B", "lgbm_A"), ("lgbm_A", "lgbm_C"),
              ("lgbm_A", "rf_A"), ("lgbm_A", "logreg_A"), ("lgbm_A", "agg+sparse_lgbm")]
ABL_BASE = {}

T0 = time.time()


def elapsed(): return (time.time() - T0) / 60
def split_arg(s): return [x.strip() for x in (s or "").split(",") if x.strip()]
def clip(p): return np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)


def holm(p):
    p = np.asarray(p, np.float64)
    adj = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    n, run = len(ok), 0.0
    for r, i in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, (n - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ================= Especificações =================
def build_specs(fd, models, ablations):
    b = lambda *bl: fd.loc[fd["block"].isin(bl), "feature"].tolist()
    agg = b("aggregate")
    A = agg + b("aggregate_norm") + b(*SPARSE_BLOCKS) + b("archetype")
    C = fd.loc[fd["in_scenario_c"].astype(bool) & (fd["block"] != "level_model_B"),
               "feature"].tolist()
    S = {"base_level": ("logit", ["level"]),
         "base_board_value": ("logit", ["board_value"]),
         "agg_lgbm": ("lgbm", agg),
         "agg+trait_lgbm": ("lgbm", agg + b("trait")),
         "agg+unit_lgbm": ("lgbm", agg + b("unit")),
         "agg+item_lgbm": ("lgbm", agg + b("item", "unit_item")),
         "agg+sparse_lgbm": ("lgbm", agg + b(*SPARSE_BLOCKS))}
    for m in models:
        S[f"{m}_A"], S[f"{m}_B"], S[f"{m}_C"] = (m, A), (m, A + ["level"]), (m, C)
    pairs = list(BASE_PAIRS)

    abl = ablations or {}
    for hname, spec in abl.get("specs", {}).items():
        base = spec["base"]
        if base not in S:
            raise KeyError(f"ablação {hname}: base {base} inexistente")
        feats = set(spec.get("drop", []))
        miss = feats - set(S[base][1])
        if miss:
            raise KeyError(f"ablação {hname}: features ausentes {sorted(miss)}")
        drop = feats | set(fd.loc[fd["block"].isin(spec.get("drop_blocks", [])), "feature"])
        S[f"abl_{hname}"] = (S[base][0], [c for c in S[base][1] if c not in drop])
        ABL_BASE[f"abl_{hname}"] = base
        pairs.append((base, f"abl_{hname}"))
    pairs += [tuple(c) for c in abl.get("contrasts", [])]

    log.info(f"[specs] agg={len(agg)} | A={len(A)} | C={len(C)} | "
             f"volume em C: {sorted(set(C) & set(fd.loc[fd.is_volume.astype(bool), 'feature']))}")
    assert "level" not in C and not (set(C) & set(agg) - {"mean_cost"}), "C contém volume"
    return S, pairs


# ================= Carga =================
def load(inp, needed):
    p = Path(inp) / "features.parquet"
    meta = pd.read_parquet(p, columns=["match_id", "puuid", "placement", "top4",
                                       "patch", "level", "split"])
    pf, blocks = pq.ParquetFile(p), []
    for i in range(0, len(needed), 100):
        ch = pf.read(columns=needed[i:i + 100]).to_pandas().to_numpy(np.float32)
        blocks.append(sparse.csr_matrix(ch)); del ch; gc.collect()
    X = sparse.hstack(blocks, format="csc", dtype=np.float32); del blocks; gc.collect()
    log.info(f"[load] {X.shape} | nnz={X.nnz:,} | patches="
             f"{meta.groupby('split', observed=True)['patch'].unique().to_dict()}")
    return meta, X, {c: i for i, c in enumerate(needed)}


# ================= Modelos =================
def build(name, p, n_rows):
    if name == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "logreg":
        return make_pipeline(MaxAbsScaler(), LogisticRegression(
            penalty="l1", solver="saga", max_iter=CFG["LOGREG_MAX_ITER"],
            tol=CFG["LOGREG_TOL"], random_state=SEED, **p))
    if name == "rf":
        ms = CFG["RF_MAX_SAMPLES"]
        ms = max(1, int(ms * n_rows)) if ms <= 1 else min(int(ms), n_rows)
        return RandomForestClassifier(n_estimators=CFG["RF_N_TREES"], max_samples=ms,
                                      bootstrap=True, n_jobs=-1, random_state=SEED, **p)
    if name == "lgbm":
        q = {k: v for k, v in p.items() if k != "n_estimators"}
        return lgb.LGBMClassifier(n_estimators=CFG["LGBM_MAX_TREES"], random_state=SEED,
                                  n_jobs=N_JOBS, verbose=-1, subsample_freq=1, max_bin=63,
                                  force_col_wise=True, **q)
    raise ValueError(name)


def space(name, t):
    if name == "logreg":
        return {"C": t.suggest_float("C", 1e-4, 1.0, log=True)}
    if name == "rf":
        return {"max_depth": t.suggest_int("max_depth", 12, 32),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 100, log=True),
                "max_features": t.suggest_categorical("max_features", ["sqrt", 0.1, 0.3])}
    return {"learning_rate": t.suggest_float("learning_rate", *CFG["LGBM_LR_RANGE"], log=True),
            "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 20, 500, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}


def dense_if_fits(X):
    if not sparse.issparse(X):
        return np.ascontiguousarray(X, dtype=np.float32)
    gb = X.shape[0] * X.shape[1] * 4 / 1e9
    if gb <= CFG["RF_DENSE_MAX_GB"]:
        return X.toarray()
    log.warning(f"[dense] {gb:.1f} GB > limite; RF em esparso")
    return X


def fit(name, p, X, y, g=None, Xv=None, yv=None):
    if name == "logit":
        X = X.toarray()
    if name == "rf":
        X = dense_if_fits(X)
    if name == "lgbm":
        if Xv is None:   # early stopping com holdout de partidas DENTRO do treino
            tr, va = next(GroupShuffleSplit(1, test_size=CFG["LGBM_ES_FRAC"],
                                            random_state=SEED).split(X, y, g))
            X, Xv, y, yv = X[tr], X[va], y[tr], y[va]
        m = build(name, p, X.shape[0])
        m.fit(X, y, **_lgbm_eval_kwargs(Xv, yv), eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(CFG["LGBM_ES_ROUNDS"], verbose=False)])
        bi = int(m.best_iteration_ or CFG["LGBM_MAX_TREES"])
        return m, {"n_iter": bi, "hit_cap": bi >= CFG["LGBM_MAX_TREES"]}
    m = build(name, p, X.shape[0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(X, y)
    diag = {}
    if name == "logreg":
        lr = m[-1]
        conv = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        diag = {"n_iter": int(np.max(lr.n_iter_)), "converged": conv,
                "nnz_coef": int((lr.coef_ != 0).sum()), "n_coef": int(lr.coef_.size)}
        (log.info if conv else log.warning)(
            f"[fit] logreg C={p.get('C'):.2e} | n_iter={diag['n_iter']} | "
            f"convergiu={conv} | não nulos={diag['nnz_coef']}/{diag['n_coef']}")
    if name == "rf":
        diag = {"max_samples": m.max_samples, "n_trees": m.n_estimators}
    return m, diag


def proba(m, X, chunk=100_000):
    if isinstance(m, RandomForestClassifier) and sparse.issparse(X):
        return np.concatenate([m.predict_proba(X[i:i + chunk].toarray())[:, 1]
                               for i in range(0, X.shape[0], chunk)]).astype(np.float64)
    if hasattr(m, "steps") and isinstance(m[0], StandardScaler):
        X = X.toarray()
    return m.predict_proba(X)[:, 1].astype(np.float64)


def group_sample(g, frac, rng):
    if frac >= 1:
        return np.arange(len(g))
    u = np.unique(g)
    return np.flatnonzero(np.isin(g, rng.choice(u, int(len(u) * frac), replace=False)))


def tune(name, X, y, g, a, rng):
    if name == "logit":
        return {}, np.nan
    n_trials = a.trials_slow if name in SLOW_MODELS else a.trials
    s = group_sample(g, a.tune_frac, rng)
    X, y, g = X[s], y[s], g[s]
    if name == "rf":
        X = dense_if_fits(X)
    cv = ([next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, g))]
          if name == "lgbm" else list(GroupKFold(a.inner).split(X, y, g)))

    def obj(t):
        p, losses = space(name, t), []
        for i, (tr, va) in enumerate(cv):
            m, _ = fit(name, p, X[tr], y[tr], Xv=X[va], yv=y[va])
            losses.append(log_loss(y[va], clip(proba(m, X[va]))))
            t.report(float(np.mean(losses)), i)
            if t.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses))

    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    st.optimize(obj, n_trials=n_trials, timeout=a.tune_timeout_min * 60, gc_after_trial=True)
    log.info(f"    [tune] {name}: {len(st.trials)} trials | loss={st.best_value:.5f} | "
             f"{st.best_params}")
    return dict(st.best_params), float(st.best_value)


# ================= Métricas =================
def match_top4_acc(y, p, g):
    d = pd.DataFrame({"y": y, "p": p, "g": g})
    if (d.groupby("g").size() != 8).any():
        return np.nan
    pred = d.groupby("g")["p"].rank(ascending=False, method="first") <= 4
    return float((pred == d["y"].astype(bool)).mean())


def ranking_metrics(p, placement, g):
    """Spearman(rank previsto, colocação) e NDCG@4 (relevância = 9 - colocação) por partida."""
    d = pd.DataFrame({"g": g, "p": p, "pl": placement})
    d = d[d.groupby("g")["p"].transform("size").to_numpy() == 8]
    if d.empty:
        return {"spearman_match": np.nan, "ndcg4_match": np.nan}
    rp = d.groupby("g")["p"].rank(ascending=False, method="average")
    rho = 1 - 6 * ((rp - d["pl"]) ** 2).groupby(d["g"]).sum() / (8 * 63)
    pos = d.groupby("g")["p"].rank(ascending=False, method="first")
    gain = (2.0 ** (9 - d["pl"]) - 1) / np.log2(pos + 1)
    dcg = gain[pos <= 4].groupby(d["g"][pos <= 4]).sum()
    idcg = sum((2.0 ** (9 - k) - 1) / np.log2(k + 1) for k in range(1, 5))
    return {"spearman_match": float(rho.mean()), "ndcg4_match": float((dcg / idcg).mean())}


def metrics(y, p, g, placement):
    p = clip(p)
    return {"auc": roc_auc_score(y, p), "logloss": log_loss(y, p),
            "brier": brier_score_loss(y, p), "f1": f1_score(y, p >= 0.5),
            "match_top4_acc": match_top4_acc(y, p, g), **ranking_metrics(p, placement, g)}


def delong_components(y, p):
    pos = y == 1
    px, py = p[pos], p[~pos]
    m, n = len(px), len(py)
    tz = rankdata(np.concatenate([px, py]))
    return ((tz[:m].sum() - m * (m + 1) / 2) / (m * n),
            (tz[:m] - rankdata(px)) / n, 1.0 - (tz[m:] - rankdata(py)) / m)


def delong_test(c1, c2):
    (a1, x1, y1), (a2, x2, y2) = c1, c2
    S = np.cov(np.vstack([x1, x2])) / len(x1) + np.cov(np.vstack([y1, y2])) / len(y1)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return float(z), float(2 * norm.sf(abs(z)))


def bootstrap(y, preds, g, n_boot, seed):
    rng = np.random.default_rng(seed)
    gc_ = pd.factorize(g)[0]; n_m = gc_.max() + 1
    y = y.astype(np.float64)
    pre = {k: (np.argsort(clip(p), kind="stable"),
               -(y * np.log(clip(p)) + (1 - y) * np.log(1 - clip(p))), (clip(p) - y) ** 2)
           for k, p in preds.items()}
    out = {k: {"auc": [], "logloss": [], "brier": []} for k in preds}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_m, n_m), minlength=n_m)[gc_].astype(np.float64)
        W = w.sum()
        for k, (o, ll, br) in pre.items():
            wo, yo = w[o], y[o]
            wneg = wo * (1 - yo)
            cneg = np.cumsum(wneg) - wneg
            out[k]["auc"].append((wo * yo * cneg).sum() / ((wo * yo).sum() * wneg.sum()))
            out[k]["logloss"].append((w * ll).sum() / W)
            out[k]["brier"].append((w * br).sum() / W)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def paired_cv(fa, fb, n_tr, n_te):
    """Δ fold a fold com t corrigido (Nadeau & Bengio, 2003): var·(1/k + n_te/n_tr)."""
    d = np.asarray(fa) - np.asarray(fb); k = len(d)
    se = np.sqrt(d.var(ddof=1) * (1 / k + n_te / n_tr))
    tq = student_t.ppf(0.975, k - 1)
    tstat = d.mean() / se if se > 0 else 0.0
    return d.mean(), d.mean() - tq * se, d.mean() + tq * se, float(2 * student_t.sf(abs(tstat), k - 1))


# ================= Execução de uma especificação =================
def run_spec(rn, model, cols, Xall, colidx, y, g, tr_idx, te_idx, a, out, fixed=None):
    rng = np.random.default_rng(SEED)            # semente por spec (independe da ordem)
    X = Xall[:, [colidx[c] for c in cols]].tocsr()
    Xtr, ytr, gtr = X[tr_idx], y[tr_idx], g[tr_idx]
    oof, folds = np.zeros(len(tr_idx)), []
    save_folds = model in SAVE_FOLD_MODELS and not rn.startswith("abl_")
    for k, (ftr, fte) in enumerate(GroupKFold(a.outer).split(Xtr, ytr, gtr)):
        tf = time.time()
        bp, loss = ((fixed["folds"][k], np.nan) if fixed
                    else tune(model, Xtr[ftr], ytr[ftr], gtr[ftr], a, rng))
        m, diag = fit(model, bp, Xtr[ftr], ytr[ftr], g=gtr[ftr])
        oof[fte] = proba(m, Xtr[fte])
        pc = clip(oof[fte])
        folds.append({"spec": rn, "fold": k, "auc": roc_auc_score(ytr[fte], pc),
                      "logloss": log_loss(ytr[fte], pc), "n_tr": len(ftr), "n_te": len(fte),
                      "inner_loss": loss, "params": json.dumps(bp),
                      "inherited": bool(fixed), **diag})
        if save_folds:
            joblib.dump({"model": m, "features": cols, "fold": k,
                         "test_rows_global": tr_idx[fte]},
                        out / "models" / f"fold_{rn}_{k}.joblib", compress=3)
        log.info(f"  [{rn}] fold {k + 1}/{a.outer}: AUC={folds[-1]['auc']:.4f} | "
                 f"árvores={diag.get('n_iter', '—')} | {(time.time() - tf) / 60:.1f} min | "
                 f"total {elapsed():.0f} min")
        del m; gc.collect()
    # Modelo final: tuning no treino inteiro (ou herdado) -> avaliação única no 16.5
    bp, loss = (fixed["final"], np.nan) if fixed else tune(model, Xtr, ytr, gtr, a, rng)
    m, diag = fit(model, bp, Xtr, ytr, g=gtr)
    pte = proba(m, X[te_idx])
    joblib.dump({"model": m, "features": cols, "params": bp, "fit_scope": "train_only"},
                out / "models" / f"final_{rn}.joblib", compress=3)
    log.info(f"[{rn}] OOF AUC={roc_auc_score(ytr, oof):.4f} | "
             f"16.5 AUC={roc_auc_score(y[te_idx], pte):.4f} | {len(cols)} feats")
    del m, X, Xtr; gc.collect()
    return oof, pte, folds, {"params": bp, "inner_loss": loss, "n_features": len(cols), **diag}


def player_disjoint(Xall, colidx, cols, params, meta, y, g, tr_idx, seed):
    """Sensibilidade por PUUID: teste = jogadores sorteados; treino = partidas SEM esses
    jogadores (disjunto em partida E jogador). Controle: holdout por partida, mesmo tamanho."""
    rng = np.random.default_rng(seed)
    X = Xall[:, [colidx[c] for c in cols]].tocsr()[tr_idx]
    yt, gt = y[tr_idx], g[tr_idx]
    pu = meta["puuid"].to_numpy()[tr_idx]
    u = pd.unique(pu[pd.notna(pu)])
    test_p = set(rng.choice(u, int(0.2 * len(u)), replace=False))
    is_te = np.array([x in test_p for x in pu])
    touched = pd.Series(is_te).groupby(gt).transform("any").to_numpy()
    res = {}
    m, _ = fit("lgbm", params, X[~touched], yt[~touched], g=gt[~touched])
    res["player_disjoint_auc"] = roc_auc_score(yt[is_te], proba(m, X[is_te]))
    res["player_disjoint_n_train"], res["player_disjoint_n_test"] = int((~touched).sum()), int(is_te.sum())
    um = np.unique(gt)
    te_m = rng.choice(um, int(len(um) * is_te.mean()), replace=False)
    hte = np.isin(gt, te_m)
    m, _ = fit("lgbm", params, X[~hte], yt[~hte], g=gt[~hte])
    res["match_holdout_auc"] = roc_auc_score(yt[hte], proba(m, X[hte]))
    res["delta_auc_player_minus_match"] = res["player_disjoint_auc"] - res["match_holdout_auc"]
    log.info(f"[puuid] {res}")
    return res


def lambdamart(Xall, colidx, cols, params, y, g, pl, tr_idx, te_idx, p_ref, out):
    """LambdaMART (relevância = 8 − colocação) com os hiperparâmetros do lgbm_A; avaliado no 16.5."""
    t0 = time.time()
    X = Xall[:, [colidx[c] for c in cols]].tocsr()
    rel = (8 - pl).astype(np.int32)

    def grp(idx):
        idx = idx[np.argsort(g[idx], kind="stable")]
        return idx, np.unique(g[idx], return_counts=True)[1]

    tr, va = next(GroupShuffleSplit(1, test_size=CFG["LGBM_ES_FRAC"], random_state=SEED)
                  .split(tr_idx, groups=g[tr_idx]))
    itr, gtr = grp(tr_idx[tr]); iva, gva = grp(tr_idx[va])
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=CFG["LGBM_MAX_TREES"],
                       random_state=SEED, n_jobs=N_JOBS, verbose=-1, subsample_freq=1, max_bin=63,
                       force_col_wise=True, **{k: v for k, v in params.items() if k != "n_estimators"})
    ev = ({"eval_X": (X[iva],), "eval_y": (rel[iva],)}
          if "eval_X" in inspect.signature(lgb.LGBMRanker.fit).parameters
          else {"eval_set": [(X[iva], rel[iva])]})
    m.fit(X[itr], rel[itr], group=gtr, eval_group=[gva], eval_at=[4], **ev,
          callbacks=[lgb.early_stopping(CFG["LGBM_ES_ROUNDS"], verbose=False)])
    p = 1 / (1 + np.exp(-np.clip(m.predict(X[te_idx]), -30, 30)))
    yt, gt, pt = y[te_idx], g[te_idx], pl[te_idx]
    r = {"n_iter": int(m.best_iteration_ or 0), "auc": roc_auc_score(yt, p),
         "match_top4_acc": match_top4_acc(yt, p, gt), **ranking_metrics(p, pt, gt)}
    _, pv = delong_test(delong_components(yt, clip(p)), delong_components(yt, clip(p_ref)))
    ref = ranking_metrics(p_ref, pt, gt)
    r.update({"delta_auc_vs_lgbm_A": r["auc"] - roc_auc_score(yt, p_ref), "delong_p": pv,
              "lgbm_A_spearman": ref["spearman_match"], "lgbm_A_ndcg4": ref["ndcg4_match"],
              "minutes": (time.time() - t0) / 60})
    json.dump(r, open(out / "lambdamart.json", "w"), indent=2, default=float)
    log.info(f"[lambdamart] {r}")


def inherited_params(rn, folds, final, a):
    """Hiperparâmetros da spec-base, fold a fold (None => tunar)."""
    if rn not in ABL_BASE or a.abl_retune:
        return None
    b = ABL_BASE[rn]
    fb = sorted((f for f in folds if f["spec"] == b), key=lambda f: f["fold"])
    if b not in final or len(fb) != a.outer:
        log.warning(f"[{rn}] base {b} não rodou nesta execução: tunando do zero")
        return None
    log.info(f"[{rn}] herdando hiperparâmetros de {b}")
    return {"folds": [json.loads(f["params"]) for f in fb], "final": final[b]["params"]}


# ================= Main =================
def main(a):
    out = Path(a.out); (out / "models").mkdir(parents=True, exist_ok=True)
    CFG.update({"LOGREG_MAX_ITER": a.logreg_max_iter, "RF_N_TREES": a.rf_trees,
                "RF_MAX_SAMPLES": a.rf_max_samples,
                "LGBM_LR_RANGE": (a.lgbm_lr_min, 0.20),
                "LGBM_ES_ROUNDS": a.es_rounds})
    log.info(f"[main] {SCRIPT_VERSION} | args={vars(a)} | CFG={CFG}")

    fd = pd.read_csv(Path(a.inp) / "feature_dict.csv").drop_duplicates("feature")
    fd = fd[~fd["block"].isin(split_arg(a.drop_blocks)) &
            ~fd["feature"].isin(split_arg(a.drop_agg))]
    abl = json.load(open(a.ablations)) if a.ablations else {}
    specs, pairs = build_specs(fd, split_arg(a.models), abl)
    if a.only:
        specs = {k: v for k, v in specs.items() if k in set(split_arg(a.only))}
    pairs = [p for p in pairs if p[0] in specs and p[1] in specs]
    needed = sorted({c for _, cols in specs.values() for c in cols} | {"level"})
    meta, Xall, colidx = load(a.inp, needed)

    if a.min_level:
        keep = (meta["level"] >= a.min_level).to_numpy()
        meta, Xall = meta[keep].reset_index(drop=True), Xall[keep]
        log.info(f"[main] level >= {a.min_level}: {keep.sum():,} linhas")
    if a.placements:   # modelo restrito à fronteira (ex.: 3,6)
        lo, hi = map(int, a.placements.split(","))
        keep = meta["placement"].between(lo, hi).to_numpy()
        meta, Xall = meta[keep].reset_index(drop=True), Xall[keep]
        log.info(f"[main] colocações {lo}–{hi}: {keep.sum():,} linhas")

    y = meta["top4"].to_numpy(np.int8)
    g = pd.factorize(meta["match_id"])[0]
    pl = meta["placement"].to_numpy()
    tr_idx = np.flatnonzero(meta["split"].astype(str).to_numpy() == "train")
    te_idx = np.flatnonzero(meta["split"].astype(str).to_numpy() == "test")
    assert set(meta["patch"].astype(str).iloc[te_idx]) == {"16.5"}
    assert not set(meta["match_id"].iloc[tr_idx]) & set(meta["match_id"].iloc[te_idx])
    log.info(f"[main] treino={len(tr_idx):,} | teste(16.5)={len(te_idx):,} | {len(specs)} specs")

    json.dump({"script": SCRIPT_VERSION, "args": vars(a), "cfg": CFG, "specs":
               {k: {"model": m, "n_features": len(c), "inherits": ABL_BASE.get(k)}
                for k, (m, c) in specs.items()},
               "pairs": pairs, "n_train": len(tr_idx), "n_test": len(te_idx),
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__,
                            "sklearn": sklearn.__version__, "lightgbm": lgb.__version__,
                            "optuna": optuna.__version__,
                            "lightgbm_eval_api": "eval_X/eval_y" if _LGBM_NEW_EVAL else "eval_set"
                            }},
              open(out / "run_config.json", "w"), indent=2, default=str)

    # specs-base antes das ablações (garante herança)
    order = sorted(specs, key=lambda k: k.startswith("abl_"))
    oof, pte, folds, final = {}, {}, [], {}
    for rn in order:
        model, cols = specs[rn]
        log.info(f"== {rn} | {model} | {len(cols)} feats | decorrido {elapsed():.0f} min")
        fixed = inherited_params(rn, folds, final, a)
        oof[rn], pte[rn], f, final[rn] = run_spec(rn, model, cols, Xall, colidx, y, g,
                                                  tr_idx, te_idx, a, out, fixed=fixed)
        folds += f
        pd.DataFrame(folds).to_csv(out / "folds.csv", index=False)   # checkpoint
    fdf = pd.DataFrame(folds)

    # ---- Métricas: OOF (treino) e 16.5 (out-of-time) com IC por bootstrap de partidas ----
    bt = bootstrap(y[te_idx], pte, g[te_idx], a.n_boot, SEED)
    rows_cv, rows_te = [], []
    for rn in specs:
        rows_cv.append({"model": rn, **metrics(y[tr_idx], oof[rn], g[tr_idx], pl[tr_idx]),
                        "auc_fold_mean": fdf.loc[fdf.spec == rn, "auc"].mean(),
                        "auc_fold_sd": fdf.loc[fdf.spec == rn, "auc"].std(ddof=1)})
        r = {"model": rn, "test_patch": "16.5",
             **metrics(y[te_idx], pte[rn], g[te_idx], pl[te_idx])}
        for mt in ("auc", "logloss", "brier"):
            r[f"{mt}_lo"], r[f"{mt}_hi"] = np.percentile(bt[rn][mt], [2.5, 97.5])
        rows_te.append(r)
    pd.DataFrame(rows_cv).sort_values("auc", ascending=False).to_csv(out / "metrics_cv.csv", index=False)
    tdf = pd.DataFrame(rows_te).sort_values("auc", ascending=False)
    tdf.to_csv(out / "temporal.csv", index=False)
    tdf.to_csv(out / "metrics.csv", index=False)   # o 05 v10 lê métricas do 16.5 com IC
    log.info("\n16.5 (out-of-time):\n" + tdf.round(4).to_string(index=False))

    # ---- Comparações pareadas (Parecer 1.3 e 1.5) ----
    dl = {k: delong_components(y[te_idx], clip(p)) for k, p in pte.items()}
    comp = []
    for m1, m2 in pairs:
        f1, f2 = fdf[fdf.spec == m1].sort_values("fold"), fdf[fdf.spec == m2].sort_values("fold")
        n_tr, n_te = f1["n_tr"].mean(), f1["n_te"].mean()
        for mt in ("auc", "logloss"):
            d, lo, hi, p = paired_cv(f1[mt].to_numpy(), f2[mt].to_numpy(), n_tr, n_te)
            db = bt[m1][mt] - bt[m2][mt]
            row = {"m1": m1, "m2": m2, "metric": mt, "cv_diff": d, "cv_lo": lo, "cv_hi": hi,
                   "cv_p": p, "test_diff": float(db.mean()), "test_lo": db.quantile(.025),
                   "test_hi": db.quantile(.975),
                   "test_boot_p": min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()))}
            if mt == "auc":
                row["delong_z"], row["delong_p"] = delong_test(dl[m1], dl[m2])
            comp.append(row)
    cdf = pd.DataFrame(comp)
    for col in ("cv_p", "test_boot_p", "delong_p"):
        if col in cdf:
            cdf[f"{col}_holm"] = np.nan
            for _, idx in cdf.groupby("metric").groups.items():
                cdf.loc[idx, f"{col}_holm"] = holm(cdf.loc[idx, col])
    cdf.to_csv(out / "comparisons.csv", index=False)
    if len(cdf):
        log.info("\nComparações (ΔAUC):\n" + cdf[cdf.metric == "auc"][
            ["m1", "m2", "cv_diff", "cv_lo", "cv_hi", "test_diff", "test_lo", "test_hi"]
        ].round(4).to_string(index=False))

    # ---- Sensibilidade de dependência por jogador (Parecer 2.1) ----
    if "lgbm_A" in specs and not a.skip_puuid:
        sens = player_disjoint(Xall, colidx, specs["lgbm_A"][1], final["lgbm_A"]["params"],
                               meta, y, g, tr_idx, SEED)
        seen = set(meta["puuid"].iloc[tr_idx].dropna())
        new = ~meta["puuid"].iloc[te_idx].isin(seen).to_numpy()
        if new.sum() > 100 and len(np.unique(y[te_idx][new])) == 2:
            sens["test_unseen_players_auc"] = roc_auc_score(y[te_idx][new], pte["lgbm_A"][new])
            sens["test_unseen_players_n"] = int(new.sum())
        json.dump(sens, open(out / "puuid_sensitivity.json", "w"), indent=2, default=float)

    if a.lambdamart and "lgbm_A" in specs:
        lambdamart(Xall, colidx, specs["lgbm_A"][1], final["lgbm_A"]["params"], y, g, pl,
                   tr_idx, te_idx, pte["lgbm_A"], out)

    cols_meta = ["match_id", "puuid", "placement", "top4", "patch", "level"]
    o = meta.iloc[tr_idx][cols_meta].copy()
    for k, p in oof.items(): o[f"p_{k}"] = p.astype(np.float32)
    o.to_parquet(out / "oof_predictions.parquet", index=False)
    t = meta.iloc[te_idx][cols_meta].copy()
    for k, p in pte.items(): t[f"p_{k}"] = p.astype(np.float32)
    t.to_parquet(out / "test_predictions.parquet", index=False)
    json.dump(final, open(out / "best_params.json", "w"), indent=2, default=str)
    log.info(f"[main] saídas em {out.resolve()} | total {elapsed():.0f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="logreg,rf,lgbm")
    ap.add_argument("--only", default="", help="subconjunto de specs (vírgula)")
    ap.add_argument("--ablations", default="", help="JSON {specs: {nome: {base, drop, drop_blocks}}, contrasts}")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--trials-slow", type=int, default=10, help="trials para rf e logreg")
    ap.add_argument("--tune-timeout-min", type=float, default=20)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--tune-frac", type=float, default=0.25)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--logreg-max-iter", type=int, default=5000)
    ap.add_argument("--rf-trees", type=int, default=300)
    ap.add_argument("--rf-max-samples", type=float, default=0.20, help="<=1 fração; >1 absoluto")
    ap.add_argument("--min-level", type=int, default=0)
    ap.add_argument("--drop-agg", default="")
    ap.add_argument("--drop-blocks", default="")
    ap.add_argument("--skip-puuid", action="store_true")
    ap.add_argument("--abl-retune", action="store_true", help="re-tunar ablações (padrão: herdar da base)")
    ap.add_argument("--lgbm-lr-min", type=float, default=0.03, help="piso do learning_rate no tuning")
    ap.add_argument("--es-rounds", type=int, default=200, help="paciência do early stopping")
    ap.add_argument("--placements", default="", help="faixa de colocações, ex.: 3,6")
    ap.add_argument("--lambdamart", action="store_true", help="LambdaMART com params do lgbm_A")
    main(ap.parse_args())
