"""
02b_features.py — Arquétipos de composição (Seção 2.4 do TCC).

- MiniBatchKMeans sobre vetores binários de traits e unidades. Em dados binários, a
  distância euclidiana ao quadrado é igual à distância de Hamming.
- Os centróides, os nomes e as taxas de referência são ajustados SÓ no treino
  (Parecer 1.4). O teste (16.5) apenas recebe o rótulo via predict.
- Não usa o rótulo (top4) para formar os clusters.
- Caracteriza cada cluster: nível, n_units, board_value e intensidade (Parecer 2.3).
- Estabilidade: ARI entre sementes para K ∈ {8, 12, 16} (Parecer, Etapa 2).
- Sempre gera exatamente K colunas one-hot. A escrita é atômica.
"""
import argparse
import json
import logging
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

SEED = 42
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("clusters")
PROFILE_COLS = ["level", "n_units", "board_value", "board_value_per_unit",
                "full_items_per_unit", "share_2star", "n_full_items"]


def get_cluster_name(centroid, cols):
    s = pd.Series(centroid, index=cols)
    traits = s[s.index.str.startswith("trait_")].nlargest(2)
    units = s[s.index.str.startswith("unit_")].nlargest(1)
    parts = ([t.replace("trait_", "") for t in traits.index if traits[t] > 0.1] +
             [u.replace("unit_", "") for u in units.index if units[u] > 0.1])
    return "Mix" if not parts else "_".join(parts)


def top4_ci(y, g, n_boot=500, seed=SEED):
    """IC 95% por bootstrap de partidas (os 8 jogadores de uma partida são dependentes)."""
    if len(y) == 0:
        return np.nan, np.nan
    d = pd.DataFrame({"y": y, "g": g}).groupby("g")["y"].agg(["sum", "count"])
    s, c = d["sum"].to_numpy(), d["count"].to_numpy()
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(s), (n_boot, len(s)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def stability(X, ks, seeds, batch):
    rows = []
    for k in ks:
        labs = [MiniBatchKMeans(n_clusters=k, random_state=s, batch_size=batch,
                                n_init=3).fit_predict(X) for s in seeds]
        aris = [adjusted_rand_score(labs[i], labs[j])
                for i, j in combinations(range(len(seeds)), 2)]
        rows.append({"k": k, "ari_mean": np.mean(aris), "ari_sd": np.std(aris, ddof=1),
                     "ari_min": np.min(aris), "n_pairs": len(aris)})
        log.info(f"  estabilidade K={k}: ARI={np.mean(aris):.3f} ± {np.std(aris, ddof=1):.3f}")
    return pd.DataFrame(rows)


def main(inp, out_report, k, batch_size, n_boot, train_patches, ks, n_seeds, n_sub):
    inp, rep = Path(inp), Path(out_report)
    rep.mkdir(parents=True, exist_ok=True)
    fp = inp / "features.parquet"

    fd = pd.read_csv(inp / "feature_dict.csv")
    fd = fd[fd["block"] != "archetype"]
    cluster_cols = fd[fd["block"].isin(["trait", "unit"])]["feature"].tolist()
    if not cluster_cols:
        raise ValueError("Nenhuma feature de trait ou unit para clusterizar.")

    tbl = pq.read_table(fp)
    tbl = tbl.select([c for c in tbl.column_names if not c.startswith("archetype_")])
    patch = tbl["patch"].to_pandas().astype(str).to_numpy()
    is_tr = np.isin(patch, list(train_patches))
    log.info(f"treino={is_tr.sum():,} | teste={(~is_tr).sum():,} | {len(cluster_cols)} colunas")

    X = np.column_stack([(tbl[c].to_numpy() > 0) for c in cluster_cols]).astype(np.float32)
    y = tbl["top4"].to_numpy()
    g = tbl["match_id"].to_numpy()
    prof = {c: tbl[c].to_numpy() for c in PROFILE_COLS if c in tbl.column_names}

    # ---- Ajuste SÓ no treino, predição em tudo ----
    km = MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=batch_size, n_init=3)
    km.fit(X[is_tr])
    labels = km.predict(X)
    np.save(rep / f"kmeans_centroids_k{k}.npy", km.cluster_centers_)

    names, rows = {}, []
    for c in range(k):
        mtr, mte = (labels == c) & is_tr, (labels == c) & ~is_tr
        names[c] = f"archetype_{get_cluster_name(X[mtr].mean(0) if mtr.any() else km.cluster_centers_[c], cluster_cols)}_{c}"
        lo, hi = top4_ci(y[mtr], g[mtr], n_boot)
        row = {"cluster_id": c, "archetype_name": names[c],
               "n_train": int(mtr.sum()), "share_train": mtr.sum() / is_tr.sum(),
               "top4_train": y[mtr].mean() if mtr.any() else np.nan,
               "top4_lo": lo, "top4_hi": hi,
               "diff_vs_50": bool(lo > 0.5 or hi < 0.5),
               "n_test": int(mte.sum()),
               "top4_test": y[mte].mean() if mte.any() else np.nan}
        for pc, v in prof.items():                    # perfil (Parecer 2.3, cluster Vi)
            row[f"mean_{pc}"] = float(v[mtr].mean()) if mtr.any() else np.nan
        rows.append(row)
        log.info(f"  {names[c]}: n_tr={row['n_train']:,} | top4={row['top4_train']:.3f} "
                 f"[{lo:.3f}; {hi:.3f}] | lvl={row.get('mean_level', np.nan):.2f} | "
                 f"bv/u={row.get('mean_board_value_per_unit', np.nan):.2f}")
    rdf = pd.DataFrame(rows).sort_values("top4_train", ascending=False)
    rdf.round(4).to_csv(rep / "archetypes_report.csv", index=False)

    # ---- Estabilidade (subamostra do treino) ----
    rng = np.random.default_rng(SEED)
    tr_idx = np.flatnonzero(is_tr)
    sub = rng.choice(tr_idx, min(n_sub, len(tr_idx)), replace=False)
    stab = stability(X[sub], ks, list(range(n_seeds)), batch_size)
    stab.round(4).to_csv(rep / "archetypes_stability.csv", index=False)
    del X

    # ---- One-hot com exatamente K colunas e escrita atômica ----
    onehot = (labels[:, None] == np.arange(k)[None, :]).astype(np.int8)
    for c in range(k):
        tbl = tbl.append_column(names[c], pa.array(onehot[:, c]))
    tmp = fp.with_suffix(".tmp.parquet")
    pq.write_table(tbl, tmp, compression="zstd", row_group_size=500_000)
    os.replace(tmp, fp)

    fd = pd.concat([fd, pd.DataFrame([{
        "feature": names[c], "block": "archetype",
        "freq_train": float(onehot[is_tr, c].mean()),
        "is_volume": False, "in_scenario_c": True} for c in range(k)])], ignore_index=True)
    fd.to_csv(inp / "feature_dict.csv", index=False)

    with open(rep / "archetypes_meta.json", "w") as fh:
        json.dump({"k": k, "seed": SEED, "fit_scope": "train_only",
                   "train_patches": sorted(train_patches), "n_cluster_cols": len(cluster_cols),
                   "inertia_train": float(km.inertia_),
                   "stability": stab.to_dict("records")}, fh, indent=2, default=float)
    log.info(f"Concluído: {k} arquétipos adicionados (ajuste somente no treino).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed")
    ap.add_argument("--out-report", default="report")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--batch", type=int, default=10_000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--train-patches", default="16.1,16.2,16.3,16.4")
    ap.add_argument("--stab-ks", default="8,12,16")
    ap.add_argument("--stab-seeds", type=int, default=5)
    ap.add_argument("--stab-sub", type=int, default=200_000)
    a = ap.parse_args()
    main(a.inp, a.out_report, a.k, a.batch, a.n_boot, set(a.train_patches.split(",")),
         [int(x) for x in a.stab_ks.split(",")], a.stab_seeds, a.stab_sub)
