"""
02b_clusters.py — Identificação de arquétipos de composição (Seção 2.4 do TCC).

- Agrupa os vetores binários de traits e unidades (MiniBatchKMeans).
  [v6] Em vetores binários, a distância euclidiana ao quadrado é igual à distância de Hamming
  (nº de traits/unidades diferentes). Isso é declarado no texto como alternativa escalável
  ao Jaccard/HDBSCAN.
- Clusterização não supervisionada: o rótulo (top4) NÃO é usado para formar os clusters.
- Taxa de Top 4 por cluster com IC 95% por bootstrap de partidas [v6].
- Insere o cluster como atributo one-hot.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("clusters")


def get_cluster_name(centroid, cols):
    s = pd.Series(centroid, index=cols)
    traits = s[s.index.str.startswith("trait_")].nlargest(2)
    units = s[s.index.str.startswith("unit_")].nlargest(1)
    t_names = [t.replace("trait_", "") for t in traits.index if traits[t] > 0.1]
    u_names = [u.replace("unit_", "") for u in units.index if units[u] > 0.1]
    parts = t_names + u_names
    return "Mix" if not parts else "_".join(parts)


def top4_ci(y, g, n_boot=500, seed=42):   # [v6] bootstrap por partida (8 jogadores dependentes)
    d = pd.DataFrame({"y": y, "g": g}).groupby("g")["y"].agg(["sum", "count"])
    s, c = d["sum"].to_numpy(), d["count"].to_numpy()
    r = np.random.default_rng(seed)
    b = [s[i].sum() / c[i].sum() for i in (r.integers(0, len(s), len(s)) for _ in range(n_boot))]
    return np.percentile(b, [2.5, 97.5])


def main(inp, out_report, k, batch_size, n_boot):
    inp = Path(inp)
    Path(out_report).mkdir(parents=True, exist_ok=True)

    fd = pd.read_csv(inp / "feature_dict.csv")
    cluster_cols = fd[fd["block"].isin(["trait", "unit"])]["feature"].tolist()
    if not cluster_cols:
        raise ValueError("Nenhuma feature de trait ou unit encontrada para clusterizar.")

    log.info("Carregando base de dados parquet...")
    df = pq.read_table(inp / "features.parquet").to_pandas()

    old_clusters = fd[fd["block"] == "archetype"]["feature"].tolist()
    old_clusters += [c for c in df.columns if c.startswith("archetype_")]   # [v6] garante limpeza
    if old_clusters:
        log.info(f"Removendo {len(set(old_clusters))} colunas de clusters antigos...")
        df = df.drop(columns=list(set(old_clusters)), errors="ignore")
        fd = fd[fd["block"] != "archetype"]

    X = (df[cluster_cols] > 0).astype(np.float32)
    log.info(f"MiniBatchKMeans K={k} (sem uso do rótulo; euclidiana² = Hamming em binário)")
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=batch_size, n_init=3)
    cluster_labels = kmeans.fit_predict(X)

    names, report_rows = {}, []
    y, g = df["top4"].to_numpy(), df["match_id"].to_numpy()
    for c in range(k):
        mask = cluster_labels == c
        size = int(mask.sum())
        name = get_cluster_name(X[mask].mean(axis=0), cluster_cols)
        col_name = f"archetype_{name}_{c}"
        names[c] = col_name
        rate = float(y[mask].mean()) if size else np.nan
        lo, hi = top4_ci(y[mask], g[mask], n_boot) if size else (np.nan, np.nan)   # [v6]
        report_rows.append({"cluster_id": c, "archetype_name": col_name, "n_players": size,
                            "share": size / len(df), "top4_rate": round(rate, 4),
                            "top4_lo": round(lo, 4), "top4_hi": round(hi, 4),
                            "diff_vs_50": lo > 0.5 or hi < 0.5})
        log.info(f"  {col_name}: n={size:,} | top4={rate:.3f} [{lo:.3f}; {hi:.3f}]")

    report_df = pd.DataFrame(report_rows).sort_values("top4_rate", ascending=False)
    report_path = Path(out_report) / "archetypes_report.csv"
    report_df.to_csv(report_path, index=False)
    log.info(f"Relatório de arquétipos salvo em {report_path}")

    dummies = pd.get_dummies(cluster_labels).rename(columns=names).astype(np.int8)
    new_features = [{"feature": c, "block": "archetype", "freq": float((dummies[c] > 0).mean())}
                    for c in dummies.columns]
    fd = pd.concat([fd, pd.DataFrame(new_features)], ignore_index=True)
    fd.to_csv(inp / "feature_dict.csv", index=False)

    df_final = pd.concat([df, dummies], axis=1)
    pq.write_table(pa.Table.from_pandas(df_final, preserve_index=False),
                   inp / "features.parquet", compression="zstd")
    log.info(f"Processo concluído! {k} arquétipos adicionados aos dados.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed")
    ap.add_argument("--out-report", default="report")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=500)   # [v6]
    args = ap.parse_args()
    main(args.inp, args.out_report, args.k, args.batch, args.n_boot)
