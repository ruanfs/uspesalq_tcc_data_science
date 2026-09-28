"""
02_features.py — Matriz de features (1 linha = jogador × partida), baixo uso de memória.
Chave de junção: (match_id, pidx).

[v6] n_traits_gold_plus / n_traits_prismatic (H1) e contagem de partidas por patch.

Uso:
    python 02_features.py --inp data/interim --out data/processed --min-freq 0.005
"""
import argparse
import gc
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("features")

PREFIX_RE = re.compile(r"^TFT\d*_(?:Item_)?(?:Artifact_)?")


def short(name) -> str:
    """TFT_Item_X / TFT9_Item_X / TFT16_X / TFT_Item_Artifact_X -> X"""
    return PREFIX_RE.sub("", str(name))


COMPONENTS = {short(c) for c in (
    "TFT_Item_BFSword", "TFT_Item_RecurveBow", "TFT_Item_NeedlesslyLargeRod",
    "TFT_Item_TearOfTheGoddess", "TFT_Item_ChainVest", "TFT_Item_NegatronCloak",
    "TFT_Item_GiantsBelt", "TFT_Item_SparringGloves", "TFT_Item_Spatula",
    "TFT_Item_FryingPan",
)}
TOP_K_PAIRS = 10_000
CARRY_MIN_ITEMS = 3
S = 16                       # máx. de slots por tabuleiro (rid*S + slot)
KEY = ["match_id", "pidx"]
STYLE_GOLD, STYLE_PRISMATIC = 4, 5   # [v6] ver TRAIT_STYLE no 01_extract


class Vocab:
    """Mapeia strings (normalizadas) -> códigos int32 globais (estáveis entre lotes)."""
    def __init__(self):
        self.idx, self.names = {}, []

    def _add(self, s):
        if s not in self.idx:
            self.idx[s] = len(self.names); self.names.append(s)
        return self.idx[s]

    def encode(self, ser: pd.Series, norm=None) -> np.ndarray:
        cat = ser.astype("category")
        f = norm or (lambda x: x)
        lut = np.array([self._add(f(c)) for c in cat.cat.categories], dtype=np.int32)
        return lut[cat.cat.codes.to_numpy()]      # chamar só com valores não nulos


def batches(path, cols, bs):
    for b in pq.ParquetFile(path).iter_batches(batch_size=bs, columns=cols):
        yield b.to_pandas()


def to_rid(df, key_index):
    return key_index.get_indexer(pd.MultiIndex.from_arrays(
        [df["match_id"].astype(str).to_numpy(), df["pidx"].to_numpy(np.int16)]))


def build_sparse(rid, col, val, n, m, agg):
    rid, col, val = map(np.concatenate, (rid, col, val))
    if agg == "max":                               # dedup (rid, col) com max
        k = rid.astype(np.int64) * m + col
        s = pd.Series(val).groupby(k).max()
        k = s.index.to_numpy()
        rid, col, val = k // m, k % m, s.to_numpy()
    X = sp.csr_matrix((val.astype(np.int16), (rid, col)), shape=(n, m))  # soma duplicatas
    X.eliminate_zeros()
    return X


def main(inp, out, min_freq, bs, save_sparse):
    inp, out = Path(inp), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- Base ----------
    agg_cols = ["n_units", "sum_cost", "mean_cost", "n_1star", "n_2star",
                "n_3star", "n_items", "n_active_traits", "n_unique_traits"]
    P = pd.read_parquet(inp / "participants.parquet",
                        columns=KEY + ["puuid", "placement", "top4", "level"] + agg_cols)
    M = pd.read_parquet(inp / "matches.parquet",
                        columns=["match_id", "patch"]).drop_duplicates("match_id")
    P = P.merge(M, on="match_id", how="left"); del M
    P["match_id"] = P["match_id"].astype(str)
    P["pidx"] = P["pidx"].astype(np.int16)

    dup = P.duplicated(KEY, keep=False)
    if dup.any():
        bad = P.loc[dup, "match_id"].unique()
        log.warning(f"{int(dup.sum()):,} linhas com chave duplicada em {len(bad):,} partidas "
                    f"-> removendo. Exemplos: {list(bad[:5])}")
        P = P[~P["match_id"].isin(bad)].reset_index(drop=True)
    n = len(P)
    key_index = pd.MultiIndex.from_arrays([P["match_id"].to_numpy(), P["pidx"].to_numpy()])
    assert key_index.is_unique, "Chave duplicada em participants"
    # [v6] partidas por patch (seção "volume por patch" do documento)
    patch_counts = {str(k): int(v) for k, v in
                    P.drop_duplicates("match_id")["patch"].astype(str).value_counts().items()}
    log.info(f"{n:,} linhas-base | partidas por patch: {patch_counts}")

    A = {k: np.zeros(n, np.int16) for k in
         ("n_hi_cost", "n_hi_cost_2star", "n_hi_cost_1star", "n_carries", "n_components",
          "n_artifacts", "n_radiants", "n_emblems", "n_full_items", "n_full_items_on_hi_cost",
          "n_traits_gold_plus", "n_traits_prismatic")}   # [v6]
    A["board_value"] = np.zeros(n, np.float32)
    slot_cost = np.zeros(n * S, np.int8)
    full_per_unit = np.zeros(n * S, np.int8)

    # ---------- Units ----------
    Vu, ur, uc, uv = Vocab(), [], [], []
    for df in batches(inp / "units.parquet", KEY + ["slot", "character_id", "star", "cost"], bs):
        rid = to_rid(df, key_index)
        ok = (rid >= 0) & df["character_id"].notna().to_numpy()
        df, rid = df[ok], rid[ok]
        star = df["star"].fillna(0).to_numpy(np.int8)
        cost = df["cost"].fillna(0).to_numpy(np.int8)
        slot = df["slot"].to_numpy(np.int64)
        assert slot.max(initial=0) < S
        hi = cost >= 4
        A["n_hi_cost"] += np.bincount(rid, hi, n).astype(np.int16)
        A["n_hi_cost_2star"] += np.bincount(rid, hi & (star >= 2), n).astype(np.int16)
        A["n_hi_cost_1star"] += np.bincount(rid, hi & (star == 1), n).astype(np.int16)
        A["board_value"] += np.bincount(rid, cost * 3.0 ** (star - 1), n).astype(np.float32)
        slot_cost[rid * S + slot] = cost
        ur.append(rid.astype(np.int32)); uc.append(Vu.encode(df["character_id"], short))
        uv.append(star)
        del df
    Xu = build_sparse(ur, uc, uv, n, len(Vu.names), "max"); del ur, uc, uv; gc.collect()

    # ---------- Traits ----------
    Vt, tr, tc, tv = Vocab(), [], [], []
    for df in batches(inp / "traits.parquet", KEY + ["trait", "tier_current", "style"], bs):  # [v6]
        rid = to_rid(df, key_index)
        ok = (rid >= 0) & df["trait"].notna().to_numpy()
        df, rid = df[ok], rid[ok]
        st = df["style"].fillna(0).to_numpy(np.int8)                                    # [v6]
        A["n_traits_gold_plus"] += np.bincount(rid, st >= STYLE_GOLD, n).astype(np.int16)
        A["n_traits_prismatic"] += np.bincount(rid, st == STYLE_PRISMATIC, n).astype(np.int16)
        tr.append(rid.astype(np.int32)); tc.append(Vt.encode(df["trait"], short))
        tv.append(df["tier_current"].fillna(0).to_numpy(np.int8))
        del df
    Xt = build_sparse(tr, tc, tv, n, len(Vt.names), "max"); del tr, tc, tv; gc.collect()

    # ---------- Items ----------
    Vi, Vc, ir, ic, ich = Vocab(), Vocab(), [], [], []
    for df in batches(inp / "items.parquet",
                      KEY + ["slot", "character_id", "item", "item_category"], bs):
        rid = to_rid(df, key_index)
        ok = (rid >= 0) & df["item"].notna().to_numpy()
        df, rid = df[ok], rid[ok]
        cat = df["item_category"].astype(str).to_numpy()
        comp = df["item"].astype(str).map(short).isin(COMPONENTS).to_numpy()
        A["n_components"] += np.bincount(rid, comp, n).astype(np.int16)
        for c, k in (("artifact", "n_artifacts"), ("radiant", "n_radiants"), ("emblem", "n_emblems")):
            A[k] += np.bincount(rid, cat == c, n).astype(np.int16)

        full = ~comp
        df, rid = df[full], rid[full]
        sk = rid.astype(np.int64) * S + df["slot"].to_numpy(np.int64)
        A["n_full_items"] += np.bincount(rid, minlength=n).astype(np.int16)
        A["n_full_items_on_hi_cost"] += np.bincount(rid, slot_cost[sk] >= 4, n).astype(np.int16)
        np.add.at(full_per_unit, sk, 1)
        ir.append(rid.astype(np.int32)); ic.append(Vi.encode(df["item"], short))
        ich.append(Vc.encode(df["character_id"].fillna("NA"), short))
        del df
    del slot_cost
    A["n_carries"] = np.bincount(np.nonzero(full_per_unit >= CARRY_MIN_ITEMS)[0] // S,
                                 minlength=n).astype(np.int16)
    del full_per_unit; gc.collect()

    ir, ic, ich = map(np.concatenate, (ir, ic, ich))
    Xi = build_sparse([ir], [ic], [np.ones(len(ir), np.int8)], n, len(Vi.names), "sum")

    pcode = ich.astype(np.int64) * len(Vi.names) + ic
    del ic, ich
    uniq, cnt = np.unique(pcode, return_counts=True)
    top = uniq[np.argsort(-cnt, kind="stable")[:TOP_K_PAIRS]]
    mask = np.isin(pcode, top)
    pcol = np.searchsorted(top_sorted := np.sort(top), pcode[mask])
    Xp = build_sparse([ir[mask]], [pcol], [np.ones(mask.sum(), np.int8)], n, len(top), "sum")
    pair_names = [f"{Vc.names[c // len(Vi.names)]}__{Vi.names[c % len(Vi.names)]}"
                  for c in top_sorted]
    del ir, pcode, mask, pcol; gc.collect()

    # ---------- Filtro de frequência ----------
    blocks = [("trait_", "trait", Xt, Vt.names),
              ("unit_", "unit", Xu, Vu.names),
              ("item_", "item", Xi, Vi.names),
              ("pair_", "unit_item", Xp, pair_names)]
    kept_mats, kept_names, kept_blocks, kept_freq = [], [], [], []
    n_dropped = 0
    for prefix, bname, X, names in blocks:
        freq = X.getnnz(axis=0) / n
        keep = np.nonzero((freq >= min_freq) & (freq < 1.0))[0]
        n_dropped += X.shape[1] - len(keep)
        kept_mats.append(X[:, keep])
        kept_names += [f"{prefix}{names[j]}" for j in keep]
        kept_blocks += [bname] * len(keep)
        kept_freq += freq[keep].tolist()
    del Xt, Xu, Xi, Xp, blocks
    Xs = sp.hstack(kept_mats, format="csc"); del kept_mats; gc.collect()

    dupn = pd.Index(kept_names)[pd.Index(kept_names).duplicated()].unique()
    assert len(dupn) == 0, f"Colunas duplicadas: {list(dupn)}"

    if save_sparse:
        sp.save_npz(out / "features_sparse.npz", Xs.tocsr())
        pd.Series(kept_names).to_csv(out / "features_sparse_cols.csv", index=False, header=False)

    # ---------- Montagem colunar ----------
    cols = {}
    cols["match_id"] = pa.array(P["match_id"].to_numpy())
    cols["pidx"] = pa.array(P["pidx"].to_numpy(np.int8))
    cols["puuid"] = pa.array(P["puuid"].astype(object).to_numpy(), pa.string())
    cols["patch"] = pa.array(P["patch"].astype(str).to_numpy()).dictionary_encode()
    for c in ("placement", "top4", "level"):
        cols[c] = pa.array(P[c].fillna(0).to_numpy(np.int8))

    const, fdict = [], []
    dense_aggs = {c: P[c].fillna(0).to_numpy(np.float32 if c == "mean_cost" else np.int16)
                  for c in agg_cols}
    dense_aggs.update(A)
    del P, A; gc.collect()
    for c, v in dense_aggs.items():
        if v.min() == v.max():
            const.append(c); continue
        cols[c] = pa.array(v)
        fdict.append((c, "aggregate", float((v > 0).mean())))
    if const: log.warning(f"agregados constantes removidos: {const}")   # [v6]
    del dense_aggs

    for j, (c, b, f) in enumerate(zip(kept_names, kept_blocks, kept_freq)):
        assert c not in cols, f"Colisão de nome: {c}"
        s, e = Xs.indptr[j], Xs.indptr[j + 1]
        v = np.zeros(n, np.int8)
        v[Xs.indices[s:e]] = np.clip(Xs.data[s:e], -128, 127)
        cols[c] = pa.array(v)
        fdict.append((c, b, f))
    nnz_total = Xs.nnz
    del Xs; gc.collect()

    table = pa.table(cols); del cols
    top4 = table["top4"].to_numpy()
    mids = table["match_id"].to_pandas()
    assert pd.Series(top4).groupby(mids.to_numpy()).sum().eq(4).all()
    assert not any(c.startswith("leak_") for c in table.column_names)
    assert len(set(table.column_names)) == len(table.column_names), "Colunas duplicadas na tabela"
    del mids

    pq.write_table(table, out / "features.parquet", compression="zstd", row_group_size=500_000)

    fdict = pd.DataFrame(fdict, columns=["feature", "block", "freq"])
    fdict = pd.concat([fdict, pd.DataFrame([{"feature": "level", "block": "level_model_B",
                                             "freq": 1.0}])], ignore_index=True)
    fdict = fdict.drop_duplicates("feature")
    fdict.to_csv(out / "feature_dict.csv", index=False)

    n_feat = len(fdict) - 1
    summary = {
        "n_rows": n, "n_matches": int(pd.unique(table["match_id"].to_numpy()).size),
        "n_features": n_feat,
        "by_block": fdict["block"].value_counts().to_dict(),
        "min_freq": min_freq, "n_dropped_rare": int(n_dropped),
        "n_dropped_constant": len(const),
        "unit_item_pairs_kept": int((fdict["block"] == "unit_item").sum()),
        "sparsity_sparse_blocks": 1 - nnz_total / max(n * len(kept_names), 1),
        "patches": patch_counts,   # [v6]
    }
    with open(out / "features_log.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/interim")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--min-freq", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=2_000_000, help="linhas por lote de leitura")
    ap.add_argument("--save-sparse", action="store_true")
    a = ap.parse_args()
    main(a.inp, a.out, a.min_freq, a.batch, a.save_sparse)
