"""
02_features.py — Matriz de features (1 linha = jogador × partida), com baixo uso de memória.

Chave: (match_id, pidx).
Split temporal: TREINO = --train-patches (padrão 16.1–16.4); TESTE = demais (16.5).

Regras contra vazamento (Parecer 1.1 e 1.4):
  * O filtro de frequência mínima, a remoção de constantes e a seleção dos top-K pares
    unidade–item são ajustados SOMENTE nas linhas de treino.
  * EmptyBag (marcador de fim de jogo: 12,6% no 1º lugar contra ~0,01% do 5º ao 8º)
    é descartado de TODAS as features.
  * ThiefsGloves fica como item esparso, mas sai dos contadores de volume.
Cenário C (Parecer 1.2): o bloco 'aggregate_norm' traz métricas de intensidade
  independentes do tamanho do tabuleiro.
Definição de "ouro+" (H1): style ∈ {4: gold, 5: prismatic}. Traits únicas (style=3)
  NÃO entram e são contadas à parte em n_unique_traits.

Saídas: features.parquet (com a coluna 'split'), feature_dict.csv,
        features_log.json, fit_artifacts.json

Uso:
    python 02_features.py --inp data/interim --out data/processed --min-freq 0.005
"""
import argparse
import gc
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

SCRIPT_VERSION = "02_features v7"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
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
LEAK_CATEGORIES = {"empty_bag"}            # descarte total
NO_VOLUME_CATEGORIES = {"thiefs_gloves"}   # fora dos contadores de volume
TOP_K_PAIRS = 10_000
CARRY_MIN_ITEMS = 3
S = 16
KEY = ["match_id", "pidx"]
STYLE_GOLD, STYLE_PRISMATIC = 4, 5
INTENSIVE_AGGS = {"mean_cost"}             # agregado original que já é de intensidade


def sha256_file(p: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(block), b""):
            h.update(c)
    return h.hexdigest()


class Vocab:
    """Mapeia strings normalizadas -> códigos int32 estáveis entre lotes."""
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
        return lut[cat.cat.codes.to_numpy()]


def batches(path, cols, bs):
    for b in pq.ParquetFile(path).iter_batches(batch_size=bs, columns=cols):
        yield b.to_pandas()


def to_rid(df, key_index):
    return key_index.get_indexer(pd.MultiIndex.from_arrays(
        [df["match_id"].astype(str).to_numpy(), df["pidx"].to_numpy(np.int16)]))


def build_sparse(rid, col, val, n, m, agg):
    rid, col, val = map(np.concatenate, (rid, col, val))
    if agg == "max":
        k = rid.astype(np.int64) * m + col
        s = pd.Series(val).groupby(k).max()
        k = s.index.to_numpy()
        rid, col, val = k // m, k % m, s.to_numpy()
    X = sp.csr_matrix((val.astype(np.int16), (rid, col)), shape=(n, m))
    X.eliminate_zeros()
    return X


def main(inp, out, min_freq, bs, save_sparse, train_patches):
    started = datetime.now(timezone.utc).isoformat()
    inp, out = Path(inp), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- Base ----------
    agg_cols = ["n_units", "sum_cost", "mean_cost", "n_1star", "n_2star",
                "n_3star", "n_items", "n_active_traits", "n_unique_traits"]
    P = pd.read_parquet(inp / "participants.parquet",
                        columns=KEY + ["puuid", "placement", "top4", "level",
                                       "n_units", "sum_cost", "mean_cost", "n_1star",
                                       "n_2star", "n_3star", "n_items_clean",
                                       "n_active_traits", "n_unique_traits"])
    P = P.rename(columns={"n_items_clean": "n_items"})   # n_items = sem itens suspeitos
    M = pd.read_parquet(inp / "matches.parquet",
                        columns=["match_id", "patch"]).drop_duplicates("match_id")
    P = P.merge(M, on="match_id", how="left"); del M
    P["match_id"] = P["match_id"].astype(str)
    P["pidx"] = P["pidx"].astype(np.int16)
    P["patch"] = P["patch"].astype(str)

    dup = P.duplicated(KEY, keep=False)
    if dup.any():
        bad = P.loc[dup, "match_id"].unique()
        log.warning(f"{int(dup.sum()):,} linhas com chave duplicada em {len(bad):,} partidas; removendo")
        P = P[~P["match_id"].isin(bad)].reset_index(drop=True)
    n = len(P)
    key_index = pd.MultiIndex.from_arrays([P["match_id"].to_numpy(), P["pidx"].to_numpy()])
    assert key_index.is_unique, "Chave duplicada em participants"

    is_train = P["patch"].isin(train_patches).to_numpy()
    n_tr = int(is_train.sum())
    if n_tr == 0 or n_tr == n:
        raise ValueError(f"Split inválido: n_train={n_tr}, n={n}")
    tr_idx = np.flatnonzero(is_train)
    patch_counts = {str(k): int(v) for k, v in
                    P.drop_duplicates("match_id")["patch"].value_counts().items()}
    log.info(f"{n:,} linhas | treino={n_tr:,} ({sorted(train_patches)}) | "
             f"teste={n - n_tr:,} | partidas por patch: {patch_counts}")

    A = {k: np.zeros(n, np.int16) for k in
         ("n_hi_cost", "n_hi_cost_2star", "n_hi_cost_1star", "n_carries", "n_components",
          "n_artifacts", "n_radiants", "n_emblems", "n_full_items", "n_full_items_on_hi_cost",
          "n_traits_gold_plus", "n_traits_prismatic")}
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
    for df in batches(inp / "traits.parquet", KEY + ["trait", "tier_current", "style"], bs):
        rid = to_rid(df, key_index)
        ok = (rid >= 0) & df["trait"].notna().to_numpy()
        df, rid = df[ok], rid[ok]
        st = df["style"].fillna(0).to_numpy(np.int8)
        A["n_traits_gold_plus"] += np.bincount(rid, st >= STYLE_GOLD, n).astype(np.int16)
        A["n_traits_prismatic"] += np.bincount(rid, st == STYLE_PRISMATIC, n).astype(np.int16)
        tr.append(rid.astype(np.int32)); tc.append(Vt.encode(df["trait"], short))
        tv.append(df["tier_current"].fillna(0).to_numpy(np.int8))
        del df
    Xt = build_sparse(tr, tc, tv, n, len(Vt.names), "max"); del tr, tc, tv; gc.collect()

    # ---------- Items ----------
    Vi, Vc, ir, ic, ich = Vocab(), Vocab(), [], [], []
    n_leak_rows = n_novol_rows = 0
    for df in batches(inp / "items.parquet",
                      KEY + ["slot", "character_id", "item", "item_category"], bs):
        rid = to_rid(df, key_index)
        cat = df["item_category"].astype(str).to_numpy()
        ok = (rid >= 0) & df["item"].notna().to_numpy()
        leak = np.isin(cat, list(LEAK_CATEGORIES))
        n_leak_rows += int((ok & leak).sum())
        ok &= ~leak                                         # EmptyBag fora de tudo
        df, rid, cat = df[ok], rid[ok], cat[ok]

        comp = df["item"].astype(str).map(short).isin(COMPONENTS).to_numpy()
        A["n_components"] += np.bincount(rid, comp, n).astype(np.int16)
        for c, k in (("artifact", "n_artifacts"), ("radiant", "n_radiants"),
                     ("emblem", "n_emblems")):
            A[k] += np.bincount(rid, cat == c, n).astype(np.int16)

        novol = np.isin(cat, list(NO_VOLUME_CATEGORIES))
        n_novol_rows += int(novol.sum())
        full = ~comp
        vol = full & ~novol                                 # ThiefsGloves fora do volume
        sk = rid.astype(np.int64) * S + df["slot"].to_numpy(np.int64)
        A["n_full_items"] += np.bincount(rid, vol, n).astype(np.int16)
        A["n_full_items_on_hi_cost"] += np.bincount(
            rid, vol & (slot_cost[sk] >= 4), n).astype(np.int16)
        np.add.at(full_per_unit, sk[vol], 1)

        dff = df[full]                                      # bloco esparso: itens completos
        ir.append(rid[full].astype(np.int32)); ic.append(Vi.encode(dff["item"], short))
        ich.append(Vc.encode(dff["character_id"].fillna("NA"), short))
        del df, dff
    del slot_cost
    A["n_carries"] = np.bincount(np.nonzero(full_per_unit >= CARRY_MIN_ITEMS)[0] // S,
                                 minlength=n).astype(np.int16)
    del full_per_unit; gc.collect()
    log.info(f"Itens descartados (vazamento EmptyBag): {n_leak_rows:,} | "
             f"fora do volume (ThiefsGloves): {n_novol_rows:,}")

    ir, ic, ich = map(np.concatenate, (ir, ic, ich))
    Xi = build_sparse([ir], [ic], [np.ones(len(ir), np.int8)], n, len(Vi.names), "sum")

    # ---------- Top-K pares unidade–item: ajustados SÓ no treino ----------
    nI = len(Vi.names)
    pcode = ich.astype(np.int64) * nI + ic
    del ic, ich
    uniq, cnt = np.unique(pcode[is_train[ir]], return_counts=True)
    top_sorted = np.sort(uniq[np.argsort(-cnt, kind="stable")[:TOP_K_PAIRS]])
    mask = np.isin(pcode, top_sorted)
    pcol = np.searchsorted(top_sorted, pcode[mask])
    Xp = build_sparse([ir[mask]], [pcol], [np.ones(int(mask.sum()), np.int8)],
                      n, len(top_sorted), "sum")
    pair_names = [f"{Vc.names[c // nI]}__{Vi.names[c % nI]}" for c in top_sorted]
    del ir, pcode, mask, pcol; gc.collect()

    # ---------- Filtro de frequência: ajustado SÓ no treino ----------
    blocks = [("trait_", "trait", Xt, Vt.names), ("unit_", "unit", Xu, Vu.names),
              ("item_", "item", Xi, Vi.names), ("pair_", "unit_item", Xp, pair_names)]
    kept_mats, kept_names, kept_blocks, kept_freq = [], [], [], []
    n_dropped = 0
    for prefix, bname, X, names in blocks:
        freq_tr = X[tr_idx].getnnz(axis=0) / n_tr
        keep = np.nonzero((freq_tr >= min_freq) & (freq_tr < 1.0))[0]
        n_dropped += X.shape[1] - len(keep)
        kept_mats.append(X[:, keep])
        kept_names += [f"{prefix}{names[j]}" for j in keep]
        kept_blocks += [bname] * len(keep)
        kept_freq += freq_tr[keep].tolist()
        log.info(f"bloco {bname:9s}: {X.shape[1]:>6,} -> {len(keep):>5,} colunas")
    del Xt, Xu, Xi, Xp, blocks
    Xs = sp.hstack(kept_mats, format="csc"); del kept_mats; gc.collect()
    assert not pd.Index(kept_names).duplicated().any(), "Colunas duplicadas"

    if save_sparse:
        sp.save_npz(out / "features_sparse.npz", Xs.tocsr())
        pd.Series(kept_names).to_csv(out / "features_sparse_cols.csv", index=False, header=False)

    # ---------- Montagem colunar ----------
    cols = {"match_id": pa.array(P["match_id"].to_numpy()),
            "pidx": pa.array(P["pidx"].to_numpy(np.int8)),
            "puuid": pa.array(P["puuid"].astype(object).to_numpy(), pa.string()),
            "patch": pa.array(P["patch"].to_numpy()).dictionary_encode(),
            "split": pa.array(np.where(is_train, "train", "test")).dictionary_encode()}
    for c in ("placement", "top4", "level"):
        cols[c] = pa.array(P[c].fillna(0).to_numpy(np.int8))

    dense = {c: P[c].fillna(0).to_numpy(np.float32 if c == "mean_cost" else np.int16)
             for c in agg_cols}
    dense.update(A)
    del P, A; gc.collect()

    # Cenário C: métricas de intensidade (desacoplam a qualidade do tamanho do tabuleiro)
    nu = np.maximum(dense["n_units"], 1).astype(np.float32)
    norm = {
        "board_value_per_unit": dense["board_value"] / nu,
        "full_items_per_unit": dense["n_full_items"] / nu,
        "carries_per_unit": dense["n_carries"] / nu,
        "share_2star": dense["n_2star"] / nu,
        "share_3star": dense["n_3star"] / nu,
        "share_hi_cost": dense["n_hi_cost"] / nu,
        "share_hi_cost_2star": dense["n_hi_cost_2star"] / np.maximum(dense["n_hi_cost"], 1),
        "component_share": dense["n_components"] /
            np.maximum(dense["n_components"] + dense["n_full_items"], 1),
        "gold_plus_share": dense["n_traits_gold_plus"] /
            np.maximum(dense["n_active_traits"], 1),
    }
    norm = {k: v.astype(np.float32) for k, v in norm.items()}

    const, fdict = [], []
    for block, src in (("aggregate", dense), ("aggregate_norm", norm)):
        for c, v in src.items():
            vt = v[is_train]
            if vt.min() == vt.max():                       # constância avaliada no treino
                const.append(c); continue
            cols[c] = pa.array(v)
            is_vol = block == "aggregate" and c not in INTENSIVE_AGGS
            fdict.append((c, block, float((vt > 0).mean()), is_vol, not is_vol))
    if const:
        log.warning(f"agregados constantes no treino removidos: {const}")
    n_vol = sum(1 for f in fdict if f[3])
    log.info(f"agregados de volume: {n_vol} | intensidade (Cenário C): "
             f"{sum(1 for f in fdict if f[1] != 'aggregate' or not f[3])}")
    del dense, norm

    for j, (c, b, f) in enumerate(zip(kept_names, kept_blocks, kept_freq)):
        assert c not in cols, f"Colisão de nome: {c}"
        s, e = Xs.indptr[j], Xs.indptr[j + 1]
        v = np.zeros(n, np.int8)
        v[Xs.indices[s:e]] = np.clip(Xs.data[s:e], -128, 127)
        cols[c] = pa.array(v)
        fdict.append((c, b, f, False, True))
    nnz_total = Xs.nnz
    del Xs; gc.collect()

    table = pa.table(cols); del cols
    names_all = table.column_names
    top4 = table["top4"].to_numpy()
    mids = table["match_id"].to_numpy()
    assert pd.Series(top4).groupby(mids).sum().eq(4).all()
    assert not any(c.startswith(("leak_", "susp_")) for c in names_all)
    assert not any("emptybag" in c.lower() for c in names_all), "EmptyBag vazou"
    assert len(set(names_all)) == len(names_all)
    n_matches = int(pd.unique(mids).size); del mids

    pq.write_table(table, out / "features.parquet", compression="zstd", row_group_size=500_000)
    del table; gc.collect()

    fdict = pd.DataFrame(fdict, columns=["feature", "block", "freq_train",
                                         "is_volume", "in_scenario_c"])
    fdict = pd.concat([fdict, pd.DataFrame([{
        "feature": "level", "block": "level_model_B", "freq_train": 1.0,
        "is_volume": True, "in_scenario_c": False}])], ignore_index=True)
    fdict = fdict.drop_duplicates("feature")
    fdict.to_csv(out / "feature_dict.csv", index=False)

    summary = {
        "n_rows": n, "n_train": n_tr, "n_test": n - n_tr, "n_matches": n_matches,
        "train_patches": sorted(train_patches),
        "n_features": len(fdict) - 1,
        "by_block": fdict["block"].value_counts().to_dict(),
        "n_volume_aggregates": n_vol,
        "min_freq": min_freq, "n_dropped_rare": int(n_dropped),
        "n_dropped_constant": len(const),
        "unit_item_pairs_kept": int((fdict["block"] == "unit_item").sum()),
        "sparsity_sparse_blocks": 1 - nnz_total / max(n * len(kept_names), 1),
        "items_dropped_empty_bag": n_leak_rows,
        "items_excluded_from_volume_thiefs_gloves": n_novol_rows,
        "gold_plus_definition": "style in {4,5}; unique (style=3) excluída",
        "patches": patch_counts,
    }
    with open(out / "features_log.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    upstream = inp / "extraction_manifest.json"
    artifacts = {
        "script": SCRIPT_VERSION, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "fit_scope": "train_only", "train_patches": sorted(train_patches),
        "min_freq": min_freq, "top_k_pairs": TOP_K_PAIRS,
        "leak_categories": sorted(LEAK_CATEGORIES),
        "no_volume_categories": sorted(NO_VOLUME_CATEGORIES),
        "kept_sparse_features": kept_names,
        "upstream_raw_sha256": (json.load(open(upstream))["raw_dataset_sha256"]
                                if upstream.exists() else None),
        "features_parquet_sha256": sha256_file(out / "features.parquet"),
    }
    with open(out / "fit_artifacts.json", "w") as fh:
        json.dump(artifacts, fh, indent=2)
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/interim")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--min-freq", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=2_000_000)
    ap.add_argument("--save-sparse", action="store_true")
    ap.add_argument("--train-patches", default="16.1,16.2,16.3,16.4")
    a = ap.parse_args()
    main(a.inp, a.out, a.min_freq, a.batch, a.save_sparse,
         set(a.train_patches.split(",")))
