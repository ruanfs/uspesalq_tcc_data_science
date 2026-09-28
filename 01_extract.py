"""
01_extract.py — Extração em streaming (baixo uso de memória) dos JSONs da Riot API (TFT Set 16).

Chave de junção entre tabelas: (match_id, pidx), onde pidx = posição do jogador (0–7).

Uso:
    python 01_extract.py --raw raw --out data/interim --chunk 200000
"""
import argparse
import gc
import json
import logging
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import orjson
    def load_json(p: Path):
        return orjson.loads(p.read_bytes())
except ImportError:
    def load_json(p: Path):
        with open(p, encoding="utf-8") as f:
            return json.load(f)

# ---------------- Configuração ----------------
VALID_QUEUES = {1100}
VALID_SET = 16
VALID_PATCHES = {"16.1", "16.2", "16.3", "16.4", "16.5"}
N_PLAYERS = 8
RARITY_TO_COST = {0: 1, 1: 2, 2: 3, 4: 4, 6: 5, 7: 5, 8: 6, 9: 7}
TRAIT_STYLE = {0: "inactive", 1: "bronze", 2: "silver", 3: "unique",
               4: "gold", 5: "prismatic"}
ELO = "Master+ (Master, Grandmaster, Challenger) via League-V1"   # [v6]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("extract")

# ---------------- Schemas (tipos compactos) ----------------
DSTR = pa.dictionary(pa.int32(), pa.string())   # strings repetitivas
SCHEMAS = {
    "matches": pa.schema([
        ("match_id", pa.string()), ("patch", DSTR),
        ("game_datetime", pa.timestamp("ms")), ("game_length_s", pa.float32()),
        ("set_core", DSTR), ("source_file", DSTR),
    ]),
    "participants": pa.schema([
        ("match_id", pa.string()), ("pidx", pa.int8()), ("puuid", pa.string()),
        ("placement", pa.int8()), ("top4", pa.int8()), ("level", pa.int8()),
        ("n_units", pa.int8()), ("sum_cost", pa.int16()), ("mean_cost", pa.float32()),
        ("n_1star", pa.int8()), ("n_2star", pa.int8()), ("n_3star", pa.int8()),
        ("n_items", pa.int8()), ("n_active_traits", pa.int8()), ("n_unique_traits", pa.int8()),
        ("leak_last_round", pa.int16()), ("leak_time_eliminated", pa.float32()),
        ("leak_damage_to_players", pa.int16()), ("leak_players_eliminated", pa.int8()),
        ("leak_gold_left", pa.int16()),
    ]),
    "units": pa.schema([
        ("match_id", pa.string()), ("pidx", pa.int8()), ("puuid", pa.string()),
        ("slot", pa.int8()), ("character_id", DSTR), ("star", pa.int8()),
        ("rarity", pa.int8()), ("cost", pa.int8()), ("n_items", pa.int8()),
    ]),
    "traits": pa.schema([
        ("match_id", pa.string()), ("pidx", pa.int8()), ("puuid", pa.string()),
        ("trait", DSTR), ("num_units", pa.int8()), ("tier_current", pa.int8()),
        ("tier_total", pa.int8()), ("style", pa.int8()), ("style_name", DSTR),
    ]),
    "items": pa.schema([
        ("match_id", pa.string()), ("pidx", pa.int8()), ("puuid", pa.string()),
        ("slot", pa.int8()), ("character_id", DSTR), ("item", DSTR),
        ("item_category", DSTR),
    ]),
}


# ---------------- Writer com buffer ----------------
class BufferedParquet:
    def __init__(self, path: Path, schema: pa.Schema, chunk: int, compression: str):
        self.schema, self.chunk = schema, chunk
        self.writer = pq.ParquetWriter(path, schema, compression=compression)
        self.cols = {n: [] for n in schema.names}
        self.n = 0
        self.total = 0

    def append(self, **row):
        for k, lst in self.cols.items():
            lst.append(row.get(k))
        self.n += 1
        if self.n >= self.chunk:
            self.flush()

    def flush(self):
        if not self.n:
            return
        arrays = []
        for field in self.schema:
            vals = self.cols[field.name]
            if pa.types.is_dictionary(field.type):
                arrays.append(pa.array(vals, pa.string()).dictionary_encode())
            else:
                arrays.append(pa.array(vals, field.type))
        self.writer.write_table(pa.Table.from_arrays(arrays, schema=self.schema))
        self.total += self.n
        self.cols = {n: [] for n in self.schema.names}
        self.n = 0

    def close(self):
        self.flush()
        self.writer.close()


# ---------------- Helpers ----------------
PATCH_RE = re.compile(r"Releases/(\d+\.\d+)")

def parse_patch(v):
    m = PATCH_RE.search(v or "")
    return m.group(1) if m else None


def item_category(name: str) -> str:
    n = name.lower()
    if "radiant" in n: return "radiant"
    if "artifact" in n or "ornn" in n: return "artifact"
    if "emblem" in n: return "emblem"
    if "darkin" in n or n.startswith("tft16_"): return "set_special"
    return "standard"


def validate_match(m: dict, stats: Counter) -> bool:
    info = m.get("info", {})
    if info.get("queue_id", info.get("queueId")) not in VALID_QUEUES:
        stats["drop_queue"] += 1; return False
    if info.get("tft_set_number") != VALID_SET:
        stats["drop_set"] += 1; return False
    if info.get("endOfGameResult", "GameComplete") != "GameComplete":
        stats["drop_incomplete"] += 1; return False
    parts = info.get("participants", [])
    if len(parts) != N_PLAYERS:
        stats["drop_n_players"] += 1; return False
    if sorted(p.get("placement") or 0 for p in parts) != list(range(1, N_PLAYERS + 1)):
        stats["drop_placement"] += 1; return False  # garante exatamente 4 top4
    patch = parse_patch(info.get("game_version"))
    if patch in {"15.24"} or patch not in VALID_PATCHES:
        stats["drop_patch"] += 1; return False

    return True


# ---------------- Extração ----------------
def process_match(m, source, W, agg, stats):
    info, mid = m["info"], m["metadata"]["match_id"]
    patch = parse_patch(info.get("game_version"))
    ts = info.get("game_datetime")

    W["matches"].append(match_id=mid, patch=patch, game_datetime=ts,
                        game_length_s=info.get("game_length"),
                        set_core=info.get("tft_set_core_name"), source_file=source)
    agg["patches"][patch] += 1
    if ts is not None:
        agg["dt_min"] = ts if agg["dt_min"] is None else min(agg["dt_min"], ts)
        agg["dt_max"] = ts if agg["dt_max"] is None else max(agg["dt_max"], ts)

    puuids_seen = set()
    for pidx, p in enumerate(info["participants"]):
        puuid = p.get("puuid") or None
        if puuid is None:
            stats["puuid_missing"] += 1
        elif puuid in puuids_seen:
            stats["puuid_dup_in_match"] += 1
        puuids_seen.add(puuid)

        units, traits = p.get("units", []), p.get("traits", [])
        costs = [RARITY_TO_COST.get(u.get("rarity")) for u in units]
        sum_cost = sum(c for c in costs if c)
        top4 = int(p["placement"] <= 4)
        agg["n_part"] += 1; agg["n_top4"] += top4
        k = {"match_id": mid, "pidx": pidx, "puuid": puuid}

        W["participants"].append(
            **k, placement=p["placement"], top4=top4,
            level=p.get("level"), n_units=len(units), sum_cost=sum_cost,
            mean_cost=(sum_cost / len(units)) if units else 0.0,
            n_1star=sum(u.get("tier") == 1 for u in units),
            n_2star=sum(u.get("tier") == 2 for u in units),
            n_3star=sum(u.get("tier") == 3 for u in units),
            n_items=sum(len(u.get("itemNames", [])) for u in units),
            n_active_traits=sum(t.get("tier_current", 0) > 0 for t in traits),
            n_unique_traits=sum(t.get("style") == 3 for t in traits),
            leak_last_round=p.get("last_round"),
            leak_time_eliminated=p.get("time_eliminated"),
            leak_damage_to_players=p.get("total_damage_to_players"),
            leak_players_eliminated=p.get("players_eliminated"),
            leak_gold_left=p.get("gold_left"),
        )

        for slot, (u, cost) in enumerate(zip(units, costs)):
            cid, items = u.get("character_id"), u.get("itemNames", [])
            agg["units"].add(cid)
            if cost is None:
                agg["unknown_rarity"].add(u.get("rarity"))
            W["units"].append(**k, slot=slot, character_id=cid, star=u.get("tier"),
                              rarity=u.get("rarity"), cost=cost, n_items=len(items))
            for it in items:
                agg["items"].add(it)
                W["items"].append(**k, slot=slot, character_id=cid, item=it,
                                  item_category=item_category(it))

        for t in traits:
            agg["traits"].add(t.get("name"))
            W["traits"].append(**k, trait=t.get("name"), num_units=t.get("num_units"),
                               tier_current=t.get("tier_current"),
                               tier_total=t.get("tier_total"), style=t.get("style"),
                               style_name=TRAIT_STYLE.get(t.get("style"), "other"))


def main(raw_dir, out_dir, chunk, compression):
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("tft_worker_*/batch_*.json"))
    log.info(f"{len(files)} arquivos encontrados em {raw}")

    W = {k: BufferedParquet(out / f"{k}.parquet", s, chunk, compression)
         for k, s in SCHEMAS.items()}
    agg = {"patches": Counter(), "dt_min": None, "dt_max": None, "n_part": 0,
           "n_top4": 0, "units": set(), "traits": set(), "items": set(),
           "unknown_rarity": set()}
    seen, stats = set(), Counter()

    try:
        for f in tqdm(files, desc="Extraindo"):
            try:
                data = load_json(f)
            except Exception as e:
                log.warning(f"Falha ao ler {f}: {e}")
                stats["bad_files"] += 1
                continue
            if isinstance(data, dict):
                data = [data]
            source = f"{f.parent.name}/{f.name}"

            for m in data:
                stats["raw_matches"] += 1
                mid = m.get("metadata", {}).get("match_id")
                if not mid:
                    stats["drop_no_id"] += 1; continue
                if mid in seen:
                    stats["drop_duplicate"] += 1; continue
                if not validate_match(m, stats):
                    continue
                seen.add(mid)
                process_match(m, source, W, agg, stats)
                stats["kept_matches"] += 1

            del data          # libera o JSON bruto imediatamente
            gc.collect()
    finally:
        for name, w in W.items():
            w.close()
            log.info(f"{name:13s}: {w.total:>11,} linhas")

    to_iso = lambda ms: None if ms is None else str(pa.scalar(ms, pa.timestamp("ms")).as_py())
    summary = {
        **stats,
        "n_participants": agg["n_part"],
        "top4_rate": agg["n_top4"] / agg["n_part"] if agg["n_part"] else None,
        "patches": dict(agg["patches"]),
        "date_range": [to_iso(agg["dt_min"]), to_iso(agg["dt_max"])],
        "n_unique_units": len(agg["units"]),
        "n_unique_traits": len(agg["traits"]),
        "n_unique_items": len(agg["items"]),
        "unknown_rarities": sorted(r for r in agg["unknown_rarity"] if r is not None),
        "elo": ELO,   # [v6] documentado para a banca (seção 6)
    }
    with open(out / "extraction_log.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    log.info(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="data/interim")
    ap.add_argument("--chunk", type=int, default=200_000, help="linhas por flush")
    ap.add_argument("--compression", default="zstd")
    a = ap.parse_args()
    main(a.raw, a.out, a.chunk, a.compression)
