"""
01_extract.py — Extração em streaming dos JSONs da Riot API (TFT Set 16).

Saídas (em --out):
    matches.parquet, participants.parquet, units.parquet, traits.parquet, items.parquet
    extraction_log.json       -> estatísticas de filtragem e contagens
    extraction_manifest.json  -> SHA-256 dos brutos e das saídas, versões e status

Chave de junção: (match_id, pidx), em que pidx = posição do jogador (0 a 7).

Notas metodológicas:
  * Este script NÃO particiona os dados nem aplica filtros dependentes da distribuição.
    O split temporal (16.1–16.4 treino, 16.5 teste) acontece apenas nas etapas seguintes.
  * Colunas 'leak_*' são estados de fim de jogo e NUNCA devem entrar como preditoras.
  * Itens suspeitos (EmptyBag, ThiefsGloves) recebem categoria própria e contadores
    segregados ('susp_*'), para auditoria (Parecer, item 1.1).

Uso:
    python 01_extract.py --raw raw --out data/interim --chunk 200000
"""
import argparse
import hashlib
import json
import logging
import platform as py_platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import orjson

    def parse_bytes(b: bytes):
        return orjson.loads(b)
    JSON_BACKEND = f"orjson {orjson.__version__}"
except ImportError:
    def parse_bytes(b: bytes):
        return json.loads(b.decode("utf-8"))
    JSON_BACKEND = "json (stdlib)"

# ---------------- Configuração ----------------
SCRIPT_VERSION = "01_extract v7"
VALID_QUEUES = {1100}                                   # Ranked TFT
VALID_SET = 16
VALID_PATCHES = {"16.1", "16.2", "16.3", "16.4", "16.5"}
N_PLAYERS = 8
# rarity da API -> custo em ouro. 8 e 9 são unidades especiais do Set 16 (custo 6 e 7).
RARITY_TO_COST = {0: 1, 1: 2, 2: 3, 4: 4, 6: 5, 7: 5, 8: 6, 9: 7}
TRAIT_STYLE = {0: "inactive", 1: "bronze", 2: "silver", 3: "unique",
               4: "gold", 5: "prismatic"}
# Seleção de jogadores: Master+ na data da coleta. As partidas incluem adversários
# de elo menor (viés de seleção documentado; ver ameaças à validade).
ELO = "Seed players Master+ (Master/GM/Challenger) via League-V1; partidas históricas"

# Itens suspeitos auditados (Parecer 1.1); a busca é feita por substring, em minúsculas.
SUSPECT_ITEMS = {"emptybag": "empty_bag", "thiefsgloves": "thiefs_gloves"}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("extract")

# ---------------- Schemas ----------------
DSTR = pa.dictionary(pa.int32(), pa.string())
SCHEMAS = {
    "matches": pa.schema([
        ("match_id", pa.string()), ("platform", DSTR), ("patch", DSTR),
        ("game_datetime", pa.timestamp("ms")), ("game_length_s", pa.float32()),
        ("set_core", DSTR), ("source_file", DSTR),
    ]),
    "participants": pa.schema([
        ("match_id", pa.string()), ("pidx", pa.int8()), ("puuid", pa.string()),
        ("placement", pa.int8()), ("top4", pa.int8()), ("level", pa.int8()),
        ("n_units", pa.int8()), ("n_units_unknown_cost", pa.int8()),
        ("sum_cost", pa.int16()), ("mean_cost", pa.float32()),
        ("n_1star", pa.int8()), ("n_2star", pa.int8()), ("n_3star", pa.int8()),
        ("n_items", pa.int8()),            # bruto (compatibilidade com 02_features)
        ("n_items_clean", pa.int8()),      # sem itens suspeitos
        ("n_active_traits", pa.int8()), ("n_unique_traits", pa.int8()),
        # Itens suspeitos, segregados (NÃO usar como preditores até concluir a auditoria)
        ("susp_n_empty_bag", pa.int8()), ("susp_n_thiefs_gloves", pa.int8()),
        # Estados de fim de jogo (vazamento direto)
        ("leak_last_round", pa.int16()), ("leak_time_eliminated", pa.float32()),
        ("leak_damage_to_players", pa.int16()), ("leak_players_eliminated", pa.int8()),
        ("leak_gold_left", pa.int16()),
    ]),
    "units": pa.schema([
        ("match_id", pa.string()), ("pidx", pa.int8()), ("puuid", pa.string()),
        ("slot", pa.int8()), ("character_id", DSTR), ("star", pa.int8()),
        ("rarity", pa.int8()), ("cost", pa.int8()), ("n_items", pa.int8()),
        ("n_items_clean", pa.int8()),
    ]),
    "traits": pa.schema([
        ("match_id", pa.string()), ("pidx", pa.int8()), ("puuid", pa.string()),
        ("trait", DSTR), ("num_units", pa.int8()), ("tier_current", pa.int8()),
        ("tier_total", pa.int8()), ("style", pa.int8()), ("style_name", DSTR),
    ]),
    "items": pa.schema([
        ("match_id", pa.string()), ("pidx", pa.int8()), ("puuid", pa.string()),
        ("slot", pa.int8()), ("character_id", DSTR), ("item", DSTR),
        ("item_category", DSTR), ("is_suspect", pa.int8()),
    ]),
}


# ---------------- Writer com buffer ----------------
class BufferedParquet:
    def __init__(self, path: Path, schema: pa.Schema, chunk: int, compression: str):
        self.path, self.schema, self.chunk = path, schema, chunk
        self.writer = pq.ParquetWriter(path, schema, compression=compression)
        self.cols = {n: [] for n in schema.names}
        self.n = 0
        self.total = 0

    def append(self, **row):
        unknown = set(row) - set(self.cols)
        if unknown:
            raise KeyError(f"{self.path.name}: colunas fora do schema {unknown}")
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
PLATFORM_RE = re.compile(r"^([A-Z0-9]+)_")


def parse_patch(v):
    m = PATCH_RE.search(v or "")
    return m.group(1) if m else None


def parse_platform(match_id: str):
    m = PLATFORM_RE.match(match_id or "")
    return m.group(1) if m else None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def suspect_tag(name: str):
    n = (name or "").lower()
    for key, tag in SUSPECT_ITEMS.items():
        if key in n:
            return tag
    return None


def item_category(name: str) -> str:
    tag = suspect_tag(name)
    if tag:
        return tag                                   # categoria própria e segregada
    n = (name or "").lower()
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
        stats["drop_placement"] += 1; return False      # garante exatamente 4 no Top 4
    if parse_patch(info.get("game_version")) not in VALID_PATCHES:
        stats["drop_patch"] += 1; return False
    return True


# ---------------- Extração ----------------
def process_match(m, source, W, agg, stats):
    info, mid = m["info"], m["metadata"]["match_id"]
    patch = parse_patch(info.get("game_version"))
    plat = parse_platform(mid)
    ts = info.get("game_datetime")

    W["matches"].append(match_id=mid, platform=plat, patch=patch, game_datetime=ts,
                        game_length_s=info.get("game_length"),
                        set_core=info.get("tft_set_core_name"), source_file=source)
    agg["patches"][patch] += 1
    agg["platforms"][plat] += 1
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
        known = [c for c in costs if c is not None]
        n_unknown = len(costs) - len(known)
        if n_unknown:
            stats["participants_with_unknown_cost"] += 1
        sum_cost = sum(known)
        mean_cost = (sum_cost / len(known)) if known else 0.0   # denominador coerente

        n_items_raw = n_items_clean = n_eb = n_tg = 0
        for u in units:
            for it in u.get("itemNames", []):
                n_items_raw += 1
                tag = suspect_tag(it)
                if tag == "empty_bag": n_eb += 1
                elif tag == "thiefs_gloves": n_tg += 1
                else: n_items_clean += 1

        top4 = int(p["placement"] <= 4)
        agg["n_part"] += 1; agg["n_top4"] += top4
        agg["susp_by_place"][p["placement"]]["empty_bag"] += int(n_eb > 0)
        agg["susp_by_place"][p["placement"]]["thiefs_gloves"] += int(n_tg > 0)
        agg["susp_by_place"][p["placement"]]["n"] += 1
        k = {"match_id": mid, "pidx": pidx, "puuid": puuid}

        W["participants"].append(
            **k, placement=p["placement"], top4=top4, level=p.get("level"),
            n_units=len(units), n_units_unknown_cost=n_unknown,
            sum_cost=sum_cost, mean_cost=mean_cost,
            n_1star=sum(u.get("tier") == 1 for u in units),
            n_2star=sum(u.get("tier") == 2 for u in units),
            n_3star=sum(u.get("tier") == 3 for u in units),
            n_items=n_items_raw, n_items_clean=n_items_clean,
            n_active_traits=sum((t.get("tier_current") or 0) > 0 for t in traits),
            n_unique_traits=sum(t.get("style") == 3 for t in traits),
            susp_n_empty_bag=n_eb, susp_n_thiefs_gloves=n_tg,
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
                              rarity=u.get("rarity"), cost=cost, n_items=len(items),
                              n_items_clean=sum(suspect_tag(i) is None for i in items))
            for it in items:
                agg["items"].add(it)
                W["items"].append(**k, slot=slot, character_id=cid, item=it,
                                  item_category=item_category(it),
                                  is_suspect=int(suspect_tag(it) is not None))

        for t in traits:
            agg["traits"].add(t.get("name"))
            W["traits"].append(**k, trait=t.get("name"), num_units=t.get("num_units"),
                               tier_current=t.get("tier_current"),
                               tier_total=t.get("tier_total"), style=t.get("style"),
                               style_name=TRAIT_STYLE.get(t.get("style"), "other"))


def lib_versions():
    import pandas, numpy  # noqa: E401 (apenas para registrar as versões)
    return {"python": sys.version.split()[0], "platform": py_platform.platform(),
            "pyarrow": pa.__version__, "pandas": pandas.__version__,
            "numpy": numpy.__version__, "json_backend": JSON_BACKEND}


def main(raw_dir, out_dir, chunk, compression):
    started = datetime.now(timezone.utc).isoformat()
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("tft_worker_*/batch_*.json"))
    log.info(f"{len(files)} arquivos encontrados em {raw}")
    if not files:
        raise FileNotFoundError(f"Nenhum batch_*.json em {raw}/tft_worker_*/")

    W = {k: BufferedParquet(out / f"{k}.parquet", s, chunk, compression)
         for k, s in SCHEMAS.items()}
    agg = {"patches": Counter(), "platforms": Counter(), "dt_min": None, "dt_max": None,
           "n_part": 0, "n_top4": 0, "units": set(), "traits": set(), "items": set(),
           "unknown_rarity": set(),
           "susp_by_place": {r: Counter() for r in range(1, N_PLAYERS + 1)}}
    seen, rejected, stats = set(), set(), Counter()
    raw_hashes, status = {}, "FAILED"

    try:
        for f in tqdm(files, desc="Extraindo"):
            rel = f"{f.parent.name}/{f.name}"
            try:
                b = f.read_bytes()
                raw_hashes[rel] = sha256_bytes(b)
                data = parse_bytes(b)
                del b
            except (OSError, ValueError) as e:
                log.warning(f"Falha ao ler {rel}: {e}")
                stats["bad_files"] += 1
                continue
            if isinstance(data, dict):
                data = [data]

            for m in data:
                stats["raw_matches"] += 1
                mid = (m.get("metadata") or {}).get("match_id")
                if not mid:
                    stats["drop_no_id"] += 1; continue
                if mid in seen or mid in rejected:
                    stats["drop_duplicate"] += 1; continue
                if not validate_match(m, stats):
                    rejected.add(mid); continue
                seen.add(mid)
                process_match(m, rel, W, agg, stats)
                stats["kept_matches"] += 1
            del data
        status = "OK"
    finally:
        for name, w in W.items():
            w.close()
            log.info(f"{name:13s}: {w.total:>11,} linhas")
        if status != "OK":
            log.error("Extração INTERROMPIDA: Parquets parciais. Não usar.")

    to_iso = lambda ms: None if ms is None else str(pa.scalar(ms, pa.timestamp("ms")).as_py())
    susp = {str(r): {"n": c["n"],
                     "rate_empty_bag": c["empty_bag"] / c["n"] if c["n"] else None,
                     "rate_thiefs_gloves": c["thiefs_gloves"] / c["n"] if c["n"] else None}
            for r, c in agg["susp_by_place"].items()}
    summary = {
        **stats,
        "n_participants": agg["n_part"],
        "top4_rate": agg["n_top4"] / agg["n_part"] if agg["n_part"] else None,
        "patches": dict(agg["patches"]),
        "platforms": dict(agg["platforms"]),
        "date_range": [to_iso(agg["dt_min"]), to_iso(agg["dt_max"])],
        "n_unique_units": len(agg["units"]),
        "n_unique_traits": len(agg["traits"]),
        "n_unique_items": len(agg["items"]),
        "unknown_rarities": sorted(r for r in agg["unknown_rarity"] if r is not None),
        "suspect_items_by_placement": susp,
        "elo": ELO,
    }
    with open(out / "extraction_log.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

    raw_digest = sha256_bytes("".join(f"{k}:{v}\n" for k, v in sorted(raw_hashes.items()))
                              .encode())
    manifest = {
        "script": SCRIPT_VERSION, "status": status,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "args": {"raw": str(raw), "out": str(out), "chunk": chunk,
                 "compression": compression},
        "env": lib_versions(),
        "raw_dataset_sha256": raw_digest,
        "raw_files": raw_hashes,
        "outputs": {f"{k}.parquet": sha256_file(out / f"{k}.parquet") for k in SCHEMAS},
    }
    with open(out / "extraction_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info(f"status={status} | raw_dataset_sha256={raw_digest}")
    log.info(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="data/interim")
    ap.add_argument("--chunk", type=int, default=200_000, help="linhas por flush")
    ap.add_argument("--compression", default="zstd")
    a = ap.parse_args()
    main(a.raw, a.out, a.chunk, a.compression)
