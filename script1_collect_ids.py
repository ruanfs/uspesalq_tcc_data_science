"""
================================================================================
  SCRIPT 1 — COLETA DE MATCH IDs
  TFT KR · Challenger (todos) + Grandmaster (todos) + Master

  O que faz:
    1. Busca todos os PUUIDs do Challenger
    2. Busca todos os PUUIDs do Grandmaster
    3. Busca (ordenados por LP)
    4. Remove PUUIDs duplicados entre os três tiers
    5. Para cada jogador coleta até 300 Match IDs da API
    6. Remove Match IDs duplicados e divide igualmente em 5 arquivos .txt
       (tft_match_ids_part_1.txt  …  tft_match_ids_part_5.txt)

  Esses arquivos serão consumidos pelo Script 2 (download das partidas).

  Checkpoint: tft_collect_progress.json
    → Se interrompido, ao rodar novamente continua de onde parou
      sem repetir jogadores já processados.
================================================================================
"""

import requests
import time
import json
import os
import math
from datetime import datetime

# =============================================================================
# CONFIGURAÇÃO — edite aqui
# =============================================================================
API_KEY            = ""
REGION_ROUTING     = "asia"    # americas | asia | europe
REGION_PLATFORM    = "kr"      # br1 | na1 | kr | euw1 | la1 | la2 ...

MASTER_TOP_N       = 5000      # Quantos jogadores pegar do Master (os melhores por LP)
MATCHES_PER_PLAYER = 300       # IDs coletados por jogador
NUM_OUTPUT_FILES   = 10        # Número de arquivos de saída

REQUEST_DELAY      = 1.2       # Delay (s) entre chamadas à API
CHECKPOINT_EVERY   = 50        # Salva progresso a cada N jogadores

PROGRESS_FILE      = "tft_collect_progress.json"
OUTPUT_PREFIX      = "tft_match_ids_part"   # tft_match_ids_part_1.txt …

HEADERS  = {"X-Riot-Token": API_KEY}
LOG_FILE = "tft_collect_log.txt"   # Arquivo de log com horários de cada etapa

# =============================================================================
# LOG DE HORÁRIOS
# =============================================================================

def _now() -> str:
    """Retorna timestamp formatado: 2025-06-10 14:32:07"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _elapsed(seconds: float) -> str:
    """Converte segundos em string legível: 1h 23m 45s"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"

def log(msg: str):
    """Imprime no terminal e grava no arquivo de log com timestamp."""
    line = f"[{_now()}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def log_stage_start(label: str) -> float:
    """Loga início de etapa e retorna o timestamp de início (para calcular duração)."""
    log(f"{'─' * 56}")
    log(f"▶  INÍCIO  │ {label}")
    log(f"{'─' * 56}")
    return time.time()

def log_stage_end(label: str, start_time: float, extra: str = ""):
    """Loga fim de etapa com duração total."""
    duration = time.time() - start_time
    log(f"{'─' * 56}")
    log(f"■  FIM     │ {label}")
    log(f"   Duração │ {_elapsed(duration)}")
    if extra:
        log(f"   Resumo  │ {extra}")
    log(f"{'─' * 56}\n")

# =============================================================================
# CHECKPOINT
# =============================================================================

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Checkpoint] Progresso anterior encontrado → retomando execução.")
        return data
    return {
        "stage":            1,   # 1=puuids  2=match_ids  3=split  done=concluído
        "puuids":           [],
        "puuid_sources":    {},  # puuid → tier de origem (para log)
        "processed_puuids": [],
        "unique_match_ids": [],
    }

def save_progress(p: dict):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PROGRESS_FILE)

# =============================================================================
# API — RIOT  (coleta de PUUIDs por tier)
# =============================================================================

def _fetch_tier_puuids(endpoint: str, label: str, limit: int | None = None) -> list[str]:
    """
    Busca entradas de um tier (challenger/grandmaster/master),
    ordena por LP decrescente e retorna até `limit` PUUIDs (None = todos).
    """
    url = f"https://{REGION_PLATFORM}.api.riotgames.com{endpoint}?queue=RANKED_TFT"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 30))
            print(f"\n  [Rate Limit] Aguardando {wait}s...")
            time.sleep(wait)
            return _fetch_tier_puuids(endpoint, label, limit)
        r.raise_for_status()
        entries = r.json().get("entries", [])
        entries = sorted(entries, key=lambda x: x.get("leaguePoints", 0), reverse=True)
        if limit:
            entries = entries[:limit]
        puuids = [e["puuid"] for e in entries if "puuid" in e]
        print(f"  [OK] {label:<12} → {len(puuids):>5} jogadores")
        return puuids
    except Exception as e:
        print(f"  [ERRO] Falha ao buscar {label}: {e}")
        return []

def collect_all_puuids() -> tuple[list[str], dict[str, str]]:
    """
    Coleta PUUIDs dos três tiers, remove duplicatas e retorna:
      - lista ordenada de PUUIDs únicos
      - dicionário puuid → tier (para log)
    """
    print("\n  Buscando rankings...")
    challenger  = _fetch_tier_puuids("/tft/league/v1/challenger",  "Challenger")
    time.sleep(REQUEST_DELAY)
    grandmaster = _fetch_tier_puuids("/tft/league/v1/grandmaster", "Grandmaster")
    time.sleep(REQUEST_DELAY)
    master      = _fetch_tier_puuids("/tft/league/v1/master",      "Master",      limit=MASTER_TOP_N)

    # Monta dicionário de origem (o tier mais alto prevalece em caso de duplicata)
    sources: dict[str, str] = {}
    for p in master:      sources[p] = "Master"
    for p in grandmaster: sources[p] = "Grandmaster"
    for p in challenger:  sources[p] = "Challenger"

    # Preserva ordem: Challenger → Grandmaster → Master (sem repetições)
    seen   = set()
    unique = []
    for p in (challenger + grandmaster + master):
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique, sources

def get_match_ids(puuid: str) -> list[str]:
    url = (f"https://{REGION_ROUTING}.api.riotgames.com"
           f"/tft/match/v1/matches/by-puuid/{puuid}/ids?count={MATCHES_PER_PLAYER}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 30))
            print(f"\n  [Rate Limit] Aguardando {wait}s...")
            time.sleep(wait)
            return get_match_ids(puuid)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

# =============================================================================
# SAÍDA — divide IDs em N arquivos .txt
# =============================================================================

def split_and_save(match_ids: list[str], num_files: int):
    """Divide a lista de IDs em num_files partes iguais e salva como .txt."""
    total = len(match_ids)
    size  = math.ceil(total / num_files)

    print(f"\n--- Dividindo {total} IDs em {num_files} arquivos ---")
    for i in range(num_files):
        chunk    = match_ids[i * size : (i + 1) * size]
        filename = f"{OUTPUT_PREFIX}_{i + 1}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk))
        print(f"  [OK] {filename}  →  {len(chunk)} IDs")
    print(f"\n✅ Arquivos prontos para o Script 2.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    p = load_progress()

    # Cabeçalho geral no log (apenas na primeira execução ou sempre que rodar)
    log(f"{'═' * 56}")
    log(f"  TFT SCRIPT 1 — Coleta de Match IDs")
    log(f"  Região: {REGION_PLATFORM} / {REGION_ROUTING}")
    log(f"{'═' * 56}\n")

    # ── ETAPA 1 — Coleta de PUUIDs (Challenger + Grandmaster + Master) ─
    if p["stage"] == 1:
        t1 = log_stage_start("ETAPA 1/3 · Coleta de PUUIDs  (Challenger · Grandmaster · Master)")

        puuids, sources = collect_all_puuids()

        if not puuids:
            log("[ERRO FATAL] Não foi possível obter jogadores. Verifique a API Key.")
            raise SystemExit(1)

        # Resumo por tier
        tiers = {"Challenger": 0, "Grandmaster": 0, "Master": 0}
        for tier in sources.values():
            tiers[tier] += 1

        resumo = " | ".join(f"{t}: {c}" for t, c in tiers.items())
        resumo += f" | TOTAL: {len(puuids)}"
        print(f"\n  Resumo (únicos):")
        for tier, count in tiers.items():
            print(f"    {tier:<12} : {count}")
        print(f"    {'TOTAL':<12} : {len(puuids)}")

        p["puuids"]        = puuids
        p["puuid_sources"] = sources
        p["stage"]         = 2
        save_progress(p)

        log_stage_end("ETAPA 1/3 · Coleta de PUUIDs", t1, resumo)

    # ── ETAPA 2 — Coleta de Match IDs ─────────────────────────────────────────
    if p["stage"] == 2:
        t2 = log_stage_start(f"ETAPA 2/3 · Coleta de Match IDs  (até {MATCHES_PER_PLAYER} por jogador)")

        puuids        = p["puuids"]
        processed_set = set(p["processed_puuids"])
        unique_ids    = set(p["unique_match_ids"])
        total_players = len(puuids)
        pending       = total_players - len(processed_set)

        log(f"   Jogadores totais  : {total_players}")
        log(f"   Já processados    : {len(processed_set)}")
        log(f"   Pendentes         : {pending}\n")

        for idx, puuid in enumerate(puuids):
            if puuid in processed_set:
                continue

            tier = p["puuid_sources"].get(puuid, "?")
            ids  = get_match_ids(puuid)
            unique_ids.update(ids)
            processed_set.add(puuid)

            print(
                f"  [{tier:<12}] Jogador {idx + 1:>5}/{total_players} | "
                f"IDs únicos acumulados: {len(unique_ids):>8}",
                end="\r"
            )
            time.sleep(REQUEST_DELAY)

            # Checkpoint periódico
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                p["processed_puuids"] = list(processed_set)
                p["unique_match_ids"] = list(unique_ids)
                save_progress(p)

        # Salva estado final da etapa 2
        p["processed_puuids"] = list(processed_set)
        p["unique_match_ids"] = list(unique_ids)
        p["stage"]            = 3
        save_progress(p)

        resumo = f"{len(unique_ids)} Match IDs únicos de {total_players} jogadores"
        print(f"\n[OK] {resumo}")
        log_stage_end("ETAPA 2/3 · Coleta de Match IDs", t2, resumo)

    # ── ETAPA 3 — Dividir e salvar nos arquivos de saída ──────────────────────
    if p["stage"] == 3:
        t3 = log_stage_start(f"ETAPA 3/3 · Divisão em {NUM_OUTPUT_FILES} arquivos de saída")

        split_and_save(p["unique_match_ids"], NUM_OUTPUT_FILES)
        p["stage"] = "done"
        save_progress(p)

        total_ids = len(p["unique_match_ids"])
        resumo    = (f"{total_ids} IDs divididos em {NUM_OUTPUT_FILES} arquivos  "
                     f"(~{math.ceil(total_ids / NUM_OUTPUT_FILES)} por arquivo)")
        log_stage_end("ETAPA 3/3 · Divisão em arquivos", t3, resumo)

    # Rodapé geral
    log(f"{'═' * 56}")
    log(f"  🎉 Script 1 concluído com sucesso!")
    log(f"  Arquivos: {OUTPUT_PREFIX}_1.txt  …  {OUTPUT_PREFIX}_{NUM_OUTPUT_FILES}.txt")
    log(f"  Log completo salvo em: {LOG_FILE}")
    log(f"{'═' * 56}")