"""
================================================================================
  SCRIPT 2 — DOWNLOAD DE PARTIDAS (worker individual)
  TFT Challenger KR

  Como usar:
    Cada uma das 5 pessoas recebe um arquivo de IDs diferente gerado pelo
    Script 1 (ex.: tft_match_ids_part_1.txt) e executa:

        python script2_download_matches.py --part 1
        python script2_download_matches.py --part 2
        ...
        python script2_download_matches.py --part 5

  Estrutura de arquivos gerada por cada worker (exemplo part 1):

    tft_worker_1/                        ← pasta exclusiva deste worker
    ├── batch_0001.json                  ← lote 1  (até 100 partidas)
    ├── batch_0002.json                  ← lote 2  (até 100 partidas)
    ├── batch_NNNN.json                  ← ...
    ├── tft_matches_part_1_final.json    ← arquivo consolidado (gerado ao final)
    └── tft_worker_1.log                 ← log de horários desta partição

    tft_download_progress_part_1.json    ← checkpoint deste worker

  Checkpoint:
    → Guarda quais IDs já foram baixados e em qual lote estamos.
    → Se perder conexão ou o script travar, basta rodar novamente —
      ele retoma do ponto exato onde parou, sem re-baixar nada.
================================================================================
"""

import argparse
import requests
import time
import json
import os
from datetime import datetime

# =============================================================================
# CONFIGURAÇÃO — edite aqui
# =============================================================================
API_KEY          = ""
REGION_ROUTING   = "asia"    # americas | asia | europe

BATCH_SIZE       = 100       # Partidas por arquivo de lote
REQUEST_DELAY    = 1.2       # Delay (s) entre chamadas (respeita rate limit)
MAX_RETRIES      = 5         # Tentativas antes de pular uma partida com erro
CHECKPOINT_EVERY = 25        # Salva checkpoint a cada N downloads com sucesso

IDS_PREFIX      = "tft_match_ids_part"          # Arquivo de IDs gerado pelo Script 1
WORKER_DIR      = "tft_worker_{part}"            # Pasta de trabalho do worker
BATCH_FILENAME  = "batch_{num:04d}.json"         # Lotes dentro da pasta
FINAL_FILENAME  = "tft_matches_part_{part}_final.json"  # Consolidado final
PROGRESS_PREFIX = "tft_download_progress_part"   # Arquivo de checkpoint

HEADERS = {"X-Riot-Token": API_KEY}

# =============================================================================
# ARGUMENTO DE LINHA DE COMANDO
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download TFT match details for one partition."
    )
    parser.add_argument(
        "--part", "-p",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Número da partição que este worker vai processar (1–10)."
    )
    return parser.parse_args()

# =============================================================================
# HELPERS DE TEMPO / LOG
# =============================================================================

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:   return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:   return f"{m}m {s:02d}s"
    return f"{s}s"

def log(msg: str, log_path: str):
    """Imprime no terminal e grava no arquivo de log do worker."""
    line = f"[{_now()}] {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def log_stage_start(label: str, log_path: str) -> float:
    log("─" * 58, log_path)
    log(f"▶  INÍCIO  │ {label}", log_path)
    log("─" * 58, log_path)
    return time.time()

def log_stage_end(label: str, start: float, log_path: str, extra: str = ""):
    log("─" * 58, log_path)
    log(f"■  FIM     │ {label}", log_path)
    log(f"   Duração │ {_elapsed(time.time() - start)}", log_path)
    if extra:
        log(f"   Resumo  │ {extra}", log_path)
    log("─" * 58 + "\n", log_path)

# =============================================================================
# CHECKPOINT
# =============================================================================

def load_progress(part: int) -> dict:
    """
    Estrutura do checkpoint:
      - downloaded_ids  : lista de IDs baixados com sucesso
      - failed_ids      : lista de IDs que falharam após MAX_RETRIES tentativas
      - current_batch   : número do lote em escrita no momento
      - batch_count     : quantas partidas já estão no lote atual
      - last_success_id : último ID salvo (para debug)
    """
    path = f"{PROGRESS_PREFIX}{part}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return {
        "part":            part,
        "downloaded_ids":  [],
        "failed_ids":      [],
        "current_batch":   1,
        "batch_count":     0,
        "last_success_id": None,
    }

def save_progress(p: dict):
    """Escrita atômica — salva em .tmp e renomeia para nunca corromper."""
    path = f"{PROGRESS_PREFIX}{p['part']}.json"
    tmp  = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# =============================================================================
# API — RIOT
# =============================================================================

def get_match_details(match_id: str, attempt: int = 1) -> dict | None:
    """
    Baixa os detalhes de uma partida com tratamento completo de erros:
      - 429 → aguarda Retry-After e retenta (sem contar como attempt)
      - 5xx → back-off exponencial até MAX_RETRIES
      - ConnectionError / Timeout → aguarda e retenta
      - 404 / 403 → retorna None imediatamente
    """
    if attempt > MAX_RETRIES:
        return None

    url = (f"https://{REGION_ROUTING}.api.riotgames.com"
           f"/tft/match/v1/matches/{match_id}")
    try:
        res = requests.get(url, headers=HEADERS, timeout=20)

        if res.status_code == 429:
            wait = int(res.headers.get("Retry-After", 30))
            print(f"\n  [Rate Limit] Aguardando {wait}s...")
            time.sleep(wait)
            return get_match_details(match_id, attempt)       # não incrementa attempt

        if res.status_code == 200:
            return res.json()

        if res.status_code >= 500:
            wait = 2 ** attempt
            print(f"\n  [Erro {res.status_code}] Tentativa {attempt}/{MAX_RETRIES} — aguardando {wait}s...")
            time.sleep(wait)
            return get_match_details(match_id, attempt + 1)

        return None   # 404, 403, etc.

    except requests.exceptions.ConnectionError:
        print(f"\n  [Sem conexão] Tentativa {attempt}/{MAX_RETRIES} — aguardando 15s...")
        time.sleep(15)
        return get_match_details(match_id, attempt + 1)

    except requests.exceptions.Timeout:
        print(f"\n  [Timeout] Tentativa {attempt}/{MAX_RETRIES} — aguardando 10s...")
        time.sleep(10)
        return get_match_details(match_id, attempt + 1)

    except Exception as e:
        print(f"\n  [Erro inesperado] {e}")
        return None

# =============================================================================
# GERENCIAMENTO DE LOTES
# =============================================================================

def batch_path(worker_dir: str, batch_num: int) -> str:
    return os.path.join(worker_dir, BATCH_FILENAME.format(num=batch_num))

def load_batch(worker_dir: str, batch_num: int) -> list:
    """Carrega lote existente do disco (para retomada segura)."""
    path = batch_path(worker_dir, batch_num)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_batch(worker_dir: str, batch_num: int, data: list):
    """Salva lote de forma atômica."""
    path = batch_path(worker_dir, batch_num)
    tmp  = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def count_existing_batches(worker_dir: str) -> int:
    if not os.path.exists(worker_dir):
        return 0
    return sum(
        1 for f in os.listdir(worker_dir)
        if f.startswith("batch_") and f.endswith(".json")
    )

# =============================================================================
# CONSOLIDAÇÃO FINAL
# =============================================================================

def consolidate(worker_dir: str, part: int, log_path: str) -> str:
    """
    Lê todos os lotes da pasta do worker, consolida em um único JSON
    e salva dentro da mesma pasta. Retorna o caminho do arquivo final.
    """
    t_start = log_stage_start("Consolidação dos lotes → arquivo final", log_path)

    batch_files = sorted(
        f for f in os.listdir(worker_dir)
        if f.startswith("batch_") and f.endswith(".json")
    )

    all_data = []
    for fname in batch_files:
        fpath = os.path.join(worker_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            batch = json.load(f)
        all_data.extend(batch)
        print(f"  Lendo {fname}  ({len(batch)} partidas)  total acumulado: {len(all_data)}", end="\r")

    print()  # quebra a linha do end="\r"

    final_path = os.path.join(worker_dir, FINAL_FILENAME.format(part=part))
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    resumo = (f"{len(all_data)} partidas consolidadas de {len(batch_files)} lotes"
              f" → {os.path.basename(final_path)}")
    log_stage_end("Consolidação", t_start, log_path, resumo)

    return final_path

# =============================================================================
# ENTRADA DE DADOS
# =============================================================================

def load_ids_from_file(part: int) -> list[str]:
    path = f"{IDS_PREFIX}_{part}.txt"
    if not os.path.exists(path):
        print(f"[ERRO FATAL] Arquivo de IDs não encontrado: {path}")
        print( "             Execute o Script 1 primeiro e copie o arquivo correto.")
        raise SystemExit(1)
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    print(f"[OK] {len(ids)} IDs carregados de '{path}'.")
    return ids

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    args = parse_args()
    part = args.part

    # Cria pasta do worker e define caminho do log
    worker_dir = WORKER_DIR.format(part=part)
    os.makedirs(worker_dir, exist_ok=True)
    log_path = os.path.join(worker_dir, f"tft_worker_{part}.log")

    # ── Cabeçalho ──────────────────────────────────────────────────────────────
    log("═" * 58, log_path)
    log(f"  TFT SCRIPT 2 — Worker  Partição {part}/5", log_path)
    log(f"  Pasta de trabalho : {worker_dir}/", log_path)
    log(f"  Tamanho dos lotes : {BATCH_SIZE} partidas por arquivo", log_path)
    log("═" * 58, log_path)

    # ── Carrega IDs e progresso ────────────────────────────────────────────────
    all_ids    = load_ids_from_file(part)
    p          = load_progress(part)
    downloaded = set(p["downloaded_ids"])
    failed     = set(p["failed_ids"])
    total      = len(all_ids)
    pending    = [m for m in all_ids if m not in downloaded and m not in failed]

    already_done     = len(downloaded)
    already_failed   = len(failed)
    existing_batches = count_existing_batches(worker_dir)

    log(f"  Total de IDs nesta partição : {total}", log_path)
    log(f"  Já baixados (checkpoint)    : {already_done}", log_path)
    log(f"  Falhas registradas          : {already_failed}", log_path)
    log(f"  Lotes já salvos em disco    : {existing_batches}", log_path)
    log(f"  Pendentes nesta sessão      : {len(pending)}", log_path)

    if not pending:
        log("✅ Todos os IDs já foram processados! Iniciando consolidação...", log_path)
        final = consolidate(worker_dir, part, log_path)
        log(f"🎉 Arquivo final gerado: {final}", log_path)
        raise SystemExit(0)

    # ── Download ───────────────────────────────────────────────────────────────
    t_download = log_stage_start("Download das partidas pendentes", log_path)

    # Retoma o lote atual do checkpoint (pode estar parcialmente preenchido)
    current_batch_num  = p["current_batch"]
    current_batch_data = load_batch(worker_dir, current_batch_num)

    # Se o lote atual já estava cheio, começa um novo
    if len(current_batch_data) >= BATCH_SIZE:
        current_batch_num += 1
        current_batch_data = []

    success_count = 0
    error_count   = 0

    for idx, match_id in enumerate(pending):

        data = get_match_details(match_id)

        if data:
            current_batch_data.append(data)
            downloaded.add(match_id)
            p["last_success_id"] = match_id
            success_count += 1

            # Lote cheio → salva e abre o próximo
            if len(current_batch_data) >= BATCH_SIZE:
                save_batch(worker_dir, current_batch_num, current_batch_data)
                print(
                    f"\n  ✔ Lote {current_batch_num:04d} salvo "
                    f"({BATCH_SIZE} partidas) → {batch_path(worker_dir, current_batch_num)}"
                )
                current_batch_num += 1
                current_batch_data = []
        else:
            failed.add(match_id)
            error_count += 1

        # Log inline de progresso
        total_done = already_done + success_count + already_failed + error_count
        print(
            f"  [Part {part}] {idx + 1:>6}/{len(pending)}"
            f"  |  Lote: {current_batch_num:04d} [{len(current_batch_data):>3}/{BATCH_SIZE}]"
            f"  |  ✔ {success_count + already_done:>6}  ✘ {error_count + already_failed:>4}"
            f"  |  Total: {total_done}/{total}",
            end="\r"
        )

        time.sleep(REQUEST_DELAY)

        # Checkpoint periódico — inclui flush do lote parcial
        if success_count % CHECKPOINT_EVERY == 0 and success_count > 0:
            if current_batch_data:
                save_batch(worker_dir, current_batch_num, current_batch_data)
            p["downloaded_ids"] = list(downloaded)
            p["failed_ids"]     = list(failed)
            p["current_batch"]  = current_batch_num
            p["batch_count"]    = len(current_batch_data)
            save_progress(p)

    # Salva o último lote (pode ser incompleto — é o lote final da partição)
    if current_batch_data:
        save_batch(worker_dir, current_batch_num, current_batch_data)
        print(
            f"\n  ✔ Lote {current_batch_num:04d} salvo "
            f"({len(current_batch_data)} partidas — lote final)"
        )

    # Checkpoint final
    p["downloaded_ids"] = list(downloaded)
    p["failed_ids"]     = list(failed)
    p["current_batch"]  = current_batch_num
    p["batch_count"]    = len(current_batch_data)
    save_progress(p)

    total_success = already_done + success_count
    total_errors  = already_failed + error_count
    total_batches = count_existing_batches(worker_dir)

    resumo = (f"✔ {success_count} baixadas nesta sessão ({total_success} total) | "
              f"✘ {error_count} erros ({total_errors} total) | "
              f"{total_batches} lotes em disco")
    log_stage_end("Download das partidas", t_download, log_path, resumo)

    if total_errors > 0:
        log(f"⚠  {total_errors} IDs falharam após {MAX_RETRIES} tentativas.", log_path)
        log( "   Para re-tentar: remova-os de 'failed_ids' no checkpoint e execute novamente.", log_path)

    # ── Consolidação final ─────────────────────────────────────────────────────
    final_path = consolidate(worker_dir, part, log_path)

    # ── Rodapé ─────────────────────────────────────────────────────────────────
    log("═" * 58, log_path)
    log(f"  🎉 Worker {part} finalizado com sucesso!", log_path)
    log(f"  Lotes salvos     : {worker_dir}/batch_0001.json … batch_{total_batches:04d}.json", log_path)
    log(f"  Arquivo final    : {final_path}", log_path)
    log(f"  Log              : {log_path}", log_path)
    log("═" * 58, log_path)