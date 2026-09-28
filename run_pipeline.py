"""
run_pipeline.py — Executa o pipeline completo do TCC em sequência e grava tudo em arquivo de log.

- A saída de cada script aparece no console e vai para o log ao mesmo tempo.
- Registra o início, o fim, a duração e o código de retorno de cada etapa.
- Por padrão, para na primeira falha. Use --keep-going para continuar mesmo com erro.
- Permite retomar a partir de uma etapa (--from 4) ou rodar etapas específicas (--only 6,7).

Uso:
    python run_pipeline.py
    python run_pipeline.py --from 4          # retoma a partir do 1º treino
    python run_pipeline.py --only 7,8        # só SHAP e relatório
    python run_pipeline.py --list            # lista as etapas
"""
import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("extract",         ["01_extract.py"]),
    ("features",        ["02_features.py", "--inp", "data/interim", "--out", "data/processed",
                         "--min-freq", "0.005"]),
    ("clusters",        ["02b_features.py"]),
    ("train_main",      ["03_train.py", "--budget-min", "480", "--trials", "50", "--outer", "5",
                         "--n-boot", "2000"]),
    ("train_lvl8",      ["03_train.py", "--min-level", "8",
                         "--drop-agg", "n_units,board_value,n_items,n_components",
                         "--out", "results_lvl8"]),
    ("train_no_units",  ["03_train.py", "--drop-blocks", "unit,unit_item",
                         "--out", "results_no_units"]),
    ("shap",            ["04_shap.py", "--res", "results", "--models", "lgbm", "--top-inter", "40", "--threads", "8"]),
    ("shap_lvl8",       ["04_shap.py", "--res", "results_lvl8", "--models", "lgbm", "--top-inter", "40", "--threads", "8"]),
    ("shap_no_units",   ["04_shap.py", "--res", "results_no_units", "--models", "lgbm", "--top-inter", "40", "--threads", "8"]),
    ("report",          ["05_report.py", "--res", "results", "--out", "report", "--model", "lgbm"]),
    ("report_lvl8",     ["05_report.py", "--res", "results_lvl8", "--out", "report_lvl8", "--model", "lgbm"]),
    ("report_no_units", ["05_report.py", "--res", "results_no_units", "--out", "report_no_units", "--model", "lgbm"]),
    ("sensitivity",     ["06_sensitivity.py"]),
]


def fmt_dur(s):
    return str(dt.timedelta(seconds=int(s)))


class Tee:
    """Escreve no console e no arquivo de log."""
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, msg, end="\n"):
        sys.stdout.write(msg + end); sys.stdout.flush()
        self.f.write(msg + end)

    def close(self):
        self.f.close()


def run_step(i, name, cmd, log, step_dir):
    full = [sys.executable, "-u", *cmd]
    step_log = step_dir / f"{i:02d}_{name}.log"
    log.write("=" * 90)
    log.write(f"[{i}/{len(STEPS)}] {name} | {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    log.write(f"$ {' '.join(cmd)}")
    log.write("-" * 90)
    t0 = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with open(step_log, "w", encoding="utf-8", buffering=1) as sf:
        p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
        try:
            for line in p.stdout:
                line = line.rstrip("\n")
                log.write(line); sf.write(line + "\n")
            rc = p.wait()
        except KeyboardInterrupt:
            p.terminate(); p.wait()
            log.write("!! Interrompido pelo usuário")
            raise
    dur = time.time() - t0
    status = "OK" if rc == 0 else f"FALHOU (código {rc})"
    log.write("-" * 90)
    log.write(f"[{i}/{len(STEPS)}] {name}: {status} | duração {fmt_dur(dur)} | log: {step_log}")
    return {"step": i, "name": name, "cmd": " ".join(cmd), "returncode": rc,
            "seconds": round(dur, 1), "log": str(step_log)}


def main(a):
    if a.list:
        for i, (n, c) in enumerate(STEPS, 1):
            print(f"{i}. {n:15s} python {' '.join(c)}")
        return 0

    sel = ([int(x) for x in a.only.split(",")] if a.only
           else list(range(a.start, len(STEPS) + 1)))
    for s in sel:
        if not 1 <= s <= len(STEPS):
            sys.exit(f"Etapa inválida: {s}")
    for s in sel:   # confere se os scripts existem antes de começar
        if not Path(STEPS[s - 1][1][0]).exists():
            sys.exit(f"Script não encontrado: {STEPS[s - 1][1][0]}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(a.log_dir) / stamp; log_dir.mkdir(parents=True, exist_ok=True)
    log = Tee(log_dir / "pipeline.log")
    log.write(f"Pipeline TCC TFT | início {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    log.write(f"Python {platform.python_version()} | {platform.platform()} | cwd={Path.cwd()}")
    log.write(f"Etapas: {sel} | keep-going={a.keep_going} | logs em {log_dir}")

    T0, results = time.time(), []
    try:
        for s in sel:
            name, cmd = STEPS[s - 1]
            r = run_step(s, name, cmd, log, log_dir)
            results.append(r)
            if r["returncode"] != 0 and not a.keep_going:
                log.write(f"!! Parando o pipeline. Para retomar: python run_pipeline.py --from {s}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        log.write("=" * 90)
        log.write("RESUMO")
        for r in results:
            st = "OK " if r["returncode"] == 0 else "ERRO"
            log.write(f"  [{st}] {r['step']}. {r['name']:15s} {fmt_dur(r['seconds']):>9s}")
        log.write(f"Tempo total: {fmt_dur(time.time() - T0)}")
        json.dump(results, open(log_dir / "summary.json", "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        log.close()
    return 0 if results and all(r["returncode"] == 0 for r in results) and \
        len(results) == len(sel) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1, help="etapa inicial (1-based)")
    ap.add_argument("--only", default="", help="etapas específicas, ex.: 7,8")
    ap.add_argument("--keep-going", action="store_true", help="continua mesmo se uma etapa falhar")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--list", action="store_true")
    sys.exit(main(ap.parse_args()))
