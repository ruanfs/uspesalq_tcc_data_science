"""
run_pipeline.py v4 — Pipeline 01 → 05 com teto rígido de 12 h.
Uso:
  python run_pipeline.py                  # todas as etapas
  python run_pipeline.py --from 4         # retoma no treino (dados já extraídos)
  python run_pipeline.py --list
Teto: um watchdog mata a etapa que ultrapassar o tempo restante. As etapas centrais rodam
primeiro (treino + SHAP), seguidas das de sensibilidade. O relatório sempre roda, com uma
reserva de 15 min. O requirements.txt é gerado no início, com as versões instaladas.
"""
import argparse, datetime as dt, json, os, platform, subprocess, sys, threading, time
from importlib import metadata
from pathlib import Path

PKGS = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "optuna", "pyarrow",
        "joblib", "shap", "matplotlib", "tqdm", "orjson"]

# Parâmetros comuns de treino, enxutos para caber em 12 h.
# Ablações herdam os hiperparâmetros da base (sem tuning).
TRAIN = ["--trials", "6", "--tune-frac", "0.10", "--lgbm-lr-min", "0.06", "--es-rounds", "100",
         "--rf-trees", "300", "--rf-max-samples", "0.30", "--trials-slow", "6",
         "--tune-timeout-min", "25"]
ABL_H = ["abl_H1_noQ", "abl_H1_noR", "abl_H1b_noQ", "abl_H1b_noR",
         "abl_H2_noQ", "abl_H2_noR", "abl_H3_noQ", "abl_H3_noR"]
SPECS = ",".join(["base_level", "base_board_value", "agg_lgbm", "agg+sparse_lgbm", "lgbm_A",
                  "lgbm_C", "rf_A", "abl_agg_trait", "abl_agg_unit", "abl_agg_item", *ABL_H])
CORE = {"extract", "features", "clusters", "train_main"}   # se falhar, o resto não faz sentido

STEPS = [
    ("extract", ["01_extract.py"]),
    ("features", ["02_features.py", "--inp", "data/interim", "--out", "data/processed",
                  "--min-freq", "0.005"]),
    ("clusters", ["02b_features.py"]),
    ("train_main", ["03_train.py", "--only", SPECS, "--models", "lgbm,rf", "--outer", "5", *TRAIN,
                    "--n-boot", "500", "--ablations", "ablations.json", "--lambdamart",
                    "--out", "results"]),
    ("shap", ["04_shap.py", "--res", "results", "--specs", "lgbm_A,lgbm_C", "--n-shap", "50000",
              "--n-stab", "10000", "--n-inter", "2000", "--top-inter", "30", "--n-boot", "500"]),
    ("train_logreg", ["03_train.py", "--only", "logreg_A", "--models", "logreg", "--outer", "3",
                      "--trials-slow", "3", "--tune-frac", "0.1", "--tune-timeout-min", "10",
                      "--logreg-max-iter", "5000", "--n-boot", "300", "--skip-puuid",
                      "--out", "results_logreg"]),
    ("train_lvl8", ["03_train.py", "--only", "base_level,agg_lgbm,lgbm_A," + ",".join(ABL_H),
                    "--models", "lgbm", "--outer", "3", *TRAIN, "--n-boot", "300",
                    "--ablations", "ablations.json", "--min-level", "8", "--skip-puuid",
                    "--out", "results_lvl8"]),
    ("train_mid", ["03_train.py", "--only", "lgbm_A", "--models", "lgbm", "--outer", "3", *TRAIN,
                   "--n-boot", "300", "--placements", "3,6", "--skip-puuid", "--out", "results_mid"]),
    ("report", ["05_report.py", "--res", "results", "--sens", "results_lvl8",
                "--logreg", "results_logreg", "--mid", "results_mid", "--inp", "data/processed",
                "--interim", "data/interim", "--out", "report"]),
]


def fmt(s): return str(dt.timedelta(seconds=int(max(s, 0))))


class Tee:
    def __init__(self, p): self.f = open(p, "a", encoding="utf-8", buffering=1)
    def write(self, m): sys.stdout.write(m + "\n"); sys.stdout.flush(); self.f.write(m + "\n")
    def close(self): self.f.close()


def freeze(path="requirements.txt"):
    L = [f"# gerado em {dt.datetime.now():%Y-%m-%d %H:%M} | Python {platform.python_version()} "
         f"| {platform.platform()}"]
    for p in PKGS:
        try: L.append(f"{p}=={metadata.version(p)}")
        except metadata.PackageNotFoundError: L.append(f"# {p}: não instalado")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def run_step(i, name, cmd, log, d, timeout):
    sl = d / f"{i:02d}_{name}.log"
    log.write("=" * 90)
    log.write(f"[{i}] {name} | {dt.datetime.now():%H:%M:%S} | limite {fmt(timeout)}")
    log.write("$ python " + " ".join(cmd)); t0, hit = time.time(), []
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with open(sl, "w", encoding="utf-8", buffering=1) as sf:
        p = subprocess.Popen([sys.executable, "-u", *cmd], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", bufsize=1, env=env)
        timer = threading.Timer(timeout, lambda: (hit.append(1), p.kill())); timer.start()
        try:
            for line in p.stdout:
                line = line.rstrip("\n"); log.write(line); sf.write(line + "\n")
            rc = p.wait()
        except KeyboardInterrupt:
            p.terminate(); p.wait(); log.write("!! interrompido"); raise
        finally:
            timer.cancel()
    dur = time.time() - t0
    st = "TEMPO ESGOTADO" if hit else ("OK" if rc == 0 else f"FALHOU ({rc})")
    log.write(f"[{i}] {name}: {st} | {fmt(dur)}")
    return {"step": i, "name": name, "returncode": rc, "timeout": bool(hit),
            "seconds": round(dur, 1), "log": str(sl)}


def main(a):
    if a.list:
        for i, (n, c) in enumerate(STEPS, 1): print(f"{i}. {n:13s} python {' '.join(c)}")
        return 0
    sel = [int(x) for x in a.only.split(",")] if a.only else list(range(a.start, len(STEPS) + 1))
    for s in sel:
        if not Path(STEPS[s - 1][1][0]).exists(): sys.exit(f"Script ausente: {STEPS[s - 1][1][0]}")
    if not Path("ablations.json").exists(): sys.exit("ablations.json ausente")

    d = Path(a.log_dir) / dt.datetime.now().strftime("%Y%m%d_%H%M%S"); d.mkdir(parents=True)
    log = Tee(d / "pipeline.log")
    log.write(f"Pipeline v4 | Python {platform.python_version()} | etapas {sel} | teto {a.budget_h} h")
    log.write(f"requirements.txt: {len(freeze()) - 1} pacotes")
    T0, res, stop = time.time(), [], False
    deadline = T0 + a.budget_h * 3600
    try:
        for s in sel:
            n, c = STEPS[s - 1]; rep = n == "report"
            if stop and not rep: continue
            if rep and not Path("results/metrics.csv").exists():
                log.write("-- report: pulado (results/metrics.csv ausente)"); continue
            rem = deadline - time.time() - (0 if rep else a.reserve_min * 60)
            if rem < 120:
                log.write(f"-- {n}: pulada por falta de tempo"); continue
            r = run_step(s, n, c, log, d, rem); res.append(r)
            if r["returncode"] and n in CORE and not a.keep_going:
                log.write(f"!! etapa central falhou. Retome: python run_pipeline.py --from {s}")
                stop = True
    except KeyboardInterrupt:
        pass
    finally:
        log.write("=" * 90)
        for r in res:
            tag = "OK " if r["returncode"] == 0 else ("TEMPO" if r["timeout"] else "ERRO")
            log.write(f"  [{tag}] {r['step']}. {r['name']:13s} {fmt(r['seconds'])}")
        log.write(f"Total: {fmt(time.time() - T0)} de {a.budget_h} h")
        json.dump(res, open(d / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        log.close()
    return 0 if res and len(res) == len(sel) and all(r["returncode"] == 0 for r in res) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--budget-h", type=float, default=16)
    ap.add_argument("--reserve-min", type=float, default=15, help="reserva para o relatório")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--list", action="store_true")
    sys.exit(main(ap.parse_args()))
