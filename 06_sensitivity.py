from pathlib import Path
import pandas as pd

R = {"Principal": "results", "Level ≥ 8": "results_lvl8", "Sem unidades": "results_no_units"}
rows = []
for nome, d in R.items():
    f = Path(d) / "metrics.csv"
    if not f.exists():
        print(f"[aviso] {f} não encontrado, pulando"); continue
    m = pd.read_csv(f).set_index("model")
    for k in ("baseline_level", "lgbm_A", "lgbm_B"):
        if k in m.index:
            r = m.loc[k]
            rows.append({"Rodada": nome, "Modelo": k,
                         "AUC [IC 95%]": f"{r.auc:.3f} [{r.auc_lo:.3f}; {r.auc_hi:.3f}]",
                         "Log loss": f"{r.logloss:.3f}", "Brier": f"{r.brier:.3f}"})
Path("report/tables").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_latex("report/tables/sensibilidade.tex", index=False, escape=True,
                            caption="Análises de sensibilidade.", label="tab:sens")
