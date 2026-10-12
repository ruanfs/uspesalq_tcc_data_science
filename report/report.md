# Resultados — Previsão de Top 4 em TFT (16.5 out-of-time)

**Configuração:** {"inp": "data/processed", "out": "results", "models": "lgbm,rf", "only": "base_level,base_board_value,agg_lgbm,agg+sparse_lgbm,lgbm_A,lgbm_C,rf_A,abl_agg_trait,abl_agg_unit,abl_agg_item,abl_H1_noQ,abl_H1_noR,abl_H1b_noQ,abl_H1b_noR,abl_H2_noQ,abl_H2_noR,abl_H3_noQ,abl_H3_noR", "ablations": "ablations.json", "trials": 6, "trials_slow": 6, "tune_timeout_min": 25.0, "outer": 5, "inner": 3, "tune_frac": 0.1, "n_boot": 500, "logreg_max_iter": 5000, "rf_trees": 300, "rf_max_samples": 0.3, "min_level": 0, "drop_agg": "", "drop_blocks": "", "skip_puuid": false, "abl_retune": false, "lgbm_lr_min": 0.06, "es_rounds": 100, "placements": "", "lambdamart": true}

## Desempenho

- Melhor: **LGBM: agregados + esparsos**, AUC 0.9379 [0.9371; 0.9387].
- LGBM: agregados + esparsos − LGBM: agregados: ΔAUC +0.0143 [+0.0138; +0.0148]
- LGBM: Cenário A − LGBM: Cenário C (sem volume): ΔAUC +0.0021 [+0.0019; +0.0024]
- Agregados + traits − LGBM: agregados: ΔAUC +0.0103 [+0.0098; +0.0107]
- Agregados + unidades − LGBM: agregados: ΔAUC +0.0115 [+0.0109; +0.0119]
- Agregados + itens − LGBM: agregados: ΔAUC +0.0104 [+0.0100; +0.0109]

## Hipóteses (ablação)

- H1 (compl.): traits vs unidades: ΔAUC 16.5 +0.0000 [-0.0003; +0.0003] → **Sem diferença**
- H1: traits ouro/prismáticas vs nº de traits ativas: ΔAUC 16.5 -0.0008 [-0.0010; -0.0006] → **R > Q**
- H2: 4-5 custos 2★ vs custo total: ΔAUC 16.5 +0.0002 [+0.0000; +0.0004] → **Q > R**
- H3: itens nos carregadores vs total de itens: ΔAUC 16.5 +0.0007 [+0.0005; +0.0010] → **Q > R**

**Level ≥ 8:**
- H1 (compl.): traits vs unidades: ΔAUC 16.5 +0.0003 [-0.0001; +0.0006] → **Sem diferença**
- H1: traits ouro/prismáticas vs nº de traits ativas: ΔAUC 16.5 -0.0004 [-0.0006; -0.0002] → **R > Q**
- H2: 4-5 custos 2★ vs custo total: ΔAUC 16.5 +0.0004 [+0.0002; +0.0006] → **Q > R**
- H3: itens nos carregadores vs total de itens: ΔAUC 16.5 +0.0008 [+0.0006; +0.0010] → **Q > R**

## Erros

- AUC geral 0.9370 | só colocações 3–6: 0.8228 (fronteira do Top 4)

## Auditoria EmptyBag/ThiefsGloves

- EmptyBag: 12.62% no 1º contra 0.01% no 5º–8º | AUC isolada 0.517 | corr com a última rodada 0.144
- ThiefsGloves: 41.21% no 1º contra 30.63% no 5º–8º | AUC isolada 0.540 | corr com a última rodada 0.116

## Convergência

- Logística: 4/4 ajustes convergiram (n_iter máx. 70)

## Robustez

- Estabilidade SHAP entre folds (LGBM: Cenário A): Spearman 0.941; Jaccard@20 0.840
- Estabilidade SHAP entre folds (LGBM: Cenário C (sem volume)): Spearman 0.954; Jaccard@20 0.830
- Level ≥ 8: LGBM: Cenário A: AUC 0.9349 [0.9340; 0.9358]
- Level ≥ 8: LGBM: agregados: AUC 0.9212 [0.9203; 0.9222]
- Level ≥ 8: Baseline: nível: AUC 0.6875 [0.6852; 0.6893]
- Logística: Logística L1: Cenário A: AUC 0.9164 [0.9153; 0.9174]
- Colocações 3–6: LGBM: Cenário A: AUC 0.8231 [0.8211; 0.8254]
- LambdaMART (Cenário A) − LGBM: Cenário A, 16.5: AUC 0.9300 (Δ -0.0070, DeLong p = <0,001); Spearman 0.8762 vs 0.8812; NDCG@4 0.9180 vs 0.9058