# Resultados — Previsão de Top 4 em TFT

**Base:** 206.654 partidas, 1.653.232 observações, 811 atributos.

**Recorte:** level ≥ 8 → 1.551.450 observações (Top 4 = 0.526).

**Removidos:** features ['n_units', 'board_value', 'n_items', 'n_components'] | blocos —.

## Desempenho

- Melhor modelo: **LightGBM (B)**, AUC 0.938 [0.938; 0.938].
- LightGBM (A) (só composição): AUC 0.937 vs baseline de nível 0.718.
- Incluir o nível (B) altera a AUC em +0.001.

## Hipóteses

- H1 (compl.): traits > unidades [A]: diferença -0.250 [-0.269; -0.232] → **Não**
- H1: traits ouro/prismáticas > nº de traits ativas [A]: diferença -0.187 [-0.194; -0.179] → **Não**
- H2: 4-5 custos 2★ > custo total [A]: diferença -0.167 [-0.179; -0.155] → **Não**
- H3: itens nos carregadores > total de itens completos [A]: diferença -0.425 [-0.439; -0.411] → **Não**
- H1 (compl.): traits > unidades [B]: diferença -0.216 [-0.235; -0.197] → **Não**
- H1: traits ouro/prismáticas > nº de traits ativas [B]: diferença -0.155 [-0.161; -0.148] → **Não**
- H2: 4-5 custos 2★ > custo total [B]: diferença 0.184 [0.174; 0.193] → **Sim**
- H3: itens nos carregadores > total de itens completos [B]: diferença -0.436 [-0.448; -0.424] → **Não**

## Arquétipos

- Mais forte: **archetype_Ionia_TheBoss_Wukong_0** (Top 4 = 0.545)
- Mais fraco: **archetype_Piltover_Defender_Vi_1** (Top 4 = 0.443)

## Robustez

| Análise                                    | Spearman      | Jaccard@20    |
|:-------------------------------------------|:--------------|:--------------|
| Estabilidade entre folds — modelos OOF (A) | 0.899 ± 0.004 | 0.937         |
| Estabilidade entre folds — modelos OOF (B) | 0.895 ± 0.006 | 1.000         |
| SHAP vs coef. Logística (A)                | 0.562         | 0.333         |
| SHAP vs coef. Logística (B)                | 0.564         | 0.290         |
| Temporal: teste no patch 16.5 (A)          | AUC=0.936     | LogLoss=0.329 |
| Temporal: teste no patch 16.5 (B)          | AUC=0.937     | LogLoss=0.325 |

## Arquivos

- Tabelas LaTeX: `tables/` (use `\input{tables/metricas.tex}`)
- Figuras: `figures/`