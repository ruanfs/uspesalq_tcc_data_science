# Resultados — Previsão de Top 4 em TFT

**Base:** 206.654 partidas, 1.653.232 observações, 811 atributos.

**Removidos:** features — | blocos ['unit', 'unit_item'].

## Desempenho

- Melhor modelo: **LightGBM (B)**, AUC 0.940 [0.939; 0.940].
- LightGBM (A) (só composição): AUC 0.938 vs baseline de nível 0.741.
- Incluir o nível (B) altera a AUC em +0.001.

## Hipóteses

- H1: traits ouro/prismáticas > nº de traits ativas [A]: diferença -0.202 [-0.210; -0.195] → **Não**
- H2: 4-5 custos 2★ > custo total [A]: diferença 0.346 [0.339; 0.353] → **Sim**
- H3: itens nos carregadores > total de itens completos [A]: diferença -0.165 [-0.173; -0.157] → **Não**
- H1: traits ouro/prismáticas > nº de traits ativas [B]: diferença -0.191 [-0.198; -0.185] → **Não**
- H2: 4-5 custos 2★ > custo total [B]: diferença 0.302 [0.294; 0.308] → **Sim**
- H3: itens nos carregadores > total de itens completos [B]: diferença -0.147 [-0.154; -0.138] → **Não**

## Arquétipos

- Mais forte: **archetype_Ionia_TheBoss_Wukong_0** (Top 4 = 0.545)
- Mais fraco: **archetype_Piltover_Defender_Vi_1** (Top 4 = 0.443)

## Robustez

| Análise                                    | Spearman      | Jaccard@20    |
|:-------------------------------------------|:--------------|:--------------|
| Estabilidade entre folds — modelos OOF (A) | 0.988 ± 0.002 | 1.000         |
| Estabilidade entre folds — modelos OOF (B) | 0.987 ± 0.003 | 1.000         |
| SHAP vs coef. Logística (A)                | 0.555         | 0.212         |
| SHAP vs coef. Logística (B)                | 0.526         | 0.290         |
| Temporal: teste no patch 16.5 (A)          | AUC=0.938     | LogLoss=0.319 |
| Temporal: teste no patch 16.5 (B)          | AUC=0.939     | LogLoss=0.317 |

## Arquivos

- Tabelas LaTeX: `tables/` (use `\input{tables/metricas.tex}`)
- Figuras: `figures/`