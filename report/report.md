# Resultados — Previsão de Top 4 em TFT

**Base:** 206.654 partidas, 1.653.232 observações, 811 atributos.

## Desempenho

- Melhor modelo: **LightGBM (B)**, AUC 0.942 [0.942; 0.942].
- LightGBM (A) (só composição): AUC 0.941 vs baseline de nível 0.741.
- Incluir o nível (B) altera a AUC em +0.001.

## Hipóteses

- H1 (compl.): traits > unidades [A]: diferença -0.248 [-0.263; -0.233] → **Não**
- H1: traits ouro/prismáticas > nº de traits ativas [A]: diferença -0.157 [-0.163; -0.151] → **Não**
- H2: 4-5 custos 2★ > custo total [A]: diferença 0.262 [0.254; 0.269] → **Sim**
- H3: itens nos carregadores > total de itens completos [A]: diferença -0.107 [-0.114; -0.100] → **Não**
- H1 (compl.): traits > unidades [B]: diferença -0.209 [-0.222; -0.196] → **Não**
- H1: traits ouro/prismáticas > nº de traits ativas [B]: diferença -0.136 [-0.141; -0.132] → **Não**
- H2: 4-5 custos 2★ > custo total [B]: diferença 0.218 [0.213; 0.224] → **Sim**
- H3: itens nos carregadores > total de itens completos [B]: diferença -0.060 [-0.067; -0.053] → **Não**

## Arquétipos

- Mais forte: **archetype_Ionia_TheBoss_Wukong_0** (Top 4 = 0.545)
- Mais fraco: **archetype_Piltover_Defender_Vi_1** (Top 4 = 0.443)

## Robustez

| Análise                                    | Spearman      | Jaccard@20    |
|:-------------------------------------------|:--------------|:--------------|
| Estabilidade entre folds — modelos OOF (A) | 0.933 ± 0.010 | 0.933         |
| Estabilidade entre folds — modelos OOF (B) | 0.942 ± 0.010 | 0.891         |
| SHAP vs coef. Logística (A)                | 0.596         | 0.250         |
| SHAP vs coef. Logística (B)                | 0.588         | 0.379         |
| Temporal: teste no patch 16.5 (A)          | AUC=0.939     | LogLoss=0.320 |
| Temporal: teste no patch 16.5 (B)          | AUC=0.940     | LogLoss=0.316 |

## Arquivos

- Tabelas LaTeX: `tables/` (use `\input{tables/metricas.tex}`)
- Figuras: `figures/`