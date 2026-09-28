# Comparação Cenário A vs B

## Desempenho (B − A)

- Reg. Logística L1 | Principal: ΔAUC +0.0070 [+0.0069; +0.0072] (DeLong Holm p=<0,001)
- Random Forest | Principal: ΔAUC +0.0027 [+0.0027; +0.0028] (DeLong Holm p=<0,001)
- LightGBM | Principal: ΔAUC +0.0008 [+0.0008; +0.0009] (DeLong Holm p=<0,001)
- Reg. Logística L1 | Level ≥ 8: ΔAUC +0.0098 [+0.0096; +0.0100] (DeLong Holm p=<0,001)
- Random Forest | Level ≥ 8: ΔAUC +0.0038 [+0.0037; +0.0039] (DeLong Holm p=<0,001)
- LightGBM | Level ≥ 8: ΔAUC +0.0008 [+0.0008; +0.0009] (DeLong Holm p=<0,001)
- Reg. Logística L1 | Sem unidades: ΔAUC +0.0108 [+0.0106; +0.0110] (DeLong Holm p=<0,001)
- Random Forest | Sem unidades: ΔAUC +0.0027 [+0.0026; +0.0028] (DeLong Holm p=<0,001)
- LightGBM | Sem unidades: ΔAUC +0.0011 [+0.0011; +0.0012] (DeLong Holm p=<0,001)

## SHAP

| Rodada       |   Spearman (ranking A vs B) |   Jaccard@20 | Participação do level em B   | Importância da composição retida em B   |
|:-------------|----------------------------:|-------------:|:-----------------------------|:----------------------------------------|
| Principal    |                       0.982 |        0.739 | 6.3%                         | 92.3%                                   |
| Level ≥ 8    |                       0.978 |        0.818 | 6.2%                         | 88.3%                                   |
| Sem unidades |                       0.971 |        0.818 | 7.9%                         | 92.2%                                   |

## Hipóteses

| Rodada       | Hipótese                            | Dif. A [IC 95%]         | Dif. B [IC 95%]         | Conclusão           |
|:-------------|:------------------------------------|:------------------------|:------------------------|:--------------------|
| Principal    | H1 compl.: traits > unidades        | -0.248 [-0.263; -0.233] | -0.209 [-0.222; -0.196] | Não sustentada      |
| Principal    | H1: traits ouro+ > nº traits ativas | -0.157 [-0.163; -0.151] | -0.136 [-0.141; -0.132] | Não sustentada      |
| Principal    | H2: 4-5 custos 2★ > custo total     | +0.262 [+0.254; +0.269] | +0.218 [+0.213; +0.224] | Sustentada em A e B |
| Principal    | H3: itens em carregadores > total   | -0.107 [-0.114; -0.100] | -0.060 [-0.067; -0.053] | Não sustentada      |
| Level ≥ 8    | H1 compl.: traits > unidades        | -0.250 [-0.269; -0.232] | -0.216 [-0.235; -0.197] | Não sustentada      |
| Level ≥ 8    | H1: traits ouro+ > nº traits ativas | -0.187 [-0.194; -0.179] | -0.155 [-0.161; -0.148] | Não sustentada      |
| Level ≥ 8    | H2: 4-5 custos 2★ > custo total     | -0.167 [-0.179; -0.155] | +0.184 [+0.174; +0.193] | Só em B             |
| Level ≥ 8    | H3: itens em carregadores > total   | -0.425 [-0.439; -0.411] | -0.436 [-0.448; -0.424] | Não sustentada      |
| Sem unidades | H1: traits ouro+ > nº traits ativas | -0.202 [-0.210; -0.195] | -0.191 [-0.198; -0.185] | Não sustentada      |
| Sem unidades | H2: 4-5 custos 2★ > custo total     | +0.346 [+0.339; +0.353] | +0.302 [+0.294; +0.308] | Sustentada em A e B |
| Sem unidades | H3: itens em carregadores > total   | -0.165 [-0.173; -0.157] | -0.147 [-0.154; -0.138] | Não sustentada      |