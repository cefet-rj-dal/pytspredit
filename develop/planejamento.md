# Planejamento — Port do tspredit (R) para pytspredit (Python)

> Documento vivo. Atualizado a cada checkpoint. Mesmo método validado no
> **pydaltoolbox** (já publicado no PyPI). Objetivo: você saber todos os passos,
> o que já foi feito, o que vem, e por quê — sem surpresas.

Última atualização: 2026-11-18

---

## 1. O todo: o ecossistema DAL

```
          pydaltoolbox  (fundação: Linha de Experimento) — no PyPI
                 │
        ┌────────┴────────┐
   pytspredit         pyharbinger
 (previsão de TS)   (detecção de eventos em TS)
        (ESTE)        (depende dos dois)
```

`tspredit` depende **apenas** do `daltoolbox` → `pytspredit` depende apenas do
`pydaltoolbox` (já no PyPI). Este é o passo natural **antes** do `pyharbinger`.

Princípio central (herdado): **não reimplementar algoritmos** — envolver
`scikit-learn`/`numpy`/`pandas`/`statsmodels` com a interface uniforme da Linha de
Experimento (`fit`/`action`/`predict`/`transform`/`evaluate`/`set_params`). Nossa
contribuição é a **interface**, não os algoritmos.

---

## 2. Decisões travadas (2026-11-18)

| Tema | Decisão |
|---|---|
| Fundação | Reusar `pydaltoolbox` do PyPI (`Regression`, `Predictor`, `DalTransform`, `DalTune`, `adjust_*`). |
| Estilo de API | Métodos de instância; `fit` retorna `self` (encadeável). |
| Nomenclatura | **Classes** em PascalCase (`TsReg`, `TsKnn`, `TsNormGminmax`, `TsLagmap`); **funções construtoras de dados** em snake_case (`ts_data`, `ts_projection`, `ts_sample`) — idiomático (são funções, como no R). |
| Modelo de dados | Representação ts_data = `pandas.DataFrame` com colunas `t{sw-1}..t0` (plano, sem subclasse — facilita a aritmética das normalizações). |
| **Lag map** | **Portar as 12 estratégias na v1** (recent/even/geom/acf/pacf/peaks/seasonal/acf_seasonal/pacf_seasonal/blocks/mi/mrmr) — fidelidade total ao R. |
| **ARIMA/VAR** | **statsmodels = dependência core** da v1. `ts_arima`/`ts_arimax` (univariados) entram; `ts_var` é multivariado → vai com o bloco MV. |
| **Multivariado** | **Adiado para o Bloco B** (`ts_data_mv`, `ts_reg_mv`, `ts_regsw_mv`, `ts_lm_mv`, `ts_var`, `ts_plot_mv`). |
| Dependências v1 | `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `pydaltoolbox`. |
| Fidelidade | Adapters fiéis aos defaults do backend R; **comparar resultado R vs Python** nos pontos sensíveis. |
| Publicação | Um único release (`0.1.0`) ao final da última fase. Eduardo faz/autoriza o upload (token `daldeploy` de conta inteira, já no `.pypirc`). PyPI sem prazo; versões imutáveis; beta até `1.0.0`. |
| Workflow | Branch por fase → push → Eduardo revisa no GitHub → ff-merge na `main`. `gh` não instalado. Verde sempre (ruff+pytest+build dos exemplos). |

---

## 3. Arquitetura a portar (árvore)

```
pydaltoolbox.Predictor
└── TsReg ───────── evaluate = MSE/sMAPE/R2 ; baseline predict = t0
    └── TsRegSW ─── preprocess (TsNorm*) + input_map (TsLagmap) + backend
        ├── TsKnn   → KNeighborsRegressor
        ├── TsMlp   → MLPRegressor
        ├── TsRf    → RandomForestRegressor
        ├── TsSvm   → SVR
        ├── TsElm   → ELM (implementação própria, leve)
        └── ...
    └── TsArima / TsArimax → statsmodels (operam na série, sem janela)

pydaltoolbox.DalTransform
├── TsNormNone / TsNormGminmax / TsNormSwminmax / TsNormDiff / TsNormAn
├── TsFil* (filtros)      — ma, ema, ses, qes, kalman, lowess, spline, winsor, hp...
└── TsAug* (augmentation) — jitter, flip, stretch, shrink, wormhole, awareness...

pydaltoolbox.DalBase
└── TsLagmap  — 12 estratégias de seleção de lag (fit aprende as posições)

pydaltoolbox.DalTune
└── TsTune / TsIntegtune — CV por janelas, minimiza erro
```

---

## 4. Estratégia de garantia (três camadas)

1. **Unitário** → `pytest` por componente (`fit → predict/transform → evaluate`).
2. **Integrado** → exemplos ponta-a-ponta (tutoriais).
3. **Documentação executável** → `notebooks/**/*.md` (jupytext) renderizado em
   `examples/**/*.md` (saída real embutida) por `tools/build_examples.py`. Falha se
   houver erro → é smoke test. Tabelas em texto puro (estilo `##` do R). **Um
   exemplo didático por classe** (objetivo + referência do artigo + fit/predict/evaluate).

**Comandos:** `pip install -e ".[dev,docs]"` · `ruff check .` · `pytest` ·
`python tools/build_examples.py`.

---

## 5. Roadmap

Legenda: ✅ feito · 🔜 próximo · ⬜ pendente (v1) · 🔭 Bloco B (pós-v1)

### BLOCO A — v1 publicável

#### ✅ Fase 0 — Scaffolding
`pyproject` (deps incl. `pydaltoolbox`+`statsmodels`), layout `src/pytspredit/`,
`tools/build_examples.py`, `develop/` (PROMPT+planejamento+deploy), README.

#### ✅ Fase 1 — Núcleo de dados
`ts_data` (janela deslizante), `ts_projection`, `ts_sample`, `adjust_ts_data`,
`ts_head`, `datasets.tsd()` (CSV extraído do `.RData` — idêntico ao R). Testes +
notebooks `tsdata/`. **Verificado R vs Python: dim/colunas/valores idênticos.**

> **Meta: publicar `0.1.0` HOJE.** Para caber em um dia, consolidei o roadmap em
> **milestones** (não mais 10 fases com 10 pausas). Resta decidir **o escopo de
> hoje** (ver §5.1) e a **cadência de revisão** (ver §5.2).

### 5.1 Escopo do 0.1.0 de hoje — ✅ DECIDIDO: Trilha A (Enxuta) + pausar a cada milestone (M2→M3→M4)

Duas trilhas. A recomendada é a **Enxuta**: entrega o *pipeline de previsão
univariado completo* (o contrato central e o que o pyharbinger consome), e deixa
filtros + augmentation como *fast-follow* `0.2.0` (poucos dias depois).

**TRILHA A — Enxuta (recomendada p/ hoje): 3 milestones restantes**

| # | Milestone | Conteúdo |
|---|---|---|
| ✅ M0 | Scaffolding | feito |
| ✅ M1 | Núcleo de dados | feito |
| ✅ **M2** | Substrato de modelagem | `TsReg` + `TsRegSW` (com `steps_ahead`) + **normalizações (5)** + `TsLagmap` (**12 estratégias**) — **paridade R vs Py verificada** (norms ≤1e-13, 12 lagmaps idênticos, evaluate idêntico) |
| ✅ **M3** | Regressores | `TsKnn`, `TsMlp`, `TsRf`, `TsSvm`, `TsElm` + `TsArima` (univariado) — **paridade R vs Py end-to-end verificada** (ver abaixo). *(`TsArimax` é multivariado → Bloco B.)* |
| 🔜 **M4** | Tuning + fecho → 🚀 publica | `TsTune`, `TsIntegtune` + revisão API/README + build/twine/install limpo + **upload `0.1.0`** |

**Paridade M3 (tsd, previsão 5 passos, R vs Python):** knn `max|Δ|=4.7e-9` (exato) ·
elm `4.8e-9` (exato, equivale a OLS) · svm `2.5e-4` · arima `6.0e-4` · rf `3.6e-2` ·
mlp `2.2e-2`. Determinísticos (knn, elm) batem; svm/arima quase exatos; rf/mlp dentro
da tolerância de RNG do backend (esperado).

→ Filtros (13) e augmentation (8) viram **`0.2.0`** (uma milestone cada, dias depois).

**TRILHA B — Completa hoje: 4 milestones restantes**

Igual à Enxuta, mas insere antes do fecho:

| **M4** | Filtros + Augmentation | 13 filtros simples + 8 augmentations |
| **M5** | Tuning + fecho → 🚀 publica | idem M4 da trilha A |

> Risco da Trilha B: ~21 classes a mais com exemplo+teste+comparação R/Python no
> mesmo dia. Factível, mas aperta e sobra menos revisão sua. Por isso recomendo a A.

### 5.2 Cadência de revisão — DECISÃO EM ABERTO
- **(i) Pausar a cada milestone** (você revisa M2, M3, M4 no GitHub) — mais controle.
- **(ii) Correr direto até UMA revisão final** antes do `twine upload` — mais rápido;
  você aprova o pacote inteiro de uma vez. **Recomendada para fechar hoje.**

### 5.3 Convenção de exemplos (ajuste para caber hoje)
Mantemos **um exemplo por classe**, mas **agrupados por família num notebook** (ex.:
`normalization/` num só `.md` com uma seção por norma; idem regressores), em vez de
40 arquivos. Reduz o volume de docs sem perder o “didático por classe”.

### BLOCO B — pós-v1 (lib extra/nicho ou complexidade)
- 🔭 **Filtros/Augmentation** (se formos pela Trilha A → viram o `0.2.0`).
- 🔭 **Multivariado:** `ts_data_mv`, `ts_reg_mv`, `ts_regsw_mv`, `ts_lm_mv`, `ts_var`, `ts_plot_mv`.
- 🔭 **Filtros avançados:** `ts_fil_emd`/`remd` (PyEMD), `wavelet` (PyWavelets),
  `gabor`, `fft`, state-space avançado (KFAS).
- 🔭 **ARIMA avançado / MV:** `ts_arimax` (multivariado, xreg), `ts_darima` (distributed ARIMA), `ts_warma`, `ts_deterministic`.
- 🔭 **Benchmark + datasets grandes:** `tspredbench*`, EUNITE/NN3/NN5/M4/… (`.RData`).
- 🔭 **Gráficos:** `ts_plot_mv` e afins (matplotlib).

---

## 6. Como trabalhamos
- Incremental, um passo por vez; você confere o `examples/` processado antes de escalar.
- Verde sempre (`ruff` + `pytest` + build dos exemplos).
- Decisões sempre registradas aqui.
- **Branch por milestone** (como no pydaltoolbox); **exemplos sempre commitados** e
  empurrados; eu te passo os **links do GitHub** de cada exemplo para você conferir.
  O PR você abre pelo link (sem `gh` aqui). Sem merge/commit na `main` sem você pedir.
- **Comparação R vs Python a cada passo (obrigatório):** para cada componente
  portado, rodar o equivalente no `tspredit` (R 4.5.0 disponível) e no `pytspredit`
  sobre o mesmo dado (`tsd`) e **conferir que os resultados batem** (dim/colunas/
  valores em dados; métricas MSE/sMAPE/R2 em modelos). Divergência esperada só onde
  o backend difere (ex.: RNG do MLP/RF) — nesses casos, documentar a tolerância. Isso
  dá segurança a cada etapa. (Já feito no núcleo: `ts_data` idêntico ao R.)

## 7. Pendências / decisões em aberto
- [x] Backend do ELM (M3): implementação própria (random hidden uniform_positive + pinv,
      sem bias). Com `actfun="purelin"` e `nhid >= nº de lags` equivale a OLS → bate com
      o R independente do RNG. Com `nhid` menor, projeção aleatória (diverge por RNG).
- [x] `TsArima` sobre statsmodels: ordem manual `(p,d,q)` fiel; `auto` via busca AICc
      (aproxima `forecast::auto.arima`, pode não reproduzir a ordem exata). Backend MLE
      difere do `forecast` → previsões próximas, não bit-idênticas.
- Nota: `TsArimax` é multivariado (xreg) → movido para o Bloco B junto da família MV.
- [x] `ts_norm_an` portado e com paridade R vs Py (os 4 operadores). Nota: o default
      `outliers_boxplot` é aplicado sobre os dados já transformados (como no R) — crítico
      no `divide` (recorta explosões de divisão por média ~0).
- **Tolerância documentada (seasonality):** as 3 estratégias sazonais com `seasonality`
  explícito são idênticas ao R. Com `seasonality=None`, o R usa `forecast::findfrequency`
  (busca espectral AR); no Python uso uma estimativa por periodograma — o período pode
  divergir. Recomendação: passar `seasonality` quando quiser paridade exata.
- **Correção intencional (`TsNormNone`):** no caminho `x`, retorno `x` inalterado (o R
  retornava a matriz de janelas, o que quebraria o pipeline sliding-window).
