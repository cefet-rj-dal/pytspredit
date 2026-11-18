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

#### 🔜 Fase 2 — Base de regressão + 1º regressor
- [ ] `TsReg` (sobre `Predictor`; `evaluate` MSE/sMAPE/R2; baseline predict = t0)
- [ ] `TsRegSW` (preprocess + input_map + backend; `predict` com `steps_ahead`)
- [ ] `TsLagmap` (começar por `"recent"`; demais estratégias na Fase 4b)
- [ ] 1º regressor ponta-a-ponta (`TsKnn` ou `TsLm`)
- Exemplos: `prediction/`.

#### ⬜ Fase 3 — Normalizações
`TsNormNone`, `TsNormGminmax`, `TsNormSwminmax`, `TsNormDiff`, `TsNormAn`
(todas com `inverse_transform`). Exemplos: `normalization/`.

#### ⬜ Fase 4 — Regressores ML completos + lag maps
- [ ] **4a** `TsKnn`, `TsMlp`, `TsRf`, `TsSvm`, `TsElm` (fiéis aos defaults do R)
- [ ] **4b** `TsLagmap` completo: as 12 estratégias (`tslagutils`)
- Exemplos: `prediction/` (um por regressor), `lagmapping/`.

#### ⬜ Fase 5 — Modelos estatísticos (statsmodels)
`TsArima`, `TsArimax` (univariados). Exemplos: `prediction/`.

#### ⬜ Fase 6 — Filtros simples
`TsFilMa`, `TsFilEma`, `TsFilSes`, `TsFilQes`, `TsFilKalman`, `TsFilLowess`,
`TsFilSpline`, `TsFilWinsor`, `TsFilHp`, `TsFilSeasAdj`, `TsFilRecursive`,
`TsFilSmooth`, `TsFilNone`. Exemplos: `filter/`.

#### ⬜ Fase 7 — Augmentation
`TsAugNone`, `TsAugJitter`, `TsAugFlip`, `TsAugStretch`, `TsAugShrink`,
`TsAugWormhole`, `TsAugAwareness`, `TsAugAwaresmooth`. Exemplos: `augment/`.

#### ⬜ Fase 8 — Tuning
`TsTune`, `TsIntegtune` sobre `DalTune`. Exemplos: `tutorial/`.

#### ⬜ Fase 9 — Fechamento da v1 → 🚀 Publicação (`0.1.0`)
Revisão de API pública, docstrings, `__all__`; suíte consolidada; README final;
build + `twine check` + install em venv limpo. **Eduardo faz/autoriza o upload.**

### BLOCO B — pós-v1 (lib extra/nicho ou complexidade)
- 🔭 **Multivariado:** `ts_data_mv`, `ts_reg_mv`, `ts_regsw_mv`, `ts_lm_mv`, `ts_var`, `ts_plot_mv`.
- 🔭 **Filtros avançados:** `ts_fil_emd`/`remd` (PyEMD), `wavelet` (PyWavelets),
  `gabor`, `fft`, state-space avançado (KFAS).
- 🔭 **ARIMA avançado:** `ts_darima` (distributed ARIMA), `ts_warma`, `ts_deterministic`.
- 🔭 **Benchmark + datasets grandes:** `tspredbench*`, EUNITE/NN3/NN5/M4/… (`.RData`).
- 🔭 **Gráficos:** `ts_plot_mv` e afins (matplotlib).

---

## 6. Como trabalhamos
- Incremental, um passo por vez; você confere o `examples/` processado antes de escalar.
- Verde sempre (`ruff` + `pytest` + build dos exemplos).
- Decisões sempre registradas aqui.
- Trabalho em **branch**; push para você revisar no GitHub. Sem merge/commit na `main`
  sem você pedir.

## 7. Pendências / decisões em aberto
- [ ] Backend do ELM: implementação própria leve (random hidden + solução fechada) — confirmar na Fase 4a.
- [ ] `ts_norm_an` (adaptive normalization): confirmar fórmula exata na Fase 3.
