# Kickoff prompt — portar `tspredit` (R) para Python (`pytspredit`)

> Cole/aponte este arquivo no início de uma nova sessão. Ele te dá tudo para retomar
> o port no mesmo padrão com que já fizemos o **pydaltoolbox** (publicado no PyPI).
> Primeira coisa a fazer na sessão: ler a memória do projeto e as pastas de referência.

---

## 0. Missão

Portar o pacote R **`tspredit`** (previsão de séries temporais) para Python como
**`pytspredit`**, reaproveitando o **`pydaltoolbox`** (já portado e publicado) como
fundação. `tspredit` depende **apenas** do `daltoolbox` — então este é o próximo passo
natural depois do pydaltoolbox, e **antes** do pyharbinger (que depende dos dois).

Princípio central (o mesmo do pydaltoolbox): **não reimplementar algoritmos** — envolver
`scikit-learn` / `numpy` / `pandas` / `statsmodels` com a **interface uniforme e
extensível da Linha de Experimento** (`fit` / `action` / `predict` / `transform` /
`evaluate` / `set_params`). Nossa contribuição é a interface, não os algoritmos.

## 1. Leia primeiro (pastas de referência)

| Pasta | O que é |
|---|---|
| `C:\Git\dal\pydaltoolbox` | **O PADRÃO A COPIAR.** Estrutura, convenções, testes, exemplos, empacotamento, deploy. Imite tudo. |
| `C:\Git\dal\daltoolbox` | Pacote R fundacional (origem do pydaltoolbox). |
| `C:\Git\dal\tspredit` | **O código R a portar** (fonte desta missão). |
| Memória | `C:\Users\eduar\.claude\projects\c--Git-dal-pydaltoolbox\memory\pydaltoolbox-port.md` — leia para o contexto completo das decisões. |

A pasta atual (`C:\Git\dal\pytspredit`) é um scaffold vazio (só `.git`, `.gitignore`,
`LICENSE`, `README.md`), exatamente como o pydaltoolbox começou.

## 2. Convenções já estabelecidas (replique do pydaltoolbox)

- **API idiomática:** métodos de instância; `fit` retorna `self` (encadeável).
- **Nomes:** PascalCase (PEP 8) — ex.: `TsData`, `TsReg`, `TsMlp`, `TsNormGminmax`.
- **Dados:** `pandas`/`numpy`. Séries e janelas deslizantes como DataFrame/ndarray.
- **Adapters finos** sobre sklearn/statsmodels, **fiéis aos defaults do backend R**
  (replicar comportamento, não só chamar a lib — e **comparar resultado R vs Python**).
- **Layout:** `src/pytspredit/`, `pyproject.toml` com `hatchling`, deps mínimas.
- **3 camadas de garantia:** (1) `pytest` unit por componente; (2) `pytest` integração
  (tutoriais ponta-a-ponta); (3) **documentação executável** — notebooks jupytext em
  `notebooks/**/*.md` renderizados para `examples/**/*.md` por `tools/build_examples.py`
  (copie o script do pydaltoolbox; tabelas saem em texto puro, sem HTML). **Um exemplo
  didático por classe** (objetivo + referência do artigo + fit/predict/evaluate).
- **Pasta `develop/`** para material de dev (este PROMPT.md, planejamento.md, deploy.md),
  **excluída do sdist** via `[tool.hatch.build.targets.sdist]` (veja o pyproject do
  pydaltoolbox: `include = src+README+LICENSE`, `exclude = develop/notebooks/examples/tests/tools`).
- **Qualidade sempre verde:** `ruff check .` + `pytest` + build dos exemplos a cada passo.
- **Git:** trabalhe em branch, faça push para o Eduardo revisar no GitHub, **pause a cada
  fase/sub-bloco** para ele conferir, depois ff-merge na `main`. `gh` NÃO está instalado.
  Commits terminam com `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Deploy:** PyPI via `twine` (o `.pypirc` do Eduardo já tem o token `daldeploy` de
  conta inteira — serve para todos os pacotes DAL). Guia completo em
  `C:\Git\dal\pydaltoolbox\develop\deploy.md`. Versões são **imutáveis**; é **beta** até
  a `1.0.0`. **O Eduardo autoriza/roda o deploy** (ou autoriza você a rodar).

## 3. Dependência — reutilize o pydaltoolbox

`pip install pydaltoolbox` (já está no PyPI). O `tspredit` R consome do `daltoolbox`
(confirmado por análise do fonte): **`regression`/`predictor` (muito), `dal_transform`
(muito — base das normalizações), `dal_tune`, `sample_random`/`train_test`/`k_fold`,
`set_params`, `fit/transform/predict/inverse_transform`, `minmax`, `outliers_boxplot`**.

No Python, importe essas peças do `pydaltoolbox`:
```python
from pydaltoolbox import (Regression, SklearnRegression, Predictor, DalTransform,
                          DalTune, SampleRandom, train_test_from_folds,
                          MinMax, ZScore, OutliersBoxplot)
```
Nota: os genéricos `do_fit`/`do_predict` são **do próprio tspredit** (definidos em
`ts_reg.R`), não do daltoolbox.

## 4. Arquitetura do `tspredit` (60 arquivos em `R/`) — o que portar

- **Estruturas de dados:**
  - `ts_data(y, sw)` — janela deslizante: matriz com colunas de lag `t{sw-1}..t0`
    (`ts_data.R`). `ts_projection` separa entrada (`t{sw-1}..t1`) e saída (`t0`).
    `ts_data_mv` (multivariada), `ts_lagmap`, `tslagutils`.
  - Em Python: um `TsData` (ndarray/DataFrame com colunas `t{k}`) + `ts_projection` →
    `{input, output}`. (Veja `ts_data.R` para a lógica de janelamento exata.)
- **Bases de regressão de TS** (`ts_reg.R`, `ts_regsw.R`, `ts_reg_mv.R`, `ts_regsw_mv.R`):
  `ts_reg` estende `predictor` do daltoolbox; usa os genéricos próprios `do_fit(obj,x,y)` /
  `do_predict(obj,x)`; `fit` recebe `(x, y)` das janelas projetadas; `evaluate` = MSE/sMAPE/R2
  (igual ao `reg_regression` do daltoolbox). Um regressor de TS combina **normalização +
  backend**: ex. `ts_mlp(ts_norm_gminmax(), input_size, ...)`.
- **Regressores** (adapters finos → sklearn/statsmodels):
  `ts_knn`→`KNeighborsRegressor`, `ts_mlp`→`MLPRegressor`, `ts_rf`→`RandomForestRegressor`,
  `ts_svm`→`SVR`, `ts_lm_mv`→`LinearRegression`; `ts_elm` (ELM — simples, implementável),
  `ts_arima`/`ts_arimax`/`ts_darima`→`statsmodels` (ARIMA/SARIMAX), `ts_var`→`statsmodels VAR`,
  `ts_warma`.
- **Normalizações** (`DalTransform`, com `inverse_transform`): `ts_norm_gminmax` (global
  min-max), `ts_norm_swminmax` (por janela), `ts_norm_an` (adaptive), `ts_norm_diff`
  (diferenciação), `ts_norm_none`.
- **Filtros** (`ts_fil_*`, ~19): ma, ema, ses, qes, kalman, lowess, spline, winsor, hp,
  seas_adj, recursive, smooth... (deixe EMD/`remd`/`gabor`/`wavelet`/`fft` que precisam
  de PyEMD/PyWavelets/scipy para depois, se complicar).
- **Augmentation** (`ts_aug_*`, ~8): jitter, flip, stretch, shrink, wormhole, awareness...
- **Tuning:** `ts_tune`, `ts_integtune` — estende `dal_tune` (reutilize `DalTune` do
  pydaltoolbox; CV por janelas, minimiza erro).
- **Datasets:** `data.R` traz `tsd` (série exemplo) — exponha em `datasets`.
- **Utils/bench/plot:** `tsanutils`, `tslagutils`, `tspredbench*` (benchmark), `ts_plot_mv`
  (plot — pode ficar para depois, como no pydaltoolbox).

## 5. Critério de escopo da v1 (mesmo princípio do pydaltoolbox)

Entra na v1 tudo que é **wrapper fino** de `sklearn`/`numpy`/`pandas`/`statsmodels`.
Adie para "Bloco B" o que exige lib extra/nicho ou complexidade: filtros baseados em
EMD/wavelets/gabor (PyEMD, PyWavelets), ELM com backend pesado, `KFAS`/state-space
avançado, e gráficos. Mantenha as **deps da v1 enxutas** (numpy/pandas/scikit-learn +
pydaltoolbox; statsmodels se ARIMA entrar na v1 — avalie como extra se preferir).

## 6. Roadmap sugerido (em fases/sub-blocos, pause a cada um)

0. Scaffolding: `pyproject` (dep `pydaltoolbox`), `src/`, `develop/` (mova este PROMPT +
   planejamento.md + deploy.md para lá), `tools/build_examples.py`, README com `pip install`.
1. **Núcleo de dados:** `TsData` (janela deslizante) + `ts_projection` + datasets (`tsd`).
2. **Base de regressão:** `TsReg` (contrato `fit(x,y)`/`predict(x)`/`evaluate`) + um
   regressor simples (ex.: `TsKnn` ou `TsLm`) rodando um experimento ponta-a-ponta.
3. **Normalizações:** `TsNormGminmax`, `TsNormSwminmax`, `TsNormDiff`, `TsNormNone`
   (com `inverse_transform`).
4. **Regressores ML completos:** knn, mlp, rf, svm, elm.
5. **Modelos estatísticos:** arima/arimax/var (statsmodels).
6. **Filtros simples** (ma, ema, ses, kalman, lowess, spline, winsor...).
7. **Augmentation.**
8. **Tuning** (`TsTune`/`TsIntegtune` sobre `DalTune`).
9. Fechamento + publicar `pytspredit` 0.1.0 no PyPI.
Bloco B: filtros EMD/wavelet/gabor/fft, state-space, benchmark, gráficos.

*(Confirme o roadmap com o Eduardo e registre num `develop/planejamento.md` próprio,
como fizemos no pydaltoolbox — documento vivo, marcando fases feitas.)*

## 7. Ritual a cada fase
Implementar + testes → criar notebook do tema → `python tools/build_examples.py` →
`ruff` + `pytest` verdes → commit no branch + push → **Eduardo revisa no GitHub** →
ff-merge na main. Sempre comparar qualidade R vs Python nos pontos sensíveis.

## 8. Primeiro passo concreto
Leia `pydaltoolbox/pyproject.toml`, `tools/build_examples.py`, `src/pydaltoolbox/base.py`
e `regression/`, depois `tspredit/R/ts_data.R`, `ts_reg.R`, `ts_norm_gminmax.R`,
`ts_mlp.R`. Proponha o planejamento ao Eduardo e comece pela Fase 0+1.
