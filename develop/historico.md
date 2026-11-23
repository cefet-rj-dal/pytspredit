# Histórico — pytspredit (port Python do tspredit)

Relato cronológico do trabalho concluído. Os próximos passos estão em
`planejamento.md`; a correspondência de classes em `mapeamento.md`; a metodologia de
paridade em `parity.md`. Ao concluir uma atividade, registramos aqui.

Projeto: port Python do pacote R **tspredit** (CEFET/RJ, lab DAL) — previsão de séries
temporais com pré-processamento sintonizável —, preservando a **Experiment Line** (API
uniforme `fit`/`action`/`predict`/`transform`/`evaluate`/`set_params`). Depende apenas do
`pydaltoolbox` (já no PyPI). É o passo **antes** do `pyharbinger` (que depende dos dois).
Princípios: embrulhar scikit-learn/numpy/pandas/statsmodels/scipy com adapters finos e
**fiéis ao default do R** (quality-faithful); back-implementar nativamente só o que é
nicho/atípico; docstrings com Teoria + `References:` + `Porting:`; bug em qualquer lado
corrige-se nos dois repositórios; comparar R vs Python a cada passo.

---

## Bloco A / v1 — pipeline de previsão univariado

### 0.1.0 — fundação (publicada no PyPI, 2026-11-18)
Consolidada em milestones (M0→M4), cada uma em branch própria, com paridade R vs Python
verificada antes de pausar para revisão do Eduardo.
- **M0 — Scaffolding:** `pyproject` (deps `pydaltoolbox` + `statsmodels`), layout
  `src/pytspredit/`, pipeline de docs executáveis (`notebooks/**/*.md` → `examples/**/*.md`),
  `develop/` (PROMPT + planejamento + deploy), README.
- **M1 — Núcleo de dados:** `ts_data` (janela deslizante), `ts_projection`, `ts_sample`,
  `adjust_ts_data`, `ts_head`, `datasets.tsd()` (CSV extraído do `.RData`). **Paridade:
  dim/colunas/valores idênticos ao R.** (Descoberta: `tsd` tem 41 linhas, `y=sin(x)`,
  `x=seq(0,10,0.25)` — não 100 como a `data.R` sugere.)
- **M2 — Substrato de modelagem:** `TsReg` + `TsRegSW` (com `steps_ahead`); normalizações
  (`TsNormNone/Gminmax/Swminmax/Diff/An`); `TsLagmap` (**12 estratégias**). **Paridade:**
  7 configs de norm ≤1e-13; **as 12 estratégias de lag idênticas** (MI/mRMR com
  `method='median_unbiased'` = quantile type 8 do R; acf/pacf via statsmodels
  `adjusted=False`/`method='ywm'`); evaluate (MSE/sMAPE/R2) idêntico. Duas tolerâncias
  documentadas: lag sazonal com `seasonality=None` (periodograma vs `forecast::findfrequency`
  — passar `seasonality` explícito p/ paridade exata); `TsNormNone` caminho-x devolve `x`
  inalterado (o R devolvia a matriz de janelas, bug latente no pipeline sliding-window).
- **M3 — Regressores:** `TsKnn/TsElm/TsRf/TsSvm/TsMlp` (wrappers sklearn; ELM próprio) +
  `TsArima` (statsmodels; estende `TsReg`, opera na série `sw=1`). **Paridade end-to-end
  (tsd, 5 passos):** knn `max|Δ|=4.7e-9` e elm `4.8e-9` (exatos — determinísticos; ELM
  purelin+nhid≥nlags == OLS), svm `2.5e-4`, arima `6.0e-4` (quase-exato), rf `3.6e-2` e
  mlp `2.2e-2` (tolerância de RNG do backend). SVM = SVR + StandardScaler em X +
  TransformedTargetRegressor (espelha `e1071` scale=TRUE), gamma='scale'.
- **M4 — Tuning + fecho:** `TsTune` + `TsIntegtune` (sobre `DalTune`; grid CV minimizando
  MSE) + `TsAugNone` (identidade). Build + `twine check` OK; **install em venv limpo OK**
  (pegou um bug real: `row.to_numpy()` read-only no numpy novo → cópia explícita no predict
  multi-step do `TsRegSW`). 60 testes, 28 símbolos públicos.
- **Descoberta R:** `ts_norm_none` do R tem bug real (transform caminho-x devolve a matriz
  de janelas; `ts_mlp`+`ts_norm_none` também quebra). Patch de uma linha proposto e depois
  **merjado no `tspredit` main (2026-11-19, b2b1467)** — paridade futura pode rodar contra
  o main direto.

---

## Ciclo 1 — baixa dependência, um tema por release

- **0.2.0 — Zoo de modelos** *(0 dep nova)* **[PUBLICADO 2026-11-19]**: `TsDeterministic`
  (+`TsPeriodic`/`TsPersist`, nativos) · `TsDarima` (modelo **linear sobre janelas**, não
  ARIMA distribuído) · `TsWarma` (lm + normalização local por passo + `ndiffs` via KPSS do
  statsmodels). Ambos usam um `_RLinear` privado que **reproduz o `lm` do R** em designs
  rank-deficientes (descarta colunas aliased, **não** min-norm lstsq do sklearn).
  **Paridade (vs o branch R corrigido): deterministic exato; darima/warma ≤4.8e-9**
  (inclusive no caso degenerado seno-puro). Lição reusável: `lm` do R vs
  `LinearRegression` do sklearn divergem em designs colineares — portar modelos lineares
  de janela com a abordagem de descarte de colunas do `_RLinear`.
- **0.3.0 — Augmentation** *(0 dep, nativo)* **[PUBLICADO 2026-11-20]**: `TsAugFlip/
  Stretch/Shrink/Wormhole` (determinísticos, paridade exata ≤5e-15; wormhole 0.0) +
  `TsAugJitter/Awareness/Awaresmooth` (RNG: rnorm/rexp/sample → valores diferem do R, mas
  algoritmo + escala de ruído fiéis; jitter sd=0.4395701 bate). Ativa o slot `augment` do
  `TsIntegtune`.
- **0.4.0 — Filtros (baixa dep)** *(+`scipy` core)* **[PUBLICADO 2026-11-20]**: 16 filtros.
  **Paridade:** 9 EXATOS ~5e-9 (`none/ma/ema/winsor/recursive/fft/smooth` + `hp` [solve
  manual] + `gabor` [kernel+conv manual]); `lowess` (statsmodels, `delta=0.01*(n-1)`,
  it=3) e `kalman` (KF local-level, init difusa P0=1e7) também ~5e-9; `ses` quase-exato
  (8e-5). Tolerância de backend documentada: `qes` (HoltWinters-trend ~0.13), `seas_adj`
  (STL R vs statsmodels ~1.2), `spline` (`smooth.spline` vs `make_smoothing_spline` ~0.14).
  (Nota: committado direto na main, sem branch de milestone — ff/limpo mesmo assim.)
- **0.5.0 — Multivariado + plot** *(+`matplotlib` core)* **[PUBLICADO 2026-11-21]**:
  `multivariate.py` = `TsDataMv` (subclasse de DataFrame com metadados) + `ts_data_mv`/
  `adjust_ts_data_mv` + `ts_mv_spec`/`TsMvSpec` + `TsRegswMv` (orquestrador sliding-window)
  + `TsRegMv` + `TsLmMv` (reusa `_RLinear`) + `TsVar` (OLS nativo + AICc) + `TsArimax`
  (statsmodels ARIMA+exog) + `TsMvPrediction`. `plot.py` = `plot_ts_pred` (port matplotlib
  do `grf_ts_pred` do daltoolbox, ausente no pydaltoolbox) + `plot_ts_pred_mv`.
  **Paridade (orquestração exata):** `TsVar` ~1e-6 · `TsLmMv` (futuro conhecido) 4.97e-9 ·
  `TsRegswMv` (KNN + aux periódico) idêntico a 8 casas · `TsArimax` tolerância de backend
  (~1.4e-1) · ELM/RF/MLP MV = tolerância RNG/backend. Lições reusáveis: (1) ELM-purelin ==
  OLS exato só quando `nhid == n_features`; com `nhid>features` a extrapolação multi-step
  depende da projeção aleatória; (2) `stats::filter(y, rep(1/3,3), sides=1)` dá NA nos 2
  primeiros → no Python `rolling(3)` (min_periods=3) + preencher NA com `y`.
- **Varredura de doc (0.5.0):** toda classe pública passou a trazer `References:` (citações
  copiadas do `@references` do R) + `Porting:` (arquivo R + nível exact/approximate/adapted
  + adaptações). Cobertos todos os módulos. Fica o padrão para as próximas releases.
- **0.6.0 — Benchmark (parte leve)** *(0 dep nova)* **[IMPLEMENTADO, branch
  `milestone-0.6.0-benchmark`, aguardando revisão]**: fecha o Ciclo 1. `datasets.py`
  reescrito com loader genérico por `kind` + 20 acessores (+`tsd`) das **miniaturas
  Eidolon** (CSV/JSON + `manifest.json{url,kind,schema}`) + `catalog`/`list_datasets`/
  `dataset_url`. `benchmark.py` = `TsPredBench` (harness datasets×modelos → DataFrame
  tidy `[dataset, model, mse, smape, r2]`; factories ou instâncias; célula que falha →
  `nan`+`error`). **Arquitetura Eidolon** (Calmon et al., 2026, BreSci): miniatura
  compacta + url do completo, preservando o contrato (identidade/schema/url); a
  serialização Python é a instanciação do contrato, não do `.RData`. **Gerador canônico
  no `tspredbench`:** `etl/02-build-minidata.R` estendido para emitir R `.RData` (`data/`,
  inalterado) **e** Python (`data/python/`) do mesmo objeto encolhido → R↔Python idênticos
  por construção; sincronizado para `src/pytspredit/data/` (+`tsd` local). **Paridade
  committada (primeira sob o ritual):** `develop/parity/datasets.R` gera
  `datasets_ref.json`; `tests/test_quality_datasets.py` compara as 20 miniaturas ao R
  elemento-a-elemento (≤1e-9), R-free no teste — **todas batem** (numérico + labels
  `split`/`date`). Achados reusáveis: (1) `rdata` descarta `attr(url)` + puxa `xarray`
  → CSV/JSON+manifest é mais leve/fiel; (2) colunas all-null viram `object` no pandas (vs
  `NA_real_` no R) e rownames viram `_row` → loader corrige (`_coerce_numeric`+`_row`→índice).
  **Decisão:** `TsPredBench` é interface nova (o R `tspredbench.R` é só doc de datasets, sem
  harness) → `Porting: n/a`. 185 testes, ruff limpo, 13 notebooks (novo `benchmark/01`).

> **Estado ao fim do Ciclo 1 (parcial):** 0.1.0→0.5.0 publicados; `main` é a única branch
> (todas as milestone branches, já merjadas, foram deletadas local+remoto a pedido do
> Eduardo). 140 testes verdes, 68 símbolos públicos, 12 notebooks de exemplo. Falta o
> **0.6.0** (benchmark, parte leve) para fechar o Ciclo 1, depois o Ciclo 2 (deps pesadas).

---

## Maturidade — adoção das práticas do pydaltoolbox (2026-11-23)

Decidido com o Eduardo trazer as quatro passadas de maturidade do pydaltoolbox
(0.21→0.24) para o pytspredit. **Este passo (scaffold):** criados `develop/historico.md`
(este arquivo), `develop/mapeamento.md` (tabela R↔Python) e `develop/parity.md`
(metodologia de paridade); `planejamento.md` atualizado com a §1c e as passadas de
maturidade no roadmap. A partir do **0.6.0** a paridade passa a ser **committada**
(`develop/parity/` + `tests/test_quality_*`). Exemplos 1:1 (um por classe portada) e
auditoria de crédito (Python + R) ficam como passadas finais perto da 1.0.0. Ver
`planejamento.md` §1c e §5.4/Maturidade.

---

## Estado atual (base para o `planejamento.md`)
- Última release publicada: **pytspredit 0.5.0** (PyPI). **0.6.0 implementado** na branch
  `milestone-0.6.0-benchmark` (aguardando revisão/`twine` do Eduardo).
- Cobertura: todo o pipeline univariado + zoo de modelos + augmentation + filtros (baixa
  dep) + multivariado + plot + **benchmark (miniaturas Eidolon + TsPredBench)** → **fecha
  o Ciclo 1**. **Falta:** Ciclo 2 (emd/wavelet/loadfulldata) com deps pesadas; e as
  passadas de maturidade (exemplos 1:1, auditoria de crédito) perto da 1.0.0.
- Deps: núcleo `numpy`/`pandas`/`scikit-learn`/`statsmodels`/`scipy`/`matplotlib` +
  `pydaltoolbox`. Extras planejados: `[emd]`, `[wavelet]`, `[bench]`.
- Próximo: **Ciclo 2** (0.7.0 EMD → 0.8.0 wavelet → 0.9.0 `loadfulldata`), depois as
  passadas de maturidade e a **1.0.0**. Eduardo refina o Ciclo 2.
