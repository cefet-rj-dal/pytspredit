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

## Ciclo 2 — dependências pesadas (extras) + back-implementation/++

- **0.7.0 — Filtros de decomposição (EMD), NATIVO** *(0 dep nova)* **[IMPLEMENTADO, branch
  `milestone-0.7.0-emd`]**: inicia o Ciclo 2, mas com um **pivô** (Eduardo, 2026-11-23). O
  plano era embrulhar PyEMD (extra `[emd]`); após análise, **back-implementamos EMD/EEMD
  nativo** (`emd.py`, numpy+scipy já no núcleo → **0 dep**, cai o extra). `TsFilEmd`/`TsFilRemd`
  em `filters.py` sobre o engine nativo. **Análise que motivou o pivô:** benchmark R `hht`
  vs PyEMD vs nativo — o `hht` é **~60× mais lento** que PyEMD e **~1800×** que um EMD nativo
  (9,3s vs 5ms em n=2048); o nativo tem **reconstrução perfeita** e **denoising ≥ PyEMD**
  (R² 0.971 vs 0.963), com scipy já no núcleo. **++ honesto:** redução de dep + velocidade +
  canonicalização (Python dono; R backporta e **dropa `hht`**), **não** um algoritmo novo —
  **CEEMDAN (candidato da literatura) foi testado e REJEITADO pelo gate §1b** (não melhora o
  denoising dos filtros, só custa 2.7×; protótipo guardado). **Paridade committada:**
  `develop/parity/filters_emd.R` (referência `hht`) + `tests/test_quality_emd.py` com
  invariante de **qualidade** (melhora de R² + remove outlier tão bem ou melhor que `hht`),
  não valor-a-valor (EMD estocástico/backend). PyEMD/`hht` **só dev** (validação). Achado:
  o `tsd` (seno puro) é degenerado p/ EMD (IMF1 = o próprio sinal) → correlação não serve;
  o invariante é o resultado do denoising. 194 testes, ruff limpo, notebook
  `filter/02-emd-filters`.

- **0.8.0 — Wavelet (MODWT), NATIVO** *(0 dep nova)* **[IMPLEMENTADO, branch
  `milestone-0.8.0-wavelet`]**: segundo back-implementation do Ciclo 2, mesmo pivô do EMD
  (Eduardo, 2026-11-23). Em vez de embrulhar PyWavelets (extra `[wavelet]`), **MODWT nativo**
  em `wavelet.py` (**numpy puro — nem scipy → 0 dep**). `TsFilWavelet` (zera D1, `fit`
  seleciona melhor filtro por R²) sobre ele; haar/d4/la8/bl14/c6 (coef. do R `wavelets`
  hardcoded). **Diferença-chave vs EMD:** MODWT é transformada **linear determinística** →
  paridade **valor-a-valor ~5e-6** com o R (não quality-level); mais fiel que `pywt.swt`
  (convenção diferente) — e o PyWavelets nem estava instalado. **Análise do ++:** thresholding
  de Donoho-Johnstone (método canônico de denoising) **avaliado e REJEITADO pelo gate** (soft/
  hard/por-nível → R² 0.23–0.25 vs 0.97 do zero-D1; MODWT redundante exige translation-invariant,
  pesquisa sem ganho garantido). ++ honesto = redução de dep + exatidão + canonicalização
  (R dropa `wavelets`). Achado: c6 reconstrói só a ~1e-6 (coef. coiflet publicados são
  aproximados, igual no R). **Paridade committada valor-a-valor:** `develop/parity/
  filters_wavelet.R` + `tests/test_quality_wavelet.py` (≤1e-5 vs R, R-free). 203 testes, ruff
  limpo, notebook `filter/03-wavelet-filter`.

- **0.9.0 — Datasets completos (`loadfulldata`), 0 DEP** **[IMPLEMENTADO, branch
  `milestone-0.9.0-loadfulldata`]**: terceiro e último back-implementation do Ciclo 2.
  **Pivô (Eduardo):** em vez do extra `[bench]=rdata` planejado, **CSV.gz/JSON.gz
  pandas-nativo, 0 dep**. Decidido por benchmark real de tamanho: CSV puro ~9× o `.RData`
  (grande demais), mas **CSV.gz ≈ `.RData`** (NN5: 157/1422/**352**/693KB p/ RData/CSV/CSV.gz/
  parquet) — **parquet perdeu em tamanho E exige `pyarrow`**; `gzip` é stdlib e
  `pd.read_csv(compression="gzip")` é pandas-núcleo → 0 dep. `datasets.loadfulldata(nome)`
  baixa a `full_url` do manifest (urllib stdlib) → pandas/gzip → mesma estrutura do loader da
  miniatura. **ETL do tspredbench estendido** (`02-build-minidata.R`, branch `full-data-python`):
  publica os completos em `data/python_full/*.csv.gz|.json.gz` + `full_url` no manifest.
  Miniaturas seguem CSV/JSON puro. Paridade: miniatura = `head(30)` do full (exato por
  construção); teste via `file://` (R-free, sem rede). 207 testes, ruff limpo. **Com isso o
  Ciclo 2 fechou SEM nenhuma dep nova** — hht, wavelets e rdata todos eliminados.

- **0.10.0 — ARIMA++ (auto-seleção nativa), 0 DEP** **[IMPLEMENTADO, branch
  `milestone-0.10.0-autoarima`]**: fecha o Ciclo 2 e é o **primeiro ++ ALGORÍTMICO** (os
  outros foram redução-de-dep). `TsArima(auto="cv")` ataca 2 fraquezas do `auto.arima`:
  (1) **candidatos dirigidos por ACF/PACF** (lags significativos entram; sazonal 12 testado)
  em vez do cap fixo 5; (2) **seleção por CV rolling-origin casada ao horizonte** (sMAPE
  out-of-sample) em vez do AICc in-sample. `auto="aicc"` continua default (compat). **Processo
  de validação (iterativo, gate §1b):** 1º teste (seno puro) reprovou — seno é determinístico,
  não-AR; 2º (AR(12) real) ganhou; 3º (NN5 real) **reprovou com CV de 1-passo** (0.1655 vs
  0.1565) → diagnóstico: **a CV tem que casar o horizonte da previsão**; com CV multi-step
  casada **passou: ~14% melhor (0.1352 vs 0.1565, 4/4), recupera o lag 7**. Lição reusável: em
  auto-seleção por CV, o horizonte da CV deve ser o horizonte de previsão. Benchmark committado
  `develop/parity/autoarima_benchmark.py`. Notebook `prediction/04-arima-plus`.

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

### Publicações 0.6.0→0.10.0 e backports R (2026-11-23)
- **0.6.0→0.10.0 PUBLICADOS no PyPI**; todas as milestone branches do Ciclo 2 ff-merjadas na
  main. O Ciclo 2 inteiro fechou **sem nenhuma dep nova**.
- **Backport R completo (tspredit)**, espelhando os nativos do Python e **removendo 3
  dependências**: EMD/EEMD nativo (base R + `stats::splinefun`) → **dropa `hht`**; MODWT
  nativo (base R) → **dropa `wavelets`**; ARIMA++ + seleção de ordem nativa (`stats::arima`,
  KPSS/ACF/PACF próprios) + `findfrequency` nativo → **dropa `forecast`** (eram 4 usos:
  ts_arima, ts_arimax, ts_warma, tslagutils). DESCRIPTION → **v2.0.737** (numeração Boeing).
  Validado `R CMD INSTALL` limpo sem as 3 libs. Gotcha corrigido: `stats::arima(xreg=NULL)`
  guardava símbolo `xreg` no call → omitir `xreg` quando NULL.

### M1 — organização de docs (espelha pydaltoolbox 0.20) · M2 — exemplos 1:1 (2026-11-23)
- **M1:** regenerar `examples/` sem `hht`/`wavelets`/`forecast`; R já usa notebooks/examples.
- **M2 COMPLETO** (branch `milestone-m2-examples`, dev-only, depois **já ff-merjada na main**):
  **~104 notebooks um-por-classe** em 11 famílias (tsdata/normalization/lagmapping/augment/
  filter/prediction/tuning/multivariate/datasets/custom/tutorial), todos verdes; famílias torch
  fora de escopo. Pacote 1 (núcleo) + Pacote 2 (datasets/custom/tutorial), espelhando o split
  0.21/0.22 do pydaltoolbox. Gotchas: conversor aceita UM argumento; `TsNormDiff` perde 1
  coluna (input_size < sw-1 ao diferenciar); `prediction/10-arima-plus` (auto="cv") é o lento.

### 0.11.0 — `fit_arima` exportado + sentinela (0,0,0) (2026-11-24) **[PUBLICADO no PyPI]**
A auto.arima nativa (sem `forecast`) virou função **pública/exportada** `fit_arima(series,
auto="aicc"|"cv", ...)` nos **dois lados** (Python `pytspredit.fit_arima`; R `fit_arima()` com
`@export`/NAMESPACE/Rd) — o substituto do `forecast::auto.arima`, reusável (ex.: harbinger).
Gatilho de auto-fit **aditivo**: `TsArima(0,0,0)`/`ts_arima(0,0,0)` dispara auto, junto com
`None`/`NULL` (default); só o (0,0,0) cheio é sentinela ((0,1,1) etc. seguem manuais; ARIMA(0,0,0)
puro fica inacessível, documentado). Nome `fit_arima` e não `fit.arima` (o `.` é inválido em
Python e colidiria com o método S3 `fit` da classe `arima` em R). pytspredit **0.11.0 publicado**;
tspredit segue em v2.0.737. ff-merjado nas duas mains.

### M3 — testes pareados R↔Python retroativos (2026-11-24) **[na main, dev-only]**
Backfill do ritual de paridade committada para as famílias **0.1→0.5**: cada uma = um gerador
`develop/parity/<fam>.R` (gera `<fam>_ref.json` do input congelado `tsd.csv`, + `logistic.csv`
congelado p/ o zoo) + um guard R-free `tests/test_quality_<fam>.py`. **6 passadas:**
(1) **normalização** — matriz transformada ≤1e-9 (8 configs, todos operadores do `an`);
(2) **lagmap** — conjunto de lags exato (12 estratégias; comparar LAGS, não posições — R é
1-based, Python 0-based); (3) **regressores** — vetor 5-step: knn/elm(purelin nhid==n_lags=OLS)
≤1e-9, svm 2.5e-4 / arima 5.98e-4 backend, rf/mlp estocásticos via **correlação-com-actual ≥0.8**
(sMAPE inútil — o seno cruza zero); (4) **model zoo** — periodic/persist + darima/warma ≤1e-9
(inclui o caso degenerado rank-3 no tsd); (5) **filtros** — 11 exatos ≤1e-9, ses ~8e-5,
qes/seas_adj/spline backend (tol por-filtro); (6) **multivariado** — var/lm_mv/regsw_mv ≤1e-6,
arimax backend 1.37e-1. (datasets/emd/wavelet já tinham guard desde 0.6–0.8.) Consolidados os 3
testes de paridade hard-coded do `test_multivariate.py` no guard committado. Suíte verde
(~261 testes), ruff limpo. `parity.md` §4/§6 atualizado (tabela + item fechado).
- **🐞 Bug REAL do R achado e corrigido** (objetivo do M3 — "corrigir bugs nos dois lados"):
  o `ts_arimax` estava **quebrado** no predict — `object 'xreg' not found`. O `stats::arima(xreg=)`
  guardava o *símbolo* `xreg` no `$call` e o `predict.Arima` o avaliava fora de escopo (regressão
  da remoção do `forecast` em 2.0.737; o `forecast::Arima` mascarava). Fix em `arima_fit`:
  `fit$call$xreg <- xreg` (embute o valor da matriz). **Python/statsmodels já estava correto** —
  paridade é concordância depois dos dois certos. tspredit **v2.0.737→2.0.747**, ff-merjado na main.
  **O alvo do CRAN passou a ser 2.0.747.**

### M4 — metadados + classificador Beta (2026-11-24) **[branch `milestone-m4-metadata`, dev-only]**
`pyproject.toml`: `maintainers`/`authors` preenchidos com os **contribuidores do port Python**
(papéis na convenção do R: cre=maintainer, aut, ths, ctb; cph=CEFET/RJ) e **Development Status
`3 - Alpha` → `4 - Beta`**. **Não é espelho cego do `Authors@R` do R** — o Eduardo corrigiu: o
port Python tem um conjunto distinto e menor (Eduardo + Diego Carvalho, Eduardo Bezerra, Esther
Pacitti, Fabio Porto [ctb]; Fernando Alexandrino, Rebecca Salles [aut; email INRIA]; CEFET/RJ
[cph]). Excluídos vs R: Arthur Garcia, Cristiane Gea, Diogo Santos, Luciano Mello, Vitória
Birindiba. Dev-only: versão segue 0.11.0; o classificador Beta + autores vão ao PyPI só no
`1.0.0b1` (FECHO). `twine check` ok.

---

## Estado atual (base para o `planejamento.md`)
- Última release publicada: **pytspredit 0.11.0** (PyPI, 2026-11-24). **Ciclo 2 completo** e
  **todo o `tspredit` portado**; o ciclo inteiro **não adicionou nenhuma dep**.
- Cobertura: todo o Ciclo 1 + EMD (0.7.0) + MODWT/wavelet (0.8.0) + `loadfulldata` (0.9.0) +
  ARIMA++ (0.10.0) + `fit_arima` exportado (0.11.0). Extras `[emd]`/`[wavelet]`/`[bench]`
  **eliminados** (back-implementation nativa). Deps: núcleo `numpy`/`pandas`/`scikit-learn`/
  `statsmodels`/`scipy`/`matplotlib` + `pydaltoolbox`.
- **tspredit R na main em v2.0.747** (EMD/wavelet/ARIMA++/`fit_arima` nativos; `hht`+`wavelets`+
  `forecast` droppados; fix do `ts_arimax`). **Pendente externo: Eduardo submete 2.0.747 ao CRAN.**
- **Maturidade (espelha pydaltoolbox 0.20→0.24, fecha com BETA `1.0.0b1`):** **M1 ✓** (docs),
  **M2 ✓** (exemplos 1:1, ~104 notebooks, na main), **M3 ✓** (testes pareados R↔Python 0.1→0.5,
  committados, na main — achou+corrigiu o bug do `ts_arimax`), **M4 ✓** (metadados: autores do
  port + Development Status `4-Beta`, branch `milestone-m4-metadata`, dev-only). **Falta: M5**
  (parity.md: estudos de caso + matriz R↔Py completa) → **M6** (auditoria de crédito Py+R:
  Teoria+References+Porting, atribuição das back-impls/++). **FECHO:** publicar `1.0.0b1` (beta ~1 semana) → promover a **1.0.0**; pós-1.0.0
  pytspredit também adota numeração Boeing.
- Branches já mescladas (podem ser deletadas): `milestone-m3-paired-tests` (pytspredit),
  `arimax-xreg-fix` (tspredit). `main` é a única branch de código nos dois repos.
- Próximo: **M4**. Depois M5, M6, e o beta `1.0.0b1`.
