# Parity Verification for the `tspredit` → `pytspredit` Port

> **Purpose.** This document records how we verify that `pytspredit` (the Python port)
> faithfully reproduces `tspredit` (the original R package), specialized to
> **time-series forecasting**. It **shares the general methodology** of the sibling
> document `C:\Git\dal\pydaltoolbox\develop\parity.md` (written as the backbone for a
> scientific article on verifying behavioral parity between two implementations of the
> same data-science/AI toolkit). Read that document for the full treatment of the central
> hazard, the taxonomy of disagreement, bug localization, and threats to validity; this
> one adapts the invariants and records the `tspredit`-specific case studies.

---

## 1. The shared protocol (condensed)

From the sibling `parity.md` (§8 there):

1. **Freeze** the input once; export to CSV; both sides read the same bytes/rows.
2. **Choose** the coarsest decision-level invariant per family.
3. **Compare** decisions, not representations.
4. **Classify** each disagreement: exact / benign-stochastic / benign-numeric-or-solver /
   weak-default / structural / real defect.
5. **Evidence** every "benign" verdict (an agreeing regime, a seed, or a formula).
6. **Localize** defects by bisecting to the smallest primitive and instrumenting it.
7. **Fix** defects on **both** sides (ask before editing the reference); **re-check the
   ripple** through every reuse of a shared primitive.
8. **Lock** the outcome with a guard test and keep the runnable parity script next to its
   frozen input.

**Central hazard (from §2 there):** never compare two runs on their own splits. Our whole
workflow uses the shared `tsd` series (and, for multivariate, the same constructed
covariates), so the input is already frozen; the committed ritual (below) makes that
explicit per release.

---

## 2. Invariants per time-series family

Time series exposes different natural invariants than tabular ML. Pick the coarsest output
that still reflects the method's decision:

| Family                         | Invariant compared                                             |
|--------------------------------|---------------------------------------------------------------|
| Forecast models (knn/mlp/rf/svm/elm/arima/darima/warma) | the **predicted vector** over an `n`-step horizon (`max\|Δ\|`) and the metrics MSE/sMAPE/R² on the shared split |
| Normalization (`TsNorm*`)      | the **full transformed window matrix** (tol ≤1e-9), plus `transform`∘`inverse_transform` round-trip |
| Lag selection (`TsLagmap`)     | the **set/positions of selected lags** (exact) |
| Filters (`TsFil*`)             | the **full filtered series** (tol; backend-dependent for some) |
| Augmentation (`TsAug*`)        | the **generated window matrix** (deterministic ones exact; RNG ones: algorithm + noise scale) |
| Multivariate (`TsVar`/`TsLmMv`/`TsRegswMv`/`TsArimax`) | the **predicted system** / per-variable forecast vectors |
| Tuning (`TsTune`/`TsIntegtune`)| the **selected hyper-parameters** (and the resulting score) |

**Deterministic transforms are the most stringent case** (normalization, deterministic
augmentation, most filters): demand the *entire output matrix equal* up to ≈1e-9 — there is
no split variance, RNG, or CV to hide behind (sibling §9).

---

## 3. Reference side: run against the *fixed* `tspredit`

The R `ts_norm_none` had a real bug (transform's `x`-path returned the window matrix
instead of the target), which silently corrupted every default-preprocess model
(darima/warma especially). It was **merged into `tspredit` main (2026-11-19, commit
`b2b1467`)**. Therefore:

- Parity references are generated against **`tspredit` main** (R 4.5.0 at `/c/R/R-4.5.0`;
  `pkgload::load_all` of `C:\Git\dal\tspredit` when a not-yet-released fix is needed).
- **Reference is not ground truth** (sibling §7): where the R behavior is itself wrong, fix
  it on the R side with Eduardo's OK — parity means *agreement after both are correct*.

---

## 4. The committed-parity ritual (from release 0.6.0 onward)

Until 0.5.0, parity was verified *ad-hoc* (numbers recorded in memory / `planejamento.md`).
From 0.6.0, each release lands parity as **reproducible artifacts**, mirroring pydaltoolbox:

- `develop/parity/<family>.py` (Python side) + `<family>_samesplit.R` (R side) + a small
  `<family>_compare.py`, reading one **frozen CSV** input.
- A guard test in `tests/test_quality_<family>.py`: **exact** hard-coded invariants for
  deterministic code; **loose** guards (thresholds, "not degenerate", "beats baseline") for
  stochastic/backend cases — tight enough to catch a collapse, loose enough to survive
  version drift.

The **retroactive pass (M3) is DONE**: `develop/parity/` + `tests/test_quality_*` now
cover 0.1→0.5 as committed, R-free guards, each with its runnable `*.R` generator and a
frozen CSV input:

| Family | Generator | Guard | Invariant / verdict |
|---|---|---|---|
| normalization | `normalization.R` | `test_quality_normalization.py` | transformed matrix ≤1e-9 (8 configs) |
| lagmap | `lagmap.R` | `test_quality_lagmap.py` | selected lag SET exact (12 strategies) |
| regressors | `regressors.R` | `test_quality_regressors.py` | 5-step vector: knn/elm ≤1e-9, svm/arima backend, rf/mlp corr≥0.8 |
| model zoo | `modelzoo.R` | `test_quality_modelzoo.py` | periodic/persist + darima/warma ≤1e-9 (incl. degenerate rank-3 tsd) |
| filters | `filters.R` | `test_quality_filters.py` | filtered series: 11 exact ≤1e-9, ses ~8e-5, qes/seas_adj/spline backend |
| multivariate | `multivariate.R` | `test_quality_multivariate.py` | y 5-step: var/lm_mv/regsw_mv ≤1e-6, arimax backend |
| datasets / EMD / wavelet | (0.6–0.8) | `test_quality_{datasets,emd,wavelet}.py` | already committed from their release |

**Bug found & fixed by M3 (both-sides discipline, §7):** generating the multivariate
reference crashed R `ts_arimax` with `object 'xreg' not found` — `stats::arima(xreg=)`
stored the `xreg` *symbol* in its call and `predict.Arima` evaluated it out of scope
(regression from the 2.0.737 `forecast` removal; `forecast::Arima` used to shield it).
Fixed in R `arima_fit` by embedding the regressor matrix value in the call (tspredit
2.0.747). Python/statsmodels was already correct — parity is agreement after both are right.

---

## 5. Case studies

The four benign disagreement classes (stochastic / numeric-solver / weak-default /
structural) all appeared in this port, but the instructive cases are the *defects* and the
*judgement calls*. Each is reproducible from `develop/parity/` and locked by a guard.

### 5.1 `ts_norm_none` x-path — a silent transform bug in the reference (fixed on both sides)
This is the `tspredit` analog of the sibling's `cla_glm` inversion: **parity testing found a
bug in the original, not the port.** R `transform.ts_norm_none` returned the *window matrix*
instead of the *target* on its `x`-path, silently corrupting every model whose default
preprocess is `ts_norm_none` — `ts_darima`/`ts_warma` worst.

It surfaced the way the central hazard (§2) predicts. A naive `darima` comparison on the
pure-sine `tsd` showed R `[0.87]` vs Python `[0.41]` — superficially a linear-fit
divergence. **Bisection to the smallest primitive (§ sibling-5)** ruled that out: against a
*fixed* R branch, `_RLinear` reproduces R's rank-deficient `lm` (column-dropping, not
sklearn's min-norm `lstsq`) to ~1e-9 **even on the degenerate rank-3 pure-sine design**
(both sides select `[0.4121, 0.1739, −0.0752, −0.3195, −0.544]`, both rank 3). The entire
`[0.87]`→`[0.41]` gap was the *corrupted target* upstream, not the fit. Fixed on the R side
(`transform.ts_norm_none` honors `x=`, plus an identity `inverse_transform`; merged to
`tspredit` main `b2b1467`). Python was already correct (it returns `x` unchanged on purpose).
Lessons: **reference is not ground truth**; **a transform bug is invisible downstream** (it
shifts every score without crashing); **re-check the ripple** (every default-preprocess
model). Parity references are therefore generated against the *fixed* R (§3).

### 5.2 `ts_arimax` `object 'xreg' not found` — a regression surfaced by writing the reference (M3)
Generating the multivariate reference (`multivariate.R`) crashed R `ts_arimax` on
*predict*: `object 'xreg' not found`. Root cause: `stats::arima(xreg=)` stores the `xreg`
**symbol** (not its value) in `fit$call`, and `predict.Arima` re-evaluates that call in a
scope where `xreg` no longer exists. It was a **regression introduced by the 2.0.737
`forecast` removal** — `forecast::Arima` used to shield the call; plain `stats::arima` does
not. Fix in R `arima_fit`: embed the matrix value with `fit$call$xreg <- xreg` (tspredit
`2.0.737`→`2.0.747`). Python/statsmodels was already correct. Lesson: **the act of writing a
parity reference exercises the reference** and flushes out its latent bugs — parity is
*agreement after both are right*, and the retroactive M3 backfill paid for itself by finding
this one. (Changed Eduardo's CRAN target to `2.0.747`.)

### 5.3 ARIMA++ — the `++` gate and the horizon-matched-CV finding (0.10.0)
`TsArima(auto="cv")` is **not a port but an algorithmic improvement**, so it is held to a
different bar: it must *beat the baseline on real data* (the §1b `++` gate) or it is not an
improvement. The validation was iterative and is the clearest "approximate is a claim that
must be evidenced" story in the project:

1. **Pure sine `tsd` → rejected the *test*, not the method.** A deterministic non-AR signal
   is a bad discriminator for an AR-order search.
2. **Synthetic AR(12) → won** (recovers the true lag), but synthetic data is not evidence.
3. **NN5 real data, 1-step CV → FAILED** (sMAPE 0.1655 vs AICc-auto 0.1565 — *worse* than
   the baseline it claims to beat).
4. **Diagnosis:** in CV-based model selection **the CV horizon must match the forecast
   horizon**. 1-step CV picks orders that generalize poorly multi-step.
5. **NN5, horizon-matched multi-step CV → passed:** 0.1352 vs 0.1565, **~14% better, 4/4**,
   and recovers the weekly lag 7 (vs auto.arima's cap-of-5 candidates).

Reusable law: **match the CV horizon to the forecast horizon.** Evidence lives in
`develop/parity/autoarima_benchmark.py` (run manually; ~15–20 s/series), not the unit suite.

### 5.4 EMD on `tsd` — choosing the invariant when the natural one degenerates (0.7.0)
`tsd` (a pure sine) is **degenerate for EMD**: IMF1 *is* the whole signal, so `y − IMF1` is a
near-constant residual and any correlation-with-clean-signal invariant is meaningless. This
is protocol rule 2 (compare *decisions*, not representations) under stress: drop to the
*coarsest decision-level* invariant — the **denoising outcome** (does it remove an injected
outlier and improve R²?), not value-wise agreement (EMD is adaptive/stochastic with a
different backend than R `hht`). The native engine reconstructs exactly (Σ IMF = signal,
4e-16), denoises R² 0.971 ≥ PyEMD 0.963, removes the outlier at least as well as `hht`, and
runs ~60× (EEMD) to ~1800× (EMD) faster. Guard: `tests/test_quality_emd.py` (R² improves;
outlier removed ≥ `hht`). Same threat as the sibling's "dataset difficulty."

### 5.5 Read-only numpy rows — the clean-venv install is a parity gate (M4)
A clean-venv install on numpy 2.5.5 caught `row.to_numpy()` becoming read-only, which broke
`TsRegSW` multi-step predict (fixed with an explicit `np.array(...)` copy). Lesson: **version
drift is a parity threat** — the from-PyPI clean install is itself a validation step, not
only a packaging check.

---

## 6. The complete R↔Python parity matrix

Every ported family, its compared invariant, its verdict, the evidence (a number, an
agreeing regime, or a formula), and the committed guard that locks it. **This is the
authoritative matrix (M5).** As of M3 every row below is a **committed, R-free guard** with
a runnable `develop/parity/<fam>.R` generator and a frozen CSV input (`tsd.csv`; plus
`logistic.csv` for the model zoo) — the pre-0.6.0 "ad-hoc" caveat no longer applies.

Verdict legend: **exact** (≤~1e-9, deterministic) · **near-exact** (≤~1e-3, same method,
MLE/float path) · **backend** (benign-numeric: params match, solver differs) ·
**stochastic** (benign-RNG: algorithm + noise scale faithful, values differ) ·
**quality** (adaptive/stochastic method → compare the outcome, not values) ·
**value-wise** (deterministic transform, compared element-for-element) ·
**++** (algorithmic improvement; gate = beats the baseline) ·
**✨** (new interface; no R counterpart) · **🐞** (bug fixed on both sides).

| Family / component (R → Py) | Invariant | Verdict | Evidence | Guard / generator |
|---|---|---|---|---|
| `ts_data`/`ts_projection`/`ts_sample`/`ts_reg`/`ts_regsw` → core | structure + metrics | **exact** | dims/columns/values identical; MSE/sMAPE/R² identical | (exercised across all guards) |
| `ts_norm_*` → `TsNormNone/Gminmax/Swminmax/Diff/An` (8 configs, all `an` ops) | transformed window matrix | **exact** 🐞 | `max\|Δ\|` ≤1e-9 (≤1e-13 ad-hoc); round-trip holds; R `ts_norm_none` x-path bug fixed (§5.1) | `normalization.R` → `test_quality_normalization.py` |
| `ts_lagmap` (12 strategies) → `TsLagmap` | selected lag **set** | **exact** | identical lags (compare lags, not positions: R 1-based, Py 0-based); MI/mRMR via `median_unbiased`=R quantile type 8; acf/pacf via statsmodels `adjusted=False`/`ywm` | `lagmap.R` → `test_quality_lagmap.py` |
| `ts_knn` → `TsKnn` | 5-step vector | **exact** | 4.7e-9 | `regressors.R` → `test_quality_regressors.py` |
| `ts_elm` → `TsElm` (native ELM) | 5-step vector | **exact** | 4.8e-9 (purelin, `nhid==n_lags` ⇒ OLS, RNG-invariant) | `regressors.R` → `test_quality_regressors.py` |
| `ts_svm` → `TsSvm` | 5-step vector | **near-exact** | 2.5e-4 (SVR + StandardScaler + TTR mirrors `e1071` scale=TRUE) | `regressors.R` → `test_quality_regressors.py` |
| `ts_arima` → `TsArima` | 5-step vector | **near-exact** | 5.98e-4 (statsmodels MLE vs native order-select) | `regressors.R` → `test_quality_regressors.py` |
| `ts_rf`/`ts_mlp` → `TsRf`/`TsMlp` | 5-step vector | **stochastic** | corr-with-actual ≥0.8 (sMAPE unusable — sine crosses zero); backend RNG | `regressors.R` → `test_quality_regressors.py` |
| `ts_deterministic` → `TsDeterministic`/`TsPeriodic`/`TsPersist` | 5-step vector | **exact** | rule-based native; ≤1e-9 | `modelzoo.R` → `test_quality_modelzoo.py` |
| `ts_darima`/`ts_warma` → `TsDarima`/`TsWarma` | 5-step vector | **exact** | ≤1e-9 vs fixed R; `_RLinear`=R `lm` column-dropping, incl. degenerate rank-3 `tsd` (§5.1) | `modelzoo.R` → `test_quality_modelzoo.py` |
| `ts_aug_flip/stretch/shrink/wormhole` → `TsAug*` | window matrix | **exact** | ≤5e-15; wormhole 0.0 | (ad-hoc 0.3.0; deterministic) |
| `ts_aug_jitter/awareness/awaresmooth` → `TsAug*` | window matrix | **stochastic** | values differ; noise scale faithful (jitter sd=0.4395701 matches) | (ad-hoc 0.3.0) |
| `ts_fil_none/ma/ema/winsor/recursive/fft/smooth/hp/gabor/lowess/kalman` → `TsFil*` | filtered series | **exact** | 11 at ≤1e-9 (hp manual solve; gabor manual kernel+conv; lowess `delta=0.01*(n-1)`,it=3; KF local-level diffuse init) | `filters.R` → `test_quality_filters.py` |
| `ts_fil_ses` → `TsFilSes` | filtered series | **near-exact** | ~8e-5 | `filters.R` → `test_quality_filters.py` |
| `ts_fil_qes/seas_adj/spline` → `TsFil*` | filtered series | **backend** | ~0.13 / ~1.2 / ~0.14 — params match, backend differs (HoltWinters / statsmodels STL / `make_smoothing_spline`) | `filters.R` → `test_quality_filters.py` |
| `ts_fil_emd`/`ts_fil_remd` → `TsFilEmd`/`TsFilRemd` (+native `emd`/`eemd`) | denoising outcome | **quality ⧺** | native; Σ IMF = signal 4e-16; R² 0.971 ≥ PyEMD 0.963; outlier removed ≥ `hht`; ~60–1800× faster; R dropped `hht` (§5.4) | `filters_emd.R` → `test_quality_emd.py` |
| `ts_fil_wavelet` → `TsFilWavelet` (+native MODWT) | filtered series | **value-wise ⧺** | ~5e-6 vs R `wavelets`, all 5 families (c6 ~1e-6: coiflet coeffs only approx-orthogonal, same in R); R dropped `wavelets` | `filters_wavelet.R` → `test_quality_wavelet.py` |
| `ts_var` → `TsVar` | y 5-step vector | **exact** | ~1e-6 (native OLS + AICc) | `multivariate.R` → `test_quality_multivariate.py` |
| `ts_lm_mv` → `TsLmMv` (known future) | y 5-step vector | **exact** | 4.97e-9 (`_RLinear`=R `lm`) | `multivariate.R` → `test_quality_multivariate.py` |
| `ts_regsw_mv` → `TsRegswMv` (KNN target + periodic aux) | y 5-step vector | **exact** | identical to 8 dp (orchestration) | `multivariate.R` → `test_quality_multivariate.py` |
| `ts_arimax` → `TsArimax` | y 5-step vector | **backend** 🐞 | ~1.37e-1 (statsmodels vs `forecast::Arima`+xreg); R `xreg` predict bug fixed (§5.2) | `multivariate.R` → `test_quality_multivariate.py` |
| `ts_tune`/`ts_integtune` → `TsTune`/`TsIntegtune` | selected hyper-params | **stochastic** | same selection class; fold-partition RNG ≠ R `sample_random` (purelin-ELM ⇒ all `nhid≥features` equivalent) | (ad-hoc 0.1.0/M4) |
| `data/*.RData` (Eidolon minis) → `datasets.<name>()` (20 + `tsd`) | element-wise values + labels | **exact** | ≤1e-9, R-free; miniatures generated from the same shrunk object by the tspredbench ETL | `datasets.R` → `test_quality_datasets.py` |
| `loadfulldata` → `datasets.loadfulldata()` | full table = `head(30)` ⇒ miniature | **exact by construction ✨** | 0 dep (urllib+pandas+gzip); CSV.gz ≈ `.RData`, beats parquet on size+deps | `test_loadfulldata.py` (`file://`, R-free) |
| `TsArima(auto="cv")` — **ARIMA++** | sMAPE vs auto.arima on NN5 | **++** | ~14% better (0.1352 vs 0.1565, 4/4); horizon-matched CV (§5.3) | `autoarima_benchmark.py` (manual) |
| `fit_arima` (exported both sides) → `pytspredit.fit_arima` | selected order + fit | **exact/near** | delegates to the AICc/CV selectors above; `(0,0,0)` sentinel triggers auto | `test_quality_regressors.py` + unit tests |
| `tspredbench.R` (doc only) → `TsPredBench` | — | **✨** | new interface; no R harness exists → `Porting: n/a` | `test_benchmark.py` |
| `plot_ts`/`plot_series`/`grf_ts_pred` (ex-daltoolbox) → `plot_ts`/`plot_series`/`plot_ts_pred`/`plot_ts_pred_mv` | — | **✨ (ownership moved)** | TS plotting family relocated out of (py)daltoolbox into (py)tspredit — the package that uses them now owns them; matplotlib port mirrors the R ggplot2 family 1:1 | `test_plot.py` |

### Documented tolerances / fidelity notes
- **Seasonal lag auto-estimation** (`seasonality=None`): periodogram estimate vs R
  `forecast::findfrequency` (now native `tslag_findfrequency`) — pass explicit `seasonality`
  for exact parity.
- **`TsNormNone` x-path** (port): returns `x` unchanged on purpose (R returned the window
  matrix — a latent sliding-window bug, §5.1). Fidelity: **adapted** (bug fix).
- **ELM-purelin == OLS exact** only when `nhid == n_features`; with `nhid>features`
  multi-step extrapolation depends on the random projection (true in both R and Python).
- **`_RLinear`** reproduces R `lm` on rank-deficient/collinear designs (drops aliased
  columns), **not** sklearn min-norm `lstsq`. Use it for every window-linear model.

---

## 7. Release status

| Side | Package | Version | State (2026-11-25) |
|---|---|---|---|
| Python | **pytspredit** | **0.12.1** | **Released to PyPI** (https://pypi.org/project/pytspredit/0.12.1/); `main` pushed. Adds the augment source-window map `df.attrs["idx"]` (mirrors R `attr(xa, "idx")`) that the augment examples use for the faceted original×augmented plot. |
| Python | **pydaltoolbox** | 0.25.0 | Released to PyPI; TS plotting (`plot_series`/`plot_ts`/`plot_ts_pred`) **removed** — ownership moved to pytspredit. |
| R | **tspredit** | 2.0.757 | On `main` (pushed); **CRAN-ready** (`R CMD check --as-cran`: 0 ERROR / 0 WARNING / 1 benign NOTE). Owns the TS plotting family (`grf_ts`/`grf_ts_pred`/`grf_series`), moved in from daltoolbox. **Eduardo submits to CRAN himself.** |
| R | **daltoolbox** | 1.4.717 | On `main`; TS plot functions removed (ownership moved to tspredit). Eduardo publishes the R side. |

**Plot-responsibility migration (sealed).** The time-series plotting family was moved out of
(py)daltoolbox — a responsibility error, since the TS package is the consumer — into
(py)tspredit, and uniformized R↔Python. Hard move (breaking), core packages only
(daltoolbox/pydaltoolbox/tspredit/pytspredit); `daltoolboxdp` and `reproduce` are deferred.
The example notebooks were re-audited for R↔Python plot parity (filters use noisy inputs so
gabor/EMD are visible; normalization compares the `t0` column; augment draws the faceted
window comparison via the new `idx`; multivariate dropped the `Agg` backend; prediction/
tutorial plot the in-sample fit + windowed outputs). Full error scan of rendered
`examples/**/*.md` is clean; 70/104 notebooks plot.

## 8. Open items for the parity maturity pass
- ~~Backfill `develop/parity/` scripts + frozen CSVs + `tests/test_quality_*` for 0.1→0.5.~~
  **DONE (M3)** — see §4; six families committed as R-free guards, one R bug fixed.
- ~~Build the full R↔Python **parity matrix**.~~ **DONE (M5)** — §6, authoritative and keyed
  to each committed guard; the category view lives in `mapeamento.md`.
- ~~Record the M3 case studies.~~ **DONE (M5)** — §5.
- Re-run every documented "benign" verdict as a seeded/agreeing-regime check so each one
  is *evidenced*, not asserted (sibling §7, "approximate as a refuge"). Partially done: the
  stochastic rows cite a seed or an agreeing regime; the few remaining ad-hoc rows
  (augmentation RNG, tuning fold RNG) still assert rather than re-run under a seed — finish
  at the 1.0.0 hardening pass.
