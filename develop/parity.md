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

A **retroactive pass** will backfill `develop/parity/` + `tests/test_quality_*` for
0.1→0.5 as one of the maturity releases before 1.0.0 (see `planejamento.md` §5.4/Maturidade).

---

## 5. Results so far (ad-hoc, 0.1.0 → 0.5.0)

All on the shared `tsd` series unless noted. To be re-expressed as committed guards.

| Component | Verdict | Evidence |
|---|---|---|
| `ts_data`/`ts_projection`/`ts_sample` | **exact** | dims/columns/values identical to R |
| Normalization (7 configs) | **exact** | `max\|Δ\|` ≤1e-13; round-trip holds |
| `TsLagmap` (12 strategies) | **exact** | identical lags/positions (MI/mRMR via `median_unbiased`=R quantile type 8; acf/pacf via statsmodels `adjusted=False`/`ywm`) |
| `TsKnn` / `TsElm` | **exact** | 4.7e-9 / 4.8e-9 (deterministic; ELM purelin+nhid≥nlags == OLS) |
| `TsArima` | near-exact | 6.0e-4 (MLE backend vs `forecast`) |
| `TsSvm` | near-exact | 2.5e-4 |
| `TsRf` / `TsMlp` | benign-stochastic | 3.6e-2 / 2.2e-2 (backend RNG; documented tolerance) |
| `TsDeterministic`/`TsPeriodic`/`TsPersist` | **exact** | rule-based, native |
| `TsDarima` / `TsWarma` | **exact** | ≤4.8e-9 vs fixed R branch; `_RLinear` reproduces R `lm` column-dropping, incl. degenerate pure-sine (both rank 3) |
| Augmentation (deterministic) | **exact** | ≤5e-15; wormhole 0.0 |
| Augmentation (RNG: jitter/awareness/awaresmooth) | benign-stochastic | values differ; algorithm + noise scale faithful (jitter sd=0.4395701 matches) |
| Filters: none/ma/ema/winsor/recursive/fft/smooth/hp/gabor | **exact** | ~5e-9 (hp manual solve; gabor manual kernel+conv) |
| Filters: lowess/kalman | **exact** | ~5e-9 (lowess `delta=0.01*(n-1)`,it=3; KF local-level diffuse init) |
| Filters: ses | near-exact | ~8e-5 |
| Filters: qes/seas_adj/spline | benign-numeric (backend) | ~0.13 / ~1.2 / ~0.14 — params match, backend differs (HoltWinters / statsmodels STL / `make_smoothing_spline`) |
| `TsVar` | **exact** | ~1e-6 (native OLS + AICc) |
| `TsLmMv` (known future) | **exact** | 4.97e-9 (`_RLinear`=R `lm`) |
| `TsRegswMv` (KNN target + periodic aux) | **exact** | identical to 8 dp (orchestration) |
| `TsArimax` | benign-numeric (backend) | ~1.4e-1 (statsmodels vs `forecast::Arima`+xreg) |
| `TsTune`/`TsIntegtune` | benign-stochastic | same selection class; fold partition RNG ≠ R `sample_random` (documented) |

### Documented tolerances / fidelity notes
- **Seasonal lag auto-estimation** (`seasonality=None`): periodogram estimate vs R
  `forecast::findfrequency` — pass explicit `seasonality` for exact parity.
- **`TsNormNone` x-path** (port): returns `x` unchanged on purpose (R returned the window
  matrix — a latent sliding-window bug). Fidelity: **adapted** (bug fix).
- **ELM-purelin == OLS exact** only when `nhid == n_features`; with `nhid>features`
  multi-step extrapolation depends on the random projection (true in both R and Python).
- **`_RLinear`** reproduces R `lm` on rank-deficient/collinear designs (drops aliased
  columns), **not** sklearn min-norm `lstsq`. Use it for every window-linear model.

---

## 6. Open items for the parity maturity pass
- Backfill `develop/parity/` scripts + frozen CSVs + `tests/test_quality_*` for 0.1→0.5.
- Build the full R↔Python **parity matrix** (goes into `mapeamento.md` / 1.0.0).
- Re-run every documented "benign" verdict as a seeded/agreeing-regime check so each one
  is *evidenced*, not asserted (sibling §7, "approximate as a refuge").
