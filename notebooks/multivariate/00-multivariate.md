---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Multivariate Forecasting (target-centered)

The multivariate branch keeps one distinguished **target** `y` while the auxiliary
variables `x1, ..., xn` either help forecast `y` or are themselves forecast by
their own pipelines. Every multivariate `predict` returns the forecast of `y`
(as an array) with the whole-system forecast attached (`.system`).

Two orchestration styles mirror the univariate split:

* **`TsRegswMv`** — the sliding-window orchestrator: one univariate learner per
  variable (`ts_mv_spec`), coordinated over a shared `window_size`.
* **`TsRegMv`** — the singular (aligned) branch: `TsLmMv` (linear regression),
  `TsVar` (vector autoregression) and `TsArimax` (ARIMA with external regressors).

```python
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytspredit as ts

# Build the canonical multivariate data from tsd (as in the R examples):
# y = sine, x1 = y shifted one step, x2 = 3-point moving average of y.
y = ts.datasets.tsd()["y"].to_numpy()
x1 = np.concatenate([y[1:], y[-1:]])
x2 = pd.Series(y).rolling(3).mean().to_numpy()
mask = np.isnan(x2); x2[mask] = y[mask]

mv = ts.ts_data_mv(pd.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y")
ts.ts_head(mv, 3)
```

`ts_data_mv` carries the multivariate metadata (target, auxiliaries, representation)
so it travels through sampling:

```python
samp = ts.ts_sample(mv, test_size=5)
(mv.mv_y, mv.mv_x, mv.mv_representation, samp.train.mv_representation)
```

## Vector autoregression — `TsVar`

`TsVar` learns the whole system jointly by OLS over lagged observations, choosing
the lag order by AICc. *(Lütkepohl, 2005; Tsay, 2014.)*

```python
model = ts.TsVar(p_max=3).fit(samp.train)
pred = model.predict(steps_ahead=5)
pred.system
```

The forecast of the target is the array itself; it evaluates like any other model:

```python
model.evaluate(samp.test["y"].to_numpy(), np.asarray(pred))["metrics"]
```

## Multivariate linear regression — `TsLmMv`

`TsLmMv` regresses the target on the synchronized auxiliaries. When the future
auxiliary values are known (held-out data), pass them to `predict`. *(Montgomery,
Peck & Vining, 2021.)*

```python
lm = ts.TsLmMv(formula="y ~ x1 + x2").fit(samp.train)
np.asarray(lm.predict(samp.test, steps_ahead=5)).round(6)
```

When the future auxiliaries are unknown, provide one univariate model per
auxiliary (`models_x`) to generate their paths:

```python
lm2 = ts.TsLmMv(models_x={"x1": ts.TsArima(), "x2": ts.TsArima()}).fit(samp.train)
np.asarray(lm2.predict(steps_ahead=5)).round(6)
```

## ARIMA with external regressors — `TsArimax`

`TsArimax` is the singular-multivariate counterpart of `TsArima`: the auxiliaries
enter as external regressors. *(Box et al., 2015; Hyndman & Athanasopoulos, 2021.)*

```python
ax = ts.TsArimax(p=2, d=0, q=0).fit(samp.train)
np.asarray(ax.predict(samp.test, steps_ahead=5)).round(6)
```

## Sliding-window orchestration — `TsRegswMv`

One sliding-window learner forecasts the target while each auxiliary is forecast
by its own pipeline; the system is rolled forward recursively. Here the target is
a deterministic k-NN and the auxiliaries are periodic — a fully deterministic
configuration that matches R exactly.

```python
orch = ts.TsRegswMv(
    model_y=ts.ts_mv_spec(ts.TsKnn(ts.TsNormGminmax(), k=3), variables=["y", "x1", "x2"]),
    models_x={"x1": ts.ts_mv_spec(ts.TsPeriodic(7)),
              "x2": ts.ts_mv_spec(ts.TsPeriodic(7))},
    window_size=5).fit(samp.train)
orch.predict(steps_ahead=5).system
```

## Plotting the forecast paths

`plot_ts_pred_mv` returns one matplotlib figure per variable (or a single figure
when `variable` is given), reusing `plot_ts_pred` variable by variable.

```python
plots = ts.plot_ts_pred_mv(samp.train, samp.test, model.predict(steps_ahead=5))
{name: type(fig).__name__ for name, fig in plots.items()}
```

## R vs Python parity

On this data the native and linear models match R to numerical precision; the
statistical backends carry a documented tolerance:

| Model | R vs Python | Note |
|---|---|---|
| `TsVar` | ~1e-6 (exact) | native OLS VAR + AICc |
| `TsLmMv` (known future) | ~5e-9 (exact) | reuses R's `lm` column-dropping |
| `TsRegswMv` (deterministic) | exact | k-NN target + periodic aux |
| `TsArimax` | backend tolerance | statsmodels vs `forecast::Arima` + xreg |
