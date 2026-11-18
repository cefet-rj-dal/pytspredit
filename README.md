# pytspredit

[![PyPI version](https://img.shields.io/pypi/v/pytspredit.svg)](https://pypi.org/project/pytspredit/)
[![Python versions](https://img.shields.io/pypi/pyversions/pytspredit.svg)](https://pypi.org/project/pytspredit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python port of the R [tspredit](https://github.com/cefet-rj-dal/tspredit) — **time series
prediction with tunable preprocessing**, built on the DAL Toolbox *Experiment Line*.

`pytspredit` provides a uniform, extensible interface for time series forecasting:
sliding-window modeling, normalization, filtering and augmentation, machine-learning and
statistical regressors, and hyperparameter tuning — reusing the best of the Python
ecosystem (scikit-learn, numpy, pandas, statsmodels) through thin adapters. Its value is
**not** reimplementing algorithms, but the rigid, extensible interface that wraps them.

It is part of the DAL Python family:

- Builds on **[pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox)** — the
  foundation (Python port of the R `daltoolbox`), on PyPI.
- Is, in turn, a dependency of **pyharbinger** (time series event detection).

## Installation

```bash
pip install pytspredit
```

> **Beta:** the current release (`0.1.0`) is in a testing phase on the way to `1.0.0`.
> Feedback and issues are very welcome.

For a local, editable clone (contributing or running the examples):

```bash
git clone https://github.com/cefet-rj-dal/pytspredit.git
cd pytspredit
pip install -e ".[dev,docs]"
pytest
```

## A forecast in a nutshell

```python
import pytspredit as ts

series = ts.datasets.tsd()["y"]                  # example series
windows = ts.ts_data(series, sw=10)              # sliding windows: columns t9 ... t0
samp = ts.ts_sample(windows, test_size=5)        # temporal train/test split
io_train = ts.ts_projection(samp.train)          # -> (input X, output t0)
io_test = ts.ts_projection(samp.test)

# a kNN regressor with min-max normalization and PACF-based lag selection
model = ts.TsKnn(ts.TsNormGminmax(), input_size=4, input_map=ts.TsLagmap("pacf"), k=3)
model.fit(io_train.input, io_train.output)

forecast = model.predict(io_test.input.iloc[[0]], steps_ahead=5)   # 5 steps ahead
print(model.evaluate(io_test.output.to_numpy(), forecast)["metrics"])
```

## Status

**v1 published (`0.1.0`, beta).** The port follows the method validated for
`pydaltoolbox` (already on PyPI), with **R-vs-Python parity checked at every step** on
the shared `tsd` series. Implemented:

- **Core data structures:** `ts_data` (sliding windows), `ts_projection`, `ts_sample`,
  `adjust_ts_data`, `ts_head`, `datasets.tsd()` (numerically identical to R).
- **Regression base:** `TsReg` (MSE / sMAPE / R2 evaluation) and `TsRegSW`
  (preprocess + lag map + backend; one-step and iterative multi-step forecasting).
- **Normalizations:** `TsNormNone`, `TsNormGminmax`, `TsNormSwminmax`, `TsNormDiff`,
  `TsNormAn` (adaptive; 4 operators) — all invertible.
- **Lag mapping:** `TsLagmap` with 12 strategies (recent, even, geom, acf, pacf, peaks,
  seasonal, acf/pacf-seasonal, blocks, mutual information, mRMR).
- **Regressors:** `TsKnn`, `TsElm`, `TsRf`, `TsSvm`, `TsMlp` (sliding-window) and
  `TsArima` (univariate ARIMA).
- **Tuning:** `TsTune` and `TsIntegtune` (integrated tuning of input size,
  preprocessing and model hyperparameters).

Thin adapters over scikit-learn / numpy / pandas / statsmodels reproduce the R
backends' defaults (e.g. SVM standardizes like `e1071`). Deferred to a later release
(see [develop/planejamento.md](develop/planejamento.md)): the multivariate family
(`ts_data_mv`, `ts_var`, `ts_arimax`), filters, the full augmentation family, and
graphics.

## Documentation

- [Processed examples](examples/) — notebooks executed end to end, with real output
  embedded. Sources live under [notebooks/](notebooks/) (jupytext Markdown), rendered by
  `python tools/build_examples.py`.
- [develop/planejamento.md](develop/planejamento.md) — migration roadmap (phases, scope, status).
- [develop/deploy.md](develop/deploy.md) — PyPI publishing guide.

## Related DAL projects

- [pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox) — foundation (Python)
- [daltoolbox](https://github.com/cefet-rj-dal/daltoolbox) / [tspredit](https://github.com/cefet-rj-dal/tspredit) / [harbinger](https://github.com/cefet-rj-dal/harbinger) — R originals
- [Data Analytics Lab](https://eic.cefet-rj.br/~dal/)

## License

MIT
