# pytspredit

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

Once released:

```bash
pip install pytspredit
```

For a local, editable clone (contributing or running the examples):

```bash
git clone https://github.com/cefet-rj-dal/pytspredit.git
cd pytspredit
pip install -e ".[dev,docs]"
pytest
```

## Sliding windows in a nutshell

```python
import pytspredit as ts

series = ts.datasets.tsd()["y"]       # synthetic sine example series

windows = ts.ts_data(series, sw=10)   # sliding windows: columns t9 ... t0
samp = ts.ts_sample(windows, test_size=5)       # temporal train/test split
io_train = ts.ts_projection(samp.train)         # -> (input X, output t0)
io_test = ts.ts_projection(samp.test)

io_train.input.head()
```

Regressors, normalizations, filters, augmentation and tuning build on top of this core
(see the roadmap).

## Status

🚧 **In development.** The port follows the method validated for `pydaltoolbox`
(already on PyPI). Done so far:

- **Core data structures:** `ts_data` (sliding windows), `ts_projection`, `ts_sample`,
  `adjust_ts_data`, `ts_head`, and `datasets.tsd()` (numerically identical to the R
  original).

Coming next: regression base + regressors, normalizations, lag-mapping strategies,
ARIMA, filters, augmentation and tuning. The full kickoff brief and roadmap are in
[develop/PROMPT.md](develop/PROMPT.md) and [develop/planejamento.md](develop/planejamento.md).

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
