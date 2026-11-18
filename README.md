# pytspredit

Python port of the R [tspredit](https://github.com/cefet-rj-dal/tspredit) — **time series
prediction with tunable preprocessing**, built on the DAL Toolbox *Experiment Line*.

`pytspredit` provides a uniform, extensible interface for time series forecasting:
sliding-window modeling, normalization, filtering and augmentation, machine-learning and
statistical regressors, and hyperparameter tuning — reusing the best of the Python
ecosystem (scikit-learn, numpy, pandas, statsmodels) through thin adapters.

It is part of the DAL Python family:

- Builds on **[pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox)** — the
  foundation (Python port of the R `daltoolbox`), already on PyPI.
- Is, in turn, a dependency of **pyharbinger** (time series event detection).

## Status

🚧 **Planned / in development.** The port follows the same method validated for
`pydaltoolbox`. The full kickoff brief and roadmap are in
[develop/PROMPT.md](develop/PROMPT.md).

Once released:

```bash
pip install pytspredit
```

## Related DAL projects

- [pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox) — foundation (Python)
- [daltoolbox](https://github.com/cefet-rj-dal/daltoolbox) / [tspredit](https://github.com/cefet-rj-dal/tspredit) / [harbinger](https://github.com/cefet-rj-dal/harbinger) — R originals
- [Data Analytics Lab](https://eic.cefet-rj.br/~dal/)

## License

MIT
