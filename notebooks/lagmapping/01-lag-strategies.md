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

# Lag-Mapping Strategies

A sliding-window regressor does not have to use *every* lag. `TsLagmap` learns, during
`fit`, which `input_size` lag columns to feed the backend. Twelve strategies are
available, grouped by the evidence they use:

* **positional:** `recent`, `even`, `geom`
* **correlation-driven:** `acf`, `pacf`, `peaks`, `seasonal`, `acf_seasonal`,
  `pacf_seasonal`, `blocks`
* **supervised:** `mi` (mutual information), `mrmr` (min-redundancy max-relevance)

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis*;
Peng, Long & Ding (2005), *mRMR feature selection*, IEEE TPAMI.

```python
import pytspredit as ts

io = ts.ts_projection(ts.ts_data(ts.datasets.tsd()["y"], 10))
print("available lag columns:", list(io.input.columns))
```

Each strategy returns the selected lags (`lag 1` = most recent feature) for a given
`input_size`:

```python
methods = ["recent", "even", "geom", "acf", "pacf", "peaks",
           "seasonal", "acf_seasonal", "pacf_seasonal", "blocks", "mi", "mrmr"]

for m in methods:
    lm = ts.TsLagmap(method=m, seasonality=4).fit(io.input, io.output, input_size=4)
    print(f"{m:14s} lags={list(lm.lags)}  columns={lm.columns}")
```

A lag map is just another component of a sliding-window regressor; passing
`input_map=ts.TsLagmap('pacf')` tells the model to select lags by partial
autocorrelation instead of simply keeping the most recent ones.
