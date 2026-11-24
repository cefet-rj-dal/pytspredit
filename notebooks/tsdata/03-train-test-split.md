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

# Train/test split of `ts_data`

**About the technique**

* `ts_sample()` splits the windowed `ts_data` into training and test sets **in time
  order** (the last `test_size` windows are held out), as required for forecasting.

**Didactic goal:** produce an honest, temporally ordered holdout for evaluation.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```

Visualize the series used in this example:

```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```


```python
y = ts.datasets.tsd()["y"]
w = ts.ts_data(y, 5)
samp = ts.ts_sample(w, test_size=5)
print("train rows:", len(samp.train), "| test rows:", len(samp.test))
```

The training set keeps the earlier windows.

```python
ts.ts_head(samp.train)
```

The test set is the final `test_size` windows, untouched during training.

```python
ts.ts_head(samp.test)
```

> **What to observe.** The split is chronological (no shuffling): the test windows come
> strictly after the training windows, so evaluation never peeks at the future.

**References:** Tashman (2000), *Out-of-sample tests of forecasting accuracy*,
International Journal of Forecasting, 16(4), 437-450.
