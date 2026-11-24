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

# Data augmentation -- Awareness (`TsAugAwareness`)

**About the technique**

* `TsAugAwareness` oversamples the more recent windows so training weights recent behavior more.
* Every augmenter is a `DalTransform` over the full window and, by default, stacks the original windows on top of the augmented ones (native; no new dependency).

**Didactic goal:** bias the sample toward recent dynamics.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

windows = ts.ts_data(ts.datasets.tsd()["y"], 10)
print("original windows:", windows.shape)
```

Visualize the series used in this example:

```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```


```python
out = ts.TsAugAwareness(factor=1, random_state=0).fit(windows).transform(windows)
print('after augmentation:', out.shape)
ts.ts_head(out, 3)
```

> **What to observe.** The output stacks the original and the augmented windows (roughly double the rows); it plugs into `TsIntegtune`'s augmentation slot.

**References:** Wen, Q. et al. (2021). *Time Series Data Augmentation for Deep Learning: A Survey.* IJCAI.
