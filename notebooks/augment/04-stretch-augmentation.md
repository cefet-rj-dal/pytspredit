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

# Data augmentation -- Stretch (`TsAugStretch`)

**About the technique**

* `TsAugStretch` scales the within-window deviations up (amplifies variation; the target is kept).
* Every augmenter is a `DalTransform` over the full window and, by default, stacks the original windows on top of the augmented ones (native; no new dependency).

**Didactic goal:** add higher-variance variants.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

windows = ts.ts_data(ts.datasets.tsd()["y"], 10)
print("original windows:", windows.shape)
```

```python
out = ts.TsAugStretch(scale_factor=1.2).fit(windows).transform(windows)
print('after augmentation:', out.shape)
ts.ts_head(out, 3)
```

> **What to observe.** The output stacks the original and the augmented windows (roughly double the rows); it plugs into `TsIntegtune`'s augmentation slot.

**References:** Wen, Q. et al. (2021). *Time Series Data Augmentation for Deep Learning: A Survey.* IJCAI.
