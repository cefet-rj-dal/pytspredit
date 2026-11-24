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

# IBOVESPA most-traded stocks (`datasets.stocks`)

**About the dataset**

* `datasets.stocks()` returns the Eidolon **miniature** -- a dict of data frames -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```

```python
d = ts.datasets.stocks()
print("keys:", list(d))
next(iter(d.values())).head(3)
```

```python
ts.datasets.dataset_url("stocks")   # full artifact (loadfulldata)
```

> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** B3 (2025), *Historical stock trading data*. https://www.b3.com.br
