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

# Santa Fe series D (`datasets.santafe_d`)

**About the dataset**

* `datasets.santafe_d()` returns the Eidolon **miniature** -- a data frame (one column per series / interval) -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```

```python
d = ts.datasets.santafe_d()
print("shape:", d.shape)
d.head(3)
```

```python
ts.datasets.dataset_url("SantaFe.D")   # full artifact (loadfulldata)
```

> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** Weigend (1993), *Time Series Prediction*, Westview Press.
