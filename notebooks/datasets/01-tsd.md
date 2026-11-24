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

# Synthetic example series (`tsd`) (`datasets.tsd`)

**About the dataset**

* `datasets.tsd()` returns the Eidolon **miniature** -- a data frame (one column per series / interval) -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```

```python
d = ts.datasets.tsd()
print("shape:", d.shape)
d.head(3)
```

> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
