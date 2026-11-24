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

# Ipea monthly macroeconomic series (`datasets.ipeadata_m`)

**About the dataset**

* `datasets.ipeadata_m()` returns the Eidolon **miniature** -- a data frame (one column per series / interval) -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```

```python
d = ts.datasets.ipeadata_m()
print("shape:", d.shape)
d.head(3)
```

```python
ts.datasets.dataset_url("ipeadata.m")   # full artifact (loadfulldata)
```

> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** Ipea (2017), *Ipeadata*. http://www.ipeadata.gov.br
