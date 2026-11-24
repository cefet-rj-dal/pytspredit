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

# Model comparison

**Didactic goal:** Compare several forecasters on the same split with TsPredBench.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts
```

```python
bench = ts.TsPredBench(
    datasets={"tsd": ts.datasets.tsd()["y"]},
    models={"persist": lambda: ts.TsPersist(),
            "knn": lambda: ts.TsKnn(ts.TsNormGminmax(), input_size=5),
            "mlp": lambda: ts.TsMlp(ts.TsNormGminmax(), input_size=5)},
    input_size=5, steps_ahead=5)
bench.run()
```

> **What to observe.** A tidy leaderboard makes model selection explicit.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
