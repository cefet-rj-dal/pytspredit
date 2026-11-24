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




      dataset    model       mse     smape        r2
    0     tsd  persist  0.599168  1.507829 -4.175042
    1     tsd      knn  0.000966  0.157581  0.991658
    2     tsd      mlp  0.000700  0.147120  0.993955



> **What to observe.** A tidy leaderboard makes model selection explicit.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
