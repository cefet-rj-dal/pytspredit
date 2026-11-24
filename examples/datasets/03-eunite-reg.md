# EUNITE load-forecasting regressors (`datasets.eunite_reg`)

**About the dataset**

* `datasets.eunite_reg()` returns the Eidolon **miniature** -- a data frame (one column per series / interval) -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.eunite_reg()
print("shape:", d.shape)
d.head(3)
```

    shape: (30, 3)





       Holiday  Weekday  split
    0        1        4  train
    1        0        5  train
    2        0        6  train




```python
ts.datasets.dataset_url("EUNITE.Reg")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/EUNITE.Reg.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** Chen, Chang & Lin (2004), IEEE Trans. Power Systems, 19(4).
