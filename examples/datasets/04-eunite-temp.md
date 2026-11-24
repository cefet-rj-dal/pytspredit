# EUNITE daily temperatures (`datasets.eunite_temp`)

**About the dataset**

* `datasets.eunite_temp()` returns the Eidolon **miniature** -- a data frame (one column per series / interval) -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.eunite_temp()
print("shape:", d.shape)
d.head(3)
```

    shape: (30, 2)





       Temperature  split
    0          0.0  train
    1         -1.6  train
    2         -1.9  train




```python
ts.datasets.dataset_url("EUNITE.Temp")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/EUNITE.Temp.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** Chen, Chang & Lin (2004), IEEE Trans. Power Systems, 19(4).
