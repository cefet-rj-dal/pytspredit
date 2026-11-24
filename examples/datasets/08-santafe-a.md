# Santa Fe series A (laser) (`datasets.santafe_a`)

**About the dataset**

* `datasets.santafe_a()` returns the Eidolon **miniature** -- a data frame (one column per series / interval) -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.santafe_a()
print("shape:", d.shape)
d.head(3)
```

    shape: (30, 2)





        V1  split
    0   86  train
    1  141  train
    2   95  train




```python
ts.datasets.dataset_url("SantaFe.A")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/SantaFe.A.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** Weigend (1993), *Time Series Prediction*, Westview Press.
