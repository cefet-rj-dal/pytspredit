# CATS artificial series with missing blocks (`datasets.cats`)

**About the dataset**

* `datasets.cats()` returns the Eidolon **miniature** -- a data frame (one column per series / interval) -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.cats()
print("shape:", d.shape)
d.head(3)
```

    shape: (30, 6)





          V1      V2      V3    V4      V5  split
    0  -2.85  150.21  328.40  2.00  290.28  train
    1 -13.07  136.41  311.23  5.11  298.11  train
    2 -17.81  126.87  312.86  5.40  286.62  train




```python
ts.datasets.dataset_url("CATS")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/CATS.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** Lendasse et al. (2007), Neurocomputing, 70(13-15), 2325-2329.
