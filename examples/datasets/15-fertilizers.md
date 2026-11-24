# FAOSTAT fertilizers by nutrient (`datasets.fertilizers`)

**About the dataset**

* `datasets.fertilizers()` returns the Eidolon **miniature** -- a dict of named series -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.fertilizers()
print("series:", list(d))
next(iter(d.values())).head()
```

    series: ['usa_n']





    0    12.96
    1    15.18
    2    16.85
    3    18.34
    4    20.69
    Name: usa_n, dtype: float64




```python
ts.datasets.dataset_url("fertilizers")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/fertilizers.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** FAO (2024), *FAOSTAT Fertilizers by Nutrient*.
