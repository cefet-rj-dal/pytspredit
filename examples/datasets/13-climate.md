# FAOSTAT/NASA-GISS temperature change (`datasets.climate`)

**About the dataset**

* `datasets.climate()` returns the Eidolon **miniature** -- a dict of named series -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.climate()
print("series:", list(d))
next(iter(d.values())).head()
```

    series: ['usa_climate']





    0    0.003
    1   -0.027
    2    0.425
    3   -0.281
    4   -0.360
    Name: usa_climate, dtype: float64




```python
ts.datasets.dataset_url("climate")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/climate.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** FAO (2024); GISTEMP Team (2024), NASA GISS.
