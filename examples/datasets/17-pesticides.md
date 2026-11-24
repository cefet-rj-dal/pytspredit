# FAOSTAT pesticides use (`datasets.pesticides`)

**About the dataset**

* `datasets.pesticides()` returns the Eidolon **miniature** -- a dict of named series -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.pesticides()
print("series:", list(d))
next(iter(d.values())).head()
```

    series: ['india_pesticides']





    0    75000.0
    1    72133.0
    2    70791.0
    3    66074.0
    4    61357.0
    Name: india_pesticides, dtype: float64




```python
ts.datasets.dataset_url("pesticides")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/pesticides.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** FAO (2024), *FAOSTAT Pesticides Use*.
