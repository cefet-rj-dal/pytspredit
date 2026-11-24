# FAOSTAT bioenergy series (`datasets.bioenergy`)

**About the dataset**

* `datasets.bioenergy()` returns the Eidolon **miniature** -- a dict of named series -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.bioenergy()
print("series:", list(d))
next(iter(d.values())).head()
```

    series: ['usa_bio_consumption']





    0     980185
    1    1130112
    2    1384457
    3    1016991
    4    1022212
    Name: usa_bio_consumption, dtype: int64




```python
ts.datasets.dataset_url("bioenergy")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/bioenergy.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** FAO (2024), *FAOSTAT Bioenergy*.
