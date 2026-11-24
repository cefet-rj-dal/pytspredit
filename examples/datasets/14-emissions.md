# FAOSTAT greenhouse-gas emissions (`datasets.emissions`)

**About the dataset**

* `datasets.emissions()` returns the Eidolon **miniature** -- a dict of named series -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.emissions()
print("series:", list(d))
next(iter(d.values())).head()
```

    series: ['usa_ch4']





    0    8771.6993
    1    8946.6938
    2    9093.2367
    3    9177.3663
    4    9084.8934
    Name: usa_ch4, dtype: float64




```python
ts.datasets.dataset_url("emissions")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/emissions.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** FAO (2023), *FAOSTAT Emissions Totals*.
