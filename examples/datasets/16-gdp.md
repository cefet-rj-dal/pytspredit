# FAOSTAT GDP / agriculture value added (`datasets.gdp`)

**About the dataset**

* `datasets.gdp()` returns the Eidolon **miniature** -- a dict of named series -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.gdp()
print("series:", list(d))
next(iter(d.values())).head()
```

    series: ['usa_gdp']





    0    1073303.0
    1    1164850.0
    2    1279110.0
    3    1425376.0
    4    1545243.0
    Name: usa_gdp, dtype: float64




```python
ts.datasets.dataset_url("gdp")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/gdp.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** FAO (2024), *FAOSTAT Macro Indicators*. doi:10.4060/cd0763en
