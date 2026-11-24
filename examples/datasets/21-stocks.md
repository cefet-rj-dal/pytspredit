# IBOVESPA most-traded stocks (`datasets.stocks`)

**About the dataset**

* `datasets.stocks()` returns the Eidolon **miniature** -- a dict of data frames -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.stocks()
print("keys:", list(d))
next(iter(d.values())).head(3)
```

    keys: ['VALE3']





             date      open      high       low     close   volume
    0  2000-01-03  3.500000  3.542500  3.500000  3.500000   585600
    1  2000-01-04  3.466666  3.474166  3.416666  3.416666   782400
    2  2000-01-05  3.375000  3.416666  3.375000  3.416666  1876800




```python
ts.datasets.dataset_url("stocks")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/stocks.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** B3 (2025), *Historical stock trading data*. https://www.b3.com.br
