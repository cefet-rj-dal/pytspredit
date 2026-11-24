# Lag mapping -- mRMR lags (`TsLagmap(method="mrmr")`)

**About the technique**

* `TsLagmap(method="mrmr")` selects lags that are relevant to the target yet non-redundant with each other (min-redundancy max-relevance).

**Didactic goal:** avoid redundant lags while keeping relevance.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

io = ts.ts_projection(ts.ts_data(ts.datasets.tsd()["y"], 10))
```


```python
lm = ts.TsLagmap(method="mrmr", seasonality=4).fit(io.input, io.output, input_size=4)
print("selected lags:", list(lm.lags))
print("columns:", lm.columns)
```

    selected lags: [np.int64(9), np.int64(6), np.int64(2), np.int64(1)]
    columns: ['t9', 't6', 't2', 't1']


> **What to observe.** The strategy returns `input_size` lag positions (`lag 1` = most recent); these are the columns actually fed to the backend.

**References:** Peng, Long & Ding (2005), *Feature selection based on mutual information: mRMR*, IEEE TPAMI 27(8).
