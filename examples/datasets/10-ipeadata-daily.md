# Ipea daily macroeconomic series (`datasets.ipeadata_d`)

**About the dataset**

* `datasets.ipeadata_d()` returns the Eidolon **miniature** -- a data frame (one column per series / interval) -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.ipeadata_d()
print("shape:", d.shape)
d.head(3)
```

    shape: (30, 13)





       GM366_IBVSP366  GM366_ERC366  GM366_EREURO366  GM366_ERPV366  GM366_ERV366  GM366_TJOVER366  GM366_TJTR366  SECEX366_MVTOT366  SECEX366_XVTOT366  JPM366_EMBI366  BM366_TJOVER366  GM366_TJOVERV366  split
    0          3580.8  1.152000e-09         0.841157   1.345450e-13  1.157820e-09             17.8         0.7353              126.7              133.3            1120          25.3401          0.004332  train
    1          3564.3  1.152000e-09         0.847678   1.381820e-13  1.157820e-09             18.5         0.8169              170.4              220.4            1131          25.3401          0.009892  train
    2          3753.5  1.152000e-09         0.859567   1.309090e-13  1.157820e-09             18.3         0.7903              212.6              161.0            1081          25.3401          0.013051  train




```python
ts.datasets.dataset_url("ipeadata.d")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/ipeadata.d.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** Ipea (2017), *Ipeadata*. http://www.ipeadata.gov.br
