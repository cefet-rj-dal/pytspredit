# Ipea monthly macroeconomic series (`datasets.ipeadata_m`)

**About the dataset**

* `datasets.ipeadata_m()` returns the Eidolon **miniature** -- a data frame (one column per series / interval) -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.ipeadata_m()
print("shape:", d.shape)
d.head(3)
```

    shape: (30, 24)





         BM12_ERC12    BM12_ERV12  IGP12_IGPDI12  FUNCEX12_MDPT12  FUNCEX12_XPT12  PRECOS12_INPC12  PRECOS12_INPCBR12  PRECOS12_IPCA12  SEADE12_TDAGSP12  SEADE12_TDOTSP12  SEADE12_TDOPSP12  GAC12_SALMINRE12  IGP12_IGPM12  PRECOS12_IPCAG12  IGP12_IGPDIG12  IGP12_IGPMG12  IGP12_IGPOGG12  PRECOS12_IPCA15G12  BM12_PIB12  MTE12_SALMIN12  BM12_TJOVER12  SEADE12_TDTGSP12  PMEN12_TD12  split
    0  6.683640e-15  3.272730e-15   8.371380e-14            59.42           68.05     5.576500e-09               3.45     7.618300e-09               7.3               3.4               2.4        805.981426      0.000031          6.615649            1.45          35.91            1.36                0.09         0.2    8.727270e-14           1.46              12.4         12.9  train
    1  1.337830e-14  3.272730e-15   8.492700e-14            59.50           68.22     5.768900e-09               1.76     8.122300e-09               7.2               3.7               2.7        803.730979      0.000042          4.616919            1.43          36.92            0.50                0.08         0.4    8.727270e-14           1.15              12.0         12.5  train
    2  1.337830e-14  3.163640e-15   8.614020e-14            60.01           73.92     5.870400e-09               3.00     8.497300e-09               7.9               3.9               2.9        803.008271      0.000057          6.038389            1.41          39.92            1.34                0.78         0.8    8.727270e-14           1.16              13.1         11.9  train




```python
ts.datasets.dataset_url("ipeadata.m")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/ipeadata.m.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** Ipea (2017), *Ipeadata*. http://www.ipeadata.gov.br
