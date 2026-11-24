# EUNITE half-hourly electrical loads (`datasets.eunite_loads`)

**About the dataset**

* `datasets.eunite_loads()` returns the Eidolon **miniature** -- a data frame (one column per series / interval) -- numerically identical to the R package. The full artifact is resolved on demand with `datasets.loadfulldata`.

**Didactic goal:** load and inspect the dataset's structure.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```


```python
d = ts.datasets.eunite_loads()
print("shape:", d.shape)
d.head(3)
```

    shape: (30, 49)





       X00.30  X01.00  X01.30  X02.00  X02.30  X03.00  X03.30  X04.00  X04.30  X05.00  X05.30  X06.00  X06.30  X07.00  X07.30  X08.00  X08.30  X09.00  X09.30  X10.00  X10.30  X11.00  X11.30  X12.00  X12.30  X13.00  X13.30  X14.00  X14.30  X15.00  X15.30  X16.00  X16.30  X17.00  X17.30  X18.00  X18.30  X19.00  X19.30  X20.00  X20.30  X21.00  X21.30  X22.00  X22.30  X23.00  X23.30  X24.00  split
    0     797     794     784     787     763     749     745     730     707     706     720     657     633     595     560     540     519     601     631     621     640     643     654     653     688     688     690     690     684     679     674     677     644     660     654     683     688     698     719     733     700     671     692     685     717     694     692     686  train
    1     704     697     704     676     664     668     668     662     665     666     703     677     669     660     650     672     648     682     692     724     727     739     739     733     741     754     767     768     738     734     747     733     751     746     737     750     759     776     777     777     746     724     697     708     745     705     702     722  train
    2     753     720     710     705     691     698     702     673     709     703     701     702     704     682     673     694     688     700     736     730     758     772     753     749     755     774     797     771     775     758     764     760     756     757     738     753     744     774     768     771     731     729     692     701     719     694     698     733  train




```python
ts.datasets.dataset_url("EUNITE.Loads")   # full artifact (loadfulldata)
```




    'https://raw.githubusercontent.com/cefet-rj-dal/tspredbench/refs/heads/main/tspredit/EUNITE.Loads.RData'



> **What to observe.** The miniature preserves the schema of the full dataset (same columns/keys); use `loadfulldata` to fetch the complete series.

**References:** Chen, Chang & Lin (2004), IEEE Trans. Power Systems, 19(4), 1821-1830.
