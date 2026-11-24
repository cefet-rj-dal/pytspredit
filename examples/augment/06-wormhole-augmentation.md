# Data augmentation -- Wormhole (`TsAugWormhole`)

**About the technique**

* `TsAugWormhole` recombines lag content from older windows into new plausible windows (deterministic).
* Every augmenter is a `DalTransform` over the full window and, by default, stacks the original windows on top of the augmented ones (native; no new dependency).

**Didactic goal:** synthesize windows by recombining past lags.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

windows = ts.ts_data(ts.datasets.tsd()["y"], 10)
print("original windows:", windows.shape)
```

    original windows: (32, 10)


Visualize the series used in this example:


```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```





![png](fig/06-wormhole-augmentation/output_3_0.png)






![png](fig/06-wormhole-augmentation/output_3_1.png)




```python
out = ts.TsAugWormhole().fit(windows).transform(windows)
print('after augmentation:', out.shape)
ts.ts_head(out, 3)
```

    after augmentation: (63, 10)





             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073
    1  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472
    2  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472  0.381661



> **What to observe.** The output stacks the original and the augmented windows (roughly double the rows); it plugs into `TsIntegtune`'s augmentation slot.

**References:** Wen, Q. et al. (2021). *Time Series Data Augmentation for Deep Learning: A Survey.* IJCAI.
