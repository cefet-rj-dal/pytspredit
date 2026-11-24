# Data augmentation -- No augmentation (`TsAugNone`)

**About the technique**

* `TsAugNone` is the identity augmenter: it returns the windows unchanged. Baseline / control for the augmentation slot.
* Every augmenter is a `DalTransform` over the full window and, by default, stacks the original windows on top of the augmented ones (native; no new dependency).

**Didactic goal:** have a no-op control in the augmentation slot.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

windows = ts.ts_data(ts.datasets.tsd()["y"], 10)
print("original windows:", windows.shape)
```

    original windows: (32, 10)



```python
out = ts.TsAugNone().fit(windows).transform(windows)
print('after augmentation:', out.shape)
ts.ts_head(out, 3)
```

    after augmentation: (32, 10)





             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073
    1  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472
    2  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472  0.381661



> **What to observe.** The output stacks the original and the augmented windows (roughly double the rows); it plugs into `TsIntegtune`'s augmentation slot.

**References:** Wen, Q. et al. (2021). *Time Series Data Augmentation for Deep Learning: A Survey.* IJCAI.
