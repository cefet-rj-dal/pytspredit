# Adaptive normalization -- soft divide (`TsNormAn`, `operation="softdivide"`)

**About the technique**

* The *soft divide* operator of `TsNormAn` divides by a stabilized average (a small
  epsilon / soft floor prevents blow-ups), combining the scale-invariance of division
  with robustness near zero.

**Didactic goal:** keep the division behavior while taming the explosions that a raw
divide causes when the local average is tiny.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

y = ts.datasets.tsd()["y"]
w = ts.ts_data(y, 5)
```

Visualize the series used in this example:


```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```





![png](fig/07-adaptive-softdivide-normalization/output_3_0.png)






![png](fig/07-adaptive-softdivide-normalization/output_3_1.png)




```python
norm = ts.TsNormAn(operation="softdivide").fit(w)
t = norm.transform(w)
ts.ts_head(t)
```




             t4        t3        t2        t1        t0
    0  0.177999  0.353874  0.518814  0.662564  0.776186
    1  0.273842  0.406324  0.521786  0.613048  0.674437
    2  0.346362  0.445082  0.523112  0.575599  0.599282
    3  0.405199  0.476056  0.523720  0.545226  0.539237
    4  0.457570  0.503306  0.523943  0.518196  0.486423
    5  0.508824  0.529696  0.523883  0.491748  0.435287




```python
ts.ts_head(norm.inverse_transform(t))
```




             t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471
    1  0.247404  0.479426  0.681639  0.841471  0.948985
    2  0.479426  0.681639  0.841471  0.948985  0.997495
    3  0.681639  0.841471  0.948985  0.997495  0.983986
    4  0.841471  0.948985  0.997495  0.983986  0.909297
    5  0.948985  0.997495  0.983986  0.909297  0.778073



> **What to observe.** Values are divided by a stabilized average, so the transform stays
> finite even where a plain divide would explode.

**References:** Ogasawara et al. (2010), *Adaptive Normalization*, IJCNN.
