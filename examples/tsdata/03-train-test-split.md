# Train/test split of `ts_data`

**About the technique**

* `ts_sample()` splits the windowed `ts_data` into training and test sets **in time
  order** (the last `test_size` windows are held out), as required for forecasting.

**Didactic goal:** produce an honest, temporally ordered holdout for evaluation.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```

Visualize the series used in this example:


```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```





![png](fig/03-train-test-split/output_3_0.png)






![png](fig/03-train-test-split/output_3_1.png)




```python
y = ts.datasets.tsd()["y"]
w = ts.ts_data(y, 5)
samp = ts.ts_sample(w, test_size=5)
print("train rows:", len(samp.train), "| test rows:", len(samp.test))
```

    train rows: 32 | test rows: 5


The training set keeps the earlier windows.


```python
ts.ts_head(samp.train)
```




             t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471
    1  0.247404  0.479426  0.681639  0.841471  0.948985
    2  0.479426  0.681639  0.841471  0.948985  0.997495
    3  0.681639  0.841471  0.948985  0.997495  0.983986
    4  0.841471  0.948985  0.997495  0.983986  0.909297
    5  0.948985  0.997495  0.983986  0.909297  0.778073



The test set is the final `test_size` windows, untouched during training.


```python
ts.ts_head(samp.test)
```




             t4        t3        t2        t1        t0
    0  0.989358  0.922604  0.798487  0.624724  0.412118
    1  0.922604  0.798487  0.624724  0.412118  0.173889
    2  0.798487  0.624724  0.412118  0.173889 -0.075151
    3  0.624724  0.412118  0.173889 -0.075151 -0.319519
    4  0.412118  0.173889 -0.075151 -0.319519 -0.544021



> **What to observe.** The split is chronological (no shuffling): the test windows come
> strictly after the training windows, so evaluation never peeks at the future.

**References:** Tashman (2000), *Out-of-sample tests of forecasting accuracy*,
International Journal of Forecasting, 16(4), 437-450.
