# Train/Test Split and Projection

Once the series is in sliding windows, two steps prepare it for modeling:

* **`ts_sample`** splits the windows into train and test *temporally* — the test set
  is taken from the **end** of the series, never shuffled, so evaluation respects the
  arrow of time.
* **`ts_projection`** splits each window into **input** features (`t{sw-1} ... t1`)
  and the **output** target (`t0`).


```python
import pytspredit as ts

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=10)
print("total windows:", windows.shape[0])
```

    total windows: 32



```python
# hold out the last 5 windows as the test set
samp = ts.ts_sample(windows, test_size=5)
print("train:", samp.train.shape, " test:", samp.test.shape)
ts.ts_head(samp.test)
```

    train: (27, 10)  test: (5, 10)





             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.450044  0.656987  0.823081  0.938000  0.994599  0.989358  0.922604  0.798487  0.624724  0.412118
    1  0.656987  0.823081  0.938000  0.994599  0.989358  0.922604  0.798487  0.624724  0.412118  0.173889
    2  0.823081  0.938000  0.994599  0.989358  0.922604  0.798487  0.624724  0.412118  0.173889 -0.075151
    3  0.938000  0.994599  0.989358  0.922604  0.798487  0.624724  0.412118  0.173889 -0.075151 -0.319519
    4  0.994599  0.989358  0.922604  0.798487  0.624724  0.412118  0.173889 -0.075151 -0.319519 -0.544021




```python
# project train/test windows into (input X, output y)
io_train = ts.ts_projection(samp.train)
io_test = ts.ts_projection(samp.test)

print("input columns:", list(io_train.input.columns))
print("output column :", io_train.output.name)
io_train.input.head(3)
```

    input columns: ['t9', 't8', 't7', 't6', 't5', 't4', 't3', 't2', 't1']
    output column : t0





             t9        t8        t7        t6        t5        t4        t3        t2        t1
    0  0.000000  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297
    1  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073
    2  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472




```python
# the output is the most recent value of each window (t0)
io_train.output.head(3)
```




    0    0.778073
    1    0.598472
    2    0.381661
    Name: t0, dtype: float64



With `io_train.input` / `io_train.output` and `io_test.input` / `io_test.output`,
any `pytspredit` regressor can be fitted and evaluated on the held-out horizon.
