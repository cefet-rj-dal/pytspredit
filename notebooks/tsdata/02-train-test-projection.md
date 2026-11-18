---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

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

```python
# hold out the last 5 windows as the test set
samp = ts.ts_sample(windows, test_size=5)
print("train:", samp.train.shape, " test:", samp.test.shape)
ts.ts_head(samp.test)
```

```python
# project train/test windows into (input X, output y)
io_train = ts.ts_projection(samp.train)
io_test = ts.ts_projection(samp.test)

print("input columns:", list(io_train.input.columns))
print("output column :", io_train.output.name)
io_train.input.head(3)
```

```python
# the output is the most recent value of each window (t0)
io_train.output.head(3)
```

With `io_train.input` / `io_train.output` and `io_test.input` / `io_test.output`,
any `pytspredit` regressor can be fitted and evaluated on the held-out horizon.
