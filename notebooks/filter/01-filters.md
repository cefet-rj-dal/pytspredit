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

# Filters (smoothing / denoising)

Filters clean a raw series before modeling. Each is a `DalTransform` that takes a
1-D series and returns a filtered series of the same length.

Most are native or reuse existing dependencies; `scipy` is the only new (lightweight)
core dependency, used by the smoothing spline.

**References:** Holt (1957, EMA); Hodrick & Prescott (1997, HP); Cleveland (1979,
LOWESS); Gabor (1946); Cooley & Tukey (1965, FFT); Kalman (1960).

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

y = ts.datasets.tsd()["y"].to_numpy().copy()
y[8] = 2 * y[8]          # inject an outlier
```

## Local averaging

```python
print("ma  :", np.round(ts.TsFilMa(3).fit(y).transform(y)[:6], 3))      # leading NaNs during warm-up
print("ema :", np.round(ts.TsFilEma(3).fit(y).transform(y)[:6], 3))
```

## Outlier-robust

```python
print("winsor:", np.round(ts.TsFilWinsor().fit(y).transform(y)[6:11], 3))  # extreme capped
print("smooth:", np.round(ts.TsFilSmooth().fit(y).transform(y)[6:11], 3))
```

## Trend / low-pass

```python
for name, f in [("recursive", ts.TsFilRecursive(0.05)), ("hp", ts.TsFilHp(100)),
                ("lowess", ts.TsFilLowess(0.2)), ("spline", ts.TsFilSpline(0.5)),
                ("fft", ts.TsFilFft()), ("gabor", ts.TsFilGabor(3, 0.1)),
                ("kalman", ts.TsFilKalman())]:
    out = np.asarray(f.fit(y).transform(y), float)
    print(f"{name:10s} -> smoothed length {len(out)}")
```

## Exponential smoothing and seasonal adjustment

```python
print("ses :", np.round(ts.TsFilSes().fit(y).transform(y)[:5], 3))    # leading NaN warm-up
print("qes :", np.round(ts.TsFilQes().fit(y).transform(y)[:5], 3))
print("seas_adj:", np.round(ts.TsFilSeasAdj(frequency=12).fit(y).transform(y)[:5], 3))
```

Nine filters reproduce R exactly (`none`, `ma`, `ema`, `winsor`, `recursive`, `fft`,
`smooth`, `hp`, `gabor`); `lowess` and `kalman` match too. `ses`, `qes`, `seas_adj` and
`spline` wrap different statistical backends than R (`HoltWinters`, `stl`,
`smooth.spline`), so their output is close but not bit-identical.
```python
# every filter preserves the series length
print("all same length:",
      all(len(np.asarray(f.fit(y).transform(y), float)) == len(y)
          for f in [ts.TsFilNone(), ts.TsFilMa(3), ts.TsFilKalman(), ts.TsFilSpline(0.5)]))
```
