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

    ma  : [  nan   nan 0.242 0.469 0.668 0.824]
    ema : [  nan   nan 0.345 0.562 0.744 0.88 ]


## Outlier-robust


```python
print("winsor:", np.round(ts.TsFilWinsor().fit(y).transform(y)[6:11], 3))  # extreme capped
print("smooth:", np.round(ts.TsFilSmooth().fit(y).transform(y)[6:11], 3))
```

    winsor: [0.995 0.984 0.995 0.778 0.598]
    smooth: [0.997 1.408 0.881 0.778 0.598]


## Trend / low-pass


```python
for name, f in [("recursive", ts.TsFilRecursive(0.05)), ("hp", ts.TsFilHp(100)),
                ("lowess", ts.TsFilLowess(0.2)), ("spline", ts.TsFilSpline(0.5)),
                ("fft", ts.TsFilFft()), ("gabor", ts.TsFilGabor(3, 0.1)),
                ("kalman", ts.TsFilKalman())]:
    out = np.asarray(f.fit(y).transform(y), float)
    print(f"{name:10s} -> smoothed length {len(out)}")
```

    recursive  -> smoothed length 41
    hp         -> smoothed length 41
    lowess     -> smoothed length 41
    spline     -> smoothed length 41
    fft        -> smoothed length 41
    gabor      -> smoothed length 41
    kalman     -> smoothed length 41


## Exponential smoothing and seasonal adjustment


```python
print("ses :", np.round(ts.TsFilSes().fit(y).transform(y)[:5], 3))    # leading NaN warm-up
print("qes :", np.round(ts.TsFilQes().fit(y).transform(y)[:5], 3))
print("seas_adj:", np.round(ts.TsFilSeasAdj(frequency=12).fit(y).transform(y)[:5], 3))
```

    ses : [   nan -0.     0.247  0.479  0.682]
    qes : [  nan   nan 0.472 0.709 0.913]
    seas_adj: [1.09  1.024 0.959 0.894 0.828]


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

    all same length: True

