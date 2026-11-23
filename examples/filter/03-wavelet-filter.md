# Wavelet (MODWT) denoising filter (native, dependency-free)

`TsFilWavelet` denoises a series with the **Maximal Overlap Discrete Wavelet
Transform** (MODWT): it decomposes the signal, zeroes the finest-scale detail, and
reconstructs (`x - D1`). If a *list* of filters is given, `fit` selects the one that
best preserves the signal (highest R2).

The MODWT is a fixed, deterministic linear transform, so unlike EMD this is a
**native pure-numpy** implementation (`pytspredit.wavelet`) that matches the R
`wavelets` package **value-for-value (~5e-6)** — with **no new dependency** (not even
scipy), and more faithfully than wrapping PyWavelets' `swt` (a different convention).

Filter families (as in R): `haar`, `d4`, `la8`, `bl14`, `c6`.

**Reference:** Mallat (1989), *A theory for multiresolution signal decomposition*,
IEEE TPAMI; Percival & Walden (2000), *Wavelet Methods for Time Series Analysis*,
Cambridge University Press (MODWT).


```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts
from pytspredit import wavelet
```

## The native MODWT

The transform perfectly reconstructs the signal (sum of detail levels + smooth).


```python
t = np.linspace(0, 1, 256)
signal = np.sin(2*np.pi*7*t) + 0.3*t
details, smooth = wavelet.modwt(signal, filter="la8")
print("levels:", len(details))
print("reconstruction max error:", float(np.max(np.abs(wavelet.imodwt(details, smooth, "la8") - signal))))
```

    levels: 5
    reconstruction max error: 1.1259881915748338e-12


## Denoising with TsFilWavelet


```python
y = ts.datasets.tsd()["y"].to_numpy()
yo = y.copy(); yo[8] = 2 * yo[8]          # inject an outlier

filtered = ts.TsFilWavelet("haar").fit(yo).transform(yo)
spike = lambda s: abs(s[8] - 0.5*(s[7] + s[9]))
print("spike raw:      %.3f" % spike(yo))
print("spike filtered: %.3f" % spike(filtered))
```

    spike raw:      0.938
    spike filtered: 0.255


## Automatic filter selection

Pass a list; `fit` keeps the filter with the highest R2 against the input.


```python
sel = ts.TsFilWavelet(["haar", "d4", "la8", "bl14", "c6"])
sel.fit(y)
print("selected filter:", sel.filter)
```

    selected filter: d4


> **What to observe.** The MODWT is translation-invariant and deterministic, so the
> native engine reproduces R value-for-value (~5e-6). The simple "zero the finest
> detail" denoiser matches the R package; proper wavelet thresholding
> (Donoho-Johnstone) was evaluated and did not improve denoising here (it needs a
> careful translation-invariant formulation) — see `develop/parity.md`.
