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

# EMD decomposition filters (native, dependency-free)

Empirical Mode Decomposition (EMD) splits a signal into Intrinsic Mode Functions
(IMFs), from fastest to slowest, plus a residual trend. Two denoising filters build
on it:

* **`TsFilEmd`** — reconstruct the series without the first (highest-frequency) IMF:
  `result = data - imf1`.
* **`TsFilRemd`** — score each IMF's roughness, pick how many high-frequency IMFs are
  noise from the cumulative-roughness curve (via `FitCurvatureMin`, capped by
  `max_imfs`), and remove them.

Unlike the R originals (which wrap the `hht` package) and the common Python route
(PyEMD), this is a **native numpy + scipy** implementation — **no new dependency**
(scipy is already core) and orders of magnitude faster than R `hht`. The engine is in
`pytspredit.emd` (`emd`, `eemd`).

**Reference:** Huang et al. (1998), *The empirical mode decomposition and the Hilbert
spectrum...*, Proc. Royal Society A; Wu & Huang (2009), *Ensemble EMD*, Adv. Adaptive
Data Analysis.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts
```

## The native engine

`emd` returns IMFs whose rows sum exactly back to the signal (the last row is the
residual trend).

```python
t = np.linspace(0, 1, 300)
signal = np.sin(2*np.pi*5*t) + 0.4*np.sin(2*np.pi*19*t) + 0.3*t
imfs = ts.emd.emd(signal)
print("number of components (IMFs + residual):", imfs.shape[0])
print("reconstruction max error:", float(np.max(np.abs(imfs.sum(0) - signal))))
```

## TsFilEmd: remove the noisiest IMF

We inject an outlier into the smooth `tsd` series and remove the first IMF.

```python
y = ts.datasets.tsd()["y"].to_numpy()
yo = y.copy(); yo[8] = 2 * yo[8]          # inject an outlier

filtered = ts.TsFilEmd().fit(yo).transform(yo)
# how far the point at index 8 sits from its neighbours (spike size)
spike = lambda s: abs(s[8] - 0.5*(s[7] + s[9]))
print("spike raw:      %.3f" % spike(yo))
print("spike filtered: %.3f" % spike(filtered))
```

The ensemble engine (`engine="eemd"`, the analogue of the R `CEEMD`) is available too
and is made deterministic with `random_state`:

```python
ts.TsFilEmd(engine="eemd", trials=50, random_state=1).fit(yo).transform(yo)[:5]
```

## TsFilRemd: remove several high-frequency IMFs

On a noisy signal, `TsFilRemd` adaptively removes the high-frequency IMFs and improves
the fit to the underlying clean signal.

```python
rng = np.random.default_rng(0)
clean = np.sin(2*np.pi*5*t) + 0.3*t
noisy = clean + 0.2*rng.standard_normal(len(t))
den = ts.TsFilRemd().fit(noisy).transform(noisy)

def r2(a, p):
    a, p = np.asarray(a), np.asarray(p)
    return 1 - np.sum((a-p)**2)/np.sum((a-np.mean(a))**2)

print("R2 noisy vs clean:    %.3f" % r2(clean, noisy))
print("R2 denoised vs clean: %.3f" % r2(clean, den))
```

> **What to observe.** The EMD filters attenuate the spike / noise while preserving the
> trend. Because EMD is an adaptive, data-driven (and, for `eemd`, stochastic) method,
> parity with R `hht` is at the *quality* level (does it denoise as well?), not
> value-for-value — see `develop/parity.md`.
