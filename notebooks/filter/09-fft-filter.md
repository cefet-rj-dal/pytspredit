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

# Filter -- FFT low-pass (`TsFilFft`)

**About the technique**

* `TsFilFft` keeps the low-frequency Fourier components (spectral denoising).
* A filter is a `DalTransform` taking a 1-D series and returning a filtered series of the same length.

**Didactic goal:** see how the filter cleans the raw series before modeling.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

x = np.linspace(0, 4 * np.pi, 128)
y = np.sin(x) + 0.25 * np.sin(12 * x)   # low-freq signal + high-freq component
```

Visualize the raw series:

```python
ts.plot_ts(y=y)
```

```python
out = ts.TsFilFft().fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

Compare the original series with the filtered one:

```python
ts.plot_ts_pred(y=y, yadj=out)
```

> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Oppenheim, A. V. & Schafer, R. W. (2010). Discrete-Time Signal Processing, 3rd ed., Pearson.
