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

# Filter -- Hodrick-Prescott (`TsFilHp`)

**About the technique**

* `TsFilHp` extracts a smooth trend via the Hodrick-Prescott penalty.
* A filter is a `DalTransform` taking a 1-D series and returning a filtered series of the same length.

**Didactic goal:** see how the filter cleans the raw series before modeling.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

_rng = np.random.default_rng(0)
_base = ts.datasets.tsd()["y"].to_numpy()
y = _base + _rng.normal(0, _base.std(ddof=1) / 10, len(_base))  # add noise
_spike = _rng.normal(0, _base.std(ddof=1))
y[[9, 19, 29]] += _spike                                        # inject a few spikes
```

Visualize the raw series:

```python
ts.plot_ts(y=y)
```

```python
out = ts.TsFilHp().fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

Compare the original series with the filtered one:

```python
ts.plot_ts_pred(y=y, yadj=out)
```

> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Hodrick, R. J. & Prescott, E. C. (1997). Postwar U.S. business cycles. Journal of Money, Credit and Banking, 29(1), 1-16.
