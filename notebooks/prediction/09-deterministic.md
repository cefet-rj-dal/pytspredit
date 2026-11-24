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

# Deterministic baselines (`TsPersist`, `TsPeriodic`)

**About the technique**

* `TsPersist` forecasts by repeating the most recent value; `TsPeriodic` repeats a learned
  cycle. Both are native, rule-based baselines (no training beyond storing the pattern).

**Didactic goal:** establish honest naive baselines that any real model must beat.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```

```python
# persistence: repeat the last observed value
ts.TsPersist().fit([10, 11, 11, 12]).predict(steps_ahead=3)
```

```python
# periodic: learn a 7-step cycle and repeat it
ts.TsPeriodic(7).fit([4, 5, 6, 7, 1, 2, 3]).predict(steps_ahead=5)
```

> **What to observe.** These are the reference baselines; `TsDeterministic` is the general
> rule-based class they specialize.

**References:** Hyndman & Athanasopoulos (2021), *Forecasting: Principles and Practice*,
3rd ed., OTexts (benchmark methods).
