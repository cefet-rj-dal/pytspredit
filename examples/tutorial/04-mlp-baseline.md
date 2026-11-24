# MLP baseline

**Didactic goal:** Train a plain MLP forecaster as a baseline.


```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

w = ts.ts_data(ts.datasets.tsd()["y"], 6)
samp = ts.ts_sample(w, test_size=5)
tr, te = ts.ts_projection(samp.train), ts.ts_projection(samp.test)
out = te.output.to_numpy().ravel()
```


```python
model = ts.TsMlp(ts.TsNormGminmax(), input_size=5).fit(tr.input, tr.output)
model.evaluate(out, model.predict(te.input.iloc[[0]], steps_ahead=5))["metrics"]
```




          mse    smape        R2
    0  0.0007  0.14712  0.993955



> **What to observe.** A normalized MLP is a strong non-linear baseline.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
