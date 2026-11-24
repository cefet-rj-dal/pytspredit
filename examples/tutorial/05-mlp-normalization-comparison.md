# MLP: normalization comparison

**Didactic goal:** Compare how different normalizations affect the same MLP.


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
for nm, norm in [("gminmax", ts.TsNormGminmax()), ("swminmax", ts.TsNormSwminmax()), ("diff", ts.TsNormDiff())]:
    m = ts.TsMlp(norm, input_size=4).fit(tr.input, tr.output)
    r2 = m.evaluate(out, m.predict(te.input.iloc[[0]], steps_ahead=5))["R2"]
    print(f"{nm:9} R2={r2:.4f}")
```

    gminmax   R2=0.9674
    swminmax  R2=0.9025
    diff      R2=0.9767


> **What to observe.** The preprocessing choice materially changes accuracy -- motivating integrated tuning.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
