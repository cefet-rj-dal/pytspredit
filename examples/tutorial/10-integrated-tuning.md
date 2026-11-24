# Integrated tuning workflow

**Didactic goal:** Co-optimize input size, preprocessing and model together.


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
itune = ts.TsIntegtune(input_size=[3, 4, 5], base_model=ts.TsElm(), folds=5,
    ranges={"nhid": [1, 2, 3, 4, 5], "actfun": ["purelin"]},
    preprocess=[ts.TsNormGminmax(), ts.TsNormSwminmax()], seed=0)
best = itune.fit(tr.input, tr.output)
print("selected:", itune.best_params)
best.evaluate(out, best.predict(te.input.iloc[[0]], steps_ahead=5))["metrics"]
```

    selected: {'input_size': 4, 'preprocess': 1, 'augment': 0, 'nhid': 3, 'actfun': 'purelin'}





                mse         smape   R2
    0  2.088139e-29  1.309686e-14  1.0



> **What to observe.** Integrated tuning is the idea at the heart of TSPredIT.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
