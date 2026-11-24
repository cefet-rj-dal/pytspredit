# ELM regression (`TsElm`)

**About the technique**

* `TsElm` is a native Extreme Learning Machine (random hidden weights + closed-form output).
* A sliding-window regressor: preprocessing + lag map + backend, trained on the projected `(input, output)`.

**Didactic goal:** fit, forecast 5 steps ahead, and evaluate (MSE / sMAPE / R2).


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

w = ts.ts_data(ts.datasets.tsd()["y"], 6)
samp = ts.ts_sample(w, test_size=5)
tr, te = ts.ts_projection(samp.train), ts.ts_projection(samp.test)
```


```python
model = ts.TsElm(ts.TsNormGminmax(), input_size=5).fit(tr.input, tr.output)
pred = model.predict(te.input.iloc[[0]], steps_ahead=5)
model.evaluate(te.output.to_numpy().ravel(), pred)["metrics"]
```




            mse    smape        R2
    0  0.853836  1.59474 -6.374628



Plot the 5-step forecast against the observed series (solid = observed, dashed = forecast):


```python
y_all = ts.datasets.tsd()["y"].to_numpy()
ts.plot_ts_pred(y=y_all, yadj=y_all[:-5], ypred=pred)
```





![png](fig/06-elm-regression/output_4_0.png)






![png](fig/06-elm-regression/output_4_1.png)



> **What to observe.** `fit` learns on the training windows; `predict(steps_ahead=5)` rolls forward from the first test window; `evaluate` reports MSE/sMAPE/R2.

**References:** Huang, G.-B., Zhu, Q.-Y. & Siew, C.-K. (2006). Extreme Learning Machine. Neurocomputing, 70(1-3), 489-501.
