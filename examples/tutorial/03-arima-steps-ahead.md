# ARIMA multi-step forecast

**Didactic goal:** Forecast several steps ahead directly from the fitted ARIMA.


```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

rser = ts.ts_data(ts.datasets.tsd()["y"], 1)
samp = ts.ts_sample(rser, test_size=5)
```


```python
model = ts.TsArima(p=2, d=0, q=0).fit(samp.train)
pred = model.predict(samp.test.iloc[[0]], steps_ahead=5)
model.evaluate(samp.test.to_numpy().ravel(), pred)["metrics"]
```




                mse   smape   R2
    0  6.630410e-10  0.0001  1.0



Plot the 5-step forecast against the observed series:


```python
import numpy as np
adjust = np.asarray(model.predict(samp.train)).ravel()          # in-sample fit
yvalues = np.concatenate([samp.train.to_numpy().ravel(), samp.test.to_numpy().ravel()])
ts.plot_ts_pred(y=yvalues, yadj=adjust, ypred=pred)
```





![png](fig/03-arima-steps-ahead/output_4_0.png)






![png](fig/03-arima-steps-ahead/output_4_1.png)



> **What to observe.** A single multi-step forecast from the last training point.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis*, Wiley.
