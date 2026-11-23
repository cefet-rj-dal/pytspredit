# Machine-Learning Regressors

Five sliding-window regressors, each a thin adapter over a scikit-learn backend (ELM
is implemented directly). All share the `TsRegSW` contract: a preprocessor
(normalization), a lag map, and a backend — `fit(X, y)` / `predict` (one-step or
`steps_ahead`) / `evaluate`.

**References:** Cover & Hart (1967, k-NN); Breiman (2001, Random Forest);
Cortes & Vapnik (1995, SVR); Rumelhart, Hinton & Williams (1986, MLP);
Huang, Zhu & Siew (2006, ELM).


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

w = ts.ts_data(ts.datasets.tsd()["y"], 10)
samp = ts.ts_sample(w, test_size=5)
io_train = ts.ts_projection(samp.train)
io_test = ts.ts_projection(samp.test)
output = io_test.output.to_numpy()


def evaluate_5step(model):
    model.fit(io_train.input, io_train.output)
    pred = model.predict(io_test.input.iloc[[0]], steps_ahead=5)
    return model.evaluate(output, pred)["metrics"]
```

## k-Nearest Neighbors (`TsKnn`)

Predicts the average target of the `k` most similar windows. Deterministic — matches
the R `FNN::knn.reg` backend exactly.


```python
evaluate_5step(ts.TsKnn(ts.TsNormGminmax(), input_size=4, input_map=ts.TsLagmap("pacf"), k=3))
```




            mse     smape        R2
    0  0.037273  0.889007  0.678074



## Extreme Learning Machine (`TsElm`)

Single hidden layer with random weights and closed-form output weights. With
`actfun="purelin"` and `nhid >= n_lags` it reduces to least squares.


```python
evaluate_5step(ts.TsElm(ts.TsNormGminmax(), 4, ts.TsLagmap("acf"), nhid=5, actfun="purelin"))
```




                mse         smape   R2
    0  2.936693e-27  1.760780e-13  1.0



## Random Forest (`TsRf`)

Ensemble of regression trees; captures nonlinearities without heavy tuning.


```python
evaluate_5step(ts.TsRf(ts.TsNormGminmax(), input_size=9, ntree=100, nodesize=1))
```




            mse     smape        R2
    0  0.020558  0.688541  0.822437



## Support Vector Regression (`TsSvm`)

Epsilon-insensitive SVR with an RBF kernel; inputs and target are standardized to
mirror the R `e1071` backend's `scale=TRUE` default.


```python
evaluate_5step(ts.TsSvm(ts.TsNormGminmax(), 4, ts.TsLagmap("seasonal", seasonality=4)))
```




                mse     smape   R2
    0  1.385132e-08  0.000332  1.0



## Multilayer Perceptron (`TsMlp`)

Single hidden layer (logistic activation, L-BFGS), approximating the R `nnet` backend.


```python
evaluate_5step(ts.TsMlp(ts.TsNormGminmax(), 4, ts.TsLagmap("even"), size=4, decay=0))
```




            mse     smape        R2
    0  0.000309  0.078969  0.997332



All five expose the same interface, so swapping the backend — or the normalization, or
the lag map — is a one-line change.
