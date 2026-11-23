# Hyperparameter Tuning

`pytspredit` tunes time series models by cross-validated grid search. There are two
tuners:

* **`TsTune`** searches candidate `input_size` values and model hyperparameter
  ranges, minimizing cross-validated MSE, and returns the best model refitted on all
  the training data.
* **`TsIntegtune`** (*integrated tuning*) additionally searches over a list of
  preprocessors (and augmenters) — co-optimizing preprocessing and modeling, the
  idea at the heart of TSPredIT.

**Reference:** Salles, R. et al. (2023). *TSPredIT: Integrated Tuning of Data
Preprocessing and Time Series Prediction Models.* LNCS.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

w = ts.ts_data(ts.datasets.tsd()["y"], 10)
samp = ts.ts_sample(w, test_size=5)
io_train = ts.ts_projection(samp.train)
io_test = ts.ts_projection(samp.test)
output = io_test.output.to_numpy()
```

## TsTune — input size + model hyperparameters


```python
tune = ts.TsTune(
    input_size=[3, 4, 5],
    base_model=ts.TsElm(ts.TsNormGminmax()),
    folds=5,
    ranges={"nhid": [1, 2, 3, 4, 5], "actfun": ["purelin"]},
    seed=0,
)
best = tune.fit(io_train.input, io_train.output)
print("selected:", tune.best_params)

pred = best.predict(io_test.input.iloc[[0]], steps_ahead=5)
best.evaluate(output, pred)["metrics"]
```

    selected: {'input_size': 4, 'nhid': 3, 'actfun': 'purelin'}





                mse         smape   R2
    0  4.748564e-28  7.448503e-14  1.0



## TsIntegtune — also choose the preprocessing


```python
itune = ts.TsIntegtune(
    input_size=[3, 4, 5],
    base_model=ts.TsElm(),
    folds=5,
    ranges={"nhid": [1, 2, 3, 4, 5], "actfun": ["purelin"]},
    preprocess=[ts.TsNormGminmax(), ts.TsNormSwminmax()],
    seed=0,
)
best = itune.fit(io_train.input, io_train.output)
print("selected:", itune.best_params)

pred = best.predict(io_test.input.iloc[[0]], steps_ahead=5)
best.evaluate(output, pred)["metrics"]
```

    selected: {'input_size': 5, 'preprocess': 1, 'augment': 0, 'nhid': 5, 'actfun': 'purelin'}





                mse         smape   R2
    0  1.976158e-30  8.625306e-15  1.0



The `preprocess` index in `best_params` points into the `preprocess` list; the
`augment` index points into the augmenters (only the identity `TsAugNone` ships in
this release — the full augmentation family arrives later).
