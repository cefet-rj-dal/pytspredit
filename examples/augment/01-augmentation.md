# Data Augmentation

Augmentation enlarges the training set with plausible variants of each sliding
window, which can improve robustness and reduce overfitting. Every augmenter is a
`DalTransform` that operates on the full window (features + target `t0`) and, by
default, stacks the **original** windows on top of the **augmented** ones.

All are native (no new dependencies). `flip`, `stretch`, `shrink` and `wormhole` are
deterministic; `jitter`, `awareness` and `awaresmooth` add random noise (a
`random_state` makes them reproducible).

**Reference:** Wen, Q. et al. (2021). *Time Series Data Augmentation for Deep
Learning: A Survey.* IJCAI.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

windows = ts.ts_data(ts.datasets.tsd()["y"], 10)
print("original windows:", windows.shape)
```

    original windows: (32, 10)


## Deterministic augmenters


```python
# flip: mirror each window around its mean
flipped = ts.TsAugFlip().fit(windows).transform(windows)
print("flip:", flipped.shape, "(original + augmented)")
ts.ts_head(flipped, 3)
```

    flip: (64, 10) (original + augmented)





             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073
    1  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472
    2  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472  0.381661




```python
# stretch / shrink: scale the within-window deviations (target kept unchanged)
print("stretch:", ts.TsAugStretch(scale_factor=1.2).fit(windows).transform(windows).shape)
print("shrink :", ts.TsAugShrink(scale_factor=0.8).fit(windows).transform(windows).shape)
# wormhole: recombine lag content from older windows
print("wormhole:", ts.TsAugWormhole().fit(windows).transform(windows).shape)
```

    stretch: (64, 10)
    shrink : (64, 10)
    wormhole: (63, 10)


## Stochastic augmenters


```python
# jitter: add low-amplitude Gaussian noise (scale estimated from the windows)
jit = ts.TsAugJitter(random_state=0).fit(windows)
print("jitter noise sd:", round(jit.sd, 4))
jit.transform(windows).shape
```

    jitter noise sd: 0.4396





    (64, 10)




```python
# awareness / awaresmooth: oversample recent windows (+ progressive smoothing)
print("awareness  :", ts.TsAugAwareness(factor=1, random_state=0).transform(windows).shape)
print("awaresmooth:", ts.TsAugAwaresmooth(factor=1, random_state=0).transform(windows).shape)
```

    awareness  : (64, 10)
    awaresmooth: (64, 10)


## Inside integrated tuning

Augmenters plug into `TsIntegtune`: it co-optimizes the input size, the
preprocessing, the augmentation and the model hyperparameters.


```python
samp = ts.ts_sample(windows, test_size=5)
io_train = ts.ts_projection(samp.train)

itune = ts.TsIntegtune(
    input_size=[3, 4, 5],
    base_model=ts.TsElm(),
    folds=5,
    ranges={"nhid": [1, 2, 3, 4, 5], "actfun": ["purelin"]},
    preprocess=[ts.TsNormGminmax()],
    augment=[ts.TsAugNone(), ts.TsAugStretch(), ts.TsAugFlip()],
    seed=0,
)
best = itune.fit(io_train.input, io_train.output)
itune.best_params
```




    {'input_size': 4,
     'preprocess': 0,
     'augment': 0,
     'nhid': 3,
     'actfun': 'purelin'}


