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

## Deterministic augmenters

```python
# flip: mirror each window around its mean
flipped = ts.TsAugFlip().fit(windows).transform(windows)
print("flip:", flipped.shape, "(original + augmented)")
ts.ts_head(flipped, 3)
```

```python
# stretch / shrink: scale the within-window deviations (target kept unchanged)
print("stretch:", ts.TsAugStretch(scale_factor=1.2).fit(windows).transform(windows).shape)
print("shrink :", ts.TsAugShrink(scale_factor=0.8).fit(windows).transform(windows).shape)
# wormhole: recombine lag content from older windows
print("wormhole:", ts.TsAugWormhole().fit(windows).transform(windows).shape)
```

## Stochastic augmenters

```python
# jitter: add low-amplitude Gaussian noise (scale estimated from the windows)
jit = ts.TsAugJitter(random_state=0).fit(windows)
print("jitter noise sd:", round(jit.sd, 4))
jit.transform(windows).shape
```

```python
# awareness / awaresmooth: oversample recent windows (+ progressive smoothing)
print("awareness  :", ts.TsAugAwareness(factor=1, random_state=0).transform(windows).shape)
print("awaresmooth:", ts.TsAugAwaresmooth(factor=1, random_state=0).transform(windows).shape)
```

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
