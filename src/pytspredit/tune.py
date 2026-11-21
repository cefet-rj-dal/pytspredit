"""Hyperparameter tuning for time series models.

Python counterpart of ``ts_tune.R`` and ``ts_integtune.R``, built on
``pydaltoolbox.DalTune``.

* :class:`TsTune` grid-searches candidate ``input_size`` values and model
  hyperparameter ranges with k-fold cross-validation, minimizing MSE, and returns
  the best model refitted on all the data.
* :class:`TsIntegtune` additionally searches over a list of preprocessors and
  augmenters (*integrated tuning*). In v1 only the identity augmenter
  (:class:`~pytspredit.TsAugNone`) is available; the full augmentation family is
  deferred to a later release.

Parity note: candidate models match R given a configuration (see M3), and the
selection rule (minimum mean CV MSE, ties -> first) mirrors ``select_hyper``. The
cross-validation *fold partition* is drawn with a seeded NumPy RNG, which differs
from R's ``sample_random``; the chosen configuration may therefore differ on
borderline cases. Fix ``seed`` for reproducibility.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd
from pydaltoolbox import DalTune

from .augment import TsAugNone
from .norm import TsNormGminmax
from .tsdata import adjust_ts_data, ts_projection

__all__ = ["TsIntegtune", "TsTune"]


def _as_list(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    if isinstance(v, range):
        return list(v)
    return [v]


def _cv_folds(n, k, seed):
    """Return a list of (train_idx, test_idx) with a seeded shuffle (k folds)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    parts = np.array_split(idx, k)
    folds = []
    for j in range(len(parts)):
        test = parts[j]
        train = np.concatenate([parts[t] for t in range(len(parts)) if t != j])
        folds.append((np.sort(train), np.sort(test)))
    return folds


def _mse(values, prediction):
    values = np.asarray(values, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()
    return float(np.mean((values - prediction) ** 2))


class TsTune(DalTune):
    """Cross-validated grid search over ``input_size`` and model ranges (``ts_tune``).

    References:
        - Kohavi R (1995). A study of cross-validation and bootstrap for accuracy
          estimation and model selection. IJCAI.
        - Salles R, Pacitti E, Bezerra E, Marques C, Pacheco C, Oliveira C, Porto
          F, Ogasawara E (2023). TSPredIT: Integrated Tuning of Data Preprocessing
          and Time Series Prediction Models. Lecture Notes in Computer Science.

    Porting:
        Mirrors ``ts_tune.R``. Fidelity: approximate.
        - CV grid search minimizing MSE (min-mean-error ties -> first). Fold
          partition uses a seeded numpy RNG (not R's ``sample``), so the chosen
          configuration can differ when ties are broken by fold assignment.
    """

    def __init__(self, input_size, base_model, folds: int = 10, ranges=None, seed: int = 0) -> None:
        super().__init__(base_model, folds, ranges)
        self.input_size = input_size
        self.seed = seed
        self.hyperparameters: pd.DataFrame | None = None
        self.best_params: dict | None = None

    # -- grid -----------------------------------------------------------------
    def _grid(self) -> list[dict]:
        ranges = {"input_size": _as_list(self.input_size)}
        ranges.update({k: _as_list(v) for k, v in self.ranges.items()})
        keys = list(ranges)
        return [dict(zip(keys, combo)) for combo in itertools.product(*(ranges[k] for k in keys))]

    # -- candidate model ------------------------------------------------------
    def _build(self, params: dict, x, y):
        model = copy.deepcopy(self.base_model)
        model.input_size = params["input_size"]
        model.set_params({k: v for k, v in params.items() if k != "input_size"})
        model.fit(x, y)
        return model

    def _iloc(self, data, idx):
        return data.iloc[idx] if isinstance(data, (pd.DataFrame, pd.Series)) else np.asarray(data)[idx]

    # -- fit ------------------------------------------------------------------
    def fit(self, x, y, **kwargs) -> object:
        x = x if isinstance(x, pd.DataFrame) else adjust_ts_data(x)
        y = pd.Series(np.asarray(y, dtype=float).ravel())
        grid = self._grid()
        records = []
        best_idx = 0
        if len(grid) > 1:
            for train, test in _cv_folds(len(x), self.folds, self.seed):
                for key, params in enumerate(grid):
                    try:
                        model = self._build(params, self._iloc(x, train), self._iloc(y, train))
                        pred = model.predict(self._iloc(x, test))
                        err = _mse(self._iloc(y, test), pred)
                    except Exception:  # noqa: BLE001 - a bad config should not abort the search
                        err = np.nan
                    records.append({**params, "key": key, "error": err})
            self.hyperparameters = pd.DataFrame(records)
            best_idx = self._select_hyper(self.hyperparameters)

        self.best_params = grid[best_idx]
        best = self._build(self.best_params, x, y)
        if len(grid) == 1:
            self.hyperparameters = pd.DataFrame(
                [{**self.best_params, "key": 0, "error": _mse(y, best.predict(x))}])
        return best

    @staticmethod
    def _select_hyper(hyper: pd.DataFrame) -> int:
        summary = hyper.groupby("key")["error"].mean()
        min_error = summary.min()
        winners = summary.index[summary == min_error]
        return int(min(winners))


class TsIntegtune(TsTune):
    """Integrated tuning over input size, preprocessing, augmentation and model
    hyperparameters (``ts_integtune``).

    References:
        - Salles R, Pacitti E, Bezerra E, Marques C, Pacheco C, Oliveira C, Porto
          F, Ogasawara E (2023). TSPredIT: Integrated Tuning of Data Preprocessing
          and Time Series Prediction Models. Lecture Notes in Computer Science.

    Porting:
        Mirrors ``ts_integtune.R``. Fidelity: approximate.
        - Same CV strategy as :class:`TsTune`, jointly searching the preprocessing
          and augmentation slots (see the fold-RNG note on :class:`TsTune`).
    """

    def __init__(self, input_size, base_model, folds: int = 10, ranges=None,
                 preprocess=None, augment=None, seed: int = 0) -> None:
        super().__init__(input_size, base_model, folds, ranges, seed)
        self.preprocess = preprocess if preprocess is not None else [TsNormGminmax()]
        self.augment = augment if augment is not None else [TsAugNone()]

    def _grid(self) -> list[dict]:
        ranges = {
            "input_size": _as_list(self.input_size),
            "preprocess": list(range(len(self.preprocess))),
            "augment": list(range(len(self.augment))),
        }
        ranges.update({k: _as_list(v) for k, v in self.ranges.items()})
        keys = list(ranges)
        return [dict(zip(keys, combo)) for combo in itertools.product(*(ranges[k] for k in keys))]

    def _build(self, params: dict, x, y):
        model = copy.deepcopy(self.base_model)
        model.input_size = params["input_size"]
        skip = {"input_size", "preprocess", "augment"}
        model.set_params({k: v for k, v in params.items() if k not in skip})
        model.preprocess = copy.deepcopy(self.preprocess[params["preprocess"]])
        augment = copy.deepcopy(self.augment[params["augment"]])
        # augment the training windows before fitting (identity in v1)
        data = adjust_ts_data(pd.concat([pd.DataFrame(np.asarray(x)),
                                         pd.Series(np.asarray(y).ravel())], axis=1))
        data = adjust_ts_data(augment.fit(data).transform(data))
        io = ts_projection(data)
        model.fit(io.input, io.output)
        return model
