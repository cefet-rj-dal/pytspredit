"""Time series augmentation.

Python counterpart of ``ts_aug_none.R`` and the seven augmenters
(``ts_aug_jitter``, ``ts_aug_flip``, ``ts_aug_stretch``, ``ts_aug_shrink``,
``ts_aug_wormhole``, ``ts_aug_awareness``, ``ts_aug_awaresmooth``).

Each augmenter is a :class:`pydaltoolbox.DalTransform` that operates on the full
sliding window (features + target ``t0``). With ``preserve_data=True`` (the R
default) the result stacks the original windows on top of the augmented ones.

Parity with R:

* ``flip``, ``stretch``, ``shrink``, ``wormhole`` are deterministic -> exact.
* ``jitter``, ``awareness``, ``awaresmooth`` draw random noise/row samples, so the
  augmented *values* depend on the RNG stream and cannot match R bit-for-bit
  (expected tolerance); the noise scale, the recency weighting and the stacking
  structure are faithful. A ``random_state`` makes them reproducible (R uses the
  global RNG); set it to ``None`` for fresh randomness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydaltoolbox import DalTransform

from .tsdata import ts_data

__all__ = [
    "TsAugAwareness",
    "TsAugAwaresmooth",
    "TsAugFlip",
    "TsAugJitter",
    "TsAugNone",
    "TsAugShrink",
    "TsAugStretch",
    "TsAugWormhole",
]


def _values(data) -> np.ndarray:
    arr = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=float)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def _frame(arr: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(arr, dtype=float)
    cols = [f"t{k}" for k in range(arr.shape[1] - 1, -1, -1)]
    return pd.DataFrame(arr, columns=cols)


def _rowmean(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1)


def _sd_within(x: np.ndarray) -> float:
    """Std of within-window deviations (R ``sd`` over the demeaned matrix, ddof=1)."""
    dev = x - _rowmean(x)[:, None]
    return float(np.std(dev, ddof=1))


class TsAugNone(DalTransform):
    """Identity augmentation (``ts_aug_none``): returns the windows unchanged.

    Porting:
        Mirrors ``ts_aug_none.R``. Fidelity: exact.
    """

    def fit(self, data=None, **kwargs) -> TsAugNone:
        return self

    def transform(self, data, **kwargs):
        return data


class _AugStack(DalTransform):
    """Base for augmenters that stack original + augmented windows."""

    preserve_data = True

    def fit(self, data=None, **kwargs) -> _AugStack:
        return self

    def _augment(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def transform(self, data, **kwargs) -> pd.DataFrame:
        x = _values(data)
        aug = self._augment(x)
        out = np.vstack([x, aug]) if self.preserve_data else aug
        return _frame(out)


class TsAugFlip(_AugStack):
    """Mirror each window around its mean (``ts_aug_flip``).

    References:
        - Wen Q, et al. (2021). Time Series Data Augmentation for Deep Learning:
          A Survey. IJCAI.

    Porting:
        Mirrors ``ts_aug_flip.R``. Fidelity: exact (deterministic; ~5e-15).
    """

    def _augment(self, x: np.ndarray) -> np.ndarray:
        return 2 * _rowmean(x)[:, None] - x


class TsAugStretch(_AugStack):
    """Amplify within-window deviations by ``scale_factor`` (``ts_aug_stretch``).

    References:
        - Wen Q, et al. (2021). Time Series Data Augmentation for Deep Learning:
          A Survey. IJCAI.

    Porting:
        Mirrors ``ts_aug_stretch.R``. Fidelity: exact (deterministic).
    """

    def __init__(self, scale_factor: float = 1.2) -> None:
        self.scale_factor = scale_factor

    def _augment(self, x: np.ndarray) -> np.ndarray:
        dev = (x - _rowmean(x)[:, None]) * self.scale_factor
        dev[:, -1] = 0.0  # keep the target column unchanged
        return x + dev


class TsAugShrink(TsAugStretch):
    """Shrink within-window deviations by ``scale_factor`` (``ts_aug_shrink``).

    References:
        - Wen Q, et al. (2021). Time Series Data Augmentation for Deep Learning:
          A Survey. IJCAI.

    Porting:
        Mirrors ``ts_aug_shrink.R``. Fidelity: exact (deterministic).
    """

    def __init__(self, scale_factor: float = 0.8) -> None:
        super().__init__(scale_factor)


class TsAugJitter(_AugStack):
    """Add low-amplitude Gaussian noise (``ts_aug_jitter``); RNG-dependent.

    References:
        - Um JT, et al. (2017). Data augmentation of wearable sensor data for
          Parkinson's disease monitoring using convolutional neural networks. ICMI.
        - Wen Q, et al. (2021). Time Series Data Augmentation for Deep Learning:
          A Survey. IJCAI.

    Porting:
        Mirrors ``ts_aug_jitter.R``. Fidelity: approximate.
        - Noise scale faithful (sd 0.4395701 matches R); values differ because the
          RNG differs (pass ``random_state`` for reproducibility).
    """

    def __init__(self, random_state: int | None = 0) -> None:
        self.random_state = random_state
        self.sd: float | None = None

    def fit(self, data, **kwargs) -> TsAugJitter:
        self.sd = _sd_within(_values(data))
        return self

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.sd is None:
            self.sd = _sd_within(x)
        rng = np.random.default_rng(self.random_state)
        noise = rng.normal(0.0, self.sd, size=x.shape)
        noise[:, -1] = 0.0
        return x + noise


class TsAugWormhole(_AugStack):
    """Replace lag terms with older values via combinatorial windows (``ts_aug_wormhole``).

    References:
        - Wen Q, et al. (2021). Time Series Data Augmentation for Deep Learning:
          A Survey. IJCAI.

    Porting:
        Mirrors ``ts_aug_wormhole.R``. Fidelity: exact (deterministic; 0.0).
    """

    def __init__(self) -> None:
        self.fold = 1

    def _augment(self, x: np.ndarray) -> np.ndarray:
        from itertools import combinations
        n = x.shape[1]
        series = np.concatenate([x[0, :n - 1], x[:, n - 1]])
        tsw = ts_data(series, n + 1).to_numpy(dtype=float)  # width n+1
        combos = list(combinations(range(1, n + 1), n - 1))  # n combos, 1-based, lexicographic
        blocks = []
        for i in range(1, self.fold + 1):
            combo = combos[len(combos) - i - 1]  # R: space[, ncol(space) - i]
            cols = [c - 1 for c in combo] + [n]   # selected lags + last col (index n of width n+1)
            blocks.append(tsw[:, cols])
        return np.vstack(blocks)


class TsAugAwareness(DalTransform):
    """Recency-aware augmentation (``ts_aug_awareness``); RNG-dependent.

    References:
        - Wen Q, et al. (2021). Time Series Data Augmentation for Deep Learning:
          A Survey. IJCAI.

    Porting:
        Mirrors ``ts_aug_awareness.R``. Fidelity: approximate.
        - Algorithm + noise scale faithful; values differ because the RNG differs
          (pass ``random_state`` for reproducibility).
    """

    def __init__(self, factor: int = 1, random_state: int | None = 0) -> None:
        self.factor = factor
        self.random_state = random_state

    def fit(self, data=None, **kwargs) -> TsAugAwareness:
        return self

    def _recent_index(self, n: int, rng) -> np.ndarray:
        rate = 10.0 / n
        i = np.ceil(rng.exponential(1.0 / rate, size=10 * n)).astype(int)
        i = i[(i > 0) & (i < n)]
        size = self.factor * n
        replace = len(i) < size  # R sample() is without replacement; fall back if too few
        i = rng.choice(i, size=size, replace=replace)
        i = n - i + 1           # 1-based, biased toward recent rows
        return np.sort(i)

    def _add_noise(self, block: np.ndarray, sd: float, rng) -> np.ndarray:
        noise = rng.normal(0.0, sd, size=block.shape)
        noise[:, -1] = 0.0
        return block + noise

    def transform(self, data, **kwargs) -> pd.DataFrame:
        x = _values(data)
        rng = np.random.default_rng(self.random_state)
        sd = _sd_within(x)
        idx = self._recent_index(x.shape[0], rng)       # 1-based
        block = self._add_noise(x[idx - 1], sd, rng)
        return _frame(np.vstack([x, block]))


def _fivenum(x: np.ndarray) -> np.ndarray:
    """R's ``fivenum`` (Tukey hinges): min, lower-hinge, median, upper-hinge, max."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    n4 = np.floor((n + 3) / 2) / 2
    d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n])
    lo = np.floor(d).astype(int) - 1
    hi = np.ceil(d).astype(int) - 1
    return 0.5 * (x[lo] + x[hi])


def _boxplot_outlier_idx(diff: np.ndarray) -> np.ndarray:
    """1-based indices of Tukey-fence outliers (as R ``boxplot.stats``)."""
    fn = _fivenum(diff)
    iqr = fn[3] - fn[1]
    lo, hi = fn[1] - 1.5 * iqr, fn[3] + 1.5 * iqr
    return np.where((diff < lo) | (diff > hi))[0] + 1


class TsAugAwaresmooth(TsAugAwareness):
    """Recency-aware augmentation with progressive smoothing (``ts_aug_awaresmooth``).

    References:
        - Wen Q, et al. (2021). Time Series Data Augmentation for Deep Learning:
          A Survey. IJCAI.

    Porting:
        Mirrors ``ts_aug_awaresmooth.R``. Fidelity: approximate (RNG-dependent;
        see :class:`TsAugAwareness`).
    """

    @staticmethod
    def _progressive_smoothing(series: np.ndarray) -> np.ndarray:
        s = np.asarray(series, dtype=float).copy()
        s = s[np.isfinite(s)]
        while True:
            diff = s[1:] - s[:-1]
            out = _boxplot_outlier_idx(diff)
            rj = out[(out > 1) & (out < len(s))]
            s[rj - 1] = (s[rj - 2] + s[rj]) / 2  # 1-based -> 0-based: serie[rj] uses rj-1
            diff2 = s[1:] - s[:-1]
            out2 = _boxplot_outlier_idx(diff2)
            if len(out2) == 0 or len(out2) == len(out):
                break
        return s

    def transform(self, data, **kwargs) -> pd.DataFrame:
        x = _values(data)
        n = x.shape[1]
        series = np.concatenate([x[0, :n - 1], x[:, n - 1]])
        smoothed = self._progressive_smoothing(series)
        windows = ts_data(smoothed, n).to_numpy(dtype=float)
        rng = np.random.default_rng(self.random_state)
        sd = _sd_within(windows)
        idx = self._recent_index(windows.shape[0], rng)
        block = self._add_noise(windows[idx - 1], sd, rng)
        return _frame(np.vstack([windows, block]))
