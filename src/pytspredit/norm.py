"""Time series normalizations.

Python counterpart of ``ts_norm_none.R``, ``ts_norm_gminmax.R``,
``ts_norm_swminmax.R``, ``ts_norm_diff.R`` and ``ts_norm_an.R`` (with
``tsanutils.R``).

Each normalizer is a :class:`pydaltoolbox.DalTransform` with ``fit`` /
``transform`` / ``inverse_transform``. Following the R contract, ``transform``
and ``inverse_transform`` take an optional ``x`` argument: when ``x`` is ``None``
the *window matrix* ``data`` is (de)normalized; when ``x`` is given, it is the
aligned target/forecast vector, (de)normalized with the statistics computed for
the corresponding windows.

For the per-window normalizers (``TsNormSwminmax``, ``TsNormAn``) the R code
stores the per-row statistics as attributes on the returned matrix. Here we keep
them as instance state set during the ``x is None`` call and reused by the
immediately following ``x``-path / inverse call, which matches how the
sliding-window pipeline invokes them sequentially.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydaltoolbox import DalTransform, OutliersBoxplot

__all__ = [
    "TsNormAn",
    "TsNormDiff",
    "TsNormGminmax",
    "TsNormNone",
    "TsNormSwminmax",
]


def _values(data) -> np.ndarray:
    """Return ``data`` as a 2-D float ndarray."""
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=float)
    arr = np.asarray(data, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _frame(values: np.ndarray, like) -> pd.DataFrame:
    """Wrap ``values`` as a DataFrame, reusing ``like``'s columns when shapes match."""
    values = np.asarray(values, dtype=float)
    cols = None
    if isinstance(like, pd.DataFrame) and like.shape[1] == values.shape[1]:
        cols = like.columns
    else:
        cols = [f"t{k}" for k in range(values.shape[1] - 1, -1, -1)]
    return pd.DataFrame(values, columns=cols)


class TsNormNone(DalTransform):
    """Identity normalization (``ts_norm_none``).

    Leaves data unchanged but honours the pre/post-processing interface. Unlike
    the R version, the ``x``-path returns ``x`` unchanged (rather than the window
    matrix), so it composes correctly inside a sliding-window regressor.
    """

    def fit(self, data=None, **kwargs) -> TsNormNone:
        return self

    def transform(self, data, x=None, **kwargs):
        return data if x is None else x

    def inverse_transform(self, data, x=None, **kwargs):
        return data if x is None else x


class _OutliersMixin:
    """Shared optional outlier mitigation before range estimation (as in R)."""

    def _apply_outliers(self, values: np.ndarray) -> np.ndarray:
        if self.outliers is None:
            return values
        frame = pd.DataFrame(values)
        out = self.outliers.fit(frame).transform(frame)
        return _values(out)


class TsNormGminmax(_OutliersMixin, DalTransform):
    """Global min-max normalization (``ts_norm_gminmax``).

    Maps the global training minimum to 0 and maximum to 1; the same scaling is
    inverted on predictions.
    """

    def __init__(self, outliers="default") -> None:
        self.outliers = OutliersBoxplot() if outliers == "default" else outliers
        self.gmin: float | None = None
        self.gmax: float | None = None

    def fit(self, data, **kwargs) -> TsNormGminmax:
        values = self._apply_outliers(_values(data))
        self.gmin = float(np.nanmin(values))
        self.gmax = float(np.nanmax(values))
        return self

    def _range(self) -> float:
        return self.gmax - self.gmin

    def transform(self, data, x=None, **kwargs):
        if x is not None:
            return (np.asarray(x, dtype=float) - self.gmin) / self._range()
        return _frame((_values(data) - self.gmin) / self._range(), data)

    def inverse_transform(self, data, x=None, **kwargs):
        if x is not None:
            return np.asarray(x, dtype=float) * self._range() + self.gmin
        return _frame(_values(data) * self._range() + self.gmin, data)


class TsNormSwminmax(_OutliersMixin, DalTransform):
    """Sliding-window (per-row) min-max normalization (``ts_norm_swminmax``)."""

    def __init__(self, outliers="default") -> None:
        self.outliers = OutliersBoxplot() if outliers == "default" else outliers
        self.i_min: np.ndarray | None = None
        self.i_max: np.ndarray | None = None

    def fit(self, data, **kwargs) -> TsNormSwminmax:
        # R applies the outlier step here but stores nothing; keep it faithful.
        self._apply_outliers(_values(data))
        return self

    def transform(self, data, x=None, **kwargs):
        if x is not None:
            rng = self.i_max - self.i_min
            return (np.asarray(x, dtype=float) - self.i_min) / rng
        values = _values(data)
        self.i_min = values.min(axis=1)
        self.i_max = values.max(axis=1)
        rng = (self.i_max - self.i_min)[:, None]
        return _frame((values - self.i_min[:, None]) / rng, data)

    def inverse_transform(self, data, x=None, **kwargs):
        rng = self.i_max - self.i_min
        if x is not None:
            return np.asarray(x, dtype=float) * rng + self.i_min
        return _frame(_values(data) * rng[:, None] + self.i_min[:, None], data)


class TsNormDiff(_OutliersMixin, DalTransform):
    """First-difference normalization + global min-max (``ts_norm_diff``).

    Reduces the window width by one (``t9..t0`` -> consecutive differences), then
    scales the differenced windows with global min-max. The target is expressed
    as the difference from the most recent input lag.
    """

    def __init__(self, outliers="default") -> None:
        self.outliers = OutliersBoxplot() if outliers == "default" else outliers
        self.gmin: float | None = None
        self.gmax: float | None = None
        self._ref: np.ndarray | None = None

    @staticmethod
    def _diff(values: np.ndarray) -> np.ndarray:
        # columns 2..n minus columns 1..n-1 -> n-1 difference columns
        return values[:, 1:] - values[:, :-1]

    def _range(self) -> float:
        return self.gmax - self.gmin

    def fit(self, data, **kwargs) -> TsNormDiff:
        diff = self._apply_outliers(self._diff(_values(data)))
        self.gmin = float(np.nanmin(diff))
        self.gmax = float(np.nanmax(diff))
        return self

    def transform(self, data, x=None, **kwargs):
        if x is not None:
            x = np.asarray(x, dtype=float) - self._ref
            return (x - self.gmin) / self._range()
        values = _values(data)
        self._ref = values[:, -1].copy()  # most recent input lag (reference)
        diff = self._diff(values)
        return _frame((diff - self.gmin) / self._range(), None)

    def inverse_transform(self, data, x=None, **kwargs):
        if x is not None:
            x = np.asarray(x, dtype=float) * self._range() + self.gmin
            return x + self._ref
        # reconstruct the series from consecutive differences and the reference
        diff = _values(data) * self._range() + self.gmin
        ref = self._ref[:, None]
        cols = np.concatenate([diff, ref], axis=1)
        for j in range(cols.shape[1] - 2, -1, -1):
            cols[:, j] = cols[:, j + 1] - cols[:, j]
        return _frame(cols, None)


def _row_apply(values: np.ndarray, fun) -> np.ndarray:
    """Apply ``fun`` per row with non-finite results replaced by 0 (R contract)."""
    out = np.array([fun(row[np.isfinite(row)]) for row in values], dtype=float)
    out[~np.isfinite(out)] = 0.0
    return out


def _ema(row: np.ndarray) -> float:
    n = len(row)
    if n == 0:
        return 0.0
    alpha = 1 - 2.0 / (n + 1)
    # weight for position k (k=0 oldest .. n-1 newest) is alpha^(n-1-k): newest -> 1
    weights = alpha ** np.arange(n - 1, -1, -1)
    return float(np.sum(weights * row) / np.sum(weights))


def _mad(row: np.ndarray) -> float:
    if len(row) == 0:
        return 0.0
    med = np.median(row)
    return float(1.4826 * np.median(np.abs(row - med)))


def _sd(row: np.ndarray) -> float:
    return float(np.std(row, ddof=1)) if len(row) > 1 else 0.0


def _zero(row: np.ndarray) -> float:
    return 0.0


class TsNormAn(DalTransform):
    """Adaptive normalization (``ts_norm_an``).

    Normalizes each window relative to an adaptive reference level (mean or EMA)
    and an optional local scale (sd or MAD), via one of four operators
    (``divide``, ``subtract``, ``softdivide``, ``asinh``), followed by global
    min-max. Mirrors ``ts_norm_an.R`` + ``tsanutils.R``.
    """

    def __init__(
        self,
        outliers="default",
        nw: int = 0,
        average: str = "mean",
        operation: str = "divide",
        scale: str = "sd",
        lambda_: float = 1.0,
        epsilon: float = 1e-8,
    ) -> None:
        if average not in ("mean", "ema"):
            raise ValueError("average must be 'mean' or 'ema'.")
        if operation not in ("divide", "subtract", "softdivide", "asinh"):
            raise ValueError("operation must be divide/subtract/softdivide/asinh.")
        if scale not in ("sd", "mad", "none"):
            raise ValueError("scale must be sd/mad/none.")
        self.outliers = OutliersBoxplot() if outliers == "default" else outliers
        self.nw = nw
        self.average = average
        self.operation = operation
        self.scale = scale
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.gmin: float | None = None
        self.gmax: float | None = None
        self.grange: float | None = None
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    # -- reference estimation -------------------------------------------------
    def _subset_recent(self, values: np.ndarray) -> np.ndarray:
        if self.nw <= 0 or self.nw >= values.shape[1]:
            return values
        return values[:, -self.nw:]

    def _reference(self, values: np.ndarray):
        window = self._subset_recent(values)
        center_fun = np.mean if self.average == "mean" else _ema
        scale_fun = {"sd": _sd, "mad": _mad, "none": _zero}[self.scale]
        center = _row_apply(window, center_fun)
        scale_value = _row_apply(window, scale_fun)
        return center, scale_value

    # -- operators (per-row broadcast) ---------------------------------------
    def _stabilize(self, center: np.ndarray) -> np.ndarray:
        sign = np.sign(center)
        sign[sign == 0] = 1
        return sign * np.maximum(np.abs(center), self.epsilon)

    def _ref_scale(self, center: np.ndarray, scale_value: np.ndarray) -> np.ndarray:
        return np.sqrt(scale_value ** 2 + (self.lambda_ * center) ** 2 + self.epsilon ** 2)

    def _forward(self, data: np.ndarray, center: np.ndarray, scale_value: np.ndarray) -> np.ndarray:
        c = center[:, None] if data.ndim == 2 else center
        s = scale_value[:, None] if data.ndim == 2 else scale_value
        if self.operation == "divide":
            return data / (self._stabilize(center)[:, None] if data.ndim == 2 else self._stabilize(center))
        if self.operation == "subtract":
            return data - c
        if self.operation == "softdivide":
            return (data - c) / self._ref_scale(c, s)
        rs = self._ref_scale(c, s)  # asinh
        return np.arcsinh(data / rs) - np.arcsinh(c / rs)

    def _inverse(self, data: np.ndarray, center: np.ndarray, scale_value: np.ndarray) -> np.ndarray:
        c = center[:, None] if data.ndim == 2 else center
        s = scale_value[:, None] if data.ndim == 2 else scale_value
        if self.operation == "divide":
            stab = self._stabilize(center)
            return data * (stab[:, None] if data.ndim == 2 else stab)
        if self.operation == "subtract":
            return data + c
        if self.operation == "softdivide":
            return c + data * self._ref_scale(c, s)
        rs = self._ref_scale(c, s)  # asinh
        return rs * np.sinh(data + np.arcsinh(c / rs))

    # -- DalTransform API -----------------------------------------------------
    def fit(self, data, **kwargs) -> TsNormAn:
        values = _values(data)
        center, scale_value = self._reference(values)
        transformed = self._forward(values, center, scale_value)
        if self.outliers is not None:
            frame = pd.DataFrame(transformed)
            transformed = _values(self.outliers.fit(frame).transform(frame))
        self.gmin = float(np.nanmin(transformed))
        self.gmax = float(np.nanmax(transformed))
        self.grange = 1.0 if abs(self.gmax - self.gmin) < self.epsilon else self.gmax - self.gmin
        return self

    def transform(self, data, x=None, **kwargs):
        if x is not None:
            x = self._forward(np.asarray(x, dtype=float), self._center, self._scale)
            return (x - self.gmin) / self.grange
        values = _values(data)
        center, scale_value = self._reference(values)
        transformed = self._forward(values, center, scale_value)
        transformed = (transformed - self.gmin) / self.grange
        self._center, self._scale = center, scale_value
        return _frame(transformed, data)

    def inverse_transform(self, data, x=None, **kwargs):
        if x is not None:
            x = np.asarray(x, dtype=float) * self.grange + self.gmin
            return self._inverse(x, self._center, self._scale)
        values = _values(data) * self.grange + self.gmin
        return _frame(self._inverse(values, self._center, self._scale), data)
