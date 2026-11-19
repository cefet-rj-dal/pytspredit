"""Window linear models: DARIMA and WARMA.

Python counterpart of ``ts_darima.R`` and ``ts_warma.R``. Both are light
sliding-window **linear** regressors (``stats::lm`` -> scikit-learn
``LinearRegression``) in the ``ts_regsw`` lineage -- *not* heavy ARIMA backends.

* ``TsDarima`` delegates the integration (differencing) to the preprocessor
  (e.g. :class:`~pytspredit.TsNormDiff`) and fits a linear model over the selected
  lag columns.
* ``TsWarma`` adds a local per-window normalization in ``steps`` (0: none; 1:
  subtract the window mean; 2: also divide by the window sd) before the linear
  model; ``steps=None`` picks the smallest step reaching integration order zero
  (``ndiffs`` via a KPSS test, mirroring ``forecast::ndiffs``).

Both are deterministic (OLS), so they match R exactly given the same configuration
(WARMA's automatic ``steps`` selection may differ if the KPSS decision differs;
pass an explicit ``steps`` for guaranteed parity).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .lagmap import _complete_positions
from .ts_reg import TsRegSW, _vec

__all__ = ["TsDarima", "TsWarma"]


class _RLinear:
    """OLS that reproduces R's ``lm`` on rank-deficient designs.

    R's ``lm`` uses a pivoted QR (``dqrdc2``) that keeps the leftmost independent
    columns and sets the coefficients of aliased columns to 0. scikit-learn's
    ``LinearRegression`` instead returns a minimum-norm ``lstsq`` solution, which
    extrapolates differently when the lag columns are collinear (common for smooth
    series). We greedily select independent columns (modified Gram-Schmidt,
    relative tolerance ``1e-7`` like R) so DARIMA/WARMA match R exactly.
    """

    def __init__(self, intercept: bool = True, tol: float = 1e-7) -> None:
        self.intercept = bool(intercept)
        self.tol = tol
        self.coef_ = None

    def _design(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.column_stack([np.ones(x.shape[0]), x]) if self.intercept else x

    def fit(self, x, y) -> _RLinear:
        d = self._design(x)
        y = _vec(y)
        orig = np.linalg.norm(d, axis=0)
        orig[orig == 0] = 1.0
        kept, basis = [], []
        for j in range(d.shape[1]):
            v = d[:, j].astype(float).copy()
            for q in basis:
                v = v - (q @ v) * q
            nv = np.linalg.norm(v)
            if nv > self.tol * orig[j]:
                kept.append(j)
                basis.append(v / nv)
        coef = np.zeros(d.shape[1])
        if kept:
            bk, *_ = np.linalg.lstsq(d[:, kept], y, rcond=None)
            coef[kept] = bk
        self.coef_ = coef
        return self

    def predict(self, x) -> np.ndarray:
        return self._design(x) @ self.coef_


def _mat(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        arr = x.to_numpy(dtype=float)
    else:
        arr = np.asarray(x, dtype=float)
    return arr.reshape(1, -1) if arr.ndim == 1 else np.array(arr)


def _adjust_positions(positions0, n_before, n_after, input_size):
    """Shift/complete lag positions when the preprocessor changed the width.

    Mirrors ``ts_darima_positions``/``ts_warma_positions``. ``positions0`` are
    0-based (from ``TsLagmap``); returns 0-based positions into the transformed
    matrix of width ``n_after``.
    """
    pos = np.asarray(positions0, dtype=int) + 1  # to 1-based (R space)
    shift = max(0, int(n_before) - int(n_after))
    if shift > 0:
        pos = pos[pos > shift] - shift
    pos = pos[(pos >= 1) & (pos <= n_after)]
    needed = min(int(input_size), int(n_after))
    if len(pos) == 0:
        pos = np.arange(max(1, n_after - needed + 1), n_after + 1)
    if len(pos) < needed:
        pos = _complete_positions(pos, n_after, needed)
    return np.asarray(pos, dtype=int) - 1  # back to 0-based


def _inverse_safe(preprocess, data, x=None):
    try:
        return preprocess.inverse_transform(data, x)
    except Exception:  # noqa: BLE001 - identity fallback, as in ts_darima.R
        return data if x is None else x


class TsDarima(TsRegSW):
    """Delegated-differencing linear regressor (``ts_darima``)."""

    def __init__(self, preprocess=None, input_size=None, input_map=None, intercept=True) -> None:
        super().__init__(preprocess, input_size, input_map)
        self.intercept = bool(intercept)
        self._positions = None

    def fit(self, x, y, **kwargs) -> TsDarima:
        x_raw = _mat(x)
        y_raw = _vec(y)
        self.input_map.fit(x_raw, y_raw, input_size=self.input_size)
        self.preprocess.fit(x_raw)
        x_tr = _mat(self.preprocess.transform(x_raw))
        y_tr = _vec(self.preprocess.transform(x_tr, y_raw))
        self._positions = _adjust_positions(self.input_map.positions, x_raw.shape[1],
                                            x_tr.shape[1], self.input_size)
        self.model = _RLinear(intercept=self.intercept).fit(x_tr[:, self._positions], y_tr)
        return self

    def _predict_one(self, x_roll):
        x_tr = _mat(self.preprocess.transform(x_roll))
        y = self.model.predict(x_tr[:, self._positions])
        return x_tr, _vec(_inverse_safe(self.preprocess, x_tr, _vec(y)))

    def predict(self, x, steps_ahead: int = 1, **kwargs) -> np.ndarray:
        x = _mat(x)
        if steps_ahead == 1:
            _, y = self._predict_one(x)
            return _vec(y)
        if x.shape[0] > 1:
            raise ValueError("in steps_ahead mode, x must have a single row")
        prediction = []
        x_roll = x[:1, :].copy()
        for _ in range(steps_ahead):
            x_tr, y = self._predict_one(x_roll)
            yi = float(y[0])
            x_roll = _mat(_inverse_safe(self.preprocess, x_tr))
            x_roll[0, :-1] = x_roll[0, 1:]
            x_roll[0, -1] = yi
            prediction.append(yi)
        return np.asarray(prediction, dtype=float)


def _ndiffs(series, alpha: float = 0.05, max_d: int = 2) -> int:
    """Number of differences for stationarity via a KPSS test (~ forecast::ndiffs)."""
    from statsmodels.tsa.stattools import kpss
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    d = 0
    while d < max_d and len(x) > 10:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, pvalue, *_ = kpss(x, regression="c", nlags="auto")
        except Exception:  # noqa: BLE001 - treat test failure as stationary
            break
        if pvalue >= alpha:  # fail to reject stationarity -> stop
            break
        x = np.diff(x)
        d += 1
    return d


def _warma_normalize(x: np.ndarray, y=None, steps: int = 0):
    means = x.mean(axis=1)
    scales = np.ones(x.shape[0])
    x_norm = x.copy()
    y_norm = None if y is None else _vec(y).astype(float).copy()
    if steps >= 1:
        x_norm = x_norm - means[:, None]
        if y_norm is not None:
            y_norm = y_norm - means
    if steps >= 2:
        scales = x_norm.std(axis=1, ddof=1)
        nz = scales > 0
        if nz.any():
            x_norm[nz] = x_norm[nz] / scales[nz][:, None]
            if y_norm is not None:
                y_norm[nz] = y_norm[nz] / scales[nz]
        if (~nz).any():
            x_norm[~nz] = 0.0
            if y_norm is not None:
                y_norm[~nz] = 0.0
            scales[~nz] = 1.0
    return x_norm, y_norm, means, scales


def _warma_inverse_y(y, means, scales, steps):
    y = _vec(y).astype(float)
    if steps >= 2:
        y = y * scales
    if steps >= 1:
        y = y + means
    return y


class TsWarma(TsRegSW):
    """Window ARMA-inspired linear regressor with local step normalization (``ts_warma``)."""

    def __init__(self, preprocess=None, input_size=None, input_map=None,
                 steps=None, intercept=True) -> None:
        super().__init__(preprocess, input_size, input_map)
        if steps is not None and steps not in (0, 1, 2):
            raise ValueError("steps must be 0, 1, 2, or None.")
        self.steps = steps
        self.intercept = bool(intercept)
        self._positions = None

    def _select_steps(self, series_tr, sw):
        from .tsdata import ts_data
        for steps in (0, 1, 2):
            windows = ts_data(series_tr, sw=sw).to_numpy(dtype=float)
            local = _warma_normalize(windows, steps=steps)[0]
            if _ndiffs(local[:, -1]) == 0:
                return steps
        return 2

    def fit(self, x, y, **kwargs) -> TsWarma:
        x_raw = _mat(x)
        y_raw = _vec(y)
        self.input_map.fit(x_raw, y_raw, input_size=self.input_size)
        self.preprocess.fit(x_raw)
        x_tr = _mat(self.preprocess.transform(x_raw))
        y_tr = _vec(self.preprocess.transform(x_tr, y_raw))
        if self.steps is None:
            series_tr = np.concatenate([x_tr[0, :], y_tr])
            self.steps = self._select_steps(series_tr, x_tr.shape[1])
        x_local, y_local, _, _ = _warma_normalize(x_tr, y_tr, steps=self.steps)
        self._positions = _adjust_positions(self.input_map.positions, x_raw.shape[1],
                                            x_local.shape[1], self.input_size)
        self.model = _RLinear(intercept=self.intercept).fit(x_local[:, self._positions], y_local)
        return self

    def _predict_one(self, x_roll):
        x_tr = _mat(self.preprocess.transform(x_roll))
        local, _, means, scales = _warma_normalize(x_tr, steps=self.steps)
        y_norm = self.model.predict(local[:, self._positions])
        y = _warma_inverse_y(y_norm, means, scales, self.steps)
        return x_tr, _vec(_inverse_safe(self.preprocess, x_tr, y))

    def predict(self, x, steps_ahead: int = 1, **kwargs) -> np.ndarray:
        x = _mat(x)
        if steps_ahead == 1:
            _, y = self._predict_one(x)
            return _vec(y)
        if x.shape[0] > 1:
            raise ValueError("in steps_ahead mode, x must have a single row")
        prediction = []
        x_roll = x[:1, :].copy()
        for _ in range(steps_ahead):
            x_tr, y = self._predict_one(x_roll)
            yi = float(y[0])
            x_roll = _mat(_inverse_safe(self.preprocess, x_tr))
            x_roll[0, :-1] = x_roll[0, 1:]
            x_roll[0, -1] = yi
            prediction.append(yi)
        return np.asarray(prediction, dtype=float)
