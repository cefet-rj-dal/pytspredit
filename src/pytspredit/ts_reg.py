"""Time series regression base classes.

Python counterpart of ``ts_reg.R`` and ``ts_regsw.R``.

``TsReg`` is the ancestor of all time series regressors: it extends the
``pydaltoolbox`` ``Predictor`` contract and provides the standard forecasting
``evaluate`` (MSE / sMAPE / R2), faithful to ``ts_reg.R`` (note the sMAPE drops
zero-denominator terms and R2 returns ``nan`` for a constant target).

``TsRegSW`` is the base for sliding-window regressors. It combines a
*preprocessor* (a ``TsNorm*``), a *lag map* (:class:`~pytspredit.TsLagmap`) and a
*backend* model. Concrete regressors implement :meth:`TsRegSW._do_fit` and
:meth:`TsRegSW._do_predict`. ``predict`` supports one-step-ahead (per row) and
iterative multi-step (``steps_ahead > 1``) forecasting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydaltoolbox import Predictor

from .lagmap import TsLagmap
from .norm import TsNormNone
from .tsdata import adjust_ts_data

__all__ = ["TsReg", "TsRegSW", "mse_ts", "r2_ts", "smape_ts"]


def _vec(v) -> np.ndarray:
    return np.asarray(v, dtype=float).ravel()


def mse_ts(actual, prediction) -> float:
    """Mean squared error (``MSE.ts``)."""
    a, p = _vec(actual), _vec(prediction)
    if len(a) != len(p):
        raise ValueError("actual and prediction have different lengths")
    return float(np.mean((a - p) ** 2))


def smape_ts(actual, prediction) -> float:
    """Symmetric MAPE (``sMAPE.ts``): drops zero-denominator terms, divides by n."""
    a, p = _vec(actual), _vec(prediction)
    if len(a) != len(p):
        raise ValueError("actual and prediction have different lengths")
    n = len(a)
    num = np.abs(a - p)
    denom = (np.abs(a) + np.abs(p)) / 2
    keep = denom != 0
    return float(np.sum(num[keep] / denom[keep]) / n)


def r2_ts(actual, prediction) -> float:
    """Coefficient of determination (``R2.ts``); ``nan`` if the target is constant."""
    a, p = _vec(actual), _vec(prediction)
    if len(a) != len(p):
        raise ValueError("actual and prediction have different lengths")
    sse = np.sum((p - a) ** 2)
    sst = np.sum((a - np.mean(a)) ** 2)
    if sst == 0:
        return float("nan")
    return float(1 - sse / sst)


class TsReg(Predictor):
    """Base class for time series regression models (non-sliding-window)."""

    def __init__(self) -> None:
        super().__init__()

    def fit(self, x=None, y=None, **kwargs) -> TsReg:
        return self

    def predict(self, x, **kwargs) -> np.ndarray:
        """Baseline predictor: return the last column (``t0``) of ``x``."""
        xv = x.to_numpy(dtype=float) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=float)
        if xv.ndim == 1:
            return xv
        return xv[:, -1]

    def evaluate(self, values, prediction, **kwargs) -> dict:
        """Forecasting metrics, faithful to ``evaluate.ts_reg``."""
        mse = mse_ts(values, prediction)
        smape = smape_ts(values, prediction)
        r2 = r2_ts(values, prediction)
        return {
            "values": _vec(values),
            "prediction": _vec(prediction),
            "smape": smape,
            "mse": mse,
            "R2": r2,
            "metrics": pd.DataFrame([{"mse": mse, "smape": smape, "R2": r2}]),
        }


class TsRegSW(TsReg):
    """Base for sliding-window regressors (``ts_regsw``).

    Parameters
    ----------
    preprocess:
        A ``TsNorm*`` preprocessor (default :class:`~pytspredit.TsNormNone`).
    input_size:
        Number of lagged inputs fed to the backend.
    input_map:
        Lag-selection strategy (default :class:`~pytspredit.TsLagmap`).
    """

    def __init__(self, preprocess=None, input_size: int | None = None, input_map=None) -> None:
        super().__init__()
        self.preprocess = preprocess if preprocess is not None else TsNormNone()
        self.input_size = input_size
        self.input_map = input_map if input_map is not None else TsLagmap()
        self.model = None

    # -- backend hooks (overridden by concrete regressors) --------------------
    def _do_fit(self, x, y):  # pragma: no cover - overridden
        raise NotImplementedError

    def _do_predict(self, x) -> np.ndarray:
        """Default: delegate to the backend's ``predict`` (``do_predict.ts_regsw``)."""
        return np.asarray(self.model.predict(np.asarray(x, dtype=float)))

    # -- helpers --------------------------------------------------------------
    def _as_matrix(self, data):
        """Select the lag columns learned by the input map."""
        df = data if isinstance(data, pd.DataFrame) else adjust_ts_data(data)
        if len(self.input_map.positions) == 0:
            self.input_map.fit(df, input_size=self.input_size)
        return df.iloc[:, self.input_map.positions]

    # -- fit/predict ----------------------------------------------------------
    def fit(self, x, y, **kwargs) -> TsRegSW:
        self.preprocess.fit(x)
        x = self.preprocess.transform(x)
        y = self.preprocess.transform(x, _vec(y))
        self.input_map.fit(x, y, input_size=self.input_size)
        x_model = self._as_matrix(x)
        self._do_fit(x_model, np.asarray(y, dtype=float))
        return self

    def predict(self, x, steps_ahead: int = 1, **kwargs) -> np.ndarray:
        if steps_ahead == 1:
            xt = self.preprocess.transform(x)
            data = self._as_matrix(xt)
            y = self._do_predict(data)
            y = self.preprocess.inverse_transform(xt, _vec(y))
            return _vec(y)

        xdf = x if isinstance(x, pd.DataFrame) else adjust_ts_data(x)
        if xdf.shape[0] > 1:
            raise ValueError("in steps_ahead mode, x must have a single row")
        cnames = list(xdf.columns)
        row = xdf.iloc[[0]].copy()
        prediction = []
        for _ in range(steps_ahead):
            row.columns = cnames
            xt = self.preprocess.transform(row)
            y = self._do_predict(self._as_matrix(xt))
            row = adjust_ts_data(self.preprocess.inverse_transform(xt))
            y = _vec(self.preprocess.inverse_transform(xt, _vec(y)))[0]
            vals = np.array(row.to_numpy(dtype=float))  # ensure a writable copy
            vals[0, :-1] = vals[0, 1:]
            vals[0, -1] = y
            row = pd.DataFrame(vals, columns=cnames)
            prediction.append(y)
        return np.asarray(prediction, dtype=float)
