"""ARIMA regressor (``ts_arima.R``).

Univariate ARIMA built on ``statsmodels``. Unlike the sliding-window regressors,
``TsArima`` extends :class:`~pytspredit.TsReg` and operates on the raw series (a
``ts_data`` with ``sw = 1``).

Parity note: the R package uses ``forecast::Arima`` / ``forecast::auto.arima``.
``statsmodels`` is a different maximum-likelihood implementation, so forecasts are
close but not bit-identical (expected backend tolerance). With an explicit
``(p, d, q)`` the orders match exactly; ``auto`` order selection here is a compact
AICc search that approximates (but may not reproduce) ``auto.arima``'s choice.
"""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .ts_reg import TsReg, _vec

__all__ = ["TsArima"]


def _trend(d: int, include_mean: bool, include_drift: bool) -> str:
    if d == 0 and include_mean:
        return "c"
    if d == 1 and include_drift:
        return "t"
    return "n"


def _fit_arima(series, order, trend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(series, order=order, trend=trend).fit()


def _try_fit(series, order, trend):
    """Fit an ARIMA, returning ``None`` if the order cannot be estimated."""
    try:
        return _fit_arima(series, order, trend)
    except Exception:  # noqa: BLE001 - many fit errors are possible; skip the order
        return None


def _auto_order(series):
    """Compact AICc search approximating auto.arima (p,q in 0..3, d in 0..2)."""
    best = None
    for d in (0, 1, 2):
        trend = "c" if d == 0 else ("t" if d == 1 else "n")
        for p in range(4):
            for q in range(4):
                if p == 0 and q == 0 and d == 0:
                    continue
                res = _try_fit(series, (p, d, q), trend)
                if res is None:
                    continue
                if best is None or res.aicc < best[0]:
                    best = (res.aicc, (p, d, q), trend, res)
    if best is None:
        return (1, 0, 0), "c", _fit_arima(series, (1, 0, 0), "c")
    return best[1], best[2], best[3]


class TsArima(TsReg):
    """ARIMA(p, d, q) forecaster. Leave orders ``None`` for automatic selection."""

    def __init__(self, p=None, d=None, q=None) -> None:
        super().__init__()
        self.p, self.d, self.q = p, d, q
        self.manual_order = p is not None and d is not None and q is not None
        self.include_mean = bool(self.manual_order and d == 0)
        self.include_drift = bool(self.manual_order and d == 1)
        self.model = None
        self.train = None

    def fit(self, x, y=None, **kwargs) -> TsArima:
        series = _vec(x)
        self.train = series
        if self.manual_order:
            trend = _trend(self.d, self.include_mean, self.include_drift)
            self.model = _fit_arima(series, (self.p, self.d, self.q), trend)
        else:
            (self.p, self.d, self.q), _, self.model = _auto_order(series)
            self.include_mean = self.d == 0
            self.include_drift = self.d == 1
        return self

    def _order(self):
        return (self.p, self.d, self.q)

    def predict(self, x, y=None, steps_ahead=None, **kwargs) -> np.ndarray:
        xv = _vec(x)
        # in-sample fit when x is the training series
        if len(xv) == len(self.train) and np.allclose(xv, self.train):
            return np.asarray(self.model.fittedvalues, dtype=float)

        if steps_ahead is None:
            steps_ahead = len(xv)

        if steps_ahead == 1 and len(xv) != 1:
            # rolling one-step-ahead: refit with known order as actuals arrive
            trend = _trend(self.d, self.include_mean, self.include_drift)
            preds = []
            history = list(self.train)
            model = self.model
            for i in range(len(xv)):
                preds.append(float(np.asarray(model.forecast(1))[0]))
                history.append(xv[i])
                refit = _try_fit(np.asarray(history, dtype=float), self._order(), trend)
                if refit is None:  # fall back to auto selection, as in ts_arima.R
                    (self.p, self.d, self.q), trend, refit = _auto_order(
                        np.asarray(history, dtype=float))
                model = refit
            return np.asarray(preds, dtype=float)

        return np.asarray(self.model.forecast(steps_ahead), dtype=float)
