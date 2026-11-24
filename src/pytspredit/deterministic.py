"""Deterministic (rule-based) univariate predictors.

Python counterpart of ``ts_deterministic.R`` (+ ``ts_periodic``, ``ts_persist``).
These forecast by a deterministic *rule* rather than a statistical learner.

Theory:
    Rule-based baselines are the standard reference models in forecasting (Hyndman &
    Athanasopoulos). ``"periodic"`` (seasonal naive) repeats a learned cycle of fixed
    length; ``"persist"`` (naive / random walk) repeats the most recent observed value.
    Despite their simplicity they are strong baselines on seasonal and trending series and
    are the yardstick a learned model must beat.

* ``"periodic"``: repeat a learned cycle of fixed length (seasonal naive).
* ``"persist"``: repeat the most recent observed value (naive forecast).

Like :class:`~pytspredit.TsArima`, they extend :class:`~pytspredit.TsReg` and work
on a raw series (``ts_data`` with ``sw = 1``) or on sliding windows.

References:
    - Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
      Practice* (3rd ed.), OTexts (simple benchmark methods: naive, seasonal naive).

Porting:
    Mirrors ``ts_deterministic.R`` (which also defines ``ts_periodic`` and ``ts_persist``).
    Fidelity: exact (rule-based; matches R to floating-point tolerance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ts_reg import TsReg, _vec

__all__ = ["TsDeterministic", "TsPeriodic", "TsPersist"]

_EPS = np.sqrt(np.finfo(float).eps)


def _as_series(x, y=None) -> np.ndarray:
    """Reconstruct a 1-D series from raw input or (windows, output)."""
    if y is not None:
        xv = np.asarray(x, dtype=float)
        xv = xv.reshape(1, -1) if xv.ndim == 1 else xv
        return np.concatenate([xv[0, :], _vec(y)]) if xv.shape[0] else _vec(y)
    if isinstance(x, (pd.DataFrame, np.ndarray)) and np.asarray(x).ndim == 2:
        arr = np.asarray(x, dtype=float)
        if arr.shape[1] == 1:
            return arr[:, 0]
        if arr.shape[0] == 1:
            return arr[0, :]
    return np.asarray(x, dtype=float).ravel()


def _infer_context_size(cycle: np.ndarray) -> int:
    """Smallest context length that makes the next cycle value unambiguous."""
    period = len(cycle)
    extended = np.concatenate([cycle, cycle])
    for k in range(1, period + 1):
        groups: dict[tuple, set] = {}
        ok = True
        for i in range(period):
            key = tuple(np.round(extended[i:i + k], 12))
            groups.setdefault(key, set()).add(round(float(extended[i + k]), 12))
        if all(len(v) == 1 for v in groups.values()) and ok:
            return k
    return period


def _periodic_next(cycle: np.ndarray, context_size: int, history: np.ndarray) -> float:
    if len(cycle) == 0:
        raise ValueError("periodic predictor has not been fitted yet.")
    if len(history) == 0:
        return float(cycle[0])
    period = len(cycle)
    extended = np.concatenate([cycle, cycle])
    k_max = min(len(history), context_size)
    for k in range(k_max, 0, -1):
        context = history[-k:]
        matches = []
        for i in range(period):
            candidate = extended[i:i + k]
            if np.allclose(candidate, context, atol=_EPS, rtol=0):
                matches.append(extended[i + k])
        uniq = {round(float(m), 12) for m in matches}
        if len(uniq) == 1:
            return float(matches[0])
    return float(cycle[0])


class TsDeterministic(TsReg):
    """Rule-based univariate predictor (``ts_deterministic``).

    References:
        - Hyndman RJ, Athanasopoulos G (2021). Forecasting: Principles and
          Practice, 3rd ed. OTexts. https://otexts.com/fpp3/ (naive/seasonal-naive
          benchmarks).

    Porting:
        Mirrors ``ts_deterministic.R``. Fidelity: exact.
        - Native rule-based forecasting (no learner); matches R exactly. The R
          file carries no academic reference; the naive/seasonal-naive baselines
          are standard forecasting benchmarks.
    """

    def __init__(self, mode: str = "periodic", period=None, context_size=None) -> None:
        super().__init__()
        if mode not in ("periodic", "persist"):
            raise ValueError("mode must be 'periodic' or 'persist'.")
        if mode == "periodic":
            if not isinstance(period, (int, np.integer)) or period < 1:
                raise ValueError("period must be a positive integer for mode='periodic'.")
            period = int(period)
        if context_size is not None:
            if not isinstance(context_size, (int, np.integer)) or context_size < 1:
                raise ValueError("context_size must be a positive integer.")
            context_size = int(context_size)
        self.mode = mode
        self.period = period
        self.context_size = context_size
        self.cycle = np.empty(0)
        self.last_value = float("nan")
        self.history = np.empty(0)

    def fit(self, x, y=None, **kwargs) -> TsDeterministic:
        series = _as_series(x, y)
        if len(series) == 0:
            raise ValueError("ts_deterministic requires at least one observation.")
        self.history = np.asarray(series, dtype=float)
        if self.mode == "periodic":
            if len(series) < self.period:
                raise ValueError("periodic mode requires at least 'period' observations.")
            self.cycle = self.history[-self.period:]
            if self.context_size is None:
                self.context_size = _infer_context_size(self.cycle)
            else:
                self.context_size = min(self.context_size, self.period)
        else:  # persist
            self.last_value = float(self.history[-1])
            self.context_size = 1
        return self

    def predict(self, x=None, y=None, steps_ahead=None, **kwargs) -> np.ndarray:
        steps_ahead = 1 if steps_ahead is None else int(steps_ahead)
        is_matrix = isinstance(x, (pd.DataFrame, np.ndarray)) and np.asarray(x).ndim == 2

        if steps_ahead == 1 and is_matrix:
            arr = np.asarray(x, dtype=float)
            if self.mode == "persist":
                return arr[:, -1]
            k = min(self.context_size, arr.shape[1])
            contexts = arr[:, arr.shape[1] - k:]
            return np.array([_periodic_next(self.cycle, self.context_size, row) for row in contexts])

        if self.mode == "persist":
            if steps_ahead == 1 and x is not None:
                value = _as_series(x)[-1] if not is_matrix else np.asarray(x, float)[-1, -1]
            elif not np.isnan(self.last_value):
                value = self.last_value
            else:
                raise ValueError("persist predictor has not been fitted yet.")
            return np.repeat(float(value), steps_ahead)

        # periodic, multi-step recursive
        if x is None:
            history = self.history.copy()
        elif is_matrix:
            arr = np.asarray(x, dtype=float)
            if arr.shape[0] > 1 and steps_ahead > 1:
                raise ValueError("multi-step deterministic forecasting needs a single history row.")
            history = arr[-1, :].copy()
        else:
            history = _as_series(x).copy()

        prediction = np.empty(steps_ahead, dtype=float)
        for i in range(steps_ahead):
            prediction[i] = _periodic_next(self.cycle, self.context_size, history)
            history = np.append(history, prediction[i])
        return prediction


class TsPeriodic(TsDeterministic):
    """Periodic deterministic predictor (``ts_periodic``).

    Porting:
        Mirrors ``ts_periodic.R`` (``ts_deterministic`` with ``mode='periodic'``).
        Fidelity: exact.
    """

    def __init__(self, period, context_size=None) -> None:
        super().__init__("periodic", period=period, context_size=context_size)


class TsPersist(TsDeterministic):
    """Persistence deterministic predictor (``ts_persist``).

    Porting:
        Mirrors ``ts_persist.R`` (``ts_deterministic`` with ``mode='persist'``).
        Fidelity: exact.
    """

    def __init__(self) -> None:
        super().__init__("persist")
