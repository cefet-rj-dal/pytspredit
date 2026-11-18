"""Lag-mapping strategies for sliding-window predictors.

Python counterpart of ``ts_lagmap.R`` and ``tslagutils.R``. ``TsLagmap`` learns,
during ``fit``, which ``input_size`` lag columns of the window matrix to feed the
backend regressor. Twelve strategies are supported:

* **positional:** ``recent``, ``even``, ``geom``
* **correlation-driven:** ``acf``, ``pacf``, ``peaks``, ``seasonal``,
  ``acf_seasonal``, ``pacf_seasonal``, ``blocks``
* **supervised:** ``mi`` (mutual information), ``mrmr`` (min-redundancy
  max-relevance)

Positions are 1-based column indices internally (as in R); the public
``positions`` attribute is 0-based for direct ``iloc`` column selection.

Seasonal auto-estimation differs from R: when ``seasonality`` is ``None`` the R
code calls ``forecast::findfrequency`` (an AR spectral-density search). Here we
use a lighter periodogram-based estimate, so the three seasonal strategies match
R exactly when an explicit ``seasonality`` is given; with auto-estimation the
detected period may differ. Pass ``seasonality`` for guaranteed parity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydaltoolbox import DalBase
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import pacf as sm_pacf

__all__ = ["TsLagmap"]

_METHODS = (
    "recent", "even", "geom", "acf", "pacf", "peaks",
    "seasonal", "acf_seasonal", "pacf_seasonal", "blocks", "mi", "mrmr",
)
_NEEDS_TARGET = {
    "acf", "pacf", "peaks", "seasonal", "acf_seasonal",
    "pacf_seasonal", "blocks", "mi", "mrmr",
}


# --- positional / index helpers (1-based, mirroring R) -----------------------

def _lag_to_position(lags, total):
    return total - np.asarray(lags, dtype=int) + 1


def _position_to_lag(positions, total):
    return total - np.asarray(positions, dtype=int) + 1


def _complete_positions(candidates, total, input_size):
    """Dedupe valid candidates, pad with most-recent-first fillers, sort asc."""
    cand = np.asarray(candidates, dtype=float)
    cand = cand[np.isfinite(cand)].astype(int)
    cand = cand[(cand >= 1) & (cand <= total)]
    selected = list(dict.fromkeys(cand.tolist()))  # unique, order-preserving
    if len(selected) < input_size:
        fillers = [p for p in range(total, 0, -1) if p not in set(selected)]
        selected = selected + fillers
    selected = selected[:min(input_size, len(selected))]
    return np.array(sorted(set(selected)), dtype=int)


def _order_desc(values):
    """R ``order(values, decreasing=TRUE)`` -> 1-based indices, ties asc by index."""
    values = np.asarray(values, dtype=float)
    idx = np.arange(len(values))
    order = np.lexsort((idx, -values))  # primary: -values (desc); tie: idx asc
    return order + 1


# --- series reconstruction & correlation scores ------------------------------

def _reconstruct_series(x, y):
    x = np.asarray(x, dtype=float)
    if x.shape[0] == 0:
        return np.asarray(y, dtype=float)
    return np.concatenate([x[0, :], np.asarray(y, dtype=float).ravel()])


def _score_acf(series, lag_max):
    vals = sm_acf(series, nlags=lag_max, adjusted=False, fft=False)
    return np.asarray(vals[1:], dtype=float)


def _score_pacf(series, lag_max):
    vals = sm_pacf(series, nlags=lag_max, method="ywm")
    return np.asarray(vals[1:], dtype=float)


def _peak_lags(scores):
    """1-based indices of local maxima of |scores| (interior points)."""
    v = np.abs(np.asarray(scores, dtype=float))
    if len(v) <= 2:
        return np.array([], dtype=int)
    peaks = [i + 1 for i in range(1, len(v) - 1) if v[i] >= v[i - 1] and v[i] >= v[i + 1]]
    return np.array(peaks, dtype=int)


def _select_top_scores(scores, total, input_size):
    if len(scores) == 0:
        return _complete_positions(range(total - input_size + 1, total + 1), total, input_size)
    ranked_lags = _order_desc(np.abs(scores))
    positions = _lag_to_position(ranked_lags, total)
    return _complete_positions(positions, total, input_size)


def _seasonal_lags(total, seasonality):
    if not np.isfinite(seasonality) or seasonality < 2 or seasonality > total:
        return np.array([], dtype=int)
    return np.arange(seasonality, total + 1, seasonality, dtype=int)


def _merge_lags(primary, secondary, total, input_size):
    merged = list(dict.fromkeys(list(np.asarray(primary, int)) + list(np.asarray(secondary, int))))
    positions = _lag_to_position(merged, total)
    return _complete_positions(positions, total, input_size)


def _estimate_seasonality(series, fallback=1):
    """Periodogram-based dominant-period estimate (approx. of findfrequency)."""
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 4:
        return int(fallback)
    detr = series - np.polyval(np.polyfit(np.arange(n), series, 1), np.arange(n))
    spec = np.abs(np.fft.rfft(detr)) ** 2
    freqs = np.fft.rfftfreq(n)
    if len(spec) > 1:
        k = 1 + int(np.argmax(spec[1:]))
        if freqs[k] > 0:
            period = int(np.floor(1.0 / freqs[k] + 0.5))
            if period >= 2:
                return period
    return int(fallback)


# --- mutual information helpers ----------------------------------------------

def _discretize(x, bins=8):
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if len(np.unique(finite)) <= 1:
        return np.ones(len(x), dtype=int)
    probs = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(finite, probs, method="median_unbiased"))  # R type 8
    if len(breaks) <= 2:
        ranks = pd.Series(x).rank(method="average").to_numpy()
        edges = np.linspace(ranks.min(), ranks.max(), bins + 1)
        return np.clip(np.searchsorted(edges, ranks, side="left"), 1, bins).astype(int)
    return np.clip(np.searchsorted(breaks, x, side="left"), 1, len(breaks) - 1).astype(int)


def _mutual_information(x, y, bins=8):
    xd = _discretize(x, bins)
    yd = _discretize(y, bins)
    tab = pd.crosstab(xd, yd).to_numpy(dtype=float)
    prob = tab / tab.sum()
    px = prob.sum(axis=1)
    py = prob.sum(axis=0)
    mi = 0.0
    for i in range(prob.shape[0]):
        for j in range(prob.shape[1]):
            pij = prob[i, j]
            if pij > 0 and px[i] > 0 and py[j] > 0:
                mi += pij * np.log(pij / (px[i] * py[j]))
    return mi


class TsLagmap(DalBase):
    """Lag-selection strategy for sliding-window predictors (``ts_lagmap``)."""

    def __init__(
        self,
        method: str = "recent",
        seasonality: int | None = None,
        peak_basis: str = "acf",
        block_radius: int = 1,
        bins: int = 8,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if peak_basis not in ("acf", "pacf"):
            raise ValueError("peak_basis must be 'acf' or 'pacf'.")
        self.method = method
        self.seasonality = seasonality
        self.peak_basis = peak_basis
        self.block_radius = block_radius
        self.bins = bins
        self.positions: np.ndarray = np.array([], dtype=int)  # 0-based
        self.lags: np.ndarray = np.array([], dtype=int)
        self.columns: list = []
        self.seasonality_estimate: int | None = None

    def fit(self, x, y=None, input_size: int | None = None, **kwargs) -> TsLagmap:
        xv = np.asarray(x, dtype=float)
        if xv.ndim == 1:
            xv = xv.reshape(-1, 1)
        if input_size is None or input_size < 1:
            raise ValueError("ts_lagmap requires a positive input_size.")
        total = xv.shape[1]
        if input_size > total:
            raise ValueError("input_size cannot exceed the number of lag columns.")
        if self.method in _NEEDS_TARGET and y is None:
            raise ValueError("this lag-mapping method requires aligned training outputs.")

        series = _reconstruct_series(xv, y) if y is not None else None
        positions = self._select(xv, y, series, total, input_size)  # 1-based, sorted asc

        colnames = list(x.columns) if isinstance(x, pd.DataFrame) else \
            [f"t{lag}" for lag in _position_to_lag(np.arange(1, total + 1), total)]

        self.positions = positions - 1  # expose 0-based for iloc
        self.lags = _position_to_lag(positions, total)
        self.columns = [colnames[p - 1] for p in positions]
        if self.seasonality is None and self.method in ("seasonal", "acf_seasonal", "pacf_seasonal"):
            self.seasonality_estimate = _estimate_seasonality(series)
        return self

    def _season(self, series):
        return self.seasonality if self.seasonality is not None else _estimate_seasonality(series)

    def _select(self, xv, y, series, total, input_size):
        m = self.method
        if m == "recent":
            return _complete_positions(range(total - input_size + 1, total + 1), total, input_size)
        if m == "even":
            pos = np.round(np.linspace(1, total, input_size)).astype(int)
            return _complete_positions(pos, total, input_size)
        if m == "geom":
            lags = np.round(np.exp(np.linspace(np.log(1), np.log(total), input_size))).astype(int)
            return _complete_positions(_lag_to_position(lags, total), total, input_size)
        if m == "acf":
            return _select_top_scores(_score_acf(series, total), total, input_size)
        if m == "pacf":
            return _select_top_scores(_score_pacf(series, total), total, input_size)
        if m == "peaks":
            return self._peaks(series, total, input_size)
        if m == "seasonal":
            lags = _seasonal_lags(total, self._season(series))
            return _complete_positions(_lag_to_position(lags, total), total, input_size)
        if m == "acf_seasonal":
            sl = _seasonal_lags(total, self._season(series))
            return _merge_lags(sl, _order_desc(np.abs(_score_acf(series, total))), total, input_size)
        if m == "pacf_seasonal":
            sl = _seasonal_lags(total, self._season(series))
            return _merge_lags(sl, _order_desc(np.abs(_score_pacf(series, total))), total, input_size)
        if m == "blocks":
            return self._blocks(series, total, input_size)
        if m == "mi":
            return self._mi(xv, y, total, input_size)
        return self._mrmr(xv, y, total, input_size)  # mrmr

    def _peaks(self, series, total, input_size):
        scores = _score_acf(series, total) if self.peak_basis == "acf" else _score_pacf(series, total)
        peaks = _peak_lags(scores)
        if len(peaks) == 0:
            return _select_top_scores(scores, total, input_size)
        peaks = peaks[_order_desc(np.abs(scores[peaks - 1])) - 1]
        return _complete_positions(_lag_to_position(peaks, total), total, input_size)

    def _blocks(self, series, total, input_size):
        scores = _score_acf(series, total) if self.peak_basis == "acf" else _score_pacf(series, total)
        centers = _peak_lags(scores)
        if len(centers) == 0:
            centers = _order_desc(np.abs(scores))
        else:
            centers = centers[_order_desc(np.abs(scores[centers - 1])) - 1]
        block = []
        for c in centers:
            block.extend(range(max(1, c - self.block_radius), min(total, c + self.block_radius) + 1))
        return _complete_positions(_lag_to_position(block, total), total, input_size)

    def _relevance(self, xv, y, total):
        return np.array([_mutual_information(xv[:, j], y, self.bins) for j in range(total)])

    def _mi(self, xv, y, total, input_size):
        relevance = self._relevance(xv, y, total)
        positions = _order_desc(relevance)  # column indices directly (1-based)
        return _complete_positions(positions, total, input_size)

    def _mrmr(self, xv, y, total, input_size):
        relevance = self._relevance(xv, y, total)
        selected: list[int] = []
        candidates = list(range(total))  # 0-based columns
        while len(selected) < min(input_size, total) and candidates:
            if not selected:
                best = candidates[int(np.argmax(relevance[candidates]))]
            else:
                scores = []
                for idx in candidates:
                    redundancy = np.mean([
                        _mutual_information(xv[:, idx], xv[:, sel], self.bins) for sel in selected
                    ])
                    scores.append(relevance[idx] - redundancy)
                best = candidates[int(np.argmax(scores))]
            selected.append(best)
            candidates.remove(best)
        return _complete_positions([s + 1 for s in selected], total, input_size)
