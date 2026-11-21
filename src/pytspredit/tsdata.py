"""Sliding-window time series data structures.

Python counterpart of the R ``ts_data.R`` and ``ts_projection.R`` files.

A *ts_data representation* is a :class:`pandas.DataFrame` whose columns are the
lagged observations of a sliding window, named ``t{sw-1}, ..., t1, t0`` from
oldest to most recent. ``t0`` is always the most recent value (the one-step
target when the window is projected). With ``sw = 1`` the series is stored as a
single ``t0`` column.

We keep the representation a plain :class:`~pandas.DataFrame` (rather than a
subclass) so downstream normalization and modeling code can do ordinary pandas
arithmetic and indexing on it. Window size is simply ``ts.shape[1]``.

References:
    - Box GEP, Jenkins GM, Reinsel GC, Ljung GM (2015). Time Series Analysis:
      Forecasting and Control, 5th ed. Wiley. (sliding-window / lag framing)

Porting:
    Mirrors ``ts_data.R`` and ``ts_projection.R``. Fidelity: exact.
    - Column naming ``t{sw-1}..t0``, window building, projection and sampling
      verified against R (dims/columns/values identical on ``tsd``).
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "TsProjection",
    "TsSample",
    "adjust_ts_data",
    "ts_data",
    "ts_head",
    "ts_projection",
    "ts_sample",
]

#: Result of :func:`ts_projection`: input features and output target.
TsProjection = namedtuple("TsProjection", ["input", "output"])

#: Result of :func:`ts_sample`: train and test partitions.
TsSample = namedtuple("TsSample", ["train", "test"])


def _lag_columns(ncol: int) -> list[str]:
    """Column names ``t{ncol-1} ... t0`` (oldest to most recent)."""
    return [f"t{k}" for k in range(ncol - 1, -1, -1)]


def ts_data(y, sw: int = 1) -> pd.DataFrame:
    """Build a sliding-window representation of a time series.

    Parameters
    ----------
    y:
        Numeric vector (raw series) or a matrix already organized in windows.
    sw:
        Sliding-window size (number of lag columns). ``sw = 1`` keeps the series
        as a single ``t0`` column; ``sw > 1`` produces lagged windows
        ``t{sw-1} ... t0`` with one row per window.

    Returns
    -------
    pandas.DataFrame
        Window matrix with ``sw`` columns named ``t{sw-1} ... t0``. For a series
        of length ``n`` and ``sw > 1`` there are ``n - sw + 1`` rows, where row
        ``i`` holds ``y[i], y[i+1], ..., y[i+sw-1]``.
    """
    if not isinstance(sw, (int, np.integer)) or sw < 1:
        raise ValueError("sw must be a positive integer.")
    sw = int(sw)

    arr = np.asarray(y, dtype=float)
    if sw > 1:
        a = arr.ravel()
        if sw > a.shape[0]:
            raise ValueError("sw cannot be greater than the series length.")
        window = sliding_window_view(a, sw)  # row i -> a[i : i + sw]
    else:
        window = arr.reshape(-1, 1) if arr.ndim == 1 else arr

    window = np.asarray(window, dtype=float)
    return pd.DataFrame(window, columns=_lag_columns(window.shape[1]))


def adjust_ts_data(data) -> pd.DataFrame:
    """Coerce a compatible object into a ts_data :class:`~pandas.DataFrame`.

    Sets the canonical ``t{k}`` column names. Mirrors R's ``adjust_ts_data``.
    """
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(np.asarray(data, dtype=float))
    df = df.copy()
    df.columns = _lag_columns(df.shape[1])
    return df


def ts_head(x, n: int = 6):
    """Return the first ``n`` rows of a ts_data representation."""
    return adjust_ts_data(x).head(n) if not isinstance(x, pd.DataFrame) else x.head(n)


def ts_projection(ts) -> TsProjection:
    """Split a ts_data window matrix into input features and output target.

    All columns except the last (``t0``) are the input features; the last column
    (``t0``) is the one-step-ahead output. Mirrors R's ``ts_projection``.

    Returns
    -------
    TsProjection
        Named tuple ``(input, output)`` where ``input`` is a
        :class:`~pandas.DataFrame` of the lag features and ``output`` is a
        :class:`~pandas.Series` of the ``t0`` target.
    """
    ts = ts if isinstance(ts, pd.DataFrame) else adjust_ts_data(ts)
    if ts.shape[1] < 2:
        raise ValueError("ts_projection needs at least 2 lag columns (sw >= 2).")
    input_ = ts.iloc[:, :-1]
    output = ts.iloc[:, -1]
    return TsProjection(input=input_, output=output)


def ts_sample(ts, test_size: int = 1, offset: int = 0) -> TsSample:
    """Split a ts_data representation into train and test partitions.

    Extracts ``test_size`` rows from the end (minus an optional ``offset``) as
    the test set; the remaining initial rows form the training set. Mirrors R's
    ``ts_sample``.
    """
    ts = ts if isinstance(ts, pd.DataFrame) else adjust_ts_data(ts)
    n = ts.shape[0]
    split = n - test_size - offset
    if split < 1:
        raise ValueError("test_size (+ offset) is too large for the number of rows.")
    train = ts.iloc[:split].reset_index(drop=True)
    test = ts.iloc[split:split + test_size].reset_index(drop=True)
    return TsSample(train=train, test=test)
