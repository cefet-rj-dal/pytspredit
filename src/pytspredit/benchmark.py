"""Time-series forecasting benchmark harness.

:class:`TsPredBench` runs a grid of *datasets x models*, fitting each model on the
training windows of each univariate series and evaluating a multi-step forecast on
the held-out tail, returning a tidy table of error metrics (MSE / sMAPE / R2).

It reuses the package's canonical Experiment Line path end to end
(``ts_data`` -> ``ts_sample`` -> ``ts_projection`` -> ``fit`` -> multi-step
``predict`` -> metrics), so a benchmark cell behaves exactly like a hand-written
single-model evaluation.

Porting:
    No R counterpart. The R ``tspredbench.R`` is dataset *documentation* only (the
    collection has no harness class); ``TsPredBench`` is a Python-native
    benchmarking convenience built on the Experiment Line. Fidelity: n/a (new
    interface, not a port of R behavior).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .ts_reg import mse_ts, r2_ts, smape_ts
from .tsdata import ts_data, ts_projection, ts_sample

__all__ = ["TsPredBench"]


class TsPredBench:
    """Grid benchmark of univariate forecasting models over several datasets.

    Parameters
    ----------
    datasets:
        Mapping ``name -> series``, where ``series`` is a 1-D array-like
        (``numpy.ndarray``, ``pandas.Series``, or ``list``) of numeric values.
    models:
        Mapping ``name -> model``. Each ``model`` is either a **factory**
        (a zero-argument callable returning a fresh predictor, recommended, so no
        state leaks between datasets) or a ready predictor **instance** (re-fit on
        each dataset). A predictor is any object exposing the Experiment Line
        ``fit(x, y)`` / ``predict(x, steps_ahead=...)`` contract (e.g. ``TsKnn``,
        ``TsMlp``, ``TsArima``, ``TsDarima``, ...).
    input_size:
        Number of lagged inputs per window (the sliding window has
        ``input_size + 1`` columns: ``input_size`` lags plus the target ``t0``).
    steps_ahead:
        Forecast horizon. The model produces a single multi-step forecast of this
        length from the first held-out window.
    test_size:
        Number of trailing windows held out for evaluation. Defaults to
        ``steps_ahead`` (the forecast is compared against the held-out tail).
    raise_on_error:
        If ``False`` (default), a model that fails on a dataset yields a row with
        ``nan`` metrics and the error message in ``error``; if ``True``, the
        exception propagates.

    Examples
    --------
    >>> import pytspredit as ts
    >>> bench = ts.TsPredBench(
    ...     datasets={"tsd": ts.datasets.tsd()["y"]},
    ...     models={
    ...         "knn": lambda: ts.TsKnn(ts.TsNormGminmax(), input_size=5),
    ...         "persist": lambda: ts.TsPersist(),
    ...     },
    ...     input_size=5, steps_ahead=5,
    ... )
    >>> res = bench.run()  # doctest: +SKIP
    """

    def __init__(
        self,
        datasets: Mapping[str, Any],
        models: Mapping[str, Any],
        input_size: int = 5,
        steps_ahead: int = 5,
        test_size: int | None = None,
        raise_on_error: bool = False,
    ) -> None:
        self.datasets = dict(datasets)
        self.models = dict(models)
        self.input_size = int(input_size)
        self.steps_ahead = int(steps_ahead)
        self.test_size = int(test_size) if test_size is not None else int(steps_ahead)
        self.raise_on_error = bool(raise_on_error)

    @staticmethod
    def _as_series(series: Any) -> np.ndarray:
        return np.asarray(series, dtype=float).ravel()

    @staticmethod
    def _make_model(model: Any) -> Any:
        return model() if isinstance(model, Callable) else model

    def _evaluate_cell(self, series: np.ndarray, model: Any) -> dict:
        window = ts_data(series, sw=self.input_size + 1)
        sample = ts_sample(window, test_size=self.test_size)
        io_train = ts_projection(sample.train)
        io_test = ts_projection(sample.test)
        actual = io_test.output.to_numpy().ravel()[: self.steps_ahead]

        mdl = self._make_model(model)
        mdl.fit(io_train.input, io_train.output)
        prediction = np.asarray(
            mdl.predict(io_test.input.iloc[[0]], steps_ahead=self.steps_ahead)
        ).ravel()

        return {
            "mse": mse_ts(actual, prediction),
            "smape": smape_ts(actual, prediction),
            "r2": r2_ts(actual, prediction),
        }

    def run(self) -> pd.DataFrame:
        """Run the full grid and return a tidy ``DataFrame``.

        Columns: ``dataset``, ``model``, ``mse``, ``smape``, ``r2`` (and ``error``
        when ``raise_on_error`` is ``False`` and a cell failed). One row per
        (dataset, model) pair, in insertion order.
        """
        rows: list[dict] = []
        for dname, series in self.datasets.items():
            s = self._as_series(series)
            for mname, model in self.models.items():
                row = {"dataset": dname, "model": mname}
                try:
                    row.update(self._evaluate_cell(s, model))
                except Exception as exc:
                    if self.raise_on_error:
                        raise
                    row.update({"mse": np.nan, "smape": np.nan, "r2": np.nan,
                                "error": f"{type(exc).__name__}: {exc}"})
                rows.append(row)

        columns = ["dataset", "model", "mse", "smape", "r2"]
        if any("error" in r for r in rows):
            columns.append("error")
        return pd.DataFrame(rows, columns=columns)
