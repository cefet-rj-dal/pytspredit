"""pytspredit -- Time series prediction with tunable preprocessing.

Python port of the R ``tspredit`` framework, built on top of ``pydaltoolbox``
(the Experiment Line foundation). It provides a uniform, extensible interface for
time series forecasting: sliding-window modeling, normalization, filtering,
augmentation, machine-learning and statistical regressors, and tuning -- wrapping
scikit-learn / numpy / pandas / statsmodels with thin adapters.
"""
from __future__ import annotations

from . import datasets
from .lagmap import TsLagmap
from .norm import (
    TsNormAn,
    TsNormDiff,
    TsNormGminmax,
    TsNormNone,
    TsNormSwminmax,
)
from .ts_reg import TsReg, TsRegSW, mse_ts, r2_ts, smape_ts
from .tsdata import (
    TsProjection,
    TsSample,
    adjust_ts_data,
    ts_data,
    ts_head,
    ts_projection,
    ts_sample,
)

__version__ = "0.1.0"

__all__ = [
    "TsLagmap",
    "TsNormAn",
    "TsNormDiff",
    "TsNormGminmax",
    "TsNormNone",
    "TsNormSwminmax",
    "TsProjection",
    "TsReg",
    "TsRegSW",
    "TsSample",
    "adjust_ts_data",
    "datasets",
    "mse_ts",
    "r2_ts",
    "smape_ts",
    "ts_data",
    "ts_head",
    "ts_projection",
    "ts_sample",
]
