"""pytspredit -- Time series prediction with tunable preprocessing.

Python port of the R ``tspredit`` framework, built on top of ``pydaltoolbox``
(the Experiment Line foundation). It provides a uniform, extensible interface for
time series forecasting: sliding-window modeling, normalization, filtering,
augmentation, machine-learning and statistical regressors, and tuning -- wrapping
scikit-learn / numpy / pandas / statsmodels with thin adapters.
"""
from __future__ import annotations

from . import datasets
from .arima import TsArima, fit_arima
from .augment import (
    TsAugAwareness,
    TsAugAwaresmooth,
    TsAugFlip,
    TsAugJitter,
    TsAugNone,
    TsAugShrink,
    TsAugStretch,
    TsAugWormhole,
)
from .benchmark import TsPredBench
from .darima_warma import TsDarima, TsWarma
from .deterministic import TsDeterministic, TsPeriodic, TsPersist
from .filters import (
    TsFilEma,
    TsFilEmd,
    TsFilFft,
    TsFilGabor,
    TsFilHp,
    TsFilKalman,
    TsFilLowess,
    TsFilMa,
    TsFilNone,
    TsFilQes,
    TsFilRecursive,
    TsFilRemd,
    TsFilSeasAdj,
    TsFilSes,
    TsFilSmooth,
    TsFilSpline,
    TsFilWavelet,
    TsFilWinsor,
)
from .lagmap import TsLagmap
from .multivariate import (
    TsArimax,
    TsDataMv,
    TsLmMv,
    TsMvPrediction,
    TsMvSpec,
    TsRegMv,
    TsRegswMv,
    TsVar,
    adjust_ts_data_mv,
    ts_data_mv,
    ts_mv_spec,
)
from .norm import (
    TsNormAn,
    TsNormDiff,
    TsNormGminmax,
    TsNormNone,
    TsNormSwminmax,
)
from .plot import plot_series, plot_ts, plot_ts_pred, plot_ts_pred_mv
from .regressors import TsElm, TsKnn, TsMlp, TsRf, TsSvm
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
from .tune import TsIntegtune, TsTune

__version__ = "0.12.0"

__all__ = [
    "TsArima",
    "TsArimax",
    "TsAugAwareness",
    "TsAugAwaresmooth",
    "TsAugFlip",
    "TsAugJitter",
    "TsAugNone",
    "TsAugShrink",
    "TsAugStretch",
    "TsAugWormhole",
    "TsDarima",
    "TsDataMv",
    "TsDeterministic",
    "TsElm",
    "TsFilEma",
    "TsFilEmd",
    "TsFilFft",
    "TsFilGabor",
    "TsFilHp",
    "TsFilKalman",
    "TsFilLowess",
    "TsFilMa",
    "TsFilNone",
    "TsFilQes",
    "TsFilRecursive",
    "TsFilRemd",
    "TsFilSeasAdj",
    "TsFilSes",
    "TsFilSmooth",
    "TsFilSpline",
    "TsFilWavelet",
    "TsFilWinsor",
    "TsIntegtune",
    "TsKnn",
    "TsLagmap",
    "TsLmMv",
    "TsMlp",
    "TsMvPrediction",
    "TsMvSpec",
    "TsNormAn",
    "TsNormDiff",
    "TsNormGminmax",
    "TsNormNone",
    "TsNormSwminmax",
    "TsPeriodic",
    "TsPersist",
    "TsPredBench",
    "TsProjection",
    "TsReg",
    "TsRegMv",
    "TsRegSW",
    "TsRegswMv",
    "TsRf",
    "TsSample",
    "TsSvm",
    "TsTune",
    "TsVar",
    "TsWarma",
    "adjust_ts_data",
    "adjust_ts_data_mv",
    "datasets",
    "fit_arima",
    "mse_ts",
    "plot_series",
    "plot_ts",
    "plot_ts_pred",
    "plot_ts_pred_mv",
    "r2_ts",
    "smape_ts",
    "ts_data",
    "ts_data_mv",
    "ts_head",
    "ts_mv_spec",
    "ts_projection",
    "ts_sample",
]
