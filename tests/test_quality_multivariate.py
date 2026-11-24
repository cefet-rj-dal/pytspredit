"""R-vs-Python parity guard for the 0.5 multivariate family (retroactive M3 backfill).

Invariant = the target (y) 5-step forecast on the canonical tsd-based MV system
(x1 = shifted y; x2 = 3-point moving average, NA->y). TsVar (native OLS), TsLmMv and
TsRegswMv (orchestration + _RLinear/knn) are value-wise exact; TsArimax is backend
(stats::arima + xreg vs statsmodels). Reference develop/parity/multivariate_ref.json is
produced by develop/parity/multivariate.R. See develop/parity.md.

Generating the reference uncovered and fixed a real R bug: stats::arima stored the `xreg`
symbol in its call, so predict.Arima's eval.parent(call$xreg) crashed with
"object 'xreg' not found", breaking every ts_arimax forecast (fixed in R arima_fit by
embedding the regressor matrix value in the call).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pytspredit as ts

warnings.filterwarnings("ignore")

_REF = Path(__file__).resolve().parents[1] / "develop" / "parity" / "multivariate_ref.json"


def _load():
    with open(_REF, encoding="utf-8") as fh:
        return json.load(fh)


def _mv():
    y = ts.datasets.tsd()["y"].to_numpy()
    x1 = np.concatenate([y[1:], y[-1:]])
    x2 = pd.Series(y).rolling(3).mean().to_numpy()  # R stats::filter(sides=1)
    mask = np.isnan(x2)
    x2[mask] = y[mask]
    return ts.ts_data_mv(pd.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y")


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/multivariate.R")
def test_var_matches_r():
    ref = _load()
    samp = ts.ts_sample(_mv(), test_size=int(ref["test_size"]))
    got = np.asarray(ts.TsVar(p_max=3).fit(samp.train).predict(steps_ahead=int(ref["test_size"])), float)
    np.testing.assert_allclose(got, np.asarray(ref["pred"]["var"], float), atol=1e-6)


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/multivariate.R")
def test_lm_mv_matches_r():
    ref = _load()
    samp = ts.ts_sample(_mv(), test_size=int(ref["test_size"]))
    model = ts.TsLmMv(formula="y ~ x1 + x2").fit(samp.train)
    got = np.asarray(model.predict(samp.test, steps_ahead=int(ref["test_size"])), float)
    np.testing.assert_allclose(got, np.asarray(ref["pred"]["lm_mv"], float), atol=1e-7)


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/multivariate.R")
def test_regsw_mv_matches_r():
    ref = _load()
    samp = ts.ts_sample(_mv(), test_size=int(ref["test_size"]))
    model = ts.TsRegswMv(
        model_y=ts.ts_mv_spec(ts.TsKnn(ts.TsNormGminmax(), k=3), variables=["y", "x1", "x2"]),
        models_x={"x1": ts.ts_mv_spec(ts.TsPeriodic(7)),
                  "x2": ts.ts_mv_spec(ts.TsPeriodic(7))},
        window_size=5).fit(samp.train)
    got = np.asarray(model.predict(steps_ahead=int(ref["test_size"])), float)
    np.testing.assert_allclose(got, np.asarray(ref["pred"]["regsw_mv"], float), atol=1e-6)


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/multivariate.R")
def test_arimax_matches_r():
    # backend tolerance: stats::arima + xreg vs statsmodels ARIMA + exog (~1.4e-1)
    ref = _load()
    samp = ts.ts_sample(_mv(), test_size=int(ref["test_size"]))
    model = ts.TsArimax(p=2, d=0, q=0).fit(samp.train)
    got = np.asarray(model.predict(samp.test, steps_ahead=int(ref["test_size"])), float)
    np.testing.assert_allclose(got, np.asarray(ref["pred"]["arimax"], float), atol=3e-1)
