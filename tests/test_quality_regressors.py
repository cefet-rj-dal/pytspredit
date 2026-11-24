"""R-vs-Python parity guard for the forecast regressors (retroactive M3 backfill).

The invariant is the 5-step-ahead predicted vector on the shared tsd split, with three
fidelity classes (see develop/parity.md):
  * deterministic (knn; elm purelin nhid==n_lags == OLS) -> value-wise <= 1e-9
  * backend (svm = e1071 SVR; arima = stats::arima MLE vs statsmodels) -> near-exact tol
  * stochastic (rf = randomForest; mlp = nnet) -> loose: finite + tracks the series
    (correlation with the actual future >= 0.8), since the RNG/backends differ by design
    (sMAPE is unusable here -- the sine crosses zero, collapsing its denominator).
Reference develop/parity/regressors_ref.json is produced by develop/parity/regressors.R.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import pytspredit as ts

warnings.filterwarnings("ignore")

_REF = Path(__file__).resolve().parents[1] / "develop" / "parity" / "regressors_ref.json"


def _load():
    with open(_REF, encoding="utf-8") as fh:
        return json.load(fh)


# sliding-window models: builder + value-wise tolerance (None => loose/stochastic guard)
_SW_MODELS = {
    "knn": (lambda: ts.TsKnn(ts.TsNormGminmax(), 4, ts.TsLagmap("pacf"), k=3), 1e-9),
    "elm_ols": (lambda: ts.TsElm(ts.TsNormGminmax(), 4, ts.TsLagmap("recent"),
                                 nhid=4, actfun="purelin"), 1e-9),
    "svm": (lambda: ts.TsSvm(ts.TsNormGminmax(), 4, ts.TsLagmap("seasonal", seasonality=4)), 1e-3),
    "rf": (lambda: ts.TsRf(ts.TsNormGminmax(), 9), None),
    "mlp": (lambda: ts.TsMlp(ts.TsNormGminmax(), 4, ts.TsLagmap("even"), size=4, decay=0), None),
}


def _corr(a, p):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(p, float))[0, 1])


def _sw_split(ref):
    w = ts.ts_data(np.asarray(ref["input"], float), int(ref["sw"]))
    samp = ts.ts_sample(w, test_size=int(ref["test_size"]))
    return ts.ts_projection(samp.train), ts.ts_projection(samp.test)


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/regressors.R")
@pytest.mark.parametrize("name", list(_SW_MODELS))
def test_regressor_matches_r(name):
    ref = _load()
    tr, te = _sw_split(ref)
    builder, tol = _SW_MODELS[name]
    model = builder().fit(tr.input, tr.output)
    got = np.asarray(model.predict(te.input.iloc[[0]], steps_ahead=int(ref["test_size"])), float)
    expected = np.asarray(ref["pred"][name], float)
    assert got.shape == expected.shape
    if tol is not None:
        np.testing.assert_allclose(got, expected, atol=tol, rtol=0,
                                   err_msg=f"{name}: forecast differs from R beyond {tol}")
    else:
        # stochastic: RNG/backend differ -> require the forecast to track the series
        assert np.isfinite(got).all()
        assert _corr(ref["actual"], got) >= 0.8, f"{name}: forecast does not track the series"


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/regressors.R")
def test_arima_matches_r():
    # arima is a ts_reg on the RAW series (sw=1), not a sliding-window model
    ref = _load()
    rw = ts.ts_data(np.asarray(ref["input"], float), 1)
    rs = ts.ts_sample(rw, test_size=int(ref["test_size"]))
    am = ts.TsArima(5, 0, 0).fit(rs.train)
    got = np.asarray(am.predict(rs.test.iloc[[0]], steps_ahead=int(ref["test_size"])), float)
    expected = np.asarray(ref["pred"]["arima"], float)
    np.testing.assert_allclose(got, expected, atol=5e-3, rtol=0,
                               err_msg="arima: stats::arima vs statsmodels beyond backend tol")
