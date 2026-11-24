"""Unit tests for the 0.5.0 multivariate family."""
import warnings

import numpy as np
import pandas as pd
import pytest

import pytspredit as ts

warnings.filterwarnings("ignore")


def _mv():
    """Build the canonical tsd-based multivariate data (mirrors the R examples)."""
    y = ts.datasets.tsd()["y"].to_numpy()
    x1 = np.concatenate([y[1:], y[-1:]])
    x2 = pd.Series(y).rolling(3).mean().to_numpy()  # R stats::filter(sides=1)
    mask = np.isnan(x2)
    x2[mask] = y[mask]
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return ts.ts_data_mv(df, y="y")


# -- ts_data_mv / adjust_ts_data_mv --------------------------------------------

def test_ts_data_mv_aligned_metadata():
    mv = _mv()
    assert isinstance(mv, ts.TsDataMv)
    assert mv.mv_y == "y"
    assert mv.mv_x == ["x1", "x2"]
    assert mv.mv_variables == ["y", "x1", "x2"]
    assert mv.mv_representation == "aligned"
    assert list(mv.columns) == ["y", "x1", "x2"]


def test_ts_data_mv_windowed_columns():
    mvw = ts.ts_data_mv(_mv(), sw=3)
    assert mvw.mv_representation == "windowed"
    # three variables x three lags each (t2, t1, t0)
    assert list(mvw.columns) == [
        "y_t2", "y_t1", "y_t0", "x1_t2", "x1_t1", "x1_t0", "x2_t2", "x2_t1", "x2_t0",
    ]


def test_ts_data_mv_requires_valid_target():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError):
        ts.ts_data_mv(df, y="missing")


def test_ts_sample_preserves_mv_metadata():
    samp = ts.ts_sample(_mv(), test_size=5)
    assert isinstance(samp.train, ts.TsDataMv)
    assert samp.train.mv_y == "y"
    assert samp.train.mv_variables == ["y", "x1", "x2"]
    assert samp.test.shape[0] == 5


def test_adjust_ts_data_mv_roundtrip():
    adj = ts.adjust_ts_data_mv(pd.DataFrame({"y": [1.0, 2.0], "x1": [3.0, 4.0]}),
                               y="y", x=["x1"])
    assert adj.mv_representation == "aligned"
    assert adj.mv_variables == ["y", "x1"]


# -- ts_mv_spec ----------------------------------------------------------------

def test_ts_mv_spec_holds_fields():
    spec = ts.ts_mv_spec(ts.TsArima(), variables=["y"], lags={"y": [0, 1]})
    assert isinstance(spec, ts.TsMvSpec)
    assert spec.variables == ["y"]
    assert spec.lags == {"y": [0, 1]}


# -- TsVar ---------------------------------------------------------------------

def test_var_fit_predict_system():
    model = ts.TsVar(p_max=3).fit(ts.ts_sample(_mv(), test_size=5).train)
    pred = model.predict(steps_ahead=5)
    assert isinstance(pred, ts.TsMvPrediction)
    assert pred.shape == (5,)
    assert pred.system.shape == (5, 3)
    assert list(pred.system.columns) == ["y", "x1", "x2"]
    assert np.isfinite(np.asarray(pred)).all()


def test_var_manual_order():
    model = ts.TsVar(p=2).fit(ts.ts_sample(_mv(), test_size=5).train)
    assert model.p == 2
    assert model.predict(steps_ahead=3).shape == (3,)


# -- TsLmMv --------------------------------------------------------------------

def test_lm_mv_known_future():
    samp = ts.ts_sample(_mv(), test_size=5)
    model = ts.TsLmMv(formula="y ~ x1 + x2").fit(samp.train)
    pred = model.predict(samp.test, steps_ahead=5)
    assert pred.shape == (5,)
    assert model.feature_names == ["x1", "x2"]


def test_lm_mv_forecasts_aux_when_unknown():
    samp = ts.ts_sample(_mv(), test_size=3)
    model = ts.TsLmMv(models_x={"x1": ts.TsPersist(), "x2": ts.TsPersist()}).fit(samp.train)
    pred = model.predict(steps_ahead=3)
    assert pred.shape == (3,)
    # persistence auxiliaries -> constant aux paths
    assert np.allclose(pred.prediction_x["x1"], pred.prediction_x["x1"][0])


def test_lm_mv_requires_future_or_models():
    model = ts.TsLmMv().fit(ts.ts_sample(_mv(), test_size=3).train)
    with pytest.raises(ValueError):
        model.predict(steps_ahead=3)


# -- TsArimax ------------------------------------------------------------------

def test_arimax_manual_order_known_future():
    samp = ts.ts_sample(_mv(), test_size=5)
    model = ts.TsArimax(p=2, d=0, q=0).fit(samp.train)
    pred = model.predict(samp.test, steps_ahead=5)
    assert pred.shape == (5,)
    assert np.isfinite(np.asarray(pred)).all()


# -- TsRegswMv -----------------------------------------------------------------

def test_regsw_mv_orchestration():
    samp = ts.ts_sample(_mv(), test_size=5)
    model = ts.TsRegswMv(
        model_y=ts.ts_mv_spec(ts.TsKnn(ts.TsNormGminmax(), k=3),
                              variables=["y", "x1", "x2"]),
        models_x={"x1": ts.ts_mv_spec(ts.TsPeriodic(7)),
                  "x2": ts.ts_mv_spec(ts.TsPeriodic(7))},
        window_size=5).fit(samp.train)
    pred = model.predict(steps_ahead=5)
    assert pred.shape == (5,)
    assert list(pred.system.columns) == ["y", "x1", "x2"]
    assert np.isfinite(np.asarray(pred)).all()


def test_regsw_mv_target_must_be_sliding_window():
    samp = ts.ts_sample(_mv(), test_size=5)
    model = ts.TsRegswMv(
        model_y=ts.ts_mv_spec(ts.TsArima(), variables=["y", "x1", "x2"]),
        models_x={"x1": ts.ts_mv_spec(ts.TsPeriodic(7)),
                  "x2": ts.ts_mv_spec(ts.TsPeriodic(7))},
        window_size=5)
    with pytest.raises(ValueError):
        model.fit(samp.train)


def test_regsw_mv_requires_one_model_per_aux():
    samp = ts.ts_sample(_mv(), test_size=5)
    model = ts.TsRegswMv(
        model_y=ts.ts_mv_spec(ts.TsKnn(ts.TsNormGminmax(), k=3)),
        models_x={"x1": ts.ts_mv_spec(ts.TsPeriodic(7))},  # missing x2
        window_size=5)
    with pytest.raises(ValueError):
        model.fit(samp.train)


# -- R-vs-Python parity --------------------------------------------------------
# Consolidated into the committed-parity guard tests/test_quality_multivariate.py
# (reproducible R reference develop/parity/multivariate.R, + ts_arimax coverage). See
# develop/parity.md.
