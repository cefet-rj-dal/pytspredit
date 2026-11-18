"""Unit tests for the regression base classes (TsReg / TsRegSW)."""
import numpy as np

import pytspredit as ts


def test_metrics_match_hand_computation():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.1, 1.9, 3.2, 3.7])
    ev = ts.TsReg().evaluate(vals, pred)
    assert abs(ev["mse"] - np.mean((vals - pred) ** 2)) < 1e-12
    sse = np.sum((pred - vals) ** 2)
    sst = np.sum((vals - vals.mean()) ** 2)
    assert abs(ev["R2"] - (1 - sse / sst)) < 1e-12
    assert list(ev["metrics"].columns) == ["mse", "smape", "R2"]


def test_smape_skips_zero_denominator():
    # a point where actual=pred=0 has zero denominator and must be dropped
    vals = np.array([0.0, 2.0])
    pred = np.array([0.0, 3.0])
    ev = ts.TsReg().evaluate(vals, pred)
    expected = (abs(2 - 3) / ((2 + 3) / 2)) / 2  # divided by n=2
    assert abs(ev["smape"] - expected) < 1e-12


def test_r2_nan_for_constant_target():
    ev = ts.TsReg().evaluate([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])
    assert np.isnan(ev["R2"])


def test_baseline_predict_returns_last_column():
    w = ts.ts_data(ts.datasets.tsd()["y"], 5)
    pred = ts.TsReg().predict(w)
    np.testing.assert_allclose(pred, w["t0"].to_numpy())


class _MeanBackend(ts.TsRegSW):
    """Trivial backend (predicts the mean of the selected lags) to exercise TsRegSW."""

    def _do_fit(self, x, y):
        return self

    def _do_predict(self, x):
        return np.asarray(x, dtype=float).mean(axis=1)


def test_tsregsw_fit_predict_one_step():
    io = ts.ts_projection(ts.ts_data(ts.datasets.tsd()["y"], 10))
    model = _MeanBackend(ts.TsNormGminmax(), input_size=4, input_map=ts.TsLagmap("recent"))
    model.fit(io.input, io.output)
    pred = model.predict(io.input)
    assert pred.shape[0] == io.input.shape[0]
    assert np.isfinite(pred).all()


def test_tsregsw_multistep_forecast():
    series = ts.datasets.tsd()["y"]
    w = ts.ts_data(series, 10)
    samp = ts.ts_sample(w, test_size=5)
    io_train = ts.ts_projection(samp.train)
    io_test = ts.ts_projection(samp.test)
    model = _MeanBackend(ts.TsNormGminmax(), input_size=4, input_map=ts.TsLagmap("recent"))
    model.fit(io_train.input, io_train.output)
    pred = model.predict(io_test.input.iloc[[0]], steps_ahead=5)
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
