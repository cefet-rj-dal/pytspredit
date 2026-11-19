"""Unit tests for the 0.2.0 model zoo: deterministic, DARIMA, WARMA."""
import warnings

import numpy as np
import pytest

import pytspredit as ts

warnings.filterwarnings("ignore")


# -- deterministic --------------------------------------------------------------

def test_periodic_repeats_cycle():
    model = ts.TsPeriodic(7).fit([4, 5, 6, 7, 1, 2, 3])
    np.testing.assert_array_equal(model.predict(steps_ahead=5), [4, 5, 6, 7, 1])


def test_persist_repeats_last_value():
    model = ts.TsPersist().fit([10, 11, 11, 12])
    np.testing.assert_array_equal(model.predict(steps_ahead=3), [12, 12, 12])


def test_persist_rowwise_returns_last_column():
    w = ts.ts_data(ts.datasets.tsd()["y"], 5)
    io = ts.ts_projection(w)
    model = ts.TsPersist().fit(io.input, io.output)
    pred = model.predict(io.input, steps_ahead=1)
    np.testing.assert_allclose(pred, io.input.iloc[:, -1].to_numpy())


def test_periodic_validation():
    with pytest.raises(ValueError):
        ts.TsDeterministic("periodic")  # period required
    with pytest.raises(ValueError):
        ts.TsDeterministic("bogus")


# -- darima / warma -------------------------------------------------------------

def _logistic(n=80, x0=0.3):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = 4 * x[i - 1] * (1 - x[i - 1])
    return x


def _split(series, sw=8, test_size=5):
    w = ts.ts_data(series, sw)
    samp = ts.ts_sample(w, test_size=test_size)
    return ts.ts_projection(samp.train), ts.ts_projection(samp.test)


@pytest.mark.parametrize("model_factory", [
    lambda: ts.TsDarima(ts.TsNormDiff(), input_size=5),
    lambda: ts.TsDarima(ts.TsNormNone(), input_size=5),
    lambda: ts.TsWarma(input_size=5, steps=1),
    lambda: ts.TsWarma(input_size=5, steps=None),
])
def test_darima_warma_fit_predict(model_factory):
    tr, te = _split(_logistic())
    model = model_factory().fit(tr.input, tr.output)
    one = model.predict(te.input)
    multi = model.predict(te.input.iloc[[0]], steps_ahead=5)
    assert one.shape[0] == te.input.shape[0]
    assert multi.shape == (5,)
    assert np.isfinite(one).all() and np.isfinite(multi).all()


def test_darima_is_deterministic():
    tr, te = _split(_logistic())
    a = ts.TsDarima(ts.TsNormNone(), input_size=5).fit(tr.input, tr.output).predict(te.input)
    b = ts.TsDarima(ts.TsNormNone(), input_size=5).fit(tr.input, tr.output).predict(te.input)
    np.testing.assert_allclose(a, b)


def test_warma_auto_steps_selected():
    tr, _ = _split(_logistic())
    model = ts.TsWarma(input_size=5, steps=None).fit(tr.input, tr.output)
    assert model.steps in (0, 1, 2)


def test_warma_steps_validation():
    with pytest.raises(ValueError):
        ts.TsWarma(input_size=5, steps=3)


def test_rlinear_matches_ols_when_full_rank():
    # on a full-rank design, _RLinear == ordinary least squares
    from pytspredit.darima_warma import _RLinear
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = x @ np.array([1.5, -2.0, 0.5]) + 0.1 * rng.normal(size=40)
    coef = _RLinear(intercept=True).fit(x, y).coef_
    xd = np.column_stack([np.ones(40), x])
    ols = np.linalg.lstsq(xd, y, rcond=None)[0]
    np.testing.assert_allclose(coef, ols, atol=1e-9)
