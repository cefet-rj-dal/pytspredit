"""Unit tests for the ARIMA regressor."""
import warnings

import numpy as np

import pytspredit as ts

warnings.filterwarnings("ignore")


def _raw_split():
    rser = ts.ts_data(ts.datasets.tsd()["y"], 1)
    return ts.ts_sample(rser, test_size=5)


def test_manual_order_forecast():
    sr = _raw_split()
    model = ts.TsArima(p=5, d=0, q=0).fit(sr.train)
    assert model._order() == (5, 0, 0)
    pred = model.predict(sr.test.iloc[[0]], steps_ahead=5)
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
    # tsd is a clean sine -> AR(5) should forecast it very well
    ev = ts.TsReg().evaluate(sr.test.to_numpy().ravel(), pred)
    assert ev["R2"] > 0.9


def test_in_sample_fit_when_x_is_training():
    sr = _raw_split()
    model = ts.TsArima(p=3, d=0, q=0).fit(sr.train)
    fitted = model.predict(sr.train)
    assert fitted.shape[0] == sr.train.shape[0]


def test_auto_order_selects_something():
    sr = _raw_split()
    model = ts.TsArima().fit(sr.train)  # no orders -> auto
    assert not model.manual_order
    assert all(v is not None for v in model._order())
    pred = model.predict(sr.test.iloc[[0]], steps_ahead=5)
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()


def test_rolling_one_step():
    sr = _raw_split()
    model = ts.TsArima(p=2, d=0, q=0).fit(sr.train)
    pred = model.predict(sr.test, steps_ahead=1)  # len(x)=5, steps_ahead=1 -> rolling
    assert pred.shape[0] == sr.test.shape[0]
    assert np.isfinite(pred).all()
