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


# --- ARIMA++ (auto="cv") -------------------------------------------------------

def test_ndiffs_detects_integration():
    from pytspredit.arima import _ndiffs
    rng = np.random.default_rng(0)
    stationary = rng.normal(0, 1, 200)
    random_walk = np.cumsum(rng.normal(0, 1, 200))
    assert _ndiffs(stationary) == 0
    assert _ndiffs(random_walk) >= 1


def test_acf_pacf_candidates_find_seasonal_lag():
    from pytspredit.arima import _acf_pacf_candidates
    rng = np.random.default_rng(1)
    n = 300
    e = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(12, n):
        y[t] = 0.4 * y[t - 1] + 0.5 * y[t - 12] + e[t]  # strong lag-12 structure
    y = y[40:]
    p_cands, q_cands = _acf_pacf_candidates(y, d=0, max_order=12)
    # the ACF/PACF-driven candidates must reach beyond auto.arima's fixed cap of 5
    assert max(p_cands + q_cands) > 5


def test_arima_cv_runs_and_forecasts():
    # small max_order keeps the grid tiny so the CV search is fast
    rng = np.random.default_rng(2)
    n = 90
    y = np.cumsum(rng.normal(0, 1, n)) + 10
    model = ts.TsArima(auto="cv", max_order=3, cv_folds=2, horizon=3).fit(y)
    assert model.auto == "cv"
    assert all(v is not None for v in model._order())
    pred = model.predict(y[-1:], steps_ahead=4)
    assert pred.shape == (4,) and np.isfinite(pred).all()
