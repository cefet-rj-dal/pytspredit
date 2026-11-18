"""Unit tests for the ML sliding-window regressors."""
import warnings

import numpy as np
import pytest

import pytspredit as ts

warnings.filterwarnings("ignore")


def _split():
    w = ts.ts_data(ts.datasets.tsd()["y"], 10)
    samp = ts.ts_sample(w, test_size=5)
    return ts.ts_projection(samp.train), ts.ts_projection(samp.test)


def _factories():
    return {
        "knn": ts.TsKnn(ts.TsNormGminmax(), 4, ts.TsLagmap("pacf"), k=3),
        "elm": ts.TsElm(ts.TsNormGminmax(), 4, ts.TsLagmap("acf"), nhid=5),
        "rf": ts.TsRf(ts.TsNormGminmax(), 9),
        "svm": ts.TsSvm(ts.TsNormGminmax(), 4, ts.TsLagmap("seasonal", seasonality=4)),
        "mlp": ts.TsMlp(ts.TsNormGminmax(), 4, ts.TsLagmap("even"), size=4, decay=0),
    }


@pytest.mark.parametrize("name", ["knn", "elm", "rf", "svm", "mlp"])
def test_fit_predict_one_step(name):
    tr, te = _split()
    model = _factories()[name].fit(tr.input, tr.output)
    pred = model.predict(te.input)
    assert pred.shape[0] == te.input.shape[0]
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("name", ["knn", "elm", "rf", "svm", "mlp"])
def test_multistep_forecast(name):
    tr, te = _split()
    model = _factories()[name].fit(tr.input, tr.output)
    pred = model.predict(te.input.iloc[[0]], steps_ahead=5)
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()


def test_knn_matches_sklearn_backend():
    from sklearn.neighbors import KNeighborsRegressor
    tr, _ = _split()
    model = ts.TsKnn(ts.TsNormGminmax(), 4, ts.TsLagmap("recent"), k=3).fit(tr.input, tr.output)
    # replicate the backend step on the same normalized/selected inputs
    xt = model.preprocess.transform(tr.input)
    xm = xt.iloc[:, model.input_map.positions]
    ref = KNeighborsRegressor(n_neighbors=3).fit(np.asarray(xm, float),
                                                 np.asarray(model.preprocess.transform(xt, tr.output.to_numpy())))
    np.testing.assert_allclose(model.model.predict(np.asarray(xm, float)),
                               ref.predict(np.asarray(xm, float)))


def test_elm_purelin_is_rng_invariant():
    # with purelin and nhid >= n_lags, ELM reduces to OLS -> independent of the seed
    tr, te = _split()
    a = ts.TsElm(ts.TsNormGminmax(), 4, ts.TsLagmap("recent"), nhid=6,
                 actfun="purelin", random_state=0).fit(tr.input, tr.output)
    b = ts.TsElm(ts.TsNormGminmax(), 4, ts.TsLagmap("recent"), nhid=6,
                 actfun="purelin", random_state=123).fit(tr.input, tr.output)
    np.testing.assert_allclose(a.predict(te.input), b.predict(te.input), atol=1e-6)


def test_elm_invalid_actfun():
    tr, _ = _split()
    with pytest.raises(ValueError):
        ts.TsElm(ts.TsNormGminmax(), 4, nhid=3, actfun="bogus").fit(tr.input, tr.output)
