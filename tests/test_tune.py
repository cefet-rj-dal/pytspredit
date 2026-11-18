"""Unit tests for TsTune / TsIntegtune."""
import warnings

import numpy as np

import pytspredit as ts

warnings.filterwarnings("ignore")


def _split():
    w = ts.ts_data(ts.datasets.tsd()["y"], 10)
    samp = ts.ts_sample(w, test_size=5)
    return ts.ts_projection(samp.train), ts.ts_projection(samp.test)


def test_tstune_grid_search_selects_and_refits():
    tr, te = _split()
    tune = ts.TsTune(
        input_size=[3, 4, 5],
        base_model=ts.TsElm(ts.TsNormGminmax()),
        folds=5,
        ranges={"nhid": [1, 2, 3, 4, 5], "actfun": ["purelin"]},
        seed=0,
    )
    best = tune.fit(tr.input, tr.output)
    assert tune.best_params["input_size"] in (3, 4, 5)
    assert tune.hyperparameters is not None and len(tune.hyperparameters) > 1
    pred = best.predict(te.input.iloc[[0]], steps_ahead=5)
    assert pred.shape == (5,)
    # purelin ELM on a clean sine tunes to a near-perfect fit
    assert best.evaluate(te.output.to_numpy(), pred)["R2"] > 0.9


def test_tstune_single_config():
    tr, _ = _split()
    tune = ts.TsTune(input_size=[4], base_model=ts.TsKnn(ts.TsNormGminmax(), k=3),
                     folds=5, ranges=None, seed=0)
    best = tune.fit(tr.input, tr.output)
    assert tune.best_params == {"input_size": 4}
    assert len(tune.hyperparameters) == 1
    assert np.isfinite(best.predict(tr.input)).all()


def test_integtune_varies_preprocess():
    tr, te = _split()
    it = ts.TsIntegtune(
        input_size=[3, 4],
        base_model=ts.TsElm(),
        folds=5,
        ranges={"nhid": [2, 3], "actfun": ["purelin"]},
        preprocess=[ts.TsNormGminmax(), ts.TsNormSwminmax()],
        seed=0,
    )
    best = it.fit(tr.input, tr.output)
    assert "preprocess" in it.best_params and "augment" in it.best_params
    assert it.best_params["preprocess"] in (0, 1)
    pred = best.predict(te.input.iloc[[0]], steps_ahead=5)
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
