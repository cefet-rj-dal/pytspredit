"""Unit tests for the augmentation family."""
import warnings

import numpy as np

import pytspredit as ts

warnings.filterwarnings("ignore")


def _windows():
    return ts.ts_data(ts.datasets.tsd()["y"], 10)


def test_none_is_identity():
    w = _windows()
    out = ts.TsAugNone().fit(w).transform(w)
    np.testing.assert_allclose(np.asarray(out, float), np.asarray(w, float))


def test_preserve_data_stacks_original_on_top():
    w = _windows()
    for aug in [ts.TsAugFlip(), ts.TsAugStretch(), ts.TsAugShrink(), ts.TsAugJitter()]:
        out = np.asarray(aug.fit(w).transform(w), float)
        assert out.shape[0] == 2 * w.shape[0]
        np.testing.assert_allclose(out[: w.shape[0]], np.asarray(w, float))  # originals first


def test_flip_mirrors_around_row_mean():
    w = _windows()
    x = w.to_numpy(float)
    out = np.asarray(ts.TsAugFlip().fit(w).transform(w), float)
    aug = out[w.shape[0]:]
    np.testing.assert_allclose(aug, 2 * x.mean(axis=1)[:, None] - x, atol=1e-9)


def test_stretch_amplifies_and_keeps_target():
    w = _windows()
    x = w.to_numpy(float)
    aug = np.asarray(ts.TsAugStretch(scale_factor=1.5).fit(w).transform(w), float)[w.shape[0]:]
    dev = (x - x.mean(axis=1)[:, None]) * 1.5
    dev[:, -1] = 0.0
    np.testing.assert_allclose(aug, x + dev, atol=1e-9)
    np.testing.assert_allclose(aug[:, -1], x[:, -1], atol=1e-9)  # target unchanged


def test_jitter_target_unchanged_and_reproducible():
    w = _windows()
    x = w.to_numpy(float)
    a = np.asarray(ts.TsAugJitter(random_state=0).fit(w).transform(w), float)[w.shape[0]:]
    b = np.asarray(ts.TsAugJitter(random_state=0).fit(w).transform(w), float)[w.shape[0]:]
    np.testing.assert_allclose(a, b)                         # reproducible with a seed
    np.testing.assert_allclose(a[:, -1], x[:, -1], atol=1e-9)  # target column untouched


def test_wormhole_shape():
    w = _windows()
    out = np.asarray(ts.TsAugWormhole().fit(w).transform(w), float)
    assert out.shape == (2 * w.shape[0] - 1, w.shape[1])


def test_awareness_shapes_and_preserve():
    w = _windows()
    for aug in [ts.TsAugAwareness(factor=1), ts.TsAugAwaresmooth(factor=1)]:
        out = np.asarray(aug.fit(w).transform(w), float)
        assert out.shape == (2 * w.shape[0], w.shape[1])
        np.testing.assert_allclose(out[: w.shape[0]], np.asarray(w, float))
        assert np.isfinite(out).all()


def test_augmenter_plugs_into_integtune():
    w = _windows()
    samp = ts.ts_sample(w, test_size=5)
    tr = ts.ts_projection(samp.train)
    itune = ts.TsIntegtune(
        input_size=[3, 4],
        base_model=ts.TsElm(),
        folds=4,
        ranges={"nhid": [2, 3], "actfun": ["purelin"]},
        preprocess=[ts.TsNormGminmax()],
        augment=[ts.TsAugNone(), ts.TsAugStretch()],
        seed=0,
    )
    best = itune.fit(tr.input, tr.output)
    assert itune.best_params["augment"] in (0, 1)
    assert np.isfinite(best.predict(tr.input)).all()
