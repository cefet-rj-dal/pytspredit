"""Unit tests for the native EMD/EEMD core and the EMD filters."""
import warnings

import numpy as np

import pytspredit as ts
from pytspredit.emd import eemd, emd

warnings.filterwarnings("ignore")


def _signal(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    return np.sin(2 * np.pi * 5 * t) + 0.4 * np.sin(2 * np.pi * 19 * t) + 0.3 * t, rng


def test_emd_reconstruction_is_exact():
    s, _ = _signal()
    imfs = emd(s)
    assert imfs.shape[0] >= 2
    np.testing.assert_allclose(imfs.sum(axis=0), s, atol=1e-9)


def test_emd_multicomponent_extracts_several_imfs():
    s, _ = _signal()
    assert emd(s).shape[0] >= 3  # two oscillations + trend


def test_eemd_is_reproducible_with_seed():
    s, _ = _signal(n=200)
    a = eemd(s, trials=10, random_state=7)
    b = eemd(s, trials=10, random_state=7)
    np.testing.assert_allclose(a, b, atol=0)
    assert a.shape[1] == len(s)


def test_tsfilemd_preserves_length_and_removes_outlier():
    y = ts.datasets.tsd()["y"].to_numpy()
    yo = y.copy(); yo[8] = 2 * yo[8]
    filt = ts.TsFilEmd().fit(yo).transform(yo)
    assert len(filt) == len(yo)
    assert np.all(np.isfinite(filt))
    dev = lambda s: abs(s[8] - 0.5 * (s[7] + s[9]))
    assert dev(filt) < dev(yo)  # the injected spike is attenuated


def test_tsfilemd_eemd_engine_runs():
    y = ts.datasets.tsd()["y"].to_numpy()
    filt = ts.TsFilEmd(engine="eemd", trials=10, random_state=1).fit(y).transform(y)
    assert len(filt) == len(y) and np.all(np.isfinite(filt))


def test_tsfilremd_preserves_length_and_is_finite():
    s, _ = _signal(n=200)
    filt = ts.TsFilRemd().fit(s).transform(s)
    assert len(filt) == len(s) and np.all(np.isfinite(filt))


def test_tsfilremd_max_imfs_cap():
    s, _ = _signal(n=200)
    # with max_imfs=0 nothing is removed -> identity
    filt = ts.TsFilRemd(max_imfs=0).fit(s).transform(s)
    np.testing.assert_allclose(filt, s, atol=1e-9)
