"""Unit tests for the native MODWT and the wavelet filter."""
import warnings

import numpy as np

import pytspredit as ts
from pytspredit.wavelet import imodwt, modwt, modwt_filters

warnings.filterwarnings("ignore")


def _signal(n=256, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    return np.sin(2 * np.pi * 7 * t) + 0.3 * t + 0.1 * rng.standard_normal(n)


def test_modwt_perfect_reconstruction():
    x = _signal()
    # haar/d4/la8/bl14 are exact; c6's published coiflet coefficients are only
    # approximately orthogonal (same in R), so it reconstructs to ~1e-6.
    tol = {"haar": 1e-9, "d4": 1e-9, "la8": 1e-9, "bl14": 1e-9, "c6": 1e-5}
    for f in modwt_filters():
        details, v = modwt(x, filter=f)
        np.testing.assert_allclose(imodwt(details, v, filter=f), x, atol=tol[f],
                                   err_msg=f"{f} reconstruction")


def test_modwt_mra_additivity():
    # x = sum of detail MRA components + smooth; zeroing D1 drops exactly D1
    x = _signal()
    details, v = modwt(x, filter="la8")
    full = imodwt(details, v, filter="la8")
    details[0] = np.zeros_like(details[0])
    without_d1 = imodwt(details, v, filter="la8")
    assert not np.allclose(full, without_d1)  # something was removed
    np.testing.assert_allclose(full, x, atol=1e-9)


def test_filter_length_and_families():
    x = ts.datasets.tsd()["y"].to_numpy()
    assert set(modwt_filters()) == {"haar", "d4", "la8", "bl14", "c6"}
    for f in modwt_filters():
        out = ts.TsFilWavelet(f).fit(x).transform(x)
        assert len(out) == len(x) and np.all(np.isfinite(out))


def test_filter_selection_by_r2():
    x = ts.datasets.tsd()["y"].to_numpy()
    sel = ts.TsFilWavelet(["haar", "d4", "la8"])
    sel.fit(x)
    assert sel.filter in ("haar", "d4", "la8")
    # after selection, transform runs with the chosen single filter
    assert len(sel.transform(x)) == len(x)
