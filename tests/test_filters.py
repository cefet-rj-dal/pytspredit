"""Unit tests for the filter family."""
import warnings

import numpy as np
import pytest

import pytspredit as ts

warnings.filterwarnings("ignore")


def _noisy():
    y = ts.datasets.tsd()["y"].to_numpy().copy()
    y[8] = 2 * y[8]
    return y


ALL = {
    "none": ts.TsFilNone(), "ma": ts.TsFilMa(3), "ema": ts.TsFilEma(3),
    "winsor": ts.TsFilWinsor(), "recursive": ts.TsFilRecursive(0.05),
    "fft": ts.TsFilFft(), "smooth": ts.TsFilSmooth(), "hp": ts.TsFilHp(100),
    "gabor": ts.TsFilGabor(3, 0.1), "lowess": ts.TsFilLowess(0.2),
    "ses": ts.TsFilSes(), "qes": ts.TsFilQes(), "seas_adj": ts.TsFilSeasAdj(12),
    "spline": ts.TsFilSpline(0.5), "kalman": ts.TsFilKalman(),
}


@pytest.mark.parametrize("name", list(ALL))
def test_length_preserved(name):
    y = _noisy()
    out = np.asarray(ALL[name].fit(y).transform(y), float)
    assert len(out) == len(y)


def test_none_identity():
    y = _noisy()
    np.testing.assert_allclose(np.asarray(ts.TsFilNone().fit(y).transform(y), float), y)


def test_ma_is_rolling_mean_with_nan_warmup():
    y = np.arange(1.0, 8.0)
    out = ts.TsFilMa(3).fit(y).transform(y)
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [2, 3, 4, 5, 6])  # mean of 3 consecutive


def test_winsor_clips_to_percentiles():
    y = _noisy()
    out = ts.TsFilWinsor().fit(y).transform(y)
    lo, hi = np.quantile(y, [0.05, 0.95])
    assert out.min() >= lo - 1e-9 and out.max() <= hi + 1e-9


def test_recursive_recursion():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = ts.TsFilRecursive(0.5).fit(y).transform(y)
    expected = np.empty(4)
    prev = 0.0
    for i in range(4):
        prev = y[i] + 0.5 * prev
        expected[i] = prev
    np.testing.assert_allclose(out, expected)


def test_hp_preserve_blends():
    y = _noisy()
    # preserve=1 returns the original series exactly
    np.testing.assert_allclose(ts.TsFilHp(100, preserve=1.0).fit(y).transform(y), y, atol=1e-9)


def test_hp_preserve_validation():
    with pytest.raises(ValueError):
        ts.TsFilHp(preserve=1.5)


def test_fft_validation():
    with pytest.raises(ValueError):
        ts.TsFilFft(threshold=0)
    with pytest.raises(ValueError):
        ts.TsFilGabor(central_frequency=0.9)


def test_kalman_tracks_series():
    y = _noisy()
    out = ts.TsFilKalman(H=0.1, Q=1.0).fit(y).transform(y)
    assert len(out) == len(y) and np.isfinite(out).all()
    # with low obs noise the filtered level follows the series closely
    assert np.corrcoef(out, y)[0, 1] > 0.9


@pytest.mark.parametrize("name", list(ALL))
def test_deterministic(name):
    y = _noisy()
    a = np.asarray(ALL[name].fit(y).transform(y), float)
    b = np.asarray(ALL[name].fit(y).transform(y), float)
    np.testing.assert_allclose(a, b, equal_nan=True)
