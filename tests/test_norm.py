"""Unit tests for time series normalizations."""
import numpy as np

import pytspredit as ts


def _windows():
    return ts.ts_data(ts.datasets.tsd()["y"], 10)


def test_none_identity():
    w = _windows()
    n = ts.TsNormNone().fit(w)
    out = n.transform(w)
    np.testing.assert_allclose(np.asarray(out, float), np.asarray(w, float))
    # x-path returns x unchanged
    assert n.transform(w, x=[1.0, 2.0]) == [1.0, 2.0]


def test_gminmax_range_and_inverse():
    w = _windows()
    n = ts.TsNormGminmax().fit(w)
    out = np.asarray(n.transform(w), float)
    assert out.min() >= -1e-9 and out.max() <= 1 + 1e-9
    back = np.asarray(n.inverse_transform(n.transform(w)), float)
    np.testing.assert_allclose(back, np.asarray(w, float), atol=1e-9)


def test_swminmax_per_row_inverse():
    w = _windows()
    n = ts.TsNormSwminmax().fit(w)
    tr = n.transform(w)
    back = np.asarray(n.inverse_transform(tr), float)
    np.testing.assert_allclose(back, np.asarray(w, float), atol=1e-9)


def test_diff_reduces_width_and_inverts():
    w = _windows()
    n = ts.TsNormDiff().fit(w)
    tr = n.transform(w)
    assert np.asarray(tr).shape[1] == w.shape[1] - 1
    back = np.asarray(n.inverse_transform(tr), float)
    np.testing.assert_allclose(back, np.asarray(w, float), atol=1e-9)


def test_an_operators_roundtrip():
    w = _windows()
    for kwargs in [
        {},
        {"operation": "subtract"},
        {"average": "ema", "operation": "softdivide", "scale": "mad"},
        {"operation": "asinh", "scale": "sd"},
    ]:
        n = ts.TsNormAn(**kwargs).fit(w)
        tr = n.transform(w)
        back = np.asarray(n.inverse_transform(tr), float)
        np.testing.assert_allclose(back, np.asarray(w, float), atol=1e-7,
                                   err_msg=f"roundtrip failed for {kwargs}")


def test_an_validation():
    import pytest
    for bad in [{"average": "x"}, {"operation": "x"}, {"scale": "x"}]:
        with pytest.raises(ValueError):
            ts.TsNormAn(**bad)
