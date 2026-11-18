"""Unit tests for reference datasets."""
import numpy as np

import pytspredit as ts


def test_tsd_shape_and_columns():
    d = ts.datasets.tsd()
    assert list(d.columns) == ["x", "y"]
    assert d.shape == (41, 2)


def test_tsd_values_match_r_original():
    # tsd is the sine signal y = sin(x), x = seq(0, 10, 0.25), as in the R package.
    d = ts.datasets.tsd()
    assert d["x"].iloc[0] == 0.0
    assert d["x"].iloc[-1] == 10.0
    np.testing.assert_allclose(d["y"], np.sin(d["x"]), atol=1e-9)
