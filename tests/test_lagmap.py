"""Unit tests for the lag-mapping strategies."""
import pytest

import pytspredit as ts

ALL_METHODS = [
    "recent", "even", "geom", "acf", "pacf", "peaks",
    "seasonal", "acf_seasonal", "pacf_seasonal", "blocks", "mi", "mrmr",
]


def _io():
    return ts.ts_projection(ts.ts_data(ts.datasets.tsd()["y"], 10))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_each_method_selects_input_size_valid_positions(method):
    io = _io()
    lm = ts.TsLagmap(method=method, seasonality=4).fit(io.input, io.output, input_size=4)
    pos = lm.positions
    assert len(pos) == 4
    assert list(pos) == sorted(pos)             # ascending
    assert pos.min() >= 0 and pos.max() < io.input.shape[1]  # valid, 0-based
    assert len(set(pos)) == 4                    # unique


def test_recent_selects_most_recent_columns():
    io = _io()
    lm = ts.TsLagmap("recent").fit(io.input, io.output, input_size=4)
    # input has 9 columns t9..t1; most recent 4 -> last 4 (0-based 5..8)
    assert list(lm.positions) == [5, 6, 7, 8]
    assert list(lm.lags) == [4, 3, 2, 1]


def test_even_spreads_positions():
    io = _io()
    lm = ts.TsLagmap("even").fit(io.input, io.output, input_size=4)
    assert list(lm.positions) == [0, 3, 5, 8]  # 1;4;6;9 -> 0-based


def test_target_required_for_supervised_methods():
    io = _io()
    with pytest.raises(ValueError):
        ts.TsLagmap("acf").fit(io.input, None, input_size=4)


def test_input_size_validation():
    io = _io()
    with pytest.raises(ValueError):
        ts.TsLagmap("recent").fit(io.input, io.output, input_size=0)
    with pytest.raises(ValueError):
        ts.TsLagmap("recent").fit(io.input, io.output, input_size=99)


def test_invalid_method():
    with pytest.raises(ValueError):
        ts.TsLagmap("nope")
