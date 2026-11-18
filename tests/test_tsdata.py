"""Unit tests for the sliding-window core (ts_data / ts_projection / ts_sample)."""
import numpy as np
import pandas as pd
import pytest

import pytspredit as ts


def test_ts_data_single_column():
    d = ts.ts_data([1, 2, 3, 4], sw=1)
    assert list(d.columns) == ["t0"]
    assert d.shape == (4, 1)
    assert d["t0"].tolist() == [1, 2, 3, 4]


def test_ts_data_sliding_windows_shape_and_names():
    d = ts.ts_data(list(range(10)), sw=4)
    # n - sw + 1 rows, sw columns named t3..t0
    assert d.shape == (7, 4)
    assert list(d.columns) == ["t3", "t2", "t1", "t0"]


def test_ts_data_window_content():
    # row i holds y[i], y[i+1], ..., y[i+sw-1] as t{sw-1}..t0
    y = [10, 20, 30, 40, 50]
    d = ts.ts_data(y, sw=3)
    assert d.iloc[0].tolist() == [10, 20, 30]  # t2, t1, t0
    assert d.iloc[1].tolist() == [20, 30, 40]
    assert d.iloc[-1].tolist() == [30, 40, 50]


def test_ts_data_sw_validation():
    with pytest.raises(ValueError):
        ts.ts_data([1, 2, 3], sw=0)
    with pytest.raises(ValueError):
        ts.ts_data([1, 2, 3], sw=5)  # larger than series length


def test_ts_projection_splits_last_column():
    d = ts.ts_data(list(range(10)), sw=4)
    io = ts.ts_projection(d)
    assert list(io.input.columns) == ["t3", "t2", "t1"]
    assert io.output.name == "t0"
    # the target equals the last column of the windows
    assert io.output.tolist() == d["t0"].tolist()


def test_ts_sample_temporal_split():
    d = ts.ts_data(list(range(20)), sw=5)  # 16 rows
    samp = ts.ts_sample(d, test_size=3)
    assert samp.train.shape[0] == d.shape[0] - 3
    assert samp.test.shape[0] == 3
    # test is the tail; train is the head, order preserved
    assert samp.test.iloc[0]["t0"] == d.iloc[-3]["t0"]
    assert samp.train.iloc[0]["t0"] == d.iloc[0]["t0"]


def test_ts_sample_offset():
    d = ts.ts_data(list(range(20)), sw=5)
    samp = ts.ts_sample(d, test_size=2, offset=1)
    # with offset=1, the last row is held out of the test window
    assert samp.test.iloc[-1]["t0"] == d.iloc[-2]["t0"]


def test_adjust_ts_data_from_ndarray():
    arr = np.arange(12).reshape(4, 3)
    d = ts.adjust_ts_data(arr)
    assert isinstance(d, pd.DataFrame)
    assert list(d.columns) == ["t2", "t1", "t0"]


def test_end_to_end_projection_sample():
    series = ts.datasets.tsd()["y"]
    d = ts.ts_data(series, sw=10)
    samp = ts.ts_sample(d, test_size=5)
    io_train = ts.ts_projection(samp.train)
    io_test = ts.ts_projection(samp.test)
    assert io_train.input.shape[1] == 9
    assert io_test.output.shape[0] == 5
