"""Unit tests for the matplotlib forecast plots (0.5.0)."""
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytspredit as ts

warnings.filterwarnings("ignore")


def _mv():
    y = ts.datasets.tsd()["y"].to_numpy()
    x1 = np.concatenate([y[1:], y[-1:]])
    x2 = pd.Series(y).rolling(3).mean().to_numpy()
    mask = np.isnan(x2)
    x2[mask] = y[mask]
    return ts.ts_data_mv(pd.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y")


def test_plot_ts_pred_returns_figure():
    y = ts.datasets.tsd()["y"].to_numpy()
    fig = ts.plot_ts_pred(y=y[:20], yadj=y[:15], ypred=y[15:20])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_ts_pred_mv_all_variables():
    samp = ts.ts_sample(_mv(), test_size=5)
    pred = ts.TsVar(p_max=3).fit(samp.train).predict(steps_ahead=5)
    plots = ts.plot_ts_pred_mv(samp.train, samp.test, pred)
    assert set(plots) == {"y", "x1", "x2"}
    assert all(isinstance(v, plt.Figure) for v in plots.values())
    for v in plots.values():
        plt.close(v)


def test_plot_ts_pred_mv_single_variable():
    samp = ts.ts_sample(_mv(), test_size=5)
    pred = ts.TsVar(p_max=3).fit(samp.train).predict(steps_ahead=5)
    fig = ts.plot_ts_pred_mv(samp.train, samp.test, pred, variable="y")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_ts_pred_mv_rejects_non_mv_prediction():
    with pytest.raises(ValueError):
        ts.plot_ts_pred_mv([1, 2, 3], prediction=np.array([1.0, 2.0]))
