"""Unit tests for the TsPredBench harness."""
import warnings

import numpy as np

import pytspredit as ts

warnings.filterwarnings("ignore")


def _grid(**kw):
    return ts.TsPredBench(
        datasets={"tsd": ts.datasets.tsd()["y"], "santafe_a": ts.datasets.santafe_a()["V1"]},
        models={
            "persist": lambda: ts.TsPersist(),
            "knn": lambda: ts.TsKnn(ts.TsNormGminmax(), input_size=5),
            "darima": lambda: ts.TsDarima(ts.TsNormDiff(), input_size=5),
        },
        input_size=5, steps_ahead=5, **kw,
    )


def test_run_shape_and_columns():
    res = _grid().run()
    assert list(res.columns) == ["dataset", "model", "mse", "smape", "r2"]
    assert len(res) == 2 * 3  # datasets x models
    assert set(res["dataset"]) == {"tsd", "santafe_a"}
    assert set(res["model"]) == {"persist", "knn", "darima"}


def test_metrics_are_finite_and_reasonable():
    res = _grid().run().set_index(["dataset", "model"])
    # darima with differencing is near-exact on the smooth sine tsd
    assert res.loc[("tsd", "darima"), "mse"] < 1e-6
    # knn beats the naive persistence baseline on tsd
    assert res.loc[("tsd", "knn"), "mse"] < res.loc[("tsd", "persist"), "mse"]
    assert np.isfinite(res["mse"]).all()


def test_accepts_model_instances_not_only_factories():
    bench = ts.TsPredBench(
        datasets={"tsd": ts.datasets.tsd()["y"]},
        models={"persist": ts.TsPersist()},
        input_size=5, steps_ahead=5,
    )
    res = bench.run()
    assert len(res) == 1
    assert np.isfinite(res.loc[0, "mse"])


def test_failing_cell_is_isolated():
    class Boom:
        def fit(self, x, y):
            raise RuntimeError("boom")

    bench = ts.TsPredBench(
        datasets={"tsd": ts.datasets.tsd()["y"]},
        models={"ok": lambda: ts.TsPersist(), "bad": lambda: Boom()},
        input_size=5, steps_ahead=5,
    )
    res = bench.run()
    assert "error" in res.columns
    bad = res[res["model"] == "bad"].iloc[0]
    assert np.isnan(bad["mse"]) and "RuntimeError" in bad["error"]
    ok = res[res["model"] == "ok"].iloc[0]
    assert np.isfinite(ok["mse"])
