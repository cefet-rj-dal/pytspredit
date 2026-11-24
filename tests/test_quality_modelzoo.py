"""R-vs-Python parity guard for the 0.2 model zoo (retroactive M3 backfill).

All deterministic / value-wise exact (<= 1e-9): the rule-based trio (periodic/persist)
and the window-linear models (darima/warma), whose ``_RLinear`` reproduces R ``lm``
column-dropping exactly -- including the degenerate rank-3 pure-sine (tsd) case.
Reference develop/parity/modelzoo_ref.json is produced by develop/parity/modelzoo.R.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import pytspredit as ts

warnings.filterwarnings("ignore")

_PARITY = Path(__file__).resolve().parents[1] / "develop" / "parity"
_REF = _PARITY / "modelzoo_ref.json"


def _load():
    with open(_REF, encoding="utf-8") as fh:
        return json.load(fh)


def _series(name):
    if name == "logistic":
        return np.loadtxt(_PARITY / "logistic.csv", skiprows=1)
    return np.asarray(ts.datasets.tsd()["y"], float)


# darima/warma: builder + source series (mirrors develop/parity/modelzoo.R)
_WIN_MODELS = {
    "darima_diff_logistic": (lambda: ts.TsDarima(ts.TsNormDiff(), input_size=5), "logistic"),
    "darima_none_logistic": (lambda: ts.TsDarima(ts.TsNormNone(), input_size=5), "logistic"),
    "darima_none_tsd": (lambda: ts.TsDarima(ts.TsNormNone(), input_size=5), "tsd"),
    "warma_steps1_logistic": (lambda: ts.TsWarma(input_size=5, steps=1), "logistic"),
    "warma_auto_logistic": (lambda: ts.TsWarma(input_size=5, steps=None), "logistic"),
}


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/modelzoo.R")
def test_deterministic_rules_match_r():
    ref = _load()
    p = ts.TsPeriodic(7).fit([4, 5, 6, 7, 1, 2, 3]).predict(steps_ahead=5)
    np.testing.assert_allclose(np.asarray(p, float),
                               np.asarray(ref["models"]["periodic"]["pred"], float), atol=1e-9)
    s = ts.TsPersist().fit([10, 11, 11, 12]).predict(steps_ahead=3)
    np.testing.assert_allclose(np.asarray(s, float),
                               np.asarray(ref["models"]["persist"]["pred"], float), atol=1e-9)


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/modelzoo.R")
@pytest.mark.parametrize("name", list(_WIN_MODELS))
def test_window_linear_matches_r(name):
    ref = _load()
    builder, src = _WIN_MODELS[name]
    w = ts.ts_data(_series(src), int(ref["sw"]))
    samp = ts.ts_sample(w, test_size=int(ref["test_size"]))
    tr, te = ts.ts_projection(samp.train), ts.ts_projection(samp.test)
    model = builder().fit(tr.input, tr.output)
    one = np.asarray(model.predict(te.input), float)
    multi = np.asarray(model.predict(te.input.iloc[[0]], steps_ahead=int(ref["test_size"])), float)
    np.testing.assert_allclose(one, np.asarray(ref["models"][name]["one"], float),
                               atol=1e-9, rtol=0, err_msg=f"{name}: one-step differs from R")
    np.testing.assert_allclose(multi, np.asarray(ref["models"][name]["multi"], float),
                               atol=1e-9, rtol=0, err_msg=f"{name}: multi-step differs from R")
