"""R-vs-Python parity guard for the native MODWT wavelet filter.

MODWT is a deterministic linear transform, so unlike EMD this is checked VALUE-WISE:
the native ``TsFilWavelet`` must match the R ``wavelets`` package (modwt / zero level-1
detail / imodwt) to a tight tolerance. Reference develop/parity/filters_wavelet_ref.json
is produced by develop/parity/filters_wavelet.R. See develop/parity.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pytspredit import TsFilWavelet

_REF = Path(__file__).resolve().parents[1] / "develop" / "parity" / "filters_wavelet_ref.json"


def _load():
    with open(_REF, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.skipif(not _REF.exists(), reason="run develop/parity/filters_wavelet.R")
@pytest.mark.parametrize("filter_name", _load()["filters"] if _REF.exists() else [])
def test_wavelet_matches_r(filter_name):
    ref = _load()
    x = np.asarray(ref["input"], float)
    expected = np.asarray(ref["denoised"][filter_name], float)
    got = TsFilWavelet(filter_name).fit(x).transform(x)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=0,
                               err_msg=f"{filter_name}: MODWT differs from R wavelets")
