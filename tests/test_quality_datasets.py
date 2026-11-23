"""R <-> Python parity guard for the Eidolon dataset miniatures.

Locks the committed-parity result for 0.6.0: the shipped Python miniatures
(CSV/JSON) must be numerically identical to the R ``.RData`` miniatures. The
reference ``develop/parity/datasets_ref.json`` is produced by
``develop/parity/datasets.R`` (run against the R miniatures); this test replays the
same canonical flatten on the shipped miniatures and asserts equality, so parity is
checked without needing R at test time. See ``develop/parity.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytspredit import datasets

_REF = Path(__file__).resolve().parents[1] / "develop" / "parity" / "datasets_ref.json"


def _flatten(obj, kind):
    num: list[float] = []
    lab: list[str] = []

    def eat_df(df: pd.DataFrame):
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                num.extend(np.asarray(s, dtype=float).tolist())
            else:
                lab.extend(s.astype(str).tolist())

    if kind == "dataframe":
        eat_df(obj)
    elif kind == "series_list":
        for v in obj.values():
            num.extend(np.asarray(v, dtype=float).tolist())
    elif kind == "df_list":
        for df in obj.values():
            eat_df(df)
    elif kind == "nested_list":
        for sub in obj.values():
            for v in sub.values():
                num.extend(np.asarray(v, dtype=float).tolist())
    else:  # pragma: no cover
        raise ValueError(kind)
    return np.asarray(num, dtype=float), lab


def _load_ref():
    with open(_REF, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.skipif(not _REF.exists(), reason="parity reference not generated (run develop/parity/datasets.R)")
@pytest.mark.parametrize("name", list(_load_ref()) if _REF.exists() else [])
def test_miniature_matches_r(name):
    ref = _load_ref()[name]
    obj = datasets._load(name)
    num, lab = _flatten(obj, ref["kind"])

    ref_num = np.asarray(
        [np.nan if v is None else v for v in np.atleast_1d(ref["numeric"])], dtype=float
    )
    assert num.shape == ref_num.shape, f"{name}: {num.shape} vs {ref_num.shape}"
    np.testing.assert_allclose(num, ref_num, atol=1e-9, rtol=0, equal_nan=True,
                               err_msg=f"{name}: numeric mismatch vs R miniature")

    ref_lab = [str(x) for x in np.atleast_1d(ref["labels"]).tolist()] if ref["labels"] else []
    assert lab == ref_lab, f"{name}: label/string columns differ from R"
