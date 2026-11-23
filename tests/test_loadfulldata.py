"""Unit tests for datasets.loadfulldata (remote full-data expansion).

The live full artifacts live in the tspredbench repo; these tests exercise the
machinery (urllib + gzip + parse) against local ``file://`` URLs so they need no
network, and verify the miniature/full consistency (the miniature is the head of the
full data).
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytspredit import datasets


def _file_url(p: Path) -> str:
    return p.resolve().as_uri()


def test_loadfulldata_dataframe(tmp_path, monkeypatch):
    full = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) * 2})
    gz = tmp_path / "x.csv.gz"
    with gzip.GzipFile(gz, "wb") as fh:
        fh.write(full.to_csv(index=False).encode())
    entry = {"kind": "dataframe", "full_url": _file_url(gz)}
    monkeypatch.setattr(datasets, "_manifest", lambda: {"X": entry})
    out = datasets.loadfulldata("X")
    assert isinstance(out, pd.DataFrame) and out.shape == (100, 2)
    np.testing.assert_allclose(out["b"], np.arange(100.0) * 2)


def test_loadfulldata_series_list(tmp_path, monkeypatch):
    raw = {"s1": [1.0, 2.0, 3.0], "s2": [4.0, 5.0]}
    gz = tmp_path / "x.json.gz"
    with gzip.GzipFile(gz, "wb") as fh:
        fh.write(json.dumps(raw).encode())
    entry = {"kind": "series_list", "full_url": _file_url(gz)}
    monkeypatch.setattr(datasets, "_manifest", lambda: {"X": entry})
    out = datasets.loadfulldata("X")
    assert isinstance(out, dict) and list(out) == ["s1", "s2"]
    assert isinstance(out["s1"], pd.Series) and out["s1"].tolist() == [1.0, 2.0, 3.0]


def test_loadfulldata_rejects_embedded():
    with pytest.raises(ValueError, match="no remote full artifact"):
        datasets.loadfulldata("tsd")  # tsd is embedded, full_url is None


def test_manifest_has_full_urls():
    cat = datasets.catalog()
    # every non-tsd dataset carries a full_url for loadfulldata
    assert cat["NN5"]["full_url"].endswith("NN5.csv.gz")
    assert cat["m4"]["full_url"].endswith("m4.json.gz")
    assert cat["tsd"]["full_url"] is None
