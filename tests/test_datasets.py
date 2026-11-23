"""Unit tests for reference datasets."""
import numpy as np
import pandas as pd
import pytest

import pytspredit as ts
from pytspredit import datasets


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


# --- Eidolon miniatures (0.6.0) -------------------------------------------------

_DATAFRAME = ["cats", "eunite_loads", "eunite_reg", "eunite_temp", "nn3", "nn5",
              "santafe_a", "santafe_d", "ipeadata_d", "ipeadata_m"]
_SERIES_LIST = ["bioenergy", "climate", "emissions", "fertilizers", "gdp", "pesticides"]
_DF_LIST = ["m1", "stocks"]
_NESTED = ["m3", "m4"]


def test_catalog_and_list():
    cat = datasets.catalog()
    assert "tsd" in cat and "NN5" in cat
    assert len(datasets.list_datasets()) == 21  # 20 tspredbench + tsd
    assert datasets.dataset_url("tsd") is None
    assert datasets.dataset_url("NN5").startswith("https://raw.githubusercontent.com/")


@pytest.mark.parametrize("name", _DATAFRAME)
def test_dataframe_miniatures(name):
    df = getattr(datasets, name)()
    entry = datasets.catalog()[_PY_TO_R[name]]
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (entry["nrow"], entry["ncol"])
    assert list(df.columns) == entry["columns"]


@pytest.mark.parametrize("name", _SERIES_LIST)
def test_series_list_miniatures(name):
    obj = getattr(datasets, name)()
    assert isinstance(obj, dict)
    assert all(isinstance(v, pd.Series) for v in obj.values())


@pytest.mark.parametrize("name", _DF_LIST)
def test_df_list_miniatures(name):
    obj = getattr(datasets, name)()
    assert isinstance(obj, dict)
    assert all(isinstance(v, pd.DataFrame) for v in obj.values())


@pytest.mark.parametrize("name", _NESTED)
def test_nested_list_miniatures(name):
    obj = getattr(datasets, name)()
    assert isinstance(obj, dict)
    first = next(iter(obj.values()))
    assert isinstance(first, dict)
    assert all(isinstance(v, pd.Series) for v in first.values())


_PY_TO_R = {
    "cats": "CATS", "eunite_loads": "EUNITE.Loads", "eunite_reg": "EUNITE.Reg",
    "eunite_temp": "EUNITE.Temp", "nn3": "NN3", "nn5": "NN5",
    "santafe_a": "SantaFe.A", "santafe_d": "SantaFe.D",
    "ipeadata_d": "ipeadata.d", "ipeadata_m": "ipeadata.m",
    "bioenergy": "bioenergy", "climate": "climate", "emissions": "emissions",
    "fertilizers": "fertilizers", "gdp": "gdp", "pesticides": "pesticides",
    "m1": "m1", "m3": "m3", "m4": "m4", "stocks": "stocks",
}
