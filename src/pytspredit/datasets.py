"""Reference datasets for examples, tutorials, and benchmarking.

Python counterpart of the R ``tspredit`` data collection. Besides the small
synthetic ``tsd`` series (the didactic entry point), this module ships the
**Eidolon miniatures** of the ``tspredbench`` collection: compact local
stand-ins that preserve the same access logic as the full remote artifacts.

Eidolon architecture (Calmon et al., 2026): instead of embedding multi-gigabyte
collections, each dataset is published as a compact *miniature* (the first 30
rows of a data frame, or the first element of a list) carrying the remote URL of
the complete artifact. The miniature and the full artifact share one usage
contract -- dataset identity, structural schema, and the resolution mechanism --
so local discovery and documentation work without downloading the full data. The
full artifacts live in the separate ``cefet-rj-dal/tspredbench`` repository and
are resolved on demand (``loadfulldata``, Cycle 2).

The R package serializes miniatures as ``.RData`` (R-native). This Python
instantiation of the same contract serializes them in Python-native formats --
**CSV** for data frames, **JSON** for lists -- plus a ``manifest.json`` that
records each dataset's URL, kind, and schema. Values are extracted from the
original ``.RData`` miniatures so they are numerically identical to R.

References:
    Calmon, L., Junior, M., Santos, L., Alexandrino, F., Coutinho, R., Guedes, G.,
    Carvalho, D., & Ogasawara, E. (2026). *Eidolon: An Architecture for the
    Reproducible Publication of Scientific Data Collections*. XX Brazilian
    e-Science Workshop (BreSci).
    Ogasawara, E., Alexandrino, F., Gea, C., Santos, D., Salles, R., et al.
    (2025). *A Benchmark Time Series Dataset Collection for Prediction Models*.

Porting:
    Mirrors ``data.R`` (``tsd``) and ``tspredbench.R`` (the dataset collection).
    Fidelity: exact (values identical to the R miniatures). The serialization
    format differs by design (CSV/JSON + manifest vs ``.RData``), preserving the
    Eidolon contract rather than the R serialization.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

__all__ = [
    "bioenergy",
    "catalog",
    "cats",
    "climate",
    "dataset_url",
    "emissions",
    "eunite_loads",
    "eunite_reg",
    "eunite_temp",
    "fertilizers",
    "gdp",
    "ipeadata_d",
    "ipeadata_m",
    "list_datasets",
    "m1",
    "m3",
    "m4",
    "nn3",
    "nn5",
    "pesticides",
    "santafe_a",
    "santafe_d",
    "stocks",
    "tsd",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _manifest() -> dict:
    with open(_DATA_DIR / "manifest.json", encoding="utf-8") as fh:
        return json.load(fh)


def _records_to_df(records: list) -> pd.DataFrame:
    """Build a DataFrame from JSON row-records, honoring R row names.

    ``jsonlite`` serializes a data frame with non-default row names by adding a
    ``_row`` field per record; we restore it as the index to mirror R's rownames.
    """
    df = pd.DataFrame(records)
    if "_row" in df.columns:
        df = df.set_index("_row")
        df.index.name = None
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns whose non-null values are all numeric to a numeric dtype.

    JSON round-tripping types an all-null column as ``object``; R keeps it numeric
    (``NA_real_``). This restores the faithful dtype while leaving genuine string
    columns (e.g. ``date``, ``split``) untouched.
    """
    for col in df.columns:
        s = df[col]
        if s.dtype == object:
            coerced = pd.to_numeric(s, errors="coerce")
            if not (s.notna() & coerced.isna()).any():
                df[col] = coerced
    return df


def _load(name: str):
    """Load a miniature by its manifest name, dispatching on its ``kind``.

    Returns a :class:`pandas.DataFrame` for ``dataframe`` datasets, a ``dict`` of
    series for ``series_list``, a ``dict`` of data frames for ``df_list``, and a
    nested ``dict`` of series for ``nested_list``.
    """
    entry = _manifest()[name]
    kind = entry["kind"]
    path = _DATA_DIR / entry["file"]
    if kind == "dataframe":
        return pd.read_csv(path)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if kind == "series_list":
        return {k: pd.Series(v, name=k) for k, v in raw.items()}
    if kind == "df_list":
        return {k: _coerce_numeric(_records_to_df(v)) for k, v in raw.items()}
    if kind == "nested_list":
        return {k: {sk: pd.Series(sv, name=sk) for sk, sv in v.items()} for k, v in raw.items()}
    raise ValueError(f"unknown dataset kind {kind!r} for {name!r}")  # pragma: no cover


def catalog() -> dict:
    """Return the dataset manifest: ``{name: {url, kind, file, ...schema}}``."""
    return _manifest()


def list_datasets() -> list[str]:
    """Return the sorted list of available dataset names."""
    return sorted(_manifest())


def dataset_url(name: str) -> str | None:
    """Return the remote URL of a dataset's full artifact (``None`` for ``tsd``).

    The URL resolves to the complete ``.RData`` in the ``cefet-rj-dal/tspredbench``
    repository. Downloading and loading it (``loadfulldata``) is provided by the
    optional ``[bench]`` extra (Cycle 2).
    """
    return _manifest()[name].get("url")


def tsd() -> pd.DataFrame:
    """Synthetic example series: ``y = sin(x)`` over ``x = seq(0, 10, 0.25)``.

    Columns ``x`` (regular time index) and ``y`` (smooth sine target). This is the
    didactic entry point of the package; it is embedded (no remote URL).

    Porting:
        Mirrors ``data.R`` (``tsd``). Fidelity: exact.
    """
    return _load("tsd")


def cats() -> pd.DataFrame:
    """CATS competition: artificial series with five missing blocks (miniature).

    References:
        Lendasse, A., Oja, E., Simula, O., & Verleysen, M. (2007). *Time Series
        Prediction Competition: The CATS Benchmark*. Neurocomputing, 70(13-15),
        2325-2329.
    """
    return _load("CATS")


def eunite_loads() -> pd.DataFrame:
    """EUNITE competition: half-hourly electrical loads, wide format (miniature).

    References:
        Chen, B.-J., Chang, M.-W., & Lin, C.-J. (2004). *Load forecasting using
        support vector machines: a study on EUNITE competition 2001*. IEEE
        Transactions on Power Systems, 19(4), 1821-1830.
    """
    return _load("EUNITE.Loads")


def eunite_reg() -> pd.DataFrame:
    """EUNITE competition: holiday/weekday regressors for load forecasting (miniature).

    References:
        Chen, B.-J., Chang, M.-W., & Lin, C.-J. (2004). IEEE Transactions on Power
        Systems, 19(4), 1821-1830.
    """
    return _load("EUNITE.Reg")


def eunite_temp() -> pd.DataFrame:
    """EUNITE competition: average daily temperatures (miniature).

    References:
        Chen, B.-J., Chang, M.-W., & Lin, C.-J. (2004). IEEE Transactions on Power
        Systems, 19(4), 1821-1830.
    """
    return _load("EUNITE.Temp")


def nn3() -> pd.DataFrame:
    """NN3 competition: monthly business time series (miniature, wide format).

    References:
        Crone, S. F., Hibon, M., & Nikolopoulos, K. (2011). *Advances in
        forecasting with neural networks? Empirical evidence from the NN3
        competition*. International Journal of Forecasting, 27(3), 635-660.
    """
    return _load("NN3")


def nn5() -> pd.DataFrame:
    """NN5 competition: daily ATM withdrawal series (miniature, wide format).

    References:
        Crone, S. F. (2008). *Results of the NN5 Time Series Forecasting
        Competition*. IEEE WCCI 2008, Hong Kong.
    """
    return _load("NN5")


def santafe_a() -> pd.DataFrame:
    """Santa Fe competition series A: chaotic far-infrared laser (miniature).

    References:
        Weigend, A. S. (1993). *Time Series Prediction: Forecasting the Future and
        Understanding the Past*. Westview Press.
    """
    return _load("SantaFe.A")


def santafe_d() -> pd.DataFrame:
    """Santa Fe competition series D: nonlinear non-stationary series (miniature).

    References:
        Weigend, A. S. (1993). *Time Series Prediction: Forecasting the Future and
        Understanding the Past*. Westview Press.
    """
    return _load("SantaFe.D")


def ipeadata_d() -> pd.DataFrame:
    """Ipea daily macroeconomic series (miniature, one column per series).

    References:
        Ipea (2017). *Ipeadata - Macroeconomic and Regional Data*.
        http://www.ipeadata.gov.br
    """
    return _load("ipeadata.d")


def ipeadata_m() -> pd.DataFrame:
    """Ipea monthly macroeconomic series (miniature, one column per series).

    References:
        Ipea (2017). *Ipeadata - Macroeconomic and Regional Data*.
        http://www.ipeadata.gov.br
    """
    return _load("ipeadata.m")


def bioenergy() -> dict:
    """FAOSTAT bioenergy series (miniature): ``{series_name: pandas.Series}``.

    References:
        FAO (2024). *FAOSTAT Bioenergy*. Rome, Italy.
    """
    return _load("bioenergy")


def climate() -> dict:
    """FAOSTAT/NASA-GISS temperature-change-on-land series (miniature): ``{name: Series}``.

    References:
        FAO (2024). *FAOSTAT Climate Change Indicators*; GISTEMP Team (2024),
        NASA Goddard Institute for Space Studies.
    """
    return _load("climate")


def emissions() -> dict:
    """FAOSTAT greenhouse-gas emissions series (miniature): ``{name: Series}``.

    References:
        FAO (2023). *FAOSTAT Climate Change: Agrifood systems emissions*.
    """
    return _load("emissions")


def fertilizers() -> dict:
    """FAOSTAT fertilizers-by-nutrient series (miniature): ``{name: Series}``.

    References:
        FAO (2024). *FAOSTAT: Fertilizers by Nutrient*.
    """
    return _load("fertilizers")


def gdp() -> dict:
    """FAOSTAT GDP / agriculture value-added series (miniature): ``{name: Series}``.

    References:
        FAO (2024). *Gross domestic product and agriculture value added
        2013-2022*. FAOSTAT Analytical Briefs, No. 85. doi:10.4060/cd0763en
    """
    return _load("gdp")


def pesticides() -> dict:
    """FAOSTAT pesticides-use series (miniature): ``{series_name: pandas.Series}``.

    References:
        FAO (2024). *FAOSTAT: Pesticides Use*.
    """
    return _load("pesticides")


def m1() -> dict:
    """M1 competition series (miniature): ``{frequency: pandas.DataFrame}``.

    References:
        Makridakis, S., et al. (1982). *The accuracy of extrapolation (time
        series) methods: Results of a forecasting competition*. Journal of
        Forecasting, 1(2), 111-153.
    """
    return _load("m1")


def m3() -> dict:
    """M3 competition series (miniature): ``{frequency: {series_name: Series}}``.

    References:
        Makridakis, S., & Hibon, M. (2000). *The M3-Competition: Results,
        conclusions and implications*. International Journal of Forecasting,
        16(4), 451-476.
    """
    return _load("m3")


def m4() -> dict:
    """M4 competition series (miniature): ``{frequency: {series_name: Series}}``.

    References:
        Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). *The M4
        Competition: Results, findings, conclusion and way forward*.
        International Journal of Forecasting, 36(1), 54-74.
    """
    return _load("m4")


def stocks() -> dict:
    """IBOVESPA most-traded stocks (miniature): ``{ticker: pandas.DataFrame}``.

    Each data frame has columns ``date, open, high, low, close, volume``.

    References:
        B3 - Brasil, Bolsa, Balcao (2025). *Historical stock trading data*.
        https://www.b3.com.br
    """
    return _load("stocks")
