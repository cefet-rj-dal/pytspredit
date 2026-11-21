"""Reference datasets for examples and tutorials.

Python counterpart of the R ``data.R``. ``tsd`` is the small synthetic univariate
series that is the didactic entry point of the package (a smooth sine signal),
shipped as a CSV extracted from the original R ``tsd.RData`` so the Python port is
numerically identical to the R package.

Porting:
    Mirrors ``data.R`` (the ``tsd`` dataset). Fidelity: exact.
    - ``tsd`` is ``y = sin(x)`` over ``x = seq(0, 10, 0.25)`` (41 rows), extracted
      from ``tsd.RData`` so values are identical to R.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["tsd"]

_DATA_DIR = Path(__file__).resolve().parent / "data"


def tsd() -> pd.DataFrame:
    """Return the synthetic example series as a :class:`~pandas.DataFrame`.

    Columns:

    * ``x`` -- regular time index (0 to 10 in steps of 0.25).
    * ``y`` -- smooth sine-based signal used as the forecasting target.
    """
    return pd.read_csv(_DATA_DIR / "tsd.csv")
