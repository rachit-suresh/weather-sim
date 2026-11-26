"""Load the provided training NetCDF and split it by year. Training-only code: the server never imports this.

Data file location: env TWIN_DATA, else ../india_climate_train.nc (next to the repo; never commit it).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT.parent / "india_climate_train.nc"

# year-based split (decision I in docs/DECISIONS.md)
TRAIN_YEARS = range(2005, 2018)
VAL_YEARS = range(2018, 2020)
TEST_YEARS = range(2020, 2022)


@dataclass
class Dataset:
    data: np.ndarray        # [N, 6, 32, 36] float32, physical units
    static: np.ndarray      # [2, 32, 36]
    years: np.ndarray       # [N]
    months: np.ndarray      # [N]
    doy: np.ndarray         # [N] day of year, 1-based (Jan 1 = 1), as the API's sin/cos are built from
    ymd: np.ndarray         # [N] yyyymmdd ints (for the climatology builder)

    def mask(self, years) -> np.ndarray:
        return np.isin(self.years, list(years))


@lru_cache(maxsize=2)
def load(path: str | None = None) -> Dataset:
    import xarray as xr
    p = Path(path or os.environ.get("TWIN_DATA", DEFAULT_DATA))
    if not p.exists():
        raise FileNotFoundError(f"training data not found at {p}; set TWIN_DATA")
    ds = xr.open_dataset(p)
    data = ds["data"].values.astype("f4")
    static = ds["static"].values.astype("f4")
    t = ds["time"].values.astype("datetime64[D]")
    years = t.astype("datetime64[Y]").astype(int) + 1970
    months = t.astype("datetime64[M]").astype(int) % 12 + 1
    doy = (t - t.astype("datetime64[Y]")).astype(int) + 1
    days = (t - t.astype("datetime64[M]")).astype(int) + 1
    ymd = years * 10000 + months * 100 + days
    return Dataset(data, static, years, months, doy, ymd)


def window_starts(ds: Dataset, years, lead: int, stride: int = 1) -> np.ndarray:
    """Indices t of the LAST input day such that t-2..t and t+1..t+lead all lie inside `years`."""
    m = ds.mask(years)
    idx = np.arange(2, len(ds.data) - lead)
    ok = m[idx - 2] & m[idx + lead]
    return idx[ok][::stride]
