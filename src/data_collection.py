"""Fetch macroeconomic series from FRED (Federal Reserve Economic Data).

No API key is required: pandas-datareader pulls the public CSV endpoint
FRED exposes for each series. Raw pulls are cached to data/raw/ so the
notebook is reproducible even if FRED is temporarily unreachable.
"""
from __future__ import annotations

import pandas as pd
from pandas_datareader import data as pdr
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# FRED series used in the analysis.
# Treatment / policy variable:
#   FEDFUNDS  - Effective Federal Funds Rate (monthly, %)
# Outcome variables (US):
#   CPIAUCSL  - CPI, All Urban Consumers, All Items, SA (monthly index) -> YoY inflation
#   PCEPI     - PCE Price Index (monthly index)                         -> YoY inflation (robustness)
#   INDPRO    - Industrial Production Index (monthly)                   -> YoY growth, output proxy
#   UNRATE    - Civilian Unemployment Rate (monthly, %)
#   GDPC1     - Real GDP, chained 2017 dollars (quarterly)              -> YoY growth
# Control / donor series (used as covariates to net out global shocks common
# to all economies, so the model isolates the effect attributable to the
# Fed's own policy shift rather than shared global inflation dynamics):
#   CP0000EZ19M086NEST - Euro area HICP, all items, index (ECB began hiking ~4 months after the Fed)
#   GBRCPIALLMINMEI    - UK CPI, all items, index (BoE tightening path differed from the Fed's)
#   PALLFNFINDEXM      - Global price index of all commodities (captures global supply shocks
#                        that are not driven by the Fed's domestic rate decision)
FRED_SERIES = {
    "FEDFUNDS": "fedfunds",
    "CPIAUCSL": "cpi_us",
    "PCEPI": "pcepi_us",
    "INDPRO": "indpro_us",
    "UNRATE": "unrate_us",
    "GDPC1": "gdp_us",
    "CP0000EZ19M086NEST": "cpi_euro",
    "GBRCPIALLMINMEI": "cpi_uk",
    "PALLFNFINDEXM": "global_commodity_price",
}


def fetch_series(fred_code: str, start: str = "2005-01-01", end: str | None = None) -> pd.Series:
    """Download a single FRED series and return it as a named pandas Series."""
    df = pdr.DataReader(fred_code, "fred", start, end)
    series = df[fred_code]
    series.name = fred_code
    return series


def fetch_all(start: str = "2005-01-01", end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Fetch all series in FRED_SERIES, cache each raw pull, and return a wide DataFrame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    columns = {}
    for fred_code, friendly_name in FRED_SERIES.items():
        cache_path = RAW_DIR / f"{friendly_name}.csv"
        if use_cache and cache_path.exists():
            s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = friendly_name
        else:
            s = fetch_series(fred_code, start=start, end=end)
            s.name = friendly_name
            s.to_frame().to_csv(cache_path)
        columns[friendly_name] = s
    wide = pd.DataFrame(columns)
    wide.index.name = "date"
    return wide.sort_index()


if __name__ == "__main__":
    df = fetch_all()
    print(df.tail())
    print(f"\nFetched {df.shape[1]} series, {df.shape[0]} monthly observations.")
