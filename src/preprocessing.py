"""Clean, transform, and split the raw FRED pull for counterfactual modeling.

All series are aligned to a common monthly index. Index-level series are
converted to year-over-year (YoY) percent changes so that outcomes are
stationary-ish and comparable in scale to the policy variable. Missing
values are forward-filled up to a small gap (to bridge single-month
reporting lags) and any remaining gaps are dropped at the edges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INTERVENTION_DATE = pd.Timestamp("2022-03-01")  # first Fed hike of the 2022 tightening cycle


def to_yoy(series: pd.Series) -> pd.Series:
    """Year-over-year percent change for a monthly index-level series."""
    return series.pct_change(periods=12) * 100


def build_analysis_frame(raw: pd.DataFrame, max_ffill: int = 2) -> pd.DataFrame:
    """Transform the raw wide FRED pull into the modeling dataset.

    Returns a monthly DataFrame with:
      fedfunds                - policy rate level (%), the treatment variable
      inflation_cpi_yoy       - US CPI inflation, YoY % (primary outcome)
      inflation_pce_yoy       - US PCE inflation, YoY % (robustness outcome)
      output_indpro_yoy       - US industrial production growth, YoY % (output outcome)
      unemployment            - US unemployment rate, % (secondary outcome)
      gdp_growth_yoy          - US real GDP growth, YoY % (quarterly, forward-filled to monthly)
      cpi_euro_yoy            - Euro area HICP inflation, YoY % (control covariate)
      cpi_uk_yoy              - UK CPI inflation, YoY % (control covariate)
      commodity_price_yoy     - Global commodity price index, YoY % (control covariate)
    """
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("MS")  # month-start frequency; introduces NaN for any missing month

    out = pd.DataFrame(index=df.index)
    out["fedfunds"] = df["fedfunds"]
    out["inflation_cpi_yoy"] = to_yoy(df["cpi_us"])
    out["inflation_pce_yoy"] = to_yoy(df["pcepi_us"])
    out["output_indpro_yoy"] = to_yoy(df["indpro_us"])
    out["unemployment"] = df["unrate_us"]
    out["gdp_growth_yoy"] = to_yoy(df["gdp_us"]).ffill(limit=2)  # quarterly series -> monthly
    out["cpi_euro_yoy"] = to_yoy(df["cpi_euro"])
    out["cpi_uk_yoy"] = to_yoy(df["cpi_uk"])
    out["commodity_price_yoy"] = to_yoy(df["global_commodity_price"])

    # Bridge short, isolated reporting gaps (e.g. a single late-arriving month);
    # do not fill long stretches, which would fabricate data.
    out = out.ffill(limit=max_ffill)

    # Drop rows where the primary outcome or the treatment variable is missing
    # (typically the first 12 months, lost to the YoY transform, and any
    # not-yet-reported months at the end of the sample).
    out = out.dropna(subset=["fedfunds", "inflation_cpi_yoy"])

    return out


def train_test_split_by_date(
    df: pd.DataFrame,
    intervention_date: pd.Timestamp = INTERVENTION_DATE,
    post_window_months: int | None = 36,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into a pre-intervention training window and a post-intervention window.

    This is NOT a random train/test split: because the goal is to build a
    counterfactual (what would have happened absent the policy change), the
    model must be trained only on data untouched by the intervention (the
    "training" period) and evaluated against the actually observed
    post-intervention period (the "test" period used for comparison).
    """
    train = df.loc[df.index < intervention_date].copy()
    post = df.loc[df.index >= intervention_date].copy()
    if post_window_months is not None:
        cutoff = intervention_date + pd.DateOffset(months=post_window_months)
        post = post.loc[post.index < cutoff]
    return train, post


if __name__ == "__main__":
    from data_collection import fetch_all

    raw = fetch_all()
    analysis = build_analysis_frame(raw)
    train, post = train_test_split_by_date(analysis)
    print(f"Analysis frame: {analysis.shape}, {analysis.index.min()} -> {analysis.index.max()}")
    print(f"Train (pre-intervention): {train.shape}, {train.index.min()} -> {train.index.max()}")
    print(f"Post (evaluation window): {post.shape}, {post.index.min()} -> {post.index.max()}")
    print(analysis.isna().sum())
