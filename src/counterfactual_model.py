"""Counterfactual (intervention analysis) model for policy impact evaluation.

Methodology
-----------
This is a regression-based structural time-series counterfactual, in the
spirit of Box & Tiao (1975) intervention analysis and Google's CausalImpact
(Brodersen et al., 2015):

1. Fit a local-level structural time-series model with regression covariates
   (an "unobserved components" / Kalman-filter model) using ONLY data from
   BEFORE the policy intervention. The covariates are macro series from
   economies/markets not subject to the Fed's policy decision (Euro area and
   UK inflation, global commodity prices), which proxy for global shocks
   common to all economies.
2. Use the fitted model to forecast forward through the post-intervention
   window, feeding in the *actual* observed values of the control covariates.
   This forecast is the counterfactual: our best estimate of what the outcome
   would have been had the policy NOT changed, given how the outcome
   historically related to its own dynamics and to global conditions.
3. The policy's estimated effect at each point in time is the gap between
   the actually observed outcome and this counterfactual forecast. Summing
   that gap over the post-period gives the cumulative effect.

Normalization: the control covariates are z-scored using pre-intervention
statistics only (see `fit_counterfactual`) before entering the regression,
since they arrive on different natural scales. The outcome itself is left in
its native, interpretable unit (year-over-year % change) rather than being
rescaled -- it was already put on a comparable, roughly stationary scale by
the YoY transform in `preprocessing.py`, which is the standard normalization
applied to macroeconomic index series.

Uncertainty (95% interval) comes from the forecast's prediction interval,
which combines parameter/state uncertainty and widens the further out the
forecast horizon extends -- exactly what should happen to a counterfactual
projected further past the last point the model actually saw.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


@dataclass
class CounterfactualResult:
    outcome_name: str
    control_columns: list[str]
    fitted_model: object
    counterfactual_mean: pd.Series
    counterfactual_ci_lower: pd.Series
    counterfactual_ci_upper: pd.Series
    actual_post: pd.Series
    in_sample_fitted: pd.Series  # one-step-ahead fitted values over the training window

    @property
    def pointwise_effect(self) -> pd.Series:
        return self.actual_post - self.counterfactual_mean

    @property
    def cumulative_effect(self) -> pd.Series:
        return self.pointwise_effect.cumsum()

    @property
    def average_effect(self) -> float:
        return float(self.pointwise_effect.mean())

    @property
    def total_effect(self) -> float:
        return float(self.cumulative_effect.iloc[-1])

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "actual": self.actual_post,
                "counterfactual": self.counterfactual_mean,
                "ci_lower": self.counterfactual_ci_lower,
                "ci_upper": self.counterfactual_ci_upper,
                "pointwise_effect": self.pointwise_effect,
                "cumulative_effect": self.cumulative_effect,
            }
        )


def fit_counterfactual(
    train: pd.DataFrame,
    post: pd.DataFrame,
    outcome_col: str,
    control_cols: list[str],
    level: str = "local level",
    alpha: float = 0.05,
) -> CounterfactualResult:
    """Fit the pre-intervention model and forecast the post-intervention counterfactual."""
    train_y = train[outcome_col]
    actual_post = post[outcome_col]

    # Normalize (z-score) the control covariates using ONLY pre-intervention
    # statistics, then apply that same transform to the post-intervention
    # covariates. The three controls have very different natural scales
    # (e.g. commodity-price YoY swings are much larger than UK CPI YoY
    # swings); putting them on a common footing keeps the regression
    # coefficients comparable and stabilizes the maximum-likelihood fit.
    # Fitting the scaler on the training window only (never on post) avoids
    # leaking post-intervention information into the counterfactual model,
    # the same discipline as the date-based train/test split itself.
    control_mean = train[control_cols].mean()
    control_std = train[control_cols].std()
    train_X = (train[control_cols] - control_mean) / control_std
    post_X = (post[control_cols] - control_mean) / control_std

    model = UnobservedComponents(endog=train_y, exog=train_X, level=level)
    fitted = model.fit(disp=False, maxiter=200)

    forecast = fitted.get_forecast(steps=len(post_X), exog=post_X)
    ci = forecast.conf_int(alpha=alpha)

    mean = forecast.predicted_mean
    mean.index = post_X.index
    ci.index = post_X.index
    lower_col, upper_col = ci.columns
    ci_lower = ci[lower_col]
    ci_upper = ci[upper_col]

    in_sample_fitted = fitted.fittedvalues
    in_sample_fitted.index = train_y.index

    return CounterfactualResult(
        outcome_name=outcome_col,
        control_columns=control_cols,
        fitted_model=fitted,
        counterfactual_mean=mean,
        counterfactual_ci_lower=ci_lower,
        counterfactual_ci_upper=ci_upper,
        actual_post=actual_post,
        in_sample_fitted=in_sample_fitted,
    )


def pre_period_backtest(
    train: pd.DataFrame,
    outcome_col: str,
    control_cols: list[str],
    holdout_months: int = 12,
    level: str = "local level",
) -> pd.DataFrame:
    """Model-validation check: hold out the last N pre-intervention months (where we
    know the policy had NOT yet changed) and verify the model forecasts them well.
    A model that cannot forecast a known "no intervention" period is not trustworthy
    for constructing the actual counterfactual.
    """
    fit_window = train.iloc[:-holdout_months]
    holdout = train.iloc[-holdout_months:]
    result = fit_counterfactual(fit_window, holdout, outcome_col, control_cols, level=level)
    return result.summary_table()
