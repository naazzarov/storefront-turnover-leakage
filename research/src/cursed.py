"""Operationalize the notion of a "cursed" storefront.

The informal claim is that some addresses kill businesses: tenant after tenant
opens there and fails, while comparable addresses nearby sustain one occupant
for a decade. Turning that into a measurable quantity is the central conceptual
step of the study, and the obvious definition -- "many tenants" -- is wrong in
three ways that all inflate the phenomenon.

  Exposure          A storefront first licensed in 1996 has had thirty years to
                    churn; one first licensed in 2022 has had four. Counting raw
                    tenants makes old addresses look cursed and new ones look
                    stable, measuring nothing but age.

  Sector base rate  Restaurants and bars fail far more often than dentists and
                    law offices. An address that has hosted five restaurants is
                    not remarkable; the sector is. Without adjustment the study
                    would rediscover that hospitality is hard and label it a
                    property of the building.

  Geography         Some commercial strips churn throughout. A location in a
                    high-turnover corridor inherits that rate, and flagging it
                    as individually cursed confuses the street with the unit.

A cursed location is therefore one that turns over *more than its own context
predicts* -- more than its exposure, sector mix and neighbourhood jointly
account for. We fit that expectation with a negative-binomial count model using
log-exposure as an offset, and define curse as the upper tail of the resulting
excess. A location is not cursed for being unlucky; it is cursed for being
unlucky beyond explanation.

Three definitions are provided. ``EXCESS`` is the primary one used for the
headline results; ``RATE`` and ``COUNT`` are simpler alternatives retained so the
paper can show the findings are not an artefact of one arbitrary choice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# A location needs this much observed time before its turnover rate means
# anything. Below it, one tenancy and one vacancy can produce an extreme rate.
MIN_EXPOSURE_YEARS = 8.0

# Locations must have hosted at least one tenancy that actually ended, or there
# is no turnover to speak of.
MIN_TENANCIES = 1

# Fraction of locations flagged as cursed under the tail-based definitions.
DEFAULT_TAIL_QUANTILE = 0.95


class CurseDefinition(str, Enum):
    """The three operational definitions, in increasing order of adjustment."""

    COUNT = "count"      # raw tenant count above a threshold
    RATE = "rate"        # tenants per decade of exposure, upper tail
    EXCESS = "excess"    # upper tail of turnover in excess of model expectation


@dataclass(frozen=True)
class CurseConfig:
    definition: CurseDefinition = CurseDefinition.EXCESS
    min_exposure_years: float = MIN_EXPOSURE_YEARS
    min_tenancies: int = MIN_TENANCIES
    tail_quantile: float = DEFAULT_TAIL_QUANTILE
    count_threshold: int = 4  # used only by CurseDefinition.COUNT


def compute_exposure(
    locations: pd.DataFrame,
    *,
    snapshot_date: pd.Timestamp,
    coverage_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Attach the observation window over which turnover could have been seen.

    Exposure runs from when the location entered observation to the snapshot.
    A location's own ``first_seen`` is used, floored at ``coverage_start``: a
    storefront that first appears in 2015 was genuinely unobservable before
    then, and crediting it with exposure back to 1996 would understate its rate.

    Using first_seen as the start does mean exposure is measured from the
    location's first *known* tenancy, so a unit that sat vacant for years before
    that is credited with no exposure for the vacancy. This biases rates upward
    for late-appearing locations, which is why ``min_exposure_years`` excludes
    the short-window cases where the bias bites hardest.
    """
    df = locations.copy()
    start = df["first_seen"]
    if coverage_start is not None:
        start = start.clip(lower=coverage_start)

    df["exposure_days"] = (snapshot_date - start).dt.days
    df["exposure_years"] = df["exposure_days"] / 365.25
    df["snapshot_date"] = snapshot_date
    return df


def eligible(df: pd.DataFrame, config: CurseConfig = CurseConfig()) -> pd.Series:
    """Boolean mask of locations with enough observation to be judged."""
    return (
        (df["exposure_years"] >= config.min_exposure_years)
        & (df["n_tenants"] >= config.min_tenancies)
        & df["exposure_years"].notna()
    )


def _dominant_sector(tenancies: pd.DataFrame) -> pd.DataFrame:
    """The licence class a location has hosted most often.

    Used as the sector control. Taken over the location's whole history rather
    than its current tenant, since the question is what kind of premises it is.
    """
    counts = (
        tenancies.groupby(["location_id", "licence_description"], sort=False)
        .size()
        .rename("n")
        .reset_index()
    )
    idx = counts.groupby("location_id", sort=False)["n"].idxmax()
    return (
        counts.loc[idx, ["location_id", "licence_description"]]
        .rename(columns={"licence_description": "sector"})
        .reset_index(drop=True)
    )


def fit_expected_turnover(
    df: pd.DataFrame,
    *,
    sector_col: str = "sector",
    area_col: str = "community_area",
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Predict each location's tenant count from exposure, sector and area.

    A negative-binomial GLM with ``log(exposure_years)`` as offset, so the
    coefficients describe turnover *rates* rather than counts. Negative binomial
    rather than Poisson because turnover is strongly overdispersed -- a Poisson
    fit would understate the variance and flag far too many locations as
    extreme.

    Sector and area levels with fewer than ``min_group_size`` locations are
    pooled into an "other" category; rare levels otherwise get fitted almost
    exactly, which would drive their residuals to zero and hide real curses.

    Adds ``expected_tenants``, ``excess_tenants`` and ``turnover_ratio``.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    work = df.copy()

    for col in (sector_col, area_col):
        if col not in work:
            work[col] = "unknown"
        vals = work[col].fillna("unknown").astype(str)
        counts = vals.value_counts()
        rare = counts[counts < min_group_size].index
        work[col] = vals.where(~vals.isin(rare), "other")

    work["log_exposure"] = np.log(work["exposure_years"].clip(lower=0.5))

    formula = f"n_tenants ~ C({sector_col}) + C({area_col})"
    try:
        model = smf.glm(
            formula=formula,
            data=work,
            family=sm.families.NegativeBinomial(alpha=1.0),
            offset=work["log_exposure"],
        ).fit()
        work["expected_tenants"] = model.predict(work, offset=work["log_exposure"])
        log.info("turnover model fitted on %d locations (df=%d)",
                 int(model.nobs), int(model.df_model))
    except Exception as exc:  # noqa: BLE001 - fall back rather than abort
        # A rank-deficient design or a separation failure should degrade the
        # analysis, not kill it; the simpler expectation is still usable.
        log.warning("negative-binomial fit failed (%s); falling back to "
                    "exposure-only expectation", exc)
        rate = work["n_tenants"].sum() / work["exposure_years"].sum()
        work["expected_tenants"] = rate * work["exposure_years"]

    work["excess_tenants"] = work["n_tenants"] - work["expected_tenants"]
    work["turnover_ratio"] = work["n_tenants"] / work["expected_tenants"].clip(lower=1e-6)

    return work


def label_cursed(
    locations: pd.DataFrame,
    tenancies: pd.DataFrame | None = None,
    *,
    snapshot_date: pd.Timestamp | None = None,
    config: CurseConfig = CurseConfig(),
) -> pd.DataFrame:
    """Attach the cursed label under the configured definition.

    Returns the location table restricted to eligible rows, carrying
    ``is_cursed`` plus the intermediate quantities the label rests on, so that
    every flagged location can be audited.
    """
    if snapshot_date is None:
        snapshot_date = locations["last_seen"].max()

    df = compute_exposure(locations, snapshot_date=snapshot_date)

    if tenancies is not None and "licence_description" in tenancies:
        df = df.merge(_dominant_sector(tenancies), on="location_id", how="left")

    mask = eligible(df, config)
    log.info(
        "eligible: %d/%d locations (>=%.0f years exposure)",
        int(mask.sum()), len(df), config.min_exposure_years,
    )
    df = df.loc[mask].copy()

    if config.definition is CurseDefinition.COUNT:
        df["curse_score"] = df["n_tenants"].astype(float)
        df["is_cursed"] = df["n_tenants"] >= config.count_threshold

    elif config.definition is CurseDefinition.RATE:
        df["curse_score"] = 10.0 * df["n_tenants"] / df["exposure_years"]
        cutoff = df["curse_score"].quantile(config.tail_quantile)
        df["is_cursed"] = df["curse_score"] >= cutoff
        log.info("rate cutoff at q%.2f: %.2f tenants/decade",
                 config.tail_quantile, cutoff)

    elif config.definition is CurseDefinition.EXCESS:
        df = fit_expected_turnover(df)
        df["curse_score"] = df["excess_tenants"]
        cutoff = df["curse_score"].quantile(config.tail_quantile)
        df["is_cursed"] = df["curse_score"] >= cutoff
        log.info("excess cutoff at q%.2f: %.2f tenants above expectation",
                 config.tail_quantile, cutoff)

    else:  # pragma: no cover - the enum is exhaustive
        raise ValueError(f"unknown definition {config.definition}")

    df["curse_definition"] = config.definition.value
    log.info("cursed: %d/%d locations (%.1f%%)",
             int(df["is_cursed"].sum()), len(df), 100 * df["is_cursed"].mean())
    return df


def definition_agreement(
    locations: pd.DataFrame,
    tenancies: pd.DataFrame | None = None,
    *,
    snapshot_date: pd.Timestamp | None = None,
    base: CurseConfig = CurseConfig(),
) -> pd.DataFrame:
    """Compare which locations each definition flags.

    A finding that holds only under one arbitrary definition is an artefact of
    that choice. This table goes in the paper's robustness section: it reports
    how far the three definitions overlap, and therefore how much the results
    depend on the operationalization.
    """
    labelled: dict[str, pd.Series] = {}
    for definition in CurseDefinition:
        cfg = CurseConfig(
            definition=definition,
            min_exposure_years=base.min_exposure_years,
            min_tenancies=base.min_tenancies,
            tail_quantile=base.tail_quantile,
            count_threshold=base.count_threshold,
        )
        out = label_cursed(locations, tenancies, snapshot_date=snapshot_date, config=cfg)
        labelled[definition.value] = out.set_index("location_id")["is_cursed"]

    joined = pd.DataFrame(labelled).fillna(False)

    rows = []
    for a in joined.columns:
        for b in joined.columns:
            both = int((joined[a] & joined[b]).sum())
            either = int((joined[a] | joined[b]).sum())
            rows.append({
                "definition_a": a,
                "definition_b": b,
                "jaccard": both / either if either else np.nan,
                "n_both": both,
            })
    return pd.DataFrame(rows)
