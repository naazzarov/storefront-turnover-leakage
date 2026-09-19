"""Features describing a storefront and its surroundings.

The study's question is whether turnover is a property of the *unit* or of its
*surroundings*. A finding that cursed locations sit in high-turnover districts
would be no finding at all -- it would say only that some commercial strips
churn, which is already well known. The interesting claim is that a specific
address underperforms the storefronts on either side of it.

Testing that requires describing each location's neighbourhood well enough to
control it away, which makes two things load-bearing.

Leave-one-out.  Every contextual feature is computed with the focal location
excluded. A neighbourhood turnover rate that includes the focal storefront is
partly a copy of the label, and a model fitted on it will appear to predict
turnover while really just reading it back. For small neighbourhoods this
inflation is severe, and it is invisible in the fit statistics.

Spatial scale.  "Neighbourhood" is ambiguous: an administrative area averages
over thousands of premises, while a 50-metre radius may hold three. Both are
computed, since a curse confined to one unit and a curse shared with immediate
neighbours are different phenomena and the contrast between the two scales is
itself informative.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Radii, in metres, at which the immediate spatial context is summarized.
# 50m is roughly the adjacent storefronts; 200m is the block; 500m the strip.
NEIGHBOUR_RADII_M = (50.0, 200.0, 500.0)

# Minimum neighbours required before a leave-one-out statistic is trusted.
MIN_NEIGHBOURS = 3

_EARTH_RADIUS_M = 6_371_000.0


def _to_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Project lat/lon to local metres via equirectangular approximation.

    Accurate to well under a metre at the radii used here, and lets the
    neighbour search run in a KD-tree rather than by great-circle distance.
    """
    lat0 = np.nanmean(lat)
    x = np.radians(lon) * np.cos(np.radians(lat0)) * _EARTH_RADIUS_M
    y = np.radians(lat) * _EARTH_RADIUS_M
    return np.column_stack([x, y])


def spatial_context(
    locations: pd.DataFrame,
    *,
    radii_m: tuple[float, ...] = NEIGHBOUR_RADII_M,
    value_cols: tuple[str, ...] = ("n_tenants", "tenants_per_decade"),
    min_neighbours: int = MIN_NEIGHBOURS,
) -> pd.DataFrame:
    """Summarize each location's neighbours, excluding the location itself.

    For every radius and every value column, adds the leave-one-out mean over
    neighbours within that radius, plus the neighbour count. Locations with
    fewer than ``min_neighbours`` get NaN rather than a mean over one or two
    premises, which would be noise presented as context.
    """
    from scipy.spatial import cKDTree

    df = locations.copy()
    has_xy = df["latitude"].notna() & df["longitude"].notna()
    sub = df.loc[has_xy]
    if len(sub) < 2:
        log.warning("too few geocoded locations for spatial context")
        return df

    xy = _to_xy(sub["latitude"].to_numpy(), sub["longitude"].to_numpy())
    tree = cKDTree(xy)

    for radius in radii_m:
        # Counts including self, so subtract one for the leave-one-out count.
        n_within = tree.query_ball_point(xy, r=radius, return_length=True)
        n_neighbours = n_within - 1

        col_n = f"n_within_{int(radius)}m"
        df.loc[sub.index, col_n] = n_neighbours

        for value_col in value_cols:
            if value_col not in sub:
                continue
            values = sub[value_col].to_numpy(dtype=float)
            filled = np.nan_to_num(values, nan=0.0)
            valid = (~np.isnan(values)).astype(float)

            # Sum over the neighbourhood including self, then remove self:
            # a KD-tree gives no direct leave-one-out sum, and subtracting is
            # exact rather than approximate.
            pairs = tree.query_ball_point(xy, r=radius)
            sums = np.fromiter((filled[idx].sum() for idx in pairs), dtype=float, count=len(pairs))
            counts = np.fromiter((valid[idx].sum() for idx in pairs), dtype=float, count=len(pairs))

            loo_sum = sums - filled
            loo_count = counts - valid
            with np.errstate(invalid="ignore", divide="ignore"):
                loo_mean = np.where(loo_count >= min_neighbours, loo_sum / loo_count, np.nan)

            df.loc[sub.index, f"{value_col}_nbr_{int(radius)}m"] = loo_mean

    log.info("spatial context computed at radii %s", radii_m)
    return df


def group_context(
    locations: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("community_area", "ward"),
    value_cols: tuple[str, ...] = ("n_tenants", "tenants_per_decade"),
    min_group_size: int = MIN_NEIGHBOURS + 1,
) -> pd.DataFrame:
    """Leave-one-out group means for administrative areas.

    Computed as (group sum - own value) / (group count - 1), which is exact and
    avoids the quadratic cost of grouping repeatedly.
    """
    df = locations.copy()
    for group_col in group_cols:
        if group_col not in df:
            continue
        g = df.groupby(group_col, sort=False)
        size = g[group_col].transform("size")
        for value_col in value_cols:
            if value_col not in df:
                continue
            vals = df[value_col].astype(float)
            total = g[value_col].transform("sum")
            count = g[value_col].transform("count")
            loo = (total - vals.fillna(0)) / (count - vals.notna().astype(int))
            df[f"{value_col}_{group_col}_loo"] = loo.where(size >= min_group_size)
    return df


def building_context(locations: pd.DataFrame) -> pd.DataFrame:
    """Describe the building each unit sits in.

    A storefront in a large multi-unit building faces different dynamics from a
    standalone shop. The building is approximated by the street address with the
    unit designator stripped, which ``location_id`` carries after the "#".
    """
    df = locations.copy()
    df["building_id"] = df["location_id"].str.split(" #").str[0]

    g = df.groupby("building_id", sort=False)
    df["units_in_building"] = g["location_id"].transform("size")
    df["is_multi_unit_building"] = df["units_in_building"] > 1

    # Leave-one-out turnover among the other units of the same building: a
    # building-wide problem (bad landlord, bad structure) looks different from
    # one unit failing while its neighbours in the same building thrive.
    vals = df["n_tenants"].astype(float)
    total = g["n_tenants"].transform("sum")
    count = g["n_tenants"].transform("count")
    loo = (total - vals.fillna(0)) / (count - vals.notna().astype(int))
    df["n_tenants_building_loo"] = loo.where(df["units_in_building"] > 1)

    return df


def vacancy_features(locations: pd.DataFrame, tenancies: pd.DataFrame) -> pd.DataFrame:
    """Summarize how long a storefront sits empty between occupants.

    Vacancy separates two distinct failure modes. A unit that is re-let within
    weeks is in demand despite its churn; one that stands empty for years is
    being avoided. Both look identical in a turnover count.
    """
    df = locations.copy()

    ten = tenancies.sort_values(["location_id", "start"]).copy()
    g = ten.groupby("location_id", sort=False)
    # observed_end caps licences that run past the data cut, so a currently
    # trading tenant cannot appear to have vacated before its successor.
    prev_end = g["observed_end"].transform(lambda s: s.cummax().shift(1))
    ten["vacancy_days"] = (ten["start"] - prev_end).dt.days

    agg = ten.groupby("location_id", sort=False)["vacancy_days"].agg(
        total_vacancy_days="sum",
        max_vacancy_days="max",
        n_vacancies="count",
    ).reset_index()

    df = df.merge(agg, on="location_id", how="left")

    # Share of the observation window spent empty. Capped at 1 because vacancy
    # inferred from overlapping tenancies can otherwise exceed the window.
    with np.errstate(invalid="ignore", divide="ignore"):
        df["vacancy_share"] = (
            df["total_vacancy_days"] / df["observed_span_days"].replace(0, np.nan)
        ).clip(0, 1)

    return df


def tenure_features(locations: pd.DataFrame, tenancies: pd.DataFrame) -> pd.DataFrame:
    """Per-location summaries of how its tenancies ended.

    Durations are deliberately *not* summarized as a mean here. Licence terms
    quantize duration to renewal multiples, so a mean tenure is largely a
    statement about Chicago's renewal calendar. What survives that quantization
    is the share of tenancies that ended early, which uses the exact cancellation
    date rather than an inferred one.
    """
    df = locations.copy()
    g = tenancies.groupby("location_id", sort=False)

    agg = g.agg(
        share_closed_early=("closed_early", "mean"),
        n_renewal_licences=("n_licences", "sum"),
        mean_licences_per_tenancy=("n_licences", "mean"),
        share_single_licence=("n_licences", lambda s: float((s == 1).mean())),
    ).reset_index()

    # A tenancy that never renewed once is the sharpest available signal of a
    # short, failed occupancy, and it is immune to the quantization problem.
    return df.merge(agg, on="location_id", how="left")


def build_features(
    locations: pd.DataFrame,
    tenancies: pd.DataFrame,
    *,
    radii_m: tuple[float, ...] = NEIGHBOUR_RADII_M,
) -> pd.DataFrame:
    """Assemble the full feature table for modelling."""
    df = building_context(locations)
    df = vacancy_features(df, tenancies)
    df = tenure_features(df, tenancies)
    df = group_context(df)
    df = spatial_context(df, radii_m=radii_m)

    log.info("features: %d locations x %d columns", len(df), df.shape[1])
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs, excluding anything that leaks the outcome.

    ``n_tenants`` and everything derived from the focal location's own turnover
    is excluded: those define the label. Leave-one-out neighbourhood columns are
    kept, since they describe the surroundings rather than the location itself.
    """
    leaky = {
        "n_tenants", "tenants_per_decade", "n_closed_early", "early_closure_rate",
        "curse_score", "is_cursed", "excess_tenants", "expected_tenants",
        "turnover_ratio", "share_closed_early", "n_renewal_licences",
        "mean_licences_per_tenancy", "share_single_licence",
        "n_vacancies", "total_vacancy_days", "max_vacancy_days", "vacancy_share",
        "mean_duration_days", "median_duration_days", "min_duration_days",
        "mean_vacancy_days", "n_censored", "observed_span_days",
    }
    numeric = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in leaky]
