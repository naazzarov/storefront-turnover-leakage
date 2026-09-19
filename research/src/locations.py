"""Group businesses into physical locations and reconstruct tenancy sequences.

A *location* is a storefront that one or more businesses have occupied over
time. Yelp has no such entity, so we build it by grouping businesses on a key
that must satisfy two competing requirements:

  It must merge  the same storefront written different ways ("475 3rd St" vs
                 "475 3rd Street"), or genuine turnover is invisible.

  It must split  distinct storefronts that share an address string (suite 100 vs
                 suite 200; the same street number in two different cities), or
                 turnover is manufactured out of nothing.

``addresses.address_key`` handles the textual side. This module adds the
geographic check: businesses sharing an address key but sitting far apart are
almost certainly different places whose text merely collided, and are split
apart. The coordinate check is a guard against false merges, not a clustering
method in its own right -- Yelp coordinates are geocoded from the address, so
they are not independent evidence, and treating them as such would overstate
our confidence.

The output is one row per location with its ordered tenancy sequence, which is
the unit of analysis for the rest of the study.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .addresses import address_key

log = logging.getLogger(__name__)

# Businesses sharing an address key but separated by more than this are treated
# as distinct locations. Set well above typical geocoding jitter (tens of
# metres) so that ordinary imprecision does not split a real storefront.
MAX_INTRA_LOCATION_METERS = 250.0

# Gap between one tenant's last review and the next tenant's first review.
# A large positive gap means the unit sat empty; a negative gap means the two
# businesses overlapped, which usually indicates they were never the same
# storefront (a food court, a mall, a shared building).
MAX_TENANCY_OVERLAP_DAYS = 180


def _haversine_meters(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in metres between paired coordinate arrays."""
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def assign_address_keys(businesses: pd.DataFrame) -> pd.DataFrame:
    """Attach the textual location key to each business.

    Businesses whose address cannot produce a key (missing, or no house number)
    get an empty key and are excluded from location-level analysis. They are not
    dropped here so that the exclusion can be counted and reported.
    """
    df = businesses.copy()
    df["address_key"] = [
        address_key(a, c, s, p)
        for a, c, s, p in zip(
            df.get("address"), df.get("city"), df.get("state"), df.get("postal_code")
        )
    ]
    n_unusable = int((df["address_key"] == "").sum())
    if n_unusable:
        log.info(
            "%d/%d businesses (%.1f%%) have no usable address key",
            n_unusable, len(df), 100 * n_unusable / len(df),
        )
    return df


def split_by_geography(
    businesses: pd.DataFrame,
    *,
    max_meters: float = MAX_INTRA_LOCATION_METERS,
) -> pd.DataFrame:
    """Split address-key groups whose members are geographically far apart.

    Within each address key, members are compared to the group's median
    coordinate. Those beyond ``max_meters`` are moved into a separate location.
    The median is used rather than the mean because a single mis-geocoded row
    would drag a mean far enough to split an otherwise sound group.

    Adds ``location_id``, which supersedes ``address_key`` downstream.
    """
    df = businesses.copy()
    usable = df["address_key"] != ""

    df["location_id"] = df["address_key"]

    coords_ok = usable & df["latitude"].notna() & df["longitude"].notna()
    sub = df.loc[coords_ok, ["address_key", "latitude", "longitude"]]
    if sub.empty:
        return df

    med = sub.groupby("address_key").agg(
        med_lat=("latitude", "median"), med_lon=("longitude", "median")
    )
    joined = sub.join(med, on="address_key")
    dist = _haversine_meters(
        joined["latitude"].to_numpy(),
        joined["longitude"].to_numpy(),
        joined["med_lat"].to_numpy(),
        joined["med_lon"].to_numpy(),
    )
    outlier = pd.Series(dist > max_meters, index=sub.index)

    n_split = int(outlier.sum())
    if n_split:
        # Suffix outliers so they form their own location rather than silently
        # joining a group they do not belong to.
        df.loc[outlier[outlier].index, "location_id"] = (
            df.loc[outlier[outlier].index, "address_key"] + "|geo-outlier"
        )
        log.info(
            "%d businesses moved out of their address group (>%.0fm from group median)",
            n_split, max_meters,
        )
    return df


def build_tenancies(
    businesses: pd.DataFrame,
    lifespans: pd.DataFrame,
) -> pd.DataFrame:
    """Order the businesses at each location into a tenancy sequence.

    Args:
        businesses: business table carrying ``location_id``.
        lifespans: output of ``lifespan.build_lifespans``.

    Returns:
        One row per business, ordered within location by first review, carrying
        ``tenancy_index``, the ``gap_days`` to the previous tenant, and
        ``overlaps_previous``.
    """
    cols = ["business_id", "first_review", "last_review", "n_reviews",
            "is_open", "is_right_censored", "is_left_censored",
            "duration_days", "observed_days"]
    # ``lifespans`` carries the cleaned is_open (and n_reviews recomputed from
    # the review trail rather than the snapshot count), so drop the business
    # table's copies instead of letting the merge suffix them.
    left = businesses.drop(columns=[c for c in cols if c != "business_id"], errors="ignore")
    df = left.merge(lifespans[cols], on="business_id", how="left")

    df = df.loc[df["location_id"].notna() & (df["location_id"] != "")].copy()
    # A business with no reviews has no position in time and cannot be ordered.
    df = df.loc[df["first_review"].notna()]

    df = df.sort_values(["location_id", "first_review", "business_id"])
    g = df.groupby("location_id", sort=False)

    df["tenancy_index"] = g.cumcount()
    df["n_tenants"] = g["business_id"].transform("size")

    prev_last = g["last_review"].shift(1)
    df["gap_days"] = (df["first_review"] - prev_last).dt.days
    df["overlaps_previous"] = df["gap_days"] < -MAX_TENANCY_OVERLAP_DAYS

    return df


def build_locations(tenancies: pd.DataFrame) -> pd.DataFrame:
    """Collapse tenancy rows into one row per location.

    Returns the location-level table that the cursed-label definition and all
    downstream modelling operate on.
    """
    g = tenancies.groupby("location_id", sort=False)

    loc = g.agg(
        n_tenants=("business_id", "size"),
        n_closed=("is_open", lambda s: int((s == 0).sum())),
        first_seen=("first_review", "min"),
        last_seen=("last_review", "max"),
        total_reviews=("n_reviews", "sum"),
        mean_tenant_reviews=("n_reviews", "mean"),
        any_overlap=("overlaps_previous", "any"),
        any_left_censored=("is_left_censored", "any"),
        all_right_censored=("is_right_censored", "all"),
        mean_duration_days=("duration_days", "mean"),
        median_duration_days=("duration_days", "median"),
        min_duration_days=("duration_days", "min"),
        n_usable_durations=("duration_days", "count"),
        mean_gap_days=("gap_days", "mean"),
    ).reset_index()

    loc["closure_rate"] = loc["n_closed"] / loc["n_tenants"]
    loc["observed_span_days"] = (loc["last_seen"] - loc["first_seen"]).dt.days

    # Turnover per decade of observation, the scale-free turnover measure.
    # Locations observed briefly can show a high rate on thin evidence, so
    # ``observed_span_days`` is carried alongside for filtering.
    span_years = loc["observed_span_days"] / 365.25
    loc["tenants_per_decade"] = np.where(
        span_years >= 1.0, 10.0 * loc["n_tenants"] / span_years, np.nan
    )

    # Attach a stable, human-readable descriptor. The location_id embeds the
    # normalized address, which Yelp's terms forbid publishing, so anything
    # that reaches a figure or table must use the opaque id instead.
    loc["city_state"] = loc["location_id"].str.split("|").str[:2].str.join("|")

    return loc


def location_pipeline(
    businesses: pd.DataFrame,
    lifespans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full grouping pipeline.

    Returns ``(tenancies, locations)``.
    """
    keyed = assign_address_keys(businesses)
    split = split_by_geography(keyed)
    tenancies = build_tenancies(split, lifespans)
    locations = build_locations(tenancies)

    log.info(
        "locations: %d total | %d multi-tenant | %d single-tenant",
        len(locations),
        int((locations["n_tenants"] > 1).sum()),
        int((locations["n_tenants"] == 1).sum()),
    )
    return tenancies, locations
