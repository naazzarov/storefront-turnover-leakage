"""New York City replication of the leakage experiment.

The Chicago result rests on one city, which leaves open whether it reflects
something general about group-aggregate features or something particular to
Chicago's administrative geography. This module rebuilds the same task from the
NYC Department of Consumer and Worker Protection licence register
(``w7w3-xahh``) so the comparison can be made.

The two registers differ in ways that matter, and the differences are the point:
if leakage reproduces across them, it is unlikely to be an artefact of either.

  Depth      Chicago publishes every licence issued since 1995 (1.21M records).
             NYC publishes licences currently in its system (72k), with expiry
             dates starting in 2019. NYC therefore has far less history, and
             its turnover counts are correspondingly lower.

  Status     Chicago encodes early closure as a cancellation code; NYC uses a
             status vocabulary (Expired, Surrendered, Revoked, ...).

  Geography  Chicago has 77 community areas and 50 wards; NYC has 51 council
             districts and 59 community boards, over a larger population.

Because NYC's shallower history makes its turnover counts smaller, the
substantive urban finding is not expected to replicate at the same magnitude.
The evaluation finding is, since it depends on the feature construction rather
than on how much turnover exists.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .addresses import address_key

log = logging.getLogger(__name__)

# Statuses that mean the licence is no longer in force. "Expired" dominates and
# is the NYC analogue of Chicago's non-renewal.
CLOSED_STATUSES = {
    "Expired", "Surrendered", "Revoked", "Failed to Renew",
    "Out of Business", "Voided", "Close",
}
ACTIVE_STATUSES = {"Active", "Ready for Renewal", "Suspended"}

# Two spells by different licence holders at one address separated by less than
# this are treated as one occupancy, matching the Chicago succession rule.
SUCCESSION_GAP_DAYS = 30


def load(path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read the NYC licence export and parse its dates."""
    df = pd.read_csv(path, dtype=str, nrows=nrows, low_memory=False)
    for col in ("license_creation_date", "lic_expir_dd"):
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    log.info("loaded %d NYC licence records", len(df))
    return df


def attach_location_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Build the storefront key from NYC's split address fields.

    The building number and street name arrive separately, and the unit lives in
    ``apt_suite``. They are recombined into the single string ``address_key``
    expects, so the same normalization applies in both cities.
    """
    out = df.copy()
    building = out["address_building"].fillna("").astype(str).str.strip()
    street = out["address_street_name"].fillna("").astype(str).str.strip()
    unit = out["apt_suite"].fillna("").astype(str).str.strip()

    combined = (building + " " + street).str.strip()
    combined = np.where(unit.ne(""), combined + " #" + unit, combined)

    out["location_id"] = [
        address_key(a, c, s, z)
        for a, c, s, z in zip(combined, out.get("address_city"),
                              out.get("address_state"), out.get("address_zip"))
    ]
    n_bad = int((out["location_id"] == "").sum())
    log.info("%d/%d NYC licences have no usable address key", n_bad, len(out))
    return out


def build_tenancies(df: pd.DataFrame, *, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Collapse licences into tenancies, mirroring the Chicago procedure.

    Licences sharing a licence holder at one address are merged, then adjacent
    spells by different holders are merged when they abut within
    ``SUCCESSION_GAP_DAYS``.
    """
    work = df.loc[(df["location_id"] != "") & df["license_creation_date"].notna()].copy()

    if snapshot_date is None:
        snapshot_date = work["license_creation_date"].max()

    work["is_closed"] = work["license_status"].isin(CLOSED_STATUSES)
    # A licence still in force has no observed end; cap the rest at the snapshot
    # so nothing is credited with unelapsed time, as in Chicago.
    end = work["lic_expir_dd"].where(work["is_closed"], pd.NaT)
    work["observed_end"] = end.clip(upper=snapshot_date).fillna(snapshot_date)
    work = work.loc[work["observed_end"] >= work["license_creation_date"]]

    holder = work["business_unique_id"].fillna(work["license_nbr"])
    work["holder"] = holder

    occ = work.groupby(["location_id", "holder"], sort=False).agg(
        start=("license_creation_date", "min"),
        end=("observed_end", "max"),
        n_licences=("license_nbr", "size"),
        closed=("is_closed", "all"),
        category=("business_category", "first"),
        council_district=("council_district", "first"),
        community_board=("community_board", "first"),
        borough=("address_borough", "first"),
    ).reset_index()

    occ = occ.sort_values(["location_id", "start"])
    g = occ.groupby("location_id", sort=False)
    prev_end = g["end"].transform(lambda s: s.cummax().shift(1))
    starts_new = prev_end.isna() | ((occ["start"] - prev_end).dt.days > SUCCESSION_GAP_DAYS)
    occ["tenancy_id"] = starts_new.groupby(occ["location_id"]).cumsum()

    ten = occ.groupby(["location_id", "tenancy_id"], sort=False).agg(
        start=("start", "min"), end=("end", "max"),
        n_licences=("n_licences", "sum"), closed=("closed", "all"),
        category=("category", "first"),
        council_district=("council_district", "first"),
        community_board=("community_board", "first"),
        borough=("borough", "first"),
    ).reset_index()
    ten["duration_days"] = (ten["end"] - ten["start"]).dt.days

    log.info("NYC: %d licences -> %d occupancies -> %d tenancies",
             len(work), len(occ), len(ten))
    return ten


def build_locations(tenancies: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tenancies to storefronts, with exposure."""
    snapshot = tenancies["end"].max()
    g = tenancies.groupby("location_id", sort=False)

    loc = g.agg(
        n_tenants=("tenancy_id", "size"),
        n_closed=("closed", "sum"),
        first_seen=("start", "min"),
        last_seen=("end", "max"),
        mean_duration_days=("duration_days", "mean"),
        council_district=("council_district", "first"),
        community_board=("community_board", "first"),
        borough=("borough", "first"),
        category=("category", "first"),
    ).reset_index()

    loc["exposure_years"] = (snapshot - loc["first_seen"]).dt.days / 365.25
    loc["closure_rate"] = loc["n_closed"] / loc["n_tenants"]

    log.info("NYC locations: %d | %d multi-tenant | max tenants %d",
             len(loc), int((loc["n_tenants"] > 1).sum()), int(loc["n_tenants"].max()))
    return loc


def group_context(
    locations: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("council_district", "community_board", "borough"),
    value_col: str = "n_tenants",
) -> pd.DataFrame:
    """Leave-one-out group means, the feature family under test.

    Identical in construction to the Chicago area aggregates: (group sum minus
    own value) divided by (group count minus one).
    """
    df = locations.copy()
    for col in group_cols:
        if col not in df:
            continue
        vals = df[value_col].astype(float)
        grp = df.groupby(col, sort=False)
        total = grp[value_col].transform("sum")
        count = grp[value_col].transform("count")
        df[f"{value_col}_{col}_loo"] = (total - vals.fillna(0)) / (count - vals.notna().astype(int))
    return df


def label(locations: pd.DataFrame, *, quantile: float = 0.95,
          min_exposure_years: float = 3.0) -> pd.DataFrame:
    """Flag excess turnover, using the exposure-adjusted rate.

    The negative-binomial excess definition used for Chicago needs more history
    than NYC provides, so the simpler rate definition is used here. Chicago's
    own results were shown to be invariant across definitions, so this does not
    compromise the comparison.
    """
    df = locations.loc[locations["exposure_years"] >= min_exposure_years].copy()
    df["turnover_rate"] = df["n_tenants"] / df["exposure_years"].clip(lower=0.5)
    cutoff = df["turnover_rate"].quantile(quantile)
    df["is_cursed"] = df["turnover_rate"] >= cutoff
    log.info("NYC labelled: %d locations, %d positive (%.1f%%)",
             len(df), int(df["is_cursed"].sum()), 100 * df["is_cursed"].mean())
    return df
