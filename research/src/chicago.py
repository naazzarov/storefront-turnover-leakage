"""Ingest Chicago business licence records and reconstruct storefront tenancies.

The City of Chicago publishes every business licence it has issued since 1995
(dataset ``r5kz-chrr``, public domain). Unlike the Yelp dataset this records
real dates -- when a licence began, when it was due to expire, and whether it
was cancelled before that -- which makes tenancy observable rather than inferred.

The central difficulty is that a licence is not a storefront.

  One business holds many licences.  Licences are renewed every one or two
      years, so a shop trading continuously for a decade appears as five or ten
      separate rows. Treating each row as a tenancy would report enormous
      turnover everywhere and the study would be meaningless.

  One business holds many licence *types* at once.  A restaurant may
      simultaneously hold a food licence, a liquor licence and a sidewalk-cafe
      permit. These are concurrent, not sequential.

  A licence can lapse without the business closing.  Ownership restructuring, a
      name change or a switch of licence class ends one licence and starts
      another while trading never stops. This is the main threat to validity in
      this design: it inflates apparent turnover.

We therefore collapse licences into *tenancies* in two stages. Licences sharing
an account number at one address are merged into a single occupancy spanning
their combined date range, which absorbs renewals and concurrent licence types.
Consecutive tenancies by *different* account holders at one address are then
merged when they abut closely enough that a handover, rather than a vacancy, is
the likelier explanation (see ``SUCCESSION_GAP_DAYS``).

The result is one row per (address, occupant) with real start and end dates.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .addresses import address_key

log = logging.getLogger(__name__)

# Licence status codes used by the City of Chicago.
#   AAI  issued and in force
#   AAC  cancelled during term -- the business stopped before expiry
#   REV  revoked by the city
#   REA  revoked, under appeal
#   INQ  in inquiry
STATUS_CANCELLED = "AAC"
STATUS_REVOKED = "REV"
STATUS_ISSUED = "AAI"

# Two licence spells by the same account at one address separated by less than
# this are treated as one continuous occupancy. Renewals are usually contiguous
# or overlap slightly; a genuine reopening after a year is a separate tenancy.
RENEWAL_GAP_DAYS = 200

# Two spells by *different* accounts at one address separated by less than this
# are treated as the same occupancy under a new licence holder (restructuring,
# rename, licence-class change) rather than as turnover. Deliberately short:
# merging aggressively here suppresses the very signal the study measures, so
# the bias is kept in the conservative direction and the sensitivity of results
# to this threshold is reported in the paper.
SUCCESSION_GAP_DAYS = 30

# Licence classes that do not correspond to a public storefront. Excluding them
# keeps the analysis on retail premises, where "cursed location" is meaningful.
NON_STOREFRONT_LICENCE_PATTERNS = (
    "home occupation",
    "peddler",
    "mobile",
    "itinerant",
    "expediter",
    "day labor",
    "valet",
    "residential real estate",
)

USED_COLUMNS = {
    "ID": "record_id",
    "LICENSE ID": "licence_id",
    "ACCOUNT NUMBER": "account_number",
    "SITE NUMBER": "site_number",
    "LEGAL NAME": "legal_name",
    "DOING BUSINESS AS NAME": "dba_name",
    "ADDRESS": "address",
    "CITY": "city",
    "STATE": "state",
    "ZIP CODE": "zip_code",
    "WARD": "ward",
    "COMMUNITY AREA NAME": "community_area",
    "NEIGHBORHOOD": "neighborhood",
    "LICENSE CODE": "licence_code",
    "LICENSE DESCRIPTION": "licence_description",
    "BUSINESS ACTIVITY": "business_activity",
    "APPLICATION TYPE": "application_type",
    "LICENSE STATUS": "licence_status",
    "LICENSE TERM START DATE": "term_start",
    "LICENSE TERM EXPIRATION DATE": "term_end",
    "DATE ISSUED": "date_issued",
    "LICENSE STATUS CHANGE DATE": "status_change_date",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
}

_DATE_COLUMNS = ("term_start", "term_end", "date_issued", "status_change_date")


def load_licences(path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read the raw licence CSV, keeping and renaming the columns we use."""
    available = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in USED_COLUMNS if c in available]
    missing = set(USED_COLUMNS) - set(usecols)
    if missing:
        log.warning("licence export is missing expected columns: %s", sorted(missing))

    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        nrows=nrows,
        low_memory=False,
    ).rename(columns=USED_COLUMNS)

    for col in _DATE_COLUMNS:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    for col in ("latitude", "longitude"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("loaded %d licence records", len(df))
    return df


def filter_storefronts(df: pd.DataFrame) -> pd.DataFrame:
    """Drop licence classes that do not denote a fixed public storefront."""
    if "licence_description" not in df:
        return df
    desc = df["licence_description"].fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for pattern in NON_STOREFRONT_LICENCE_PATTERNS:
        mask |= desc.str.contains(pattern, regex=False)
    kept = df.loc[~mask].copy()
    log.info(
        "storefront filter: kept %d/%d licences (%.1f%%)",
        len(kept), len(df), 100 * len(kept) / max(len(df), 1),
    )
    return kept


def attach_location_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the storefront key from the licence address.

    Chicago addresses carry unit designators inline ("222 S RIVERSIDE PLZ 6 610"),
    which ``address_key`` already separates, so suites stay distinct.
    """
    out = df.copy()
    out["location_id"] = [
        address_key(a, c, s, z)
        for a, c, s, z in zip(
            out.get("address"), out.get("city"), out.get("state"), out.get("zip_code")
        )
    ]
    n_bad = int((out["location_id"] == "").sum())
    if n_bad:
        log.info("%d/%d licences have no usable address key", n_bad, len(out))
    return out


def _effective_end(df: pd.DataFrame) -> pd.Series:
    """The date a licence actually stopped being in force.

    A cancelled or revoked licence ends on its status-change date, not on the
    term expiry it never reached. Using the expiry date for these would
    systematically overstate how long failing businesses survived -- which is
    precisely the quantity the study measures.
    """
    ended_early = df["licence_status"].isin({STATUS_CANCELLED, STATUS_REVOKED})
    changed = df.get("status_change_date")
    if changed is None:
        return df["term_end"]
    end = df["term_end"].copy()
    use_change = ended_early & changed.notna()
    end.loc[use_change] = changed.loc[use_change]
    # A status change recorded after expiry is administrative; keep the earlier.
    return pd.concat([end, df["term_end"]], axis=1).min(axis=1)


def build_occupancies(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse licences into one row per continuous occupancy.

    Licences sharing (location, account) are merged when consecutive spells are
    separated by less than ``RENEWAL_GAP_DAYS``, absorbing renewals and
    concurrent licence types.
    """
    work = df.loc[
        (df["location_id"] != "")
        & df["term_start"].notna()
    ].copy()
    work["licence_end"] = _effective_end(work)
    # A licence with no usable end date cannot bound an occupancy.
    work = work.loc[work["licence_end"].notna() & (work["licence_end"] >= work["term_start"])]

    work = work.sort_values(["location_id", "account_number", "term_start"])
    grp = work.groupby(["location_id", "account_number"], sort=False)

    prev_end = grp["licence_end"].shift(1)
    # Running maximum of previous ends, so an earlier long licence is not
    # forgotten when a shorter one follows it.
    prev_max_end = grp["licence_end"].transform(lambda s: s.cummax().shift(1))
    gap = (work["term_start"] - prev_max_end).dt.days
    new_spell = prev_end.isna() | (gap > RENEWAL_GAP_DAYS)
    work["spell_id"] = new_spell.groupby(
        [work["location_id"], work["account_number"]]
    ).cumsum()

    occ = work.groupby(
        ["location_id", "account_number", "spell_id"], sort=False
    ).agg(
        start=("term_start", "min"),
        end=("licence_end", "max"),
        n_licences=("licence_id", "size"),
        dba_name=("dba_name", "first"),
        legal_name=("legal_name", "first"),
        licence_description=("licence_description", "first"),
        community_area=("community_area", "first"),
        neighborhood=("neighborhood", "first"),
        ward=("ward", "first"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        ever_cancelled=("licence_status", lambda s: bool((s == STATUS_CANCELLED).any())),
        ever_revoked=("licence_status", lambda s: bool((s == STATUS_REVOKED).any())),
        last_status=("licence_status", "last"),
    ).reset_index()

    log.info("collapsed %d licences into %d occupancies", len(work), len(occ))
    return occ


def merge_successive_occupants(
    occ: pd.DataFrame,
    *,
    succession_gap_days: int = SUCCESSION_GAP_DAYS,
) -> pd.DataFrame:
    """Merge back-to-back occupancies by different accounts at one address.

    A handover with no vacancy usually means the same trading business
    restructured or renamed rather than a genuine failure and replacement.
    Merging these keeps the turnover measure conservative.

    Adds ``merged_accounts``, the number of licence holders folded together, so
    the effect of this step is auditable.
    """
    df = occ.sort_values(["location_id", "start"]).copy()
    g = df.groupby("location_id", sort=False)

    prev_end = g["end"].transform(lambda s: s.cummax().shift(1))
    gap = (df["start"] - prev_end).dt.days
    # A new tenancy begins when the gap is long enough to be a real vacancy.
    # Overlapping spells (negative gap) are concurrent occupants of the same
    # address -- a mall or food hall -- and are also treated as one tenancy,
    # since the address is not a single storefront in that case.
    starts_new = prev_end.isna() | (gap > succession_gap_days)
    df["tenancy_id"] = starts_new.groupby(df["location_id"]).cumsum()

    ten = df.groupby(["location_id", "tenancy_id"], sort=False).agg(
        start=("start", "min"),
        end=("end", "max"),
        merged_accounts=("account_number", "nunique"),
        n_licences=("n_licences", "sum"),
        dba_name=("dba_name", "first"),
        licence_description=("licence_description", "first"),
        community_area=("community_area", "first"),
        neighborhood=("neighborhood", "first"),
        ward=("ward", "first"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        ever_cancelled=("ever_cancelled", "any"),
        ever_revoked=("ever_revoked", "any"),
        last_status=("last_status", "last"),
    ).reset_index()

    ten["duration_days"] = (ten["end"] - ten["start"]).dt.days
    ten["duration_years"] = ten["duration_days"] / 365.25

    n_merged = int((ten["merged_accounts"] > 1).sum())
    log.info(
        "merged %d occupancies into %d tenancies (%d involved >1 licence holder)",
        len(df), len(ten), n_merged,
    )
    return ten


def mark_censoring(
    tenancies: pd.DataFrame,
    *,
    snapshot_date: pd.Timestamp,
    margin_days: int = 180,
) -> pd.DataFrame:
    """Flag tenancies still in force when the data was cut, and cap their end.

    ``snapshot_date`` must be the date the extract was taken, not the latest
    date appearing in it. Licences are issued for terms that run into the
    future -- this extract contains expiry dates as late as 2030 -- so taking
    the maximum end date as the snapshot would place the observation point
    years after the data actually stops. Every currently-trading business would
    then look like a completed tenancy, credited with years it has not yet
    served, and right-censoring would appear not to exist. ``pipeline`` derives
    the correct value from the latest issue date.

    A tenancy whose licence extends beyond the snapshot is right-censored: it
    was still trading when observation stopped and its true end is unknown. Its
    observed end is capped at the snapshot so durations describe time actually
    elapsed.
    """
    df = tenancies.copy()

    # Still in force at the cut: the licence runs past the snapshot.
    df["is_right_censored"] = df["end"] > snapshot_date

    # Cap the observed end so no tenancy is credited with unelapsed time.
    df["observed_end"] = df["end"].clip(upper=snapshot_date)
    df["duration_days"] = (df["observed_end"] - df["start"]).dt.days
    df["duration_years"] = df["duration_days"] / 365.25

    # Closed early is the study's failure event: the occupant stopped before the
    # licence term it had already paid for ran out.
    df["closed_early"] = df["ever_cancelled"] | df["ever_revoked"]
    df["snapshot_date"] = snapshot_date

    log.info(
        "tenancies: %d | %d right-censored (still trading) | %d closed early",
        len(df), int(df["is_right_censored"].sum()), int(df["closed_early"].sum()),
    )
    return df


def build_locations(tenancies: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tenancies to one row per storefront."""
    g = tenancies.sort_values(["location_id", "start"]).groupby("location_id", sort=False)

    loc = g.agg(
        n_tenants=("tenancy_id", "size"),
        n_closed_early=("closed_early", "sum"),
        first_seen=("start", "min"),
        # observed_end, not end: a licence running past the snapshot must not
        # extend the location's observed span into the future.
        last_seen=("observed_end", "max"),
        mean_duration_days=("duration_days", "mean"),
        median_duration_days=("duration_days", "median"),
        min_duration_days=("duration_days", "min"),
        n_censored=("is_right_censored", "sum"),
        community_area=("community_area", "first"),
        neighborhood=("neighborhood", "first"),
        ward=("ward", "first"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
    ).reset_index()

    loc["observed_span_days"] = (loc["last_seen"] - loc["first_seen"]).dt.days
    span_years = loc["observed_span_days"] / 365.25
    loc["tenants_per_decade"] = np.where(
        span_years >= 1.0, 10.0 * loc["n_tenants"] / span_years, np.nan
    )
    loc["early_closure_rate"] = loc["n_closed_early"] / loc["n_tenants"]

    # Vacancy between successive tenants, averaged per location: a storefront
    # that sits empty between occupants is a different phenomenon from one that
    # is re-let immediately.
    gaps = tenancies.sort_values(["location_id", "start"]).copy()
    gg = gaps.groupby("location_id", sort=False)
    gaps["vacancy_days"] = (
        gaps["start"] - gg["observed_end"].transform(lambda s: s.cummax().shift(1))
    ).dt.days
    vac = gaps.groupby("location_id", sort=False)["vacancy_days"].mean().rename("mean_vacancy_days")
    loc = loc.merge(vac, on="location_id", how="left")

    log.info(
        "locations: %d | %d multi-tenant | max tenants %d",
        len(loc), int((loc["n_tenants"] > 1).sum()), int(loc["n_tenants"].max()),
    )
    return loc


def infer_snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """Estimate when the extract was taken.

    The latest issue date is the right anchor: licences are *issued* up to the
    moment of extraction, whereas their term end dates run years into the
    future. Using the latest term end would place the observation point after
    the data stops (see ``mark_censoring``).
    """
    issued = df.get("date_issued")
    if issued is None or issued.notna().sum() == 0:
        # Fall back to the latest term start, which is also bounded by the cut.
        return df["term_start"].max()
    return issued.max()


def pipeline(path, *, nrows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Chicago pipeline. Returns ``(tenancies, locations)``."""
    df = load_licences(path, nrows=nrows)
    snapshot = infer_snapshot_date(df)
    log.info("snapshot date inferred as %s", snapshot.date())

    df = filter_storefronts(df)
    df = attach_location_keys(df)
    occ = build_occupancies(df)
    ten = merge_successive_occupants(occ)
    ten = mark_censoring(ten, snapshot_date=snapshot)
    loc = build_locations(ten)
    return ten, loc
