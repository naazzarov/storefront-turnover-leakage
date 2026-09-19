"""Reconstruct business activity windows from review timestamps.

The Yelp Open Dataset records no opening or closing date. Each business carries
only ``is_open``, a single bit describing its status at the moment the snapshot
was taken. Turnover at an address is therefore not directly observable, and the
central methodological problem of this study is to recover it.

We use reviews as a proxy for activity. A business's *observed window* is the
interval between its first and last review. This is a biased estimator of the
true tenancy, in ways that matter and that we quantify rather than assume away:

  Left censoring   A business is invisible until its first review. Real opening
                   dates therefore precede observed first reviews, and the gap
                   is larger for businesses that opened before Yelp had traction
                   in their city. ``city_coverage_start`` measures this.

  Right censoring  A business still open at snapshot time has no closing date.
                   Its observed window is a lower bound, and these rows must be
                   excluded from, or explicitly modelled in, any duration
                   analysis. ``is_right_censored`` flags them.

  Thin evidence    A business with one review has an observed window of zero
                   days, which says nothing about its true tenancy.
                   ``MIN_REVIEWS_FOR_DURATION`` gates duration estimates.

  Trailing silence A closed business stops being reviewed some time before it
                   actually closes, so observed windows understate tenancy by a
                   roughly constant offset. This biases durations downward but
                   affects cursed and non-cursed locations alike, so it is
                   largely absorbed by comparing groups rather than reading
                   absolute durations.

Nothing here decides which locations are "cursed"; that is ``cursed.py``. This
module only produces the per-business temporal facts that decision rests on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Below this many reviews an observed window is too thin to read as a duration.
MIN_REVIEWS_FOR_DURATION = 3

# A business whose last review falls within this window of the snapshot is
# treated as still active, regardless of is_open: the absence of later reviews
# carries no information that close to the cutoff.
RIGHT_CENSOR_MARGIN_DAYS = 365

# Per-city, the review date below which Yelp coverage is too sparse to treat an
# absence of reviews as evidence of absence. Taken as a low quantile of the
# city's review dates (see ``city_coverage_start``).
COVERAGE_QUANTILE = 0.01


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse Yelp date strings, tolerating both date and datetime spellings."""
    return pd.to_datetime(series, errors="coerce", format="mixed")


def review_activity(reviews: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reviews to one row per business.

    Args:
        reviews: frame with ``business_id``, ``date`` and ``stars``.

    Returns:
        Per-business first/last review dates, counts, and rating trajectory.
        ``stars_trend`` is the slope of a least-squares line through the
        business's ratings over time, in stars per year: a negative value means
        the business was getting worse as it went.
    """
    df = reviews.loc[:, ["business_id", "date", "stars"]].copy()
    df["date"] = _to_datetime(df["date"])
    df = df.dropna(subset=["date", "business_id"])

    grouped = df.groupby("business_id", sort=False)
    out = grouped.agg(
        first_review=("date", "min"),
        last_review=("date", "max"),
        n_reviews=("date", "size"),
        mean_stars=("stars", "mean"),
        std_stars=("stars", "std"),
    ).reset_index()

    out["observed_days"] = (out["last_review"] - out["first_review"]).dt.days
    out["stars_trend"] = _rating_trend(df)

    # Mean rating over the final quarter of the observed window: a business
    # that deteriorated before closing shows a gap against its lifetime mean.
    out = out.merge(_late_window_rating(df), on="business_id", how="left")

    return out


def _rating_trend(reviews: pd.DataFrame) -> pd.Series:
    """Least-squares slope of stars against time, in stars per year.

    Computed in closed form per group rather than via ``polyfit`` in a Python
    loop: with millions of reviews the loop dominates runtime.
    """
    df = reviews.sort_values(["business_id", "date"])
    # Years since that business's own first review.
    t0 = df.groupby("business_id", sort=False)["date"].transform("min")
    x = (df["date"] - t0).dt.total_seconds() / (365.25 * 24 * 3600)
    y = df["stars"].astype(float)

    frame = pd.DataFrame({"business_id": df["business_id"], "x": x, "y": y})
    g = frame.groupby("business_id", sort=False)
    n = g["x"].transform("size")
    mx = g["x"].transform("mean")
    my = g["y"].transform("mean")
    frame["cov"] = (frame["x"] - mx) * (frame["y"] - my)
    frame["var"] = (frame["x"] - mx) ** 2

    agg = frame.groupby("business_id", sort=False).agg(cov=("cov", "sum"), var=("var", "sum"))
    # Zero variance means every review landed on one day: slope is undefined.
    slope = np.where(agg["var"] > 0, agg["cov"] / agg["var"].replace(0, np.nan), np.nan)
    result = pd.Series(slope, index=agg.index, name="stars_trend")
    del n  # retained above only to make the closed form readable
    return result.reset_index(drop=True)


def _late_window_rating(reviews: pd.DataFrame) -> pd.DataFrame:
    """Mean stars over the last 25% of each business's observed window."""
    df = reviews.sort_values(["business_id", "date"])
    g = df.groupby("business_id", sort=False)["date"]
    start = g.transform("min")
    end = g.transform("max")
    span = (end - start).dt.total_seconds()
    # Businesses whose reviews all land on one day have no meaningful late window.
    frac = np.where(span > 0, (df["date"] - start).dt.total_seconds() / span.replace(0, np.nan), 1.0)
    late = df.loc[frac >= 0.75]
    return (
        late.groupby("business_id", sort=False)["stars"]
        .mean()
        .rename("late_mean_stars")
        .reset_index()
    )


def city_coverage_start(
    businesses: pd.DataFrame,
    activity: pd.DataFrame,
    quantile: float = COVERAGE_QUANTILE,
) -> pd.DataFrame:
    """Estimate when Yelp coverage begins in each city.

    Before a city reaches meaningful Yelp adoption, the absence of reviews says
    more about Yelp than about the business. Any location whose history starts
    near this boundary has an unobservable prefix, and the analysis excludes it
    from turnover counts.

    Returns one row per ``(city, state)`` with ``coverage_start``.
    """
    merged = businesses[["business_id", "city", "state"]].merge(
        activity[["business_id", "first_review"]], on="business_id", how="inner"
    )
    return (
        merged.groupby(["city", "state"], sort=False)["first_review"]
        .quantile(quantile)
        .rename("coverage_start")
        .reset_index()
    )


def build_lifespans(
    businesses: pd.DataFrame,
    reviews: pd.DataFrame,
    *,
    snapshot_date: pd.Timestamp | None = None,
    right_censor_margin_days: int = RIGHT_CENSOR_MARGIN_DAYS,
) -> pd.DataFrame:
    """Produce the per-business lifespan table the rest of the pipeline uses.

    Args:
        businesses: the ingested business table.
        reviews: the ingested review table.
        snapshot_date: the date the dataset was cut. Defaults to the latest
            review date, which is the best available estimate.
        right_censor_margin_days: businesses last reviewed within this many days
            of the snapshot are treated as still active.

    Returns:
        One row per business carrying its observed window, censoring flags, and
        a ``duration_days`` that is populated only where it is interpretable.
    """
    activity = review_activity(reviews)

    if snapshot_date is None:
        snapshot_date = activity["last_review"].max()
        log.info("inferred snapshot date %s", snapshot_date.date())

    df = businesses.merge(activity, on="business_id", how="left")

    coverage = city_coverage_start(businesses, activity)
    df = df.merge(coverage, on=["city", "state"], how="left")

    df["has_reviews"] = df["n_reviews"].fillna(0) > 0
    df["n_reviews"] = df["n_reviews"].fillna(0).astype(int)

    # is_open reflects snapshot status; the review trail supplies the rest.
    df["is_open"] = df["is_open"].fillna(0).astype(int)
    days_silent = (snapshot_date - df["last_review"]).dt.days
    df["days_silent"] = days_silent

    # Right-censored: either flagged open, or reviewed recently enough that its
    # silence is uninformative.
    df["is_right_censored"] = (df["is_open"] == 1) | (
        days_silent.notna() & (days_silent <= right_censor_margin_days)
    )

    # Left-censored: the business was already active when city coverage began,
    # so its true opening date is unobservable.
    df["is_left_censored"] = (
        df["first_review"].notna()
        & df["coverage_start"].notna()
        & (df["first_review"] <= df["coverage_start"])
    )

    df["duration_days"] = np.where(
        (~df["is_right_censored"])
        & (df["n_reviews"] >= MIN_REVIEWS_FOR_DURATION)
        & df["observed_days"].notna(),
        df["observed_days"],
        np.nan,
    )
    df["duration_years"] = df["duration_days"] / 365.25

    df["snapshot_date"] = snapshot_date

    n = len(df)
    log.info(
        "lifespans: %d businesses | %d with reviews | %d right-censored | "
        "%d left-censored | %d with usable duration",
        n,
        int(df["has_reviews"].sum()),
        int(df["is_right_censored"].sum()),
        int(df["is_left_censored"].sum()),
        int(df["duration_days"].notna().sum()),
    )
    return df


def censoring_report(lifespans: pd.DataFrame) -> pd.DataFrame:
    """Summarize how much of the data each censoring mechanism removes.

    This table goes in the paper. Reviewers will ask what fraction of businesses
    the duration analysis can actually speak to, and the honest answer is well
    under half; stating it up front is better than being asked.
    """
    n = len(lifespans)
    rows = [
        ("businesses", n),
        ("with >=1 review", int(lifespans["has_reviews"].sum())),
        (f"with >={MIN_REVIEWS_FOR_DURATION} reviews",
         int((lifespans["n_reviews"] >= MIN_REVIEWS_FOR_DURATION).sum())),
        ("right-censored (still active)", int(lifespans["is_right_censored"].sum())),
        ("left-censored (pre-coverage)", int(lifespans["is_left_censored"].sum())),
        ("usable duration", int(lifespans["duration_days"].notna().sum())),
    ]
    out = pd.DataFrame(rows, columns=["subset", "n"])
    out["pct_of_all"] = (100 * out["n"] / n).round(1)
    return out
