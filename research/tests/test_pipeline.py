"""End-to-end recovery test against fixtures with known ground truth.

The real Yelp data carries no label for "this storefront turned over three
times", so the correctness of the grouping step cannot be checked against it.
The synthetic fixtures plant a known number of locations at each turnover level,
which makes the check possible: if the pipeline cannot recover a truth we
deliberately planted, its output on real data means nothing.

This is the test that protects the study's headline claim.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from src import ingest, lifespan, locations  # noqa: E402

# Matches the defaults in fixtures/make_fixtures.py.
N_STABLE = 300
N_MODERATE = 60
N_CURSED = 40


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a fresh synthetic dataset in a temporary directory."""
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run(
        [sys.executable, str(RESEARCH_ROOT / "fixtures" / "make_fixtures.py"),
         "--out", str(out),
         "--stable", str(N_STABLE),
         "--moderate", str(N_MODERATE),
         "--cursed", str(N_CURSED)],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def tables(fixture_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    """Ingest the fixtures to Parquet and load them back."""
    work = tmp_path_factory.mktemp("work")
    out: dict[str, pd.DataFrame] = {}
    for table in ("business", "review"):
        src = fixture_dir / f"yelp_academic_dataset_{table}.json"
        dest = work / f"{table}.parquet"
        ingest.ingest_table(table, src=src, dest=dest)
        out[table] = pd.read_parquet(dest)
    return out


@pytest.fixture(scope="module")
def built(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lifespans = lifespan.build_lifespans(tables["business"], tables["review"])
    tenancies, locs = locations.location_pipeline(tables["business"], lifespans)
    return lifespans, tenancies, locs


class TestIngest:
    def test_roundtrip_preserves_row_count(self, fixture_dir: Path, tables: dict) -> None:
        raw_lines = sum(
            1 for line in (fixture_dir / "yelp_academic_dataset_business.json").open()
            if line.strip()
        )
        assert len(tables["business"]) == raw_lines

    def test_nested_fields_are_flattened(self, tables: dict) -> None:
        b = tables["business"]
        assert b["categories"].map(lambda v: isinstance(v, str)).all()
        # attributes is stored as a JSON string, re-parseable without loss.
        sample = b["attributes"].dropna().iloc[0]
        assert isinstance(json.loads(sample), dict)


class TestLifespan:
    def test_every_business_gets_a_window(self, built: tuple) -> None:
        lifespans, _, _ = built
        # Every fixture business has at least one review by construction.
        assert lifespans["first_review"].notna().all()
        assert (lifespans["last_review"] >= lifespans["first_review"]).all()

    def test_open_businesses_are_right_censored(self, built: tuple) -> None:
        lifespans, _, _ = built
        still_open = lifespans[lifespans["is_open"] == 1]
        assert still_open["is_right_censored"].all()
        # A censored business must not contribute a duration.
        assert still_open["duration_days"].isna().all()

    def test_durations_only_where_interpretable(self, built: tuple) -> None:
        lifespans, _, _ = built
        with_duration = lifespans[lifespans["duration_days"].notna()]
        assert (with_duration["n_reviews"] >= lifespan.MIN_REVIEWS_FOR_DURATION).all()
        assert not with_duration["is_right_censored"].any()


class TestLocationRecovery:
    """The core validation: does grouping recover the planted structure?"""

    def test_total_location_count(self, built: tuple) -> None:
        _, _, locs = built
        assert len(locs) == N_STABLE + N_MODERATE + N_CURSED

    def test_tenant_count_distribution_matches_ground_truth(self, built: tuple) -> None:
        _, _, locs = built
        counts = locs["n_tenants"].value_counts()

        assert counts.get(1, 0) == N_STABLE, "single-tenant locations were split or merged"
        assert counts.get(2, 0) == N_MODERATE, "two-tenant locations were not recovered"

        multi = int((locs["n_tenants"] >= 3).sum())
        assert multi == N_CURSED, f"expected {N_CURSED} high-turnover locations, got {multi}"

    def test_no_location_exceeds_planted_maximum(self, built: tuple) -> None:
        """A count above 5 would mean distinct storefronts were wrongly merged."""
        _, _, locs = built
        assert locs["n_tenants"].max() <= 5

    def test_address_spelling_variants_were_merged(self, tables: dict, built: tuple) -> None:
        """The fixtures vary spelling per business; grouping must see through it.

        If normalization failed, every business would land in its own location
        and the location count would approach the business count.
        """
        _, _, locs = built
        assert len(locs) < len(tables["business"])

    def test_every_tenancy_sequence_is_ordered(self, built: tuple) -> None:
        _, tenancies, _ = built
        for _, grp in tenancies.groupby("location_id"):
            ordered = grp.sort_values("tenancy_index")
            assert ordered["first_review"].is_monotonic_increasing
            assert list(ordered["tenancy_index"]) == list(range(len(ordered)))


class TestCensoringAsymmetry:
    """Guards the study's most dangerous trap.

    Stable locations are stable *because* their tenant is still trading, which
    is exactly the condition that makes a duration unobservable. So durations
    are available almost exclusively for businesses that failed. Comparing mean
    observed duration between high- and low-turnover locations therefore
    compares a censored group against an uncensored one and will report a large,
    confident, meaningless difference.

    These tests pin the asymmetry in place so that no later change can quietly
    reintroduce the naive comparison. It is the reason the analysis uses
    survival methods (Kaplan-Meier, Cox) rather than a difference of means.
    """

    def test_single_tenant_locations_have_no_usable_durations(self, built: tuple) -> None:
        _, _, locs = built
        single = locs[locs["n_tenants"] == 1]
        assert len(single) == N_STABLE
        assert single["n_usable_durations"].sum() == 0, (
            "single-tenant locations yielded durations; the censoring rule is "
            "no longer excluding still-trading businesses"
        )

    def test_duration_availability_is_confounded_with_turnover(self, built: tuple) -> None:
        """Availability of a duration is itself a function of the outcome."""
        _, _, locs = built
        high_avail = (locs[locs["n_tenants"] >= 3]["n_usable_durations"] > 0).mean()
        low_avail = (locs[locs["n_tenants"] == 1]["n_usable_durations"] > 0).mean()
        assert high_avail > low_avail

    def test_shorter_tenures_recovered_among_observable_cases(self, built: tuple) -> None:
        """Within locations that do yield durations, the planted gap survives.

        Restricted to multi-tenant locations, where both groups contain closed
        businesses, so the comparison is between like and like.
        """
        _, tenancies, locs = built
        multi = locs[locs["n_tenants"] >= 2]
        high = multi[multi["n_tenants"] >= 3]["mean_duration_days"].dropna()
        mod = multi[multi["n_tenants"] == 2]["mean_duration_days"].dropna()
        assert len(high) > 0 and len(mod) > 0
        assert high.mean() < mod.mean(), "planted short tenures were not recovered"

    def test_high_turnover_locations_have_higher_closure_rate(self, built: tuple) -> None:
        _, _, locs = built
        high = locs[locs["n_tenants"] >= 3]["closure_rate"].mean()
        low = locs[locs["n_tenants"] == 1]["closure_rate"].mean()
        assert high > low
