"""Tests for the Chicago licence-to-tenancy reconstruction.

The reconstruction makes three judgement calls, each of which can move the
headline turnover number in a different direction:

  renewals must collapse       or every two-year renewal reads as turnover and
                               the study reports churn everywhere
  successors must collapse     or a rename/restructure reads as a failure
  real turnover must survive   or the aggressive merging above erases the signal

These tests build small hand-constructed licence histories where the right
answer is known by construction, and assert each case separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import chicago  # noqa: E402


def _licence(
    *,
    account: str,
    start: str,
    end: str,
    licence_id: str | None = None,
    status: str = chicago.STATUS_ISSUED,
    status_change: str | None = None,
    address: str = "100 N MAIN ST",
    zip_code: str = "60601",
    dba: str = "Some Shop",
    description: str = "Retail Food Establishment",
) -> dict:
    """One licence row in the shape ``load_licences`` produces."""
    return {
        "licence_id": licence_id or f"L{account}{start}",
        "account_number": account,
        "dba_name": dba,
        "legal_name": dba,
        "address": address,
        "city": "CHICAGO",
        "state": "IL",
        "zip_code": zip_code,
        "licence_description": description,
        "licence_status": status,
        "term_start": pd.Timestamp(start),
        "term_end": pd.Timestamp(end),
        "status_change_date": pd.Timestamp(status_change) if status_change else pd.NaT,
        "community_area": "LOOP",
        "neighborhood": "LOOP",
        "ward": "42",
        "latitude": 41.88,
        "longitude": -87.63,
    }


# Stands in for the date the extract was taken. Every hand-built history above
# ends well before it, so nothing is censored unless a test says so.
SNAPSHOT = pd.Timestamp("2020-01-01")


def _build(
    rows: list[dict], *, snapshot: pd.Timestamp = SNAPSHOT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the post-load half of the pipeline over hand-built rows."""
    df = chicago.attach_location_keys(pd.DataFrame(rows))
    occ = chicago.build_occupancies(df)
    ten = chicago.merge_successive_occupants(occ)
    ten = chicago.mark_censoring(ten, snapshot_date=snapshot)
    return ten, chicago.build_locations(ten)


class TestRenewalCollapse:
    def test_consecutive_renewals_are_one_tenancy(self) -> None:
        """A shop renewing every two years for six years is one occupant."""
        rows = [
            _licence(account="A", start="2010-01-01", end="2012-01-01"),
            _licence(account="A", start="2012-01-01", end="2014-01-01"),
            _licence(account="A", start="2014-01-01", end="2016-01-01"),
        ]
        ten, loc = _build(rows)
        assert len(ten) == 1
        assert loc["n_tenants"].iloc[0] == 1
        assert ten["start"].iloc[0] == pd.Timestamp("2010-01-01")
        assert ten["end"].iloc[0] == pd.Timestamp("2016-01-01")

    def test_concurrent_licence_types_are_one_tenancy(self) -> None:
        """A restaurant holding food + liquor licences at once is one occupant."""
        rows = [
            _licence(account="A", start="2010-01-01", end="2012-01-01",
                     licence_id="food", description="Retail Food Establishment"),
            _licence(account="A", start="2010-03-01", end="2012-03-01",
                     licence_id="liquor", description="Consumption on Premises"),
        ]
        ten, loc = _build(rows)
        assert len(ten) == 1
        assert loc["n_tenants"].iloc[0] == 1

    def test_same_account_returning_years_later_is_two_tenancies(self) -> None:
        """A long vacancy is real turnover even when the occupant returns."""
        rows = [
            _licence(account="A", start="2004-01-01", end="2006-01-01"),
            _licence(account="A", start="2014-01-01", end="2016-01-01"),
        ]
        ten, _ = _build(rows)
        assert len(ten) == 2


class TestSuccessionMerge:
    def test_immediate_handover_is_not_turnover(self) -> None:
        """A rename or restructure with no vacancy must not count as a failure."""
        rows = [
            _licence(account="A", start="2010-01-01", end="2014-01-01", dba="Old Name"),
            _licence(account="B", start="2014-01-05", end="2018-01-01", dba="New Name"),
        ]
        ten, loc = _build(rows)
        assert len(ten) == 1
        assert loc["n_tenants"].iloc[0] == 1
        assert ten["merged_accounts"].iloc[0] == 2

    def test_handover_after_long_vacancy_is_turnover(self) -> None:
        rows = [
            _licence(account="A", start="2010-01-01", end="2014-01-01", dba="Old Name"),
            _licence(account="B", start="2015-06-01", end="2018-01-01", dba="New Name"),
        ]
        ten, loc = _build(rows)
        assert len(ten) == 2
        assert loc["n_tenants"].iloc[0] == 2

    def test_overlapping_occupants_are_one_tenancy(self) -> None:
        """Concurrent occupants mean the address is not a single storefront."""
        rows = [
            _licence(account="A", start="2010-01-01", end="2016-01-01"),
            _licence(account="B", start="2012-01-01", end="2018-01-01"),
        ]
        ten, _ = _build(rows)
        assert len(ten) == 1


class TestTurnoverSurvives:
    def test_sequential_failures_are_counted(self) -> None:
        """The signal the study exists to measure must not be merged away."""
        rows = [
            _licence(account="A", start="2006-01-01", end="2007-06-01"),
            _licence(account="B", start="2008-06-01", end="2010-01-01"),
            _licence(account="C", start="2011-06-01", end="2012-09-01"),
            _licence(account="D", start="2014-01-01", end="2015-03-01"),
        ]
        ten, loc = _build(rows)
        assert len(ten) == 4
        assert loc["n_tenants"].iloc[0] == 4

    def test_distinct_suites_do_not_merge(self) -> None:
        """Two tenants of one building are two storefronts, not turnover."""
        rows = [
            _licence(account="A", start="2010-01-01", end="2014-01-01",
                     address="100 N MAIN ST STE 100"),
            _licence(account="B", start="2010-01-01", end="2014-01-01",
                     address="100 N MAIN ST STE 200"),
        ]
        _, loc = _build(rows)
        assert len(loc) == 2
        assert set(loc["n_tenants"]) == {1}


class TestEarlyClosure:
    def test_cancelled_licence_ends_at_status_change(self) -> None:
        """A business that quit early must not be credited with its full term.

        Using the unreached expiry date would overstate how long failing
        businesses survived -- the exact quantity this study measures.
        """
        rows = [
            _licence(account="A", start="2010-01-01", end="2014-01-01",
                     status=chicago.STATUS_CANCELLED, status_change="2011-01-01"),
        ]
        ten, _ = _build(rows)
        assert ten["end"].iloc[0] == pd.Timestamp("2011-01-01")
        assert ten["duration_days"].iloc[0] == pytest.approx(365, abs=1)

    def test_early_closure_is_flagged(self) -> None:
        rows = [
            _licence(account="A", start="2010-01-01", end="2014-01-01",
                     status=chicago.STATUS_CANCELLED, status_change="2011-01-01"),
        ]
        ten, loc = _build(rows)
        assert bool(ten["closed_early"].iloc[0])
        assert loc["early_closure_rate"].iloc[0] == 1.0

    def test_normal_expiry_is_not_early_closure(self) -> None:
        rows = [_licence(account="A", start="2010-01-01", end="2014-01-01")]
        ten, _ = _build(rows)
        assert not bool(ten["closed_early"].iloc[0])


class TestStorefrontFilter:
    def test_non_storefront_classes_are_dropped(self) -> None:
        df = pd.DataFrame([
            _licence(account="A", start="2010-01-01", end="2012-01-01",
                     description="Home Occupation"),
            _licence(account="B", start="2010-01-01", end="2012-01-01",
                     description="Peddler, non-food"),
            _licence(account="C", start="2010-01-01", end="2012-01-01",
                     description="Retail Food Establishment"),
        ])
        kept = chicago.filter_storefronts(df)
        assert list(kept["account_number"]) == ["C"]


class TestSnapshotAndCensoring:
    """Licences run into the future; the observation point must not.

    Terms are issued for years ahead, so the latest date in the extract is far
    beyond the date the extract was taken. Anchoring the snapshot to that
    maximum makes every currently-trading business look like a finished tenancy
    credited with years it has not served, and right-censoring vanishes.
    """

    def test_snapshot_is_taken_from_issue_dates_not_term_ends(self) -> None:
        df = pd.DataFrame([
            {"date_issued": pd.Timestamp("2026-09-18"),
             "term_start": pd.Timestamp("2026-09-01"),
             "term_end": pd.Timestamp("2030-07-15")},
        ])
        assert chicago.infer_snapshot_date(df) == pd.Timestamp("2026-09-18")

    def test_licence_running_past_snapshot_is_censored(self) -> None:
        rows = [_licence(account="A", start="2018-01-01", end="2022-01-01")]
        ten, _ = _build(rows, snapshot=pd.Timestamp("2020-01-01"))
        assert bool(ten["is_right_censored"].iloc[0])

    def test_censored_tenancy_is_not_credited_with_unelapsed_time(self) -> None:
        """Duration must stop at the snapshot, not at a future expiry."""
        rows = [_licence(account="A", start="2018-01-01", end="2022-01-01")]
        ten, _ = _build(rows, snapshot=pd.Timestamp("2020-01-01"))
        assert ten["observed_end"].iloc[0] == pd.Timestamp("2020-01-01")
        assert ten["duration_days"].iloc[0] == pytest.approx(730, abs=1)

    def test_completed_tenancy_is_not_censored(self) -> None:
        rows = [_licence(account="A", start="2010-01-01", end="2014-01-01")]
        ten, _ = _build(rows, snapshot=pd.Timestamp("2020-01-01"))
        assert not bool(ten["is_right_censored"].iloc[0])
        assert ten["observed_end"].iloc[0] == pd.Timestamp("2014-01-01")

    def test_location_span_does_not_extend_past_snapshot(self) -> None:
        rows = [_licence(account="A", start="2018-01-01", end="2022-01-01")]
        _, loc = _build(rows, snapshot=pd.Timestamp("2020-01-01"))
        assert loc["last_seen"].iloc[0] <= pd.Timestamp("2020-01-01")


class TestVacancy:
    def test_vacancy_between_tenants_is_measured(self) -> None:
        rows = [
            _licence(account="A", start="2010-01-01", end="2012-01-01"),
            _licence(account="B", start="2013-01-01", end="2015-01-01"),
        ]
        _, loc = _build(rows)
        # One gap of ~366 days; the first tenant contributes no gap.
        assert loc["mean_vacancy_days"].iloc[0] == pytest.approx(366, abs=2)
