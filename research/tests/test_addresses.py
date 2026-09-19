"""Tests for address normalization.

These cover the two failure modes that would bias the study's headline number:
under-merging (the same storefront split across keys, hiding turnover) and
over-merging (distinct storefronts collapsed into one, inventing turnover).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.addresses import (
    address_key,
    normalize_address,
    normalize_city,
    normalize_postal,
    split_unit,
)


class TestNormalizeAddress:
    @pytest.mark.parametrize(
        "variant",
        [
            "475 3rd St",
            "475 3rd Street",
            "475 3rd St.",
            "  475   3rd st  ",
            "475 3RD STREET",
            "475 3rd St,",
        ],
    )
    def test_street_spelling_variants_collapse(self, variant: str) -> None:
        assert normalize_address(variant) == "475 3 st"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1000 North Main Avenue", "1000 n main ave"),
            ("1000 N. Main Ave.", "1000 n main ave"),
            ("55 West Broadway Boulevard", "55 w broadway blvd"),
            ("200 Southeast 5th Court", "200 se 5 ct"),
        ],
    )
    def test_directionals_and_suffixes(self, raw: str, expected: str) -> None:
        assert normalize_address(raw) == expected

    def test_directional_inside_street_name_is_preserved(self) -> None:
        # "West" here is part of the name, not a directional prefix.
        assert normalize_address("12 Key West Road") == "12 key west rd"

    def test_accents_are_stripped(self) -> None:
        assert normalize_address("10 Rue Notré Dame") == "10 rue notre dame"

    def test_house_number_range_collapses_to_start(self) -> None:
        assert normalize_address("100-102 Main St") == "100 main st"

    @pytest.mark.parametrize("empty", [None, "", "   ", ",,"])
    def test_empty_input(self, empty: str | None) -> None:
        assert normalize_address(empty) == ""


class TestUnits:
    @pytest.mark.parametrize(
        ("raw", "street", "unit"),
        [
            ("123 Main St Suite 200", "123 Main St", "200"),
            ("123 Main St Ste 200", "123 Main St", "200"),
            ("123 Main St #4B", "123 Main St", "4b"),
            ("123 Main St Unit C", "123 Main St", "c"),
            ("123 Main St, Floor 3", "123 Main St", "3"),
            ("123 Main St", "123 Main St", None),
        ],
    )
    def test_split_unit(self, raw: str, street: str, unit: str | None) -> None:
        got_street, got_unit = split_unit(raw)
        assert got_street.rstrip(",") == street
        assert got_unit == unit

    def test_different_suites_do_not_merge(self) -> None:
        """Two tenants of one building are different storefronts."""
        a = normalize_address("500 Market St Suite 100")
        b = normalize_address("500 Market St Suite 200")
        assert a != b

    def test_same_suite_spelled_differently_does_merge(self) -> None:
        assert normalize_address("500 Market St Suite 100") == normalize_address(
            "500 Market Street #100"
        )


class TestPostalAndCity:
    def test_zip_plus_four_collapses(self) -> None:
        assert normalize_postal("94107-1234") == "94107"

    def test_canadian_postal(self) -> None:
        assert normalize_postal("m5v 2t6") == "M5V2T6"

    @pytest.mark.parametrize("empty", [None, "", "--"])
    def test_empty_postal(self, empty: str | None) -> None:
        assert normalize_postal(empty) == ""

    def test_city_abbreviations(self) -> None:
        assert normalize_city("Saint Louis") == "st louis"
        assert normalize_city("Montréal") == "montreal"


class TestAddressKey:
    def test_variants_produce_one_key(self) -> None:
        a = address_key("475 3rd Street", "San Francisco", "CA", "94107-1234")
        b = address_key("475 3rd St.", "San Francisco", "CA", "94107")
        assert a == b == "94107|ca|475 3 st"

    def test_same_street_different_zip_stays_separate(self) -> None:
        """The same street address recurs across cities; the ZIP separates them."""
        a = address_key("100 Main St", "Phoenix", "AZ", "85001")
        b = address_key("100 Main St", "Tampa", "FL", "33601")
        assert a != b

    def test_missing_postal_falls_back_to_city(self) -> None:
        key = address_key("100 Main St", "Phoenix", "AZ", None)
        assert key == "city:phoenix|az|100 main st"

    @pytest.mark.parametrize(
        "unusable",
        [
            ("", "Phoenix", "AZ", "85001"),
            (None, "Phoenix", "AZ", "85001"),
            # No house number: too vague to anchor a storefront.
            ("Main St", "Phoenix", "AZ", "85001"),
            # No postal code and no city: nothing to disambiguate with.
            ("100 Main St", None, "AZ", None),
        ],
    )
    def test_unusable_addresses_return_empty(self, unusable: tuple) -> None:
        assert address_key(*unusable) == ""
