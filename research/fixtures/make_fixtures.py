"""Generate synthetic Yelp-shaped JSON for testing the pipeline.

This exists for two reasons.

First, development: the real dataset is licensed and large, so the pipeline has
to be testable without it. Every file here mimics the exact schema documented in
the Yelp Dataset JSON documentation -- same field names, same types, same
newline-delimited layout.

Second, and more important for the paper: Yelp's terms forbid redistributing the
data (Section 4.A/4.E), so the replication package cannot ship the inputs that
produced our numbers. It can ship this generator instead. The fixtures embed a
*known ground truth* -- we plant a specified number of high-turnover locations
with specified properties -- which lets us verify that the detection method
recovers what was planted. That is a validation the real data cannot provide,
because the real data has no ground-truth label for a "cursed" location.

The generator is fully deterministic given a seed.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Kept deliberately small and generic: these are invented names, not Yelp data.
_CITIES = [
    ("Springfield", "IL", "62701", 39.7817, -89.6501),
    ("Riverton", "AZ", "85001", 33.4484, -112.0740),
    ("Lakeside", "FL", "33601", 27.9506, -82.4572),
    ("Northgate", "PA", "19101", 39.9526, -75.1652),
    ("Easton", "NV", "89101", 36.1699, -115.1398),
]
_STREETS = ["Main St", "Oak Ave", "Cedar Blvd", "Elm Rd", "Maple Dr", "Pine Ln", "Birch Way"]
_CATEGORIES = [
    "Restaurants", "Bars", "Coffee & Tea", "Fast Food", "Shopping",
    "Beauty & Spas", "Automotive", "Health & Medical", "Nightlife", "Bakeries",
]
_NAME_PARTS_A = ["Golden", "Blue", "Corner", "Urban", "Sunset", "Riverside", "Copper", "Iron"]
_NAME_PARTS_B = ["Kitchen", "Cafe", "Grill", "Bistro", "Lounge", "Market", "Studio", "Garage"]

_POSITIVE = [
    "Great food and friendly staff.", "Loved the atmosphere here.",
    "Excellent service, will come back.", "Best in the neighborhood.",
    "Really solid value for the price.",
]
_NEGATIVE = [
    "Service was slow and the staff seemed overwhelmed.",
    "Place was not clean, would not return.",
    "Parking here is impossible, gave up twice.",
    "Hard to even find the entrance from the street.",
    "Overpriced for what you get, disappointing.",
]

# Address-spelling variants, injected so the fixtures exercise the
# normalization path in addresses.py rather than handing it clean input.
_SUFFIX_VARIANTS = {
    "St": ["St", "Street", "St."],
    "Ave": ["Ave", "Avenue", "Ave."],
    "Blvd": ["Blvd", "Boulevard", "Blvd."],
    "Rd": ["Rd", "Road", "Rd."],
    "Dr": ["Dr", "Drive", "Dr."],
    "Ln": ["Ln", "Lane", "Ln."],
    "Way": ["Way"],
}


def _vary_address(street: str, rng: random.Random) -> str:
    """Return a randomly-spelled variant of a street, e.g. 'Oak Ave' -> 'Oak Avenue'."""
    base, _, suffix = street.rpartition(" ")
    return f"{base} {rng.choice(_SUFFIX_VARIANTS.get(suffix, [suffix]))}"


def _business_name(rng: random.Random) -> str:
    return f"{rng.choice(_NAME_PARTS_A)} {rng.choice(_NAME_PARTS_B)}"


def _random_id(rng: random.Random) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(alphabet) for _ in range(22))


def _attributes(rng: random.Random, *, cursed: bool) -> dict[str, Any]:
    """Build a plausible attribute dict.

    Cursed locations are given systematically worse amenities, which is the
    signal the analysis is supposed to recover. The gap is deliberately modest
    (not 0 vs 1) so recovering it is a real statistical task.
    """
    p_parking = 0.25 if cursed else 0.65
    p_wifi = 0.20 if cursed else 0.55
    return {
        "BusinessParking": json.dumps({
            "garage": rng.random() < p_parking * 0.4,
            "street": rng.random() < p_parking,
            "lot": rng.random() < p_parking * 0.8,
            "valet": rng.random() < 0.05,
        }),
        "WiFi": "'free'" if rng.random() < p_wifi else "'no'",
        "RestaurantsTakeOut": rng.random() < 0.7,
        "OutdoorSeating": rng.random() < (0.15 if cursed else 0.45),
        "RestaurantsPriceRange2": str(rng.randint(1, 4)),
    }


class FixtureBuilder:
    """Builds a synthetic dataset with a recorded ground truth."""

    def __init__(self, seed: int = 20260919) -> None:
        self.rng = random.Random(seed)
        self.businesses: list[dict[str, Any]] = []
        self.reviews: list[dict[str, Any]] = []
        self.tips: list[dict[str, Any]] = []
        self.checkins: list[dict[str, Any]] = []
        self.truth: list[dict[str, Any]] = []
        self._used_addresses: set[tuple] = set()

    def _new_address(self) -> tuple[str, str, str, str, float, float]:
        """Pick an unused (street, city) pair so planted locations stay distinct."""
        while True:
            city, state, postal, lat, lon = self.rng.choice(_CITIES)
            number = self.rng.randrange(100, 9999)
            street = self.rng.choice(_STREETS)
            key = (number, street, postal)
            if key in self._used_addresses:
                continue
            self._used_addresses.add(key)
            # Jitter coordinates by roughly a few hundred metres.
            return (
                f"{number} {street}", city, state, postal,
                round(lat + self.rng.uniform(-0.05, 0.05), 6),
                round(lon + self.rng.uniform(-0.05, 0.05), 6),
            )

    def _add_business(
        self,
        *,
        street: str,
        city: str,
        state: str,
        postal: str,
        lat: float,
        lon: float,
        start: date,
        months: int,
        is_open: bool,
        cursed: bool,
    ) -> str:
        """Create one business plus the reviews that imply its active window."""
        bid = _random_id(self.rng)
        n_reviews = max(1, int(self.rng.gauss(18 if not cursed else 9, 6)))
        # Closed businesses skew lower-rated, but with heavy overlap.
        mean_stars = self.rng.gauss(3.2 if cursed else 3.8, 0.7)
        mean_stars = min(5.0, max(1.0, mean_stars))

        end = start + timedelta(days=int(months * 30.44))
        span_days = max(1, (end - start).days)

        for _ in range(n_reviews):
            offset = int(self.rng.random() * span_days)
            rdate = start + timedelta(days=offset)
            stars = min(5, max(1, int(round(self.rng.gauss(mean_stars, 1.1)))))
            text = self.rng.choice(_NEGATIVE if stars <= 2 else _POSITIVE)
            self.reviews.append({
                "review_id": _random_id(self.rng),
                "user_id": _random_id(self.rng),
                "business_id": bid,
                "stars": float(stars),
                "useful": self.rng.randint(0, 5),
                "funny": self.rng.randint(0, 3),
                "cool": self.rng.randint(0, 3),
                "text": text,
                "date": f"{rdate.isoformat()} {self.rng.randint(0, 23):02d}:"
                        f"{self.rng.randint(0, 59):02d}:{self.rng.randint(0, 59):02d}",
            })

        if self.rng.random() < 0.5:
            tdate = start + timedelta(days=int(self.rng.random() * span_days))
            self.tips.append({
                "user_id": _random_id(self.rng),
                "business_id": bid,
                "text": self.rng.choice(_POSITIVE + _NEGATIVE),
                "date": tdate.isoformat(),
                "compliment_count": self.rng.randint(0, 3),
            })

        n_checkins = self.rng.randint(0, 25)
        if n_checkins:
            stamps = []
            for _ in range(n_checkins):
                cdate = start + timedelta(days=int(self.rng.random() * span_days))
                stamps.append(f"{cdate.isoformat()} {self.rng.randint(0, 23):02d}:"
                              f"{self.rng.randint(0, 59):02d}:{self.rng.randint(0, 59):02d}")
            self.checkins.append({"business_id": bid, "date": ", ".join(sorted(stamps))})

        self.businesses.append({
            "business_id": bid,
            "name": _business_name(self.rng),
            # Spelling is varied per business so the same storefront appears
            # under different address strings -- exactly the real-world case.
            "address": _vary_address(street, self.rng),
            "city": city,
            "state": state,
            "postal_code": postal,
            "latitude": lat,
            "longitude": lon,
            "stars": round(min(5.0, max(1.0, self.rng.gauss(mean_stars, 0.3))) * 2) / 2,
            "review_count": n_reviews,
            "is_open": 1 if is_open else 0,
            "attributes": _attributes(self.rng, cursed=cursed),
            "categories": ", ".join(self.rng.sample(_CATEGORIES, self.rng.randint(1, 3))),
            "hours": {"Monday": "10:00-21:00", "Tuesday": "10:00-21:00"},
        })
        return bid

    def build(
        self,
        *,
        n_stable: int = 300,
        n_cursed: int = 40,
        n_moderate: int = 60,
        first_year: int = 2006,
        last_year: int = 2021,
    ) -> None:
        """Plant three kinds of location and record which is which.

        - stable:   one long-lived tenant, still open
        - moderate: two tenants, the first closed after a normal run
        - cursed:   three to five short-lived tenants, nearly all closed
        """
        window_start = date(first_year, 1, 1)
        window_end = date(last_year, 12, 31)
        total_months = (window_end.year - window_start.year) * 12

        for _ in range(n_stable):
            street, city, state, postal, lat, lon = self._new_address()
            start = window_start + timedelta(days=self.rng.randrange(0, 365 * 6))
            self._add_business(
                street=street, city=city, state=state, postal=postal, lat=lat, lon=lon,
                start=start, months=self.rng.randint(60, 150), is_open=True, cursed=False,
            )
            self.truth.append({
                "postal_code": postal, "address": street,
                "label": "stable", "n_tenants": 1,
            })

        for _ in range(n_moderate):
            street, city, state, postal, lat, lon = self._new_address()
            cursor = window_start + timedelta(days=self.rng.randrange(0, 365 * 3))
            for i in range(2):
                months = self.rng.randint(30, 72)
                last = i == 1
                self._add_business(
                    street=street, city=city, state=state, postal=postal, lat=lat, lon=lon,
                    start=cursor, months=months, is_open=last, cursed=False,
                )
                cursor += timedelta(days=int(months * 30.44) + self.rng.randint(30, 180))
            self.truth.append({
                "postal_code": postal, "address": street,
                "label": "moderate", "n_tenants": 2,
            })

        for _ in range(n_cursed):
            street, city, state, postal, lat, lon = self._new_address()
            n_tenants = self.rng.randint(3, 5)
            cursor = window_start + timedelta(days=self.rng.randrange(0, 365 * 2))
            for i in range(n_tenants):
                months = self.rng.randint(6, 26)  # short tenures
                last = i == n_tenants - 1
                # Even the final tenant is usually already gone.
                still_open = last and self.rng.random() < 0.35
                self._add_business(
                    street=street, city=city, state=state, postal=postal, lat=lat, lon=lon,
                    start=cursor, months=months, is_open=still_open, cursed=True,
                )
                cursor += timedelta(days=int(months * 30.44) + self.rng.randint(15, 120))
                if cursor > window_end:
                    break
            self.truth.append({
                "postal_code": postal, "address": street,
                "label": "cursed", "n_tenants": n_tenants,
            })

        _ = total_months  # window bookkeeping, kept for readability

    def write(self, out_dir: Path) -> dict[str, Path]:
        """Write newline-delimited JSON matching the official filenames."""
        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        tables = {
            "yelp_academic_dataset_business.json": self.businesses,
            "yelp_academic_dataset_review.json": self.reviews,
            "yelp_academic_dataset_tip.json": self.tips,
            "yelp_academic_dataset_checkin.json": self.checkins,
        }
        for filename, rows in tables.items():
            path = out_dir / filename
            with open(path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            written[filename] = path

        truth_path = out_dir / "ground_truth.json"
        with open(truth_path, "w", encoding="utf-8") as fh:
            json.dump(self.truth, fh, indent=2)
        written["ground_truth.json"] = truth_path
        return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "synthetic")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--stable", type=int, default=300)
    parser.add_argument("--cursed", type=int, default=40)
    parser.add_argument("--moderate", type=int, default=60)
    args = parser.parse_args()

    builder = FixtureBuilder(seed=args.seed)
    builder.build(n_stable=args.stable, n_cursed=args.cursed, n_moderate=args.moderate)
    written = builder.write(args.out)

    print(f"businesses : {len(builder.businesses):,}")
    print(f"reviews    : {len(builder.reviews):,}")
    print(f"tips       : {len(builder.tips):,}")
    print(f"checkins   : {len(builder.checkins):,}")
    print(f"locations  : {len(builder.truth):,} "
          f"({sum(t['label'] == 'cursed' for t in builder.truth)} cursed)")
    print(f"\nwritten to {args.out}")
    for name, path in written.items():
        print(f"  {name:42} {path.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
