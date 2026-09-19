"""Generate a synthetic business_small dataset.

This produces a deterministic, reproducible stand-in for the real Yelp
business_small table so the benchmark can be run without the proprietary
Yelp dataset. The generator is pure stdlib (random + sqlite3 + csv) so it
has zero dependencies and produces identical output every run.

Outputs:
    benchmark/data/business_small.sqlite
    benchmark/data/business_small.csv
"""

import csv
import os
import random
import sqlite3
from pathlib import Path

SEED = 2026
NUM_BUSINESSES = 1000

DATA_DIR = Path(__file__).parent / "data"

CITIES = {
    "Philadelphia": "PA",
    "Phoenix": "AZ",
    "Tampa": "FL",
    "Toronto": "ON",
    "Tucson": "AZ",
    "Pittsburgh": "PA",
    "Orlando": "FL",
    "Mesa": "AZ",
    "Ottawa": "ON",
    "Las Vegas": "NV",
}

CATEGORY_POOL = [
    "Restaurants",
    "Bars",
    "Cafes",
    "Shopping",
    "Automotive",
    "Beauty & Spas",
    "Health & Medical",
    "Home Services",
    "Nightlife",
    "Fast Food",
]

NAME_PREFIX = [
    "Golden", "Blue", "Red", "Sunset", "Lucky", "Green", "Silver", "Crown",
    "Maple", "Iron", "Cozy", "Urban", "Royal", "Hidden", "Grand", "Little",
]
NAME_SUFFIX = [
    "Bistro", "Cafe", "Diner", "Grill", "Bar", "Market", "Pizza", "Sushi",
    "Bakery", "Salon", "Garage", "Garden", "House", "Tavern", "Lounge", "Spa",
]


def pick_stars(rng: random.Random) -> float:
    return round(rng.uniform(1.0, 5.0) * 2) / 2


def pick_review_count(rng: random.Random, stars: float) -> int:
    base = rng.randint(1, 5000)
    if stars >= 4.5:
        base = rng.randint(200, 5000)
    elif stars >= 4.0:
        base = rng.randint(50, 3000)
    return base


def pick_categories(rng: random.Random) -> str:
    n = rng.randint(1, 3)
    cats = rng.sample(CATEGORY_POOL, n)
    return ", ".join(cats)


def generate() -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    seen_names = set()
    for i in range(NUM_BUSINESSES):
        business_id = f"biz_{i:05d}"
        name = f"{rng.choice(NAME_PREFIX)} {rng.choice(NAME_SUFFIX)}"
        if name in seen_names:
            name = f"{name} #{rng.randint(2, 999)}"
        seen_names.add(name)
        city = rng.choice(list(CITIES.keys()))
        state = CITIES[city]
        stars = pick_stars(rng)
        review_count = pick_review_count(rng, stars)
        categories = pick_categories(rng)
        rows.append(
            {
                "business_id": business_id,
                "name": name,
                "city": city,
                "state": state,
                "stars": stars,
                "review_count": review_count,
                "categories": categories,
            }
        )
    return rows


def write_sqlite(rows: list[dict], path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS business_small (
            business_id   VARCHAR PRIMARY KEY,
            name          VARCHAR,
            city          VARCHAR,
            state         VARCHAR,
            stars         FLOAT,
            review_count  INTEGER,
            categories    VARCHAR
        )
        """
    )
    cur.executemany(
        """
        INSERT INTO business_small
            (business_id, name, city, state, stars, review_count, categories)
        VALUES
            (:business_id, :name, :city, :state, :stars, :review_count, :categories)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["business_id", "name", "city", "state", "stars", "review_count", "categories"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = generate()
    write_sqlite(rows, DATA_DIR / "business_small.sqlite")
    write_csv(rows, DATA_DIR / "business_small.csv")
    print(f"Generated {len(rows)} businesses -> {DATA_DIR}")


if __name__ == "__main__":
    main()
