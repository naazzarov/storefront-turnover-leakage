"""Canonical filesystem layout for the cursed-storefronts study.

Every path is derived from two roots so the pipeline can be pointed at a
different machine by setting two environment variables:

    YELP_RAW_DIR   where the official Yelp Open Dataset JSON files live
    YELP_WORK_DIR  where intermediate and output artifacts are written

Raw data is never modified. Everything the pipeline produces lands under
``YELP_WORK_DIR`` so a full rebuild is a single ``rm -rf`` away.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPO_ROOT / "research"

RAW_DIR = Path(os.getenv("YELP_RAW_DIR", Path.home() / "yelp_raw_data")).expanduser()
WORK_DIR = Path(os.getenv("YELP_WORK_DIR", RESEARCH_ROOT / "work")).expanduser()

INTERIM_DIR = WORK_DIR / "interim"
PROCESSED_DIR = WORK_DIR / "processed"

OUTPUTS_DIR = RESEARCH_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

FIXTURES_DIR = RESEARCH_ROOT / "fixtures"

# The Yelp Open Dataset ships these filenames. Yelp has renamed them across
# releases (older archives drop the ``yelp_academic_dataset_`` prefix), so each
# logical table lists the spellings we accept.
RAW_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "business": ("yelp_academic_dataset_business.json", "business.json"),
    "review": ("yelp_academic_dataset_review.json", "review.json"),
    "user": ("yelp_academic_dataset_user.json", "user.json"),
    "checkin": ("yelp_academic_dataset_checkin.json", "checkin.json"),
    "tip": ("yelp_academic_dataset_tip.json", "tip.json"),
}


def raw_file(table: str) -> Path:
    """Locate a raw Yelp JSON file, searching one level of subdirectories.

    Raises FileNotFoundError with a message that names every path tried, so a
    missing download is obvious rather than surfacing later as an empty frame.
    """
    try:
        candidates = RAW_FILE_CANDIDATES[table]
    except KeyError:
        raise KeyError(f"unknown raw table {table!r}; expected one of {sorted(RAW_FILE_CANDIDATES)}") from None

    tried: list[Path] = []
    for name in candidates:
        direct = RAW_DIR / name
        tried.append(direct)
        if direct.exists():
            return direct
        # Extracted archives often nest everything one directory deep.
        for nested in sorted(RAW_DIR.glob(f"*/{name}")):
            tried.append(nested)
            if nested.exists():
                return nested

    listing = "\n  ".join(str(p) for p in tried)
    raise FileNotFoundError(
        f"could not find the raw {table!r} table. Tried:\n  {listing}\n"
        f"Download the Yelp Open Dataset from https://www.yelp.com/dataset and "
        f"extract it into {RAW_DIR}, or set YELP_RAW_DIR."
    )


def ensure_dirs() -> None:
    """Create every writable directory the pipeline needs."""
    for d in (INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
