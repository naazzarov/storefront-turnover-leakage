"""Download the full Chicago business-licence dataset via the Socrata API.

The portal's one-click CSV export silently truncates: it returns roughly 228k of
the 1.21M rows with HTTP 200 and no warning, which would quietly bias every
result computed from it. We therefore page the API instead and verify the row
count against the server's own ``count(1)`` before declaring success.

Pagination is ordered by ``:id``, Socrata's immutable internal row identifier.
Ordering by a data column instead risks rows shifting between pages if the
dataset is updated mid-download, which would duplicate some rows and drop others.

The dataset (``r5kz-chrr``) is published by the City of Chicago as public domain
under the Chicago Open Data policy, so unlike the Yelp corpus it may be
redistributed with the paper's replication package.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

DOMAIN = "data.cityofchicago.org"
DATASET_ID = "r5kz-chrr"
BASE_URL = f"https://{DOMAIN}/resource/{DATASET_ID}.json"

PAGE_SIZE = 50_000
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 3.0

# API field names, in the order they are written to CSV.
#
# Three of these differ from the display names the portal shows, which is what
# ``chicago.load_licences`` reads: the API calls them ``license_start_date``,
# ``expiration_date`` and ``license_status_change_date``, while the CSV header
# spells them "LICENSE TERM START DATE", "LICENSE TERM EXPIRATION DATE" and
# "LICENSE STATUS CHANGE DATE". ``HEADER`` below restores the display spelling,
# so a file fetched here is interchangeable with a manual portal export.
FIELDS = (
    "id", "license_id", "account_number", "site_number",
    "legal_name", "doing_business_as_name",
    "address", "city", "state", "zip_code",
    "ward", "community_area_name", "neighborhood",
    "license_code", "license_description", "business_activity",
    "application_type", "license_status",
    "license_start_date", "expiration_date",
    "date_issued", "license_status_change_date",
    "latitude", "longitude",
)

# chicago.load_licences expects the portal's display names.
HEADER = (
    "ID", "LICENSE ID", "ACCOUNT NUMBER", "SITE NUMBER",
    "LEGAL NAME", "DOING BUSINESS AS NAME",
    "ADDRESS", "CITY", "STATE", "ZIP CODE",
    "WARD", "COMMUNITY AREA NAME", "NEIGHBORHOOD",
    "LICENSE CODE", "LICENSE DESCRIPTION", "BUSINESS ACTIVITY",
    "APPLICATION TYPE", "LICENSE STATUS",
    "LICENSE TERM START DATE", "LICENSE TERM EXPIRATION DATE",
    "DATE ISSUED", "LICENSE STATUS CHANGE DATE",
    "LATITUDE", "LONGITUDE",
)


def _get(url: str, *, timeout: int = 180) -> bytes:
    """Fetch a URL, retrying on transient failures with linear backoff."""
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt == MAX_RETRIES:
                break
            wait = RETRY_BACKOFF_SECONDS * attempt
            log.warning("request failed (%s); retry %d/%d in %.0fs",
                        exc, attempt, MAX_RETRIES, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up after {MAX_RETRIES} attempts: {last}")


def remote_row_count() -> int:
    """Ask the server how many rows the dataset holds."""
    url = f"{BASE_URL}?{urllib.parse.urlencode({'$select': 'count(1)'})}"
    payload = json.loads(_get(url, timeout=60).decode("utf-8"))
    return int(payload[0]["count_1"])


def fetch(dest: Path, *, page_size: int = PAGE_SIZE, limit: int | None = None) -> int:
    """Page the dataset into ``dest`` as CSV. Returns the row count written."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    expected = remote_row_count()
    target = min(expected, limit) if limit else expected
    log.info("dataset reports %d rows; downloading %d", expected, target)

    written = 0
    offset = 0
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)

        while written < target:
            take = min(page_size, target - written)
            query = urllib.parse.urlencode({
                "$select": ",".join(FIELDS),
                "$order": ":id",
                "$limit": take,
                "$offset": offset,
            })
            rows = json.loads(_get(f"{BASE_URL}?{query}").decode("utf-8"))
            if not rows:
                log.warning("server returned an empty page at offset %d; stopping", offset)
                break

            for row in rows:
                writer.writerow([row.get(f, "") for f in FIELDS])

            written += len(rows)
            offset += len(rows)
            log.info("%d/%d rows (%.0f%%)", written, target, 100 * written / target)

    if limit is None and written != expected:
        # Loud rather than silent: a short download is exactly the failure that
        # made the portal's own CSV export unusable.
        log.error("INCOMPLETE: wrote %d rows, expected %d", written, expected)
    else:
        log.info("wrote %d rows -> %s", written, dest)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path.home() / "chicago_raw_data" / "business_licenses_full.csv")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--limit", type=int, help="stop after N rows (smoke test)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    written = fetch(args.out, page_size=args.page_size, limit=args.limit)
    print(f"{written} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
