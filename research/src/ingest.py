"""Convert the raw Yelp JSON dump into columnar Parquet.

The raw archive is newline-delimited JSON: one object per line, ~5 GB in total,
with ``review.json`` alone accounting for most of it. Loading that with
``pandas.read_json`` needs far more memory than a laptop has, so every table is
streamed line by line and flushed to Parquet in fixed-size row groups. Peak
memory is therefore set by ``chunk_size``, not by the file size.

Only the columns the study uses are retained. For reviews in particular we drop
the review text after deriving what we need from it (see ``sentiment.py``);
carrying 6.7M review bodies through the pipeline would dominate both disk and
memory for no analytical gain.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import paths

log = logging.getLogger(__name__)

# Columns kept per table. Anything not listed is discarded at parse time.
BUSINESS_COLUMNS = (
    "business_id", "name", "address", "city", "state", "postal_code",
    "latitude", "longitude", "stars", "review_count", "is_open",
    "categories", "attributes",
)
REVIEW_COLUMNS = ("review_id", "business_id", "user_id", "stars", "date", "text")
TIP_COLUMNS = ("business_id", "user_id", "date", "text", "compliment_count")
CHECKIN_COLUMNS = ("business_id", "date")

DEFAULT_CHUNK_SIZE = 200_000


def iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one parsed object per line, skipping blank and malformed lines.

    Malformed lines are logged and skipped rather than raising: a single bad
    line in a multi-gigabyte dump should not abort a 20-minute ingest. The
    count is reported at the end so silent data loss is impossible.
    """
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 5:
                    log.warning("skipping malformed JSON at %s:%d", path.name, lineno)
    if bad:
        log.warning("skipped %d malformed line(s) in %s", bad, path.name)


def _project(record: dict[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    return {c: record.get(c) for c in columns}


def _normalize_business(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten the two nested business fields into stable scalar columns.

    ``categories`` arrives as a comma-separated string in recent releases and as
    a JSON list in older ones; ``attributes`` is a dict whose values are
    sometimes themselves stringified dicts. Both are reduced to strings here and
    parsed properly in ``features.py``, so the Parquet schema stays flat.
    """
    row = _project(record, BUSINESS_COLUMNS)

    cats = row.get("categories")
    if isinstance(cats, list):
        row["categories"] = ", ".join(str(c) for c in cats)
    elif cats is not None and not isinstance(cats, str):
        row["categories"] = str(cats)

    attrs = row.get("attributes")
    row["attributes"] = json.dumps(attrs, sort_keys=True) if isinstance(attrs, dict) else None

    return row


_NORMALIZERS = {"business": _normalize_business}
_COLUMNS = {
    "business": BUSINESS_COLUMNS,
    "review": REVIEW_COLUMNS,
    "tip": TIP_COLUMNS,
    "checkin": CHECKIN_COLUMNS,
}


def ingest_table(
    table: str,
    *,
    src: Path | None = None,
    dest: Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
) -> Path:
    """Stream one raw table to Parquet and return the written path.

    Args:
        table: one of ``business``, ``review``, ``tip``, ``checkin``.
        src: raw JSON path; defaults to the located file under ``YELP_RAW_DIR``.
        dest: output path; defaults to ``interim/<table>.parquet``.
        chunk_size: rows buffered before each Parquet row group is flushed.
            This sets peak memory.
        limit: stop after this many rows. For smoke tests only.
    """
    columns = _COLUMNS[table]
    normalize = _NORMALIZERS.get(table, lambda r: _project(r, columns))

    src = src or paths.raw_file(table)
    if dest is None:
        paths.ensure_dirs()
        dest = paths.INTERIM_DIR / f"{table}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    total = 0

    def flush() -> None:
        nonlocal writer, buffer
        if not buffer:
            return
        frame = pd.DataFrame(buffer, columns=list(columns))
        batch = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            # The first chunk defines the schema; later chunks are cast to it
            # so an all-null column in one chunk cannot change the column type.
            writer = pq.ParquetWriter(dest, batch.schema, compression="zstd")
        else:
            batch = batch.cast(writer.schema)
        writer.write_table(batch)
        buffer = []

    try:
        for record in iter_json_lines(src):
            buffer.append(normalize(record))
            total += 1
            if len(buffer) >= chunk_size:
                flush()
                log.info("%s: %d rows", table, total)
            if limit is not None and total >= limit:
                break
        flush()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError(f"no rows ingested from {src}")

    log.info("wrote %s (%d rows) -> %s", table, total, dest)
    return dest


def ingest_all(
    tables: tuple[str, ...] = ("business", "review", "tip", "checkin"),
    *,
    limit: int | None = None,
) -> dict[str, Path]:
    """Ingest every table the study uses. Returns table name -> Parquet path."""
    paths.ensure_dirs()
    return {t: ingest_table(t, limit=limit) for t in tables}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest raw Yelp JSON into Parquet.")
    parser.add_argument("--table", action="append", dest="tables",
                        choices=sorted(_COLUMNS), help="ingest only this table (repeatable)")
    parser.add_argument("--limit", type=int, help="stop after N rows (smoke test)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    selected = tuple(args.tables) if args.tables else ("business", "review", "tip", "checkin")
    for name, out in ingest_all(selected, limit=args.limit).items():
        size_mb = out.stat().st_size / 1e6
        print(f"{name:10} -> {out}  ({size_mb:.1f} MB)")
