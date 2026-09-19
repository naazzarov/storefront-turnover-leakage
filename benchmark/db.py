"""SQLite helpers for the benchmark.

Runs SQL against the generated business_small database and returns results
as lists of tuples suitable for set comparison. Uses only the stdlib sqlite3.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "business_small.sqlite"

FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER"]


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    return conn


def is_select(sql: str) -> bool:
    return sql.strip().upper().startswith("SELECT")


def is_safe(sql: str) -> bool:
    upper = sql.upper()
    return not any(kw in upper for kw in FORBIDDEN_KEYWORDS)


def run_query(sql: str) -> list[tuple]:
    """Execute a query and return rows as a list of tuples, or raise."""
    if not is_select(sql):
        raise ValueError("Only SELECT queries are allowed")
    if not is_safe(sql):
        raise ValueError("Forbidden SQL keyword detected")

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return [tuple(r) for r in rows]
    finally:
        conn.close()


def gold_result(sql: str) -> list[tuple]:
    """Ground-truth result for a question's gold SQL."""
    return run_query(sql)
