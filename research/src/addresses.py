"""Address normalization and location keys.

The central object of this study is a *location* -- a physical storefront that
one or more businesses have occupied over time. The Yelp dataset has no such
entity: it has businesses, each carrying a free-text address. Two businesses
that occupied the same unit therefore appear as unrelated rows whose addresses
agree only up to punctuation, abbreviation and whitespace:

    "475 3rd St"      "475 3rd Street"      "475 3rd St."      "475  3rd st"

Grouping those into one location is the precondition for everything downstream,
and doing it wrongly biases the headline result in both directions: under-merging
hides turnover, over-merging invents it. We therefore normalize conservatively
and pair the textual key with a coordinate key (see ``locations.py``), so a
merge has to be defensible on both address text and geography.
"""

from __future__ import annotations

import re
import unicodedata

# Street-type abbreviations, mapped to a single canonical token. Sourced from
# the USPS Publication 28 suffix table, restricted to suffixes that actually
# occur in the Yelp metropolitan areas.
_STREET_SUFFIXES: dict[str, str] = {
    "street": "st", "str": "st", "st": "st",
    "avenue": "ave", "aven": "ave", "av": "ave", "ave": "ave",
    "boulevard": "blvd", "boul": "blvd", "blvd": "blvd",
    "road": "rd", "rd": "rd",
    "drive": "dr", "driv": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
    "circle": "cir", "circ": "cir", "cir": "cir",
    "place": "pl", "pl": "pl",
    "plaza": "plz", "plz": "plz",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "highway": "hwy", "hwy": "hwy",
    "square": "sq", "sq": "sq",
    "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl",
    "way": "way",
    "expressway": "expy", "expy": "expy",
    "freeway": "fwy", "fwy": "fwy",
    "turnpike": "tpke", "tpke": "tpke",
    "loop": "loop",
    "route": "rte", "rte": "rte",
    "crossing": "xing", "xing": "xing",
    "center": "ctr", "centre": "ctr", "ctr": "ctr",
}

# Directionals, normalized to their compass abbreviation.
_DIRECTIONALS: dict[str, str] = {
    "north": "n", "n": "n",
    "south": "s", "s": "s",
    "east": "e", "e": "e",
    "west": "w", "w": "w",
    "northeast": "ne", "ne": "ne",
    "northwest": "nw", "nw": "nw",
    "southeast": "se", "se": "se",
    "southwest": "sw", "sw": "sw",
}

# Secondary-unit designators. These are the part of an address that
# distinguishes tenants *within* one building -- suite 100 vs suite 200 are
# different storefronts and must not be merged.
_UNIT_DESIGNATORS = (
    "suite", "ste", "unit", "apt", "apartment", "building", "bldg",
    "floor", "fl", "room", "rm", "space", "spc", "lot", "trailer", "trlr",
    "department", "dept", "office", "ofc", "kiosk", "stall", "booth",
)

_UNIT_RE = re.compile(
    r"\b(?:" + "|".join(_UNIT_DESIGNATORS) + r")\b\.?\s*#?\s*([0-9a-z\-]+)",
    re.IGNORECASE,
)
# A bare "#12" suffix is the other common way Yelp records a unit.
_HASH_UNIT_RE = re.compile(r"#\s*([0-9a-z\-]+)", re.IGNORECASE)

_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[.,;:'\"()\[\]/\\]")
_WS_RE = re.compile(r"\s+")
_HOUSE_NUMBER_RE = re.compile(r"^(\d+)(?:\s*-\s*\d+)?\b")


def _as_text(value: object) -> str:
    """Coerce a field to text, treating missing values as empty.

    Real exports deliver missing cells as ``None``, as float ``nan`` (pandas
    does this even under ``dtype=str``), or as the literal strings "nan" and
    "none". All of them mean "absent" and must not become part of a key.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        # nan is the only float that is not equal to itself.
        return "" if value != value else str(value)
    if not isinstance(value, str):
        value = str(value)
    return "" if value.strip().lower() in {"", "nan", "none", "null", "n/a"} else value


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def split_unit(address: str) -> tuple[str, str | None]:
    """Separate the unit/suite designator from the street address.

    Returns ``(street_part, unit)`` where ``unit`` is ``None`` when the address
    names no secondary unit. The unit is kept rather than discarded: two tenants
    of the same building at different suites are different storefronts.

    >>> split_unit("123 Main St Suite 200")
    ('123 Main St', '200')
    >>> split_unit("123 Main St #4B")
    ('123 Main St', '4b')
    >>> split_unit("123 Main St")
    ('123 Main St', None)
    """
    address = _as_text(address)
    if not address:
        return "", None

    unit: str | None = None

    def _capture(match: re.Match[str]) -> str:
        nonlocal unit
        if unit is None:
            unit = match.group(1).lower().lstrip("-")
        return " "

    remainder = _UNIT_RE.sub(_capture, address)
    if unit is None:
        remainder = _HASH_UNIT_RE.sub(_capture, remainder)

    return _WS_RE.sub(" ", remainder).strip(" ,-"), unit


def normalize_address(address: str | None) -> str:
    """Reduce a free-text street address to a canonical comparable form.

    The output keeps the house number, the normalized street name, and the unit
    (appended as ``#unit``) so that distinct tenancies stay distinct.

    >>> normalize_address("475 3rd Street")
    '475 3 st'
    >>> normalize_address("475 3rd St.")
    '475 3 st'
    >>> normalize_address("  475   3rd st ")
    '475 3 st'
    >>> normalize_address("1000 North Main Avenue Suite 5")
    '1000 n main ave #5'
    >>> normalize_address(None)
    ''
    """
    address = _as_text(address)
    if not address:
        return ""

    street, unit = split_unit(address)

    text = _strip_accents(street).lower()
    text = _PUNCT_RE.sub(" ", text)
    # "3rd" -> "3" so ordinal spellings collapse onto one another.
    text = _ORDINAL_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return f"#{unit}" if unit else ""

    tokens = text.split(" ")
    out: list[str] = []
    for i, tok in enumerate(tokens):
        # Only map directionals at the edges of the street name; "west" in
        # "Key West Rd" is part of the name, not a directional prefix.
        if i in (0, 1, len(tokens) - 1) and tok in _DIRECTIONALS and len(tokens) > 2:
            out.append(_DIRECTIONALS[tok])
            continue
        if i == len(tokens) - 1 and tok in _STREET_SUFFIXES:
            out.append(_STREET_SUFFIXES[tok])
            continue
        out.append(tok)

    normalized = " ".join(out)
    # A leading house number that is itself a directional-looking token stays
    # as-is; we only canonicalize the number range "100-102" down to "100".
    normalized = _HOUSE_NUMBER_RE.sub(r"\1", normalized, count=1)

    return f"{normalized} #{unit}" if unit else normalized


def normalize_city(city: str | None) -> str:
    """Canonicalize a city name for use in a grouping key.

    >>> normalize_city(" Saint   Louis ")
    'st louis'
    >>> normalize_city("Montréal")
    'montreal'
    """
    city = _as_text(city)
    if not city:
        return ""
    text = _strip_accents(city).lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    text = re.sub(r"\bsaint\b", "st", text)
    text = re.sub(r"\bmount\b", "mt", text)
    text = re.sub(r"\bfort\b", "ft", text)
    return text


def normalize_postal(postal_code: str | None) -> str:
    """Reduce a postal code to its comparable prefix.

    US ZIP+4 collapses to the 5-digit ZIP; Canadian codes keep the forward
    sortation area plus the local unit, without the space.

    >>> normalize_postal("94107-1234")
    '94107'
    >>> normalize_postal("m5v 2t6")
    'M5V2T6'
    """
    postal_code = _as_text(postal_code)
    if not postal_code:
        return ""
    text = re.sub(r"[^0-9a-zA-Z]", "", postal_code).upper()
    if not text:
        return ""
    if text[:5].isdigit():
        return text[:5]
    return text


def address_key(
    address: str | None,
    city: str | None,
    state: str | None,
    postal_code: str | None,
) -> str:
    """Build the textual location key used to group businesses.

    The postal code carries most of the disambiguating power (the same street
    address recurs across cities), so it is included ahead of the city name,
    which is the noisiest of the four fields.

    Returns ``""`` when the address is unusable, which callers must treat as
    "not groupable" rather than as a group of its own.

    >>> address_key("475 3rd Street", "San Francisco", "CA", "94107-1234")
    '94107|ca|475 3 st'
    >>> address_key("475 3rd St.", "San Francisco", "CA", "94107")
    '94107|ca|475 3 st'
    """
    street = normalize_address(address)
    if not street:
        return ""
    # An address with no house number ("Main St") is too vague to anchor a
    # storefront; treat it as unusable.
    if not re.match(r"^\d", street):
        return ""

    postal = normalize_postal(postal_code)
    st = _as_text(state).strip().lower()
    if not postal:
        # Fall back to the city when the postal code is missing.
        postal = f"city:{normalize_city(city)}"
        if postal == "city:":
            return ""

    return f"{postal}|{st}|{street}"
