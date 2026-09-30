# Storefront turnover and evaluation leakage

Two related pieces of work in one repository.

**`research/`** — the pipeline and paper for *Spatial Blocking Does Not Prevent
Group-Aggregate Leakage: Mechanism, Magnitude and Limits in Urban Prediction*.
Built on the City of Chicago business licence register (1.21M records,
1995–2026) with a replication on New York City.

**`app/` and `benchmark/`** — an earlier student project: a FastAPI service that
turns natural-language questions into SQL over a Yelp business table, plus a
hand-written benchmark for evaluating that translation. Unrelated to the paper
and kept for history.

---

## research/

### What the paper argues

Splitting data randomly for cross-validation leaks information when features are
built from group averages. The standard fix for geographic data — splitting by
spatial block — does not close that leak, because the leak runs through *group
membership* rather than distance.

On this task the default protocol reports ROC AUC 0.970. Partitioning by
administrative unit reports 0.693.

| | |
|---|---|
| Leak source | ward / community-area aggregates, +0.321 AUC |
| Not the leak | 50–500 m neighbourhood features, +0.034 AUC |
| Mechanism | leave-one-out withholds only a 1/\|G\| share of the signal |
| Bound | New York reproduces it at one fifth the magnitude |

### Layout

```
research/
├── src/                    pipeline modules
│   ├── fetch_chicago.py    paged download, verifies row count against server
│   ├── chicago.py          licences -> tenancies -> locations
│   ├── nyc.py              the same for New York
│   ├── addresses.py        address normalisation and location keys
│   ├── cursed.py           target definition (three variants)
│   ├── features.py         leave-one-out contextual features
│   ├── models.py           evaluation protocols and survival
│   └── figures.py          every figure, drawn from saved tables
├── tests/                  70 tests
├── run_experiments.py      reproduces every number in the paper
├── paper/                  main.tex, main.pdf
├── arxiv/                  flat submission package
└── outputs/                figures and result tables
```

### Running it

Needs pandas, scikit-learn, scipy, statsmodels and matplotlib. On this machine
that means the Anaconda interpreter, not the system Python:

```bash
cd research
/opt/anaconda3/bin/python3 -m src.fetch_chicago     # ~440 MB
/opt/anaconda3/bin/python3 run_experiments.py
/opt/anaconda3/bin/python3 -m pytest tests/ -q
```

`fetch_chicago.py` checks the downloaded row count against the server's own
total. This matters: the portal's one-click CSV export silently returns about
228k of 1.21M rows with HTTP 200 and no warning, which produces plausible
output and is very hard to notice downstream.

Raw data lives outside the repository (`~/chicago_raw_data/`, `~/nyc_raw_data/`)
and is regenerable from the public source.

---

## app/ and benchmark/

The earlier text-to-SQL project. `app/` is a FastAPI service that sends a
question plus a schema description to an LLM, checks the returned SQL is a
SELECT, and runs it against PostgreSQL. `benchmark/` holds 100 hand-written
question/SQL pairs and a harness reporting execution rate and exact-match
accuracy.

```bash
pip install -r requirements.txt
cp .env.example .env          # needs OPENAI_API_KEY and DB_* values
uvicorn app.main:app --reload
```

The credentials in `.env.example` are placeholders; the database host it points
at no longer exists.

---

## Data sources

| Source | Licence |
|---|---|
| Chicago business licences (`r5kz-chrr`) | public domain |
| NYC issued licences (`w7w3-xahh`) | public domain |
| Yelp Open Dataset (`app/`, `benchmark/`) | restricted; not redistributed here |

Chicago and New York publish their registers openly, so anything derived from
them can be rebuilt from the published source.
