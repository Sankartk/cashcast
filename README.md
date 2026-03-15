# CashCast

**Intelligent branch cash demand forecasting for retail banks.**

Banks overfund ATM and teller vaults by an average of 15-20%, keeping millions in idle cash that earns nothing. With the Fed funds rate elevated throughout 2025-2026, each dollar sitting in a vault costs real money. CashCast applies machine learning to 2 years of daily transaction history to predict exactly how much cash each branch needs — eliminating that idle-cash overhead.

## What it does

- **Per-branch Ridge regression model** with 17 engineered features (cyclic calendar encoding, lag features, rolling statistics)
- **Isolation Forest anomaly detection** — flags unusual demand days so operators can investigate before they become a problem
- **14-day rolling forecast** with confidence bands and safety-buffered order recommendations
- **Operations dashboard** — Plotly.js charts, branch health cards, AI-generated demand narratives, cost savings calculator
- **Zero external dependencies** — runs entirely on SQLite with synthetic data out of the box

## Problem context

| Manual process | CashCast |
|---|---|
| Branch manager estimates next week's demand from memory | Ridge regression trained on 730 days of actual history |
| Orders 20-25% buffer "just in case" | Orders upper-bound forecast + 10% buffer |
| Idle vault cash costs ~$4.2B/year industry-wide (Fed study) | 15% reduction → ~$630M annual savings across US community banks |
| Anomalies discovered only after a branch runs low | Isolation Forest flags deviations in advance |

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.115, Python 3.12 |
| Database | SQLAlchemy 2.0 + SQLite (swap `DATABASE_URL` env var for Postgres) |
| ML | scikit-learn 1.5 — Ridge regression + Isolation Forest |
| Data | Pandas 2.2, NumPy 2.1 |
| Frontend | Plotly.js 2.27 (CDN), Jinja2 templates |
| Tests | pytest 8.3 |

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/Sankartk/cashcast.git
cd cashcast
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Run (starts on port 8001 to avoid conflicts)
uvicorn app.main:app --reload --port 8001

# 3. Open dashboard
# http://localhost:8001
```

On first startup, CashCast:
1. Creates the SQLite schema
2. Generates 730 days of synthetic cash data for 6 Chicago-area bank branches
3. Trains a Ridge regression model per branch (~1 second per branch)
4. Runs Isolation Forest anomaly detection across all history
5. Stores 14-day forecasts with confidence bands

All subsequent starts skip seeding and load immediately.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Operations dashboard |
| `GET` | `/api/summary` | Network-level KPIs |
| `GET` | `/api/branches` | List all branches |
| `GET` | `/api/branches/{id}/forecast` | Branch history + 14-day forecast |
| `GET` | `/api/forecast/all` | All branches health snapshot |

## Running tests

```bash
pytest tests/ -v
```

## Project layout

```
cashcast/
├── app/
│   ├── main.py            # FastAPI app, startup seeding
│   ├── database.py        # SQLAlchemy models
│   ├── schemas.py         # Pydantic response schemas
│   ├── routes.py          # API endpoints
│   ├── forecaster.py      # Ridge + Isolation Forest engine
│   ├── synthetic_data.py  # 730-day data generator
│   └── templates/
│       └── dashboard.html # Plotly.js operations dashboard
└── tests/
    └── test_forecaster.py # Unit tests (no DB required)
```
