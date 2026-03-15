"""CashCast FastAPI application.

Startup sequence:
  1. Create SQLite tables
  2. If database is empty: generate 2 years of synthetic cash data,
     train a per-branch Ridge regression model, run anomaly detection,
     and store 14-day forecasts.

Run:
    uvicorn app.main:app --reload --port 8001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from .database import Branch, DailyCashRecord, ForecastResult, SessionLocal, create_tables
from .forecaster import BranchForecaster, generate_narrative
from .routes import router
from .synthetic_data import BRANCH_CONFIGS, generate_all_branches

log = logging.getLogger("cashcast")
templates = Jinja2Templates(directory="app/templates")


# ── Startup seeding ──────────────────────────────────────────────────────────

def _seed_and_forecast() -> None:
    db = SessionLocal()
    try:
        if db.query(Branch).count() > 0:
            log.info("Database already seeded — skipping.")
            return

        log.info("Seeding database with synthetic branch data…")

        # Insert branches
        branch_id_map: dict[str, int] = {}
        for cfg in BRANCH_CONFIGS:
            b = Branch(
                code=cfg["code"],
                name=cfg["name"],
                city=cfg["city"],
                state=cfg["state"],
                branch_type=cfg["branch_type"],
                avg_daily_traffic=cfg["avg_daily_traffic"],
                vault_capacity=cfg["vault_capacity"],
            )
            db.add(b)
            db.flush()
            branch_id_map[cfg["code"]] = b.id

        # Generate 730-day history
        raw_records = generate_all_branches()

        # Group by branch so we can bulk-insert then train per branch
        records_by_branch: dict[str, list] = {c: [] for c in branch_id_map}
        for r in raw_records:
            records_by_branch[r["branch_code"]].append(r)

        for code, rows in records_by_branch.items():
            bid = branch_id_map[code]
            for r in rows:
                db.add(
                    DailyCashRecord(
                        branch_id=bid,
                        record_date=r["record_date"],
                        cash_dispensed=r["cash_dispensed"],
                        cash_deposited=r["cash_deposited"],
                        net_outflow=r["net_outflow"],
                        opening_vault=r["opening_vault"],
                        closing_vault=r["closing_vault"],
                        transaction_count=r["transaction_count"],
                    )
                )
        db.commit()
        log.info("Daily records inserted.")

        # Train models and generate forecasts
        for code, bid in branch_id_map.items():
            rows_db = (
                db.query(DailyCashRecord)
                .filter(DailyCashRecord.branch_id == bid)
                .all()
            )
            df = pd.DataFrame(
                [{"record_date": r.record_date, "cash_dispensed": r.cash_dispensed} for r in rows_db]
            )

            fc = BranchForecaster(bid)
            fc.fit(df)
            log.info("Branch %s — MAPE %.1f%%", code, fc.mape)

            # Flag anomalies in historical data
            anomaly_set = set(fc.anomaly_flags(df))
            for row in rows_db:
                if str(row.record_date) in anomaly_set:
                    row.is_anomaly = True

            hist_median = float(df["cash_dispensed"].median())
            preds = fc.predict_14d()
            narrative = generate_narrative(code, preds, hist_median, len(anomaly_set))

            for i, p in enumerate(preds):
                db.add(
                    ForecastResult(
                        branch_id=bid,
                        forecast_date=p["forecast_date"],
                        generated_at=datetime.now(),
                        predicted_dispensed=p["predicted_dispensed"],
                        lower_bound=p["lower_bound"],
                        upper_bound=p["upper_bound"],
                        recommended_order=p["recommended_order"],
                        confidence_score=p["confidence_score"],
                        narrative=narrative if i == 0 else None,
                    )
                )
        db.commit()
        log.info("Forecasts stored. Startup complete.")

    finally:
        db.close()


# ── Application ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    create_tables()
    _seed_and_forecast()
    yield


app = FastAPI(
    title="CashCast",
    description="""
## CashCast — Branch Cash Demand Forecasting

Retail bank branches typically pad vault orders by **15–20%** as a safety buffer.
CashCast replaces that buffer with a **per-branch Ridge regression** forecast trained
on 730 days of cash dispensing history, cutting idle vault float without creating gaps.

### How it works
- **17-feature model**: cyclic day/month encodings, payday flags, lag & rolling stats
- **Isolation Forest** anomaly detection on historical residuals  
- **Safety-buffered order rec** rounded to nearest $1,000  
- **AI narrative** per branch: peak day, seasonal trend, idle cash risk

### Explore
| Endpoint | Description |
|----------|-------------|
| `GET /` | Operations dashboard (Plotly.js) |
| `GET /docs` | Swagger UI ← you are here |
| `GET /redoc` | ReDoc documentation |
| `GET /api/summary` | Portfolio KPIs |
| `GET /api/branches` | Branch registry |
| `GET /api/forecast/all` | All-branch health snapshot |
""",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "branches",
            "description": "Branch registry — list and query branch metadata.",
        },
        {
            "name": "forecasts",
            "description": "ML forecasts, demand outlooks, 14-day order recommendations, "
            "and portfolio-level KPIs.",
        },
        {
            "name": "health",
            "description": "Service health and readiness checks.",
        },
    ],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})
