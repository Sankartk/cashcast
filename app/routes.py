"""CashCast API routes."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .database import Branch, DailyCashRecord, ForecastResult, get_db
from .schemas import (
    BranchDetailResponse,
    BranchOut,
    DailyCashOut,
    DashboardSummary,
    ForecastOut,
)

router = APIRouter()


@router.get("/api/summary", response_model=DashboardSummary)
def get_summary(db: Session = Depends(get_db)) -> DashboardSummary:
    branches = db.query(Branch).all()
    today = date.today()
    week_end = today + timedelta(days=7)

    accuracies: list[float] = []
    weekly_demands: list[float] = []
    low_cash_count = 0

    for b in branches:
        fcs = (
            db.query(ForecastResult)
            .filter(
                ForecastResult.branch_id == b.id,
                ForecastResult.forecast_date >= today,
                ForecastResult.forecast_date < week_end,
            )
            .all()
        )
        weekly_demands.append(sum(f.predicted_dispensed for f in fcs))
        if fcs:
            accuracies.append(sum(f.confidence_score for f in fcs) / len(fcs) * 100)

        latest = (
            db.query(DailyCashRecord)
            .filter(DailyCashRecord.branch_id == b.id)
            .order_by(DailyCashRecord.record_date.desc())
            .first()
        )
        if latest and latest.closing_vault / b.vault_capacity < 0.15:
            low_cash_count += 1

    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    total_demand_7d = sum(weekly_demands)
    avg_daily = total_demand_7d / 7 if total_demand_7d else 0
    # Assume current practice over-orders by 15%; cost at 5.5% annual rate
    daily_rate = 0.055 / 365
    weekly_overfunding = avg_daily * 0.15 * 7 * daily_rate

    return DashboardSummary(
        total_branches=len(branches),
        avg_forecast_accuracy=round(avg_accuracy, 1),
        total_projected_demand_7d=round(total_demand_7d, 0),
        branches_low_cash=low_cash_count,
        weekly_overfunding_cost=round(weekly_overfunding, 2),
        savings_potential_monthly=round(weekly_overfunding * 4.33, 2),
    )


@router.get("/api/branches", response_model=list[BranchOut])
def list_branches(db: Session = Depends(get_db)) -> list[Branch]:
    return db.query(Branch).all()


@router.get("/api/branches/{branch_id}/forecast", response_model=BranchDetailResponse)
def branch_forecast(branch_id: int, db: Session = Depends(get_db)) -> BranchDetailResponse:
    branch = db.query(Branch).filter(Branch.id == branch_id).first()
    if not branch:
        raise HTTPException(status_code=404, detail="Branch not found")

    since = date.today() - timedelta(days=90)
    history = (
        db.query(DailyCashRecord)
        .filter(DailyCashRecord.branch_id == branch_id, DailyCashRecord.record_date >= since)
        .order_by(DailyCashRecord.record_date)
        .all()
    )

    forecasts = (
        db.query(ForecastResult)
        .filter(
            ForecastResult.branch_id == branch_id,
            ForecastResult.forecast_date >= date.today(),
        )
        .order_by(ForecastResult.forecast_date)
        .limit(14)
        .all()
    )

    anomaly_dates = [str(r.record_date) for r in history if r.is_anomaly]

    avg_conf = (
        sum(f.confidence_score for f in forecasts) / len(forecasts) * 100 if forecasts else 0.0
    )

    # Weekly demand pattern (all history)
    all_records = db.query(DailyCashRecord).filter(DailyCashRecord.branch_id == branch_id).all()
    df = pd.DataFrame(
        [(r.record_date, r.cash_dispensed) for r in all_records], columns=["d", "v"]
    )
    df["dow"] = df["d"].apply(lambda x: x.weekday())
    dow_means = df.groupby("dow")["v"].mean().to_dict()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_pattern = {day_names[k]: round(float(v), 2) for k, v in dow_means.items()}

    return BranchDetailResponse(
        branch=BranchOut.model_validate(branch),
        history_90d=[DailyCashOut.model_validate(r) for r in history],
        forecasts_14d=[ForecastOut.model_validate(f) for f in forecasts],
        forecast_accuracy=round(avg_conf, 1),
        anomaly_dates=anomaly_dates,
        weekly_pattern=weekly_pattern,
    )


@router.get("/api/forecast/all")
def all_branch_health(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    branches = db.query(Branch).all()
    today = date.today()
    result: list[dict[str, Any]] = []

    for b in branches:
        latest = (
            db.query(DailyCashRecord)
            .filter(DailyCashRecord.branch_id == b.id)
            .order_by(DailyCashRecord.record_date.desc())
            .first()
        )

        forecasts = (
            db.query(ForecastResult)
            .filter(
                ForecastResult.branch_id == b.id,
                ForecastResult.forecast_date >= today,
            )
            .order_by(ForecastResult.forecast_date)
            .limit(7)
            .all()
        )

        vault_pct = (latest.closing_vault / b.vault_capacity * 100) if latest else 0.0
        status = "CRITICAL" if vault_pct < 10 else "LOW" if vault_pct < 20 else "OK"

        anomaly_recent = (
            db.query(DailyCashRecord)
            .filter(
                DailyCashRecord.branch_id == b.id,
                DailyCashRecord.is_anomaly == True,  # noqa: E712
                DailyCashRecord.record_date >= today - timedelta(days=14),
            )
            .count()
        )
        if anomaly_recent > 0:
            status = "ANOMALY"

        avg_conf = (
            sum(f.confidence_score for f in forecasts) / len(forecasts) * 100
            if forecasts
            else 0.0
        )

        recent = (
            db.query(DailyCashRecord)
            .filter(DailyCashRecord.branch_id == b.id)
            .order_by(DailyCashRecord.record_date.desc())
            .limit(7)
            .all()
        )
        avg_net = float(np.mean([r.net_outflow for r in recent])) if recent else 0.0
        current_vault = latest.closing_vault if latest else 0.0
        low_threshold = b.vault_capacity * 0.15
        days_until_low = (
            int((current_vault - low_threshold) / avg_net) if avg_net > 0 else 999
        )

        narrative = next(
            (f.narrative for f in forecasts if f.narrative), ""
        )

        result.append(
            {
                "branch": {
                    "id": b.id,
                    "code": b.code,
                    "name": b.name,
                    "city": b.city,
                    "state": b.state,
                    "branch_type": b.branch_type,
                    "vault_capacity": b.vault_capacity,
                },
                "current_vault": round(current_vault, 0),
                "vault_pct": round(vault_pct, 1),
                "days_until_low": max(days_until_low, 0),
                "forecast_accuracy": round(avg_conf, 1),
                "status": status,
                "narrative": narrative or "No forecast narrative available.",
                "forecasts_7d": [
                    {
                        "date": str(f.forecast_date),
                        "predicted": round(f.predicted_dispensed, 0),
                        "order": round(f.recommended_order, 0),
                    }
                    for f in forecasts
                ],
            }
        )

    return result
