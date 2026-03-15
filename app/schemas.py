"""Pydantic schemas for CashCast API responses."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class BranchOut(BaseModel):
    id: int
    code: str
    name: str
    city: str
    state: str
    branch_type: str
    vault_capacity: float

    model_config = {"from_attributes": True}


class DailyCashOut(BaseModel):
    id: int
    branch_id: int
    record_date: date
    cash_dispensed: float
    cash_deposited: float
    net_outflow: float
    closing_vault: float
    transaction_count: int
    is_anomaly: bool

    model_config = {"from_attributes": True}


class ForecastOut(BaseModel):
    branch_id: int
    forecast_date: date
    predicted_dispensed: float
    lower_bound: float
    upper_bound: float
    recommended_order: float
    confidence_score: float
    narrative: Optional[str] = None

    model_config = {"from_attributes": True}


class DashboardSummary(BaseModel):
    total_branches: int
    avg_forecast_accuracy: float
    total_projected_demand_7d: float
    branches_low_cash: int
    weekly_overfunding_cost: float
    savings_potential_monthly: float


class BranchHealthCard(BaseModel):
    branch: BranchOut
    current_vault: float
    vault_pct: float
    days_until_low: int
    forecast_accuracy: float
    status: str  # OK | LOW | CRITICAL | ANOMALY
    narrative: str
    forecasts_7d: list[dict]


class BranchDetailResponse(BaseModel):
    branch: BranchOut
    history_90d: list[DailyCashOut]
    forecasts_14d: list[ForecastOut]
    forecast_accuracy: float
    anomaly_dates: list[str]
    weekly_pattern: dict[str, float]
