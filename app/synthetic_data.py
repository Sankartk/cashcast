"""Generates 2 years of realistic daily cash demand data for 6 bank branches.

Patterns modelled:
- Day-of-week seasonality (Friday peaks, weekend low)
- Monthly payday spikes (1st and 15th)
- Annual seasonality (December +45%, January -17%)
- US federal holidays (day-before surge, holiday-day drop)
- Branch-type profile (URBAN base >> RURAL base)
- Controlled random noise (std ≈ 11%)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ── US Federal Holidays 2024-2026 ────────────────────────────────────────────
US_HOLIDAYS: set[date] = {
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 5, 27),
    date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 11),
    date(2024, 11, 28), date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 5, 26),
    date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 11),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 4), date(2026, 9, 7), date(2026, 11, 11),
    date(2026, 11, 26), date(2026, 12, 25),
}

# 0 = Mon … 6 = Sun
_DOW_MULT: list[float] = [1.10, 0.92, 0.88, 0.94, 1.38, 0.68, 0.45]

_MONTH_MULT: dict[int, float] = {
    1: 0.83, 2: 0.87, 3: 0.98, 4: 1.02, 5: 1.05,
    6: 1.08, 7: 1.12, 8: 1.10, 9: 1.00, 10: 1.05,
    11: 1.22, 12: 1.45,
}

BRANCH_CONFIGS: list[dict[str, Any]] = [
    {
        "code": "BRK-001", "name": "Downtown Financial Branch",
        "city": "Chicago", "state": "IL", "branch_type": "URBAN",
        "avg_daily_traffic": 420, "vault_capacity": 500_000, "base": 38_000,
    },
    {
        "code": "BRK-002", "name": "Midway Suburban Branch",
        "city": "Oak Park", "state": "IL", "branch_type": "SUBURBAN",
        "avg_daily_traffic": 215, "vault_capacity": 300_000, "base": 22_000,
    },
    {
        "code": "BRK-003", "name": "Southside Community Branch",
        "city": "Chicago", "state": "IL", "branch_type": "URBAN",
        "avg_daily_traffic": 310, "vault_capacity": 400_000, "base": 29_000,
    },
    {
        "code": "BRK-004", "name": "Naperville Main Street",
        "city": "Naperville", "state": "IL", "branch_type": "SUBURBAN",
        "avg_daily_traffic": 185, "vault_capacity": 250_000, "base": 18_500,
    },
    {
        "code": "BRK-005", "name": "Waukegan North Branch",
        "city": "Waukegan", "state": "IL", "branch_type": "SUBURBAN",
        "avg_daily_traffic": 155, "vault_capacity": 200_000, "base": 14_000,
    },
    {
        "code": "BRK-006", "name": "Rockford Rural Express",
        "city": "Rockford", "state": "IL", "branch_type": "RURAL",
        "avg_daily_traffic": 95, "vault_capacity": 150_000, "base": 9_500,
    },
]


def _build_dispensed(base: float, dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    vals = np.empty(len(dates))
    for i, ts in enumerate(dates):
        d: date = ts.date()
        v = base * _DOW_MULT[d.weekday()] * _MONTH_MULT[d.month]

        # Payday spikes
        if d.day in (1, 15):
            v *= 1.85
        elif d.day in (2, 14, 16):
            v *= 1.32
        elif d.day in (3, 13, 17):
            v *= 1.12

        # Pre-holiday surge / holiday dip
        tomorrow = d + timedelta(days=1)
        if tomorrow in US_HOLIDAYS:
            v *= 1.45
        if d in US_HOLIDAYS:
            v *= 0.28  # ATM only, tellers closed

        # Controlled noise
        v *= float(rng.normal(1.0, 0.11))
        vals[i] = max(v, base * 0.08)

    return vals


def generate_all_branches(end_date: date | None = None) -> list[dict[str, Any]]:
    """Return synthetic daily records for all branches covering the last 730 days."""
    today = end_date or date.today()
    start = today - timedelta(days=730)
    dates = pd.date_range(start=start, end=today - timedelta(days=1), freq="D")

    all_records: list[dict[str, Any]] = []

    for idx, cfg in enumerate(BRANCH_CONFIGS):
        rng = np.random.default_rng(seed=42 + idx * 17)
        dispensed = _build_dispensed(cfg["base"], dates, rng)
        # Deposited ≈ 38-52% of dispensed (net positive outflow most days)
        deposited = dispensed * rng.uniform(0.38, 0.52, len(dates))

        vault = cfg["vault_capacity"] * 0.60
        for i, ts in enumerate(dates):
            net = dispensed[i] - deposited[i]
            closing = max(vault - net, cfg["vault_capacity"] * 0.05)
            all_records.append(
                {
                    "branch_code": cfg["code"],
                    "record_date": ts.date(),
                    "cash_dispensed": round(float(dispensed[i]), 2),
                    "cash_deposited": round(float(deposited[i]), 2),
                    "net_outflow": round(float(net), 2),
                    "opening_vault": round(float(vault), 2),
                    "closing_vault": round(float(closing), 2),
                    "transaction_count": max(int(dispensed[i] / 85), 1),
                }
            )
            vault = closing

    return all_records
