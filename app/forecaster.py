"""CashCast forecasting engine.

Per-branch approach:
  1. Ridge regression with 17 engineered features (cyclic calendar + lag features)
  2. Train on oldest 80% of history, evaluate MAPE on newest 20%
  3. Isolation Forest on model residuals for anomaly detection
  4. Rule-based narrative summarising the 14-day outlook
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


# ── Feature engineering ───────────────────────────────────────────────────────

def _cyclic(series: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * math.pi * series / period
    return np.sin(angle), np.cos(angle)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from a DataFrame with 'record_date' and 'cash_dispensed' columns."""
    df = df.sort_values("record_date").reset_index(drop=True)
    dt = pd.to_datetime(df["record_date"])
    dow = dt.dt.dayofweek          # 0 = Mon
    dom = dt.dt.day                # 1-31
    month = dt.dt.month            # 1-12

    feats = pd.DataFrame(index=df.index)

    feats["dow_sin"], feats["dow_cos"] = _cyclic(dow, 7)
    feats["dom_sin"], feats["dom_cos"] = _cyclic(dom, 31)
    feats["month_sin"], feats["month_cos"] = _cyclic(month, 12)

    feats["is_friday"] = (dow == 4).astype(float)
    feats["is_monday"] = (dow == 0).astype(float)
    feats["is_weekend"] = (dow >= 5).astype(float)
    feats["is_payday_1"] = (dom == 1).astype(float)
    feats["is_payday_15"] = (dom == 15).astype(float)
    feats["is_near_payday"] = dom.isin([14, 16, 2]).astype(float)

    s = df["cash_dispensed"]
    feats["lag_7"] = s.shift(7).bfill()
    feats["lag_14"] = s.shift(14).bfill()
    feats["roll_7_mean"] = s.rolling(7, min_periods=1).mean()
    feats["roll_30_mean"] = s.rolling(30, min_periods=1).mean()
    feats["roll_7_std"] = s.rolling(7, min_periods=2).std().fillna(0.0)

    return feats


_FEATURE_COLS = [
    "dow_sin", "dow_cos", "dom_sin", "dom_cos", "month_sin", "month_cos",
    "is_friday", "is_monday", "is_weekend",
    "is_payday_1", "is_payday_15", "is_near_payday",
    "lag_7", "lag_14", "roll_7_mean", "roll_30_mean", "roll_7_std",
]

_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ── Forecaster ────────────────────────────────────────────────────────────────

class BranchForecaster:
    """Fits a per-branch Ridge regression model and predicts 14 days ahead."""

    def __init__(self, branch_id: int) -> None:
        self.branch_id = branch_id
        self._ridge = Ridge(alpha=10.0)
        self._scaler = StandardScaler()
        self._iso = IsolationForest(contamination=0.04, random_state=42)
        self.mape: float = 0.0
        self._fitted = False
        self._history_tail: pd.DataFrame = pd.DataFrame()

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """Fit model. df must have 'record_date' and 'cash_dispensed' columns."""
        df = df.sort_values("record_date").reset_index(drop=True)
        X = build_features(df)[_FEATURE_COLS]
        y = df["cash_dispensed"].values

        split = int(len(df) * 0.80)
        Xtr_s = self._scaler.fit_transform(X.iloc[:split])
        self._ridge.fit(Xtr_s, y[:split])

        yte_pred = np.maximum(self._ridge.predict(self._scaler.transform(X.iloc[split:])), 0)
        self.mape = float(mean_absolute_percentage_error(y[split:], yte_pred) * 100)

        # Anomaly detection on full-dataset residuals
        residuals = (y - np.maximum(self._ridge.predict(self._scaler.transform(X)), 0)).reshape(-1, 1)
        self._iso.fit(residuals)

        self._history_tail = df[["record_date", "cash_dispensed"]].copy()
        self._fitted = True

    # ── predict ─────────────────────────────────────────────────────────────

    def predict_14d(self) -> list[dict]:
        """Return list of 14 forecast dicts, one per day starting tomorrow."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_14d()")

        tail_vals = list(self._history_tail.sort_values("record_date")["cash_dispensed"].values)
        today = date.today()
        results: list[dict] = []

        for offset in range(1, 15):
            fdate = today + timedelta(days=offset)
            dow = fdate.weekday()
            dom = fdate.day
            month = fdate.month

            lag7 = tail_vals[-7] if len(tail_vals) >= 7 else float(np.mean(tail_vals))
            lag14 = tail_vals[-14] if len(tail_vals) >= 14 else float(np.mean(tail_vals))
            roll7_vals = tail_vals[-7:] if len(tail_vals) >= 7 else tail_vals
            roll30_vals = tail_vals[-30:] if len(tail_vals) >= 30 else tail_vals
            roll7_mean = float(np.mean(roll7_vals))
            roll30_mean = float(np.mean(roll30_vals))
            roll7_std = float(np.std(roll7_vals)) if len(roll7_vals) > 1 else 0.0

            feat = np.array([[
                math.sin(2 * math.pi * dow / 7),
                math.cos(2 * math.pi * dow / 7),
                math.sin(2 * math.pi * dom / 31),
                math.cos(2 * math.pi * dom / 31),
                math.sin(2 * math.pi * month / 12),
                math.cos(2 * math.pi * month / 12),
                float(dow == 4), float(dow == 0), float(dow >= 5),
                float(dom == 1), float(dom == 15), float(dom in (14, 16, 2)),
                lag7, lag14, roll7_mean, roll30_mean, roll7_std,
            ]])

            pred = float(max(self._ridge.predict(self._scaler.transform(feat))[0], 0))

            # Confidence band: wider for volatile branches
            volatility = roll7_std / (roll7_mean + 1e-6)
            band_pct = min(0.10 + volatility * 0.40, 0.22)
            lower = pred * (1 - band_pct)
            upper = pred * (1 + band_pct)

            # 10% safety buffer, rounded to nearest $1 000
            recommended = round((upper * 1.10) / 1_000) * 1_000

            confidence = float(max(0.60, min(1.0 - (self.mape / 100) - volatility * 0.25, 0.99)))

            results.append(
                {
                    "forecast_date": fdate,
                    "predicted_dispensed": round(pred, 2),
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                    "recommended_order": float(recommended),
                    "confidence_score": round(confidence, 3),
                }
            )
            tail_vals.append(pred)

        return results

    # ── anomaly flags ────────────────────────────────────────────────────────

    def anomaly_flags(self, df: pd.DataFrame) -> list[str]:
        """Return ISO date strings where Isolation Forest flagged an anomaly."""
        if not self._fitted:
            raise RuntimeError("Call fit() before anomaly_flags()")
        df = df.sort_values("record_date").reset_index(drop=True)
        X = build_features(df)[_FEATURE_COLS]
        preds = np.maximum(self._ridge.predict(self._scaler.transform(X)), 0)
        residuals = (df["cash_dispensed"].values - preds).reshape(-1, 1)
        flags = self._iso.predict(residuals)  # -1 = anomaly
        return [str(d) for d in df.loc[flags == -1, "record_date"].values]


# ── Narrative generator ───────────────────────────────────────────────────────

def generate_narrative(
    branch_name: str,
    forecasts: list[dict],
    hist_median: float,
    anomaly_count: int,
) -> str:
    """Produce a plain-English outlook for the next 14 days."""
    avg_7d = sum(f["predicted_dispensed"] for f in forecasts[:7]) / 7
    delta_pct = (avg_7d - hist_median) / (hist_median + 1e-6) * 100

    peak = max(forecasts, key=lambda x: x["predicted_dispensed"])
    peak_str = peak["forecast_date"].strftime("%b %d")

    parts: list[str] = []

    if delta_pct > 12:
        parts.append(f"Next-week demand tracks {delta_pct:.0f}% above the 90-day median.")
    elif delta_pct < -10:
        parts.append(f"Next-week demand is expected {abs(delta_pct):.0f}% below seasonal baseline.")
    else:
        parts.append("Next-week demand aligns with the seasonal baseline.")

    parts.append(f"Peak cash need: {peak_str} at ${peak['predicted_dispensed']:,.0f}.")

    idle_risk = sum(
        max(f["recommended_order"] - f["predicted_dispensed"] * 1.10, 0)
        for f in forecasts
    )
    if idle_risk > 8_000:
        parts.append(
            f"Ordering at current cadence risks ~${idle_risk:,.0f} in idle vault cash over 14 days "
            f"(opportunity cost at 5.5% APR)."
        )

    if anomaly_count > 0:
        parts.append(
            f"{anomaly_count} anomalous demand day(s) detected in recent history — manual review recommended."
        )

    return " ".join(parts)
