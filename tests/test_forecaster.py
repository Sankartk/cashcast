"""Unit tests for the CashCast forecasting engine.

These tests do not require a database — they exercise BranchForecaster
and generate_narrative directly with synthetic data.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.forecaster import BranchForecaster, build_features, generate_narrative


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_df(n_days: int = 400, base: float = 22_000, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic cash history DataFrame."""
    rng = np.random.default_rng(seed)
    today = date.today()
    dates = [today - timedelta(days=n_days - i) for i in range(n_days)]
    dow_mult = [1.10, 0.92, 0.88, 0.94, 1.38, 0.68, 0.45]
    vals = [
        max(base * dow_mult[d.weekday()] * float(rng.normal(1.0, 0.09)), 500.0)
        for d in dates
    ]
    return pd.DataFrame({"record_date": dates, "cash_dispensed": vals})


# ── Feature tests ─────────────────────────────────────────────────────────────

def test_build_features_shape():
    df = make_df(100)
    feats = build_features(df)
    assert feats.shape == (100, 17)
    assert feats.isnull().sum().sum() == 0


def test_build_features_cyclic_bounds():
    df = make_df(50)
    feats = build_features(df)
    for col in ("dow_sin", "dow_cos", "dom_sin", "dom_cos", "month_sin", "month_cos"):
        assert feats[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


# ── Forecaster tests ──────────────────────────────────────────────────────────

def test_fit_predict_basic():
    df = make_df(400)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    preds = fc.predict_14d()
    assert len(preds) == 14


def test_predictions_non_negative():
    df = make_df(400)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    for p in fc.predict_14d():
        assert p["predicted_dispensed"] >= 0, "Prediction went negative"


def test_confidence_band_ordering():
    df = make_df(400)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    for p in fc.predict_14d():
        assert p["lower_bound"] <= p["predicted_dispensed"] <= p["upper_bound"]


def test_recommended_order_rounded():
    df = make_df(400)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    for p in fc.predict_14d():
        assert p["recommended_order"] % 1_000 == 0, "Recommended order not rounded to $1k"


def test_confidence_score_in_range():
    df = make_df(400)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    for p in fc.predict_14d():
        assert 0.0 <= p["confidence_score"] <= 1.0


def test_mape_reasonable():
    df = make_df(400)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    assert fc.mape < 50.0, f"MAPE suspiciously high: {fc.mape:.1f}%"


def test_predict_without_fit_raises():
    fc = BranchForecaster(branch_id=99)
    with pytest.raises(RuntimeError):
        fc.predict_14d()


def test_anomaly_flags_format():
    df = make_df(300)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    flags = fc.anomaly_flags(df)
    assert isinstance(flags, list)
    assert all(isinstance(f, str) for f in flags)
    # Should be parseable as dates
    for f in flags:
        date.fromisoformat(f)


def test_anomaly_count_reasonable():
    df = make_df(300)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    flags = fc.anomaly_flags(df)
    # contamination=0.04 → expect ~4% flagged, allow generous range
    assert len(flags) <= len(df) * 0.10, "Too many anomalies flagged"


# ── Narrative tests ───────────────────────────────────────────────────────────

def test_narrative_returns_string():
    df = make_df(400)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    preds = fc.predict_14d()
    text = generate_narrative("Test Branch", preds, hist_median=22_000.0, anomaly_count=3)
    assert isinstance(text, str)
    assert len(text) > 20


def test_narrative_mentions_anomaly():
    df = make_df(400)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    preds = fc.predict_14d()
    text = generate_narrative("Test Branch", preds, hist_median=22_000.0, anomaly_count=5)
    assert "anomal" in text.lower()


def test_narrative_no_anomaly_clean():
    df = make_df(400)
    fc = BranchForecaster(branch_id=1)
    fc.fit(df)
    preds = fc.predict_14d()
    text = generate_narrative("Test Branch", preds, hist_median=22_000.0, anomaly_count=0)
    assert "anomal" not in text.lower()
