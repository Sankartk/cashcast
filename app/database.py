"""Database models and session management for CashCast."""

from __future__ import annotations

import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Date,
    Boolean,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cashcast.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Branch(Base):
    __tablename__ = "branches"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    city = Column(String(50), nullable=False)
    state = Column(String(2), nullable=False)
    branch_type = Column(String(20), nullable=False)  # URBAN | SUBURBAN | RURAL
    avg_daily_traffic = Column(Integer)
    vault_capacity = Column(Float, nullable=False)


class DailyCashRecord(Base):
    __tablename__ = "daily_cash_records"

    id = Column(Integer, primary_key=True, index=True)
    branch_id = Column(Integer, nullable=False, index=True)
    record_date = Column(Date, nullable=False)
    cash_dispensed = Column(Float, nullable=False)
    cash_deposited = Column(Float, nullable=False)
    net_outflow = Column(Float, nullable=False)
    opening_vault = Column(Float, nullable=False)
    closing_vault = Column(Float, nullable=False)
    transaction_count = Column(Integer, nullable=False)
    is_anomaly = Column(Boolean, default=False)

    __table_args__ = (UniqueConstraint("branch_id", "record_date", name="uq_branch_date"),)


class ForecastResult(Base):
    __tablename__ = "forecast_results"

    id = Column(Integer, primary_key=True, index=True)
    branch_id = Column(Integer, nullable=False, index=True)
    forecast_date = Column(Date, nullable=False)
    generated_at = Column(DateTime, nullable=False)
    predicted_dispensed = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=False)
    upper_bound = Column(Float, nullable=False)
    recommended_order = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    narrative = Column(String(600))

    __table_args__ = (UniqueConstraint("branch_id", "forecast_date", name="uq_branch_fc_date"),)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
