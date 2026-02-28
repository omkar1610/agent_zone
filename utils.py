"""
utils.py – Shared helpers: schema validation, config parsing, date parsing,
           and Pydantic request/response models.
"""

from __future__ import annotations

import ast
import logging
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required column sets
# ---------------------------------------------------------------------------

ORDERS_REQUIRED_COLS: frozenset[str] = frozenset(
    {
        "order_id",
        "creation_date",
        "creation_plan_type",
        "creation_billing_cycle",
        "no_of_accounts",
        "country",
        "first_payment_amount",
        "first_payment_discount_perc",
        "domain_name",
    }
)

TRANSACTIONS_REQUIRED_COLS: frozenset[str] = frozenset(
    {
        "orderid",
        "eventtimestamp",
        "action",
        "old_config",
        "new_config",
        "amount",
    }
)

VALID_ACTIONS: frozenset[str] = frozenset(
    {"purchase", "renew", "addaccount", "reduceaccount", "upgrade", "downgrade"}
)

VALID_BILLING_CYCLES: frozenset[int] = frozenset({1, 3, 6, 12, 24, 36, 48})


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_orders_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from the orders frame."""
    missing = ORDERS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Orders data is missing columns: {sorted(missing)}")


def validate_transactions_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from the transactions frame."""
    missing = TRANSACTIONS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Transactions data is missing columns: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Config string parser
# ---------------------------------------------------------------------------


def parse_config(config_str: Any) -> Optional[Tuple[str, int, Optional[date], int]]:
    """
    Parse a config string such as ``(gold, 12, 2024-06-01, 5)`` or its
    Python-repr equivalent ``('gold', 12, '2024-06-01', 5)`` into a
    ``(plan_type, billing_cycle, expiry_date, no_of_accounts)`` tuple.

    Returns *None* if the value is missing or cannot be parsed.
    """
    if config_str is None or (
        isinstance(config_str, float) and pd.isna(config_str)
    ):
        return None

    s = str(config_str).strip()
    if not s or s.lower() in ("nan", "none", "null", ""):
        return None

    # 1) Try Python literal eval (works for quoted tuples)
    try:
        result = ast.literal_eval(s)
        if isinstance(result, (tuple, list)) and len(result) >= 4:
            plan = str(result[0]).strip().strip("'\"")
            cycle = int(result[1])
            expiry = _parse_date_safe(result[2])
            accounts = int(result[3])
            return plan, cycle, expiry, accounts
    except (ValueError, SyntaxError, TypeError):
        pass

    # 2) Manual regex: strip parens, split on comma, handle spaces
    s_inner = re.sub(r"^[\(\[]|[\)\]]$", "", s).strip()
    parts = [p.strip().strip("'\"") for p in s_inner.split(",")]
    if len(parts) >= 4:
        try:
            plan = parts[0]
            cycle = int(parts[1])
            expiry = _parse_date_safe(parts[2])
            accounts = int(parts[3])
            return plan, cycle, expiry, accounts
        except (ValueError, IndexError):
            pass

    logger.debug("Could not parse config string: %r", config_str)
    return None


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
)


def _parse_date_safe(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null", "nat"):
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_date(value: Any) -> Optional[date]:
    """Public wrapper – parse various date/datetime formats to ``date``."""
    return _parse_date_safe(value)


def months_between(start: date, end: date) -> int:
    """
    Return the number of complete calendar months between *start* and *end*.
    Result is 0 when end == start, negative when end < start.
    """
    return (end.year - start.year) * 12 + (end.month - start.month)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class S3Source(BaseModel):
    orders_s3_path: str = Field(
        ..., description="S3 path for the orders CSV, e.g. s3://bucket/orders.csv"
    )
    transactions_s3_path: str = Field(
        ..., description="S3 path for the transactions CSV"
    )
    region: str = Field("us-east-1", description="AWS region")

    @field_validator("orders_s3_path", "transactions_s3_path")
    @classmethod
    def must_be_s3_path(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError("Path must start with 's3://'")
        return v


class AthenaSource(BaseModel):
    orders_query: str = Field(..., description="SQL query to fetch orders data")
    transactions_query: str = Field(
        ..., description="SQL query to fetch transactions data"
    )
    output_s3_prefix: str = Field(
        ..., description="S3 prefix for Athena query results"
    )
    database: str = Field(..., description="Athena database name")
    region: str = Field("us-east-1", description="AWS region")

    @field_validator("output_s3_prefix")
    @classmethod
    def must_be_s3_path(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError("output_s3_prefix must start with 's3://'")
        return v


class TrainRequest(BaseModel):
    """Request body for the /train endpoint when using S3 or Athena sources."""

    s3_source: Optional[S3Source] = None
    athena_source: Optional[AthenaSource] = None
    forecast_horizon_months: int = Field(
        60, ge=1, le=240, description="Number of months to forecast"
    )
    reference_date: Optional[str] = Field(
        None,
        description=(
            "Reference date for 'today' in YYYY-MM-DD format. "
            "Defaults to the current date if not supplied."
        ),
    )


class MonthlyForecast(BaseModel):
    month_index: int
    expected_cashflow: float
    survival_probability: float
    expected_revenue_if_active: float


class OrderLTVResult(BaseModel):
    order_id: str
    total_ltv: float
    forecast_60_months: List[MonthlyForecast]
    residual_value_after_60: float
    model_used: str
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    features_at_prediction: Optional[Dict[str, Any]] = None


class LTVSummary(BaseModel):
    total_orders: int
    total_aggregate_ltv: float
    mean_ltv_per_order: float
    median_ltv_per_order: float
    p25_ltv: float
    p75_ltv: float
    model_distribution: Dict[str, int]


class ModelDiagnostics(BaseModel):
    survival_model_type: str
    revenue_model_type: str
    n_training_orders: int
    n_training_observations: int
    survival_model_auc: Optional[float] = None
    revenue_model_rmse: Optional[float] = None
    fallback_cohort_stats: Optional[Dict[str, Any]] = None


class TrainResponse(BaseModel):
    status: str
    message: str
    diagnostics: ModelDiagnostics


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint when using S3 or Athena sources."""

    s3_source: Optional[S3Source] = None
    athena_source: Optional[AthenaSource] = None
    order_ids: Optional[List[str]] = Field(
        None, description="Subset of order IDs to predict. Defaults to all."
    )
    forecast_horizon_months: int = Field(60, ge=1, le=240)
    reference_date: Optional[str] = None


class PredictResponse(BaseModel):
    order_predictions: List[OrderLTVResult]
    summary: LTVSummary
    model_diagnostics: ModelDiagnostics
