"""
engine.py – Core ML pipeline for order-level Lifetime Value (LTV) prediction.

Pipeline stages
---------------
1. Reconstruct per-order state timeline from raw transactions.
2. Build a monthly observation panel (one row per order × month).
3. Engineer features (tenure, billing cycle, plan type, renewal behaviour, …).
4. Train models with automatic strategy selection:
     • XGBoost discrete-time survival  +  XGBoost revenue regression  (≥ MIN_ML_ORDERS)
     • Cohort-average fallback based on billing cycle                  (< MIN_ML_ORDERS)
5. Forecast 60-month cash-flow vectors + residual tail value per order.
6. Aggregate results.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import (
    VALID_ACTIONS,
    ModelDiagnostics,
    MonthlyForecast,
    OrderLTVResult,
    LTVSummary,
    months_between,
    parse_config,
    parse_date,
    validate_orders_schema,
    validate_transactions_schema,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_ML_ORDERS = 30          # minimum orders required for ML models
MIN_ML_EVENTS = 20          # minimum renewal events required for ML
FORECAST_HORIZON = 60       # months
GRACE_PERIOD_DAYS = 30      # days after expiry before we call it churned
RESIDUAL_DISCOUNT_RATE = 0.0  # monthly discount rate for residual value

# Attempt to import XGBoost; fall back to sklearn if unavailable
try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("XGBoost not available – using scikit-learn fallback models.")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OrderState:
    """Snapshot of an order's configuration at a point in time."""

    plan_type: str = "unknown"
    billing_cycle: int = 12
    expiry_date: Optional[date] = None
    no_of_accounts: int = 1
    timestamp: Optional[date] = None


@dataclass
class OrderTimeline:
    """Full reconstructed history for a single order."""

    order_id: str
    creation_date: date
    country: str
    first_payment_amount: float
    first_payment_discount_perc: float
    domain_name: str
    states: List[OrderState] = field(default_factory=list)  # ordered by time
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    churned: bool = False
    churn_date: Optional[date] = None
    last_known_date: Optional[date] = None  # last transaction or reference date


# ---------------------------------------------------------------------------
# Stage 1 – Timeline reconstruction
# ---------------------------------------------------------------------------

class TimelineBuilder:
    """
    Reconstruct per-order lifecycle from raw order and transaction DataFrames.
    """

    def __init__(self, reference_date: date) -> None:
        self.reference_date = reference_date

    def build(
        self,
        orders_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
    ) -> Dict[str, OrderTimeline]:
        validate_orders_schema(orders_df)
        validate_transactions_schema(transactions_df)

        # Normalise column types
        orders_df = orders_df.copy()
        transactions_df = transactions_df.copy()
        transactions_df["orderid"] = transactions_df["orderid"].astype(str)
        orders_df["order_id"] = orders_df["order_id"].astype(str)

        orders_df["creation_date"] = orders_df["creation_date"].apply(parse_date)
        transactions_df["eventtimestamp"] = transactions_df["eventtimestamp"].apply(
            parse_date
        )
        transactions_df["amount"] = pd.to_numeric(
            transactions_df["amount"], errors="coerce"
        ).fillna(0.0)
        transactions_df["action"] = (
            transactions_df["action"].str.lower().str.strip()
        )

        timelines: Dict[str, OrderTimeline] = {}

        for _, row in orders_df.iterrows():
            oid = str(row["order_id"])
            creation_date = row["creation_date"]
            if creation_date is None:
                logger.warning("Order %s has no creation date – skipping.", oid)
                continue

            tl = OrderTimeline(
                order_id=oid,
                creation_date=creation_date,
                country=str(row.get("country", "unknown")),
                first_payment_amount=float(row.get("first_payment_amount", 0) or 0),
                first_payment_discount_perc=float(
                    row.get("first_payment_discount_perc", 0) or 0
                ),
                domain_name=str(row.get("domain_name", "")),
            )

            # Seed initial state from orders table
            initial_state = OrderState(
                plan_type=str(row.get("creation_plan_type", "unknown")),
                billing_cycle=int(row.get("creation_billing_cycle", 12) or 12),
                no_of_accounts=int(row.get("no_of_accounts", 1) or 1),
                timestamp=creation_date,
            )
            tl.states.append(initial_state)

            # Attach matching transactions
            order_txns = transactions_df[transactions_df["orderid"] == oid].sort_values(
                "eventtimestamp"
            )

            current_state = initial_state
            last_expiry: Optional[date] = None

            for _, txn in order_txns.iterrows():
                action = txn["action"]
                ts = txn["eventtimestamp"]
                if ts is None:
                    continue

                new_cfg = parse_config(txn.get("new_config"))
                new_state = OrderState(
                    plan_type=current_state.plan_type,
                    billing_cycle=current_state.billing_cycle,
                    expiry_date=current_state.expiry_date,
                    no_of_accounts=current_state.no_of_accounts,
                    timestamp=ts,
                )

                if new_cfg is not None:
                    new_state.plan_type = new_cfg[0]
                    new_state.billing_cycle = new_cfg[1]
                    if new_cfg[2] is not None:
                        new_state.expiry_date = new_cfg[2]
                    new_state.no_of_accounts = new_cfg[3]

                if action in ("purchase", "renew"):
                    if new_state.expiry_date is not None:
                        last_expiry = new_state.expiry_date

                tl.transactions.append(
                    {
                        "timestamp": ts,
                        "action": action,
                        "amount": float(txn.get("amount", 0) or 0),
                        "new_state": new_state,
                    }
                )
                tl.states.append(new_state)
                current_state = new_state

            # Determine churn
            final_expiry = last_expiry or current_state.expiry_date
            tl.last_known_date = self.reference_date

            if final_expiry is not None:
                grace_deadline = final_expiry + timedelta(days=GRACE_PERIOD_DAYS)
                if self.reference_date > grace_deadline:
                    # Check if any renewal happened after expiry
                    renewed_after = any(
                        t["action"] == "renew"
                        and t["timestamp"] is not None
                        and t["timestamp"] >= final_expiry
                        for t in tl.transactions
                    )
                    if not renewed_after:
                        tl.churned = True
                        tl.churn_date = final_expiry

            timelines[oid] = tl

        logger.info(
            "Reconstructed %d order timelines (%d churned).",
            len(timelines),
            sum(t.churned for t in timelines.values()),
        )
        return timelines


# ---------------------------------------------------------------------------
# Stage 2 – Monthly panel builder
# ---------------------------------------------------------------------------

class MonthlyPanelBuilder:
    """
    Convert order timelines into a monthly observation panel.

    Each row represents one order-month with:
      - revenue realised in that month
      - churn_event flag (1 on the month of churn, 0 otherwise)
      - snapshot of order features
    """

    def build(
        self,
        timelines: Dict[str, OrderTimeline],
        reference_date: date,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for oid, tl in timelines.items():
            # Determine observation window
            end_date = tl.churn_date if tl.churned else reference_date
            if end_date is None:
                end_date = reference_date

            total_months = max(months_between(tl.creation_date, end_date), 1)

            # Bucket transactions into calendar months
            monthly_revenue: Dict[int, float] = {}
            action_counts: Dict[str, int] = {}
            for txn in tl.transactions:
                if txn["timestamp"] is None:
                    continue
                m_idx = months_between(tl.creation_date, txn["timestamp"])
                if m_idx < 0:
                    continue
                monthly_revenue[m_idx] = monthly_revenue.get(m_idx, 0.0) + txn["amount"]
                action_counts[txn["action"]] = action_counts.get(txn["action"], 0) + 1

            renewal_count = action_counts.get("renew", 0)
            upgrade_count = action_counts.get("upgrade", 0)
            downgrade_count = action_counts.get("downgrade", 0)
            addaccount_count = action_counts.get("addaccount", 0)
            reduceaccount_count = action_counts.get("reduceaccount", 0)

            # Get current billing cycle from last known state
            billing_cycle = tl.states[-1].billing_cycle if tl.states else 12
            plan_type = tl.states[-1].plan_type if tl.states else "unknown"
            accounts = tl.states[-1].no_of_accounts if tl.states else 1

            for m in range(total_months):
                revenue = monthly_revenue.get(m, 0.0)
                is_churn_month = tl.churned and m == total_months - 1

                rows.append(
                    {
                        "order_id": oid,
                        "month_index": m,
                        "revenue": revenue,
                        "churn_event": int(is_churn_month),
                        "tenure_months": m,
                        "billing_cycle": billing_cycle,
                        "plan_type": plan_type,
                        "no_of_accounts": accounts,
                        "country": tl.country,
                        "first_payment_amount": tl.first_payment_amount,
                        "first_payment_discount_perc": tl.first_payment_discount_perc,
                        "renewal_count": renewal_count,
                        "upgrade_count": upgrade_count,
                        "downgrade_count": downgrade_count,
                        "addaccount_count": addaccount_count,
                        "reduceaccount_count": reduceaccount_count,
                        "is_renewal_month": int(m > 0 and billing_cycle > 0 and m % billing_cycle == 0),
                        "censored": int(not tl.churned),
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        logger.info(
            "Built monthly panel: %d rows across %d orders.",
            len(df),
            df["order_id"].nunique(),
        )
        return df


# ---------------------------------------------------------------------------
# Stage 3 – Feature engineering
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = ["plan_type", "country"]
NUMERIC_FEATURES = [
    "tenure_months",
    "billing_cycle",
    "no_of_accounts",
    "first_payment_amount",
    "first_payment_discount_perc",
    "renewal_count",
    "upgrade_count",
    "downgrade_count",
    "addaccount_count",
    "reduceaccount_count",
    "is_renewal_month",
]


class FeatureEngineer:
    """Encode and scale panel features for ML models."""

    def __init__(self) -> None:
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scaler = StandardScaler()
        self._fitted = False
        self.feature_columns: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("unknown"))
            self._label_encoders[col] = le

        all_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
        self.feature_columns = all_cols
        X = df[all_cols].fillna(0).values.astype(float)
        X = self._scaler.fit_transform(X)
        self._fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fit before transform.")
        df = df.copy()
        for col, le in self._label_encoders.items():
            vals = df[col].astype(str).fillna("unknown")
            # Handle unseen categories
            known = set(le.classes_)
            vals = vals.apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(vals)
        all_cols = self.feature_columns
        X = df[all_cols].fillna(0).values.astype(float)
        return self._scaler.transform(X)


# ---------------------------------------------------------------------------
# Stage 4 – Model strategy
# ---------------------------------------------------------------------------

def _make_survival_model(n_samples: int) -> Any:
    if _XGB_AVAILABLE and n_samples >= MIN_ML_ORDERS * 5:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    return LogisticRegression(max_iter=1000, C=0.5, random_state=42)


def _make_revenue_model(n_samples: int) -> Any:
    if _XGB_AVAILABLE and n_samples >= MIN_ML_ORDERS * 5:
        return XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    return Ridge(alpha=1.0)


# ---------------------------------------------------------------------------
# Stage 5 – Cohort fallback
# ---------------------------------------------------------------------------

class CohortFallback:
    """
    Simple cohort statistics used when there is insufficient data for ML.

    Groups orders by billing_cycle and computes:
      - median renewal rate (renewal_count / expected_renewals)
      - median revenue per billing cycle
    """

    def __init__(self) -> None:
        self._cohort_stats: Dict[int, Dict[str, float]] = {}

    def fit(self, timelines: Dict[str, OrderTimeline]) -> None:
        records: List[Dict[str, Any]] = []
        for tl in timelines.values():
            billing_cycle = tl.states[-1].billing_cycle if tl.states else 12
            total_months = max(
                months_between(
                    tl.creation_date,
                    tl.churn_date or date.today(),
                ),
                1,
            )
            expected_renewals = max(total_months // billing_cycle, 1)
            renewal_count = sum(
                1 for t in tl.transactions if t["action"] == "renew"
            )
            total_revenue = sum(t["amount"] for t in tl.transactions)
            records.append(
                {
                    "billing_cycle": billing_cycle,
                    "renewal_rate": min(renewal_count / expected_renewals, 1.0),
                    "revenue_per_cycle": total_revenue / expected_renewals,
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            self._cohort_stats[12] = {"renewal_rate": 0.7, "revenue_per_cycle": 100.0}
            return

        for bc, grp in df.groupby("billing_cycle"):
            self._cohort_stats[int(bc)] = {
                "renewal_rate": float(grp["renewal_rate"].median()),
                "revenue_per_cycle": float(grp["revenue_per_cycle"].median()),
            }

    def get_stats(self, billing_cycle: int) -> Dict[str, float]:
        if billing_cycle in self._cohort_stats:
            return self._cohort_stats[billing_cycle]
        # Fallback: use nearest available billing cycle
        if self._cohort_stats:
            closest = min(
                self._cohort_stats.keys(),
                key=lambda k: abs(k - billing_cycle),
            )
            return self._cohort_stats[closest]
        return {"renewal_rate": 0.7, "revenue_per_cycle": 100.0}


# ---------------------------------------------------------------------------
# Core LTV engine
# ---------------------------------------------------------------------------

class LTVEngine:
    """
    Orchestrates the full pipeline: data ingestion → training → forecasting.
    """

    def __init__(self) -> None:
        self._feature_engineer = FeatureEngineer()
        self._survival_model: Optional[Any] = None
        self._revenue_model: Optional[Any] = None
        self._cohort_fallback = CohortFallback()
        self._use_ml = False
        self._diagnostics: Optional[ModelDiagnostics] = None
        self._timelines: Dict[str, OrderTimeline] = {}
        self._panel: pd.DataFrame = pd.DataFrame()
        self._reference_date: date = date.today()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train(
        self,
        orders_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        reference_date: Optional[date] = None,
    ) -> ModelDiagnostics:
        self._reference_date = reference_date or date.today()

        # Stage 1 – timelines
        builder = TimelineBuilder(self._reference_date)
        self._timelines = builder.build(orders_df, transactions_df)

        # Stage 2 – monthly panel
        panel_builder = MonthlyPanelBuilder()
        self._panel = panel_builder.build(self._timelines, self._reference_date)

        # Stage 3 – cohort fallback (always computed as safety net)
        self._cohort_fallback.fit(self._timelines)

        # Stage 4 – decide whether ML is feasible
        n_orders = len(self._timelines)
        n_renewal_events = int(
            self._panel["churn_event"].sum()
        ) if not self._panel.empty else 0

        if (
            not self._panel.empty
            and n_orders >= MIN_ML_ORDERS
            and n_renewal_events >= MIN_ML_EVENTS
        ):
            self._diagnostics = self._train_ml_models()
        else:
            logger.info(
                "Insufficient data for ML (%d orders, %d events) – "
                "using cohort fallback.",
                n_orders,
                n_renewal_events,
            )
            self._diagnostics = ModelDiagnostics(
                survival_model_type="cohort_fallback",
                revenue_model_type="cohort_fallback",
                n_training_orders=n_orders,
                n_training_observations=len(self._panel),
                fallback_cohort_stats={
                    str(k): v
                    for k, v in self._cohort_fallback._cohort_stats.items()
                },
            )

        return self._diagnostics

    def predict(
        self,
        orders_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        order_ids: Optional[List[str]] = None,
        forecast_horizon: int = FORECAST_HORIZON,
        reference_date: Optional[date] = None,
    ) -> Tuple[List[OrderLTVResult], LTVSummary]:
        ref_date = reference_date or self._reference_date

        # Re-build timelines for the prediction set
        builder = TimelineBuilder(ref_date)
        timelines = builder.build(orders_df, transactions_df)

        if order_ids:
            timelines = {k: v for k, v in timelines.items() if k in order_ids}

        results: List[OrderLTVResult] = []
        for oid, tl in timelines.items():
            result = self._forecast_order(tl, forecast_horizon, ref_date)
            results.append(result)

        summary = self._aggregate(results)
        return results, summary

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _train_ml_models(self) -> ModelDiagnostics:
        panel = self._panel

        # Build feature matrix
        X = self._feature_engineer.fit_transform(panel)
        y_churn = panel["churn_event"].values.astype(int)
        y_revenue = panel["revenue"].values.astype(float)

        n_samples = len(panel)
        survival_model_name = "XGBClassifier" if _XGB_AVAILABLE and n_samples >= MIN_ML_ORDERS * 5 else "LogisticRegression"
        revenue_model_name = "XGBRegressor" if _XGB_AVAILABLE and n_samples >= MIN_ML_ORDERS * 5 else "Ridge"

        # Survival model
        surv_model = _make_survival_model(n_samples)
        surv_model.fit(X, y_churn)
        self._survival_model = surv_model
        self._use_ml = True

        # Revenue model (only on rows where revenue > 0)
        revenue_mask = y_revenue > 0
        rev_model = _make_revenue_model(int(revenue_mask.sum()))
        if revenue_mask.sum() >= 5:
            rev_model.fit(X[revenue_mask], y_revenue[revenue_mask])
        else:
            # Degenerate case: fit on all rows
            rev_model.fit(X, y_revenue)
        self._revenue_model = rev_model

        # Diagnostics
        auc: Optional[float] = None
        rmse: Optional[float] = None
        try:
            if len(np.unique(y_churn)) > 1:
                y_pred_prob = surv_model.predict_proba(X)[:, 1]
                auc = float(roc_auc_score(y_churn, y_pred_prob))
        except Exception:
            pass
        try:
            y_rev_pred = rev_model.predict(X[revenue_mask])
            rmse = float(
                mean_squared_error(y_revenue[revenue_mask], y_rev_pred) ** 0.5
            )
        except Exception:
            pass

        diag = ModelDiagnostics(
            survival_model_type=survival_model_name,
            revenue_model_type=revenue_model_name,
            n_training_orders=self._panel["order_id"].nunique(),
            n_training_observations=len(panel),
            survival_model_auc=auc,
            revenue_model_rmse=rmse,
        )
        logger.info(
            "ML models trained – survival=%s (AUC=%.3f), revenue=%s (RMSE=%.2f)",
            survival_model_name,
            auc or 0.0,
            revenue_model_name,
            rmse or 0.0,
        )
        return diag

    def _forecast_order(
        self,
        tl: OrderTimeline,
        horizon: int,
        ref_date: date,
    ) -> OrderLTVResult:
        billing_cycle = tl.states[-1].billing_cycle if tl.states else 12
        plan_type = tl.states[-1].plan_type if tl.states else "unknown"
        accounts = tl.states[-1].no_of_accounts if tl.states else 1
        current_tenure = months_between(tl.creation_date, ref_date)

        renewal_count = sum(
            1 for t in tl.transactions if t["action"] == "renew"
        )
        upgrade_count = sum(
            1 for t in tl.transactions if t["action"] == "upgrade"
        )
        downgrade_count = sum(
            1 for t in tl.transactions if t["action"] == "downgrade"
        )

        monthly_forecasts: List[MonthlyForecast] = []
        cumulative_survival = 1.0
        total_ltv = 0.0
        model_used = "ml" if self._use_ml else "cohort_fallback"

        for future_month in range(1, horizon + 1):
            tenure_at_month = current_tenure + future_month
            is_renewal_month = int(
                billing_cycle > 0 and tenure_at_month % billing_cycle == 0
            )

            if self._use_ml and self._survival_model and self._revenue_model:
                row = pd.DataFrame(
                    [
                        {
                            "tenure_months": tenure_at_month,
                            "billing_cycle": billing_cycle,
                            "plan_type": plan_type,
                            "no_of_accounts": accounts,
                            "country": tl.country,
                            "first_payment_amount": tl.first_payment_amount,
                            "first_payment_discount_perc": tl.first_payment_discount_perc,
                            "renewal_count": renewal_count,
                            "upgrade_count": upgrade_count,
                            "downgrade_count": downgrade_count,
                            "addaccount_count": 0,
                            "reduceaccount_count": 0,
                            "is_renewal_month": is_renewal_month,
                        }
                    ]
                )
                try:
                    X_row = self._feature_engineer.transform(row)
                    monthly_churn_prob = float(
                        self._survival_model.predict_proba(X_row)[0, 1]
                    )
                    # Only predict revenue on renewal months; otherwise 0
                    if is_renewal_month:
                        rev_raw = float(self._revenue_model.predict(X_row)[0])
                        expected_rev_if_active = max(rev_raw, 0.0)
                    else:
                        expected_rev_if_active = 0.0
                except Exception as exc:
                    logger.debug("ML prediction error for %s month %d: %s", tl.order_id, future_month, exc)
                    monthly_churn_prob, expected_rev_if_active = self._cohort_estimate(
                        billing_cycle, tenure_at_month, is_renewal_month
                    )
                    model_used = "cohort_fallback"
            else:
                monthly_churn_prob, expected_rev_if_active = self._cohort_estimate(
                    billing_cycle, tenure_at_month, is_renewal_month
                )

            # Survival: at renewal months, apply churn probability
            if is_renewal_month:
                cumulative_survival *= 1.0 - monthly_churn_prob

            expected_cashflow = cumulative_survival * expected_rev_if_active
            total_ltv += expected_cashflow

            monthly_forecasts.append(
                MonthlyForecast(
                    month_index=future_month,
                    expected_cashflow=round(expected_cashflow, 4),
                    survival_probability=round(cumulative_survival, 6),
                    expected_revenue_if_active=round(expected_rev_if_active, 4),
                )
            )

        # Residual value: parametric tail beyond horizon
        residual = self._compute_residual(
            cumulative_survival, billing_cycle, tl, plan_type, accounts, current_tenure + horizon
        )
        total_ltv += residual

        # Simple CI: ±20% of LTV based on survival uncertainty
        ltv_std = total_ltv * 0.2
        ci_lower = max(total_ltv - ltv_std, 0.0)
        ci_upper = total_ltv + ltv_std

        return OrderLTVResult(
            order_id=tl.order_id,
            total_ltv=round(total_ltv, 2),
            forecast_60_months=monthly_forecasts,
            residual_value_after_60=round(residual, 2),
            model_used=model_used,
            confidence_interval_lower=round(ci_lower, 2),
            confidence_interval_upper=round(ci_upper, 2),
            features_at_prediction={
                "tenure_months": current_tenure,
                "billing_cycle": billing_cycle,
                "plan_type": plan_type,
                "no_of_accounts": accounts,
                "country": tl.country,
                "renewal_count": renewal_count,
            },
        )

    def _cohort_estimate(
        self,
        billing_cycle: int,
        tenure: int,
        is_renewal_month: int,
    ) -> Tuple[float, float]:
        """Return (monthly_churn_prob, expected_rev_if_active) from cohort stats."""
        stats = self._cohort_fallback.get_stats(billing_cycle)
        renewal_rate = stats["renewal_rate"]
        revenue_per_cycle = stats["revenue_per_cycle"]
        churn_prob_at_renewal = 1.0 - renewal_rate
        expected_rev = revenue_per_cycle if is_renewal_month else 0.0
        return churn_prob_at_renewal, expected_rev

    def _compute_residual(
        self,
        survival_at_horizon: float,
        billing_cycle: int,
        tl: OrderTimeline,
        plan_type: str,
        accounts: int,
        tenure_at_horizon: int,
    ) -> float:
        """
        Estimate residual LTV beyond the forecast horizon using a geometric
        series: assume constant monthly churn hazard and revenue per cycle.
        """
        if survival_at_horizon <= 0:
            return 0.0

        stats = self._cohort_fallback.get_stats(billing_cycle)
        renewal_rate = stats["renewal_rate"]
        revenue_per_cycle = stats["revenue_per_cycle"]

        if renewal_rate <= 0 or billing_cycle <= 0:
            return 0.0

        # Geometric series of future renewal payments (discounted)
        # E[residual] = S_horizon * R * renewal_rate / (1 - renewal_rate)
        # capped at a reasonable multiplier
        if renewal_rate >= 1.0:
            renewal_rate = 0.99
        residual = (
            survival_at_horizon
            * revenue_per_cycle
            * renewal_rate
            / (1.0 - renewal_rate)
        )
        # Apply a soft cap at 10× annual run-rate
        annual_run_rate = (12 / billing_cycle) * revenue_per_cycle
        return min(residual, 10.0 * annual_run_rate)

    @staticmethod
    def _aggregate(results: List[OrderLTVResult]) -> LTVSummary:
        if not results:
            return LTVSummary(
                total_orders=0,
                total_aggregate_ltv=0.0,
                mean_ltv_per_order=0.0,
                median_ltv_per_order=0.0,
                p25_ltv=0.0,
                p75_ltv=0.0,
                model_distribution={},
            )

        ltvs = np.array([r.total_ltv for r in results])
        model_dist: Dict[str, int] = {}
        for r in results:
            model_dist[r.model_used] = model_dist.get(r.model_used, 0) + 1

        return LTVSummary(
            total_orders=len(results),
            total_aggregate_ltv=round(float(ltvs.sum()), 2),
            mean_ltv_per_order=round(float(ltvs.mean()), 2),
            median_ltv_per_order=round(float(np.median(ltvs)), 2),
            p25_ltv=round(float(np.percentile(ltvs, 25)), 2),
            p75_ltv=round(float(np.percentile(ltvs, 75)), 2),
            model_distribution=model_dist,
        )

    @property
    def diagnostics(self) -> Optional[ModelDiagnostics]:
        return self._diagnostics
