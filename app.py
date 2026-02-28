"""
app.py – FastAPI application for the LTV prediction service.

Endpoints
---------
POST /train
    Train (or re-train) the LTV models on the provided dataset.
    Accepts multipart file upload  OR  JSON body (S3 / Athena source).

POST /predict
    Predict order-level LTV.
    Accepts multipart file upload  OR  JSON body (S3 / Athena source).

GET  /ltv-summary
    Return aggregate LTV statistics from the most recent prediction run.
"""

from __future__ import annotations

import io
import logging
import sys
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, AsyncGenerator, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from data_loader import AthenaLoader, FileLoader, S3Loader
from engine import LTVEngine
from utils import (
    AthenaSource,
    LTVSummary,
    ModelDiagnostics,
    OrderLTVResult,
    PredictRequest,
    PredictResponse,
    S3Source,
    TrainRequest,
    TrainResponse,
    parse_date,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_engine: LTVEngine = LTVEngine()
_last_predictions: List[OrderLTVResult] = []
_last_summary: Optional[LTVSummary] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("LTV Prediction Service starting up.")
    yield
    logger.info("LTV Prediction Service shut down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LTV Prediction Service",
    description=(
        "Production-ready order-level Lifetime Value forecasting. "
        "Predicts 60-month cash-flow vectors and aggregate LTV using a "
        "hybrid survival + revenue model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data_from_files(
    orders_file: UploadFile,
    transactions_file: UploadFile,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        orders_bytes = orders_file.file.read()
        transactions_bytes = transactions_file.file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded files: {exc}",
        ) from exc
    loader = FileLoader(orders_bytes, transactions_bytes)
    try:
        return loader.load()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse uploaded CSVs: {exc}",
        ) from exc


def _load_data_from_s3(source: S3Source) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = S3Loader(
        orders_s3_path=source.orders_s3_path,
        transactions_s3_path=source.transactions_s3_path,
        region=source.region,
    )
    try:
        return loader.load()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to load data from S3: {exc}",
        ) from exc


def _load_data_from_athena(
    source: AthenaSource,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = AthenaLoader(
        orders_query=source.orders_query,
        transactions_query=source.transactions_query,
        output_s3_prefix=source.output_s3_prefix,
        database=source.database,
        region=source.region,
    )
    try:
        return loader.load()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to load data via Athena: {exc}",
        ) from exc


def _resolve_reference_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    parsed = parse_date(raw)
    if parsed is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid reference_date format: {raw!r}. Use YYYY-MM-DD.",
        )
    return parsed


# ---------------------------------------------------------------------------
# /train  (file upload variant)
# ---------------------------------------------------------------------------

@app.post(
    "/train/upload",
    response_model=TrainResponse,
    summary="Train models from uploaded CSV files",
    tags=["Training"],
)
async def train_from_upload(
    orders_file: UploadFile = File(..., description="Orders CSV file"),
    transactions_file: UploadFile = File(
        ..., description="Transaction events CSV file"
    ),
    forecast_horizon_months: int = Form(60),
    reference_date: Optional[str] = Form(None),
) -> TrainResponse:
    orders_df, transactions_df = _load_data_from_files(orders_file, transactions_file)
    ref_date = _resolve_reference_date(reference_date)
    return _do_train(orders_df, transactions_df, ref_date)


# ---------------------------------------------------------------------------
# /train  (JSON body: S3 or Athena)
# ---------------------------------------------------------------------------

@app.post(
    "/train",
    response_model=TrainResponse,
    summary="Train models using S3 or Athena data sources",
    tags=["Training"],
)
async def train(body: TrainRequest) -> TrainResponse:
    if body.s3_source:
        orders_df, transactions_df = _load_data_from_s3(body.s3_source)
    elif body.athena_source:
        orders_df, transactions_df = _load_data_from_athena(body.athena_source)
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Provide either 's3_source' or 'athena_source'. "
                "For file uploads use POST /train/upload."
            ),
        )
    ref_date = _resolve_reference_date(body.reference_date)
    return _do_train(orders_df, transactions_df, ref_date)


def _do_train(
    orders_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    reference_date: Optional[date],
) -> TrainResponse:
    global _engine
    _engine = LTVEngine()
    try:
        diagnostics = _engine.train(orders_df, transactions_df, reference_date)
    except Exception as exc:
        logger.exception("Training failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training error: {exc}",
        ) from exc

    return TrainResponse(
        status="success",
        message=(
            f"Models trained on {diagnostics.n_training_orders} orders "
            f"({diagnostics.n_training_observations} monthly observations). "
            f"Strategy: {diagnostics.survival_model_type}."
        ),
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# /predict  (file upload variant)
# ---------------------------------------------------------------------------

@app.post(
    "/predict/upload",
    response_model=PredictResponse,
    summary="Predict LTV from uploaded CSV files",
    tags=["Prediction"],
)
async def predict_from_upload(
    orders_file: UploadFile = File(...),
    transactions_file: UploadFile = File(...),
    order_ids: Optional[str] = Form(
        None, description="Comma-separated order IDs to predict (leave blank for all)"
    ),
    forecast_horizon_months: int = Form(60),
    reference_date: Optional[str] = Form(None),
) -> PredictResponse:
    orders_df, transactions_df = _load_data_from_files(orders_file, transactions_file)
    ids = [x.strip() for x in order_ids.split(",")] if order_ids else None
    ref_date = _resolve_reference_date(reference_date)
    return _do_predict(orders_df, transactions_df, ids, forecast_horizon_months, ref_date)


# ---------------------------------------------------------------------------
# /predict  (JSON body)
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict LTV using S3 or Athena data sources",
    tags=["Prediction"],
)
async def predict(body: PredictRequest) -> PredictResponse:
    if body.s3_source:
        orders_df, transactions_df = _load_data_from_s3(body.s3_source)
    elif body.athena_source:
        orders_df, transactions_df = _load_data_from_athena(body.athena_source)
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Provide either 's3_source' or 'athena_source'. "
                "For file uploads use POST /predict/upload."
            ),
        )
    ref_date = _resolve_reference_date(body.reference_date)
    return _do_predict(
        orders_df,
        transactions_df,
        body.order_ids,
        body.forecast_horizon_months,
        ref_date,
    )


def _do_predict(
    orders_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    order_ids: Optional[List[str]],
    forecast_horizon: int,
    reference_date: Optional[date],
) -> PredictResponse:
    global _last_predictions, _last_summary

    if _engine.diagnostics is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Models have not been trained yet. "
                "Call POST /train (or /train/upload) first."
            ),
        )
    try:
        predictions, summary = _engine.predict(
            orders_df,
            transactions_df,
            order_ids=order_ids,
            forecast_horizon=forecast_horizon,
            reference_date=reference_date,
        )
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        ) from exc

    _last_predictions = predictions
    _last_summary = summary

    return PredictResponse(
        order_predictions=predictions,
        summary=summary,
        model_diagnostics=_engine.diagnostics,
    )


# ---------------------------------------------------------------------------
# /ltv-summary
# ---------------------------------------------------------------------------

@app.get(
    "/ltv-summary",
    response_model=LTVSummary,
    summary="Aggregate LTV statistics from the last prediction run",
    tags=["Summary"],
)
async def ltv_summary() -> LTVSummary:
    if _last_summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No predictions have been made yet. Call POST /predict first.",
        )
    return _last_summary


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_trained": _engine.diagnostics is not None,
        "last_prediction_count": len(_last_predictions),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
