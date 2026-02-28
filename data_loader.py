"""
data_loader.py – Abstract data ingestion layer.

Supports three source types:
  1. Direct file bytes  (uploaded via multipart form)
  2. S3 paths           (s3://bucket/key)
  3. Athena queries     (execute → store to S3 → download)
"""

from __future__ import annotations

import io
import logging
import time
from abc import ABC, abstractmethod
from typing import Tuple

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Base class – every subclass returns a (orders, transactions) pair."""

    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and return (orders_df, transactions_df)."""
        ...


# ---------------------------------------------------------------------------
# File upload loader
# ---------------------------------------------------------------------------

class FileLoader(DataLoader):
    """Load from raw CSV bytes (e.g. from an HTTP multipart upload)."""

    def __init__(self, orders_bytes: bytes, transactions_bytes: bytes) -> None:
        self._orders_bytes = orders_bytes
        self._transactions_bytes = transactions_bytes

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading data from in-memory file bytes.")
        orders = pd.read_csv(io.BytesIO(self._orders_bytes))
        transactions = pd.read_csv(io.BytesIO(self._transactions_bytes))
        logger.info(
            "Loaded %d orders and %d transaction rows.",
            len(orders),
            len(transactions),
        )
        return orders, transactions


# ---------------------------------------------------------------------------
# S3 loader
# ---------------------------------------------------------------------------

class S3Loader(DataLoader):
    """Download CSV files directly from S3."""

    def __init__(
        self,
        orders_s3_path: str,
        transactions_s3_path: str,
        region: str = "us-east-1",
    ) -> None:
        self._orders_s3_path = orders_s3_path
        self._transactions_s3_path = transactions_s3_path
        self._region = region

    @staticmethod
    def _split_s3_path(s3_path: str) -> Tuple[str, str]:
        """Return (bucket, key) from 's3://bucket/key'."""
        without_prefix = s3_path.removeprefix("s3://")
        bucket, _, key = without_prefix.partition("/")
        if not key:
            raise ValueError(f"Invalid S3 path: {s3_path!r}")
        return bucket, key

    def _read_s3_csv(self, s3_path: str) -> pd.DataFrame:
        s3 = boto3.client("s3", region_name=self._region)
        bucket, key = self._split_s3_path(s3_path)
        logger.info("Downloading s3://%s/%s", bucket, key)
        response = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(response["Body"])

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        orders = self._read_s3_csv(self._orders_s3_path)
        transactions = self._read_s3_csv(self._transactions_s3_path)
        logger.info(
            "Loaded %d orders and %d transaction rows from S3.",
            len(orders),
            len(transactions),
        )
        return orders, transactions


# ---------------------------------------------------------------------------
# Athena loader
# ---------------------------------------------------------------------------

class AthenaLoader(DataLoader):
    """
    Execute Athena SQL queries, wait for completion, download results from S3.

    Results land in *output_s3_prefix* as Athena-standard CSV files.
    """

    def __init__(
        self,
        orders_query: str,
        transactions_query: str,
        output_s3_prefix: str,
        database: str,
        region: str = "us-east-1",
        poll_interval_secs: int = 5,
        max_wait_secs: int = 300,
    ) -> None:
        self._orders_query = orders_query
        self._transactions_query = transactions_query
        self._output_s3_prefix = output_s3_prefix.rstrip("/")
        self._database = database
        self._region = region
        self._poll_interval = poll_interval_secs
        self._max_wait = max_wait_secs

    def _run_query(self, query: str) -> pd.DataFrame:
        athena = boto3.client("athena", region_name=self._region)
        s3 = boto3.client("s3", region_name=self._region)

        logger.info("Starting Athena query on database '%s'.", self._database)
        resp = athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": self._database},
            ResultConfiguration={"OutputLocation": self._output_s3_prefix},
        )
        exec_id = resp["QueryExecutionId"]
        logger.info("Athena query ID: %s", exec_id)

        elapsed = 0
        while elapsed < self._max_wait:
            status_resp = athena.get_query_execution(QueryExecutionId=exec_id)
            state = status_resp["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                break
            if state in ("FAILED", "CANCELLED"):
                reason = status_resp["QueryExecution"]["Status"].get(
                    "StateChangeReason", "unknown"
                )
                raise RuntimeError(
                    f"Athena query {exec_id} {state}: {reason}"
                )
            logger.debug("Athena state=%s, waiting…", state)
            time.sleep(self._poll_interval)
            elapsed += self._poll_interval
        else:
            raise TimeoutError(
                f"Athena query {exec_id} did not complete within "
                f"{self._max_wait}s."
            )

        # Results file: <output_prefix>/<exec_id>.csv
        without_prefix = self._output_s3_prefix.removeprefix("s3://")
        bucket, _, prefix_key = without_prefix.partition("/")
        if not bucket:
            raise ValueError(
                f"Invalid output_s3_prefix: {self._output_s3_prefix!r}"
            )
        result_key = f"{prefix_key}/{exec_id}.csv" if prefix_key else f"{exec_id}.csv"
        logger.info("Downloading Athena result s3://%s/%s", bucket, result_key)
        body = s3.get_object(Bucket=bucket, Key=result_key)["Body"]
        return pd.read_csv(body)

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        orders = self._run_query(self._orders_query)
        transactions = self._run_query(self._transactions_query)
        logger.info(
            "Loaded %d orders and %d transaction rows via Athena.",
            len(orders),
            len(transactions),
        )
        return orders, transactions
