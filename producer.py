"""
kinesis_producer/producer.py
=============================
Streams enhanced PaySim transactions into Amazon Kinesis Data Streams.

Design decisions that mirror production systems:
  - PutRecords batching (max 500 records / 5 MB per call) for throughput
  - Exponential backoff with jitter on throttling (ProvisionedThroughputExceededException)
  - Partition key = nameOrig (customer ID) so all events for one customer
    land on the same shard — preserving ordering per customer
  - Schema-version-aware serialization: v2 records include ip_risk_score
  - Configurable TPS rate limiter so you don't over-drive a 1-shard stream
  - Structured JSON envelope with metadata fields Kinesis consumers expect
  - Dry-run mode for local testing without AWS credentials
  - CloudWatch custom metrics emitted per batch (throughput, error rate)

Usage:
    # Dry run (no AWS needed) — prints JSON to stdout
    python producer.py --dry-run

    # Real Kinesis stream
    python producer.py \
        --stream-name fraud-transactions-dev \
        --region us-east-1 \
        --tps 100 \
        --batch-size 250
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# ─────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("kinesis_producer")


# ─────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────
KINESIS_MAX_BATCH      = 500          # AWS hard limit: 500 records per PutRecords
KINESIS_MAX_BATCH_BYTES= 5 * 1024 * 1024  # AWS hard limit: 5 MB per PutRecords call
MAX_RECORD_BYTES       = 1 * 1024 * 1024  # AWS hard limit: 1 MB per individual record
MAX_RETRIES            = 5
BASE_BACKOFF_MS        = 100          # ms — doubles on each retry


# ─────────────────────────────────────────────────────
# RECORD SERIALIZATION
# ─────────────────────────────────────────────────────

def serialize_record(row: dict, producer_id: str) -> dict:
    """
    Build the JSON event envelope that goes into Kinesis.

    Envelope design rationale:
    - metadata block: consumed by Lambda/Glue to route, version-gate, and monitor
    - payload block: the actual transaction fields
    - Separating them means consumers can cheaply check metadata without
      deserializing the full payload — important at high TPS

    Partition key = nameOrig (customer ID).
    Why: Kinesis guarantees ordering within a shard. By routing all events
    for the same customer to the same shard, the downstream Lambda/Glue
    velocity computation sees events in order for each customer.
    Tradeoff: if one customer generates extreme volume (a bot), that shard
    gets hot. At 10x scale you'd add a suffix (nameOrig[-2:]) to spread load.
    """
    schema_ver = str(row.get("schema_version", "v1"))

    # v2-only fields (ip_risk_score added in schema version 2)
    v2_extras = {}
    if schema_ver == "v2":
        ip_risk = row.get("ip_risk_score")
        v2_extras["ip_risk_score"] = float(ip_risk) if ip_risk == ip_risk else None  # NaN → None

    # Build payload — coerce NaN to None for JSON serialization
    def clean(v):
        if v != v:          # NaN check
            return None
        if isinstance(v, float) and math.isinf(v):
            return None
        return v

    payload = {
        # ── Original PaySim fields ──────────────────────────────────
        "step":              int(row["step"]),
        "type":              str(row["type"]),
        "amount":            float(row["amount"]),
        "nameOrig":          str(row["nameOrig"]),
        "oldbalanceOrg":     float(row["oldbalanceOrg"]),
        "newbalanceOrig":    float(row["newbalanceOrig"]),
        "nameDest":          str(row["nameDest"]),
        "oldbalanceDest":    float(row["oldbalanceDest"]),
        "newbalanceDest":    float(row["newbalanceDest"]),
        "isFraud":           int(row["isFraud"]),
        "isFlaggedFraud":    int(row["isFlaggedFraud"]),

        # ── Enhanced fields ─────────────────────────────────────────
        "customer_segment":              clean(row.get("customer_segment")),
        "device_id":                     str(row.get("device_id", "")),
        "device_type":                   clean(row.get("device_type")),
        "device_age_days":               clean(row.get("device_age_days")),
        "customer_country":              str(row.get("customer_country", "")),
        "merchant_country":              str(row.get("merchant_country", "")),
        "merchant_category":             clean(row.get("merchant_category")),
        "merchant_risk_score":           clean(row.get("merchant_risk_score")),
        "ip_address":                    str(row.get("ip_address", "")),
        "chargeback_status":             clean(row.get("chargeback_status")),
        "chargeback_delay_days":         clean(row.get("chargeback_delay_days")),
        "days_since_last_transaction":   clean(row.get("days_since_last_transaction")),
        "transaction_velocity_5min":     int(row.get("transaction_velocity_5min", 0)),
        "transaction_velocity_1hour":    int(row.get("transaction_velocity_1hour", 0)),
        "customer_lifetime_transactions":int(row.get("customer_lifetime_transactions", 0)),
        "customer_lifetime_spend":       float(row.get("customer_lifetime_spend", 0.0)),

        # ── Quality / operational flags ─────────────────────────────
        "geo_anomaly_flag":     int(row.get("geo_anomaly_flag", 0)),
        "new_device_flag":      int(row.get("new_device_flag", 0)),
        "duplicate_event_flag": int(row.get("duplicate_event_flag", 0)),
        "late_arrival_flag":    int(row.get("late_arrival_flag", 0)),
        "out_of_order_flag":    int(row.get("out_of_order_flag", 0)),
        "traffic_period":       str(row.get("traffic_period", "Normal")),

        # v2-only extras merged in
        **v2_extras,
    }

    envelope = {
        "metadata": {
            "event_id":        str(row["event_id"]),
            "event_time":      str(row["event_time"]),
            "arrival_time":    str(row["arrival_time"]),
            "schema_version":  schema_ver,
            "producer_id":     producer_id,
            "produced_at":     datetime.now(timezone.utc).isoformat(),
            "source":          "paysim-enhanced-v1",
        },
        "payload": payload,
    }

    return envelope


# ─────────────────────────────────────────────────────
# BATCH BUILDER
# ─────────────────────────────────────────────────────

def build_kinesis_entry(envelope: dict, partition_key: str) -> Optional[dict]:
    """
    Encode the envelope as UTF-8 JSON and return a Kinesis PutRecords entry.
    Returns None if the record exceeds the 1 MB per-record limit (log and skip).
    """
    data = json.dumps(envelope, default=str).encode("utf-8")
    if len(data) > MAX_RECORD_BYTES:
        log.warning("Record %s exceeds 1 MB (%d bytes) — skipping",
                    envelope["metadata"]["event_id"], len(data))
        return None
    return {
        "Data":         data,
        "PartitionKey": partition_key,
    }


def chunk_into_batches(entries: list) -> list:
    """
    Split a list of Kinesis entries into batches that each satisfy:
      - ≤ 500 records
      - ≤ 5 MB total

    Why: PutRecords rejects the entire call if either limit is exceeded.
    This function ensures every batch is safe to submit.
    """
    batches = []
    current_batch = []
    current_bytes = 0

    for entry in entries:
        record_bytes = len(entry["Data"]) + len(entry["PartitionKey"].encode())
        if (len(current_batch) >= KINESIS_MAX_BATCH or
                current_bytes + record_bytes > KINESIS_MAX_BATCH_BYTES):
            batches.append(current_batch)
            current_batch = []
            current_bytes = 0
        current_batch.append(entry)
        current_bytes += record_bytes

    if current_batch:
        batches.append(current_batch)
    return batches


# ─────────────────────────────────────────────────────
# RETRY WITH EXPONENTIAL BACKOFF + JITTER
# ─────────────────────────────────────────────────────

def put_records_with_retry(
    kinesis_client,
    stream_name: str,
    entries: list,
    attempt: int = 0,
) -> dict:
    """
    Submit a batch to Kinesis. On partial failure (some records in the batch
    were throttled), extract the failed records and retry them with exponential
    backoff + jitter.

    Why jitter? Without it, all producers retry at the same time after a
    throttle, creating a thundering herd that immediately throttles again.
    Adding random jitter spreads retries across time.

    Kinesis PutRecords returns HTTP 200 even on partial failure — you must
    inspect the FailedRecordCount field.
    """
    if attempt >= MAX_RETRIES:
        log.error("Max retries (%d) reached for batch of %d records — dropping",
                  MAX_RETRIES, len(entries))
        return {"FailedRecordCount": len(entries), "Records": []}

    try:
        response = kinesis_client.put_records(
            StreamName=stream_name,
            Records=entries,
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("ProvisionedThroughputExceededException", "ServiceUnavailable",
                    "InternalFailure", "ThrottlingException"):
            _backoff(attempt)
            return put_records_with_retry(kinesis_client, stream_name, entries, attempt + 1)
        raise  # non-retryable error — bubble up

    failed_count = response.get("FailedRecordCount", 0)
    if failed_count == 0:
        return response

    # Partial failure: identify and retry only failed records
    failed_entries = []
    for i, rec_result in enumerate(response["Records"]):
        if "ErrorCode" in rec_result:
            failed_entries.append(entries[i])
            log.debug("Record failed: %s — %s",
                      rec_result.get("ErrorCode"), rec_result.get("ErrorMessage"))

    log.warning("Batch partial failure: %d/%d records failed, retrying (attempt %d)",
                failed_count, len(entries), attempt + 1)
    _backoff(attempt)
    return put_records_with_retry(kinesis_client, stream_name, failed_entries, attempt + 1)


def _backoff(attempt: int):
    """Exponential backoff with full jitter (AWS recommended pattern)."""
    cap = 30_000   # max 30 seconds
    base = BASE_BACKOFF_MS * (2 ** attempt)
    sleep_ms = random.uniform(0, min(cap, base))
    time.sleep(sleep_ms / 1000)


# ─────────────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────────────

class TokenBucket:
    """
    Simple token bucket rate limiter.

    Why rate-limit? A Kinesis shard supports 1 MB/s and 1,000 records/s.
    Without a rate limiter, the producer would burst at CSV read speed
    (~50,000 records/s), immediately exhausting shard capacity and triggering
    constant throttling.

    At 1 shard: keep TPS ≤ 900 (leave 10% headroom for retries).
    At N shards: you can set TPS ≤ 900 × N.
    """
    def __init__(self, rate: float):
        self.rate = rate          # tokens per second
        self.tokens = rate
        self.last_refill = time.monotonic()

    def consume(self, n: int = 1):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens >= n:
            self.tokens -= n
        else:
            deficit = n - self.tokens
            sleep_for = deficit / self.rate
            time.sleep(sleep_for)
            self.tokens = 0


# ─────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────

class ProducerMetrics:
    """
    Tracks per-run statistics. Emits to CloudWatch if a client is provided;
    always prints a summary at the end.
    """
    def __init__(self, cw_client=None, namespace="FraudPlatform/Producer"):
        self.cw = cw_client
        self.namespace = namespace
        self.sent       = 0
        self.failed     = 0
        self.skipped    = 0
        self.batches    = 0
        self.start_time = time.monotonic()

    def record_batch(self, sent: int, failed: int):
        self.sent    += sent
        self.failed  += failed
        self.batches += 1

        if self.cw and self.batches % 10 == 0:
            self._emit_to_cloudwatch()

    def record_skip(self):
        self.skipped += 1

    def _emit_to_cloudwatch(self):
        """Emit custom metrics so CloudWatch dashboards and alarms work."""
        try:
            self.cw.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {"MetricName": "RecordsSent",
                     "Value": self.sent, "Unit": "Count"},
                    {"MetricName": "RecordsFailed",
                     "Value": self.failed, "Unit": "Count"},
                    {"MetricName": "ProducerThroughput",
                     "Value": self.tps(), "Unit": "Count/Second"},
                ],
            )
        except Exception as e:
            log.debug("CloudWatch emit failed (non-fatal): %s", e)

    def tps(self) -> float:
        elapsed = max(time.monotonic() - self.start_time, 0.001)
        return self.sent / elapsed

    def summary(self):
        elapsed = time.monotonic() - self.start_time
        log.info("═══════════════════════════════════════")
        log.info("Producer run complete")
        log.info("  Records sent:    %d", self.sent)
        log.info("  Records failed:  %d", self.failed)
        log.info("  Records skipped: %d", self.skipped)
        log.info("  Batches:         %d", self.batches)
        log.info("  Duration:        %.1f seconds", elapsed)
        log.info("  Avg TPS:         %.1f", self.tps())
        log.info("═══════════════════════════════════════")


# ─────────────────────────────────────────────────────
# MAIN PRODUCER
# ─────────────────────────────────────────────────────

def run_producer(
    csv_path: str,
    stream_name: str,
    region: str,
    tps: int,
    batch_size: int,
    dry_run: bool,
    chunk_size: int = 10_000,
    max_rows: Optional[int] = None,
    producer_id: str = "producer-01",
):
    """
    Main entry point.

    Reads enhanced_transactions.csv in chunks to avoid loading the full
    500K+ rows into memory, serializes each row into a Kinesis envelope,
    and publishes in PutRecords batches.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        log.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    # ── AWS clients ─────────────────────────────────────────────────
    kinesis = None
    cw      = None
    if not dry_run:
        session = boto3.Session(region_name=region)
        kinesis = session.client("kinesis")
        cw      = session.client("cloudwatch")
        log.info("Connected to Kinesis stream '%s' in %s", stream_name, region)
    else:
        log.info("DRY RUN mode — no records will be sent to AWS")

    metrics     = ProducerMetrics(cw_client=cw)
    rate_limiter= TokenBucket(rate=float(tps))
    rows_read   = 0

    log.info("Streaming %s → Kinesis (TPS=%d, batch=%d)", csv_path.name, tps, batch_size)

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if max_rows and rows_read >= max_rows:
            break

        # Accumulate Kinesis entries from this CSV chunk
        entries_buffer = []

        for _, row in chunk.iterrows():
            if max_rows and rows_read >= max_rows:
                break

            row_dict = row.to_dict()
            envelope = serialize_record(row_dict, producer_id)
            partition_key = str(row_dict.get("nameOrig", "unknown"))

            entry = build_kinesis_entry(envelope, partition_key)
            if entry is None:
                metrics.record_skip()
                continue

            entries_buffer.append(entry)
            rows_read += 1

            # Flush when we've accumulated batch_size entries
            if len(entries_buffer) >= batch_size:
                _flush_batch(entries_buffer, kinesis, stream_name,
                             dry_run, rate_limiter, metrics, batch_size)
                entries_buffer = []

        # Flush any remaining entries from this chunk
        if entries_buffer:
            _flush_batch(entries_buffer, kinesis, stream_name,
                         dry_run, rate_limiter, metrics, batch_size)

    metrics.summary()


def _flush_batch(
    entries: list,
    kinesis,
    stream_name: str,
    dry_run: bool,
    rate_limiter: TokenBucket,
    metrics: ProducerMetrics,
    batch_size: int,
):
    """
    Split entries into safe Kinesis batches and submit (or print in dry-run).
    Rate-limits by consuming tokens proportional to the batch size.
    """
    batches = chunk_into_batches(entries)
    for batch in batches:
        rate_limiter.consume(len(batch))

        if dry_run:
            # Print a sample record for verification
            sample = json.loads(batch[0]["Data"])
            log.info("[DRY RUN] Batch of %d | sample event_id=%s | type=%s | amount=%.2f | isFraud=%d",
                     len(batch),
                     sample["metadata"]["event_id"],
                     sample["payload"]["type"],
                     sample["payload"]["amount"],
                     sample["payload"]["isFraud"])
            metrics.record_batch(sent=len(batch), failed=0)
        else:
            response = put_records_with_retry(kinesis, stream_name, batch)
            failed   = response.get("FailedRecordCount", 0)
            sent     = len(batch) - failed
            metrics.record_batch(sent=sent, failed=failed)

            if sent > 0 and metrics.batches % 20 == 0:
                log.info("Sent %d records | cumulative: %d sent, %d failed | TPS: %.1f",
                         sent, metrics.sent, metrics.failed, metrics.tps())


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream enhanced PaySim transactions to Amazon Kinesis"
    )
    parser.add_argument("--csv",         default="enhanced_transactions.csv",
                        help="Path to enhanced_transactions.csv")
    parser.add_argument("--stream-name", default="fraud-transactions-dev",
                        help="Kinesis stream name")
    parser.add_argument("--region",      default="us-east-1")
    parser.add_argument("--tps",         type=int, default=100,
                        help="Target records per second (max 900 for 1 shard)")
    parser.add_argument("--batch-size",  type=int, default=250,
                        help="Records per PutRecords call (max 500)")
    parser.add_argument("--chunk-size",  type=int, default=10_000,
                        help="CSV rows to read per pandas chunk")
    parser.add_argument("--max-rows",    type=int, default=None,
                        help="Stop after N rows (for testing)")
    parser.add_argument("--producer-id", default="producer-01",
                        help="Producer identifier emitted in record metadata")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print records without sending to AWS")
    args = parser.parse_args()

    run_producer(
        csv_path=args.csv,
        stream_name=args.stream_name,
        region=args.region,
        tps=args.tps,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        producer_id=args.producer_id,
    )
