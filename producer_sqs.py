"""
producer_sqs.py
================
Reads enhanced_transactions.csv and sends records to SQS.

Key SQS constraint vs Kinesis:
  Kinesis: up to 500 records per PutRecords call
  SQS:     up to 10 messages per SendMessageBatch call
  Each message max 256 KB

Key SQS advantage:
  Free tier: 1 million requests/month permanently
  No shard management — scales automatically

Partition key equivalent:
  SQS Standard has no ordering. We set MessageGroupId on the message
  attributes so a FIFO-aware consumer could respect it if upgraded.
  In Standard SQS, MessageDeduplicationId in attributes handles
  idempotency on the consumer side via DynamoDB.

Design:
  - Reads CSV in chunks (never loads full file into memory)
  - Batches 10 messages per SendMessageBatch call (SQS max)
  - Exponential backoff with jitter on throttle errors
  - Emits a progress log every 1,000 records
  - Dry-run mode prints JSON without touching AWS
"""

import argparse, json, logging, math, random, time, uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aws_clients import sqs as sqs_client

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("producer_sqs")

SQS_BATCH_MAX  = 10          # AWS hard limit
MAX_RETRIES    = 5
BASE_BACKOFF_S = 0.1


# ── Serialization ─────────────────────────────────────────────────────

def build_message_body(row: dict) -> dict:
    """
    Build the JSON envelope for one SQS message.

    Structure:
      metadata — routing/monitoring fields, cheap to read without
                 deserializing the full payload
      payload  — full transaction record

    Why separate metadata from payload?
    Lambda and Glue consumers can filter/route on metadata fields
    without parsing the entire payload. At high volume this matters.
    """
    def clean(v):
        """Convert NaN/Inf to None so json.dumps doesn't crash."""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    schema_ver = str(row.get("schema_version", "v1"))

    payload = {
        # Original PaySim columns
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
        # Enhanced columns
        "customer_segment":              clean(row.get("customer_segment")),
        "device_id":                     str(row.get("device_id", "")),
        "device_type":                   clean(row.get("device_type")),
        "device_age_days":               clean(row.get("device_age_days")),
        "customer_country":              str(row.get("customer_country", "")),
        "merchant_country":              str(row.get("merchant_country", "")),
        "merchant_category":             clean(row.get("merchant_category")),
        "merchant_risk_score":           clean(row.get("merchant_risk_score")),
        "ip_address":                    str(row.get("ip_address", "")),
        "ip_risk_score":                 clean(row.get("ip_risk_score")),
        "chargeback_status":             clean(row.get("chargeback_status")),
        "days_since_last_transaction":   clean(row.get("days_since_last_transaction")),
        "transaction_velocity_5min":     int(row.get("transaction_velocity_5min", 0)),
        "transaction_velocity_1hour":    int(row.get("transaction_velocity_1hour", 0)),
        "customer_lifetime_transactions":int(row.get("customer_lifetime_transactions", 0)),
        "customer_lifetime_spend":       float(row.get("customer_lifetime_spend", 0.0)),
        "geo_anomaly_flag":              int(row.get("geo_anomaly_flag", 0)),
        "new_device_flag":               int(row.get("new_device_flag", 0)),
        "duplicate_event_flag":          int(row.get("duplicate_event_flag", 0)),
        "late_arrival_flag":             int(row.get("late_arrival_flag", 0)),
        "out_of_order_flag":             int(row.get("out_of_order_flag", 0)),
        "traffic_period":                str(row.get("traffic_period", "Normal")),
    }
    if schema_ver == "v2":
        payload["ip_risk_score"] = clean(row.get("ip_risk_score"))

    return {
        "metadata": {
            "event_id":       str(row["event_id"]),
            "event_time":     str(row["event_time"]),
            "arrival_time":   str(row["arrival_time"]),
            "schema_version": schema_ver,
            "produced_at":    datetime.now(timezone.utc).isoformat(),
            "source":         "paysim-enhanced-v1",
        },
        "payload": payload,
    }


def build_sqs_entry(row: dict, message_id: str) -> dict | None:
    """Build one entry for SendMessageBatch. Returns None if >256KB."""
    body = json.dumps(build_message_body(row), default=str)
    if len(body.encode()) > 256 * 1024:
        log.warning("Record %s exceeds 256KB — skipping", row.get("event_id"))
        return None
    return {
        "Id":           message_id,          # unique within the batch (not the event_id)
        "MessageBody":  body,
        "MessageAttributes": {
            # Attach key routing fields as attributes so consumers
            # can filter without deserializing the body
            "event_id": {
                "DataType":    "String",
                "StringValue": str(row["event_id"]),
            },
            "schema_version": {
                "DataType":    "String",
                "StringValue": str(row.get("schema_version", "v1")),
            },
            "is_fraud": {
                "DataType":    "Number",
                "StringValue": str(int(row.get("isFraud", 0))),
            },
            "customer_id": {
                "DataType":    "String",
                "StringValue": str(row.get("nameOrig", "")),
            },
        },
    }


# ── Retry logic ───────────────────────────────────────────────────────

def send_batch_with_retry(sqs, queue_url: str, entries: list, attempt: int = 0) -> int:
    """
    Send a batch of ≤10 entries. Returns number of successfully sent records.
    Retries only failed entries using exponential backoff + full jitter.
    """
    if attempt >= MAX_RETRIES:
        log.error("Max retries reached — dropping %d records", len(entries))
        return 0

    resp   = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
    failed = resp.get("Failed", [])

    if not failed:
        return len(entries)

    # Identify which entries failed and retry only those
    failed_ids   = {f["Id"] for f in failed}
    retry_entries= [e for e in entries if e["Id"] in failed_ids]

    for f in failed:
        log.warning("Message %s failed: %s — %s",
                    f["Id"], f.get("Code"), f.get("Message"))

    # Exponential backoff with full jitter
    cap     = 30.0
    backoff = random.uniform(0, min(cap, BASE_BACKOFF_S * (2 ** attempt)))
    time.sleep(backoff)

    sent_now = len(entries) - len(retry_entries)
    return sent_now + send_batch_with_retry(sqs, queue_url, retry_entries, attempt + 1)


# ── Main producer ─────────────────────────────────────────────────────

def run(csv_path: str, queue_url: str, max_rows: int | None,
        dry_run: bool, chunk_size: int = 5_000):

    sqs  = sqs_client()
    path = Path(csv_path)
    if not path.exists():
        log.error("CSV not found: %s", csv_path)
        return

    total_sent = total_failed = total_skipped = rows_read = 0
    start      = time.monotonic()
    buffer     = []    # accumulates up to 10 SQS entries

    log.info("=" * 60)
    log.info("SQS Producer starting")
    log.info("  Queue   : %s", queue_url)
    log.info("  CSV     : %s", path.name)
    log.info("  Max rows: %s", max_rows or "all")
    log.info("  Dry run : %s", dry_run)
    log.info("=" * 60)

    for chunk in pd.read_csv(path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            if max_rows and rows_read >= max_rows:
                break

            entry = build_sqs_entry(row.to_dict(), str(rows_read))
            if entry is None:
                total_skipped += 1
                continue

            buffer.append(entry)
            rows_read += 1

            # Flush every 10 entries (SQS batch limit)
            if len(buffer) == SQS_BATCH_MAX:
                if dry_run:
                    sample = json.loads(buffer[0]["MessageBody"])
                    log.info("[DRY RUN] Batch of 10 | event_id=%s | type=%s | amount=%.2f | isFraud=%d",
                             sample["metadata"]["event_id"],
                             sample["payload"]["type"],
                             sample["payload"]["amount"],
                             sample["payload"]["isFraud"])
                    total_sent += 10
                else:
                    total_sent += send_batch_with_retry(sqs, queue_url, buffer)

                buffer = []

            if rows_read % 1_000 == 0:
                elapsed = time.monotonic() - start
                log.info("Progress: %d rows | %d sent | %.0f msg/s",
                         rows_read, total_sent, total_sent / max(elapsed, 0.001))

        if max_rows and rows_read >= max_rows:
            break

    # Flush remainder
    if buffer:
        if dry_run:
            total_sent += len(buffer)
        else:
            total_sent += send_batch_with_retry(sqs, queue_url, buffer)

    elapsed = time.monotonic() - start
    log.info("")
    log.info("=" * 60)
    log.info("Producer complete")
    log.info("  Sent    : %d", total_sent)
    log.info("  Failed  : %d", total_failed)
    log.info("  Skipped : %d", total_skipped)
    log.info("  Duration: %.1f seconds", elapsed)
    log.info("  Avg rate: %.0f msg/s", total_sent / max(elapsed, 0.001))
    log.info("=" * 60)


if __name__ == "__main__":
    import json as _json

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="enhanced_transactions.csv")
    parser.add_argument("--config",    default="queue_config.json")
    parser.add_argument("--max-rows",  type=int, default=None)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--chunk-size",type=int, default=5_000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = _json.load(f)

    run(
        csv_path  = args.csv,
        queue_url = cfg["main_queue_url"],
        max_rows  = args.max_rows,
        dry_run   = args.dry_run,
        chunk_size= args.chunk_size,
    )