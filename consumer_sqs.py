"""
consumer_sqs.py
================
Pulls transactions from SQS, runs three fraud detection checks,
writes flagged transactions to the alerts queue, marks all processed
records in DynamoDB (idempotency), and saves raw JSON to S3 (Bronze).

Three fraud checks (mirrors what Lambda would run in production):
  1. Velocity gate       — transaction_velocity_5min >= 5
  2. Merchant risk gate  — merchant_risk_score >= 0.75
  3. Geo anomaly gate    — geo_anomaly_flag == 1 AND new_device_flag == 1

Why three separate checks instead of one ML score?
  Rule-based gates run in microseconds and catch obvious fraud patterns
  with zero model infrastructure. In production these run BEFORE the
  SageMaker scoring call to avoid paying for inference on obvious cases.
  This is the exact pattern Capital One uses in their fraud pipeline.

Processing loop:
  1. ReceiveMessage (long polling, up to 10 at a time)
  2. For each message:
     a. Check DynamoDB — if event_id exists, skip (already processed)
     b. Run fraud checks
     c. If flagged → send to alerts queue
     d. Save raw JSON to S3 bronze layer
     e. Write event_id to DynamoDB with 24hr TTL
     f. Delete message from main queue (acknowledges successful processing)
  3. Repeat until queue is empty or --max-messages reached
"""

import argparse, json, logging, time
from datetime import datetime, timezone

from aws_clients import sqs as sqs_client, dynamodb as dynamodb_client, s3 as s3_client

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("consumer_sqs")

RECEIVE_BATCH   = 10      # messages per ReceiveMessage call (SQS max)
WAIT_SECONDS    = 20      # long polling window
EMPTY_POLLS_MAX = 3       # stop after this many consecutive empty polls


# ── Fraud detection rules ─────────────────────────────────────────────

def run_fraud_checks(payload: dict) -> list[str]:
    """
    Returns a list of triggered rule names.
    Empty list = transaction looks clean.

    These rules are deliberately simple — the point is demonstrating
    the pipeline pattern, not building a perfect model.
    """
    triggered = []

    # Rule 1: Card testing — many transactions in 5 minutes
    if int(payload.get("transaction_velocity_5min", 0)) >= 5:
        triggered.append("HIGH_VELOCITY_5MIN")

    # Rule 2: High-risk merchant
    risk = payload.get("merchant_risk_score")
    if risk is not None and float(risk) >= 0.75:
        triggered.append("HIGH_RISK_MERCHANT")

    # Rule 3: Impossible travel + new device (account takeover signal)
    if (int(payload.get("geo_anomaly_flag", 0)) == 1 and
            int(payload.get("new_device_flag", 0)) == 1):
        triggered.append("GEO_ANOMALY_NEW_DEVICE")

    # Rule 4: Preserve original PaySim ground truth label
    if int(payload.get("isFraud", 0)) == 1:
        triggered.append("PAYSIM_FRAUD_LABEL")

    return triggered


# ── Idempotency check ─────────────────────────────────────────────────

def already_processed(ddb, table: str, event_id: str) -> bool:
    """
    Check DynamoDB for this event_id.
    Returns True if already processed (duplicate delivery from SQS).
    """
    resp = ddb.get_item(
        TableName=table,
        Key={"event_id": {"S": event_id}},
        ProjectionExpression="event_id",
    )
    return "Item" in resp


def mark_processed(ddb, table: str, event_id: str):
    """
    Write event_id to DynamoDB with a 24-hour TTL.
    After 24 hours DynamoDB auto-deletes this record — table stays small.
    """
    ttl = int(time.time()) + 86_400   # 24 hours from now
    ddb.put_item(
        TableName=table,
        Item={
            "event_id":  {"S": event_id},
            "processed_at": {"S": datetime.now(timezone.utc).isoformat()},
            "ttl":       {"N": str(ttl)},
        },
        # Only write if NOT already there (extra safety net)
        ConditionExpression="attribute_not_exists(event_id)",
    )


# ── S3 Bronze write ───────────────────────────────────────────────────

def save_to_bronze(s3, bucket: str, event_id: str, envelope: dict):
    """
    Save raw JSON message to S3 Bronze layer.
    Path: bronze/transactions/YYYY/MM/DD/<event_id>.json

    Partitioned by date so Glue crawlers and Athena can scan efficiently.
    """
    now  = datetime.now(timezone.utc)
    key  = (f"bronze/transactions/"
            f"{now.year:04d}/{now.month:02d}/{now.day:02d}/"
            f"{event_id}.json")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(envelope, default=str).encode("utf-8"),
        ContentType="application/json",
    )


# ── Main consumer loop ────────────────────────────────────────────────

def run(main_queue_url: str, alert_queue_url: str,
        dedup_table: str, s3_bucket: str,
        max_messages: int | None, save_to_s3: bool):

    sqs = sqs_client()
    ddb = dynamodb_client()
    s3  = s3_client()

    stats = {
        "processed": 0, "skipped_dedup": 0, "flagged": 0,
        "errors": 0,    "empty_polls":   0,
    }
    start = time.monotonic()

    log.info("=" * 60)
    log.info("SQS Consumer starting")
    log.info("  Main queue  : %s", main_queue_url)
    log.info("  Alert queue : %s", alert_queue_url)
    log.info("  DynamoDB    : %s", dedup_table)
    log.info("  S3 bucket   : %s", s3_bucket)
    log.info("  Max messages: %s", max_messages or "unlimited")
    log.info("  Save to S3  : %s", save_to_s3)
    log.info("=" * 60)

    while True:
        if max_messages and stats["processed"] >= max_messages:
            log.info("Reached max_messages limit (%d)", max_messages)
            break

        # ── Receive up to 10 messages ─────────────────────────────
        resp = sqs.receive_message(
            QueueUrl            = main_queue_url,
            MaxNumberOfMessages = RECEIVE_BATCH,
            WaitTimeSeconds     = WAIT_SECONDS,     # long polling
            MessageAttributeNames=["All"],
        )
        messages = resp.get("Messages", [])

        if not messages:
            stats["empty_polls"] += 1
            log.info("Empty poll %d/%d — queue may be drained",
                     stats["empty_polls"], EMPTY_POLLS_MAX)
            if stats["empty_polls"] >= EMPTY_POLLS_MAX:
                log.info("Queue appears empty. Stopping.")
                break
            continue

        stats["empty_polls"] = 0   # reset on any non-empty poll

        # ── Process each message ──────────────────────────────────
        for msg in messages:
            receipt_handle = msg["ReceiptHandle"]
            try:
                envelope = json.loads(msg["Body"])
                metadata = envelope["metadata"]
                payload  = envelope["payload"]
                event_id = metadata["event_id"]

                # Step 1: Idempotency check
                if already_processed(ddb, dedup_table, event_id):
                    log.debug("Duplicate — skipping %s", event_id)
                    stats["skipped_dedup"] += 1
                    # Still delete from queue — no point reprocessing
                    sqs.delete_message(
                        QueueUrl=main_queue_url,
                        ReceiptHandle=receipt_handle,
                    )
                    continue

                # Step 2: Fraud checks
                triggered_rules = run_fraud_checks(payload)
                is_flagged      = len(triggered_rules) > 0

                # Step 3: Send to alerts queue if flagged
                if is_flagged:
                    alert_body = {
                        "event_id":      event_id,
                        "event_time":    metadata["event_time"],
                        "customer_id":   payload["nameOrig"],
                        "amount":        payload["amount"],
                        "type":          payload["type"],
                        "triggered_rules":triggered_rules,
                        "merchant_risk": payload.get("merchant_risk_score"),
                        "geo_anomaly":   payload.get("geo_anomaly_flag"),
                        "velocity_5min": payload.get("transaction_velocity_5min"),
                        "isFraud_label": payload.get("isFraud"),
                        "alerted_at":    datetime.now(timezone.utc).isoformat(),
                    }
                    sqs.send_message(
                        QueueUrl   =alert_queue_url,
                        MessageBody=json.dumps(alert_body),
                    )
                    stats["flagged"] += 1
                    log.info("FLAGGED  event_id=%-36s | rules=%s | amount=%10.2f | isFraud=%d",
                             event_id, triggered_rules,
                             payload["amount"], payload.get("isFraud", 0))

                # Step 4: Save raw JSON to S3 Bronze
                if save_to_s3:
                    save_to_bronze(s3, s3_bucket, event_id, envelope)

                # Step 5: Mark processed in DynamoDB
                try:
                    mark_processed(ddb, dedup_table, event_id)
                except ddb.exceptions.ConditionalCheckFailedException:
                    # Race condition: another consumer processed it first
                    log.debug("Conditional write failed for %s — race condition, safe to ignore", event_id)

                # Step 6: Delete from main queue (acknowledge)
                sqs.delete_message(
                    QueueUrl=main_queue_url,
                    ReceiptHandle=receipt_handle,
                )
                stats["processed"] += 1

                if stats["processed"] % 500 == 0:
                    elapsed = time.monotonic() - start
                    log.info("Progress: %d processed | %d flagged | %.0f msg/s",
                             stats["processed"], stats["flagged"],
                             stats["processed"] / max(elapsed, 0.001))

            except Exception as e:
                log.error("Error processing message: %s", e, exc_info=True)
                stats["errors"] += 1
                # Do NOT delete the message — let SQS retry it
                # After maxReceiveCount retries, SQS moves it to DLQ

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.monotonic() - start
    log.info("")
    log.info("=" * 60)
    log.info("Consumer complete")
    log.info("  Processed   : %d", stats["processed"])
    log.info("  Flagged     : %d (%.1f%%)",
             stats["flagged"],
             stats["flagged"] / max(stats["processed"], 1) * 100)
    log.info("  Dedup skips : %d", stats["skipped_dedup"])
    log.info("  Errors      : %d", stats["errors"])
    log.info("  Duration    : %.1f seconds", elapsed)
    log.info("  Avg rate    : %.0f msg/s",
             stats["processed"] / max(elapsed, 0.001))
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="queue_config.json")
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--no-s3",        action="store_true",
                        help="Skip S3 writes (faster for testing)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    run(
        main_queue_url  = cfg["main_queue_url"],
        alert_queue_url = cfg["alert_queue_url"],
        dedup_table     = cfg["dedup_table"],
        s3_bucket       = cfg["s3_bucket"],
        max_messages    = args.max_messages,
        save_to_s3      = not args.no_s3,
    )