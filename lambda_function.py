"""
lambda_function.py
===================
AWS Lambda function triggered by SQS.

Processes each financial transaction in real-time:
  1. Deserialize SQS message envelope
  2. Idempotency check (fraud-dedup-dev DynamoDB table)
  3. Compute real-time velocity features from DynamoDB feature store
  4. Run 4 fraud detection rules
  5. Write enriched record to DynamoDB fraud-events table
  6. Write raw JSON to S3 Bronze layer
  7. If flagged → publish SNS alert

Why Lambda + SQS trigger vs consumer_sqs.py?
  consumer_sqs.py runs on YOUR machine — it stops when you close
  the terminal. Lambda runs in AWS, scales automatically with queue
  depth, retries on failure, and costs $0 when idle.
  This is the production pattern used at every company running
  event-driven data pipelines on AWS.

Environment variables (set when deploying):
  S3_BUCKET        fraud-platform-jeet-dev
  SNS_TOPIC_ARN    arn:aws:sns:<region>:<account-id>:fraud-alerts
  ML_ENDPOINT_NAME (optional) SageMaker endpoint for ML scoring
  ML_THRESHOLD     (optional) score threshold chosen by train_model.py
  DEDUP_TABLE      fraud-dedup-dev
  FEATURE_TABLE    fraud-feature-store
  EVENTS_TABLE     fraud-events
  AWS_REGION       us-east-1
"""

import json
import os
import time
import logging
from datetime import datetime, timezone, timedelta

import boto3
from botocore.exceptions import ClientError

from fraud_features import DECISION_TIME_FEATURES, build_feature_vector, to_csv_row

# ── Logging ────────────────────────────────────────────────────────
# Lambda automatically sends logs to CloudWatch Logs
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── AWS clients (initialized outside handler = reused across invocations)
# This is a critical Lambda optimization — creating boto3 clients inside
# the handler means recreating them on EVERY invocation. Outside = once
# per container, then reused for the lifetime of the warm Lambda instance.
_REGION       = os.environ.get("AWS_REGION", "us-east-1")
_ddb          = boto3.resource("dynamodb", region_name=_REGION)
_s3           = boto3.client("s3",         region_name=_REGION)
_sns          = boto3.client("sns",        region_name=_REGION)

# ── Config from environment variables ──────────────────────────────
S3_BUCKET     = os.environ.get("S3_BUCKET",     "fraud-platform-jeet-dev")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
DEDUP_TABLE   = os.environ.get("DEDUP_TABLE",   "fraud-dedup-dev")
FEATURE_TABLE = os.environ.get("FEATURE_TABLE", "fraud-feature-store")
EVENTS_TABLE  = os.environ.get("EVENTS_TABLE",  "fraud-events")
ML_ENDPOINT   = os.environ.get("ML_ENDPOINT_NAME", "")
ML_THRESHOLD  = float(os.environ.get("ML_THRESHOLD", "0.5"))
ML_FEATURES   = [c for c in os.environ.get("ML_FEATURE_COLUMNS", "").split(",") if c] \
                or DECISION_TIME_FEATURES
_sm_runtime   = boto3.client("sagemaker-runtime", region_name=_REGION) if ML_ENDPOINT else None

# ── DynamoDB table references ──────────────────────────────────────
_dedup_table   = _ddb.Table(DEDUP_TABLE)
_feature_table = _ddb.Table(FEATURE_TABLE)
_events_table  = _ddb.Table(EVENTS_TABLE)


# ─────────────────────────────────────────────────────────────────
# IDEMPOTENCY
# ─────────────────────────────────────────────────────────────────

def claim_event(event_id: str) -> bool:
    """
    Atomically claim an event before processing it.

    A conditional put succeeds for exactly one caller, so two Lambda instances
    receiving the same SQS message (at-least-once delivery) can never both
    process it. A get-then-put check would race: both could read "not seen"
    before either writes.

    Returns True if this invocation owns the event, False if it is a duplicate.
    """
    ttl = int(time.time()) + 86_400
    try:
        _dedup_table.put_item(
            Item={
                "event_id":   event_id,
                "status":     "IN_PROGRESS",
                "claimed_at": datetime.now(timezone.utc).isoformat(),
                "ttl":        ttl,
            },
            ConditionExpression="attribute_not_exists(event_id)",
        )
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise   # infrastructure error: let SQS retry the message


def release_claim(event_id: str):
    """Remove the claim after a processing failure so the SQS retry can run."""
    try:
        _dedup_table.delete_item(Key={"event_id": event_id})
    except ClientError as e:
        logger.warning("Could not release claim for %s: %s", event_id, e)


def mark_processed(event_id: str):
    """Mark a claimed event as DONE (claim already exists, so update it)."""
    try:
        _dedup_table.update_item(
            Key={"event_id": event_id},
            UpdateExpression="SET #s = :done, processed_at = :ts",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":done": "DONE",
                ":ts":   datetime.now(timezone.utc).isoformat(),
            },
        )
    except ClientError as e:
        logger.warning("Failed to mark %s processed: %s", event_id, e)


# ─────────────────────────────────────────────────────────────────
# VELOCITY FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────────

def get_and_update_velocity(customer_id: str, event_time: datetime) -> dict:
    """
    Compute transaction velocity for a customer at event_time.
    Uses DynamoDB as the feature store — each item represents one
    transaction timestamp for a customer.

    Pattern: write the current event, then count how many events
    exist within the 5-min and 1-hour windows.

    Why DynamoDB for this?
    - Single-digit millisecond reads at any scale
    - Atomic writes (no race conditions between Lambda invocations)
    - TTL auto-expires old velocity records
    - No server to manage (vs Redis which requires ElastiCache)

    At 10x scale: switch to DynamoDB atomic counters with
    conditional writes instead of scanning items per customer.
    """
    pk           = f"VELOCITY#{customer_id}"
    now_iso      = event_time.isoformat()
    ttl_2hr      = int(time.time()) + 7_200   # keep for 2 hours

    # Write this transaction's timestamp
    try:
        _feature_table.put_item(Item={
            "pk":  pk,
            "sk":  now_iso,
            "ttl": ttl_2hr,
        })
    except ClientError as e:
        logger.warning("Velocity write failed: %s", e)

    # Query all events for this customer in last 1 hour
    cutoff_1hr  = (event_time - timedelta(hours=1)).isoformat()
    cutoff_5min = (event_time - timedelta(minutes=5)).isoformat()

    try:
        resp = _feature_table.query(
            KeyConditionExpression=(
                "pk = :pk AND sk >= :cutoff"
            ),
            ExpressionAttributeValues={
                ":pk":     pk,
                ":cutoff": cutoff_1hr,
            },
            Select="COUNT",
        )
        count_1hr = resp.get("Count", 0)

        resp_5min = _feature_table.query(
            KeyConditionExpression=(
                "pk = :pk AND sk >= :cutoff"
            ),
            ExpressionAttributeValues={
                ":pk":     pk,
                ":cutoff": cutoff_5min,
            },
            Select="COUNT",
        )
        count_5min = resp_5min.get("Count", 0)

    except ClientError as e:
        logger.warning("Velocity query failed: %s", e)
        count_1hr  = 0
        count_5min = 0

    return {
        "velocity_5min_realtime":  count_5min,
        "velocity_1hour_realtime": count_1hr,
    }


# ─────────────────────────────────────────────────────────────────
# FRAUD DETECTION RULES
# ─────────────────────────────────────────────────────────────────

def run_fraud_rules(payload: dict, velocity: dict) -> dict:
    """
    Run 4 real-time fraud detection rules.
    Returns triggered rules and combined risk score.

    Rule design philosophy:
    - Rule-based gates are fast (microseconds) and interpretable
    - They run BEFORE any ML scoring to filter obvious cases
    - Each rule maps to a known fraud pattern with a business name
    - Risk score is additive — multiple signals compound suspicion

    At production scale these thresholds would be learned from
    historical data and tuned per merchant category and segment.
    """
    triggered = []
    risk_score = 0

    # Rule 1: Card testing — many transactions in 5 minutes
    # Fraudsters test stolen cards with small amounts in rapid succession
    v5 = velocity.get("velocity_5min_realtime", 0)
    if v5 >= 5:
        triggered.append("HIGH_VELOCITY_5MIN")
        risk_score += 4
        logger.info("Rule triggered: HIGH_VELOCITY_5MIN (count=%d)", v5)

    # Rule 2: High-risk merchant
    # 10% of merchants in our dataset have risk_score >= 0.75
    merchant_risk = float(payload.get("merchant_risk_score") or 0)
    if merchant_risk >= 0.75:
        triggered.append("HIGH_RISK_MERCHANT")
        risk_score += 3
        logger.info("Rule triggered: HIGH_RISK_MERCHANT (score=%.4f)", merchant_risk)

    # Rule 3: Impossible travel + new device (account takeover signal)
    # Customer in US suddenly transacts at foreign merchant on a brand-new device
    geo_flag    = int(payload.get("geo_anomaly_flag") or 0)
    device_flag = int(payload.get("new_device_flag") or 0)
    if geo_flag == 1 and device_flag == 1:
        triggered.append("GEO_ANOMALY_NEW_DEVICE")
        risk_score += 5
        logger.info("Rule triggered: GEO_ANOMALY_NEW_DEVICE")

    # Rule 4: Balance mismatch on TRANSFER
    # PaySim fraud pattern: balance doesn't change after a TRANSFER
    # (money moved but balance unchanged = fraudulent ledger entry)
    if payload.get("type") == "TRANSFER":
        old_bal = float(payload.get("oldbalanceOrg") or 0)
        new_bal = float(payload.get("newbalanceOrig") or 0)
        amount  = float(payload.get("amount") or 0)
        if old_bal > 0 and new_bal == old_bal and amount > 0:
            triggered.append("BALANCE_MISMATCH_TRANSFER")
            risk_score += 4
            logger.info("Rule triggered: BALANCE_MISMATCH_TRANSFER (amount=%.2f)", amount)

    # NOTE: the PaySim ground-truth label (isFraud) is deliberately NOT read here.
    # A real payment system never knows the label at decision time; reading it
    # would make every alert "correct" by construction. The label is stored with
    # the event (write_to_feature_store) only so alert precision can be measured.

    return {
        "triggered_rules": triggered,
        "risk_score":      risk_score,
        "is_flagged":      len(triggered) > 0,
    }


def score_with_model(payload: dict, velocity: dict, event_time: datetime):
    """
    Optional ML score from the SageMaker endpoint. Uses the same feature code as
    training (fraud_features.py), so there is no training/serving skew.
    Returns None when no endpoint is configured or the call fails: the rule
    engine still runs, so a model outage never blocks transaction processing.
    """
    if _sm_runtime is None:
        return None
    vector = build_feature_vector(
        payload,
        velocity_5min=velocity.get("velocity_5min_realtime", 0),
        velocity_1hour=velocity.get("velocity_1hour_realtime", 0),
        hour_of_day=event_time.hour,
        feature_columns=ML_FEATURES,
    )
    try:
        resp = _sm_runtime.invoke_endpoint(
            EndpointName=ML_ENDPOINT, ContentType="text/csv", Body=to_csv_row(vector)
        )
        return float(resp["Body"].read().decode("utf-8").strip())
    except (ClientError, ValueError) as e:
        logger.warning("ML scoring failed, falling back to rules only: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────
# DYNAMODB FEATURE STORE WRITE
# ─────────────────────────────────────────────────────────────────

def write_to_feature_store(
    event_id: str,
    metadata: dict,
    payload: dict,
    velocity: dict,
    fraud_result: dict,
):
    """
    Write the fully enriched transaction record to DynamoDB fraud-events.

    This table is the real-time feature store — downstream systems
    (model retraining, case management, dashboards) read from here.

    TTL: 7 days — enough for weekly model retraining pipeline.
    """
    ttl_7days = int(time.time()) + 7 * 86_400

    item = {
        # Keys
        "event_id":    event_id,

        # Metadata
        "event_time":  metadata.get("event_time", ""),
        "schema_version": metadata.get("schema_version", "v1"),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "processed_by": "fraud-lambda-v1",

        # Core transaction fields
        "customer_id":   str(payload.get("nameOrig", "")),
        "merchant_id":   str(payload.get("nameDest", "")),
        "amount":        str(payload.get("amount", 0)),     # DDB stores as Decimal
        "tx_type":       str(payload.get("type", "")),

        # Fraud detection results
        "triggered_rules": fraud_result["triggered_rules"],
        "risk_score":      fraud_result["risk_score"],
        "is_flagged":      fraud_result["is_flagged"],
        # DynamoDB rejects Python floats, so store the score as a string
        "ml_score":        "" if fraud_result.get("ml_score") is None
                           else f"{fraud_result['ml_score']:.6f}",

        # Real-time computed velocity (from DynamoDB feature store)
        "velocity_5min_realtime":  velocity.get("velocity_5min_realtime", 0),
        "velocity_1hour_realtime": velocity.get("velocity_1hour_realtime", 0),

        # Pre-computed features from producer
        "velocity_5min_precomputed":  int(payload.get("transaction_velocity_5min") or 0),
        "velocity_1hour_precomputed": int(payload.get("transaction_velocity_1hour") or 0),
        "geo_anomaly_flag":    int(payload.get("geo_anomaly_flag") or 0),
        "new_device_flag":     int(payload.get("new_device_flag") or 0),
        "merchant_risk_score": str(payload.get("merchant_risk_score") or ""),
        "merchant_category":   str(payload.get("merchant_category") or ""),
        "customer_segment":    str(payload.get("customer_segment") or ""),
        "customer_country":    str(payload.get("customer_country") or ""),
        "merchant_country":    str(payload.get("merchant_country") or ""),

        # Original fraud label (ground truth — never modified)
        "is_fraud_label":   int(payload.get("isFraud") or 0),
        "is_flagged_fraud":  int(payload.get("isFlaggedFraud") or 0),

        # TTL
        "ttl": ttl_7days,
    }

    _events_table.put_item(Item=item)


# ─────────────────────────────────────────────────────────────────
# S3 BRONZE WRITE
# ─────────────────────────────────────────────────────────────────

def write_to_s3_bronze(event_id: str, envelope: dict):
    """
    Write raw message to S3 Bronze layer.
    Path: bronze/transactions/YYYY/MM/DD/<event_id>.json

    Date-partitioned so Glue crawlers and Athena queries
    can scan only the partitions they need (partition pruning).
    """
    now = datetime.now(timezone.utc)
    key = (
        f"bronze/transactions/"
        f"{now.year:04d}/{now.month:02d}/{now.day:02d}/"
        f"{event_id}.json"
    )
    _s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(envelope, default=str).encode("utf-8"),
        ContentType="application/json",
    )


# ─────────────────────────────────────────────────────────────────
# SNS ALERT PUBLISH
# ─────────────────────────────────────────────────────────────────

def publish_fraud_alert(
    event_id: str,
    metadata: dict,
    payload: dict,
    velocity: dict,
    fraud_result: dict,
):
    """
    Publish a structured fraud alert to SNS.

    SNS fan-out pattern: one SNS topic → multiple subscribers:
      - Email notification to fraud ops team
      - SQS queue for case management system
      - Lambda for real-time dashboard update

    We publish to the topic; subscribers decide what to do with it.
    This decouples detection from notification — a core microservices pattern.
    """
    if not SNS_TOPIC_ARN:
        logger.warning("SNS_TOPIC_ARN not set — skipping alert publish")
        return

    alert = {
        "alert_type":     "FRAUD_DETECTED",
        "event_id":       event_id,
        "event_time":     metadata.get("event_time"),
        "customer_id":    payload.get("nameOrig"),
        "amount":         payload.get("amount"),
        "tx_type":        payload.get("type"),
        "merchant_id":    payload.get("nameDest"),
        "merchant_category": payload.get("merchant_category"),
        "triggered_rules":fraud_result["triggered_rules"],
        "risk_score":     fraud_result["risk_score"],
        "velocity_5min":  velocity.get("velocity_5min_realtime"),
        "geo_anomaly":    payload.get("geo_anomaly_flag"),
        "new_device":     payload.get("new_device_flag"),
        "is_fraud_label": payload.get("isFraud"),
        "alerted_at":     datetime.now(timezone.utc).isoformat(),
    }

    _sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"[FRAUD ALERT] Risk Score {fraud_result['risk_score']} — "
                f"{payload.get('type')} ${payload.get('amount', 0):.2f}",
        Message=json.dumps(alert, indent=2, default=str),
        MessageAttributes={
            "risk_score": {
                "DataType":    "Number",
                "StringValue": str(fraud_result["risk_score"]),
            },
            "tx_type": {
                "DataType":    "String",
                "StringValue": str(payload.get("type", "UNKNOWN")),
            },
        },
    )
    logger.info("SNS alert published | risk_score=%d | rules=%s",
                fraud_result["risk_score"], fraud_result["triggered_rules"])


# ─────────────────────────────────────────────────────────────────
# MAIN HANDLER
# ─────────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Entry point for SQS-triggered Lambda.

    SQS sends a batch of up to 10 messages per invocation.
    Lambda processes each message independently — if one fails,
    only that message goes back to the queue (partial batch failure).

    Context object contains:
      - context.function_name
      - context.aws_request_id
      - context.get_remaining_time_in_millis()

    Return format for partial batch failure reporting:
      {"batchItemFailures": [{"itemIdentifier": message_id}, ...]}
    This tells SQS exactly which messages failed so it only
    retries those — not the whole batch.
    """
    logger.info("Lambda invoked | records=%d | request_id=%s",
                len(event.get("Records", [])),
                context.aws_request_id)

    batch_failures = []
    stats = {
        "processed": 0,
        "flagged":   0,
        "skipped":   0,
        "errors":    0,
    }

    for record in event.get("Records", []):
        message_id = record["messageId"]
        event_id = ""
        claimed = False
        try:
            # ── Parse SQS message ──────────────────────────────
            envelope = json.loads(record["body"])
            metadata = envelope.get("metadata", {})
            payload  = envelope.get("payload", {})
            event_id = metadata.get("event_id", "")

            if not event_id:
                logger.warning("Missing event_id in message %s", message_id)
                stats["skipped"] += 1
                continue

            # ── Idempotency check ──────────────────────────────
            claimed = claim_event(event_id)
            if not claimed:
                logger.info("Duplicate — skipping %s", event_id)
                stats["skipped"] += 1
                continue

            # ── Parse event_time ───────────────────────────────
            event_time_str = metadata.get("event_time", "")
            try:
                event_time = datetime.fromisoformat(event_time_str)
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                event_time = datetime.now(timezone.utc)

            # ── Compute real-time velocity features ────────────
            # This is the core Lambda value-add over consumer_sqs.py:
            # velocity is computed from LIVE DynamoDB counters,
            # not pre-computed values from the CSV file.
            velocity = get_and_update_velocity(
                customer_id=payload.get("nameOrig", ""),
                event_time=event_time,
            )

            # ── Run fraud detection rules ──────────────────────
            fraud_result = run_fraud_rules(payload, velocity)

            # ── Optional ML score (SageMaker) ──────────────────
            ml_score = score_with_model(payload, velocity, event_time)
            fraud_result["ml_score"] = ml_score
            if ml_score is not None and ml_score >= ML_THRESHOLD:
                fraud_result["triggered_rules"].append("ML_MODEL_SCORE")
                fraud_result["is_flagged"] = True

            # ── Write to DynamoDB feature store ────────────────
            write_to_feature_store(
                event_id, metadata, payload, velocity, fraud_result
            )

            # ── Write to S3 Bronze ─────────────────────────────
            write_to_s3_bronze(event_id, envelope)

            # ── Publish SNS alert if flagged ───────────────────
            if fraud_result["is_flagged"]:
                publish_fraud_alert(
                    event_id, metadata, payload, velocity, fraud_result
                )
                stats["flagged"] += 1
                logger.info(
                    "FLAGGED | event_id=%s | rules=%s | risk=%d | amount=%.2f",
                    event_id,
                    fraud_result["triggered_rules"],
                    fraud_result["risk_score"],
                    float(payload.get("amount", 0)),
                )

            # ── Mark processed ─────────────────────────────────
            mark_processed(event_id)
            stats["processed"] += 1

        except Exception as e:
            if claimed:
                release_claim(event_id)
            logger.error("Error processing message %s: %s", message_id, e, exc_info=True)
            stats["errors"] += 1
            # Report this message as failed so SQS retries only it
            batch_failures.append({"itemIdentifier": message_id})

    # ── Log invocation summary ─────────────────────────────────────
    logger.info(
        "Invocation complete | processed=%d | flagged=%d | skipped=%d | errors=%d",
        stats["processed"], stats["flagged"],
        stats["skipped"],   stats["errors"],
    )

    # Return failed message IDs for partial batch failure handling
    return {"batchItemFailures": batch_failures}
