"""
verify_pipeline.py
===================
End-to-end health check. Run after producer + consumer to confirm:
  1. Main queue is drained (messages consumed)
  2. Alerts queue has flagged transactions
  3. DynamoDB has processed event IDs
  4. S3 Bronze has JSON files
  5. DLQ is empty (no failures)
"""

import json, logging
from aws_clients import sqs as sqs_client, dynamodb as dynamodb_client, s3 as s3_client

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("verify")


def queue_depth(sqs, url: str, label: str):
    attrs = sqs.get_queue_attributes(
        QueueUrl=url,
        AttributeNames=[
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
        ],
    )["Attributes"]
    visible    = int(attrs["ApproximateNumberOfMessages"])
    in_flight  = int(attrs["ApproximateNumberOfMessagesNotVisible"])
    log.info("  %-30s visible=%d  in-flight=%d", label, visible, in_flight)
    return visible


def peek_alert(sqs, url: str):
    """Read one alert message to show what a flagged transaction looks like."""
    resp = sqs.receive_message(
        QueueUrl=url, MaxNumberOfMessages=1, WaitTimeSeconds=3,
        VisibilityTimeout=5,   # short — we're peeking, not consuming
    )
    msgs = resp.get("Messages", [])
    if msgs:
        alert = json.loads(msgs[0]["Body"])
        log.info("  Sample alert:")
        for k, v in alert.items():
            log.info("    %-25s: %s", k, v)
        # Return the message to the queue (don't delete it)
        sqs.change_message_visibility(
            QueueUrl=url,
            ReceiptHandle=msgs[0]["ReceiptHandle"],
            VisibilityTimeout=0,
        )
    else:
        log.info("  No alerts visible right now")


def check_dynamodb(ddb, table: str):
    resp  = ddb.scan(TableName=table, Select="COUNT")
    count = resp["Count"]
    log.info("  DynamoDB '%s': %d processed event IDs", table, count)
    return count


def check_s3(s3, bucket: str):
    resp  = s3.list_objects_v2(Bucket=bucket, Prefix="bronze/transactions/")
    count = resp.get("KeyCount", 0)
    log.info("  S3 bronze/transactions/: %d JSON files", count)
    if count > 0:
        sample_key = resp["Contents"][0]["Key"]
        log.info("  Sample key: %s", sample_key)
    return count


def main():
    with open("queue_config.json") as f:
        cfg = json.load(f)

    sqs = sqs_client()
    ddb = dynamodb_client()
    s3  = s3_client()

    log.info("")
    log.info("=" * 60)
    log.info("PIPELINE VERIFICATION REPORT")
    log.info("=" * 60)

    log.info("")
    log.info("Queue depths:")
    main_depth  = queue_depth(sqs, cfg["main_queue_url"],  "fraud-transactions-dev")
    alert_depth = queue_depth(sqs, cfg["alert_queue_url"], "fraud-alerts-dev")
    dlq_depth   = queue_depth(sqs, cfg["dlq_url"],         "fraud-transactions-dev-dlq")

    log.info("")
    log.info("Sample flagged transaction (from alerts queue):")
    peek_alert(sqs, cfg["alert_queue_url"])

    log.info("")
    log.info("Storage:")
    ddb_count = check_dynamodb(ddb, cfg["dedup_table"])
    s3_count  = check_s3(s3, cfg["s3_bucket"])

    log.info("")
    log.info("=" * 60)
    log.info("RESULT:")
    if dlq_depth == 0 and ddb_count > 0:
        log.info("  ✓ Pipeline healthy — DLQ empty, records processed")
    elif dlq_depth > 0:
        log.warning("  ✗ DLQ has %d messages — some records failed processing", dlq_depth)
    else:
        log.warning("  ? DynamoDB empty — has the consumer run yet?")
    log.info("=" * 60)


if __name__ == "__main__":
    main()