"""
Creates all AWS resources inside LocalStack.
Run ONCE after docker-compose up -d.

Resources created:
  SQS  fraud-transactions-dev      main queue
  SQS  fraud-transactions-dev-dlq  dead-letter queue
  SQS  fraud-alerts-dev            high-risk flagged transactions
  DynamoDB  fraud-dedup-dev        idempotency (at-least-once protection)
  S3   fraud-platform-raw-dev      Bronze/Silver/Gold landing zones
"""

import json, logging, time, requests
from aws_clients import sqs, dynamodb, s3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("bootstrap")

MAIN_QUEUE  = "fraud-transactions-dev"
DLQ_NAME    = "fraud-transactions-dev-dlq"
ALERT_QUEUE = "fraud-alerts-dev"
DEDUP_TABLE = "fraud-dedup-dev"
S3_BUCKET   = "fraud-platform-raw-dev"


def wait_for_localstack():
    log.info("Waiting for LocalStack ...")
    for _ in range(30):
        try:
            r = requests.get("http://localhost:4566/_localstack/health", timeout=3)
            if r.json().get("services", {}).get("sqs") in ("running", "available"):
                log.info("LocalStack ready")
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("LocalStack not healthy. Is Docker running?")


def create_queues(sqs_client):
    urls = {}

    # Helper: create queue or return existing URL
    def get_or_create(queue_name, attributes=None):
        try:
            resp = sqs_client.create_queue(
                QueueName=queue_name,
                Attributes=attributes or {},
            )
            log.info("Created queue: %s", queue_name)
            return resp["QueueUrl"]
        except Exception as e:
            if "QueueAlreadyExists" in str(e) or "QueueNameExists" in str(e):
                log.info("Queue already exists, fetching URL: %s", queue_name)
                resp = sqs_client.get_queue_url(QueueName=queue_name)
                return resp["QueueUrl"]
            raise

    # 1. DLQ
    dlq_url = get_or_create(DLQ_NAME)
    urls[DLQ_NAME] = dlq_url

    dlq_arn = sqs_client.get_queue_attributes(
        QueueUrl=dlq_url,
        AttributeNames=["QueueArn"],
    )["Attributes"]["QueueArn"]
    log.info("DLQ ARN: %s", dlq_arn)

    # 2. Main queue
    main_url = get_or_create(
        MAIN_QUEUE,
        {
            "VisibilityTimeout":             "60",
            "ReceiveMessageWaitTimeSeconds": "20",
            "RedrivePolicy": json.dumps({
                "deadLetterTargetArn": dlq_arn,
                "maxReceiveCount":     "3",
            }),
        },
    )
    urls[MAIN_QUEUE] = main_url

    # 3. Alerts queue
    alert_url = get_or_create(
        ALERT_QUEUE,
        {
            "VisibilityTimeout":             "30",
            "ReceiveMessageWaitTimeSeconds": "20",
        },
    )
    urls[ALERT_QUEUE] = alert_url

    return urls


def create_dedup_table(ddb):
    try:
        ddb.create_table(
            TableName=DEDUP_TABLE,
            KeySchema=[{"AttributeName": "event_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "event_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        ddb.update_time_to_live(
            TableName=DEDUP_TABLE,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
        )
        log.info("DynamoDB table created: %s (TTL enabled)", DEDUP_TABLE)
    except Exception as e:
        if "ResourceInUseException" in str(e):
            log.info("DynamoDB table already exists")
        else:
            raise


def create_s3_bucket(s3_client):
    try:
        s3_client.create_bucket(Bucket=S3_BUCKET)
        log.info("S3 bucket created: %s", S3_BUCKET)
    except Exception as e:
        if "BucketAlready" in str(e):
            log.info("S3 bucket already exists — skipping")
        else:
            raise


def main():
    wait_for_localstack()

    q_urls = create_queues(sqs())
    create_dedup_table(dynamodb())
    create_s3_bucket(s3())

    cfg = {
        "main_queue_url":  q_urls[MAIN_QUEUE],
        "dlq_url":         q_urls[DLQ_NAME],
        "alert_queue_url": q_urls[ALERT_QUEUE],
        "dedup_table":     DEDUP_TABLE,
        "s3_bucket":       S3_BUCKET,
    }
    with open("queue_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    log.info("")
    log.info("=" * 60)
    log.info("Bootstrap complete. queue_config.json written.")
    log.info("Next: python producer_sqs.py --max-rows 1000")
    log.info("=" * 60)

if __name__ == "__main__":
    main()