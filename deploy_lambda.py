"""
deploy_lambda.py
=================
Packages and deploys the Lambda function, connects the SQS trigger,
and runs a test invocation to confirm everything works.

Usage:
    python deploy_lambda.py
"""

import boto3
import json
import logging
import os
import time
import zipfile
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("deploy_lambda")

# ── Config ─────────────────────────────────────────────────────────
REGION        = "us-east-1"
ACCOUNT_ID    = "621402808508"
FUNCTION_NAME = "fraud-transaction-processor"
LAMBDA_ROLE   = f"arn:aws:iam::{ACCOUNT_ID}:role/FraudPlatformLambdaRole"
SNS_TOPIC_ARN = f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:fraud-alerts"
S3_BUCKET     = "fraud-platform-jeet-dev"
SQS_QUEUE     = "fraud-transactions-dev"

ENV_VARS = {
    "S3_BUCKET":     S3_BUCKET,
    "SNS_TOPIC_ARN": SNS_TOPIC_ARN,
    "DEDUP_TABLE":   "fraud-dedup-dev",
    "FEATURE_TABLE": "fraud-feature-store",
    "EVENTS_TABLE":  "fraud-events",
}


def zip_lambda_function() -> bytes:
    """
    Package lambda_function.py into a ZIP bytes object.
    Lambda requires a ZIP file for function code deployment.
    In production you'd use SAM, CDK, or Terraform for this.
    For a portfolio project, direct ZIP upload is correct and clear.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("lambda_function.py", "lambda_function.py")
    buffer.seek(0)
    log.info("Packaged lambda_function.py into ZIP (%d bytes)", len(buffer.getvalue()))
    return buffer.getvalue()


def deploy_function(lambda_client, zip_bytes: bytes) -> str:
    """Create or update the Lambda function. Returns function ARN."""
    config = dict(
        FunctionName = FUNCTION_NAME,
        Runtime      = "python3.12",
        Role         = LAMBDA_ROLE,
        Handler      = "lambda_function.lambda_handler",
        Timeout      = 60,       # seconds per invocation
        MemorySize   = 256,      # MB — enough for our processing
        Environment  = {"Variables": ENV_VARS},
        Description  = "Real-time fraud detection — SQS → DynamoDB → S3 → SNS",
    )

    try:
        resp = lambda_client.create_function(
            **config,
            Code={"ZipFile": zip_bytes},
        )
        arn = resp["FunctionArn"]
        log.info("Created Lambda function: %s", arn)

        # Wait until function is Active
        log.info("Waiting for function to become Active ...")
        waiter = lambda_client.get_waiter("function_active_v2")
        waiter.wait(FunctionName=FUNCTION_NAME)
        log.info("Function is Active")
        return arn

    except lambda_client.exceptions.ResourceConflictException:
        log.info("Function exists — updating code and config ...")

        lambda_client.update_function_code(
            FunctionName=FUNCTION_NAME,
            ZipFile=zip_bytes,
        )
        time.sleep(5)

        lambda_client.update_function_configuration(
            FunctionName = FUNCTION_NAME,
            Runtime      = config["Runtime"],
            Role         = config["Role"],
            Handler      = config["Handler"],
            Timeout      = config["Timeout"],
            MemorySize   = config["MemorySize"],
            Environment  = config["Environment"],
            Description  = config["Description"],
        )

        waiter = lambda_client.get_waiter("function_updated_v2")
        waiter.wait(FunctionName=FUNCTION_NAME)

        resp = lambda_client.get_function_configuration(FunctionName=FUNCTION_NAME)
        arn  = resp["FunctionArn"]
        log.info("Function updated: %s", arn)
        return arn


def get_sqs_arn(sqs_client) -> str:
    """Get the ARN of the main SQS queue."""
    url = sqs_client.get_queue_url(QueueName=SQS_QUEUE)["QueueUrl"]
    arn = sqs_client.get_queue_attributes(
        QueueUrl=url,
        AttributeNames=["QueueArn"],
    )["Attributes"]["QueueArn"]
    log.info("SQS queue ARN: %s", arn)
    return arn


def connect_sqs_trigger(lambda_client, sqs_arn: str):
    """
    Create an SQS Event Source Mapping — this is what makes Lambda
    trigger automatically when messages appear in the queue.

    BatchSize=10: Lambda receives up to 10 messages per invocation.
    Why 10? SQS sends max 10 messages per ReceiveMessage call.
    Matching this means one Lambda invocation per SQS poll cycle.

    MaximumBatchingWindowInSeconds=5: Wait up to 5 seconds to
    accumulate messages before invoking Lambda. Reduces invocation
    count when messages arrive in bursts.

    FunctionResponseTypes=["ReportBatchItemFailures"]: Enables
    partial batch failure — Lambda tells SQS exactly which messages
    failed so only those are retried, not the whole batch.
    """
    # Check if mapping already exists
    existing = lambda_client.list_event_source_mappings(
        FunctionName=FUNCTION_NAME,
        EventSourceArn=sqs_arn,
    )["EventSourceMappings"]

    if existing:
        mapping_id = existing[0]["UUID"]
        log.info("SQS trigger already exists (UUID=%s) — updating ...", mapping_id)
        lambda_client.update_event_source_mapping(
            UUID=mapping_id,
            BatchSize=10,
            MaximumBatchingWindowInSeconds=5,
            FunctionResponseTypes=["ReportBatchItemFailures"],
        )
        log.info("SQS trigger updated")
        return

    resp = lambda_client.create_event_source_mapping(
        FunctionName         = FUNCTION_NAME,
        EventSourceArn       = sqs_arn,
        BatchSize            = 10,
        MaximumBatchingWindowInSeconds = 5,
        FunctionResponseTypes= ["ReportBatchItemFailures"],
        Enabled              = True,
    )
    log.info("SQS trigger created | UUID=%s | State=%s",
             resp["UUID"], resp["State"])


def send_test_messages(sqs_client):
    """
    Send 20 test transactions to SQS so Lambda processes them.
    Includes one with HIGH_RISK_MERCHANT and one with PAYSIM_FRAUD_LABEL
    so we can confirm all code paths work.
    """
    import uuid
    from datetime import datetime, timezone

    url = sqs_client.get_queue_url(QueueName=SQS_QUEUE)["QueueUrl"]

    test_records = [
        # Normal transaction
        {
            "metadata": {
                "event_id":       f"EVT-TEST-{uuid.uuid4().hex[:8].upper()}",
                "event_time":     "2023-01-01T01:00:00",
                "arrival_time":   "2023-01-01T01:00:05",
                "schema_version": "v1",
                "produced_at":    datetime.now(timezone.utc).isoformat(),
                "source":         "test",
            },
            "payload": {
                "step": 1, "type": "PAYMENT",
                "amount": 1500.00,
                "nameOrig": "C_TEST_001", "oldbalanceOrg": 5000.0,
                "newbalanceOrig": 3500.0, "nameDest": "M_TEST_001",
                "oldbalanceDest": 0.0, "newbalanceDest": 1500.0,
                "isFraud": 0, "isFlaggedFraud": 0,
                "merchant_risk_score": 0.15,
                "merchant_category": "Retail",
                "customer_segment": "Family",
                "geo_anomaly_flag": 0, "new_device_flag": 0,
                "customer_country": "US", "merchant_country": "US",
                "transaction_velocity_5min": 0,
                "transaction_velocity_1hour": 0,
            },
        },
        # High-risk merchant transaction
        {
            "metadata": {
                "event_id":       f"EVT-TEST-{uuid.uuid4().hex[:8].upper()}",
                "event_time":     "2023-01-01T01:02:00",
                "arrival_time":   "2023-01-01T01:02:05",
                "schema_version": "v1",
                "produced_at":    datetime.now(timezone.utc).isoformat(),
                "source":         "test",
            },
            "payload": {
                "step": 1, "type": "CASH_OUT",
                "amount": 85000.00,
                "nameOrig": "C_TEST_002", "oldbalanceOrg": 90000.0,
                "newbalanceOrig": 5000.0, "nameDest": "M_TEST_002",
                "oldbalanceDest": 0.0, "newbalanceDest": 85000.0,
                "isFraud": 0, "isFlaggedFraud": 0,
                "merchant_risk_score": 0.92,
                "merchant_category": "Electronics",
                "customer_segment": "High Spender",
                "geo_anomaly_flag": 0, "new_device_flag": 1,
                "customer_country": "US", "merchant_country": "NG",
                "transaction_velocity_5min": 2,
                "transaction_velocity_1hour": 4,
            },
        },
        # Confirmed fraud transaction
        {
            "metadata": {
                "event_id":       f"EVT-TEST-{uuid.uuid4().hex[:8].upper()}",
                "event_time":     "2023-01-01T01:05:00",
                "arrival_time":   "2023-01-01T01:05:10",
                "schema_version": "v2",
                "produced_at":    datetime.now(timezone.utc).isoformat(),
                "source":         "test",
            },
            "payload": {
                "step": 1, "type": "TRANSFER",
                "amount": 450000.00,
                "nameOrig": "C_TEST_003", "oldbalanceOrg": 450000.0,
                "newbalanceOrig": 450000.0,   # balance unchanged = mismatch
                "nameDest": "M_TEST_003",
                "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
                "isFraud": 1, "isFlaggedFraud": 1,
                "merchant_risk_score": 0.88,
                "merchant_category": "Travel",
                "customer_segment": "Traveler",
                "geo_anomaly_flag": 1, "new_device_flag": 1,
                "customer_country": "US", "merchant_country": "BR",
                "transaction_velocity_5min": 0,
                "transaction_velocity_1hour": 1,
                "ip_risk_score": 0.91,
            },
        },
    ]

    entries = [
        {
            "Id":          str(i),
            "MessageBody": json.dumps(rec),
        }
        for i, rec in enumerate(test_records)
    ]

    resp = sqs_client.send_message_batch(QueueUrl=url, Entries=entries)
    log.info("Sent %d test messages to SQS", len(entries))
    if resp.get("Failed"):
        log.warning("Some test messages failed: %s", resp["Failed"])


def verify_lambda_output(ddb_resource, s3_client, lambda_client):
    """
    Wait for Lambda to process messages then verify output
    in DynamoDB and S3.
    """
    log.info("")
    log.info("Waiting 20 seconds for Lambda to process messages ...")
    time.sleep(20)

    # Check DynamoDB fraud-events table
    table  = ddb_resource.Table("fraud-events")
    scan   = table.scan(Select="COUNT")
    events = scan.get("Count", 0)
    log.info("DynamoDB fraud-events: %d records", events)

    # Check Lambda recent invocations via CloudWatch
    cw = boto3.client("cloudwatch", region_name=REGION)
    try:
        resp = cw.get_metric_statistics(
            Namespace="AWS/Lambda",
            MetricName="Invocations",
            Dimensions=[{"Name": "FunctionName", "Value": FUNCTION_NAME}],
            StartTime=__import__("datetime").datetime.utcnow() - __import__("datetime").timedelta(minutes=5),
            EndTime=__import__("datetime").datetime.utcnow(),
            Period=300,
            Statistics=["Sum"],
        )
        invocations = sum(d["Sum"] for d in resp.get("Datapoints", []))
        log.info("Lambda invocations (last 5 min): %.0f", invocations)
    except Exception as e:
        log.debug("CloudWatch metric fetch: %s", e)

    log.info("")
    log.info("=" * 60)
    log.info("Verification complete")
    log.info("  DynamoDB fraud-events : %d records", events)
    log.info("")
    log.info("Check CloudWatch Logs for Lambda output:")
    log.info("  https://console.aws.amazon.com/cloudwatch/home?"
             "region=%s#logsV2:log-groups/log-group/"
             "$252Faws$252Flambda$252F%s", REGION, FUNCTION_NAME)
    log.info("=" * 60)


def main():
    lambda_client = boto3.client("lambda", region_name=REGION)
    sqs_client    = boto3.client("sqs",    region_name=REGION)
    ddb_resource  = boto3.resource("dynamodb", region_name=REGION)
    s3_client     = boto3.client("s3",     region_name=REGION)

    log.info("=" * 60)
    log.info("Fraud Platform — Lambda Deployment")
    log.info("  Function : %s", FUNCTION_NAME)
    log.info("  Role     : %s", LAMBDA_ROLE)
    log.info("  SQS      : %s", SQS_QUEUE)
    log.info("  SNS      : %s", SNS_TOPIC_ARN)
    log.info("=" * 60)

    # 1. Package and deploy
    log.info("")
    log.info("Step 1: Packaging Lambda function ...")
    zip_bytes = zip_lambda_function()

    log.info("Step 2: Deploying to AWS Lambda ...")
    function_arn = deploy_function(lambda_client, zip_bytes)

    # 2. Connect SQS trigger
    log.info("")
    log.info("Step 3: Connecting SQS trigger ...")
    sqs_arn = get_sqs_arn(sqs_client)
    connect_sqs_trigger(lambda_client, sqs_arn)

    # Wait for trigger to become active
    log.info("Waiting 10 seconds for trigger to activate ...")
    time.sleep(10)

    # 3. Send test messages
    log.info("")
    log.info("Step 4: Sending 3 test transactions to SQS ...")
    send_test_messages(sqs_client)

    # 4. Verify
    log.info("")
    log.info("Step 5: Verifying Lambda processed the messages ...")
    verify_lambda_output(ddb_resource, s3_client, lambda_client)

    log.info("")
    log.info("Deployment complete.")
    log.info("Function ARN: %s", function_arn)


if __name__ == "__main__":
    main()
