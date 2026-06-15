"""
upload_bronze_to_aws.py
========================
Pulls Bronze JSON files from LocalStack S3 and uploads them
to real AWS S3. Run once to seed the real data lake.

This is the bridge between local development and real AWS.
In production this step doesn't exist — the consumer writes
directly to real S3. But for portfolio development it's the
correct pattern: develop locally, promote to real AWS.
"""

import boto3
import json
import logging
from botocore.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("upload_bronze")

# ── LocalStack client (source) ─────────────────────────────────────
localstack = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

# ── Real AWS client (destination) ─────────────────────────────────
aws_s3 = boto3.client("s3", region_name="us-east-1")

LOCAL_BUCKET = "fraud-platform-raw-dev"
AWS_BUCKET   = "fraud-platform-jeet-dev"
PREFIX       = "bronze/transactions/"


def upload_all():
    log.info("Listing Bronze files in LocalStack ...")

    paginator = localstack.get_paginator("list_objects_v2")
    pages     = paginator.paginate(Bucket=LOCAL_BUCKET, Prefix=PREFIX)

    uploaded = 0
    skipped  = 0
    errors   = 0

    for page in pages:
        objects = page.get("Contents", [])
        if not objects:
            log.info("No objects found under %s", PREFIX)
            break

        for obj in objects:
            key = obj["Key"]

            # Skip folder placeholder objects
            if key.endswith("/"):
                continue

            try:
                # Download from LocalStack
                response = localstack.get_object(
                    Bucket=LOCAL_BUCKET,
                    Key=key,
                )
                body = response["Body"].read()

                # Quick validation — confirm it's valid JSON
                json.loads(body)

                # Upload to real AWS S3
                aws_s3.put_object(
                    Bucket=AWS_BUCKET,
                    Key=key,
                    Body=body,
                    ContentType="application/json",
                )
                uploaded += 1

                if uploaded % 100 == 0:
                    log.info("Uploaded %d files ...", uploaded)

            except json.JSONDecodeError:
                log.warning("Invalid JSON — skipping: %s", key)
                skipped += 1
            except Exception as e:
                log.error("Failed to upload %s: %s", key, e)
                errors += 1

    log.info("")
    log.info("=" * 50)
    log.info("Upload complete")
    log.info("  Uploaded : %d", uploaded)
    log.info("  Skipped  : %d", skipped)
    log.info("  Errors   : %d", errors)
    log.info("  Bucket   : s3://%s/%s", AWS_BUCKET, PREFIX)
    log.info("=" * 50)


if __name__ == "__main__":
    upload_all()