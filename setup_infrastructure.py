"""
kinesis_producer/setup_infrastructure.py
=========================================
Creates the Kinesis stream and all required IAM resources for the
fraud detection platform.

Run this ONCE before running the producer.

Why script the infrastructure instead of using the console?
  - Reproducible: you can tear down and rebuild in minutes
  - Auditable: IAM policies are explicit and reviewable
  - Portfolio signal: it shows you understand IAM least-privilege,
    not just "click and hope"

Usage:
    python setup_infrastructure.py --region us-east-1 --env dev
    python setup_infrastructure.py --region us-east-1 --env dev --teardown
"""

import argparse
import json
import logging
import sys
import time

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("setup")

# ─────────────────────────────────────────────────────
# NAMES  (all resources prefixed so they're easy to find/delete)
# ─────────────────────────────────────────────────────
def names(env: str) -> dict:
    return {
        "stream":          f"fraud-transactions-{env}",
        "producer_policy": f"FraudPlatformProducerPolicy-{env}",
        "consumer_policy": f"FraudPlatformConsumerPolicy-{env}",
        "producer_role":   f"FraudPlatformProducerRole-{env}",
        "consumer_role":   f"FraudPlatformConsumerRole-{env}",
        "producer_user":   f"fraud-producer-{env}",
    }


# ─────────────────────────────────────────────────────
# KINESIS STREAM
# ─────────────────────────────────────────────────────

def create_stream(kinesis, stream_name: str, shard_count: int = 1):
    """
    Create the Kinesis Data Stream.

    Shard sizing rationale for dev/prototype:
      1 shard = 1 MB/s write, 2 MB/s read, 1,000 records/s write
      Our producer runs at 100 TPS → well within 1 shard's capacity
      Average record size ~1.5 KB → 100 * 1.5 KB = 150 KB/s → well under 1 MB/s

    Retention: 24 hours (default). For the full fraud platform
    we would set 7 days so failed Glue jobs can replay.
    """
    try:
        kinesis.create_stream(
            StreamName=stream_name,
            ShardCount=shard_count,
            StreamModeDetails={"StreamMode": "PROVISIONED"},
        )
        log.info("Creating Kinesis stream '%s' (%d shard(s)) ...", stream_name, shard_count)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            log.info("Stream '%s' already exists — skipping creation", stream_name)
            return
        raise

    # Wait for stream to become ACTIVE
    waiter = kinesis.get_waiter("stream_exists")
    waiter.wait(StreamName=stream_name, WaiterConfig={"MaxAttempts": 30, "Delay": 5})
    log.info("Stream '%s' is ACTIVE", stream_name)

    # Set retention to 7 days (needed for replay on pipeline failures)
    kinesis.increase_stream_retention_period(
        StreamName=stream_name,
        RetentionPeriodHours=168,  # 7 days
    )
    log.info("Retention set to 7 days")


def describe_stream(kinesis, stream_name: str) -> dict:
    resp = kinesis.describe_stream_summary(StreamName=stream_name)
    info = resp["StreamDescriptionSummary"]
    log.info("Stream info:")
    log.info("  ARN:              %s", info["StreamARN"])
    log.info("  Status:           %s", info["StreamStatus"])
    log.info("  Shards:           %d", info["OpenShardCount"])
    log.info("  Retention (hrs):  %d", info["RetentionPeriodHours"])
    return info


def delete_stream(kinesis, stream_name: str):
    try:
        kinesis.delete_stream(StreamName=stream_name, EnforceConsumerDeletion=True)
        log.info("Deleted stream '%s'", stream_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            log.info("Stream '%s' does not exist — nothing to delete", stream_name)
        else:
            raise


# ─────────────────────────────────────────────────────
# IAM POLICIES
# ─────────────────────────────────────────────────────

def producer_policy_document(stream_arn: str) -> dict:
    """
    Least-privilege policy for the Kinesis producer.

    Allowed actions (ONLY what the producer needs):
      kinesis:PutRecord      — single record (fallback)
      kinesis:PutRecords     — batch up to 500 records
      kinesis:DescribeStream — needed to validate stream exists on startup
      cloudwatch:PutMetricData — emit custom producer metrics

    Explicitly NOT allowed:
      kinesis:GetRecords     — producer should never read from the stream
      kinesis:DeleteStream   — producer should never destroy infrastructure
      kinesis:MergeShards    — producer should never reshape the stream
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "KinesisProducerAccess",
                "Effect": "Allow",
                "Action": [
                    "kinesis:PutRecord",
                    "kinesis:PutRecords",
                    "kinesis:DescribeStream",
                    "kinesis:DescribeStreamSummary",
                ],
                "Resource": stream_arn,
            },
            {
                "Sid": "CloudWatchMetrics",
                "Effect": "Allow",
                "Action": ["cloudwatch:PutMetricData"],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "cloudwatch:namespace": "FraudPlatform/Producer"
                    }
                },
            },
        ],
    }


def consumer_policy_document(stream_arn: str) -> dict:
    """
    Least-privilege policy for Kinesis consumers (Lambda, Glue).

    Allowed actions:
      kinesis:GetRecords         — read records from a shard
      kinesis:GetShardIterator   — get starting position in a shard
      kinesis:DescribeStream     — discover shard layout
      kinesis:ListShards         — enumerate shards (needed after resharding)
      kinesis:SubscribeToShard   — Enhanced Fan-Out (low latency, higher cost)
      kinesis:RegisterStreamConsumer — register an EFO consumer

    Design note: Lambda and Glue both use this same policy but are
    attached via separate IAM roles so you can revoke one independently.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "KinesisConsumerAccess",
                "Effect": "Allow",
                "Action": [
                    "kinesis:GetRecords",
                    "kinesis:GetShardIterator",
                    "kinesis:DescribeStream",
                    "kinesis:DescribeStreamSummary",
                    "kinesis:ListShards",
                    "kinesis:ListStreams",
                    "kinesis:SubscribeToShard",
                    "kinesis:RegisterStreamConsumer",
                ],
                "Resource": stream_arn,
            },
            {
                "Sid": "CloudWatchMetrics",
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:PutMetricData",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": "*",
            },
        ],
    }


def upsert_policy(iam, policy_name: str, document: dict, account_id: str) -> str:
    """Create an IAM policy or update its default version if it already exists."""
    policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
    try:
        resp = iam.create_policy(
            PolicyName=policy_name,
            PolicyDocument=json.dumps(document),
            Description=f"Fraud Platform — {policy_name}",
        )
        arn = resp["Policy"]["Arn"]
        log.info("Created policy: %s", arn)
        return arn
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            log.info("Policy '%s' exists — creating new version", policy_name)
            iam.create_policy_version(
                PolicyArn=policy_arn,
                PolicyDocument=json.dumps(document),
                SetAsDefault=True,
            )
            return policy_arn
        raise


def create_producer_user(iam, username: str, policy_arn: str) -> dict:
    """
    Create an IAM user for the local producer script.

    Note on design choice: for production, you would use an IAM Role
    attached to an EC2 instance or ECS task — never long-lived access keys.
    We create a user here ONLY for local development convenience.
    The README warns about this explicitly.
    """
    try:
        iam.create_user(UserName=username)
        log.info("Created IAM user: %s", username)
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            log.info("IAM user '%s' already exists", username)
        else:
            raise

    iam.attach_user_policy(UserName=username, PolicyArn=policy_arn)
    log.info("Attached producer policy to user '%s'", username)

    # Create access keys
    keys = iam.create_access_key(UserName=username)["AccessKey"]
    log.info("Access key created (store securely — shown once only):")
    return {
        "AccessKeyId":     keys["AccessKeyId"],
        "SecretAccessKey": keys["SecretAccessKey"],
    }


# ─────────────────────────────────────────────────────
# TEARDOWN
# ─────────────────────────────────────────────────────

def teardown(kinesis, iam, n: dict, account_id: str):
    """Delete all resources created by setup. Used for cleanup after demos."""
    log.info("Tearing down all Fraud Platform resources ...")
    delete_stream(kinesis, n["stream"])

    for policy_name in [n["producer_policy"], n["consumer_policy"]]:
        arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
        try:
            # Delete all non-default versions first
            versions = iam.list_policy_versions(PolicyArn=arn)["Versions"]
            for v in versions:
                if not v["IsDefaultVersion"]:
                    iam.delete_policy_version(PolicyArn=arn, VersionId=v["VersionId"])
            # Detach from all entities
            for u in iam.list_entities_for_policy(PolicyArn=arn)["PolicyUsers"]:
                iam.detach_user_policy(UserName=u["UserName"], PolicyArn=arn)
            iam.delete_policy(PolicyArn=arn)
            log.info("Deleted policy: %s", arn)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                log.info("Policy '%s' not found — skipping", policy_name)
            else:
                log.warning("Could not delete policy %s: %s", policy_name, e)

    try:
        # Delete access keys first
        keys = iam.list_access_keys(UserName=n["producer_user"])["AccessKeyMetadata"]
        for k in keys:
            iam.delete_access_key(UserName=n["producer_user"], AccessKeyId=k["AccessKeyId"])
        iam.delete_user(UserName=n["producer_user"])
        log.info("Deleted IAM user: %s", n["producer_user"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            log.info("User '%s' not found — skipping", n["producer_user"])
        else:
            log.warning("Could not delete user: %s", e)

    log.info("Teardown complete. All billable resources removed.")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Setup Fraud Platform Kinesis infrastructure")
    parser.add_argument("--region",   default="us-east-1")
    parser.add_argument("--env",      default="dev", choices=["dev", "staging", "prod"])
    parser.add_argument("--shards",   type=int, default=1,
                        help="Kinesis shard count (1 for dev, scale up for prod)")
    parser.add_argument("--teardown", action="store_true",
                        help="Delete all created resources")
    args = parser.parse_args()

    session    = boto3.Session(region_name=args.region)
    kinesis    = session.client("kinesis")
    iam        = session.client("iam")
    sts        = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    n          = names(args.env)

    if args.teardown:
        teardown(kinesis, iam, n, account_id)
        return

    log.info("Setting up Fraud Platform infrastructure | env=%s | region=%s",
             args.env, args.region)
    log.info("AWS Account: %s", account_id)

    # 1. Kinesis stream
    create_stream(kinesis, n["stream"], shard_count=args.shards)
    info = describe_stream(kinesis, n["stream"])
    stream_arn = info["StreamARN"]

    # 2. IAM policies
    producer_arn = upsert_policy(
        iam, n["producer_policy"],
        producer_policy_document(stream_arn),
        account_id,
    )
    consumer_arn = upsert_policy(
        iam, n["consumer_policy"],
        consumer_policy_document(stream_arn),
        account_id,
    )

    # 3. Producer IAM user (local dev only — use roles in production)
    credentials = create_producer_user(iam, n["producer_user"], producer_arn)

    # Print next steps
    log.info("")
    log.info("═══════════════════════════════════════════════════════════════")
    log.info("Infrastructure ready. Next steps:")
    log.info("")
    log.info("1. Configure AWS credentials for the producer:")
    log.info("   AWS_ACCESS_KEY_ID=%s", credentials["AccessKeyId"])
    log.info("   AWS_SECRET_ACCESS_KEY=<shown above — store in .env>")
    log.info("")
    log.info("2. Run a dry-run test (no AWS calls):")
    log.info("   python producer.py --dry-run --max-rows 100")
    log.info("")
    log.info("3. Stream 1,000 records to Kinesis:")
    log.info("   python producer.py \\")
    log.info("       --stream-name %s \\", n["stream"])
    log.info("       --region %s \\", args.region)
    log.info("       --tps 100 \\")
    log.info("       --max-rows 1000")
    log.info("")
    log.info("4. Verify records arrived:")
    log.info("   python verify_stream.py --stream-name %s --region %s",
             n["stream"], args.region)
    log.info("═══════════════════════════════════════════════════════════════")

    # Write .env file (never commit this)
    env_content = (
        f"AWS_ACCESS_KEY_ID={credentials['AccessKeyId']}\n"
        f"AWS_SECRET_ACCESS_KEY={credentials['SecretAccessKey']}\n"
        f"AWS_REGION={args.region}\n"
        f"KINESIS_STREAM_NAME={n['stream']}\n"
    )
    with open(".env", "w") as f:
        f.write(env_content)
    log.info("Credentials written to .env (add .env to .gitignore NOW)")


if __name__ == "__main__":
    main()
