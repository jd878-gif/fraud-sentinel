"""
project_config.py
=================
Account-specific settings come from the environment (or .env), never from
source code. The AWS account ID is resolved from your credentials via STS,
so the repo works for anyone who clones it.

    export AWS_REGION=us-east-1
    export FRAUD_S3_BUCKET=fraud-platform-<you>-dev   # optional
"""
import os
from functools import lru_cache

REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("FRAUD_S3_BUCKET", "fraud-platform-jeet-dev")


@lru_cache(maxsize=1)
def get_account_id() -> str:
    """AWS_ACCOUNT_ID env var if set, otherwise ask STS for the caller's account."""
    if os.environ.get("AWS_ACCOUNT_ID"):
        return os.environ["AWS_ACCOUNT_ID"]
    import boto3
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def role_arn(role_name: str) -> str:
    return f"arn:aws:iam::{get_account_id()}:role/{role_name}"


def sns_topic_arn(topic: str = "fraud-alerts") -> str:
    return f"arn:aws:sns:{REGION}:{get_account_id()}:{topic}"
