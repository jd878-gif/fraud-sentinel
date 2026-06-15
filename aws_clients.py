"""
Central boto3 client factory.
Change USE_LOCALSTACK = False to point at real AWS with zero other changes.
"""
import boto3
from botocore.config import Config

USE_LOCALSTACK      = True
LOCALSTACK_ENDPOINT = "http://localhost:4566"
AWS_REGION          = "us-east-1"

_RETRY_CONFIG = Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    max_pool_connections=20,
)

def _client(service: str):
    kwargs = dict(
        service_name = service,
        region_name  = AWS_REGION,
        config       = _RETRY_CONFIG,
    )
    if USE_LOCALSTACK:
        kwargs["endpoint_url"]         = LOCALSTACK_ENDPOINT
        kwargs["aws_access_key_id"]    = "test"
        kwargs["aws_secret_access_key"]= "test"
    return boto3.client(**kwargs)

def sqs():       return _client("sqs")
def dynamodb():  return _client("dynamodb")
def s3():        return _client("s3")
def cloudwatch():return _client("cloudwatch")