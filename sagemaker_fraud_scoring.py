"""
sagemaker_fraud_scoring.py
===========================
Trains a fraud detection model using data from the Gold layer
and deploys it as a SageMaker Serverless Inference endpoint.

Why Serverless Inference (not real-time endpoint)?
  Real-time endpoint: $0.046/hour = $33/month even with zero traffic
  Serverless endpoint: $0 when idle, pay per invocation only
  Free tier: 150,000 invocations/month
  For a portfolio project, serverless is the correct choice.

Pipeline:
  1. Download Silver Parquet from S3 (has is_fraud label + all features)
  2. Feature engineering (select and encode fraud signal features)
  3. Train XGBoost classifier (SageMaker built-in algorithm)
  4. Register model in SageMaker Model Registry
  5. Deploy Serverless Inference endpoint
  6. Test endpoint with sample transactions
  7. Update Lambda environment to call SageMaker for scoring
  8. Print cleanup command (to avoid forgetting)

Features used:
  merchant_risk_score     — pre-computed merchant risk (0-1)
  velocity_5min           — transactions in last 5 minutes
  velocity_1hour          — transactions in last hour
  geo_anomaly_flag        — impossible travel signal
  new_device_flag         — account takeover signal
  balance_mismatch_flag   — fraudulent ledger signal
  amount                  — transaction amount
  rule_signal_score       — combined rule-based score

Label: is_fraud (original PaySim ground truth, never modified)
"""

import boto3
import json
import logging
import os
import time
import io

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("sagemaker_fraud")

# ── Config ─────────────────────────────────────────────────────────
REGION        = "us-east-1"
ACCOUNT_ID    = "621402808508"
S3_BUCKET     = "fraud-platform-jeet-dev"
SILVER_PATH   = f"s3://{S3_BUCKET}/silver/transactions/"
MODEL_PREFIX  = "fraud-model"
ENDPOINT_NAME = "fraud-scoring-serverless"
FUNCTION_NAME = "fraud-transaction-processor"

# SageMaker execution role — reuse the Glue role (it has S3 access)
# In production you'd create a dedicated SageMaker role
SAGEMAKER_ROLE = f"arn:aws:iam::{ACCOUNT_ID}:role/FraudPlatformGlueRole"


# ─────────────────────────────────────────────────────────────────
# STEP 1: LOAD TRAINING DATA FROM S3 SILVER LAYER
# ─────────────────────────────────────────────────────────────────

def load_training_data(s3_client) -> pd.DataFrame:
    """
    Download Silver Parquet files from S3 and load into pandas.
    Silver layer has clean, validated data with all fraud features.

    Why Silver instead of Gold?
    Gold is aggregated (one row per merchant category, segment, etc.)
    Silver has one row per transaction — what we need for ML training.
    """
    log.info("Loading Silver layer training data from S3 ...")

    # List all Parquet files in Silver layer
    resp    = s3_client.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix="silver/transactions/",
    )
    objects = [o for o in resp.get("Contents", [])
               if o["Key"].endswith(".parquet")]

    if not objects:
        raise ValueError("No Parquet files found in Silver layer. Run Glue ETL first.")

    log.info("Found %d Parquet files in Silver layer", len(objects))

    # Download and concatenate all Parquet files
    dfs = []
    for obj in objects:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
        df = pd.read_parquet(io.BytesIO(response["Body"].read()))
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    log.info("Loaded %d records, %d columns", len(df), len(df.columns))
    log.info("Fraud rate: %.2f%%", df["is_fraud"].mean() * 100)
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "merchant_risk_score",
    "velocity_5min",
    "velocity_1hour",
    "geo_anomaly_flag",
    "new_device_flag",
    "balance_mismatch_flag",
    "amount",
    "rule_signal_score",
]

LABEL_COL = "is_fraud"


def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Select and clean features for model training.

    Feature selection rationale:
    - merchant_risk_score: strongest single predictor in our dataset
    - velocity_5min/1hour: card testing detection
    - geo_anomaly_flag + new_device_flag: account takeover signals
    - balance_mismatch_flag: PaySim-specific fraud pattern
    - amount: large amounts correlate with fraud in PaySim
    - rule_signal_score: combined rule output (meta-feature)

    We deliberately exclude:
    - customer_id, merchant_id: too high cardinality, causes overfitting
    - event_time: temporal leakage risk in small datasets
    - customer_country, merchant_country: encoded in geo_anomaly_flag
    """
    log.info("Engineering features ...")

    # Fill missing values with sensible defaults
    df["merchant_risk_score"]  = df["merchant_risk_score"].fillna(0.1)
    df["velocity_5min"]        = df["velocity_5min"].fillna(0)
    df["velocity_1hour"]       = df["velocity_1hour"].fillna(0)
    df["geo_anomaly_flag"]     = df["geo_anomaly_flag"].fillna(0)
    df["new_device_flag"]      = df["new_device_flag"].fillna(0)
    df["balance_mismatch_flag"]= df["balance_mismatch_flag"].fillna(0)
    df["amount"]               = df["amount"].fillna(0)
    df["rule_signal_score"]    = df["rule_signal_score"].fillna(0)
    df[LABEL_COL]              = df[LABEL_COL].fillna(0)

    # Select features and label
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        log.warning("Missing features (will use zeros): %s", missing)
        for f in missing:
            df[f] = 0

    X = df[FEATURE_COLS].astype(float)
    y = df[LABEL_COL].astype(int)

    log.info("Features: %s", FEATURE_COLS)
    log.info("Training samples: %d (fraud: %d, legitimate: %d)",
             len(y), y.sum(), (y == 0).sum())

    return X, y


def prepare_training_csv(X: pd.DataFrame, y: pd.Series) -> tuple:
    # Manual stratified split using numpy — no sklearn needed
    import numpy as np
    rng = np.random.default_rng(42)

    fraud_idx = y[y == 1].index.tolist()
    legit_idx = y[y == 0].index.tolist()

    rng.shuffle(fraud_idx)
    rng.shuffle(legit_idx)

    fraud_split = int(len(fraud_idx) * 0.8)
    legit_split = int(len(legit_idx) * 0.8)

    train_idx = fraud_idx[:fraud_split] + legit_idx[:legit_split]
    val_idx   = fraud_idx[fraud_split:] + legit_idx[legit_split:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    X_train = X.loc[train_idx]
    X_val   = X.loc[val_idx]
    y_train = y.loc[train_idx]
    y_val   = y.loc[val_idx]

    # SageMaker XGBoost CSV format: label first, then features
    train_df = pd.concat(
        [y_train.reset_index(drop=True),
         X_train.reset_index(drop=True)], axis=1
    )
    val_df = pd.concat(
        [y_val.reset_index(drop=True),
         X_val.reset_index(drop=True)], axis=1
    )

    log.info("Train: %d rows | Validation: %d rows", len(train_df), len(val_df))
    return train_df, val_df


def upload_training_data(s3_client, train_df, val_df) -> tuple:
    """Upload training CSV files to S3 for SageMaker."""
    train_key = f"{MODEL_PREFIX}/input/train/train.csv"
    val_key   = f"{MODEL_PREFIX}/input/validation/validation.csv"

    # Upload train
    buf = io.BytesIO()
    train_df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=train_key, Body=buf.getvalue())
    log.info("Uploaded training data: s3://%s/%s", S3_BUCKET, train_key)

    # Upload validation
    buf = io.BytesIO()
    val_df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=val_key, Body=buf.getvalue())
    log.info("Uploaded validation data: s3://%s/%s", S3_BUCKET, val_key)

    train_s3 = f"s3://{S3_BUCKET}/{MODEL_PREFIX}/input/train"
    val_s3   = f"s3://{S3_BUCKET}/{MODEL_PREFIX}/input/validation"
    return train_s3, val_s3


# ─────────────────────────────────────────────────────────────────
# STEP 3: TRAIN XGBOOST MODEL ON SAGEMAKER
# ─────────────────────────────────────────────────────────────────

def get_xgboost_image_uri() -> str:
    """Get the SageMaker built-in XGBoost container URI for us-east-1."""
    # Hardcoded for us-east-1 — no sagemaker SDK needed
    return "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"


def train_model(sm_client, train_s3: str, val_s3: str) -> str:
    """
    Launch a SageMaker Training Job using the built-in XGBoost algorithm.

    Why XGBoost for fraud detection?
    - Handles class imbalance well (scale_pos_weight parameter)
    - Fast training on tabular data
    - Interpretable feature importance
    - Industry standard for fraud detection at Stripe, Square, PayPal

    Hyperparameters:
    - max_depth=5: prevents overfitting on small dataset
    - n_estimators=100: enough trees for good performance
    - scale_pos_weight: handles class imbalance
      (legitimate:fraud ratio = ~130:1 in our dataset)
    - eval_metric=auc: AUC is the right metric for imbalanced fraud data
      (accuracy would be misleading — a model predicting all legitimate
       gets 99.3% accuracy but catches zero fraud)
    """
    job_name = f"fraud-xgboost-{int(time.time())}"
    image_uri = get_xgboost_image_uri()

    # Calculate scale_pos_weight from our dataset stats
    # ~0.68% fraud rate → ~130 legitimate per fraud transaction
    scale_pos_weight = 130

    log.info("Starting SageMaker training job: %s", job_name)
    log.info("  Algorithm : XGBoost 1.7-1")
    log.info("  Instance  : ml.m5.large (~$0.02 for this job)")
    log.info("  Features  : %d", len(FEATURE_COLS))

    sm_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage":     image_uri,
            "TrainingInputMode": "File",
        },
        RoleArn=SAGEMAKER_ROLE,
        InputDataConfig=[
            {
                "ChannelName":     "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType":             "S3Prefix",
                        "S3Uri":                  train_s3,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            },
            {
                "ChannelName":     "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType":             "S3Prefix",
                        "S3Uri":                  val_s3,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            },
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{S3_BUCKET}/{MODEL_PREFIX}/output/",
        },
        ResourceConfig={
            "InstanceType":   "ml.m5.large",
            "InstanceCount":  1,
            "VolumeSizeInGB": 5,
        },
        HyperParameters={
            "max_depth":         "5",
            "eta":               "0.2",
            "gamma":             "4",
            "min_child_weight":  "6",
            "subsample":         "0.8",
            "objective":         "binary:logistic",
            "num_round":         "100",
            "scale_pos_weight":  str(scale_pos_weight),
            "eval_metric":       "auc",
        },
        StoppingCondition={"MaxRuntimeInSeconds": 900},  # 15 min max
    )

    # Poll until complete
    log.info("Waiting for training job to complete (~5-10 minutes) ...")
    while True:
        resp   = sm_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        elapsed = resp.get("TrainingTimeInSeconds", 0)
        log.info("  Status: %-12s | Elapsed: %ds", status, elapsed)

        if status == "Completed":
            model_s3 = resp["ModelArtifacts"]["S3ModelArtifacts"]
            log.info("Training complete!")
            log.info("  Model artifacts: %s", model_s3)
            if "FinalMetricDataList" in resp:
                for m in resp["FinalMetricDataList"]:
                    log.info("  %s: %.4f", m["MetricName"], m["Value"])
            return job_name, model_s3

        if status in ("Failed", "Stopped"):
            reason = resp.get("FailureReason", "Unknown")
            raise RuntimeError(f"Training job {status}: {reason}")

        time.sleep(30)


# ─────────────────────────────────────────────────────────────────
# STEP 4: CREATE MODEL + SERVERLESS ENDPOINT
# ─────────────────────────────────────────────────────────────────

def create_sagemaker_model(sm_client, job_name: str, model_s3: str) -> str:
    """Register the trained model in SageMaker."""
    model_name = f"fraud-model-{int(time.time())}"
    image_uri  = get_xgboost_image_uri()

    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image":          image_uri,
            "ModelDataUrl":   model_s3,
            "Environment": {
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_PROGRAM":              "xgboost_scoring.py",
            },
        },
        ExecutionRoleArn=SAGEMAKER_ROLE,
    )
    log.info("Created SageMaker model: %s", model_name)
    return model_name


def deploy_serverless_endpoint(sm_client, model_name: str) -> str:
    """
    Deploy model as a Serverless Inference endpoint.

    Serverless vs Real-time:
    - Real-time: always-on, $0.046/hr, <100ms latency
    - Serverless: scales to zero, $0 idle, 1-3s cold start

    For fraud detection in production you'd want real-time.
    For a portfolio project, serverless is correct — zero idle cost.

    MemorySizeInMB=1024: enough for XGBoost inference
    MaxConcurrency=5: up to 5 simultaneous invocations
    """
    config_name = f"fraud-serverless-config-{int(time.time())}"

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName":    "AllTraffic",
                "ModelName":      model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": 1024,
                    "MaxConcurrency": 5,
                },
            }
        ],
    )
    log.info("Created endpoint config: %s", config_name)

    # Create or update endpoint
    try:
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
        log.info("Creating endpoint: %s ...", ENDPOINT_NAME)
    except sm_client.exceptions.ResourceInUse:
        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
        log.info("Updating endpoint: %s ...", ENDPOINT_NAME)

    # Wait for endpoint to be InService
    log.info("Waiting for endpoint (2-5 minutes) ...")
    while True:
        resp   = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        log.info("  Endpoint status: %s", status)

        if status == "InService":
            log.info("Endpoint is InService!")
            return ENDPOINT_NAME
        if status == "Failed":
            reason = resp.get("FailureReason", "Unknown")
            raise RuntimeError(f"Endpoint failed: {reason}")
        time.sleep(30)


# ─────────────────────────────────────────────────────────────────
# STEP 5: TEST ENDPOINT
# ─────────────────────────────────────────────────────────────────

def test_endpoint(sm_runtime):
    """
    Send test transactions to the SageMaker endpoint.
    Feature order must match training: FEATURE_COLS order.
    merchant_risk_score, velocity_5min, velocity_1hour,
    geo_anomaly_flag, new_device_flag, balance_mismatch_flag,
    amount, rule_signal_score
    """
    log.info("")
    log.info("Testing SageMaker endpoint ...")

    test_cases = [
        {
            "name":    "Normal transaction",
            "features": "0.15,0,1,0,0,0,1500.00,0",
            "expected": "low fraud probability",
        },
        {
            "name":    "High-risk merchant + new device",
            "features": "0.92,2,4,0,1,0,85000.00,5",
            "expected": "high fraud probability",
        },
        {
            "name":    "All signals — confirmed fraud",
            "features": "0.88,0,1,1,1,1,450000.00,14",
            "expected": "very high fraud probability",
        },
    ]

    for case in test_cases:
        resp = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=case["features"],
        )
        score = float(resp["Body"].read().decode("utf-8").strip())
        log.info("  %-40s → score=%.4f (%s)",
                 case["name"], score, case["expected"])


# ─────────────────────────────────────────────────────────────────
# STEP 6: UPDATE LAMBDA TO CALL SAGEMAKER
# ─────────────────────────────────────────────────────────────────

def update_lambda_with_sagemaker(lambda_client):
    """
    Add SAGEMAKER_ENDPOINT_NAME to Lambda environment variables.
    The Lambda function already has the SageMaker call logic
    ready to activate when this env var is set.
    """
    resp = lambda_client.get_function_configuration(
        FunctionName=FUNCTION_NAME
    )
    current_env = resp["Environment"]["Variables"]
    current_env["SAGEMAKER_ENDPOINT_NAME"] = ENDPOINT_NAME
    current_env["SAGEMAKER_SCORE_THRESHOLD"] = "0.5"

    lambda_client.update_function_configuration(
        FunctionName=FUNCTION_NAME,
        Environment={"Variables": current_env},
    )
    log.info("Lambda updated with SageMaker endpoint: %s", ENDPOINT_NAME)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    s3_client  = boto3.client("s3",        region_name=REGION)
    sm_client  = boto3.client("sagemaker", region_name=REGION)
    sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    lam_client = boto3.client("lambda",    region_name=REGION)

    log.info("=" * 60)
    log.info("Fraud Platform — SageMaker Fraud Scoring Pipeline")
    log.info("  Algorithm : XGBoost (built-in)")
    log.info("  Endpoint  : Serverless Inference (zero idle cost)")
    log.info("  Estimated cost: ~$0.05-0.10 total")
    log.info("=" * 60)

    # try:
    #     # Install sklearn if needed
    #     import sklearn
    # except ImportError:
    #     log.info("Installing scikit-learn ...")
    #     os.system("pip install scikit-learn --quiet")

    # 1. Load data
    df = load_training_data(s3_client)

    # 2. Feature engineering
    X, y = engineer_features(df)

    # 3. Prepare and upload training data
    train_df, val_df = prepare_training_csv(X, y)
    train_s3, val_s3 = upload_training_data(s3_client, train_df, val_df)

    # 4. Train model
    log.info("")
    log.info("Step 1/4: Training XGBoost model on SageMaker ...")
    job_name, model_s3 = train_model(sm_client, train_s3, val_s3)

    # 5. Create model
    log.info("")
    log.info("Step 2/4: Registering model in SageMaker ...")
    model_name = create_sagemaker_model(sm_client, job_name, model_s3)

    # 6. Deploy serverless endpoint
    log.info("")
    log.info("Step 3/4: Deploying Serverless Inference endpoint ...")
    deploy_serverless_endpoint(sm_client, model_name)

    # 7. Test endpoint
    log.info("")
    log.info("Step 4/4: Testing endpoint with sample transactions ...")
    test_endpoint(sm_runtime)

    # 8. Update Lambda
    log.info("")
    log.info("Updating Lambda with SageMaker endpoint name ...")
    update_lambda_with_sagemaker(lam_client)

    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("SageMaker deployment complete")
    log.info("  Training job    : %s", job_name)
    log.info("  Model artifacts : %s", model_s3)
    log.info("  Endpoint        : %s", ENDPOINT_NAME)
    log.info("  Endpoint type   : Serverless (zero idle cost)")
    log.info("")
    log.info("Console links:")
    log.info("  Training: https://console.aws.amazon.com/sagemaker/home?"
             "region=%s#/jobs", REGION)
    log.info("  Endpoint: https://console.aws.amazon.com/sagemaker/home?"
             "region=%s#/endpoints", REGION)
    log.info("")
    log.info("CLEANUP COMMAND (run when done):")
    log.info("  aws sagemaker delete-endpoint "
             "--endpoint-name %s --region %s", ENDPOINT_NAME, REGION)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
