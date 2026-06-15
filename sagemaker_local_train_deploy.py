"""
sagemaker_local_train_deploy.py
================================
Trains XGBoost fraud classifier locally, packages the model,
uploads to S3, and deploys as a SageMaker Serverless endpoint.

Why train locally instead of SageMaker Training Job?
  Free AWS accounts have zero quota for SageMaker training instances.
  Training locally and deploying to SageMaker for serving is a valid
  production pattern — many teams train on EC2/local and deploy to
  SageMaker endpoints for managed, auto-scaling inference.

  In production at scale you'd use SageMaker Training Jobs for:
  - Reproducible training runs tracked in Model Registry
  - Distributed training across multiple instances
  - Automatic hyperparameter tuning (SageMaker HyperParameter Tuning)

  For a portfolio project, local training + SageMaker serving is
  architecturally correct and demonstrates the key skill: integrating
  a trained model into a real-time AWS inference pipeline.

Pipeline:
  1. Load Silver Parquet from S3
  2. Feature engineering (8 fraud signal features)
  3. Train XGBoost locally (takes ~10 seconds on 1,000 rows)
  4. Evaluate model (AUC, precision, recall, confusion matrix)
  5. Package model as tar.gz (SageMaker format)
  6. Upload model artifacts to S3
  7. Register model in SageMaker
  8. Deploy Serverless Inference endpoint
  9. Test with 3 sample transactions
  10. Print cleanup command
"""

import boto3
import io
import json
import logging
import os
import pickle
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("sagemaker_deploy")

# ── Config ─────────────────────────────────────────────────────────
REGION        = "us-east-1"
ACCOUNT_ID    = "621402808508"
S3_BUCKET     = "fraud-platform-jeet-dev"
MODEL_PREFIX  = "fraud-model"
ENDPOINT_NAME = "fraud-scoring-serverless"
FUNCTION_NAME = "fraud-transaction-processor"
SAGEMAKER_ROLE= f"arn:aws:iam::{ACCOUNT_ID}:role/FraudPlatformGlueRole"

# XGBoost container URI for us-east-1
XGBOOST_IMAGE = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"

# Feature columns — must match order used during training
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


# ─────────────────────────────────────────────────────────────────
# STEP 1: LOAD SILVER DATA FROM S3
# ─────────────────────────────────────────────────────────────────

def load_silver_data(s3_client) -> pd.DataFrame:
    log.info("Loading Silver layer from S3 ...")
    resp    = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix="silver/transactions/")
    objects = [o for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]

    if not objects:
        raise ValueError("No Silver Parquet files found. Run Glue ETL first.")

    dfs = []
    for obj in objects:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
        dfs.append(pd.read_parquet(io.BytesIO(response["Body"].read())))

    df = pd.concat(dfs, ignore_index=True)
    log.info("  Loaded: %d rows, %d columns", len(df), len(df.columns))
    log.info("  Fraud rate: %.2f%% (%d fraud / %d total)",
             df[LABEL_COL].mean() * 100, df[LABEL_COL].sum(), len(df))
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Select and clean the 8 fraud signal features.
    Returns X (feature matrix) and y (labels) as numpy arrays.
    """
    log.info("Engineering features ...")

    # Fill missing values
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    X = df[FEATURE_COLS].astype(float).values
    y = df[LABEL_COL].astype(int).values

    log.info("  Feature matrix: %s", X.shape)
    log.info("  Class distribution: %d fraud, %d legitimate",
             y.sum(), (y == 0).sum())
    return X, y


def stratified_split(X, y, test_size=0.2, seed=42):
    """
    Manual stratified train/test split using numpy.
    Ensures both splits have the same fraud rate.
    """
    rng = np.random.default_rng(seed)

    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]

    rng.shuffle(fraud_idx)
    rng.shuffle(legit_idx)

    n_fraud_val = max(1, int(len(fraud_idx) * test_size))
    n_legit_val = int(len(legit_idx) * test_size)

    val_idx   = np.concatenate([fraud_idx[:n_fraud_val], legit_idx[:n_legit_val]])
    train_idx = np.concatenate([fraud_idx[n_fraud_val:], legit_idx[n_legit_val:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (X[train_idx], X[val_idx],
            y[train_idx], y[val_idx])


# ─────────────────────────────────────────────────────────────────
# STEP 3: TRAIN XGBOOST LOCALLY
# ─────────────────────────────────────────────────────────────────

def train_xgboost(X_train, X_val, y_train, y_val) -> xgb.Booster:
    """
    Train XGBoost fraud classifier.

    Key hyperparameters for fraud detection:
    - scale_pos_weight: compensates for class imbalance
      (if 1% fraud → scale_pos_weight ≈ 99)
    - eval_metric=auc: AUC handles imbalance better than accuracy
      (a model predicting all legitimate gets 99% accuracy but
       catches zero fraud — AUC reveals this)
    - max_depth=4: shallow trees prevent overfitting on small dataset
    - subsample=0.8: row sampling reduces overfitting
    """
    fraud_count = y_train.sum()
    legit_count = (y_train == 0).sum()
    scale_pos_weight = legit_count / max(fraud_count, 1)

    log.info("Training XGBoost locally ...")
    log.info("  scale_pos_weight: %.1f (handles %.1f:1 class imbalance)",
             scale_pos_weight, scale_pos_weight)

    params = {
        "max_depth":         4,
        "eta":               0.2,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "objective":         "binary:logistic",
        "eval_metric":       "auc",
        "scale_pos_weight":  scale_pos_weight,
        "seed":              42,
        "verbosity":         0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train,
                          feature_names=FEATURE_COLS)
    dval   = xgb.DMatrix(X_val,   label=y_val,
                          feature_names=FEATURE_COLS)

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        verbose_eval=False,
        early_stopping_rounds=15,
    )

    train_auc = evals_result["train"]["auc"][-1]
    val_auc   = evals_result["val"]["auc"][-1]
    log.info("  Training AUC  : %.4f", train_auc)
    log.info("  Validation AUC: %.4f", val_auc)

    # Feature importance
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    log.info("  Feature importance (gain):")
    for feat, score in sorted_imp[:5]:
        log.info("    %-30s : %.2f", feat, score)

    return model


def evaluate_model(model, X_val, y_val):
    """
    Compute precision, recall, and confusion matrix at threshold 0.5.
    AUC is the primary metric but precision/recall tell the
    operational story: how many alerts are real fraud?
    """
    dval   = xgb.DMatrix(X_val, feature_names=FEATURE_COLS)
    scores = model.predict(dval)
    preds  = (scores >= 0.5).astype(int)

    tp = int(((preds == 1) & (y_val == 1)).sum())
    fp = int(((preds == 1) & (y_val == 0)).sum())
    tn = int(((preds == 0) & (y_val == 0)).sum())
    fn = int(((preds == 0) & (y_val == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 0.001)

    log.info("  Confusion matrix (threshold=0.5):")
    log.info("    TP=%d  FP=%d  TN=%d  FN=%d", tp, fp, tn, fn)
    log.info("  Precision : %.3f  (of flagged transactions, %.0f%% are real fraud)",
             precision, precision * 100)
    log.info("  Recall    : %.3f  (caught %.0f%% of actual fraud)",
             recall, recall * 100)
    log.info("  F1 Score  : %.3f", f1)


# ─────────────────────────────────────────────────────────────────
# STEP 4: PACKAGE AND UPLOAD MODEL TO S3
# ─────────────────────────────────────────────────────────────────

def package_and_upload_model(model: xgb.Booster, s3_client) -> str:
    """
    SageMaker expects model artifacts as a tar.gz file containing
    xgboost-model (the binary model file).

    Package structure:
      model.tar.gz
        └── xgboost-model    ← XGBoost binary format

    Why tar.gz? SageMaker's container extracts this archive and
    loads the model file automatically during endpoint startup.
    """
    log.info("Packaging model for SageMaker ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model in XGBoost binary format
        model_path = os.path.join(tmpdir, "xgboost-model")
        model.save_model(model_path)

        # Create tar.gz archive
        tar_path = os.path.join(tmpdir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_path, arcname="xgboost-model")

        # Upload to S3
        s3_key = f"{MODEL_PREFIX}/output/model.tar.gz"
        s3_client.upload_file(tar_path, S3_BUCKET, s3_key)

    model_s3 = f"s3://{S3_BUCKET}/{s3_key}"
    log.info("  Model uploaded: %s", model_s3)
    return model_s3


# ─────────────────────────────────────────────────────────────────
# STEP 5: REGISTER + DEPLOY SERVERLESS ENDPOINT
# ─────────────────────────────────────────────────────────────────

def create_sagemaker_model(sm_client, model_s3: str) -> str:
    model_name = f"fraud-xgb-{int(time.time())}"
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image":        XGBOOST_IMAGE,
            "ModelDataUrl": model_s3,
        },
        ExecutionRoleArn=SAGEMAKER_ROLE,
    )
    log.info("SageMaker model registered: %s", model_name)
    return model_name


def deploy_serverless_endpoint(sm_client, model_name: str) -> str:
    """
    Deploy as Serverless Inference endpoint.
    Serverless = zero cost when idle, pay per invocation only.
    Free tier: 150,000 invocations/month.

    MemorySizeInMB=1024: XGBoost model is ~50KB, 1GB is plenty
    MaxConcurrency=5: handles bursts of fraud checks
    """
    config_name = f"fraud-serverless-{int(time.time())}"

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName":   model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": 1024,
                "MaxConcurrency": 5,
            },
        }],
    )
    log.info("Endpoint config created: %s", config_name)

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

    # Poll until InService
    log.info("Waiting for endpoint to be InService (2-5 minutes) ...")
    while True:
        resp   = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        log.info("  Endpoint status: %s", status)
        if status == "InService":
            log.info("Endpoint is InService!")
            return ENDPOINT_NAME
        if status in ("Failed", "OutOfService"):
            raise RuntimeError(f"Endpoint failed: {resp.get('FailureReason')}")
        time.sleep(20)


# ─────────────────────────────────────────────────────────────────
# STEP 6: TEST ENDPOINT
# ─────────────────────────────────────────────────────────────────

def test_endpoint(sm_runtime):
    """
    Send 3 test transactions to verify the endpoint works.
    CSV format: features in FEATURE_COLS order, no label.
    """
    log.info("")
    log.info("Testing SageMaker endpoint ...")
    log.info("Feature order: %s", FEATURE_COLS)

    test_cases = [
        # merchant_risk, v5min, v1hr, geo, new_dev, bal_mismatch, amount, signal
        ("Normal transaction",
         "0.15,0,1,0,0,0,1500.00,0",
         "~0.02 expected"),
        ("High-risk merchant + new device",
         "0.92,2,4,0,1,0,85000.00,5",
         "~0.70 expected"),
        ("All signals — confirmed fraud TRANSFER",
         "0.88,0,1,1,1,1,450000.00,14",
         "~0.95 expected"),
    ]

    log.info("")
    for name, features, expected in test_cases:
        resp  = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=features,
        )
        score = float(resp["Body"].read().decode("utf-8").strip())
        flag  = "🚨 FRAUD" if score >= 0.5 else "✓ CLEAN"
        log.info("  %s", name)
        log.info("    Score: %.4f | %s | %s", score, flag, expected)
        log.info("")


# ─────────────────────────────────────────────────────────────────
# STEP 7: UPDATE LAMBDA ENVIRONMENT
# ─────────────────────────────────────────────────────────────────

def update_lambda(lambda_client):
    """Add SageMaker endpoint name to Lambda env vars."""
    try:
        resp = lambda_client.get_function_configuration(
            FunctionName=FUNCTION_NAME
        )
        env = resp["Environment"]["Variables"]
        env["SAGEMAKER_ENDPOINT_NAME"]  = ENDPOINT_NAME
        env["SAGEMAKER_SCORE_THRESHOLD"]= "0.5"
        lambda_client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Environment={"Variables": env},
        )
        log.info("Lambda updated: SAGEMAKER_ENDPOINT_NAME=%s", ENDPOINT_NAME)
    except Exception as e:
        log.warning("Lambda update skipped: %s", e)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    s3_client  = boto3.client("s3",              region_name=REGION)
    sm_client  = boto3.client("sagemaker",       region_name=REGION)
    sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    lam_client = boto3.client("lambda",          region_name=REGION)

    log.info("=" * 60)
    log.info("Fraud Platform — Local Train + SageMaker Deploy")
    log.info("  Training  : Local XGBoost (no AWS quota needed)")
    log.info("  Serving   : SageMaker Serverless Inference")
    log.info("  Idle cost : $0 (serverless scales to zero)")
    log.info("=" * 60)

    # 1. Load data
    df = load_silver_data(s3_client)

    # 2. Features
    X, y = prepare_features(df)

    # 3. Split
    X_train, X_val, y_train, y_val = stratified_split(X, y)
    log.info("Split: %d train, %d validation", len(X_train), len(X_val))

    # 4. Train
    model = train_xgboost(X_train, X_val, y_train, y_val)

    # 5. Evaluate
    log.info("Evaluating model on validation set ...")
    evaluate_model(model, X_val, y_val)

    # 6. Package + upload
    model_s3 = package_and_upload_model(model, s3_client)

    # 7. Register in SageMaker
    log.info("")
    log.info("Registering model in SageMaker ...")
    model_name = create_sagemaker_model(sm_client, model_s3)

    # 8. Deploy serverless endpoint
    log.info("")
    log.info("Deploying Serverless Inference endpoint ...")
    deploy_serverless_endpoint(sm_client, model_name)

    # 9. Test
    test_endpoint(sm_runtime)

    # 10. Update Lambda
    update_lambda(lam_client)

    # Summary
    log.info("=" * 60)
    log.info("SageMaker deployment complete")
    log.info("  Model     : s3://%s/%s/output/model.tar.gz",
             S3_BUCKET, MODEL_PREFIX)
    log.info("  Endpoint  : %s (Serverless)", ENDPOINT_NAME)
    log.info("  Algorithm : XGBoost 3.0.2")
    log.info("  Features  : %d fraud signal features", len(FEATURE_COLS))
    log.info("")
    log.info("Console:")
    log.info("  https://console.aws.amazon.com/sagemaker/home?"
             "region=%s#/endpoints", REGION)
    log.info("")
    log.info("CLEANUP — run this when done:")
    log.info("  aws sagemaker delete-endpoint "
             "--endpoint-name %s --region %s", ENDPOINT_NAME, REGION)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
