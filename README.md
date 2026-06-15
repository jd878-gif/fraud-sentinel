# FraudSentinel 🛡️
### Real-Time Financial Fraud Detection Platform on AWS

[![AWS](https://img.shields.io/badge/AWS-10%20Services-orange?logo=amazon-aws)](https://aws.amazon.com)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0.2-green)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

FraudSentinel is a production-grade, end-to-end fraud detection platform built on AWS that processes financial transactions in real-time, applies rule-based and ML-based fraud scoring, orchestrates batch ETL pipelines, and delivers structured fraud alerts via email — all for under $0.15 in compute cost.

> Built as a portfolio project demonstrating production data engineering patterns: event-driven architecture, Medallion data lake design, serverless ML inference, and automated pipeline orchestration.

---

## Architecture

```
PaySim Dataset (6.36M transactions)
         │
         ▼
┌─────────────────────────────────┐
│   Enhancement Pipeline          │
│   Python · Faker · NumPy        │
│   507,500 rows · 39 columns     │
│   5 fraud signal types          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Amazon SQS                    │
│   fraud-transactions-dev        │
│   DLQ · Alerts Queue            │
│   At-least-once delivery        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   AWS Lambda                    │
│   fraud-transaction-processor   │
│   4 fraud detection rules       │
│   P99 latency: 761ms            │
│   DynamoDB feature store writes │
│   S3 Bronze layer writes        │
└──────┬──────────────┬───────────┘
       │              │
       ▼              ▼
┌────────────┐  ┌─────────────────┐
│  DynamoDB  │  │   Amazon SNS    │
│  3 Tables  │  │  fraud-alerts   │
│  TTL-based │  │  Email in <30s  │
└────────────┘  └─────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Amazon S3                     │
│   fraud-platform-jeet-dev       │
│   Bronze → Silver → Gold        │
│   Medallion Architecture        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   AWS Glue ETL (2 Jobs)         │
│   Bronze → Silver               │
│   Validates · Deduplicates      │
│   Writes Parquet · Partitioned  │
│                                 │
│   Silver → Gold                 │
│   5 Aggregation Tables          │
│   Fraud by Merchant/Segment/Hr  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   AWS Glue Data Catalog         │
│   6 Tables Registered           │
│   Auto-discovered by Crawlers   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Amazon Athena                 │
│   SQL on Gold Parquet           │
│   Fraud analytics queries       │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│   AWS Step Functions            │
│   FraudPlatform-ETL-Pipeline    │
│   Bronze→Silver→Gold sequence   │
│   Catch/Retry · SNS on failure  │
│   EventBridge: Every Sunday 2AM │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│   Amazon SageMaker              │
│   XGBoost Fraud Classifier      │
│   Serverless Inference          │
│   AUC: 1.00 · F1: 1.00         │
│   8 fraud signal features       │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│   Amazon CloudWatch             │
│   FraudPlatform-Overview        │
│   6 panels · DLQ alarm          │
│   P50/P95/P99 latency tracking  │
└─────────────────────────────────┘
```

---

## AWS Services

| Service | Role | Key Detail |
|---|---|---|
| **Amazon SQS** | Transaction ingestion queue | At-least-once delivery, DLQ after 3 failures, 20s long polling |
| **AWS Lambda** | Real-time fraud processor | 4 detection rules, P99 761ms, SQS-triggered, partial batch failure |
| **Amazon DynamoDB** | Feature store + idempotency | 3 tables, TTL-based expiry, velocity counters per customer |
| **Amazon S3** | Medallion data lake | Bronze (JSON) → Silver (Parquet) → Gold (aggregations) |
| **AWS Glue** | ETL pipeline | 2 PySpark jobs, date-partitioned Parquet, Glue Schema Registry |
| **AWS Glue Crawlers** | Schema discovery | Auto-registers Silver and Gold tables in Data Catalog |
| **Amazon Athena** | SQL analytics | Queries Gold Parquet, partition pruning, $0.00 on 1,000 records |
| **AWS Step Functions** | Pipeline orchestration | Visual workflow, Catch/Retry, weekly EventBridge trigger |
| **Amazon SNS** | Fraud alert delivery | Structured JSON alerts to Gmail in under 30 seconds |
| **Amazon SageMaker** | ML fraud scoring | XGBoost, Serverless Inference, zero idle cost |
| **Amazon CloudWatch** | Monitoring | 6-panel dashboard, DLQ depth alarm, P50/P95/P99 |
| **Amazon EventBridge** | Scheduling | `cron(0 2 ? * SUN *)` — weekly ETL trigger |

---

## Dataset

Built on the **PaySim** mobile money simulation dataset (6.36M transactions) enhanced with a custom Python pipeline to behave like production payment data.

### Enhancement Pipeline (`enhancement_pipeline.py`)

| Feature Added | Description |
|---|---|
| `event_id` | UUID-based unique identifier per transaction |
| `event_time` / `arrival_time` | ISO 8601 wall-clock timestamps (2% late arrivals simulated) |
| `schema_version` | v1/v2 evolution — `ip_risk_score` added in v2 |
| `transaction_velocity_5min` | Sliding window counter per customer (card testing detection) |
| `transaction_velocity_1hour` | 1-hour velocity window |
| `geo_anomaly_flag` | Impossible travel detection (cross-country in <30 min) |
| `new_device_flag` | Device age < 2 days (account takeover signal) |
| `merchant_risk_score` | 0.0–1.0 risk score, 10% merchants flagged HIGH |
| `duplicate_event_flag` | 1.5% duplicates injected (at-least-once delivery simulation) |
| `late_arrival_flag` | 2% late records (network jitter simulation) |
| `chargeback_status` | 85% of fraud rows get Filed chargeback |
| `traffic_period` | Payday / Holiday / Normal seasonality tags |

**Dimension tables:** `customers.csv` · `merchants.csv` · `devices.csv` · `chargebacks.csv`

**Intentional data quality issues injected:** duplicate events, late arrivals, out-of-order records, missing values, corrupted records, schema evolution, concept drift.

---

## Fraud Detection

### Rule-Based Engine (Lambda)

Four real-time rules evaluated on every transaction:

| Rule | Signal | Threshold | Risk Score |
|---|---|---|---|
| `HIGH_VELOCITY_5MIN` | Card testing — many small transactions | velocity_5min ≥ 5 | +4 |
| `HIGH_RISK_MERCHANT` | Suspicious merchant | merchant_risk_score ≥ 0.75 | +3 |
| `GEO_ANOMALY_NEW_DEVICE` | Account takeover | geo_anomaly=1 AND new_device=1 | +5 |
| `BALANCE_MISMATCH_TRANSFER` | Fraudulent ledger entry | TRANSFER with unchanged balance | +4 |
| `PAYSIM_FRAUD_LABEL` | Ground truth label | isFraud=1 | +2 |

Max risk score: **14** (all rules triggered simultaneously)

### ML Scoring (SageMaker)

- **Algorithm:** XGBoost (built-in SageMaker container)
- **Features:** 8 fraud signal features
- **Training:** Local XGBoost 3.0.2, 802 training samples
- **Deployment:** SageMaker Serverless Inference (zero idle cost)

**Model performance:**
```
Training AUC    : 1.0000
Validation AUC  : 1.0000
Precision       : 1.000
Recall          : 1.000
F1 Score        : 1.000
```

**Feature importance (gain):**
```
merchant_risk_score   89.97  ← strongest fraud predictor
amount                84.63
geo_anomaly_flag      71.14
velocity_1hour        41.33
rule_signal_score     40.59
```

**Sample scores:**
```
Normal transaction              → 0.0227  ✓ CLEAN
High-risk merchant + new device → 0.0534  ✓ CLEAN
All signals — confirmed fraud   → 0.8836  🚨 FRAUD
```

---

## Medallion Architecture

```
Bronze (Raw JSON)           Silver (Validated Parquet)      Gold (Aggregated Parquet)
─────────────────           ──────────────────────────      ─────────────────────────
1,000 JSON files            Deduplicated                    gold_fraud_by_merchant
Per-event granularity       Type-cast                       gold_fraud_by_segment
Date-partitioned            Corrupted → quarantine          gold_fraud_by_hour
                            balance_mismatch_flag added     gold_high_risk_customers
                            rule_signal_score added         gold_pipeline_run_summary
                            Date-partitioned Parquet
```

### Athena Analytics Results

```sql
SELECT merchant_category, fraud_rate_pct, total_transactions
FROM fraud_platform_dev.gold_fraud_by_merchant
ORDER BY fraud_rate_pct DESC;
```

| Merchant Category | Fraud Rate | Transactions |
|---|---|---|
| Food | 0.88% | 228 |
| Travel | 0.75% | 134 |
| Entertainment | 0.00% | 108 |
| Electronics | 0.00% | 99 |
| Retail | 0.00% | 230 |

```sql
SELECT customer_segment, fraud_rate_pct
FROM fraud_platform_dev.gold_fraud_by_segment
ORDER BY fraud_rate_pct DESC;
```

| Customer Segment | Fraud Rate |
|---|---|
| Traveler | 0.65% |
| High Spender | 0.63% |
| Family | 0.41% |
| Business | 0.00% |

---

## Pipeline Orchestration

**Step Functions state machine** (`FraudPlatform-ETL-Pipeline`):

```
Start
  │
  ▼
RunBronzeToSilver ──(on error)──► NotifyFailure ──► PipelineFailed
  │
  ▼ (on success)
RunSilverToGold ───(on error)──► NotifyFailure ──► PipelineFailed
  │
  ▼ (on success)
NotifySuccess
  │
  ▼
PipelineSucceeded
```

- Triggered weekly by EventBridge: `cron(0 2 ? * SUN *)`
- Each Glue state uses `.sync` integration — waits for completion
- Auto-retry on failure: 2 attempts with 2x backoff
- SNS notification on both success and failure

---

## Key Engineering Decisions

### SQS vs Kinesis
Used SQS Standard (permanent free tier) instead of Kinesis ($11/month/shard). Key tradeoff: Kinesis preserves per-shard ordering which matters for accurate velocity computation — all events for one customer land on the same shard. SQS Standard does not guarantee ordering. In production I would use Kinesis with customer ID as the partition key. The producer code is endpoint-agnostic — switching requires changing one configuration value.

### DynamoDB for Velocity Feature Store
Per-customer velocity counters (5-min, 1-hour) stored as DynamoDB items with TTL-based auto-expiry. Alternatives considered: Redis (requires ElastiCache, adds cost and operational overhead), in-memory (doesn't survive Lambda restarts). DynamoDB provides single-digit millisecond reads at any scale with no server to manage.

### Idempotency Pattern
SQS at-least-once delivery means the same message can arrive twice. DynamoDB dedup table stores processed `event_id` values with 24-hour TTL. Consumer checks before processing: if exists → skip. Conditional writes prevent race conditions between concurrent Lambda invocations.

### Serverless Inference vs Real-Time Endpoint
SageMaker real-time endpoint: $0.046/hour = $33/month even with zero traffic. Serverless: $0 idle, pay per invocation, 150,000 free invocations/month. For fraud detection in production you'd want real-time (sub-100ms latency). For a portfolio project and low-traffic deployments, serverless is the correct architectural choice.

### Medallion Architecture
Bronze (raw JSON) → Silver (validated Parquet) → Gold (aggregated) follows the Delta Lake / Databricks Medallion pattern used at Airbnb, Netflix, and Walmart. Each layer serves a different consumer: Bronze for audit/replay, Silver for ML training, Gold for business dashboards. Date partitioning on Silver enables Athena partition pruning — only scans the relevant date folder rather than the full dataset.

---

## Project Structure

```
fraud-sentinel/
│
├── enhancement_pipeline.py        # PaySim → production dataset
├── aws_clients.py                 # Boto3 client factory (LocalStack/AWS toggle)
├── bootstrap.py                   # Creates all LocalStack resources
├── producer_sqs.py                # CSV → SQS producer
├── consumer_sqs.py                # SQS consumer (local dev)
├── lambda_function.py             # AWS Lambda fraud processor
├── deploy_lambda.py               # Lambda deployment script
├── glue_bronze_to_silver.py       # Glue ETL Job 1
├── glue_silver_to_gold.py         # Glue ETL Job 2
├── deploy_and_run_glue.py         # Glue job deployment + execution
├── deploy_step_functions.py       # Step Functions + EventBridge setup
├── sagemaker_local_train_deploy.py # XGBoost train + SageMaker deploy
├── upload_bronze_to_aws.py        # LocalStack → real AWS S3 migration
├── verify_pipeline.py             # End-to-end health check
├── docker-compose.yml             # LocalStack for local development
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Results

| Metric | Value |
|---|---|
| Transactions processed | 507,500 |
| Flagging rate (5,000 msg run) | 7.7% |
| Food category fraud rate | 0.88% (highest) |
| Traveler segment fraud rate | 0.65% (highest) |
| Lambda P99 latency | 761ms |
| Lambda cold start | ~642ms |
| Lambda memory used | 101-103 MB / 256 MB |
| DLQ failures | 0 across all runs |
| SNS alert delivery | < 30 seconds |
| Glue Bronze→Silver | 456 seconds, $0.11 |
| Glue Silver→Gold | 89 seconds, $0.02 |
| SageMaker training AUC | 1.0000 |
| SageMaker F1 score | 1.0000 |
| **Total AWS compute cost** | **~$0.15** |

---

## Local Development Setup

### Prerequisites
- Python 3.9+
- Docker Desktop

### Quick Start

```bash
# Clone the repository
git clone https://github.com/jd878-gif/fraud-sentinel.git
cd fraud-sentinel

# Install dependencies
pip install boto3 pandas numpy faker xgboost requests

# Start LocalStack (free AWS emulation)
docker-compose up -d

# Bootstrap local AWS resources
python bootstrap.py

# Run producer (sends transactions to local SQS)
python producer_sqs.py --max-rows 1000

# Run consumer (processes transactions locally)
python consumer_sqs.py --max-messages 1000

# Verify pipeline health
python verify_pipeline.py
```

### Deploy to Real AWS

```bash
# Configure AWS credentials
aws configure

# Upload data to real S3
python upload_bronze_to_aws.py

# Deploy Lambda
python deploy_lambda.py

# Deploy Glue ETL + run pipeline
python deploy_and_run_glue.py

# Deploy Step Functions orchestration
python deploy_step_functions.py

# Train and deploy SageMaker model
python sagemaker_local_train_deploy.py
```

---

## Cost Breakdown

| Component | Cost |
|---|---|
| SQS (5,000+ messages) | $0.00 (free tier) |
| Lambda (all invocations) | $0.00 (free tier) |
| DynamoDB (all operations) | $0.00 (free tier) |
| S3 storage (~15 MB) | ~$0.01/month |
| Glue ETL (2 jobs) | $0.13 (one-time run) |
| Athena queries | $0.00 (< 10 MB scanned) |
| SNS notifications | $0.00 (free tier) |
| CloudWatch | $0.00 (free tier) |
| Step Functions | $0.00 (free tier) |
| SageMaker Serverless | $0.00 (free tier) |
| **Total** | **~$0.15** |

---

## Scalability Considerations

### At 10× Scale (5M transactions/day)
- Replace SQS with **Kinesis Data Streams** (10 shards, customer ID as partition key for per-customer ordering)
- Move velocity counters from in-memory Python dicts to **DynamoDB atomic counters**
- Scale Glue to **10 DPUs** for Silver/Gold processing
- Add **SageMaker real-time endpoint** (vs serverless) for sub-100ms inference latency

### At 100× Scale (50M transactions/day)
- **Kinesis Enhanced Fan-Out** for Lambda consumers (dedicated 2 MB/s per consumer)
- **DynamoDB DAX** for microsecond feature store reads
- **Apache Iceberg** format on S3 for ACID transactions and time-travel queries
- Separate **feature engineering service** (ECS/Fargate) from fraud scoring service

### At 1000× Scale (500M transactions/day — Stripe/PayPal tier)
- Multi-region active-active deployment
- **Apache Kafka** (MSK) instead of Kinesis for cross-region replication
- **Feature store** (SageMaker Feature Store) with online + offline stores
- **Model ensemble** (XGBoost + neural network) with A/B testing via SageMaker
- **Real-time data quality monitoring** with Great Expectations + CloudWatch

---

## Interview Talking Points

**"Why SQS instead of Kinesis?"**
> Kinesis preserves per-shard ordering, which matters for accurate velocity computation — all events for the same customer land on the same shard. SQS Standard doesn't guarantee ordering. I used SQS for cost reasons (permanent free tier vs $11/month/shard for Kinesis) and documented this tradeoff explicitly. The producer code is endpoint-agnostic — switching requires one configuration change.

**"What happens if a Lambda batch partially fails?"**
> SQS + Lambda supports partial batch failure reporting. Lambda returns `{"batchItemFailures": [{"itemIdentifier": failed_message_id}]}`. SQS only retries the failed messages, not the whole batch. After 3 failures, the message goes to the DLQ. I have a CloudWatch alarm on DLQ depth that fires SNS if any message hits the DLQ.

**"How does the idempotency work?"**
> SQS at-least-once delivery means the same message can arrive twice. Before processing each message, Lambda checks DynamoDB for the event_id. If it exists, we skip. After successful processing, we write the event_id with a 24-hour TTL. DynamoDB conditional writes prevent race conditions between concurrent Lambda instances.

**"Why Medallion architecture?"**
> Bronze preserves the raw data exactly as received — critical for audit trails and pipeline replay. Silver is where cleaning happens: deduplication, type casting, quarantining corrupted records. Gold is business-ready aggregations that answer specific questions without touching raw data. This separation means a bug in Gold only requires rerunning one Glue job, not reprocessing raw data.

---

## Tech Stack

**Languages:** Python 3.12  
**ML:** XGBoost 3.0.2  
**Data:** Pandas, NumPy, PyArrow  
**AWS:** SQS · Lambda · DynamoDB · S3 · Glue · Athena · Step Functions · SNS · SageMaker · CloudWatch · EventBridge  
**Local Dev:** LocalStack · Docker  
**Data Source:** PaySim (Kaggle) — enhanced with custom pipeline  

---

## Author

**Jeet Dave**  
MS Data Science · New Jersey Institute of Technology (Expected May 2027)  
GitHub: [@jd878-gif](https://github.com/jd878-gif)

---


