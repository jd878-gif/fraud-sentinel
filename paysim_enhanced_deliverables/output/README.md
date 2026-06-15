# PaySim Enhanced — Production-Grade Financial Transaction Dataset

> Transforms the open-source PaySim mobile money simulation into a production-realistic  
> financial transaction dataset designed for building AWS fraud detection platforms.

---

## What Was Built

The raw PaySim dataset contains 6.36 million simulated mobile money transactions with ground-truth fraud labels. This pipeline enriches it to behave like the real-time event stream a payments company such as Stripe, Square, or Mercado Pago would ingest — complete with customer profiles, device fingerprints, behavioral velocity features, geo-anomaly signals, and intentional operational failures that production pipelines must handle.

The result is a five-table dataset that feeds directly into a Kinesis → Lambda → DynamoDB → SageMaker → S3 → Glue fraud detection architecture.

---

## What Changed and Why

### 1. Timestamps (`event_time`, `arrival_time`)

**What:** Each PaySim `step` (1–743) is converted to a real ISO 8601 wall-clock timestamp starting at `2023-01-01T00:00:00`. Arrival time adds realistic network delay.

**Why:** AWS Kinesis, Glue Streaming, and SageMaker all operate on real timestamps. PaySim's integer steps cannot be used directly by any production AWS service. Separating event time from arrival time also enables late-data handling via Spark watermarks — a pattern every streaming engineer must implement.

**Production mirror:** Stripe's event log separates `created` (when the payment was initiated) from `received_at` (when the webhook arrived at the processor).

---

### 2. Schema Evolution (`schema_version`)

**What:** The first 80% of rows are tagged `v1`. The remaining 20% are tagged `v2` and include `ip_risk_score`, which is null in v1.

**Why:** Real schemas change mid-stream. A pipeline that cannot handle nullable new fields will crash in production. Glue Schema Registry and Avro/Parquet column evolution exist precisely for this reason. By baking schema evolution into the dataset, you force the pipeline to implement forward-compatible deserialization.

**Production mirror:** Airbnb's data platform blog describes their Kafka schema registry approach for managing breaking vs. non-breaking schema changes.

---

### 3. Customer Behavioral Features

**What:** `customer_segment`, `customer_country`, `customer_lifetime_transactions`, `customer_lifetime_spend`, `days_since_last_transaction`

**Why:** These are the exact features that power a customer risk model. Lifetime transaction count and spend distinguish a new account (high risk) from a 5-year customer (lower risk). Days since last transaction flags dormant accounts that suddenly become active — a classic account takeover signal.

**Engineering note:** These features are computed statelessly across chunks using a running dictionary (`O(n)` time, `O(customers)` space). At 10× scale they would move to DynamoDB as the feature store, queried by Lambda on each transaction.

---

### 4. Transaction Velocity Features

**What:** `transaction_velocity_5min` and `transaction_velocity_1hour`

**Why:** Velocity is one of the highest-signal fraud features. A customer making 8 transactions in 5 minutes is almost certainly card testing. These features are expensive to compute in real-time — they require maintaining per-customer sliding windows. In a production system, DynamoDB stores these counters and Lambda computes them on every transaction event from Kinesis.

**Production mirror:** Capital One's fraud platform uses velocity counters stored in Redis (low-latency key-value) to gate transactions in under 100ms.

---

### 5. Device & IP Context

**What:** `device_id`, `device_type`, `device_age_days`, `ip_address`, `ip_risk_score`

**Why:** Device fingerprinting is the primary defense against account takeover fraud. When a customer who always uses an iPhone suddenly transacts from a 1-day-old Android device with a high-risk IP, that is a strong fraud signal even if the transaction amount looks normal.

The `new_device_flag` captures this: `device_age_days < 2`. The `ip_risk_score` (v2 only) simulates a third-party threat intelligence feed such as MaxMind or IPQualityScore.

---

### 6. Geo Anomaly Detection

**What:** `geo_anomaly_flag` is set when `merchant_country ≠ customer_country` AND the customer made a transaction within the past 30 minutes.

**Why:** Impossible travel is a well-known fraud signal. A customer in the US cannot make a physical purchase in Nigeria 10 minutes after making one in New York. This flag is what SageMaker's feature engineering layer would compute before scoring each transaction.

---

### 7. Chargeback Simulation

**What:** 85% of fraud rows get `chargeback_status = Filed` with a delay of 7–90 days. 2% of legitimate rows also get chargebacks (friendly fraud).

**Why:** Fraud discovery is not instant. In production, chargebacks arrive weeks after the fraudulent transaction, which means ML models must be retrained continuously as ground truth evolves. The delayed chargeback table also drives Step Functions workflows for case management.

---

### 8. Operational Failures

| Challenge | Rate | How Injected | Production Source |
|-----------|------|-------------|-------------------|
| Duplicate events | 1.5% | New `event_id`, same payload | Kinesis at-least-once delivery |
| Late arrivals | 2.0% | `arrival_time` > `event_time` + 5 min | Network jitter, upstream retry |
| Out-of-order events | 4.0% | `event_time` swapped between nearby rows | Shard rebalancing, retries |
| Missing values | 3.0% | Random nulls in nullable columns | Optional API fields, upstream failures |
| Corrupted records | 0.5% | Negative amounts, bad timestamps, out-of-range scores | Bit rot, upstream bugs |

**Why inject failures?** Every AWS pipeline must handle these. A pipeline that only works on clean data will fail on day one in production. By building failure handling in from the start, you demonstrate production engineering maturity rather than tutorial-level implementation.

---

### 9. Concept Drift

**What:** In the second half of the dataset (step > 372 for the full 6.3M row run), `merchant_risk_score` for fraud transactions is multiplied by 1.4. This simulates a new fraud ring that migrates to slightly higher-risk merchants over time.

**Why:** ML models degrade when the statistical distribution of their input features shifts. This is called concept drift. A fraud detection system that does not monitor for distribution shift will quietly stop working while fraud rates climb. SageMaker Model Monitor detects this automatically — but only if the dataset gives it something to detect.

---

### 10. Seasonality

**What:** `traffic_period` tags rows as `Payday` (steps 168–192) or `Holiday` (steps 336–360).

**Why:** Fraud rates spike during high-traffic periods (payday, Black Friday, holidays) because transaction volume increases and fraud models trained on normal-period data underperform. These tags allow Glue jobs to compute separate baselines for each traffic period.

---

## AWS Service Mapping

| Dataset Feature | AWS Service | How It Is Used |
|---|---|---|
| `event_id` | Amazon Kinesis | Shard partition key; deduplication key in DynamoDB |
| `event_time` / `arrival_time` | Kinesis Data Streams | Event time vs. ingestion time for watermarking |
| `schema_version` | AWS Glue Schema Registry | Avro/JSON schema evolution; backward compatibility |
| `transaction_velocity_*` | AWS Lambda + DynamoDB | Lambda reads/updates velocity counters per transaction |
| `merchant_risk_score` | AWS Lambda | Real-time gate: if score > 0.80, flag before scoring |
| `ip_risk_score` | AWS Lambda | Third-party enrichment layer in the Lambda function |
| `geo_anomaly_flag` | SageMaker Feature Store | Pre-computed feature for fraud scoring model |
| `new_device_flag` | SageMaker | High-weight feature in gradient boosting model |
| `chargeback_status` + `chargebacks.csv` | AWS Step Functions | Chargeback case management workflow |
| `duplicate_event_flag` | Kinesis / Lambda | Idempotency key check against DynamoDB before processing |
| `late_arrival_flag` | Glue Streaming / Spark | Watermark tolerance window (e.g., 10-minute late threshold) |
| `isFraud` (original) | SageMaker | Training label; never modified by enhancement pipeline |
| `traffic_period` | CloudWatch | Separate alarm thresholds for Payday vs. Normal periods |
| `customers.csv` | Glue Data Catalog | Slowly Changing Dimension (Type 2) table in Redshift |
| `merchants.csv` | Glue Data Catalog | Merchant dimension; joined in Glue Silver layer |
| `chargebacks.csv` | S3 → Glue → Redshift | Gold layer fact table; drives weekly retraining Step Function |
| Replay modes (10×, 100×) | Kinesis load testing | Stress test Kinesis shard scaling and Lambda concurrency |

---

## How to Run

### Requirements
```
Python 3.9+
pandas >= 1.5
numpy >= 1.23
faker >= 18.0
```

### Install
```bash
pip install pandas numpy faker
```

### Run (demo — 500K row sample)
```bash
python enhancement_pipeline.py \
    --input  PS_20174392719_1491204439457_log.csv \
    --outdir ./output \
    --chunk-size 200000 \
    --replay-mode 1x
```

### Run (full 6.3M row dataset — allow ~15 min)
```bash
python enhancement_pipeline.py \
    --input  PS_20174392719_1491204439457_log.csv \
    --outdir ./output \
    --chunk-size 500000 \
    --replay-mode 1x
```

### Run with 10× replay for load testing
```bash
python enhancement_pipeline.py \
    --input  PS_20174392719_1491204439457_log.csv \
    --outdir ./output \
    --chunk-size 500000 \
    --replay-mode 10x
```

---

## Output Files

| File | Rows (demo) | Description |
|------|-------------|-------------|
| `enhanced_transactions.csv` | 507,500 | Main enriched transaction fact table |
| `customers.csv` | 499,953 | Customer dimension (one row per unique originator) |
| `merchants.csv` | 214,856 | Merchant dimension |
| `devices.csv` | 499,953 | Device dimension |
| `chargebacks.csv` | 233 | Chargeback fact table (fraud-originated) |
| `data_dictionary.md` | — | Full column documentation |
| `engineering_reflection.md` | — | Internal engineering document |

---

## Configuration

All enhancement percentages are configurable via the `CONFIG` dict in `enhancement_pipeline.py`:

```python
CONFIG = {
    "late_arrival_pct":       0.02,   # 2% late records
    "duplicate_event_pct":    0.015,  # 1.5% duplicates
    "out_of_order_pct":       0.02,   # 2% out-of-order
    "missing_value_pct":      0.03,   # 3% nulls injected
    "corrupted_record_pct":   0.005,  # 0.5% corrupted
    "high_risk_merchant_pct": 0.10,   # 10% high-risk merchants
    "chargeback_rate_fraud":  0.85,   # 85% of fraud → chargeback
    "schema_v2_step":         400,    # v2 schema starts at step 400
    "seed":                   42,     # reproducibility
}
```

---

## Reproducibility

All synthetic enhancements use `seed=42` via `numpy.random.default_rng` and `Faker.seed()`. Running the pipeline twice on the same input produces identical output.
