# Data Dictionary — PaySim Enhanced Financial Transaction Dataset

**Version:** 1.0.0  
**Seed:** 42 (all synthetic fields reproducible)  
**Base Dataset:** PaySim (Kaggle) — 6.36M transactions, mobile money simulation  
**Enhanced Sample:** 507,500 transactions (500K base + 1.5% duplicates)

---

## Table: `enhanced_transactions.csv`

### Original PaySim Columns (preserved exactly)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `step` | int | Simulation time step; 1 step = 1 hour of simulated time | `1`, `743` |
| `type` | string | Transaction type | `CASH_OUT`, `PAYMENT`, `TRANSFER`, `CASH_IN`, `DEBIT` |
| `amount` | float | Transaction amount in local currency (pesos) | `9839.64` |
| `nameOrig` | string | Originating account ID (customer or merchant prefix) | `C1231006815` |
| `oldbalanceOrg` | float | Originator balance before transaction | `170136.00` |
| `newbalanceOrig` | float | Originator balance after transaction | `160296.36` |
| `nameDest` | string | Destination account ID | `M1979787155` |
| `oldbalanceDest` | float | Destination balance before transaction | `0.00` |
| `newbalanceDest` | float | Destination balance after transaction | `0.00` |
| `isFraud` | int | Ground truth fraud label (0 = legitimate, 1 = fraud) | `0`, `1` |
| `isFlaggedFraud` | int | PaySim rule-based flag for large fraudulent transfers | `0`, `1` |

> **Critical:** `isFraud` and `isFlaggedFraud` are never modified by the enhancement pipeline. Downstream ML models must treat these as immutable ground truth.

---

### Enhanced Columns (added by pipeline)

#### Identity & Timing

| Column | Type | Nullable | Description | AWS Usage |
|--------|------|----------|-------------|-----------|
| `event_id` | string | No | UUID-based unique event identifier (`EVT-` prefix + 16 hex chars). Globally unique — duplicate rows get new event_ids. | DynamoDB partition key; Kinesis dedup key |
| `event_time` | ISO datetime | No | Wall-clock time derived from `step` (step × 60 min from epoch 2023-01-01). Represents *when the transaction occurred*. | Kinesis partition timestamp; Glue watermark |
| `arrival_time` | ISO datetime | No | When the event arrived at the ingestion layer. 2% of records arrive 5–360 minutes *after* `event_time`, simulating network delays and upstream system lag. | Kinesis approximate arrival; late-data watermarking |
| `schema_version` | string | No | `v1` (steps 1–399) or `v2` (steps 400+). In v2, `ip_risk_score` becomes available. Mirrors real schema evolution as product teams add features. | Glue schema registry versioning |

#### Customer Profile

| Column | Type | Nullable | Description | AWS Usage |
|--------|------|----------|-------------|-----------|
| `customer_segment` | string | Yes (3% chance) | Behavioral segment: `Student`, `Traveler`, `Business`, `High Spender`, `Family`. Assigned once per customer and held constant. | SageMaker feature engineering; Redshift cohort queries |
| `customer_country` | string | No | Customer's home country (ISO 2-letter). US majority; 14 other countries. | Geo anomaly detection; DynamoDB feature store |
| `customer_lifetime_transactions` | int | No | Running count of all transactions by this customer up to and including this event. Computed across chunks. | ML velocity feature; churn risk scoring |
| `customer_lifetime_spend` | float | No | Cumulative spend by this customer (all transaction types). | LTV computation; risk segmentation |
| `days_since_last_transaction` | float | Yes | Calendar days since this customer's previous transaction. Null for a customer's first transaction. | Dormancy detection; account takeover signals |

#### Device & Network

| Column | Type | Nullable | Description | AWS Usage |
|--------|------|----------|-------------|-----------|
| `device_id` | string | No | Deterministic device identifier derived from customer ID + device slot. Customers reuse the same 1–3 devices. Format: `DEV-` + 12 hex chars. | DynamoDB device history; new-device anomaly detection |
| `device_type` | string | Yes (3% chance) | `iPhone`, `Android`, `Web`, `Tablet`. Weighted: 35/35/20/10. | Feature engineering; risk stratification |
| `device_age_days` | int | Yes (3% chance) | Age of the device in days (1–1460). Devices < 2 days old set `new_device_flag = 1`. | Account takeover signal |
| `ip_address` | string | Yes (3% chance) | Synthetic IPv4 address. Per-customer: same customer uses the same IP across sessions (unless flagged as anomaly). | Velocity checks; fraud ring detection |
| `ip_risk_score` | float | Yes (null for v1) | Risk score 0.0–1.0 for the originating IP. **Only populated for schema_version = v2.** Simulates a third-party IP intelligence feed. | Lambda real-time scoring; SageMaker feature |

#### Merchant Context

| Column | Type | Nullable | Description | AWS Usage |
|--------|------|----------|-------------|-----------|
| `merchant_country` | string | No | Country where merchant is registered. | Cross-border transaction risk; geo anomaly |
| `merchant_category` | string | Yes (3% chance) | `Retail`, `Electronics`, `Travel`, `Food`, `Healthcare`, `Entertainment`, `Utilities`. | Category-level risk models; Redshift aggregations |
| `merchant_risk_score` | float | Yes (3% chance) | Merchant risk score 0.0–1.0. ~10% of merchants are high-risk (score > 0.70). In the second half of the dataset, fraud-associated merchants have scores boosted by 40% (concept drift simulation). | Real-time Lambda threshold check; SageMaker feature |

#### Behavioral Features (pre-computed for real-time scoring)

| Column | Type | Nullable | Description | AWS Usage |
|--------|------|----------|-------------|-----------|
| `transaction_velocity_5min` | int | No | Number of transactions this customer made in the 5 minutes before this event. Card-testing attacks produce values of 5+. | Lambda velocity gate; SageMaker feature |
| `transaction_velocity_1hour` | int | No | Number of transactions this customer made in the hour before this event. | DynamoDB feature store; fraud scoring |

#### Fraud Enhancement Flags

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `geo_anomaly_flag` | int (0/1) | No | 1 if merchant_country ≠ customer_country AND the previous transaction by this customer was less than 30 minutes ago (impossible travel). |
| `new_device_flag` | int (0/1) | No | 1 if device_age_days < 2. Correlates with account takeover scenarios. |
| `chargeback_status` | string | No | `Filed` or `None`. 85% of fraud rows and 2% of legitimate rows have a chargeback filed (friendly fraud). |
| `chargeback_delay_days` | float | Yes | Days between transaction and chargeback filing (7–90 days). Null if no chargeback. |

#### Operational Challenge Flags

| Column | Type | Description | Pipeline Handling |
|--------|------|-------------|-------------------|
| `duplicate_event_flag` | int (0/1) | 1 if this is a duplicate event (1.5% of rows). Duplicate rows have a new `event_id` but same payload. | Kinesis consumer deduplication via DynamoDB idempotency key |
| `late_arrival_flag` | int (0/1) | 1 if `arrival_time` > `event_time` + 5 min (2% of rows). | Kinesis Firehose buffering; Glue streaming watermarks |
| `out_of_order_flag` | int (0/1) | 1 if this event's `event_time` is swapped with a nearby event (4% of rows). | Spark watermark; Lambda sort buffer |

#### Context Tags

| Column | Type | Description |
|--------|------|-------------|
| `traffic_period` | string | `Normal`, `Payday` (steps 168–192), `Holiday` (steps 336–360). Used for anomaly baselining and dashboards. |
| `replay_round` | int | `0` for original data; 1–N for replay copies in 10x/100x mode. |

---

## Table: `customers.csv`

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Matches `nameOrig` in transactions. Primary key. |
| `customer_segment` | string | `Student`, `Traveler`, `Business`, `High Spender`, `Family` |
| `home_country` | string | ISO 2-letter country code |
| `account_creation_date` | date | Date account was created (1 month – 5 years before epoch) |
| `risk_profile` | string | `Low` (65%), `Medium` (25%), `High` (10%) |
| `preferred_device_type` | int | Device slot index (0–2) assigned to this customer |

---

## Table: `merchants.csv`

| Column | Type | Description |
|--------|------|-------------|
| `merchant_id` | string | Matches `nameDest` for merchant-prefixed destinations. Primary key. |
| `merchant_category` | string | One of 7 categories |
| `merchant_country` | string | ISO 2-letter country code |
| `merchant_risk_score` | float | 0.0–1.0. ~10% of merchants score > 0.70 (high risk). |
| `merchant_creation_date` | date | Date merchant was onboarded |

---

## Table: `devices.csv`

| Column | Type | Description |
|--------|------|-------------|
| `device_id` | string | Matches `device_id` in transactions. Primary key. |
| `device_type` | string | `iPhone`, `Android`, `Web`, `Tablet` |
| `device_age_days` | int | Age of device at time of first use |
| `operating_system` | string | e.g. `iOS 17`, `Android 14`, `Chrome/Windows` |

---

## Table: `chargebacks.csv`

| Column | Type | Description |
|--------|------|-------------|
| `chargeback_id` | string | Unique chargeback ID (`CB-` prefix). Primary key. |
| `transaction_id` | string | References `event_id` in transactions. |
| `customer_id` | string | References `nameOrig` |
| `days_to_dispute` | int | Days between transaction and filing (7–90) |
| `resolution_status` | string | `Approved` (70%), `Denied` (20%), `Under Review` (10%) |

---

## Fraud Pattern Reference

| Fraud Type | Signals in Dataset | Original Label Preserved? |
|---|---|---|
| Card Testing | `transaction_velocity_5min` > 5, small amounts | Yes — `isFraud=1` |
| Account Takeover | `new_device_flag=1`, `geo_anomaly_flag=1`, same customer | Yes |
| Impossible Travel | `geo_anomaly_flag=1`, `days_since_last_transaction` < 0.02 | Yes |
| Merchant Abuse | `merchant_risk_score` > 0.70, `type=CASH_OUT` | Yes |
| Friendly Fraud | `isFraud=0`, `chargeback_status=Filed` | Yes (legitimate rows) |
| Concept Drift | `merchant_risk_score` boosted in step > 372 for fraud rows | Enhancement only |
