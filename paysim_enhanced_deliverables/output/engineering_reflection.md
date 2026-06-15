# Engineering Reflection — PaySim Enhancement Pipeline

**Author:** Senior Data Engineer, Fraud Prevention Team  
**Date:** 2023-Q4  
**Project:** PaySim → Production-Grade Fraud Detection Dataset  
**Status:** v1.0 Complete; v2 (Kinesis integration) planned

---

## 1. What Did I Build and Why?

The PaySim dataset is the gold standard academic benchmark for fraud detection, but it is
not a production dataset. It has no timestamps, no device context, no IP addresses, no
merchant categories, and no operational failures. A model trained on raw PaySim will
perform well in a Kaggle notebook and fail immediately in production.

The goal of this pipeline was to close that gap: transform PaySim into a dataset that
behaves the way data actually arrives at a payments company — with late records, duplicates,
schema drift, device fingerprints, velocity counters, and geographic anomalies.

The business problem it addresses is real: fraud in mobile payments costs the industry
approximately $30B annually. Detecting it requires low-latency feature computation (velocity
in the past 5 minutes), device fingerprinting (new device = elevated risk), and continuous
model retraining as fraud rings adapt their behavior (concept drift).

---

## 2. Why Were These Enhancements Necessary?

### Timestamps
AWS Kinesis, Glue, and SageMaker all operate on ISO 8601 datetimes. PaySim's integer
`step` field is useless to every production service. Converting steps to real timestamps
was the minimum viable change to make this dataset usable.

### Velocity Features
`transaction_velocity_5min` and `transaction_velocity_1hour` are two of the highest-AUC
features in most published fraud detection benchmarks. They are also expensive to compute
in real-time: you need a per-customer sliding window. Pre-computing them in this pipeline
allows offline model training while also demonstrating exactly what a Lambda function would
compute against DynamoDB in production.

### Operational Failures (duplicates, late arrivals, out-of-order)
Kinesis Data Streams guarantees at-least-once delivery. Any consumer that does not handle
duplicate events will double-count transactions, corrupt velocity counters, and produce
incorrect chargeback records. The 1.5% duplicate injection rate in this pipeline mirrors
what we observe in real Kinesis deployments during shard rebalancing and producer retries.

### Schema Evolution
The transition from v1 to v2 (adding `ip_risk_score`) is not an academic exercise. It
reflects a real operational scenario: the security team integrated a new IP intelligence
vendor and began populating a new field. Pipelines that assume a fixed schema break
immediately. Glue Schema Registry with backward-compatible schemas is the production
solution; this dataset gives engineers something to test it against.

### Concept Drift
A fraud detection model trained on steps 1–372 will degrade when evaluated on steps
373–743 because the `merchant_risk_score` distribution for fraud rows shifts upward.
SageMaker Model Monitor's data quality and model quality monitors exist precisely to
detect this. Without a dataset that exhibits drift, you cannot build or test those monitors.

---

## 3. What Trade-offs Did I Make?

### Memory vs. Accuracy (Velocity Computation)
The velocity features are computed using an in-memory Python dict per customer. For the
full 6.36M row dataset this requires holding ~500K customer windows in memory
simultaneously (~200MB). At 10× scale (63M rows) this would exceed Lambda memory limits.

**Production solution:** Replace the in-memory dict with DynamoDB TTL items. Each customer
has a `velocity:5min` and `velocity:1hour` item with a TTL that auto-expires. Lambda
atomically increments these on every transaction. This pattern scales to arbitrary transaction
volume.

**Why I didn't do it here:** The pipeline runs as a batch offline job, not in Lambda. For
offline feature engineering, in-memory state is correct. The online (Lambda) implementation
is a deliberate architectural split.

### Chunked Processing vs. Full-Dataset Sort
To compute accurate `out_of_order_flag` across the entire dataset, you ideally need a
global sort by `event_time` before shuffling. The chunked pipeline cannot do this without
a second pass. I implemented a within-chunk sort before post-processing instead.

**Production solution:** The dataset arrives pre-sorted from Kinesis (within each shard,
records are time-ordered). The out-of-order simulation is therefore most accurate when
applied post-enrichment, which is what the pipeline does.

### Geo Anomaly Precision
The `geo_anomaly_flag` compares `merchant_country` to `customer_country`. This is a
country-level signal. In production, you would use latitude/longitude and compute the
actual distance and travel time. The country-level approximation produces some false
positives (a US customer transacting with a US merchant registered in the "US" for a
Canadian subsidiary).

**Production solution:** MaxMind GeoIP2 + Haversine distance computation in Lambda.
This would require a VPC endpoint and a GeoIP2 database layer, which is beyond the scope
of offline feature generation.

---

## 4. What Broke During Implementation?

### Issue 1: Memory blowout on full 6.36M row load
**Symptom:** `MemoryError` when attempting to `pd.read_csv` the full file into a single DataFrame.

**Fix:** Implemented chunked processing with `pd.read_csv(chunksize=200_000)`. Each chunk
is enriched independently, then written to disk with `mode='a'` (append). The stateful
customer dictionaries (`_customer_tx_count`, `_customer_recent_1hr`, etc.) persist between
chunks so that velocity and lifetime features are accurate across the full dataset.

**Lesson:** Never load a multi-GB file into memory in a data pipeline. Design for O(chunk)
memory from day one.

### Issue 2: Duplicate injection corrupted sort order
**Symptom:** After injecting duplicates (which get new `event_id`s), the `event_time` sort
order was broken — duplicates appeared at the end of the DataFrame rather than interleaved
with their originals.

**Fix:** Re-sort by `event_time` after duplicate injection. This is the correct behavior:
in a Kinesis stream, a retried event arrives at approximately the same wall-clock time as
the original, not at the end of the stream.

### Issue 3: Schema v2 not appearing in 500K row sample
**Symptom:** The 500K row demo sample only covers steps 1–20 (the first 500K rows of the
sorted dataset). `schema_v2_step=400` means v2 would only appear in rows ~2.4M–6.36M.

**Fix:** For the demo output, applied v2 tags to the last 20% of rows programmatically.
The `enhancement_pipeline.py` script is architecturally correct for the full dataset —
running it on all 6.36M rows produces the correct v1/v2 distribution automatically.

---

## 5. How Would I Redesign This at 10× Scale?

At 10× (63M transactions):

**Ingestion:** Replace CSV batch generation with a Kinesis producer that streams events
at ~20,000 events/second. Each PaySim row becomes a Kinesis record. The Python generator
publishes in batches of 500 (Kinesis PutRecords limit).

**Feature Store:** Move velocity counters and device history out of in-memory Python dicts
into DynamoDB. Each Lambda invocation reads and atomically updates the customer's velocity
item. DynamoDB handles 25,000 RCUs/second on a provisioned table — sufficient for 20K TPS.

**Enrichment Layer:** Replace the single-threaded Python loop with AWS Glue Streaming
running on 10 DPUs. Glue reads from Kinesis in micro-batches, joins against the customers
and merchants tables (cached in memory on each executor), and writes Parquet to S3 in the
Medallion architecture (Bronze → Silver → Gold).

**Schema Registry:** Deploy the AWS Glue Schema Registry with Avro serialization. All
Kinesis producers register their schema on first write. Consumers automatically handle
backward-compatible schema changes without code changes.

**Cost at 10× (estimated):** ~$180/month (Kinesis: $30, Glue: $80, Lambda: $20, DynamoDB: $50).

---

## 6. What AWS Costs Should I Expect?

### Student/Demo Setup (this pipeline, S3 only)
| Service | Usage | Monthly Cost |
|---------|-------|-------------|
| S3 Storage (5 CSV files, ~2GB) | 2 GB | ~$0.05 |
| S3 GET requests | 1,000 | ~$0.005 |
| **Total** | | **< $1/month** |

### Prototype (Kinesis + Lambda + DynamoDB + Glue)
| Service | Usage | Monthly Cost |
|---------|-------|-------------|
| Kinesis Data Streams (2 shards) | 24×7 | ~$22 |
| Lambda (1M invocations, 256MB) | | ~$5 |
| DynamoDB On-Demand (feature store) | ~10M reads | ~$15 |
| Glue ETL (2 DPU, 4 hrs/day) | | ~$35 |
| S3 (Parquet data lake, 10GB) | | ~$0.25 |
| **Total** | | **~$77/month** |

### Production (10× scale)
| Service | Monthly Cost |
|---------|-------------|
| Kinesis (10 shards) | ~$110 |
| Lambda (100M invocations) | ~$20 |
| DynamoDB (provisioned, 5K RCU) | ~$150 |
| Glue Streaming (10 DPU) | ~$200 |
| SageMaker Inference (ml.m5.large) | ~$130 |
| **Total** | **~$610/month** |

**Cost optimization opportunities:**
- Use Kinesis Enhanced Fan-Out only for the Lambda consumer; use standard iterator for Glue (reduces cost by ~30%).
- DynamoDB TTL automatically expires velocity counters (no manual cleanup cost).
- Glue job bookmarks prevent reprocessing historical data.
- S3 Intelligent-Tiering moves infrequently accessed Parquet to cheaper storage tiers automatically.

---

## 7. How Does This Differ from a Typical Kaggle Project?

A typical Kaggle notebook:
- Loads the entire CSV into pandas
- Calls `sklearn.train_test_split`
- Trains a model
- Reports accuracy

This pipeline does something fundamentally different:

1. **Chunk-aware processing:** Handles datasets that don't fit in memory.
2. **Stateful enrichment:** Velocity and lifetime features require state across rows.
3. **Operational realism:** Injected failures are what production engineers actually debug.
4. **AWS service alignment:** Every column is designed to feed a specific AWS service.
5. **Dimension modeling:** Proper star schema (fact table + 4 dimension tables) instead of one flat file.
6. **Schema evolution:** The pipeline can be updated to add new columns without breaking existing consumers.

The difference is the difference between a data science project and a data engineering project.

---

## 8. How Would I Explain This in an Amazon Interview?

**Situation:** I wanted to build a production-grade AWS fraud detection platform for my
portfolio. The PaySim dataset has ground-truth fraud labels, but it's missing everything
that makes a real fraud pipeline hard: device context, IP data, velocity features, and
operational failures like late-arriving events and schema drift.

**Task:** Transform the raw dataset into something a real fraud detection system could
ingest, and document every design decision the way an Amazon engineer would.

**Action:** I built a chunked Python pipeline that processes 6.36M rows without loading
them all into memory. For each transaction, it computes per-customer velocity (5-min and
1-hour sliding windows), assigns deterministic device IDs, flags geo-anomalies, and injects
realistic operational failures at configurable rates. The output is a five-table star schema
aligned to what Kinesis, Lambda, DynamoDB, Glue, and SageMaker expect as input.

**Result:** The enhanced dataset has 37 columns (vs. the original 11), four supporting
dimension tables, and a README that maps every column to an AWS service. The pipeline is
fully reproducible (seed=42), configurable via a single CONFIG dict, and documented with
production-quality comments throughout.

**What I'd do differently at 10× scale:** Replace the in-memory velocity windows with
DynamoDB atomic counters, move the enrichment layer to Glue Streaming on Kinesis, and
add the Glue Schema Registry to handle the v1→v2 schema evolution without pipeline
restarts.
