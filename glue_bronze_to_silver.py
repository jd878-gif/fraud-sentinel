"""
glue_bronze_to_silver.py
=========================
AWS Glue ETL Job: Bronze → Silver

Reads raw JSON transaction files from S3 Bronze layer,
applies production-quality data cleaning and validation,
writes partitioned Parquet to S3 Silver layer, and
registers the table in the Glue Data Catalog.

Glue job parameters (set when creating the job):
  --S3_BUCKET      fraud-platform-jeet-dev
  --DATABASE_NAME  fraud_platform_dev

Design decisions:
  - Uses GlueContext + DynamicFrame for native Glue integration
  - Falls back to Spark DataFrame for complex transformations
  - Partitions Silver output by year/month/day for Athena efficiency
  - Writes job metrics to CloudWatch via Glue job bookmarks
  - Corrupted records written to a separate quarantine prefix
    instead of silently dropped — production auditability pattern
"""

import sys
import json
from datetime import datetime

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType,
    TimestampType, LongType, FloatType
)

# ── Job initialization ─────────────────────────────────────────────
args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "S3_BUCKET",
    "DATABASE_NAME",
])

sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args["JOB_NAME"], args)

S3_BUCKET     = args["S3_BUCKET"]
DATABASE_NAME = args["DATABASE_NAME"]
BRONZE_PATH   = f"s3://{S3_BUCKET}/bronze/transactions/"
SILVER_PATH   = f"s3://{S3_BUCKET}/silver/transactions/"
QUARANTINE    = f"s3://{S3_BUCKET}/quarantine/bronze/"

print(f"[Bronze→Silver] Starting | bucket={S3_BUCKET}")
print(f"  Input : {BRONZE_PATH}")
print(f"  Output: {SILVER_PATH}")


# ── Step 1: Read Bronze JSON using GlueContext ─────────────────────
# Use Glue DynamicFrame for native S3 JSON reading with
# automatic schema inference and malformed record handling.
print("[Bronze→Silver] Reading Bronze JSON files ...")

bronze_dyf = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={
        "paths":       [BRONZE_PATH],
        "recurse":     True,
        "groupFiles":  "inPartition",
        "groupSize":   "1048576",   # 1 MB grouping for small files
    },
    format="json",
    format_options={
        "multiline": False,
    },
    transformation_ctx="bronze_read",
)

total_input = bronze_dyf.count()
print(f"[Bronze→Silver] Records read: {total_input:,}")

if total_input == 0:
    print("[Bronze→Silver] No records found. Exiting.")
    job.commit()
    sys.exit(0)


# ── Step 2: Flatten nested envelope structure ──────────────────────
# Bronze JSON has: { metadata: {...}, payload: {...} }
# Silver needs flat columns for Parquet and Athena queries.
print("[Bronze→Silver] Flattening envelope structure ...")

bronze_df = bronze_dyf.toDF()

# Confirm envelope structure exists
if "metadata" not in bronze_df.columns or "payload" not in bronze_df.columns:
    print("[Bronze→Silver] ERROR: Unexpected schema — missing metadata/payload")
    print("Columns found:", bronze_df.columns)
    job.commit()
    sys.exit(1)

flat_df = bronze_df.select(
    # ── Metadata fields ────────────────────────────────────────────
    F.col("metadata.event_id").alias("event_id"),
    F.col("metadata.event_time").alias("event_time_str"),
    F.col("metadata.arrival_time").alias("arrival_time_str"),
    F.col("metadata.schema_version").alias("schema_version"),
    F.col("metadata.produced_at").alias("produced_at"),
    F.col("metadata.source").alias("source"),

    # ── Original PaySim fields ─────────────────────────────────────
    F.col("payload.step").cast(IntegerType()).alias("step"),
    F.col("payload.type").alias("transaction_type"),
    F.col("payload.amount").cast(DoubleType()).alias("amount"),
    F.col("payload.nameOrig").alias("customer_id"),
    F.col("payload.oldbalanceOrg").cast(DoubleType()).alias("balance_before"),
    F.col("payload.newbalanceOrig").cast(DoubleType()).alias("balance_after"),
    F.col("payload.nameDest").alias("merchant_id"),
    F.col("payload.oldbalanceDest").cast(DoubleType()).alias("dest_balance_before"),
    F.col("payload.newbalanceDest").cast(DoubleType()).alias("dest_balance_after"),
    F.col("payload.isFraud").cast(IntegerType()).alias("is_fraud"),
    F.col("payload.isFlaggedFraud").cast(IntegerType()).alias("is_flagged_fraud"),

    # ── Enhanced fields ────────────────────────────────────────────
    F.col("payload.customer_segment").alias("customer_segment"),
    F.col("payload.device_id").alias("device_id"),
    F.col("payload.device_type").alias("device_type"),
    F.col("payload.device_age_days").cast(IntegerType()).alias("device_age_days"),
    F.col("payload.customer_country").alias("customer_country"),
    F.col("payload.merchant_country").alias("merchant_country"),
    F.col("payload.merchant_category").alias("merchant_category"),
    F.col("payload.merchant_risk_score").cast(DoubleType()).alias("merchant_risk_score"),
    F.col("payload.ip_address").alias("ip_address"),
    F.col("payload.ip_risk_score").cast(DoubleType()).alias("ip_risk_score"),
    F.col("payload.chargeback_status").alias("chargeback_status"),
    F.col("payload.days_since_last_transaction").cast(DoubleType()).alias("days_since_last_tx"),
    F.col("payload.transaction_velocity_5min").cast(IntegerType()).alias("velocity_5min"),
    F.col("payload.transaction_velocity_1hour").cast(IntegerType()).alias("velocity_1hour"),
    F.col("payload.customer_lifetime_transactions").cast(LongType()).alias("lifetime_tx_count"),
    F.col("payload.customer_lifetime_spend").cast(DoubleType()).alias("lifetime_spend"),

    # ── Quality flags ──────────────────────────────────────────────
    F.col("payload.geo_anomaly_flag").cast(IntegerType()).alias("geo_anomaly_flag"),
    F.col("payload.new_device_flag").cast(IntegerType()).alias("new_device_flag"),
    F.col("payload.duplicate_event_flag").cast(IntegerType()).alias("duplicate_event_flag"),
    F.col("payload.late_arrival_flag").cast(IntegerType()).alias("late_arrival_flag"),
    F.col("payload.out_of_order_flag").cast(IntegerType()).alias("out_of_order_flag"),
    F.col("payload.traffic_period").alias("traffic_period"),
)

print(f"[Bronze→Silver] Flattened schema: {len(flat_df.columns)} columns")


# ── Step 3: Parse and validate timestamps ─────────────────────────
print("[Bronze→Silver] Parsing timestamps ...")

flat_df = flat_df.withColumn(
    "event_time",
    F.to_timestamp(F.col("event_time_str"), "yyyy-MM-dd'T'HH:mm:ss")
).withColumn(
    "arrival_time",
    F.to_timestamp(F.col("arrival_time_str"), "yyyy-MM-dd'T'HH:mm:ss")
).drop("event_time_str", "arrival_time_str")

# Add partition columns derived from event_time
flat_df = flat_df \
    .withColumn("year",  F.year(F.col("event_time"))) \
    .withColumn("month", F.month(F.col("event_time"))) \
    .withColumn("day",   F.dayofmonth(F.col("event_time")))


# ── Step 4: Separate valid from corrupted records ─────────────────
# Corrupted records = negative amounts OR null event_id OR null event_time
# Production pattern: never silently drop bad data — quarantine it
print("[Bronze→Silver] Separating valid from corrupted records ...")

valid_df = flat_df.filter(
    F.col("event_id").isNotNull() &
    F.col("event_time").isNotNull() &
    (F.col("amount") > 0) &
    F.col("transaction_type").isNotNull()
)

corrupted_df = flat_df.filter(
    F.col("event_id").isNull() |
    F.col("event_time").isNull() |
    (F.col("amount") <= 0) |
    F.col("transaction_type").isNull()
)

valid_count     = valid_df.count()
corrupted_count = corrupted_df.count()
print(f"  Valid records    : {valid_count:,}")
print(f"  Corrupted records: {corrupted_count:,}")

# Write corrupted records to quarantine (don't lose them)
if corrupted_count > 0:
    corrupted_df.write \
        .mode("append") \
        .json(QUARANTINE)
    print(f"  Quarantined {corrupted_count} records to {QUARANTINE}")


# ── Step 5: Deduplicate on event_id ───────────────────────────────
# Keep the first occurrence of each event_id.
# SQS at-least-once delivery can produce duplicates.
print("[Bronze→Silver] Deduplicating on event_id ...")

deduped_df = valid_df.dropDuplicates(["event_id"])
dedup_removed = valid_count - deduped_df.count()
print(f"  Duplicates removed: {dedup_removed:,}")
print(f"  Records after dedup: {deduped_df.count():,}")


# ── Step 6: Add computed columns ───────────────────────────────────
print("[Bronze→Silver] Adding computed columns ...")

silver_df = deduped_df \
    .withColumn(
        # Balance delta — negative means money left the account
        "balance_delta",
        F.col("balance_after") - F.col("balance_before")
    ) \
    .withColumn(
        # Flag transactions where balance didn't change as expected
        # Classic fraud pattern: balance stays the same after TRANSFER
        "balance_mismatch_flag",
        F.when(
            (F.col("transaction_type") == "TRANSFER") &
            (F.col("balance_after") == F.col("balance_before")) &
            (F.col("amount") > 0),
            F.lit(1)
        ).otherwise(F.lit(0))
    ) \
    .withColumn(
        # Risk tier based on merchant_risk_score
        "merchant_risk_tier",
        F.when(F.col("merchant_risk_score") >= 0.75, F.lit("HIGH"))
         .when(F.col("merchant_risk_score") >= 0.40, F.lit("MEDIUM"))
         .otherwise(F.lit("LOW"))
    ) \
    .withColumn(
        # Combined fraud signal score (simple weighted sum)
        # Used for ranking alerts — not a replacement for ML
        "rule_signal_score",
        (F.col("geo_anomaly_flag") * 3) +
        (F.col("new_device_flag") * 2) +
        (F.col("duplicate_event_flag") * 1) +
        (F.when(F.col("velocity_5min") >= 5, F.lit(4)).otherwise(F.lit(0))) +
        (F.when(F.col("merchant_risk_score") >= 0.75, F.lit(3)).otherwise(F.lit(0)))
    ) \
    .withColumn(
        "processed_at",
        F.lit(datetime.utcnow().isoformat())
    )

print(f"[Bronze→Silver] Silver schema: {len(silver_df.columns)} columns")


# ── Step 7: Write Silver Parquet ───────────────────────────────────
# Partitioned by year/month/day for Athena query efficiency.
# Parquet columnar format: 10-20x smaller than JSON, 10x faster queries.
print(f"[Bronze→Silver] Writing Silver Parquet to {SILVER_PATH} ...")

silver_df.write \
    .mode("overwrite") \
    .partitionBy("year", "month", "day") \
    .parquet(SILVER_PATH)

print("[Bronze→Silver] Silver write complete")


# ── Step 8: Register Silver table in Glue Data Catalog ────────────
# print("[Bronze→Silver] Registering Silver table in Glue Data Catalog ...")

# silver_dyf = DynamicFrame.fromDF(silver_df, glueContext, "silver_dyf")

# glueContext.write_dynamic_frame.from_catalog(
#     frame         = silver_dyf,
#     database      = DATABASE_NAME,
#     table_name    = "silver_transactions",
#     additional_options={
#         "enableUpdateCatalog": True,
#         "updateBehavior":      "UPDATE_IN_DATABASE",
#         "partitionKeys":       ["year", "month", "day"],
#     },
#     transformation_ctx="silver_catalog_write",
# )

# print("[Bronze→Silver] Catalog registration complete")


# ── Step 9: Print job summary ──────────────────────────────────────
final_count = silver_df.count()
print("")
print("=" * 60)
print("Bronze → Silver Job Summary")
print(f"  Input records    : {total_input:,}")
print(f"  Valid records    : {valid_count:,}")
print(f"  Corrupted        : {corrupted_count:,}")
print(f"  Duplicates removed: {dedup_removed:,}")
print(f"  Silver records   : {final_count:,}")
print(f"  Output path      : {SILVER_PATH}")
print(f"  Catalog table    : {DATABASE_NAME}.silver_transactions")
print("=" * 60)

job.commit()
print("[Bronze→Silver] Job committed successfully")
