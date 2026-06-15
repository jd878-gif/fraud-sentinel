"""
glue_silver_to_gold.py
========================
AWS Glue ETL Job: Silver → Gold

Reads clean Silver Parquet, computes fraud analytics aggregations,
writes Gold summary tables, registers in Glue Data Catalog.

Gold tables produced:
  gold/fraud_by_merchant/    — fraud rate and avg amount per merchant category
  gold/fraud_by_segment/     — fraud rate per customer segment
  gold/fraud_by_hour/        — fraud rate by hour of day (time pattern analysis)
  gold/high_risk_customers/  — customers with multiple fraud signals

These tables answer the business questions that matter:
  "Which merchant category has the highest fraud rate?"
  "Which customer segment is most targeted?"
  "What time of day do fraud attacks peak?"
  "Which customers show multiple risk signals?"

Glue job parameters:
  --S3_BUCKET      fraud-platform-jeet-dev
  --DATABASE_NAME  fraud_platform_dev
"""

import sys
from datetime import datetime

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

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
SILVER_PATH   = f"s3://{S3_BUCKET}/silver/transactions/"
GOLD_BASE     = f"s3://{S3_BUCKET}/gold/"

RUN_DATE = datetime.utcnow().isoformat()

print(f"[Silver→Gold] Starting | bucket={S3_BUCKET}")
print(f"  Input : {SILVER_PATH}")
print(f"  Output: {GOLD_BASE}")


# ── Read Silver ────────────────────────────────────────────────────
print("[Silver→Gold] Reading Silver Parquet ...")

silver_df = spark.read.parquet(SILVER_PATH)
total = silver_df.count()
print(f"[Silver→Gold] Silver records: {total:,}")

if total == 0:
    print("[Silver→Gold] No Silver data found. Run Bronze→Silver job first.")
    job.commit()
    sys.exit(0)

# Cache Silver since we query it multiple times
silver_df.cache()


# ── Helper: write Gold table + register in catalog ─────────────────

def write_gold_table(df, gold_path: str, table_name: str, desc: str):
    """Write one Gold table to S3 Parquet only.
    Catalog registration handled by crawler after job completes."""
    print(f"[Silver→Gold] Writing {desc} → {gold_path}")
    row_count = df.count()
    df.write \
        .mode("overwrite") \
        .parquet(gold_path)
    print(f"  Rows: {row_count:,} | Written to {gold_path}")
    return row_count


# ── Gold Table 1: Fraud by Merchant Category ───────────────────────
# Business question: "Which merchant categories attract the most fraud?"
# Used by: risk team to adjust merchant onboarding thresholds
print("[Silver→Gold] Computing fraud_by_merchant ...")

fraud_by_merchant = silver_df \
    .groupBy("merchant_category", "merchant_risk_tier") \
    .agg(
        F.count("*").alias("total_transactions"),
        F.sum("is_fraud").alias("fraud_transactions"),
        F.round(
            F.sum("is_fraud").cast(DoubleType()) /
            F.count("*").cast(DoubleType()) * 100, 2
        ).alias("fraud_rate_pct"),
        F.round(F.avg("amount"), 2).alias("avg_transaction_amount"),
        F.round(F.sum("amount"), 2).alias("total_volume"),
        F.round(F.avg("merchant_risk_score"), 4).alias("avg_merchant_risk_score"),
        F.round(F.avg("rule_signal_score"), 2).alias("avg_signal_score"),
        F.lit(RUN_DATE).alias("computed_at"),
    ) \
    .orderBy(F.col("fraud_rate_pct").desc())

write_gold_table(
    fraud_by_merchant,
    f"{GOLD_BASE}fraud_by_merchant/",
    "fraud_by_merchant",
    "Fraud by Merchant Category",
)


# ── Gold Table 2: Fraud by Customer Segment ────────────────────────
# Business question: "Which customer segments are most targeted?"
# Used by: product team to design segment-specific friction rules
print("[Silver→Gold] Computing fraud_by_segment ...")

fraud_by_segment = silver_df \
    .groupBy("customer_segment") \
    .agg(
        F.count("*").alias("total_transactions"),
        F.sum("is_fraud").alias("fraud_transactions"),
        F.round(
            F.sum("is_fraud").cast(DoubleType()) /
            F.count("*").cast(DoubleType()) * 100, 2
        ).alias("fraud_rate_pct"),
        F.round(F.avg("amount"), 2).alias("avg_transaction_amount"),
        F.round(F.avg("lifetime_tx_count"), 1).alias("avg_lifetime_transactions"),
        F.round(F.avg("lifetime_spend"), 2).alias("avg_lifetime_spend"),
        F.sum("geo_anomaly_flag").alias("geo_anomaly_count"),
        F.sum("new_device_flag").alias("new_device_count"),
        F.lit(RUN_DATE).alias("computed_at"),
    ) \
    .orderBy(F.col("fraud_rate_pct").desc())

write_gold_table(
    fraud_by_segment,
    f"{GOLD_BASE}fraud_by_segment/",
    "fraud_by_segment",
    "Fraud by Customer Segment",
)


# ── Gold Table 3: Fraud by Hour of Day ────────────────────────────
# Business question: "When do fraud attacks peak?"
# Used by: ops team to schedule enhanced monitoring windows
print("[Silver→Gold] Computing fraud_by_hour ...")

fraud_by_hour = silver_df \
    .withColumn("hour_of_day", F.hour(F.col("event_time"))) \
    .groupBy("hour_of_day", "traffic_period") \
    .agg(
        F.count("*").alias("total_transactions"),
        F.sum("is_fraud").alias("fraud_transactions"),
        F.round(
            F.sum("is_fraud").cast(DoubleType()) /
            F.count("*").cast(DoubleType()) * 100, 2
        ).alias("fraud_rate_pct"),
        F.round(F.avg("amount"), 2).alias("avg_amount"),
        F.round(F.avg("velocity_5min"), 2).alias("avg_velocity_5min"),
        F.lit(RUN_DATE).alias("computed_at"),
    ) \
    .orderBy(F.col("hour_of_day"))

write_gold_table(
    fraud_by_hour,
    f"{GOLD_BASE}fraud_by_hour/",
    "fraud_by_hour",
    "Fraud by Hour of Day",
)


# ── Gold Table 4: High Risk Customers ─────────────────────────────
# Business question: "Which customers show multiple fraud signals?"
# Used by: fraud ops team for manual review queue
print("[Silver→Gold] Computing high_risk_customers ...")

high_risk_customers = silver_df \
    .groupBy("customer_id", "customer_segment", "customer_country") \
    .agg(
        F.count("*").alias("total_transactions"),
        F.sum("is_fraud").alias("confirmed_fraud_count"),
        F.sum("geo_anomaly_flag").alias("geo_anomaly_count"),
        F.sum("new_device_flag").alias("new_device_count"),
        F.sum("duplicate_event_flag").alias("duplicate_event_count"),
        F.max("velocity_5min").alias("max_velocity_5min"),
        F.max("rule_signal_score").alias("max_signal_score"),
        F.round(F.sum("amount"), 2).alias("total_transaction_volume"),
        F.round(F.avg("merchant_risk_score"), 4).alias("avg_merchant_risk"),
        F.max("event_time").alias("last_transaction_time"),
        F.lit(RUN_DATE).alias("computed_at"),
    ) \
    .filter(
        # Customer qualifies as high risk if ANY of:
        # - Has confirmed fraud transactions
        # - Shows geo anomaly + new device (account takeover signal)
        # - Has velocity >= 5 in 5 minutes (card testing signal)
        # - Has high rule signal score
        (F.col("confirmed_fraud_count") > 0) |
        ((F.col("geo_anomaly_count") > 0) & (F.col("new_device_count") > 0)) |
        (F.col("max_velocity_5min") >= 5) |
        (F.col("max_signal_score") >= 5)
    ) \
    .orderBy(F.col("max_signal_score").desc())

write_gold_table(
    high_risk_customers,
    f"{GOLD_BASE}high_risk_customers/",
    "high_risk_customers",
    "High Risk Customers",
)


# ── Gold Table 5: Pipeline Run Summary ────────────────────────────
# Operational metadata — answers "did the pipeline run correctly?"
# Used by: CloudWatch dashboards and data quality monitoring
print("[Silver→Gold] Computing pipeline_run_summary ...")

summary_data = [
    {
        "run_date":              RUN_DATE,
        "silver_records_read":   total,
        "fraud_transactions":    int(silver_df.filter(F.col("is_fraud") == 1).count()),
        "fraud_rate_pct":        round(
            silver_df.filter(F.col("is_fraud") == 1).count() / max(total, 1) * 100, 2
        ),
        "unique_customers":      silver_df.select("customer_id").distinct().count(),
        "unique_merchants":      silver_df.select("merchant_id").distinct().count(),
        "geo_anomaly_count":     int(silver_df.filter(F.col("geo_anomaly_flag") == 1).count()),
        "new_device_count":      int(silver_df.filter(F.col("new_device_flag") == 1).count()),
        "high_risk_merchant_count": int(
            silver_df.filter(F.col("merchant_risk_tier") == "HIGH").count()
        ),
        "balance_mismatch_count":int(
            silver_df.filter(F.col("balance_mismatch_flag") == 1).count()
        ),
        "s3_bucket":             S3_BUCKET,
        "glue_database":         DATABASE_NAME,
    }
]

summary_df = spark.createDataFrame(summary_data)

write_gold_table(
    summary_df,
    f"{GOLD_BASE}pipeline_run_summary/",
    "pipeline_run_summary",
    "Pipeline Run Summary",
)


# ── Final summary ──────────────────────────────────────────────────
silver_df.unpersist()

print("")
print("=" * 60)
print("Silver → Gold Job Summary")
print(f"  Silver records processed : {total:,}")
print(f"  Gold tables written      : 5")
print(f"    gold_fraud_by_merchant")
print(f"    gold_fraud_by_segment")
print(f"    gold_fraud_by_hour")
print(f"    gold_high_risk_customers")
print(f"    gold_pipeline_run_summary")
print(f"  Glue catalog database    : {DATABASE_NAME}")
print(f"  Run date                 : {RUN_DATE}")
print("=" * 60)

job.commit()
print("[Silver→Gold] Job committed successfully")
