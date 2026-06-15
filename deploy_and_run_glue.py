"""
deploy_and_run_glue.py
=======================
Uploads Glue ETL scripts to S3, creates both Glue jobs,
runs them in sequence, and monitors until completion.

Run this after uploading Bronze files to S3.

Usage:
    python deploy_and_run_glue.py
"""

import boto3
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("glue_deploy")

# ── Config ─────────────────────────────────────────────────────────
REGION        = "us-east-1"
S3_BUCKET     = "fraud-platform-jeet-dev"
DATABASE_NAME = "fraud_platform_dev"
GLUE_ROLE     = "arn:aws:iam::621402808508:role/FraudPlatformGlueRole"

JOBS = [
    {
        "name":        "fraud-bronze-to-silver",
        "script_file": "glue_bronze_to_silver.py",
        "script_s3":   f"s3://{S3_BUCKET}/scripts/glue_bronze_to_silver.py",
        "description": "Bronze JSON → Silver Parquet (clean + validate)",
    },
    {
        "name":        "fraud-silver-to-gold",
        "script_file": "glue_silver_to_gold.py",
        "script_s3":   f"s3://{S3_BUCKET}/scripts/glue_silver_to_gold.py",
        "description": "Silver Parquet → Gold aggregations",
    },
]

DEFAULT_ARGS = {
    "--S3_BUCKET":     S3_BUCKET,
    "--DATABASE_NAME": DATABASE_NAME,
    "--job-language":  "python",
    "--job-bookmark-option": "job-bookmark-disable",
    "--enable-metrics": "",
    "--enable-continuous-cloudwatch-log": "true",
    "--TempDir": f"s3://{S3_BUCKET}/tmp/",
}


def upload_scripts(s3_client):
    """Upload both ETL scripts to S3."""
    log.info("Uploading Glue scripts to S3 ...")
    for job in JOBS:
        with open(job["script_file"], "rb") as f:
            key = f"scripts/{job['script_file']}"
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=f.read(),
                ContentType="text/x-python",
            )
        log.info("  Uploaded: s3://%s/%s", S3_BUCKET, key)


def create_or_update_job(glue_client, job_config: dict):
    """Create a Glue job, or update it if it already exists."""
    job_def = {
        "Name":        job_config["name"],
        "Description": job_config["description"],
        "Role":        GLUE_ROLE,
        "Command": {
            "Name":           "glueetl",
            "ScriptLocation": job_config["script_s3"],
            "PythonVersion":  "3",
        },
        "DefaultArguments":    DEFAULT_ARGS,
        "GlueVersion":         "4.0",
        "NumberOfWorkers":     2,
        "WorkerType":          "G.1X",
        "Timeout":             30,    # minutes — safety cutoff
        "MaxRetries":          0,     # don't retry on failure during dev
    }

    try:
        glue_client.create_job(**job_def)
        log.info("Created Glue job: %s", job_config["name"])
    except glue_client.exceptions.AlreadyExistsException:
        # Update existing job
        update_def = {k: v for k, v in job_def.items() if k != "Name"}
        glue_client.update_job(
            JobName=job_config["name"],
            JobUpdate=update_def,
        )
        log.info("Updated Glue job: %s", job_config["name"])


def run_job_and_wait(glue_client, job_name: str) -> bool:
    """
    Start a Glue job run and poll until it completes.
    Returns True on success, False on failure.

    Glue job states:
      STARTING → RUNNING → SUCCEEDED / FAILED / STOPPED / TIMEOUT
    """
    log.info("Starting Glue job: %s ...", job_name)
    response = glue_client.start_job_run(JobName=job_name)
    run_id   = response["JobRunId"]
    log.info("  Job run ID: %s", run_id)

    poll_interval = 15   # seconds between status checks
    elapsed       = 0
    max_wait      = 30 * 60   # 30 minutes max

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        run_info = glue_client.get_job_run(
            JobName=job_name,
            RunId=run_id,
        )["JobRun"]

        state       = run_info["JobRunState"]
        duration    = run_info.get("ExecutionTime", 0)
        dpu_seconds = run_info.get("DPUSeconds", 0)

        log.info("  [%3ds elapsed] State: %-12s | DPU-seconds: %.0f",
                 elapsed, state, dpu_seconds or 0)

        if state == "SUCCEEDED":
            log.info("  ✓ Job SUCCEEDED in %d seconds (%.4f DPU-hours = ~$%.4f)",
                     duration,
                     dpu_seconds / 3600,
                     (dpu_seconds / 3600) * 0.44)
            return True

        if state in ("FAILED", "STOPPED", "TIMEOUT", "ERROR"):
            error = run_info.get("ErrorMessage", "No error message available")
            log.error("  ✗ Job %s: %s", state, error)
            log.error("  Check CloudWatch logs:")
            log.error("    https://console.aws.amazon.com/cloudwatch/home?"
                      "region=%s#logGroups:prefix=/aws-glue/jobs", REGION)
            return False

    log.error("  Job timed out after %d minutes", max_wait // 60)
    return False


def verify_gold_output(s3_client):
    """Check that Gold tables were written to S3."""
    log.info("")
    log.info("Verifying Gold layer output ...")
    gold_tables = [
        "fraud_by_merchant",
        "fraud_by_segment",
        "fraud_by_hour",
        "high_risk_customers",
        "pipeline_run_summary",
    ]
    for table in gold_tables:
        prefix = f"gold/{table}/"
        resp   = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            MaxKeys=1,
        )
        count = resp.get("KeyCount", 0)
        status = "✓" if count > 0 else "✗ EMPTY"
        log.info("  %s  s3://%s/%s  (%d objects)",
                 status, S3_BUCKET, prefix, count)


def print_athena_queries():
    """Print sample Athena queries to run after the job completes."""
    log.info("")
    log.info("=" * 65)
    log.info("Athena Sample Queries")
    log.info("Run these at: https://console.aws.amazon.com/athena")
    log.info("Database: %s", DATABASE_NAME)
    log.info("=" * 65)
    queries = [
        ("Fraud rate by merchant category",
         f"SELECT merchant_category, fraud_rate_pct, total_transactions "
         f"FROM {DATABASE_NAME}.gold_fraud_by_merchant "
         f"ORDER BY fraud_rate_pct DESC;"),

        ("Fraud rate by customer segment",
         f"SELECT customer_segment, fraud_rate_pct, total_transactions "
         f"FROM {DATABASE_NAME}.gold_fraud_by_segment "
         f"ORDER BY fraud_rate_pct DESC;"),

        ("Peak fraud hours",
         f"SELECT hour_of_day, fraud_rate_pct, total_transactions "
         f"FROM {DATABASE_NAME}.gold_fraud_by_hour "
         f"ORDER BY fraud_rate_pct DESC LIMIT 5;"),

        ("Top 10 high risk customers",
         f"SELECT customer_id, confirmed_fraud_count, max_signal_score, "
         f"total_transaction_volume "
         f"FROM {DATABASE_NAME}.gold_high_risk_customers "
         f"ORDER BY max_signal_score DESC LIMIT 10;"),

        ("Pipeline run summary",
         f"SELECT * FROM {DATABASE_NAME}.gold_pipeline_run_summary;"),
    ]
    for title, query in queries:
        log.info("")
        log.info("  -- %s", title)
        log.info("  %s", query)
    log.info("=" * 65)

def run_crawlers(glue_client):
    """
    Create and run Glue Crawlers to register Silver and Gold tables
    in the Data Catalog. Crawlers auto-detect schema from Parquet files.
    This is the correct production pattern for initial table registration.
    """
    crawlers = [
        {
            "name":   "fraud-silver-crawler",
            "path":   f"s3://{S3_BUCKET}/silver/transactions/",
            "prefix": "silver_",
        },
        {
            "name":   "fraud-gold-crawler",
            "path":   f"s3://{S3_BUCKET}/gold/",
            "prefix": "gold_",
        },
    ]

    for crawler in crawlers:
        log.info("")
        log.info("Setting up crawler: %s", crawler["name"])

        # Create crawler (skip if exists)
        try:
            glue_client.create_crawler(
                Name=crawler["name"],
                Role=GLUE_ROLE,
                DatabaseName=DATABASE_NAME,
                TablePrefix=crawler["prefix"],
                Targets={
                    "S3Targets": [{"Path": crawler["path"]}]
                },
                SchemaChangePolicy={
                    "UpdateBehavior": "UPDATE_IN_DATABASE",
                    "DeleteBehavior": "DEPRECATE_IN_DATABASE",
                },
                RecrawlPolicy={"RecrawlBehavior": "CRAWL_EVERYTHING"},
            )
            log.info("  Created crawler: %s", crawler["name"])
        except glue_client.exceptions.AlreadyExistsException:
            log.info("  Crawler already exists: %s", crawler["name"])

        # Start the crawler
        try:
            glue_client.start_crawler(Name=crawler["name"])
            log.info("  Started crawler: %s", crawler["name"])
        except glue_client.exceptions.CrawlerRunningException:
            log.info("  Crawler already running: %s", crawler["name"])

        # Poll until complete
        for _ in range(40):   # max 10 minutes
            time.sleep(15)
            resp  = glue_client.get_crawler(Name=crawler["name"])
            state = resp["Crawler"]["State"]
            log.info("  Crawler state: %s", state)
            if state == "READY":
                tables = glue_client.get_tables(DatabaseName=DATABASE_NAME)
                names  = [t["Name"] for t in tables["TableList"]]
                log.info("  Tables in catalog: %s", names)
                break


def main():
    s3_client   = boto3.client("s3",   region_name=REGION)
    glue_client = boto3.client("glue", region_name=REGION)

    log.info("=" * 65)
    log.info("Fraud Platform — Glue ETL Deployment")
    log.info("  Bucket  : %s", S3_BUCKET)
    log.info("  Database: %s", DATABASE_NAME)
    log.info("  Role    : %s", GLUE_ROLE)
    log.info("=" * 65)

    # 1. Upload scripts
    upload_scripts(s3_client)

    # 2. Create/update jobs
    log.info("")
    log.info("Creating Glue jobs ...")
    for job in JOBS:
        create_or_update_job(glue_client, job)

    # 3. Run Bronze → Silver
    log.info("")
    log.info("Running Job 1: Bronze → Silver ...")
    log.info("Expected duration: 3-8 minutes on G.1X worker")
    success = run_job_and_wait(glue_client, "fraud-bronze-to-silver")

    if not success:
        log.error("Bronze→Silver failed. Stopping pipeline.")
        log.error("Fix the error above, then re-run this script.")
        return

    # 4. Run Silver → Gold
    log.info("")
    log.info("Running Job 2: Silver → Gold ...")
    log.info("Expected duration: 2-5 minutes on G.1X worker")
    success = run_job_and_wait(glue_client, "fraud-silver-to-gold")

    if not success:
        log.error("Silver→Gold failed.")
        return

    # 5. Run crawlers to register tables in Glue catalog
    if success:
        run_crawlers(glue_client)

    # 6. Verify output
    verify_gold_output(s3_client)

    # 7. Print Athena queries
    print_athena_queries()

    # 6. Print Athena queries
    print_athena_queries()

    log.info("")
    log.info("Pipeline complete. Both jobs succeeded.")
    log.info("Open Athena to query your Gold tables.")


if __name__ == "__main__":
    main()
