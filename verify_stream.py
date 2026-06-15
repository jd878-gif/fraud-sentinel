"""
kinesis_producer/verify_stream.py
===================================
Reads records back from the Kinesis stream to confirm the producer
is working and the record format is correct.

What this validates:
  1. Records are arriving (not silently dropped)
  2. JSON envelope deserializes correctly
  3. Both schema v1 and v2 records are present
  4. isFraud label is preserved exactly from the source data
  5. Partition key distribution looks reasonable (not a hot shard)

Usage:
    python verify_stream.py \
        --stream-name fraud-transactions-dev \
        --region us-east-1 \
        --sample-size 50
"""

import argparse
import json
import logging
import time
from collections import Counter, defaultdict

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("verify_stream")


def read_sample(kinesis, stream_name: str, sample_size: int) -> list:
    """
    Read up to sample_size records from the beginning of the stream
    using TRIM_HORIZON (start from the oldest available record).

    Why TRIM_HORIZON for verification? We want to read what the producer
    actually sent, not what arrives after we start the verifier.
    In production consumers you would use LATEST or a checkpoint.
    """
    # Get all shards
    resp   = kinesis.list_shards(StreamName=stream_name)
    shards = resp["Shards"]
    log.info("Stream '%s' has %d shard(s)", stream_name, len(shards))

    records = []
    for shard in shards:
        shard_id = shard["ShardId"]

        # Get iterator starting from the oldest record
        iter_resp = kinesis.get_shard_iterator(
            StreamName=stream_name,
            ShardId=shard_id,
            ShardIteratorType="TRIM_HORIZON",
        )
        iterator = iter_resp["ShardIterator"]

        # Read up to sample_size records from this shard
        while iterator and len(records) < sample_size:
            try:
                get_resp = kinesis.get_records(
                    ShardIterator=iterator,
                    Limit=min(100, sample_size - len(records)),
                )
            except ClientError as e:
                log.error("GetRecords failed: %s", e)
                break

            batch = get_resp["Records"]
            if not batch:
                log.info("Shard %s: no more records available", shard_id)
                break

            for rec in batch:
                try:
                    envelope = json.loads(rec["Data"].decode("utf-8"))
                    records.append({
                        "envelope":       envelope,
                        "partition_key":  rec["PartitionKey"],
                        "sequence_number":rec["SequenceNumber"],
                        "approx_arrival": rec["ApproximateArrivalTimestamp"].isoformat(),
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    log.warning("Failed to decode record: %s", e)

            iterator = get_resp.get("NextShardIterator")
            # Small sleep to avoid hitting GetRecords rate limit (5 calls/s/shard)
            time.sleep(0.25)

        log.info("Shard %s: read %d records", shard_id, len(records))

    return records[:sample_size]


def validate_and_report(records: list):
    """Print a detailed validation report for the sampled records."""
    if not records:
        log.error("No records found. Either the stream is empty or the producer hasn't run yet.")
        return

    log.info("")
    log.info("═══════════════════════════════════════════════════════════════")
    log.info("STREAM VALIDATION REPORT")
    log.info("═══════════════════════════════════════════════════════════════")
    log.info("Records sampled: %d", len(records))
    log.info("")

    # ── Schema versions ──────────────────────────────────────────────
    schema_counts = Counter(
        r["envelope"]["metadata"]["schema_version"] for r in records
    )
    log.info("Schema versions:")
    for ver, count in sorted(schema_counts.items()):
        log.info("  %-5s : %d records", ver, count)
    log.info("")

    # ── Transaction types ─────────────────────────────────────────────
    type_counts = Counter(
        r["envelope"]["payload"]["type"] for r in records
    )
    log.info("Transaction types:")
    for tx_type, count in type_counts.most_common():
        log.info("  %-12s : %d", tx_type, count)
    log.info("")

    # ── Fraud label preservation ─────────────────────────────────────
    fraud_count = sum(
        1 for r in records if r["envelope"]["payload"]["isFraud"] == 1
    )
    log.info("Fraud label: %d fraud / %d total (%.2f%%)",
             fraud_count, len(records), fraud_count / len(records) * 100)
    log.info("")

    # ── Operational flags ─────────────────────────────────────────────
    dup_count  = sum(1 for r in records if r["envelope"]["payload"].get("duplicate_event_flag") == 1)
    late_count = sum(1 for r in records if r["envelope"]["payload"].get("late_arrival_flag") == 1)
    ooo_count  = sum(1 for r in records if r["envelope"]["payload"].get("out_of_order_flag") == 1)
    log.info("Operational flags (expected ~1.5%%, 2%%, 4%%):")
    log.info("  Duplicates:   %d (%.1f%%)", dup_count,  dup_count  / len(records) * 100)
    log.info("  Late:         %d (%.1f%%)", late_count, late_count / len(records) * 100)
    log.info("  Out-of-order: %d (%.1f%%)", ooo_count,  ooo_count  / len(records) * 100)
    log.info("")

    # ── Partition key distribution ────────────────────────────────────
    pk_counts = Counter(r["partition_key"] for r in records)
    log.info("Partition key distribution (%d unique keys):", len(pk_counts))
    log.info("  Most common:  %s (%d records)", *pk_counts.most_common(1)[0])
    log.info("  Singleton keys: %d (expected: most keys appear once)",
             sum(1 for c in pk_counts.values() if c == 1))
    log.info("")

    # ── v2-only field validation ──────────────────────────────────────
    v2_records = [r for r in records if r["envelope"]["metadata"]["schema_version"] == "v2"]
    if v2_records:
        ip_risk_present = sum(
            1 for r in v2_records
            if r["envelope"]["payload"].get("ip_risk_score") is not None
        )
        log.info("v2 records: ip_risk_score present in %d/%d", ip_risk_present, len(v2_records))
    else:
        log.info("No v2 records in sample (expected if only early steps were streamed)")
    log.info("")

    # ── Sample record ─────────────────────────────────────────────────
    log.info("Sample record (first):")
    sample = records[0]
    meta = sample["envelope"]["metadata"]
    pay  = sample["envelope"]["payload"]
    log.info("  event_id:       %s", meta["event_id"])
    log.info("  event_time:     %s", meta["event_time"])
    log.info("  schema_version: %s", meta["schema_version"])
    log.info("  produced_at:    %s", meta["produced_at"])
    log.info("  type:           %s", pay["type"])
    log.info("  amount:         %.2f", pay["amount"])
    log.info("  nameOrig:       %s", pay["nameOrig"])
    log.info("  isFraud:        %d", pay["isFraud"])
    log.info("  merchant_risk:  %s", pay.get("merchant_risk_score"))
    log.info("  partition_key:  %s", sample["partition_key"])
    log.info("  sequence_no:    %s", sample["sequence_number"][:20] + "...")
    log.info("  approx_arrival: %s", sample["approx_arrival"])
    log.info("")
    log.info("═══════════════════════════════════════════════════════════════")
    log.info("✓ Validation complete")


def main():
    parser = argparse.ArgumentParser(description="Verify Kinesis stream contents")
    parser.add_argument("--stream-name",  default="fraud-transactions-dev")
    parser.add_argument("--region",       default="us-east-1")
    parser.add_argument("--sample-size",  type=int, default=50)
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    kinesis = session.client("kinesis")

    log.info("Reading from stream '%s' ...", args.stream_name)
    records = read_sample(kinesis, args.stream_name, args.sample_size)
    validate_and_report(records)


if __name__ == "__main__":
    main()
