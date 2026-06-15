"""
PaySim Enhancement Pipeline
============================
Transforms the raw PaySim dataset into a production-grade financial
transaction dataset suitable for building AWS fraud detection platforms.

Author: Senior Data Engineer – Fraud Prevention Team
Version: 1.0.0
Seed: 42 (all randomness reproducible)

Usage:
    python enhancement_pipeline.py \
        --input  PS_20174392719_1491204439457_log.csv \
        --outdir ./output \
        --chunk-size 200000 \
        --replay-mode 1x
"""

import argparse
import hashlib
import ipaddress
import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ─────────────────────────────────────────────
# CONFIGURATION  (all percentages configurable)
# ─────────────────────────────────────────────
CONFIG = {
    # Reproducibility
    "seed": 42,

    # Simulation epoch – step 1 maps to this timestamp
    "epoch": datetime(2023, 1, 1, 0, 0, 0),
    "step_minutes": 60,           # each PaySim step = 1 hour

    # Operational challenges
    "late_arrival_pct": 0.02,     # 2% of records arrive late
    "duplicate_event_pct": 0.015, # 1.5% duplicate events
    "out_of_order_pct": 0.02,     # 2% out-of-order events
    "missing_value_pct": 0.03,    # 3% fields randomly nulled
    "corrupted_record_pct": 0.005,# 0.5% corrupted records

    # Schema evolution – step at which v2 schema begins
    "schema_v2_step": 400,

    # Geo anomaly window – transactions within this many minutes
    # from same customer in geographically impossible locations
    "geo_anomaly_window_minutes": 30,

    # Merchant risk: fraction of merchants that are high-risk
    "high_risk_merchant_pct": 0.10,

    # Chargeback rates
    "chargeback_rate_fraud": 0.85,
    "chargeback_rate_legitimate": 0.02,

    # Device reuse: each customer reuses devices across sessions
    "devices_per_customer_max": 3,

    # Replay multiplier choices
    "replay_modes": {"1x": 1, "10x": 10, "100x": 100},
}

# ─────────────────────────────────────────────
# REFERENCE DATA
# ─────────────────────────────────────────────
CUSTOMER_SEGMENTS = ["Student", "Traveler", "Business", "High Spender", "Family"]
SEGMENT_WEIGHTS   = [0.20, 0.15, 0.25, 0.15, 0.25]

DEVICE_TYPES  = ["iPhone", "Android", "Web", "Tablet"]
DEVICE_WEIGHTS= [0.35, 0.35, 0.20, 0.10]

MERCHANT_CATEGORIES = [
    "Retail", "Electronics", "Travel", "Food",
    "Healthcare", "Entertainment", "Utilities",
]
MERCHANT_CAT_WEIGHTS = [0.25, 0.10, 0.15, 0.20, 0.10, 0.12, 0.08]

COUNTRIES = [
    "US", "US", "US", "US", "US",   # majority US
    "CA", "GB", "MX", "DE", "FR",
    "IN", "BR", "AU", "JP", "NG",
]

RISK_PROFILES = ["Low", "Medium", "High"]
RISK_WEIGHTS  = [0.65, 0.25, 0.10]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def step_to_datetime(step: int, epoch: datetime, step_minutes: int) -> datetime:
    """Convert PaySim step integer to a wall-clock datetime."""
    return epoch + timedelta(minutes=int(step) * step_minutes)


def fake_ipv4(rng: np.random.Generator) -> str:
    """Generate a plausible IPv4 address (avoids reserved ranges)."""
    while True:
        octets = rng.integers(1, 255, size=4)
        if octets[0] not in (10, 127, 169, 172, 192):
            return ".".join(str(o) for o in octets)


def deterministic_device_id(customer_id: str, device_slot: int) -> str:
    """
    Customers reuse the same device IDs across transactions.
    We derive them deterministically so the mapping is reproducible
    without storing a lookup table in memory.
    """
    raw = f"{customer_id}:slot{device_slot}"
    return "DEV-" + hashlib.md5(raw.encode()).hexdigest()[:12].upper()


def inject_missing_values(df: pd.DataFrame, pct: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Randomly null a fraction of cells in nullable columns.
    Mirrors real-world upstream API failures and optional fields.
    """
    nullable_cols = [
        "merchant_category", "ip_address", "device_age_days",
        "days_since_last_transaction", "merchant_risk_score",
    ]
    n_cells = int(len(df) * pct)
    for _ in range(n_cells):
        col = nullable_cols[rng.integers(len(nullable_cols))]
        row = rng.integers(len(df))
        df.at[row, col] = np.nan
    return df


def inject_corrupted_records(df: pd.DataFrame, pct: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Corrupt a small fraction of records to test pipeline validation logic.
    Introduces: negative amounts, malformed timestamps, out-of-range risk scores.
    """
    n = max(1, int(len(df) * pct))
    idxs = rng.choice(df.index, size=n, replace=False)
    corruption_type = rng.integers(0, 3, size=n)

    for i, idx in enumerate(idxs):
        ct = corruption_type[i]
        if ct == 0:
            df.at[idx, "amount"] = -abs(df.at[idx, "amount"])          # negative amount
        elif ct == 1:
            df.at[idx, "event_time"] = "CORRUPTED_TIMESTAMP_9999-99-99" # bad timestamp
        else:
            df.at[idx, "merchant_risk_score"] = 99.9                   # out-of-range score
    return df


# ─────────────────────────────────────────────
# DIMENSION TABLE BUILDERS
# ─────────────────────────────────────────────

def build_customers_table(
    customer_ids: list,
    segment_map: dict,
    country_map: dict,
    device_type_map: dict,
    rng: np.random.Generator,
    epoch: datetime,
) -> pd.DataFrame:
    """
    Customers dimension table.
    One row per unique customer_id observed in the transaction log.
    """
    records = []
    for cid in customer_ids:
        acct_days_ago = int(rng.integers(30, 1825))  # 1 month – 5 years
        records.append({
            "customer_id":          cid,
            "customer_segment":     segment_map.get(cid, "Family"),
            "home_country":         country_map.get(cid, "US"),
            "account_creation_date":(epoch - timedelta(days=acct_days_ago)).date().isoformat(),
            "risk_profile":         rng.choice(RISK_PROFILES, p=RISK_WEIGHTS),
            "preferred_device_type":device_type_map.get(cid, "Web"),
        })
    return pd.DataFrame(records)


def build_merchants_table(
    merchant_ids: list,
    merchant_cat_map: dict,
    merchant_country_map: dict,
    merchant_risk_map: dict,
    rng: np.random.Generator,
    epoch: datetime,
) -> pd.DataFrame:
    """
    Merchants dimension table.
    One row per unique merchant destination observed.
    """
    records = []
    for mid in merchant_ids:
        acct_days_ago = int(rng.integers(60, 3650))
        records.append({
            "merchant_id":           mid,
            "merchant_category":     merchant_cat_map.get(mid, "Retail"),
            "merchant_country":      merchant_country_map.get(mid, "US"),
            "merchant_risk_score":   round(merchant_risk_map.get(mid, 0.1), 4),
            "merchant_creation_date":(epoch - timedelta(days=acct_days_ago)).date().isoformat(),
        })
    return pd.DataFrame(records)


def build_devices_table(unique_device_ids: set, rng: np.random.Generator) -> pd.DataFrame:
    """
    Devices dimension table derived from all device_ids seen in transactions.
    """
    os_map = {
        "iPhone":  ["iOS 16", "iOS 17"],
        "Android": ["Android 13", "Android 14"],
        "Web":     ["Chrome/Windows", "Safari/macOS", "Firefox/Linux"],
        "Tablet":  ["iPadOS 16", "Android 13"],
    }
    records = []
    for did in unique_device_ids:
        dtype = rng.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS)
        age   = int(rng.integers(1, 1460))  # 1 day – 4 years
        records.append({
            "device_id":       did,
            "device_type":     dtype,
            "device_age_days": age,
            "operating_system":rng.choice(os_map[dtype]),
        })
    return pd.DataFrame(records)


def build_chargebacks_table(fraud_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Chargebacks fact table.
    Fraudulent transactions + a small slice of legitimate ones (friendly fraud).
    """
    records = []
    for _, row in fraud_df.iterrows():
        delay = int(rng.integers(7, 91))
        status = rng.choice(
            ["Approved", "Denied", "Under Review"],
            p=[0.70, 0.20, 0.10],
        )
        records.append({
            "chargeback_id":    "CB-" + uuid.uuid4().hex[:10].upper(),
            "transaction_id":   row["event_id"],
            "customer_id":      row["nameOrig"],
            "days_to_dispute":  delay,
            "resolution_status":status,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# CORE ENRICHMENT – processes one chunk at a time
# ─────────────────────────────────────────────

class PaySimEnhancer:
    """
    Stateful enricher that processes the PaySim CSV in chunks.
    Maintains per-customer running counters between chunks so that
    velocity and lifetime features are accurate across the full dataset.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.rng = np.random.default_rng(config["seed"])
        self.fake = Faker()
        Faker.seed(config["seed"])

        # Per-customer state (maintained across chunks)
        self._customer_tx_count: dict   = {}   # customer_id → cumulative tx count
        self._customer_lifetime_spend: dict = {}
        self._customer_last_tx_time: dict = {}   # customer_id → last event_time
        self._customer_recent_5min: dict  = {}   # customer_id → list of recent datetimes
        self._customer_recent_1hr: dict   = {}

        # Lookup maps assigned once on first pass
        self._customer_segment_map:   dict = {}
        self._customer_country_map:   dict = {}
        self._customer_device_slot:   dict = {}   # customer_id → slot (0-2)
        self._customer_device_type:   dict = {}
        self._merchant_cat_map:       dict = {}
        self._merchant_country_map:   dict = {}
        self._merchant_risk_map:      dict = {}
        self._ip_cache:               dict = {}   # customer_id → ip string

        # Collected device ids for the devices table
        self.all_device_ids: set = set()

        # Fraud rows for chargeback table
        self._fraud_rows: list = []

    # ── assignment helpers ──────────────────────────────────────────

    def _get_or_assign_segment(self, cid: str) -> str:
        if cid not in self._customer_segment_map:
            self._customer_segment_map[cid] = self.rng.choice(
                CUSTOMER_SEGMENTS, p=SEGMENT_WEIGHTS
            )
        return self._customer_segment_map[cid]

    def _get_or_assign_country(self, cid: str) -> str:
        if cid not in self._customer_country_map:
            self._customer_country_map[cid] = self.rng.choice(COUNTRIES)
        return self._customer_country_map[cid]

    def _get_or_assign_device_id(self, cid: str) -> str:
        """Each customer is assigned a fixed device slot per session."""
        if cid not in self._customer_device_slot:
            self._customer_device_slot[cid] = int(self.rng.integers(0, 3))
        slot = self._customer_device_slot[cid]
        did  = deterministic_device_id(cid, slot)
        self.all_device_ids.add(did)
        return did

    def _get_or_assign_ip(self, cid: str) -> str:
        if cid not in self._ip_cache:
            self._ip_cache[cid] = fake_ipv4(self.rng)
        return self._ip_cache[cid]

    def _get_or_assign_merchant_cat(self, mid: str) -> str:
        if mid not in self._merchant_cat_map:
            self._merchant_cat_map[mid] = self.rng.choice(
                MERCHANT_CATEGORIES, p=MERCHANT_CAT_WEIGHTS
            )
        return self._merchant_cat_map[mid]

    def _get_or_assign_merchant_country(self, mid: str) -> str:
        if mid not in self._merchant_country_map:
            self._merchant_country_map[mid] = self.rng.choice(COUNTRIES)
        return self._merchant_country_map[mid]

    def _get_or_assign_merchant_risk(self, mid: str) -> float:
        if mid not in self._merchant_risk_map:
            # ~10% of merchants are high-risk (score > 0.7)
            if self.rng.random() < self.cfg["high_risk_merchant_pct"]:
                score = round(float(self.rng.uniform(0.70, 1.00)), 4)
            else:
                score = round(float(self.rng.uniform(0.01, 0.50)), 4)
            self._merchant_risk_map[mid] = score
        return self._merchant_risk_map[mid]

    # ── velocity helpers ────────────────────────────────────────────

    def _compute_velocity(self, cid: str, event_time: datetime):
        """
        Returns (velocity_5min, velocity_1hour) for a customer at event_time.
        Maintains a sliding window of recent timestamps.
        """
        recent = self._customer_recent_1hr.get(cid, [])
        # Purge events older than 1 hour
        cutoff_1hr  = event_time - timedelta(hours=1)
        cutoff_5min = event_time - timedelta(minutes=5)
        recent = [t for t in recent if t >= cutoff_1hr]

        v5  = sum(1 for t in recent if t >= cutoff_5min)
        v1h = len(recent)

        # Add current event
        recent.append(event_time)
        self._customer_recent_1hr[cid] = recent
        return v5, v1h

    # ── chunk processor ─────────────────────────────────────────────

    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich one chunk of the PaySim dataframe.
        Returns the enriched chunk with all new columns.
        """
        rows = []

        for _, row in chunk.iterrows():
            cid  = row["nameOrig"]
            mid  = row["nameDest"]
            step = int(row["step"])
            amt  = float(row["amount"])

            # ── Timestamps ──────────────────────────────────────────
            event_time = step_to_datetime(
                step, self.cfg["epoch"], self.cfg["step_minutes"]
            )

            is_late = self.rng.random() < self.cfg["late_arrival_pct"]
            if is_late:
                delay_minutes = int(self.rng.integers(5, 360))
                arrival_time  = event_time + timedelta(minutes=delay_minutes)
            else:
                arrival_time = event_time + timedelta(
                    seconds=int(self.rng.integers(0, 30))
                )

            # ── Schema version ──────────────────────────────────────
            schema_ver = "v2" if step >= self.cfg["schema_v2_step"] else "v1"

            # ── Customer features ────────────────────────────────────
            segment      = self._get_or_assign_segment(cid)
            cust_country = self._get_or_assign_country(cid)
            device_id    = self._get_or_assign_device_id(cid)
            device_type  = self.rng.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS)
            device_age   = int(self.rng.integers(1, 1460))
            ip_addr      = self._get_or_assign_ip(cid)
            ip_risk      = round(float(self.rng.uniform(0.0, 1.0)), 4) if schema_ver == "v2" else np.nan

            # ── Merchant features ────────────────────────────────────
            merch_cat    = self._get_or_assign_merchant_cat(mid)
            merch_country= self._get_or_assign_merchant_country(mid)
            merch_risk   = self._get_or_assign_merchant_risk(mid)

            # ── Velocity ─────────────────────────────────────────────
            v5min, v1hr  = self._compute_velocity(cid, event_time)

            # ── Lifetime features ────────────────────────────────────
            prev_tx   = self._customer_tx_count.get(cid, 0)
            prev_spend= self._customer_lifetime_spend.get(cid, 0.0)
            self._customer_tx_count[cid]         = prev_tx + 1
            self._customer_lifetime_spend[cid]   = prev_spend + amt

            last_tx_time = self._customer_last_tx_time.get(cid)
            days_since   = None
            if last_tx_time:
                delta = event_time - last_tx_time
                days_since = round(delta.total_seconds() / 86400, 2)
            self._customer_last_tx_time[cid] = event_time

            # ── Geo anomaly ──────────────────────────────────────────
            # Flag if merchant_country differs from customer home country
            # AND the last transaction was within the anomaly window
            geo_flag = False
            if merch_country != cust_country:
                if last_tx_time:
                    elapsed_mins = (event_time - last_tx_time).total_seconds() / 60
                    if elapsed_mins < self.cfg["geo_anomaly_window_minutes"]:
                        geo_flag = True

            # ── New device flag ───────────────────────────────────────
            # Simple heuristic: device_age < 2 days → new device
            new_device_flag = device_age < 2

            # ── Chargeback simulation ─────────────────────────────────
            is_fraud = bool(row["isFraud"])
            if is_fraud:
                chargeback_rate = self.cfg["chargeback_rate_fraud"]
            else:
                chargeback_rate = self.cfg["chargeback_rate_legitimate"]

            if self.rng.random() < chargeback_rate:
                cb_status = "Filed"
                cb_delay  = int(self.rng.integers(7, 91))
            else:
                cb_status = "None"
                cb_delay  = np.nan

            # ── Seasonality / traffic spikes ──────────────────────────
            # Steps 168-192 = "payday week" – no data change, metadata only
            # Steps 336-360 = "holiday period"
            # (downstream systems can filter on these ranges)

            # ── Assemble enhanced row ─────────────────────────────────
            enriched = {
                # Original PaySim columns
                "step":              row["step"],
                "type":              row["type"],
                "amount":            amt,
                "nameOrig":          cid,
                "oldbalanceOrg":     row["oldbalanceOrg"],
                "newbalanceOrig":    row["newbalanceOrig"],
                "nameDest":          mid,
                "oldbalanceDest":    row["oldbalanceDest"],
                "newbalanceDest":    row["newbalanceDest"],
                "isFraud":           int(row["isFraud"]),
                "isFlaggedFraud":    int(row["isFlaggedFraud"]),

                # New columns
                "event_id":                     "EVT-" + uuid.uuid4().hex[:16].upper(),
                "event_time":                   event_time.isoformat(),
                "arrival_time":                 arrival_time.isoformat(),
                "schema_version":               schema_ver,
                "customer_segment":             segment,
                "device_id":                    device_id,
                "device_type":                  device_type,
                "device_age_days":              device_age,
                "customer_country":             cust_country,
                "merchant_country":             merch_country,
                "merchant_category":            merch_cat,
                "merchant_risk_score":          merch_risk,
                "ip_address":                   ip_addr,
                "ip_risk_score":                ip_risk,
                "chargeback_status":            cb_status,
                "chargeback_delay_days":        cb_delay,
                "days_since_last_transaction":  days_since,
                "transaction_velocity_5min":    v5min,
                "transaction_velocity_1hour":   v1hr,
                "customer_lifetime_transactions":self._customer_tx_count[cid],
                "customer_lifetime_spend":      round(self._customer_lifetime_spend[cid], 2),
                "geo_anomaly_flag":             int(geo_flag),
                "new_device_flag":              int(new_device_flag),
                "duplicate_event_flag":         0,   # set in post-processing
                "late_arrival_flag":            int(is_late),
                "out_of_order_flag":            0,   # set in post-processing
            }

            rows.append(enriched)
            if is_fraud:
                self._fraud_rows.append(enriched)

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# POST-PROCESSING  (inject operational challenges)
# ─────────────────────────────────────────────

def inject_duplicates(df: pd.DataFrame, pct: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Duplicate pct of rows. Each duplicate gets a new event_id but the same
    original_event_id so downstream dedup logic can detect it.
    Mirrors at-least-once delivery guarantees from Kinesis.
    """
    n = int(len(df) * pct)
    dup_idxs = rng.choice(df.index, size=n, replace=False)
    dups = df.loc[dup_idxs].copy()
    dups["event_id"] = ["EVT-" + uuid.uuid4().hex[:16].upper() for _ in range(len(dups))]
    dups["duplicate_event_flag"] = 1
    result = pd.concat([df, dups], ignore_index=True)
    result = result.sort_values("event_time").reset_index(drop=True)
    return result


def inject_out_of_order(df: pd.DataFrame, pct: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Shuffle pct of rows by randomly swapping their positions.
    Mirrors network jitter and Kinesis shard rebalancing scenarios.
    """
    n = int(len(df) * pct)
    idxs = rng.choice(df.index, size=n * 2, replace=False)
    pairs = idxs.reshape(-1, 2)
    df = df.copy()
    for i, j in pairs:
        df.loc[[i, j], "event_time"] = df.loc[[j, i], "event_time"].values
        df.at[i, "out_of_order_flag"] = 1
        df.at[j, "out_of_order_flag"] = 1
    return df


def simulate_concept_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gradually increase merchant_risk_score for fraud transactions over time.
    Simulates a new fraud ring that adapts its merchant network.
    Concept drift is visible to ML models trained on earlier time windows.
    """
    df = df.copy()
    max_step = df["step"].max()
    # Fraud rows in the second half of the dataset get a risk score boost
    mask = (df["isFraud"] == 1) & (df["step"] > max_step * 0.5)
    df.loc[mask, "merchant_risk_score"] = (
        df.loc[mask, "merchant_risk_score"] * 1.4
    ).clip(upper=1.0)
    return df


def apply_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag records that fall within known high-traffic windows.
    Downstream Glue jobs can use these tags for anomaly baselining.
    Steps 168-192 → payday spike.
    Steps 336-360 → holiday surge.
    """
    df = df.copy()
    df["traffic_period"] = "Normal"
    df.loc[(df["step"] >= 168) & (df["step"] <= 192), "traffic_period"] = "Payday"
    df.loc[(df["step"] >= 336) & (df["step"] <= 360), "traffic_period"] = "Holiday"
    return df


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_pipeline(input_path: str, outdir: str, chunk_size: int, replay_mode: str):
    set_seeds(CONFIG["seed"])
    os.makedirs(outdir, exist_ok=True)

    multiplier = CONFIG["replay_modes"].get(replay_mode, 1)
    print(f"[Pipeline] Starting | chunk_size={chunk_size:,} | replay={replay_mode} ({multiplier}x)")

    enhancer = PaySimEnhancer(CONFIG)
    rng_post = np.random.default_rng(CONFIG["seed"] + 99)

    enhanced_path = os.path.join(outdir, "enhanced_transactions.csv")
    first_chunk = True
    total_rows  = 0

    # ── Chunk-wise enrichment ────────────────────────────────────────
    print("[Pipeline] Enriching transactions chunk by chunk ...")
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        enriched_chunk = enhancer.process_chunk(chunk)
        mode = "w" if first_chunk else "a"
        enriched_chunk.to_csv(enhanced_path, index=False, mode=mode,
                              header=first_chunk)
        first_chunk   = False
        total_rows   += len(enriched_chunk)
        print(f"  chunk {i+1:03d} → {total_rows:,} rows written")

    print(f"[Pipeline] Base enrichment complete: {total_rows:,} rows")

    # ── Post-processing (load full file for set operations) ──────────
    print("[Pipeline] Loading enriched file for post-processing ...")
    df = pd.read_csv(enhanced_path)

    print("[Pipeline] Injecting duplicate events ...")
    df = inject_duplicates(df, CONFIG["duplicate_event_pct"], rng_post)

    print("[Pipeline] Injecting out-of-order events ...")
    df = inject_out_of_order(df, CONFIG["out_of_order_pct"], rng_post)

    print("[Pipeline] Injecting missing values ...")
    df = inject_missing_values(df, CONFIG["missing_value_pct"], rng_post)

    print("[Pipeline] Injecting corrupted records ...")
    df = inject_corrupted_records(df, CONFIG["corrupted_record_pct"], rng_post)

    print("[Pipeline] Applying concept drift ...")
    df = simulate_concept_drift(df)

    print("[Pipeline] Applying seasonality tags ...")
    df = apply_seasonality(df)

    # ── Replay mode ──────────────────────────────────────────────────
    if multiplier > 1:
        print(f"[Pipeline] Generating {multiplier}x replay dataset ...")
        copies = [df]
        for rep in range(1, multiplier):
            rep_df = df.copy()
            rep_df["event_id"] = ["EVT-" + uuid.uuid4().hex[:16].upper()
                                  for _ in range(len(rep_df))]
            rep_df["replay_round"] = rep
            copies.append(rep_df)
        df = pd.concat(copies, ignore_index=True)
        print(f"  Replay dataset size: {len(df):,} rows")
    else:
        df["replay_round"] = 0

    df.to_csv(enhanced_path, index=False)
    print(f"[Pipeline] enhanced_transactions.csv → {len(df):,} rows")

    # ── Dimension tables ─────────────────────────────────────────────
    print("[Pipeline] Building dimension tables ...")
    rng_dim = np.random.default_rng(CONFIG["seed"] + 1)

    customer_ids = list(enhancer._customer_segment_map.keys())
    customers_df = build_customers_table(
        customer_ids,
        enhancer._customer_segment_map,
        enhancer._customer_country_map,
        enhancer._customer_device_slot,   # repurposed as device_type proxy
        rng_dim,
        CONFIG["epoch"],
    )
    customers_df.to_csv(os.path.join(outdir, "customers.csv"), index=False)
    print(f"  customers.csv      → {len(customers_df):,} rows")

    merchant_ids = list(enhancer._merchant_cat_map.keys())
    merchants_df = build_merchants_table(
        merchant_ids,
        enhancer._merchant_cat_map,
        enhancer._merchant_country_map,
        enhancer._merchant_risk_map,
        rng_dim,
        CONFIG["epoch"],
    )
    merchants_df.to_csv(os.path.join(outdir, "merchants.csv"), index=False)
    print(f"  merchants.csv      → {len(merchants_df):,} rows")

    devices_df = build_devices_table(enhancer.all_device_ids, rng_dim)
    devices_df.to_csv(os.path.join(outdir, "devices.csv"), index=False)
    print(f"  devices.csv        → {len(devices_df):,} rows")

    # Chargebacks: fraud rows collected during enrichment
    fraud_df      = pd.DataFrame(enhancer._fraud_rows)
    chargebacks_df= build_chargebacks_table(fraud_df, rng_dim)
    chargebacks_df.to_csv(os.path.join(outdir, "chargebacks.csv"), index=False)
    print(f"  chargebacks.csv    → {len(chargebacks_df):,} rows")

    print("\n[Pipeline] ✓ All files written to:", outdir)
    return df


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaySim Enhancement Pipeline")
    parser.add_argument("--input",       default="/mnt/user-data/uploads/PS_20174392719_1491204439457_log.csv")
    parser.add_argument("--outdir",      default="/home/claude/paysim_enhanced/output")
    parser.add_argument("--chunk-size",  type=int, default=200_000)
    parser.add_argument("--replay-mode", default="1x", choices=["1x", "10x", "100x"])
    args = parser.parse_args()

    run_pipeline(args.input, args.outdir, args.chunk_size, args.replay_mode)
