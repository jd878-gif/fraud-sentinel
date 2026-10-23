"""
fraud_features.py
=================
Single source of truth for the model's features, used by BOTH training
(train_model.py, vectorised with pandas) and serving (lambda_function.py,
pure Python so the Lambda package needs no pandas).

Keeping one definition prevents training/serving skew: tests/test_features.py
asserts both paths produce identical values for the same transaction.

Feature groups
--------------
DECISION_TIME_FEATURES
    Information a payment system genuinely has *before* it approves a
    transaction: amount, type, the sender's balance before the transaction,
    time of day, velocity counters, device / geo / merchant signals.

POST_TRANSACTION_FEATURES
    Balances *after* the transaction. In PaySim these encode simulator
    artifacts (fraudulent transfers empty the account in a very regular way),
    so a model that uses them looks near-perfect. They are only used in the
    ablation study to show that effect; they are NOT in the deployed model.

NEVER_FEATURES
    Ground truth or things only known later (chargebacks arrive 7-90 days
    after the transaction). Using any of these is label leakage.
    train_model.py refuses to train if one of them appears in the feature list.
"""

from __future__ import annotations

import math

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

DECISION_TIME_FEATURES = [
    "amount",
    "log_amount",
    "old_balance_orig",
    "amount_to_balance_ratio",
    "drains_account",
    "hour_of_day",
    "velocity_5min",
    "velocity_1hour",
    "geo_anomaly_flag",
    "new_device_flag",
    "merchant_risk_score",
] + [f"type_{t}" for t in TRANSACTION_TYPES]

POST_TRANSACTION_FEATURES = [
    "new_balance_orig",
    "old_balance_dest",
    "new_balance_dest",
    "orig_balance_error",
    "balance_mismatch_flag",
]

NEVER_FEATURES = {
    "isFraud", "is_fraud", "isFlaggedFraud",
    "chargeback_status", "chargeback_delay_days",
    "rule_signal_score",          # built from other features + used for alert ranking
    "event_id", "nameOrig", "nameDest",
}


def _f(value, default: float = 0.0) -> float:
    """Float conversion that treats None / '' / NaN as `default`."""
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


# ─────────────────────────────────────────────────────────────────
# SERVING PATH (pure Python, used inside Lambda)
# ─────────────────────────────────────────────────────────────────

def build_feature_dict(payload: dict, velocity_5min: float, velocity_1hour: float,
                       hour_of_day: int) -> dict:
    """Compute every feature (both groups) for one transaction."""
    amount = _f(payload.get("amount"))
    old_orig = _f(payload.get("oldbalanceOrg"))
    new_orig = _f(payload.get("newbalanceOrig"))
    tx_type = str(payload.get("type") or "")

    feats = {
        "amount": amount,
        "log_amount": math.log1p(max(amount, 0.0)),
        "old_balance_orig": old_orig,
        "amount_to_balance_ratio": amount / (old_orig + 1.0),
        "drains_account": 1.0 if old_orig > 0 and amount >= old_orig else 0.0,
        "hour_of_day": float(hour_of_day),
        "velocity_5min": _f(velocity_5min),
        "velocity_1hour": _f(velocity_1hour),
        "geo_anomaly_flag": _f(payload.get("geo_anomaly_flag")),
        "new_device_flag": _f(payload.get("new_device_flag")),
        "merchant_risk_score": _f(payload.get("merchant_risk_score")),
        # post-transaction (ablation only)
        "new_balance_orig": new_orig,
        "old_balance_dest": _f(payload.get("oldbalanceDest")),
        "new_balance_dest": _f(payload.get("newbalanceDest")),
        "orig_balance_error": old_orig - amount - new_orig,
        "balance_mismatch_flag": 1.0 if (tx_type == "TRANSFER" and amount > 0
                                         and new_orig == old_orig) else 0.0,
    }
    for t in TRANSACTION_TYPES:
        feats[f"type_{t}"] = 1.0 if tx_type == t else 0.0
    return feats


def build_feature_vector(payload: dict, velocity_5min: float, velocity_1hour: float,
                         hour_of_day: int, feature_columns: list[str]) -> list[float]:
    feats = build_feature_dict(payload, velocity_5min, velocity_1hour, hour_of_day)
    return [feats[c] for c in feature_columns]


def to_csv_row(values: list[float]) -> str:
    """SageMaker built-in XGBoost expects one CSV row, features only, no label."""
    return ",".join(repr(float(v)) for v in values)


# ─────────────────────────────────────────────────────────────────
# TRAINING PATH (vectorised pandas)
# ─────────────────────────────────────────────────────────────────

def build_feature_frame(df):
    """
    Vectorised equivalent of build_feature_dict for the enhanced PaySim CSV
    (output of enhancement_pipeline.py). Returns a DataFrame of features.
    """
    import numpy as np
    import pandas as pd

    def num(col):
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    amount = num("amount")
    old_orig = num("oldbalanceOrg")
    new_orig = num("newbalanceOrig")
    tx_type = df["type"].astype(str) if "type" in df.columns else pd.Series("", index=df.index)

    out = pd.DataFrame(index=df.index)
    out["amount"] = amount
    out["log_amount"] = np.log1p(amount.clip(lower=0.0))
    out["old_balance_orig"] = old_orig
    out["amount_to_balance_ratio"] = amount / (old_orig + 1.0)
    out["drains_account"] = ((old_orig > 0) & (amount >= old_orig)).astype(float)
    # each PaySim step is one hour starting at midnight -> hour = step % 24
    out["hour_of_day"] = (num("step").astype(int) % 24).astype(float)
    out["velocity_5min"] = num("transaction_velocity_5min")
    out["velocity_1hour"] = num("transaction_velocity_1hour")
    out["geo_anomaly_flag"] = num("geo_anomaly_flag")
    out["new_device_flag"] = num("new_device_flag")
    out["merchant_risk_score"] = num("merchant_risk_score")

    out["new_balance_orig"] = new_orig
    out["old_balance_dest"] = num("oldbalanceDest")
    out["new_balance_dest"] = num("newbalanceDest")
    out["orig_balance_error"] = old_orig - amount - new_orig
    out["balance_mismatch_flag"] = ((tx_type == "TRANSFER") & (amount > 0)
                                    & (new_orig == old_orig)).astype(float)
    for t in TRANSACTION_TYPES:
        out[f"type_{t}"] = (tx_type == t).astype(float)
    return out
