"""End-to-end smoke test of train_model.py on a small synthetic PaySim-like dataset."""
import json

import numpy as np
import pandas as pd
import pytest

import train_model


def synthetic_paysim(n=6000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = np.sort(rng.integers(1, 300, n))
    tx_type = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], n)
    old = rng.uniform(0, 50_000, n).round(2)
    fraud = (np.isin(tx_type, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < 0.06)).astype(int)
    amount = np.where(fraud == 1, old, rng.uniform(1, 5_000, n)).round(2)
    return pd.DataFrame({
        "event_id": [f"EVT-{i}" for i in range(n)],
        "step": step, "type": tx_type, "amount": amount,
        "oldbalanceOrg": old, "newbalanceOrig": np.maximum(old - amount, 0),
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "transaction_velocity_5min": rng.integers(0, 3, n),
        "transaction_velocity_1hour": rng.integers(0, 6, n),
        "geo_anomaly_flag": rng.integers(0, 2, n), "new_device_flag": rng.integers(0, 2, n),
        "merchant_risk_score": rng.uniform(0, 1, n),
        "chargeback_status": np.where(fraud == 1, "Filed", "None"),
        "isFraud": fraud,
    })


@pytest.fixture
def csv_path(tmp_path):
    df = synthetic_paysim()
    df = pd.concat([df, df.head(50)])           # injected duplicates must be removed
    path = tmp_path / "enhanced.csv"
    df.to_csv(path, index=False)
    return path


def test_time_split_never_leaks_future_into_training(csv_path):
    df = train_model.load_and_clean(str(csv_path))
    train, val, test = train_model.time_split(df)
    assert train["step"].max() < val["step"].min() <= val["step"].max() < test["step"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_duplicates_removed(csv_path):
    assert train_model.load_and_clean(str(csv_path))["event_id"].is_unique


def test_leaky_feature_is_rejected():
    X = np.zeros((10, 1))
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="leakage"):
        train_model.train_xgb(X, y, X, y, ["chargeback_status"])


def test_run_writes_artifacts(csv_path, tmp_path):
    out = tmp_path / "artifacts"
    train_model.run(str(csv_path), str(out))
    meta = json.loads((out / "model_metadata.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    assert meta["feature_columns"] == train_model.DECISION_TIME_FEATURES
    assert 0.0 < meta["threshold"] < 1.0 or meta["threshold"] == 0.5
    assert {m["name"] for m in metrics} >= {"rule engine (baseline)", "decision_time (deployed)"}
    assert all(0.0 <= m["pr_auc"] <= 1.0 for m in metrics)
    assert (out / "model.json").exists() and (out / "results.md").exists()
