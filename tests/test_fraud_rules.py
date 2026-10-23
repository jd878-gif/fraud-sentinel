"""Unit tests for the Lambda rule engine (no AWS calls are made)."""
import pytest

from lambda_function import run_fraud_rules

CLEAN = {"type": "PAYMENT", "amount": 120.0, "oldbalanceOrg": 5000.0,
         "newbalanceOrig": 4880.0, "merchant_risk_score": 0.10,
         "geo_anomaly_flag": 0, "new_device_flag": 0, "isFraud": 0}
NO_VELOCITY = {"velocity_5min_realtime": 0, "velocity_1hour_realtime": 0}


def test_clean_transaction_not_flagged():
    result = run_fraud_rules(CLEAN, NO_VELOCITY)
    assert result == {"triggered_rules": [], "risk_score": 0, "is_flagged": False}


@pytest.mark.parametrize("is_fraud", [0, 1])
def test_ground_truth_label_never_changes_the_decision(is_fraud):
    """Regression test: the rules must not read the PaySim answer key."""
    result = run_fraud_rules({**CLEAN, "isFraud": is_fraud}, NO_VELOCITY)
    assert result["is_flagged"] is False
    assert "PAYSIM_FRAUD_LABEL" not in result["triggered_rules"]


def test_high_velocity_card_testing():
    result = run_fraud_rules(CLEAN, {"velocity_5min_realtime": 5})
    assert result["triggered_rules"] == ["HIGH_VELOCITY_5MIN"]
    assert result["risk_score"] == 4


def test_velocity_just_below_threshold():
    assert run_fraud_rules(CLEAN, {"velocity_5min_realtime": 4})["is_flagged"] is False


def test_high_risk_merchant_boundary():
    assert run_fraud_rules({**CLEAN, "merchant_risk_score": 0.75}, NO_VELOCITY)["risk_score"] == 3
    assert run_fraud_rules({**CLEAN, "merchant_risk_score": 0.7499}, NO_VELOCITY)["risk_score"] == 0


def test_geo_anomaly_needs_new_device_too():
    only_geo = run_fraud_rules({**CLEAN, "geo_anomaly_flag": 1}, NO_VELOCITY)
    both = run_fraud_rules({**CLEAN, "geo_anomaly_flag": 1, "new_device_flag": 1}, NO_VELOCITY)
    assert only_geo["is_flagged"] is False
    assert both["triggered_rules"] == ["GEO_ANOMALY_NEW_DEVICE"] and both["risk_score"] == 5


def test_balance_mismatch_only_on_transfer():
    tx = {**CLEAN, "amount": 9000.0, "oldbalanceOrg": 9000.0, "newbalanceOrig": 9000.0}
    assert run_fraud_rules({**tx, "type": "TRANSFER"}, NO_VELOCITY)["triggered_rules"] == [
        "BALANCE_MISMATCH_TRANSFER"]
    assert run_fraud_rules({**tx, "type": "CASH_OUT"}, NO_VELOCITY)["is_flagged"] is False


def test_scores_are_additive():
    tx = {**CLEAN, "merchant_risk_score": 0.9, "geo_anomaly_flag": 1, "new_device_flag": 1}
    result = run_fraud_rules(tx, {"velocity_5min_realtime": 7})
    assert result["risk_score"] == 4 + 3 + 5


def test_missing_and_null_fields_do_not_crash():
    result = run_fraud_rules({"type": None, "amount": None, "merchant_risk_score": None}, {})
    assert result["is_flagged"] is False
