"""Training/serving parity and leakage guards for fraud_features.py."""
import math

import numpy as np
import pandas as pd
import pytest

from fraud_features import (
    DECISION_TIME_FEATURES,
    NEVER_FEATURES,
    POST_TRANSACTION_FEATURES,
    build_feature_dict,
    build_feature_frame,
    build_feature_vector,
    to_csv_row,
)

ROWS = [
    dict(step=25, type="TRANSFER", amount=181.0, oldbalanceOrg=181.0, newbalanceOrig=0.0,
         oldbalanceDest=0.0, newbalanceDest=0.0, transaction_velocity_5min=0,
         transaction_velocity_1hour=1, geo_anomaly_flag=0, new_device_flag=1,
         merchant_risk_score=0.82, isFraud=1),
    dict(step=1, type="PAYMENT", amount=9839.64, oldbalanceOrg=170136.0,
         newbalanceOrig=160296.36, oldbalanceDest=0.0, newbalanceDest=0.0,
         transaction_velocity_5min=2, transaction_velocity_1hour=3, geo_anomaly_flag=1,
         new_device_flag=0, merchant_risk_score=None, isFraud=0),
    dict(step=743, type="CASH_OUT", amount=0.0, oldbalanceOrg=0.0, newbalanceOrig=0.0,
         oldbalanceDest=10.0, newbalanceDest=10.0, transaction_velocity_5min=None,
         transaction_velocity_1hour=None, geo_anomaly_flag=None, new_device_flag=None,
         merchant_risk_score=float("nan"), isFraud=0),
]


def test_no_leaky_columns_in_feature_lists():
    assert not NEVER_FEATURES.intersection(DECISION_TIME_FEATURES)
    assert not NEVER_FEATURES.intersection(POST_TRANSACTION_FEATURES)
    assert not set(DECISION_TIME_FEATURES) & set(POST_TRANSACTION_FEATURES)


@pytest.mark.parametrize("row", ROWS)
def test_training_and_serving_features_match(row):
    frame = build_feature_frame(pd.DataFrame([row])).iloc[0]
    served = build_feature_dict(
        row,
        velocity_5min=row["transaction_velocity_5min"] or 0,
        velocity_1hour=row["transaction_velocity_1hour"] or 0,
        hour_of_day=row["step"] % 24,
    )
    for col in DECISION_TIME_FEATURES + POST_TRANSACTION_FEATURES:
        assert math.isclose(frame[col], served[col], rel_tol=1e-9, abs_tol=1e-9), col


def test_label_does_not_affect_features():
    a = build_feature_frame(pd.DataFrame([{**ROWS[0], "isFraud": 0}]))
    b = build_feature_frame(pd.DataFrame([{**ROWS[0], "isFraud": 1}]))
    pd.testing.assert_frame_equal(a, b)


def test_one_hot_type():
    feats = build_feature_dict({"type": "TRANSFER"}, 0, 0, 0)
    assert feats["type_TRANSFER"] == 1.0
    assert sum(v for k, v in feats.items() if k.startswith("type_")) == 1.0


def test_vector_order_and_csv():
    vec = build_feature_vector(ROWS[0], 0, 1, 1, DECISION_TIME_FEATURES)
    assert len(vec) == len(DECISION_TIME_FEATURES)
    assert vec[DECISION_TIME_FEATURES.index("amount")] == 181.0
    assert len(to_csv_row(vec).split(",")) == len(vec)
    assert np.isfinite(vec).all()
