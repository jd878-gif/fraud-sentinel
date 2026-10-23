"""Handler tests with in-memory fakes: idempotency and partial batch failure."""
import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

import lambda_function as lf


class FakeDedupTable:
    def __init__(self):
        self.items = {}

    def put_item(self, Item, ConditionExpression=None):
        if ConditionExpression and Item["event_id"] in self.items:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
        self.items[Item["event_id"]] = dict(Item)

    def update_item(self, Key, ExpressionAttributeValues, **_):
        self.items[Key["event_id"]]["status"] = ExpressionAttributeValues[":done"]

    def delete_item(self, Key):
        self.items.pop(Key["event_id"], None)


@pytest.fixture
def env(monkeypatch):
    table = FakeDedupTable()
    calls = {"stored": [], "alerts": []}
    monkeypatch.setattr(lf, "_dedup_table", table)
    monkeypatch.setattr(lf, "get_and_update_velocity",
                        lambda **_: {"velocity_5min_realtime": 0, "velocity_1hour_realtime": 0})
    monkeypatch.setattr(lf, "write_to_feature_store", lambda *a: calls["stored"].append(a[0]))
    monkeypatch.setattr(lf, "write_to_s3_bronze", lambda *a: None)
    monkeypatch.setattr(lf, "publish_fraud_alert", lambda *a: calls["alerts"].append(a[0]))
    return table, calls


def record(msg_id, event_id, **payload):
    body = {"metadata": {"event_id": event_id, "event_time": "2023-01-02T03:00:00"},
            "payload": {"type": "PAYMENT", "amount": 10.0, "nameOrig": "C1", **payload}}
    return {"messageId": msg_id, "body": json.dumps(body)}


CTX = SimpleNamespace(aws_request_id="test")


def test_duplicate_message_processed_once(env):
    table, calls = env
    event = {"Records": [record("m1", "EVT-1"), record("m2", "EVT-1")]}
    assert lf.lambda_handler(event, CTX) == {"batchItemFailures": []}
    assert calls["stored"] == ["EVT-1"]
    assert table.items["EVT-1"]["status"] == "DONE"


def test_failure_reports_only_that_message_and_releases_claim(env, monkeypatch):
    table, calls = env

    def boom(*args):
        if args[0] == "EVT-BAD":
            raise RuntimeError("S3 down")

    monkeypatch.setattr(lf, "write_to_s3_bronze", boom)
    event = {"Records": [record("ok", "EVT-OK"), record("bad", "EVT-BAD")]}
    assert lf.lambda_handler(event, CTX) == {"batchItemFailures": [{"itemIdentifier": "bad"}]}
    assert "EVT-BAD" not in table.items          # claim released so the retry can run
    assert table.items["EVT-OK"]["status"] == "DONE"


def test_fraud_label_alone_does_not_trigger_alert(env):
    _, calls = env
    lf.lambda_handler({"Records": [record("m1", "EVT-9", isFraud=1)]}, CTX)
    assert calls["alerts"] == []


def test_rule_hit_triggers_alert(env):
    _, calls = env
    lf.lambda_handler({"Records": [record("m1", "EVT-7", merchant_risk_score=0.9)]}, CTX)
    assert calls["alerts"] == ["EVT-7"]
