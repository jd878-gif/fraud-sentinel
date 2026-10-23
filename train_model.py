"""
train_model.py
==============
Leak-free, reproducible training and evaluation for the FraudSentinel model.

Replaces the old evaluation (random split on ~1,000 Silver rows, which reported
AUC 1.00). What changed and why:

  1. Trains on the full enhanced PaySim dataset, not a 1,000-row sample.
     With PaySim's ~0.13% fraud rate a tiny sample has only a handful of
     fraud cases, so any AUC on it is noise.
  2. Time-based split on `step` (train = first 70% of time, validation = next
     15%, test = last 15%). A fraud model is always used on *future*
     transactions, so a random shuffle overstates performance.
  3. Only decision-time features in the deployed model. Post-transaction
     balances (PaySim simulator artifacts) are measured in an ablation, not
     shipped. Ground truth / delayed fields (isFraud, chargebacks) are blocked.
  4. Reports metrics that matter under heavy class imbalance: PR-AUC (average
     precision) with a bootstrap 95% CI, recall at >=90% precision, and the
     confusion matrix at a threshold chosen on validation (never on test).
  5. Compares the model against the rule engine on the same test set, so the
     README can say whether ML actually beats the rules.

Usage
-----
    # 1. produce the enhanced dataset (once)
    python enhancement_pipeline.py --input PS_20174392719_1491204439457_log.csv \
                                   --outdir data/enhanced
    # 2. train + evaluate (writes artifacts/)
    python train_model.py --data data/enhanced/enhanced_transactions.csv
    # 3. optional: package + deploy the decision-time model to SageMaker
    python train_model.py --data data/enhanced/enhanced_transactions.csv --deploy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from fraud_features import (
    DECISION_TIME_FEATURES,
    NEVER_FEATURES,
    POST_TRANSACTION_FEATURES,
    build_feature_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("train_model")

LABEL = "isFraud"
TARGET_PRECISION = 0.90
SEED = 42

EXPERIMENTS = {
    # name: (feature list, shipped?)
    "decision_time (deployed)": (DECISION_TIME_FEATURES, True),
    "decision_time - merchant_risk_score": (
        [f for f in DECISION_TIME_FEATURES if f != "merchant_risk_score"], False),
    # No balance information at all: shows how much of the signal is PaySim's
    # simulator pattern (fraud empties the sender's account) vs behaviour.
    "no balance features": (
        [f for f in DECISION_TIME_FEATURES
         if f not in {"old_balance_orig", "amount_to_balance_ratio", "drains_account"}], False),
    "+ post-transaction balances (leaky)": (
        DECISION_TIME_FEATURES + POST_TRANSACTION_FEATURES, False),
}


# ─────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────

def load_and_clean(path: str, max_rows: int | None = None) -> pd.DataFrame:
    log.info("Loading %s ...", path)
    df = pd.read_csv(path, nrows=max_rows, low_memory=False)
    n0 = len(df)

    if "event_id" in df.columns:                       # injected duplicate events
        df = df.drop_duplicates(subset="event_id", keep="first")
    df = df.dropna(subset=["step", "type", "amount", LABEL])
    df = df[pd.to_numeric(df["amount"], errors="coerce") >= 0]

    if "merchant_risk_score" in df.columns:            # injected corrupted values (e.g. 99.9)
        mrs = pd.to_numeric(df["merchant_risk_score"], errors="coerce")
        df["merchant_risk_score"] = mrs.where(mrs.between(0, 1))

    df[LABEL] = df[LABEL].astype(int)
    df["step"] = df["step"].astype(int)
    log.info("  rows: %d raw -> %d clean | fraud: %d (%.3f%%)",
             n0, len(df), df[LABEL].sum(), 100 * df[LABEL].mean())
    return df.reset_index(drop=True)


def time_split(df: pd.DataFrame, train_q: float = 0.70, val_q: float = 0.85):
    """Split on time (step), never shuffle across the boundary."""
    t1, t2 = df["step"].quantile([train_q, val_q]).astype(int).tolist()
    train = df[df["step"] <= t1]
    val = df[(df["step"] > t1) & (df["step"] <= t2)]
    test = df[df["step"] > t2]
    for name, part in (("train", train), ("val", val), ("test", test)):
        log.info("  %-5s steps %3d-%3d | rows %8d | fraud %6d",
                 name, part["step"].min(), part["step"].max(), len(part), part[LABEL].sum())
    if min(part[LABEL].sum() for part in (train, val, test)) < 20:
        log.warning("Fewer than 20 fraud cases in a split; metrics will be unstable.")
    return train, val, test


# ─────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────

def recall_at_precision(y, scores, target=TARGET_PRECISION) -> float:
    precision, recall, _ = precision_recall_curve(y, scores)
    ok = recall[precision >= target]
    return float(ok.max()) if ok.size else 0.0


def threshold_for_precision(y, scores, target=TARGET_PRECISION) -> float:
    """Lowest threshold whose validation precision reaches `target`."""
    precision, _, thresholds = precision_recall_curve(y, scores)
    ok = np.where(precision[:-1] >= target)[0]
    return float(thresholds[ok[0]]) if ok.size else 0.5


def bootstrap_pr_auc(y, scores, n=200, seed=SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, scores = np.asarray(y), np.asarray(scores)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(average_precision_score(y[idx], scores[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


@dataclass
class Result:
    name: str
    pr_auc: float
    pr_auc_ci: tuple[float, float]
    roc_auc: float
    recall_at_p90: float
    threshold: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int

    def row(self) -> str:
        lo, hi = self.pr_auc_ci
        return (f"| {self.name} | {self.pr_auc:.3f} ({lo:.3f}-{hi:.3f}) | {self.roc_auc:.3f} | "
                f"{self.recall_at_p90:.1%} | {self.precision:.1%} | {self.recall:.1%} | "
                f"{self.tp} / {self.fp} / {self.fn} |")


def evaluate(name, y_val, s_val, y_test, s_test) -> Result:
    thr = threshold_for_precision(y_val, s_val)          # chosen on validation only
    pred = (s_test >= thr).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    return Result(
        name=name,
        pr_auc=float(average_precision_score(y_test, s_test)),
        pr_auc_ci=bootstrap_pr_auc(y_test, s_test),
        roc_auc=float(roc_auc_score(y_test, s_test)),
        recall_at_p90=recall_at_precision(y_test, s_test),
        threshold=thr,
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


# ─────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────

def rule_engine_scores(df: pd.DataFrame) -> np.ndarray:
    """Same additive rules as lambda_function.run_fraud_rules (minus the label)."""
    f = build_feature_frame(df)
    score = (
        np.where(f["velocity_5min"] >= 5, 4, 0)
        + np.where(f["merchant_risk_score"] >= 0.75, 3, 0)
        + np.where((f["geo_anomaly_flag"] == 1) & (f["new_device_flag"] == 1), 5, 0)
        + np.where((f["balance_mismatch_flag"] == 1) & (f["old_balance_orig"] > 0), 4, 0)
    )
    return score.astype(float)


def train_xgb(X_tr, y_tr, X_val, y_val, feature_names) -> xgb.Booster:
    bad = NEVER_FEATURES.intersection(feature_names)
    if bad:
        raise ValueError(f"Label leakage: {sorted(bad)} must never be model features")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": float((y_tr == 0).sum() / max(y_tr.sum(), 1)),
        "seed": SEED,
        "verbosity": 0,
    }
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    return xgb.train(params, dtr, num_boost_round=1000, evals=[(dval, "val")],
                     early_stopping_rounds=50, verbose_eval=False)


def predict(model: xgb.Booster, X, feature_names) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=feature_names)
    return model.predict(d, iteration_range=(0, model.best_iteration + 1))


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def run(data_path: str, outdir: str, max_rows: int | None = None) -> dict:
    os.makedirs(outdir, exist_ok=True)
    df = load_and_clean(data_path, max_rows)
    train, val, test = time_split(df)
    F_tr, F_val, F_te = (build_feature_frame(p) for p in (train, val, test))
    y_tr, y_val, y_te = (p[LABEL].to_numpy() for p in (train, val, test))

    results: list[Result] = []
    test_scores: dict[str, np.ndarray] = {}
    deployed = None

    # Baseline: the rule engine
    test_scores["rule engine (baseline)"] = rule_engine_scores(test)
    results.append(evaluate("rule engine (baseline)", y_val, rule_engine_scores(val),
                            y_te, test_scores["rule engine (baseline)"]))

    for name, (cols, shipped) in EXPERIMENTS.items():
        log.info("Training: %s (%d features)", name, len(cols))
        model = train_xgb(F_tr[cols].to_numpy(), y_tr, F_val[cols].to_numpy(), y_val, cols)
        s_val = predict(model, F_val[cols].to_numpy(), cols)
        s_te = predict(model, F_te[cols].to_numpy(), cols)
        test_scores[name] = s_te
        res = evaluate(name, y_val, s_val, y_te, s_te)
        results.append(res)
        log.info("  test PR-AUC %.3f | ROC-AUC %.3f | recall@P90 %.1f%%",
                 res.pr_auc, res.roc_auc, 100 * res.recall_at_p90)
        if shipped:
            deployed = (model, cols, res)

    model, cols, res = deployed
    model.save_model(os.path.join(outdir, "model.json"))
    importance = model.get_score(importance_type="gain")
    meta = {
        "feature_columns": cols,
        "threshold": res.threshold,
        "target_precision": TARGET_PRECISION,
        "best_iteration": int(model.best_iteration),
        "xgboost_version": xgb.__version__,
        "split": "time-based on step: 70% train / 15% val / 15% test",
        "feature_importance_gain": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }
    with open(os.path.join(outdir, "model_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump([r.__dict__ for r in results], fh, indent=2)

    table = [
        "| Model | Test PR-AUC (95% CI) | ROC-AUC | Recall @ 90% precision "
        "| Precision | Recall | TP / FP / FN |",
        "|---|---|---|---|---|---|---|",
        *(r.row() for r in results),
    ]
    with open(os.path.join(outdir, "results.md"), "w") as fh:
        fh.write("\n".join(table) + "\n")
    print("\n".join(table))
    _plot_pr_curves(outdir, y_te, test_scores)
    return {"results": results, "metadata": meta}


def _plot_pr_curves(outdir, y_te, test_scores: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed; skipping PR curve plot")
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, scores in test_scores.items():
        p, r, _ = precision_recall_curve(y_te, scores)
        ax.plot(r, p, label=name, linestyle="--" if "rule" in name else "-")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall, test set (last 15% of time)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pr_curves.png"), dpi=150)
    plt.close(fig)


def deploy(outdir: str):
    """Package the decision-time model and deploy it with the existing SageMaker code."""
    import boto3

    import sagemaker_local_train_deploy as smd

    meta = json.load(open(os.path.join(outdir, "model_metadata.json")))
    if not meta["xgboost_version"].startswith("1.7"):
        log.warning("Model trained with xgboost %s but the SageMaker container is %s. "
                    "Retrain with `pip install xgboost==1.7.6` before deploying.",
                    meta["xgboost_version"], smd.XGBOOST_IMAGE.split(":")[-1])
    booster = xgb.Booster()
    booster.load_model(os.path.join(outdir, "model.json"))
    s3 = boto3.client("s3", region_name=smd.REGION)
    model_s3 = smd.package_and_upload_model(booster, s3)
    s3.upload_file(os.path.join(outdir, "model_metadata.json"), smd.S3_BUCKET,
                   f"{smd.MODEL_PREFIX}/output/model_metadata.json")
    sm = boto3.client("sagemaker", region_name=smd.REGION)
    smd.deploy_serverless_endpoint(sm, smd.create_sagemaker_model(sm, model_s3))
    log.info("Set Lambda env ML_ENDPOINT_NAME=%s and ML_THRESHOLD=%.4f",
             smd.ENDPOINT_NAME, meta["threshold"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="enhanced_transactions.csv or raw PaySim CSV")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--max-rows", type=int, default=None, help="debug: read only the first N rows")
    ap.add_argument("--deploy", action="store_true")
    args = ap.parse_args()
    run(args.data, args.outdir, args.max_rows)
    if args.deploy:
        deploy(args.outdir)
