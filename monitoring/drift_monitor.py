"""
monitoring/drift_monitor.py
Data drift and model performance monitoring for Laptop Price Predictor
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import logging
from datetime import datetime
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import r2_score, mean_squared_error

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REFERENCE_PATH  = "monitoring/reference_data.csv"
DRIFT_LOG_PATH  = "monitoring/drift_logs.json"
ALERT_LOG_PATH  = "monitoring/alerts.json"
MODEL_PATH      = "models/best_model.pkl"

# Thresholds
KS_THRESHOLD       = 0.05   # p-value below this → drift detected (numeric)
CHI2_THRESHOLD     = 0.05   # p-value below this → drift detected (categorical)
R2_DROP_THRESHOLD  = 0.10   # R² drop larger than this → model drift


# ─── Reference Data Management ───────────────────────────────────────────────

def save_reference_data(df: pd.DataFrame):
    """Save training data as the reference baseline for drift comparison"""
    os.makedirs("monitoring", exist_ok=True)
    df.to_csv(REFERENCE_PATH, index=False)
    logger.info(f"✅ Reference data saved to {REFERENCE_PATH} ({len(df)} rows)")


def load_reference_data() -> pd.DataFrame:
    """Load saved reference baseline"""
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f"Reference data not found at {REFERENCE_PATH}. "
            "Call save_reference_data() after training."
        )
    return pd.read_csv(REFERENCE_PATH)


# ─── Numeric Feature Drift (KS Test) ─────────────────────────────────────────

def detect_numeric_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
    threshold: float = KS_THRESHOLD
) -> dict:
    """
    Kolmogorov-Smirnov test for numeric feature drift.

    Returns:
        dict with drift status, KS statistic, and p-value
    """
    reference = reference.dropna()
    current   = current.dropna()

    if len(current) < 10:
        return {"feature": feature_name, "drift": False, "reason": "insufficient data"}

    ks_stat, p_value = ks_2samp(reference, current)
    drift_detected   = p_value < threshold

    result = {
        "feature":        feature_name,
        "type":           "numeric",
        "ks_statistic":   round(float(ks_stat), 4),
        "p_value":        round(float(p_value), 4),
        "drift_detected": drift_detected,
        "threshold":      threshold,
        "ref_mean":       round(float(reference.mean()), 4),
        "cur_mean":       round(float(current.mean()), 4),
        "ref_std":        round(float(reference.std()), 4),
        "cur_std":        round(float(current.std()), 4),
    }

    if drift_detected:
        logger.warning(f"⚠️  DRIFT DETECTED [{feature_name}]: KS={ks_stat:.4f}, p={p_value:.4f}")
    else:
        logger.info(f"✅ No drift [{feature_name}]: p={p_value:.4f}")

    return result


# ─── Categorical Feature Drift (Chi-Square) ───────────────────────────────────

def detect_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
    threshold: float = CHI2_THRESHOLD
) -> dict:
    """
    Chi-Square test for categorical feature drift.
    """
    all_categories = set(reference.unique()) | set(current.unique())

    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    # Align both series to same index
    ref_aligned = pd.Series(
        [ref_counts.get(c, 0) for c in all_categories], index=all_categories
    )
    cur_aligned = pd.Series(
        [cur_counts.get(c, 0) for c in all_categories], index=all_categories
    )

    contingency_table = pd.DataFrame([ref_aligned, cur_aligned]).values

    try:
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        drift_detected = p_value < threshold
    except Exception:
        drift_detected = False
        chi2, p_value = 0.0, 1.0

    result = {
        "feature":        feature_name,
        "type":           "categorical",
        "chi2_statistic": round(float(chi2), 4),
        "p_value":        round(float(p_value), 4),
        "drift_detected": drift_detected,
        "threshold":      threshold,
        "ref_categories": int(reference.nunique()),
        "cur_categories": int(current.nunique()),
    }

    if drift_detected:
        logger.warning(f"⚠️  DRIFT DETECTED [{feature_name}]: Chi²={chi2:.4f}, p={p_value:.4f}")

    return result


# ─── Full Data Drift Report ───────────────────────────────────────────────────

NUMERIC_FEATURES      = ['Inches', 'Ram', 'Weight', 'ppi', 'HDD', 'SSD']
CATEGORICAL_FEATURES  = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'OpSys']


def run_drift_report(current_df: pd.DataFrame) -> dict:
    """
    Compare current incoming data against the reference baseline.

    Args:
        current_df: New preprocessed data (same columns as training data)

    Returns:
        Full drift report dict
    """
    reference_df = load_reference_data()

    report = {
        "timestamp":          datetime.now().isoformat(),
        "reference_rows":     len(reference_df),
        "current_rows":       len(current_df),
        "numeric_drift":      [],
        "categorical_drift":  [],
        "drift_summary":      {},
    }

    # Numeric drift
    for col in NUMERIC_FEATURES:
        if col in reference_df.columns and col in current_df.columns:
            result = detect_numeric_drift(
                reference_df[col], current_df[col], col
            )
            report["numeric_drift"].append(result)

    # Categorical drift
    for col in CATEGORICAL_FEATURES:
        if col in reference_df.columns and col in current_df.columns:
            result = detect_categorical_drift(
                reference_df[col], current_df[col], col
            )
            report["categorical_drift"].append(result)

    # Summary
    num_drifted = sum(r["drift_detected"] for r in report["numeric_drift"])
    cat_drifted = sum(r["drift_detected"] for r in report["categorical_drift"])
    total       = len(report["numeric_drift"]) + len(report["categorical_drift"])

    report["drift_summary"] = {
        "total_features_checked": total,
        "features_with_drift":    num_drifted + cat_drifted,
        "numeric_drift_count":    num_drifted,
        "categorical_drift_count":cat_drifted,
        "overall_drift_detected": (num_drifted + cat_drifted) > 0,
    }

    # Save log
    _append_log(DRIFT_LOG_PATH, report)

    return report


# ─── Model Performance Drift ──────────────────────────────────────────────────

def detect_model_drift(
    X_new: pd.DataFrame,
    y_true: pd.Series,
    baseline_r2: float = 0.86,
    threshold: float = R2_DROP_THRESHOLD
) -> dict:
    """
    Compare live model performance against baseline R².

    Args:
        X_new: New feature matrix (preprocessed)
        y_true: True prices (log-transformed)
        baseline_r2: R² achieved during training
        threshold: Max allowed drop before alert

    Returns:
        dict with current performance and drift status
    """
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_new)

    current_r2   = r2_score(y_true, y_pred)
    current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_drop      = baseline_r2 - current_r2
    drift        = r2_drop > threshold

    result = {
        "timestamp":     datetime.now().isoformat(),
        "baseline_r2":   baseline_r2,
        "current_r2":    round(current_r2, 4),
        "r2_drop":       round(r2_drop, 4),
        "current_rmse":  round(current_rmse, 4),
        "drift_detected":drift,
        "threshold":     threshold,
        "action":        "RETRAIN REQUIRED" if drift else "OK"
    }

    if drift:
        logger.warning(
            f"🚨 MODEL DRIFT: R² dropped {r2_drop:.4f} "
            f"(baseline={baseline_r2}, current={current_r2:.4f})"
        )
        _append_log(ALERT_LOG_PATH, result)
    else:
        logger.info(f"✅ Model stable: R²={current_r2:.4f}")

    return result


# ─── Prediction Logging ───────────────────────────────────────────────────────

PREDICTION_LOG_PATH = "monitoring/prediction_logs.json"


def log_prediction(input_features: dict, predicted_price: float):
    """
    Log each prediction for monitoring and retraining purposes.

    Args:
        input_features: Raw input dict from user
        predicted_price: Final price in INR after exp()
    """
    record = {
        "timestamp":       datetime.now().isoformat(),
        "input":           input_features,
        "predicted_price": round(predicted_price, 2),
    }
    _append_log(PREDICTION_LOG_PATH, record)


def get_prediction_logs(last_n: int = 100) -> list:
    """Return last N prediction logs"""
    return _read_log(PREDICTION_LOG_PATH)[-last_n:]


# ─── Utility ──────────────────────────────────────────────────────────────────

def _append_log(filepath: str, record: dict):
    os.makedirs("monitoring", exist_ok=True)
    logs = _read_log(filepath)
    logs.append(record)
    with open(filepath, 'w') as f:
        json.dump(logs, f, indent=2)


def _read_log(filepath: str) -> list:
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def print_drift_report(report: dict):
    """Pretty print the drift report to console"""
    print("\n" + "=" * 60)
    print("📊 DATA DRIFT REPORT")
    print(f"   Generated: {report['timestamp']}")
    print(f"   Reference: {report['reference_rows']} rows")
    print(f"   Current:   {report['current_rows']} rows")
    print("=" * 60)

    print("\n🔢 NUMERIC FEATURES:")
    for r in report["numeric_drift"]:
        status = "⚠️  DRIFT" if r["drift_detected"] else "✅ OK"
        print(f"   {r['feature']:15s}  p={r['p_value']:.4f}  {status}")

    print("\n🏷️  CATEGORICAL FEATURES:")
    for r in report["categorical_drift"]:
        status = "⚠️  DRIFT" if r["drift_detected"] else "✅ OK"
        print(f"   {r['feature']:15s}  p={r['p_value']:.4f}  {status}")

    s = report["drift_summary"]
    print(f"\n📋 SUMMARY: {s['features_with_drift']}/{s['total_features_checked']} features drifted")
    if s["overall_drift_detected"]:
        print("🚨 ACTION: Consider retraining the model with fresh data!")
    else:
        print("✅ No significant drift detected.")
    print("=" * 60 + "\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick test: compare reference data against itself (should show no drift)"""
    try:
        ref = load_reference_data()
        report = run_drift_report(ref)
        print_drift_report(report)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run main.py first to generate reference data.")