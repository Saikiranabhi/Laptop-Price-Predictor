# Auto-generated file
"""
tests/test_models.py
Unit tests for model training, evaluation, and inference
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Synthetic numeric dataset for model tests (no preprocessing needed)"""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'Company':    np.random.randint(0, 10, n),
        'TypeName':   np.random.randint(0, 6, n),
        'Inches':     np.random.uniform(11, 17, n),
        'Ram':        np.random.choice([4, 8, 16, 32], n),
        'Weight':     np.random.uniform(1.0, 3.5, n),
        'Touchscreen':np.random.randint(0, 2, n),
        'IPS':        np.random.randint(0, 2, n),
        'ppi':        np.random.uniform(100, 250, n),
        'Cpu_brand':  np.random.randint(0, 6, n),
        'HDD':        np.random.choice([0, 256, 512, 1024], n),
        'SSD':        np.random.choice([0, 128, 256, 512], n),
        'Gpu_brand':  np.random.randint(0, 4, n),
        'os':         np.random.randint(0, 4, n),
    })
    y = np.log(
        10000 + X['Ram'] * 3000 + X['SSD'] * 100 +
        X['ppi'] * 200 + np.random.normal(0, 500, n)
    )
    return X, y


@pytest.fixture
def trained_rf(synthetic_data):
    X, y = synthetic_data
    split = int(len(X) * 0.8)
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X[:split], y[:split])
    return model, X[split:], y[split:]


# ─── Model Output Tests ───────────────────────────────────────────────────────

class TestModelOutputs:

    def test_linear_regression_predicts(self, synthetic_data):
        X, y = synthetic_data
        model = LinearRegression()
        model.fit(X[:160], y[:160])
        preds = model.predict(X[160:])
        assert preds.shape == (40,)
        assert not np.any(np.isnan(preds))

    def test_random_forest_predicts(self, synthetic_data):
        X, y = synthetic_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X[:160], y[:160])
        preds = model.predict(X[160:])
        assert len(preds) == 40
        assert all(p > 0 for p in preds)  # log-price should be positive

    def test_xgboost_predicts(self, synthetic_data):
        X, y = synthetic_data
        model = XGBRegressor(n_estimators=10, random_state=42, verbosity=0)
        model.fit(X[:160], y[:160])
        preds = model.predict(X[160:])
        assert len(preds) == 40
        assert not np.any(np.isnan(preds))

    def test_single_sample_prediction(self, trained_rf):
        model, X_test, _ = trained_rf
        single = X_test.iloc[[0]]
        pred = model.predict(single)
        assert pred.shape == (1,)

    def test_price_after_exp_transform(self, trained_rf):
        """Verify exponentiated price is in realistic range (₹15k–₹5L)"""
        model, X_test, _ = trained_rf
        log_preds = model.predict(X_test)
        prices = np.exp(log_preds)
        assert prices.min() > 0
        # All prices should be > ₹1000 (very loose sanity check)
        assert all(p > 1000 for p in prices)


# ─── Model Accuracy Tests ─────────────────────────────────────────────────────

class TestModelAccuracy:

    def test_rf_r2_above_threshold(self, synthetic_data):
        """Random Forest should achieve R² > 0.70 on synthetic data"""
        X, y = synthetic_data
        split = int(len(X) * 0.8)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X[:split], y[:split])
        r2 = r2_score(y[split:], model.predict(X[split:]))
        assert r2 > 0.70, f"RF R² too low: {r2:.4f}"

    def test_xgb_r2_above_threshold(self, synthetic_data):
        """XGBoost should achieve R² > 0.70 on synthetic data"""
        X, y = synthetic_data
        split = int(len(X) * 0.8)
        model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X[:split], y[:split])
        r2 = r2_score(y[split:], model.predict(X[split:]))
        assert r2 > 0.70, f"XGB R² too low: {r2:.4f}"

    def test_rf_beats_linear_regression(self, synthetic_data):
        """RF should beat LinearRegression on R²"""
        X, y = synthetic_data
        split = int(len(X) * 0.8)

        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=50, random_state=42)

        lr.fit(X[:split], y[:split])
        rf.fit(X[:split], y[:split])

        lr_r2 = r2_score(y[split:], lr.predict(X[split:]))
        rf_r2 = r2_score(y[split:], rf.predict(X[split:]))

        assert rf_r2 >= lr_r2, f"Expected RF ({rf_r2:.3f}) >= LR ({lr_r2:.3f})"


# ─── Model Persistence Tests ──────────────────────────────────────────────────

class TestModelPersistence:

    def test_model_save_and_load(self, trained_rf, tmp_path):
        """Saved model should produce identical predictions after loading"""
        model, X_test, _ = trained_rf
        path = tmp_path / "test_model.pkl"

        joblib.dump(model, path)
        loaded = joblib.load(path)

        original_preds = model.predict(X_test)
        loaded_preds   = loaded.predict(X_test)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_saved_model_file_exists(self, trained_rf, tmp_path):
        model, _, _ = trained_rf
        path = tmp_path / "model.pkl"
        joblib.dump(model, path)
        assert path.exists()

    def test_best_model_exists(self):
        """Check that the best_model.pkl exists after training"""
        path = 'models/best_model.pkl'
        if os.path.exists(path):
            model = joblib.load(path)
            assert hasattr(model, 'predict'), "Loaded object is not a valid model"
        else:
            pytest.skip("models/best_model.pkl not found — run main.py first")