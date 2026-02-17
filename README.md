# 💻 Laptop Price Predictor — MLOps Project

> ML-powered laptop price estimation using Random Forest & XGBoost with full MLOps pipeline

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Models Trained](#models-trained)
- [MLOps Pipeline](#mlops-pipeline)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Docker Setup](#docker-setup)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Drift Detection](#monitoring--drift-detection)
- [Testing](#testing)
- [Common Errors & Fixes](#common-errors--fixes)
- [Model Performance](#model-performance)
- [File Reference](#file-reference)

---

## 🌟 Overview

This project predicts laptop prices based on hardware specifications using machine learning. It includes:

- **10 regression models** trained and compared automatically
- **MLflow** experiment tracking for all runs
- **Streamlit** web interface for interactive predictions
- **Docker** containerization for consistent deployment
- **GitHub Actions** CI/CD pipeline for automated training and deployment
- **Drift monitoring** to detect when model needs retraining

---

## 🏗️ Architecture

```
flowchart TD
    A[laptop_data.xlsx] -->|pd.read_excel| B[DataLoader]
    B -->|validate schema| C[FeatureEngineer]
    C -->|clean + encode| D[Preprocessed DataFrame]
    D -->|train_test_split| E[ModelTrainer]
    E -->|10 models| F[MLflow Tracking]
    F -->|best R² score| G[best_model.pkl]
    G -->|joblib.load| H[Streamlit App]
    H -->|user input| I[Predict Price]
    I -->|np.exp| J[₹ Price Output]

    G --> K[monitoring/drift_monitor.py]
    K -->|KS Test + Chi²| L[Drift Alerts]
```

**Data Flow Summary:**
1. Excel file loaded → schema validated
2. Raw columns cleaned and features engineered (PPI, CPU brand, GPU brand, etc.)
3. Label encoders fitted and saved alongside model
4. 10 models trained, all logged to MLflow
5. Best model saved as `best_model.pkl`
6. Streamlit app loads model + encoders → serves predictions
7. Every prediction logged for drift monitoring

---

## 🗂️ Project Structure

```
laptop-price-predictor/
├── data/
│   ├── raw/
│   │   └── laptop_data.xlsx          ← Source data (1330 rows)
│   └── processed/
│       └── preprocessed_data.csv     ← After feature engineering
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py              ← DataLoader class
│   │   └── preprocess.py             ← FeatureEngineer class
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                  ← ModelTrainer (all 10 models)
│   │   └── evaluate.py               ← Metrics + comparison
│   └── api/
│       └── app.py                    ← Streamlit web app
│
├── models/
│   ├── best_model.pkl                ← Trained best model
│   ├── label_encoders.pkl            ← Fitted LabelEncoders
│   └── feature_cols.pkl              ← Column order used in training
│
├── mlflow/
│   └── mlruns/
│       └── mlflow_tracking.py        ← MLflow logging functions
│
├── monitoring/
│   ├── drift_monitor.py              ← KS + Chi² drift detection
│   ├── reference_data.csv            ← Training data baseline
│   ├── drift_logs.json               ← Drift check history
│   ├── alerts.json                   ← Model drift alerts
│   └── prediction_logs.json          ← Per-prediction log
│
├── tests/
│   ├── test_data.py                  ← Data pipeline unit tests
│   └── test_models.py                ← Model accuracy unit tests
│
├── notebooks/
│   └── eda.ipynb                     ← Exploratory data analysis
│
├── docker/
│   ├── Dockerfile                    ← Container definition
│   └── docker-compose.yml            ← Multi-service orchestration
│
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml           ← GitHub Actions CI/CD
│
├── main.py                           ← Training entry point
├── requirements.txt                  ← Python dependencies
├── config.yaml                       ← Hyperparameters config
└── README.md
```

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
|---|---|---|
| Language | Python 3.10 | Core language |
| Data | Pandas, NumPy | Data manipulation |
| ML | Scikit-learn | 9 regression models |
| ML | XGBoost | Gradient boosting model |
| MLOps | MLflow | Experiment tracking & model registry |
| Frontend | Streamlit | Interactive web UI |
| Serialization | Joblib | Model save/load |
| Container | Docker | Consistent deployment environment |
| CI/CD | GitHub Actions | Automated training & deployment |
| Testing | Pytest | Unit tests for data + models |
| Package Manager | uv | Fast Python package manager |
| Data Source | OpenPyXL | Excel file reading |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | `laptop_data.xlsx` |
| Rows | 1,330 laptops |
| Target | Price (INR) |
| Features | 10 raw columns |

**Raw Columns:**

| Column | Type | Example |
|---|---|---|
| Company | String | Apple, Dell, HP |
| TypeName | String | Ultrabook, Gaming |
| Inches | Float | 13.3, 15.6 |
| ScreenResolution | String | IPS Panel 2560x1600 |
| Cpu | String | Intel Core i5 2.3GHz |
| Ram | String | 8GB |
| Memory | String | 128GB SSD |
| Gpu | String | Intel Iris Plus 640 |
| OpSys | String | macOS, Windows 10 |
| Weight | String | 1.37kg |
| Price | Float | 71378.68 |

---

## ⚙️ Feature Engineering

Raw columns are transformed into 13 model-ready features:

| Feature | Source | Transformation |
|---|---|---|
| Company | Company | LabelEncoder |
| TypeName | TypeName | LabelEncoder |
| Inches | Inches | Direct (float) |
| Ram | Ram (e.g. "8GB") | Extract integer → 8 |
| Weight | Weight (e.g. "1.37kg") | Extract float → 1.37 |
| Touchscreen | ScreenResolution | 1 if "Touchscreen" in string |
| IPS | ScreenResolution | 1 if "IPS" in string |
| ppi | ScreenResolution + Inches | sqrt(x²+y²) / inches |
| Cpu_brand | Cpu | Extract i3/i5/i7/AMD |
| HDD | Memory | Extract HDD GB |
| SSD | Memory | Extract SSD GB |
| Gpu_brand | Gpu | Extract Intel/AMD/Nvidia |
| OpSys | OpSys | LabelEncoder |

**Target transformation:** `log(Price)` — log-transform applied for better regression; prediction uses `np.exp()` to reverse.

---

## 🤖 Models Trained

All 10 models below are trained, compared, and logged to MLflow:

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from xgboost import XGBRegressor
```

**Expected Results:**

| Model | Test R² | RMSE |
|---|---|---|
| Random Forest | ~0.86 | ~0.21 |
| XGBoost | ~0.85 | ~0.22 |
| Extra Trees | ~0.85 | ~0.22 |
| Gradient Boosting | ~0.84 | ~0.23 |
| Decision Tree | ~0.78 | ~0.27 |
| AdaBoost | ~0.72 | ~0.31 |
| Ridge | ~0.70 | ~0.32 |
| Linear Regression | ~0.70 | ~0.32 |
| Lasso | ~0.68 | ~0.33 |
| KNeighbors | ~0.67 | ~0.34 |

---

## 🔄 MLOps Pipeline

```
Code Push (GitHub)
        ↓
GitHub Actions triggered
        ↓
  ┌─────────────┐
  │ Lint + Test │  ← flake8, black, pytest
  └──────┬──────┘
         ↓
  ┌──────────────┐
  │ Train Models │  ← main.py runs all 10 models
  └──────┬───────┘
         ↓
  ┌─────────────────┐
  │ MLflow Tracking │  ← logs metrics, params, artifacts
  └──────┬──────────┘
         ↓
  ┌──────────────────┐
  │ Select Best Model│  ← highest test R²
  └──────┬───────────┘
         ↓
  ┌─────────────────────┐
  │ Build Docker Image  │  ← only on main branch
  └──────┬──────────────┘
         ↓
  ┌──────────────────┐
  │ Push to Docker Hub│
  └──────┬───────────┘
         ↓
  ┌───────────────┐
  │ Deploy to Cloud│
  └───────────────┘
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10
- `uv` package manager (recommended) or `pip`
- Git

### Install uv (if not installed)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/laptop-price-predictor.git
cd laptop-price-predictor

# Create virtual environment
uv venv

# Activate venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt
```

### Verify Installation

```bash
uv run python -c "import numpy; import pandas; import sklearn; import xgboost; print('✅ All packages OK')"
```

---

## ▶️ How to Run

### Step 1 — Place Dataset

Put `laptop_data.xlsx` inside:
```
data/raw/laptop_data.xlsx
```

### Step 2 — Train Model

```bash
uv run python main.py
```

**Expected output:**
```
📂 Loading data...
✅ Loaded 1330 rows
⚙️  Preprocessing...
🚀 Training Random Forest...
✅ R² Score: 0.8600
✅ Features: ['Company', 'TypeName', 'Inches', 'Ram', 'Weight',
              'Touchscreen', 'IPS', 'ppi', 'Cpu_brand', 'HDD',
              'SSD', 'Gpu_brand', 'OpSys']
💾 Models saved to models/
🎉 Done! Now run: uv run streamlit run src/api/app.py
```

This creates three files in `models/`:
- `best_model.pkl` — trained Random Forest
- `label_encoders.pkl` — fitted encoders
- `feature_cols.pkl` — column order used during training

### Step 3 — Run Streamlit App

```bash
uv run streamlit run src/api/app.py
```

Open browser at: **http://localhost:8501**

### Step 4 — View MLflow Dashboard (optional)

```bash
mlflow ui --port 5000
```

Open browser at: **http://localhost:5000**

---

## 🐳 Docker Setup

### Build & Run Locally

```bash
# Build image
docker build -f docker/Dockerfile -t laptop-price-predictor .

# Run container
docker run -p 8501:8501 laptop-price-predictor
```

### Run with Docker Compose (includes MLflow)

```bash
# Start all services (MLflow + Trainer + API)
docker-compose -f docker/docker-compose.yml up --build

# Services available:
# Streamlit app  → http://localhost:8000
# MLflow UI      → http://localhost:5000

# Stop all services
docker-compose -f docker/docker-compose.yml down
```

### Push to Docker Hub

```bash
# Login
docker login

# Tag image
docker tag laptop-price-predictor yourusername/laptop-price-predictor:latest

# Push
docker push yourusername/laptop-price-predictor:latest
```

### Deploy to Cloud (from Docker Hub image)

```bash
# AWS ECS / Google Cloud Run / Azure Container Apps
# Pull and run from any Linux server:
docker pull yourusername/laptop-price-predictor:latest
docker run -p 8501:8501 yourusername/laptop-price-predictor:latest
```

---

## ⚙️ CI/CD Pipeline

The `.github/workflows/ml_pipeline.yml` runs automatically on every push to `main` or `develop`.

**Pipeline stages:**

```
1. lint-and-test
   ├── flake8 src/        ← code style check
   ├── black --check src/ ← formatting check
   └── pytest tests/ -v   ← run all unit tests

2. train-models (runs after lint passes)
   ├── uv run python main.py
   └── Upload models/ as artifact

3. docker-build-push (runs on main branch only)
   ├── docker build
   ├── docker login (uses GitHub Secrets)
   └── docker push to Docker Hub

4. deploy
   └── Deploy to cloud provider
```

**Required GitHub Secrets:**
- `DOCKER_USERNAME` — your Docker Hub username
- `DOCKER_PASSWORD` — your Docker Hub password/token

---

## 📊 Monitoring & Drift Detection

The `monitoring/drift_monitor.py` module detects when new data has drifted from training data.

### Run Drift Check

```bash
uv run python monitoring/drift_monitor.py
```

### How It Works

**Numeric Features** → Kolmogorov-Smirnov (KS) test
- Checks: Inches, Ram, Weight, ppi, HDD, SSD
- Alert if p-value < 0.05

**Categorical Features** → Chi-Square test
- Checks: Company, TypeName, Cpu_brand, Gpu_brand, OpSys
- Alert if p-value < 0.05

**Model Performance Drift** → R² drop check
- Alert if current R² drops > 0.10 below baseline

**Prediction Logging** — every prediction saved to `monitoring/prediction_logs.json`

### Output Example

```
============================================================
📊 DATA DRIFT REPORT
   Generated: 2026-02-17T10:30:00
   Reference: 1330 rows | Current: 1330 rows
============================================================

🔢 NUMERIC FEATURES:
   Inches          p=0.9200  ✅ OK
   Ram             p=0.8100  ✅ OK
   Weight          p=0.7300  ✅ OK
   ppi             p=0.0200  ⚠️  DRIFT
   HDD             p=0.6500  ✅ OK
   SSD             p=0.5400  ✅ OK

🏷️  CATEGORICAL FEATURES:
   Company         p=0.9100  ✅ OK

📋 SUMMARY: 1/11 features drifted
✅ No critical drift detected.
============================================================
```

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run only data tests
uv run pytest tests/test_data.py -v

# Run only model tests
uv run pytest tests/test_models.py -v

# Run with coverage
uv run pytest tests/ --cov=src -v
```

**Test Coverage:**

`test_data.py` — 15 tests covering:
- Schema validation (columns, price positivity, no nulls)
- Feature engineering (RAM, weight, CPU, GPU, memory, PPI, resolution)
- Full pipeline (shape, dropped columns, new columns, encoding)

`test_models.py` — 9 tests covering:
- All 3 model types produce valid predictions
- R² thresholds (RF > 0.70, XGB > 0.70)
- RF beats Linear Regression
- Model save/load produces identical predictions

---

## 🔧 Common Errors & Fixes

### Error: `numpy.dtype size changed, binary incompatibility`

```bash
# Fix: reinstall compatible versions
uv pip uninstall numpy pandas scikit-learn xgboost -y
uv pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 xgboost==1.7.6
```

### Error: `Model not loaded. Run python main.py first`

```bash
# Fix: train the model first
uv run python main.py
```

### Error: `['Unnamed: 0'] not in index`

Already handled in `main.py` with:
```python
df = pd.read_excel('data/raw/laptop_data.xlsx', index_col=0)
```
And in `preprocess()`:
```python
drop_cols = [c for c in ['Unnamed: 0', 'Cpu', ...] if c in df.columns]
```

### Error: `X has N features, but model expects M features`

Model and app are out of sync. Retrain and restart:
```bash
uv run python main.py
uv run streamlit run src/api/app.py
```

### Error: `os is not defined` in app.py

Add `import os` at the top of `app.py`.

### Streamlit uses wrong Python (package not found)

```bash
# Always use uv run prefix
uv run streamlit run src/api/app.py

# Verify which Python uv uses
uv run python -c "import sys; print(sys.executable)"
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Best Model | Random Forest Regressor |
| Test R² | ~0.86 |
| RMSE | ~0.21 |
| MAE | ~0.15 |
| Training Samples | 1,064 (80%) |
| Test Samples | 266 (20%) |
| Target Transform | log(Price) |
| Features Used | 13 engineered features |

---

## 📁 File Reference

| File | Description |
|---|---|
| `main.py` | Entry point: loads data, preprocesses, trains, saves model |
| `src/api/app.py` | Streamlit UI with sidebar inputs and prediction display |
| `src/data/load_data.py` | DataLoader class with schema validation |
| `src/data/preprocess.py` | FeatureEngineer class with all transformations |
| `src/models/train.py` | ModelTrainer class for all 10 models with MLflow |
| `src/models/evaluate.py` | Model evaluation and comparison utilities |
| `mlflow/mlruns/mlflow_tracking.py` | MLflow logging helpers |
| `monitoring/drift_monitor.py` | KS + Chi² drift detection + prediction logging |
| `tests/test_data.py` | 15 unit tests for data pipeline |
| `tests/test_models.py` | 9 unit tests for model training and persistence |
| `docker/Dockerfile` | Container definition for deployment |
| `docker/docker-compose.yml` | MLflow + Trainer + API orchestration |
| `.github/workflows/ml_pipeline.yml` | CI/CD: lint → train → docker → deploy |
| `models/best_model.pkl` | Saved best model (generated by main.py) |
| `models/label_encoders.pkl` | Saved LabelEncoders (generated by main.py) |
| `models/feature_cols.pkl` | Column order used in training (generated by main.py) |

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first.

---

## 📄 License

MIT License

---

## 👤 Author

**Saikiranabhi**
GitHub: [https://github.com/Saikiranabhi/laptop-price-predictor](https://github.com/Saikiranabhi/laptop-price-predictor)
