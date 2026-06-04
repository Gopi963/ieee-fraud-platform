# IEEE Fraud Detection Platform

A production-grade, end-to-end financial crime detection platform built on 
Databricks, processing **590,540 real banking transactions** from the IEEE-CIS 
Fraud Detection dataset. Implements a full Medallion lakehouse architecture 
with automated CI/CD, data contracts and MLflow experiment tracking.

## 🔴 Live Results — Trained on Real Data

| Metric | Score |
|--------|-------|
| Dataset | 590,540 real transactions |
| Fraud cases | 20,663 (3.499%) |
| Model AUC-ROC | **0.8623** |
| True Positives | 1,925 fraud cases caught |
| Training records | 472,432 |
| Features engineered | 55 |
| Data contract checks | 6/6 passed |
| CI/CD tests | 18/18 passing |

## Architecture
EEE-CIS Dataset (590k transactions)
↓
Databricks Unity Catalog Volume
↓
Bronze Layer — Raw Delta tables
(590,540 transactions + 144,233 identity records)
↓
Silver Layer — Joined, cleaned, validated
(6/6 data contract checks, time features, derived columns)
↓
Gold Layer — 55 engineered features
(amount anomalies, velocity, email risk, device fingerprinting)
↓
Random Forest Classifier — MLflow tracked
(AUC-ROC: 0.8623, time-based train/test split)
↓
GitHub Actions CI/CD (18 tests on every push)
## Why this project matters

Most fraud detection portfolios on GitHub are ML notebooks. This is a 
**data engineering platform** where ML is just the output. The value is in:

- **Real data** — 590k actual banking transactions, not generated
- **Multi-table joins** — transactions + identity with time-aware splits
- **Production patterns** — data contracts, schema validation, Delta Lake ACID
- **Proper evaluation** — time-based split (not random), realistic class imbalance
- **MLflow tracking** — every experiment logged, reproducible

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Cloud platform | Databricks Free Edition (Spark 4.1.0) |
| Storage | Delta Lake (Unity Catalog) |
| Processing | PySpark — Bronze → Silver → Gold |
| Feature engineering | PySpark Window functions (55 features) |
| ML model | scikit-learn Random Forest + MLflow |
| Data quality | Custom data contracts (6 checks) |
| CI/CD | GitHub Actions (18 tests, runs on every push) |
| Language | Python 3.11 |

## Dataset

IEEE-CIS Fraud Detection (Kaggle, 2019)
- `train_transaction.csv` — 590,540 rows, 394 columns
- `train_identity.csv` — 144,233 rows, 41 columns
- Target: `isFraud` (3.499% positive rate — realistic class imbalance)

## Pipeline Notebooks

| Notebook | Description | Output |
|----------|-------------|--------|
| `01_bronze.py` | Raw ingestion + data quality checks | `main.default.bronze_transactions` |
| `02_silver.py` | Join, clean, validate, data contracts | `main.default.silver_fraud` |
| `03_gold.py` | 55 feature engineering transforms | `main.default.gold_fraud_features` |
| `04_ml.py` | Time-based split, RF training, MLflow | Logged model + metrics |

## Feature Engineering Highlights

55 features across 5 categories:

- **Amount features** — log transform, customer avg/std/max, amount anomaly ratio
- **Time features** — hour, day, is_night, is_weekend, velocity window
- **Card features** — credit/debit flag, card type encoding
- **Email features** — free email provider detection, domain mismatch flag
- **Identity features** — has_identity flag, mobile/desktop device type

## Model Notes

AUC-ROC of 0.8623 on 590k real transactions using a baseline Random Forest.
Key design decision: **time-based train/test split** (not random) — critical for
fraud detection where you train on past data and predict on future transactions.

Precision is intentionally traded for recall in fraud detection — catching more
fraud (recall: 58.55%) matters more than avoiding false alarms in most real-world
deployments. Threshold tuning and XGBoost would improve both.

## CI/CD

GitHub Actions runs 18 automated tests on every push:
- Bronze ingestion tests (schema, duplicates, data contracts)
- Silver transformation tests (derived columns, contract validation)

## How to Run

```bash
# Clone the repo
git clone https://github.com/Gopi963/ieee-fraud-platform.git
cd ieee-fraud-platform

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# For Databricks notebooks:
# 1. Upload train_transaction.csv and train_identity.csv to
#    /Volumes/main/default/fraud_detection/
# 2. Run notebooks 01 → 02 → 03 → 04 in order
```

## Related Projects

- [Kafka + PySpark Streaming](https://github.com/Gopi963/kafka-pyspark-streaming)
- [Airflow DAG Collection](https://github.com/Gopi963/airflow-dags)
- [dbt Insurance Analytics](https://github.com/Gopi963/dbt-insurance)
- [Financial Crime Detection Platform](https://github.com/Gopi963/financial-crime-detection)