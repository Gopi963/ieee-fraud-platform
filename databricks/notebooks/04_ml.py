# Databricks notebook source
# MAGIC %md
# MAGIC # ML Layer — Fraud Detection Model
# MAGIC Trains Random Forest classifier on Gold features
# MAGIC Tracks experiments with MLflow
# MAGIC Evaluates with production-grade metrics

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import json
import warnings
warnings.filterwarnings('ignore')

GOLD_TABLE = "main.default.gold_fraud_features"
MODEL_NAME = "fraud_detection_rf"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Gold features

# COMMAND ----------

print("Loading Gold features...")
df_gold = spark.table(GOLD_TABLE)
print(f"Records: {df_gold.count():,}")

# Convert to pandas for sklearn
print("Converting to pandas...")
df = df_gold.drop('TransactionID', '_processed_at').toPandas()
print(f"Shape: {df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare features

# COMMAND ----------

TARGET = 'isFraud'
EXCLUDE_COLS = [TARGET]
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

# Encode categoricals
for col_name in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col_name] = le.fit_transform(X[col_name].astype(str))

# Convert to numeric
X = X.apply(pd.to_numeric, errors='coerce')

print(f"Features: {X.shape[1]}")
print(f"Fraud rate: {y.mean()*100:.3f}%")
print(f"Class balance: {y.value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Time-based train/test split
# MAGIC Critical for fraud detection — must use time-based split
# MAGIC Never use random split for time series fraud data

# COMMAND ----------

# Use 80/20 split — first 80% for training, last 20% for testing
# This simulates real production where model is trained on past, tested on future
split_idx = int(len(df) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Training: {len(X_train):,} records ({y_train.mean()*100:.3f}% fraud)")
print(f"Testing: {len(X_test):,} records ({y_test.mean()*100:.3f}% fraud)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train with MLflow tracking

# COMMAND ----------

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_IEEEFraud"):
    
    # Build pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Training Random Forest model...")
    print("This may take a few minutes on 590k records...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc_roc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("avg_precision", avg_precision)
    mlflow.log_metric("precision_fraud",
        report.get('1', {}).get('precision', 0))
    mlflow.log_metric("recall_fraud",
        report.get('1', {}).get('recall', 0))
    mlflow.log_metric("f1_fraud",
        report.get('1', {}).get('f1-score', 0))
    mlflow.log_metric("true_positives", int(cm[1][1]))
    mlflow.log_metric("false_positives", int(cm[0][1]))
    mlflow.log_metric("false_negatives", int(cm[1][0]))
    
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"AUC-ROC:          {auc_roc:.4f}")
    print(f"Avg Precision:    {avg_precision:.4f}")
    print(f"Fraud Precision:  {report.get('1', {}).get('precision', 0):.4f}")
    print(f"Fraud Recall:     {report.get('1', {}).get('recall', 0):.4f}")
    print(f"Fraud F1:         {report.get('1', {}).get('f1-score', 0):.4f}")
    print(f"True Positives:   {cm[1][1]:,}")
    print(f"False Positives:  {cm[0][1]:,}")
    print(f"False Negatives:  {cm[1][0]:,}")
    print(f"{'='*50}")
    
    # Feature importance
    model = pipeline.named_steps['model']
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Save metrics
    metrics = {
        'auc_roc': round(auc_roc, 4),
        'avg_precision': round(avg_precision, 4),
        'precision_fraud': round(report.get('1', {}).get('precision', 0), 4),
        'recall_fraud': round(report.get('1', {}).get('recall', 0), 4),
        'f1_fraud': round(report.get('1', {}).get('f1-score', 0), 4),
        'true_positives': int(cm[1][1]),
        'false_positives': int(cm[0][1]),
        'false_negatives': int(cm[1][0]),
        'training_records': len(X_train),
        'test_records': len(X_test),
        'feature_count': len(FEATURE_COLS),
        'top_features': feature_importance.head(10).to_dict('records')
    }
    
    mlflow.log_dict(metrics, "metrics.json")
    
    # Register model
    mlflow.sklearn.log_model(pipeline, MODEL_NAME)
    
    run_id = mlflow.active_run().info.run_id
    print(f"\n✓ MLflow run ID: {run_id}")
    print("✓ Model logged to MLflow")
    print("ML layer complete!")