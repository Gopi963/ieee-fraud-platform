"""
Central configuration for the IEEE Fraud Detection Platform
All paths, settings and constants in one place
"""

# Databricks Volume paths
VOLUME_PATH = "/Volumes/main/default/fraud_detection"
TRAIN_TRANSACTION_PATH = f"{VOLUME_PATH}/train_transaction.csv"
TRAIN_IDENTITY_PATH = f"{VOLUME_PATH}/train_identity.csv"

# Delta Lake table paths
BRONZE_TRANSACTION_TABLE = "main.default.bronze_transactions"
BRONZE_IDENTITY_TABLE = "main.default.bronze_identity"
SILVER_TABLE = "main.default.silver_fraud"
GOLD_TABLE = "main.default.gold_fraud_features"

# ML settings
ML_EXPERIMENT_NAME = "/ieee-fraud-detection"
MODEL_NAME = "fraud_detection_rf"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature columns
NUMERIC_FEATURES = [
    'TransactionAmt',
    'card1', 'card2', 'card3', 'card5',
    'addr1', 'addr2',
    'dist1', 'dist2',
    'C1', 'C2', 'C3', 'C4', 'C5',
    'C6', 'C7', 'C8', 'C9', 'C10',
    'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5',
    'D6', 'D7', 'D8', 'D9', 'D10',
    'D11', 'D12', 'D13', 'D14', 'D15',
]

CATEGORICAL_FEATURES = [
    'ProductCD',
    'card4', 'card6',
    'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5',
    'M6', 'M7', 'M8', 'M9',
]

TARGET = 'isFraud'
ID_COL = 'TransactionID'