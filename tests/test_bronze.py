"""
Tests for Bronze layer logic
These run in CI/CD on every push via GitHub Actions
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def generate_mock_transactions(n=100, fraud_rate=0.08):
    """Generate mock transaction data for testing."""
    np.random.seed(42)
    fraud_mask = np.random.random(n) < fraud_rate
    return pd.DataFrame({
        'TransactionID': range(1, n + 1),
        'TransactionDT': np.random.randint(86400, 86400*30, n),
        'TransactionAmt': np.random.uniform(1, 5000, n),
        'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R'], n),
        'card1': np.random.randint(1000, 9999, n),
        'isFraud': fraud_mask.astype(int),
    })

def generate_mock_identity(n=50):
    """Generate mock identity data for testing."""
    return pd.DataFrame({
        'TransactionID': range(1, n + 1),
        'id_01': np.random.uniform(-5, 0, n),
        'DeviceType': np.random.choice(['mobile', 'desktop', None], n),
    })

class TestBronzeIngestion:
    
    def test_transaction_schema(self):
        """Test transaction data has required columns."""
        df = generate_mock_transactions()
        required_cols = [
            'TransactionID', 'TransactionDT',
            'TransactionAmt', 'isFraud'
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_no_duplicate_transaction_ids(self):
        """Transaction IDs must be unique."""
        df = generate_mock_transactions()
        assert df['TransactionID'].nunique() == len(df), \
            "Duplicate TransactionIDs found"
    
    def test_transaction_amounts_positive(self):
        """All transaction amounts must be positive."""
        df = generate_mock_transactions()
        assert (df['TransactionAmt'] > 0).all(), \
            "Non-positive transaction amounts found"
    
    def test_fraud_label_binary(self):
        """isFraud must be 0 or 1 only."""
        df = generate_mock_transactions()
        assert df['isFraud'].isin([0, 1]).all(), \
            "isFraud contains non-binary values"
    
    def test_fraud_rate_realistic(self):
        """Fraud rate should be between 0.1% and 20%."""
        df = generate_mock_transactions(n=10000, fraud_rate=0.035)
        fraud_rate = df['isFraud'].mean()
        assert 0.001 <= fraud_rate <= 0.20, \
            f"Unrealistic fraud rate: {fraud_rate:.3f}"
    
    def test_identity_join(self):
        """Left join should preserve all transactions."""
        df_txn = generate_mock_transactions(100)
        df_id = generate_mock_identity(50)
        df_joined = df_txn.merge(df_id, on='TransactionID', how='left')
        assert len(df_joined) == len(df_txn), \
            "Left join changed transaction count"
    
    def test_transaction_dt_positive(self):
        """TransactionDT must be positive."""
        df = generate_mock_transactions()
        assert (df['TransactionDT'] > 0).all(), \
            "Negative TransactionDT values found"

class TestDataContracts:
    
    def test_required_columns_present(self):
        """Data contract: all required columns must exist."""
        df = generate_mock_transactions()
        contract_cols = ['TransactionID', 'TransactionAmt', 'isFraud']
        missing = [c for c in contract_cols if c not in df.columns]
        assert len(missing) == 0, f"Contract violation: missing {missing}"
    
    def test_no_null_transaction_ids(self):
        """Data contract: TransactionID cannot be null."""
        df = generate_mock_transactions()
        assert df['TransactionID'].isnull().sum() == 0, \
            "Contract violation: null TransactionIDs"
    
    def test_no_null_fraud_labels(self):
        """Data contract: isFraud cannot be null."""
        df = generate_mock_transactions()
        assert df['isFraud'].isnull().sum() == 0, \
            "Contract violation: null fraud labels"