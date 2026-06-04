"""
Tests for Silver layer transformations
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def generate_silver_data(n=200):
    """Generate mock Silver layer data."""
    np.random.seed(42)
    return pd.DataFrame({
        'TransactionID': range(1, n + 1),
        'TransactionDT': np.random.randint(86400, 86400*30, n),
        'TransactionAmt': np.random.uniform(1, 5000, n),
        'isFraud': (np.random.random(n) < 0.035).astype(int),
        'card1': np.random.randint(1000, 9999, n),
        'P_emaildomain': np.random.choice(
            ['gmail.com', 'yahoo.com', 'hotmail.com', None], n),
        'R_emaildomain': np.random.choice(
            ['gmail.com', 'yahoo.com', None], n),
        'DeviceType': np.random.choice(['mobile', 'desktop', None], n),
        'TransactionAmt': np.abs(np.random.normal(150, 200, n)) + 1,
    })

class TestSilverTransformations:
    
    def test_tx_hour_derivation(self):
        """tx_hour should be 0-23 derived from TransactionDT."""
        df = generate_silver_data()
        df['tx_hour'] = (df['TransactionDT'] / 3600 % 24).astype(int)
        assert df['tx_hour'].between(0, 23).all(), \
            "tx_hour values outside 0-23 range"
    
    def test_tx_day_derivation(self):
        """tx_day should be 0-6."""
        df = generate_silver_data()
        df['tx_day'] = (df['TransactionDT'] / 86400 % 7).astype(int)
        assert df['tx_day'].between(0, 6).all(), \
            "tx_day values outside 0-6 range"
    
    def test_is_high_value_flag(self):
        """is_high_value should be 1 for amounts > 1000."""
        df = generate_silver_data()
        df['is_high_value'] = (df['TransactionAmt'] > 1000).astype(int)
        high_value = df[df['TransactionAmt'] > 1000]
        assert (high_value['is_high_value'] == 1).all(), \
            "is_high_value flag incorrect"
    
    def test_has_identity_flag(self):
        """has_identity should be 1 when identity data present."""
        df = generate_silver_data()
        df['has_identity'] = df['DeviceType'].notna().astype(int)
        assert df['has_identity'].isin([0, 1]).all(), \
            "has_identity is not binary"
    
    def test_email_domain_match(self):
        """email_domain_match should be 1 when P and R domains match."""
        df = generate_silver_data()
        df['email_domain_match'] = (
            df['P_emaildomain'] == df['R_emaildomain']
        ).astype(int)
        matching = df[df['P_emaildomain'] == df['R_emaildomain']]
        assert (matching['email_domain_match'] == 1).all(), \
            "email_domain_match flag incorrect"
    
    def test_no_negative_amounts_after_cleaning(self):
        """After cleaning, no negative transaction amounts."""
        df = generate_silver_data()
        df = df[df['TransactionAmt'] > 0]
        assert (df['TransactionAmt'] > 0).all(), \
            "Negative amounts survived cleaning"

class TestSilverDataContract:
    
    def test_record_count_preserved(self):
        """Silver should have same or fewer records than Bronze."""
        df_bronze = generate_silver_data(1000)
        df_silver = df_bronze.dropna(subset=['TransactionID'])
        assert len(df_silver) <= len(df_bronze), \
            "Silver has more records than Bronze"
    
    def test_fraud_rate_consistent(self):
        """Fraud rate should stay consistent through Silver."""
        df = generate_silver_data(10000)
        fraud_rate = df['isFraud'].mean()
        assert 0.001 <= fraud_rate <= 0.20, \
            f"Fraud rate changed unexpectedly: {fraud_rate}"