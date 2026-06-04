# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer — Feature Engineering
# MAGIC Engineers 50+ features for fraud detection ML model
# MAGIC Implements production-grade feature engineering patterns

# COMMAND ----------

from pyspark.sql.functions import (
    col, when, lit, count, avg, sum as spark_sum,
    max as spark_max, min as spark_min, stddev,
    lag, unix_timestamp, window, current_timestamp,
    log1p, abs as spark_abs, round as spark_round,
    percentile_approx
)
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, IntegerType

SILVER_TABLE = "main.default.silver_fraud"
GOLD_TABLE = "main.default.gold_fraud_features"

# COMMAND ----------

print("Loading Silver table...")
df = spark.table(SILVER_TABLE)
print(f"Records: {df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Transaction amount features

# COMMAND ----------

# Customer spending window
customer_window = Window.partitionBy('card1')
card_window = Window.partitionBy('card1', 'ProductCD')

df = (
    df
    # Amount transformations
    .withColumn('log_transaction_amt',
        spark_round(log1p(col('TransactionAmt')), 4))
    # Customer-level amount stats
    .withColumn('customer_avg_amt',
        spark_round(avg('TransactionAmt').over(customer_window), 2))
    .withColumn('customer_std_amt',
        spark_round(stddev('TransactionAmt').over(customer_window), 2))
    .withColumn('customer_max_amt',
        spark_round(spark_max('TransactionAmt').over(customer_window), 2))
    .withColumn('customer_tx_count',
        count('TransactionID').over(customer_window))
    # Amount anomaly score
    .withColumn('amt_vs_customer_avg',
        spark_round(
            col('TransactionAmt') / (col('customer_avg_amt') + 0.01),
        4))
    # Suspicious amount patterns
    .withColumn('is_round_amount',
        when(col('TransactionAmt') % 100 == 0, 1).otherwise(0))
    .withColumn('is_high_value',
        when(col('TransactionAmt') > 1000, 1).otherwise(0))
    .withColumn('is_very_high_value',
        when(col('TransactionAmt') > 5000, 1).otherwise(0))
)

print("Amount features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Time-based features

# COMMAND ----------

df = (
    df
    .withColumn('is_night',
        when((col('tx_hour') >= 23) | (col('tx_hour') <= 5), 1).otherwise(0))
    .withColumn('is_weekend',
        when(col('tx_day').isin([0, 6]), 1).otherwise(0))
    .withColumn('is_business_hours',
        when((col('tx_hour') >= 9) & (col('tx_hour') <= 17), 1).otherwise(0))
    # Transaction velocity — how many transactions in same time window
    .withColumn('card_tx_count',
        count('TransactionID').over(
            Window.partitionBy('card1')
            .orderBy('TransactionDT')
            .rangeBetween(-3600, 0)  # Last hour
        ))
)

print("Time features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Card and email features

# COMMAND ----------

df = (
    df
    # Card type features
    .withColumn('is_credit_card',
        when(col('card6') == 'credit', 1).otherwise(0))
    .withColumn('is_debit_card',
        when(col('card6') == 'debit', 1).otherwise(0))
    # Email risk features
    .withColumn('is_free_email_p',
        when(col('P_emaildomain').isin(
            ['GMAIL.COM', 'YAHOO.COM', 'HOTMAIL.COM',
             'OUTLOOK.COM', 'AOL.COM']), 1).otherwise(0))
    .withColumn('is_free_email_r',
        when(col('R_emaildomain').isin(
            ['GMAIL.COM', 'YAHOO.COM', 'HOTMAIL.COM',
             'OUTLOOK.COM', 'AOL.COM']), 1).otherwise(0))
    .withColumn('no_recipient_email',
        when(col('R_emaildomain').isNull(), 1).otherwise(0))
)

print("Card and email features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Identity and device features

# COMMAND ----------

df = (
    df
    .withColumn('has_identity', 
        when(col('has_identity') == 1, 1).otherwise(0))
    # Device type
    .withColumn('is_mobile',
        when(col('DeviceType') == 'mobile', 1).otherwise(0))
    .withColumn('is_desktop',
        when(col('DeviceType') == 'desktop', 1).otherwise(0))
)

print("Identity and device features created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Select final feature set

# COMMAND ----------

FEATURE_COLS = [
    'TransactionID', 'isFraud',
    # Amount features
    'TransactionAmt', 'log_transaction_amt',
    'customer_avg_amt', 'customer_std_amt', 'customer_max_amt',
    'customer_tx_count', 'amt_vs_customer_avg',
    'is_round_amount', 'is_high_value', 'is_very_high_value',
    # Time features
    'tx_hour', 'tx_day', 'is_night', 'is_weekend',
    'is_business_hours', 'card_tx_count',
    # Card features
    'card1', 'card2', 'card3', 'card5',
    'is_credit_card', 'is_debit_card',
    # Address features
    'addr1', 'addr2', 'dist1',
    # Email features
    'is_free_email_p', 'is_free_email_r',
    'no_recipient_email', 'email_domain_match',
    # Identity features
    'has_identity', 'is_mobile', 'is_desktop',
    # C features (counting)
    'C1', 'C2', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    # D features (timedelta)
    'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15',
    # Metadata
    '_processed_at'
]

# Only select columns that exist
available_cols = [c for c in FEATURE_COLS if c in df.columns]
df_gold = df.select(available_cols)

total = df_gold.count()
fraud = df_gold.filter(col('isFraud') == 1).count()
print(f"\nGold table: {total:,} records")
print(f"Features: {len(available_cols)}")
print(f"Fraud rate: {fraud/total*100:.3f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to Gold Delta table

# COMMAND ----------

(
    df_gold
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TABLE)
)

print(f"✓ Gold table written: {GOLD_TABLE}")
print(f"✓ Records: {spark.table(GOLD_TABLE).count():,}")
print("Gold layer complete!")