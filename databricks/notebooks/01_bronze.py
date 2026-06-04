# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer — Raw Data Ingestion
# MAGIC Ingests raw IEEE-CIS fraud detection data into Delta Lake Bronze tables
# MAGIC Implements data contracts and quality checks at ingestion

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, isnan, when, lit, current_timestamp
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# Configuration
VOLUME_PATH = "/Volumes/main/default/fraud_detection"
TRAIN_TRANSACTION_PATH = f"{VOLUME_PATH}/train_transaction.csv"
TRAIN_IDENTITY_PATH = f"{VOLUME_PATH}/train_identity.csv"
BRONZE_TRANSACTION_TABLE = "main.default.bronze_transactions"
BRONZE_IDENTITY_TABLE = "main.default.bronze_identity"

print(f"Reading from: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingest Transaction Data

# COMMAND ----------

print("Loading transaction data...")
df_transactions = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(TRAIN_TRANSACTION_PATH)
)

print(f"Transaction records: {df_transactions.count():,}")
print(f"Columns: {len(df_transactions.columns)}")
df_transactions.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ingest Identity Data

# COMMAND ----------

print("Loading identity data...")
df_identity = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(TRAIN_IDENTITY_PATH)
)

print(f"Identity records: {df_identity.count():,}")
print(f"Columns: {len(df_identity.columns)}")
df_identity.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Quality Checks

# COMMAND ----------

def check_data_quality(df, table_name):
    """Run data quality checks and log results."""
    total = df.count()
    
    # Check null rates for key columns
    key_cols = ['TransactionID', 'TransactionAmt', 'TransactionDT']
    if 'isFraud' in df.columns:
        key_cols.append('isFraud')
    
    print(f"\n{'='*50}")
    print(f"Data Quality Report: {table_name}")
    print(f"{'='*50}")
    print(f"Total records: {total:,}")
    
    for col_name in key_cols:
        if col_name in df.columns:
            null_count = df.filter(col(col_name).isNull()).count()
            null_pct = (null_count / total) * 100
            status = "✓" if null_pct == 0 else "✗"
            print(f"{status} {col_name}: {null_count:,} nulls ({null_pct:.2f}%)")
    
    # Check fraud rate if available
    if 'isFraud' in df.columns:
        fraud_count = df.filter(col('isFraud') == 1).count()
        fraud_rate = (fraud_count / total) * 100
        print(f"\nFraud rate: {fraud_count:,} ({fraud_rate:.3f}%)")
    
    # Check duplicate TransactionIDs
    distinct = df.select('TransactionID').distinct().count()
    duplicates = total - distinct
    status = "✓" if duplicates == 0 else "✗"
    print(f"{status} Duplicate TransactionIDs: {duplicates:,}")
    
    print(f"{'='*50}\n")
    return {
        'total_records': total,
        'duplicates': duplicates,
    }

# Run quality checks
txn_quality = check_data_quality(df_transactions, "bronze_transactions")
id_quality = check_data_quality(df_identity, "bronze_identity")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Add metadata and write to Delta Lake

# COMMAND ----------

# Add ingestion metadata
df_transactions_bronze = (
    df_transactions
    .withColumn('_ingested_at', current_timestamp())
    .withColumn('_source', lit('ieee_cis_kaggle'))
    .withColumn('_layer', lit('bronze'))
)

df_identity_bronze = (
    df_identity
    .withColumn('_ingested_at', current_timestamp())
    .withColumn('_source', lit('ieee_cis_kaggle'))
    .withColumn('_layer', lit('bronze'))
)

# COMMAND ----------

# Write transactions to Delta Lake Bronze
print("Writing transactions to Delta Lake Bronze...")
(
    df_transactions_bronze
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(BRONZE_TRANSACTION_TABLE)
)
print(f"✓ Transactions written to {BRONZE_TRANSACTION_TABLE}")

# Write identity to Delta Lake Bronze
print("Writing identity to Delta Lake Bronze...")
(
    df_identity_bronze
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(BRONZE_IDENTITY_TABLE)
)
print(f"✓ Identity written to {BRONZE_IDENTITY_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify Delta tables

# COMMAND ----------

print("\nVerifying Bronze tables...")
txn_count = spark.table(BRONZE_TRANSACTION_TABLE).count()
id_count = spark.table(BRONZE_IDENTITY_TABLE).count()

print(f"✓ bronze_transactions: {txn_count:,} records")
print(f"✓ bronze_identity: {id_count:,} records")

# Show sample
print("\nTransaction sample:")
spark.table(BRONZE_TRANSACTION_TABLE).select(
    'TransactionID', 'TransactionAmt', 'isFraud', '_ingested_at'
).show(5)

print("Bronze layer complete!")