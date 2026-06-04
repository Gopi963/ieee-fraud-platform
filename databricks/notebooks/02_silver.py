# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer — Data Cleaning and Joining
# MAGIC Joins transaction and identity tables
# MAGIC Cleans, validates and standardises data
# MAGIC Implements data contracts

# COMMAND ----------

from pyspark.sql.functions import (
    col, when, lit, count, isnan, upper, trim,
    current_timestamp, coalesce, median
)
from pyspark.sql.types import FloatType, IntegerType

BRONZE_TRANSACTION_TABLE = "main.default.bronze_transactions"
BRONZE_IDENTITY_TABLE = "main.default.bronze_identity"
SILVER_TABLE = "main.default.silver_fraud"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Bronze tables

# COMMAND ----------

print("Loading Bronze tables...")
df_transactions = spark.table(BRONZE_TRANSACTION_TABLE)
df_identity = spark.table(BRONZE_IDENTITY_TABLE)

print(f"Transactions: {df_transactions.count():,}")
print(f"Identity: {df_identity.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Join transactions and identity
# MAGIC Left join — not all transactions have identity data
# MAGIC This is a key data engineering decision: we keep all transactions

# COMMAND ----------

print("Joining transactions with identity data...")
df_joined = (
    df_transactions
    .join(df_identity, on='TransactionID', how='left')
)

joined_count = df_joined.count()
identity_match_count = df_joined.filter(col('id_01').isNotNull()).count()
match_rate = (identity_match_count / joined_count) * 100

print(f"Joined records: {joined_count:,}")
print(f"With identity data: {identity_match_count:,} ({match_rate:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Clean and standardise

# COMMAND ----------

# Columns with high null rates that we'll drop
HIGH_NULL_COLS = [
    'D6', 'D7', 'D8', 'D9',
    'id_07', 'id_08', 'id_18', 'id_21',
    'id_22', 'id_23', 'id_24', 'id_25',
    'id_26', 'id_27'
]

# Drop high-null columns
cols_to_drop = [c for c in HIGH_NULL_COLS if c in df_joined.columns]
df_clean = df_joined.drop(*cols_to_drop)
print(f"Dropped {len(cols_to_drop)} high-null columns")

# Standardise email domains
if 'P_emaildomain' in df_clean.columns:
    df_clean = df_clean.withColumn(
        'P_emaildomain',
        upper(trim(col('P_emaildomain')))
    )
if 'R_emaildomain' in df_clean.columns:
    df_clean = df_clean.withColumn(
        'R_emaildomain',
        upper(trim(col('R_emaildomain')))
    )

# Fill numeric nulls with median
numeric_cols = ['TransactionAmt', 'dist1', 'dist2']
for c in numeric_cols:
    if c in df_clean.columns:
        median_val = df_clean.approxQuantile(c, [0.5], 0.01)[0]
        df_clean = df_clean.fillna({c: median_val})

print(f"Cleaned records: {df_clean.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Add derived columns

# COMMAND ----------

df_silver = (
    df_clean
    # Transaction hour from timedelta
    .withColumn('tx_hour',
        (col('TransactionDT') / 3600 % 24).cast(IntegerType()))
    # Transaction day
    .withColumn('tx_day',
        (col('TransactionDT') / 86400 % 7).cast(IntegerType()))
    # High value transaction flag
    .withColumn('is_high_value',
        when(col('TransactionAmt') > 1000, 1).otherwise(0))
    # Has identity data flag
    .withColumn('has_identity',
        when(col('id_01').isNotNull(), 1).otherwise(0))
    # Email domain match flag
    .withColumn('email_domain_match',
        when(col('P_emaildomain') == col('R_emaildomain'), 1).otherwise(0))
    # Silver metadata
    .withColumn('_processed_at', current_timestamp())
    .withColumn('_layer', lit('silver'))
)

print(f"Silver records: {df_silver.count():,}")
print(f"Silver columns: {len(df_silver.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data contract validation

# COMMAND ----------

def validate_silver_contract(df):
    """Validate Silver layer data contracts."""
    print("\n" + "="*50)
    print("Silver Layer Data Contract Validation")
    print("="*50)
    
    total = df.count()
    passed = 0
    failed = 0
    
    checks = [
        ("No null TransactionIDs",
         df.filter(col('TransactionID').isNull()).count() == 0),
        ("No null TransactionAmt",
         df.filter(col('TransactionAmt').isNull()).count() == 0),
        ("No null isFraud labels",
         df.filter(col('isFraud').isNull()).count() == 0),
        ("All amounts positive",
         df.filter(col('TransactionAmt') <= 0).count() == 0),
        ("isFraud is binary",
         df.filter(~col('isFraud').isin([0, 1])).count() == 0),
        ("tx_hour in range 0-23",
         df.filter(
             (col('tx_hour') < 0) | (col('tx_hour') > 23)
         ).count() == 0),
    ]
    
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    fraud_rate = df.filter(col('isFraud') == 1).count() / total * 100
    print(f"\nFraud rate: {fraud_rate:.3f}%")
    print(f"Checks passed: {passed}/{passed+failed}")
    print("="*50)
    
    if failed > 0:
        raise ValueError(f"Silver data contract failed: {failed} checks failed!")
    
    return True

validate_silver_contract(df_silver)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to Delta Lake Silver

# COMMAND ----------

print(f"\nWriting to Silver table: {SILVER_TABLE}")
(
    df_silver
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

# Verify
silver_count = spark.table(SILVER_TABLE).count()
fraud_count = spark.table(SILVER_TABLE).filter(col('isFraud') == 1).count()
print(f"✓ Silver table written: {silver_count:,} records")
print(f"✓ Fraud transactions: {fraud_count:,} ({fraud_count/silver_count*100:.3f}%)")
print("Silver layer complete!")