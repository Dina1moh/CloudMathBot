from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col

# ── Spark Session
spark = SparkSession.builder \
    .appName("OpenMath-Preprocessing") \
    .getOrCreate()

# ── Paths 
S3_INPUT = "s3://cloud-project-time4/dataset/openmath_raw.jsonl"

S3_OUTPUT = "s3://cloud-project-time4/dataset/final/"

# ── Step 1: Load JSONL 
print(">>> Loading data...")
df = spark.read.json(S3_INPUT)          # <-- read JSONL, not parquet
print(f"Total records: {df.count()}")
df.printSchema()                        # confirm column names

# ── Step 2: Rename columns to match your pipeline ──
# OpenMathInstruct-2 uses: problem, generated_solution
df = df.withColumnRenamed("problem", "instruction") \
       .withColumnRenamed("generated_solution", "response")

# ── Step 3: Format to TinyLlama Format 
print(">>> Formatting...")
df = df.withColumn(
    "text",
    F.concat(
        F.lit("<|user|>\n"),
        col("instruction"),
        F.lit("</s>\n<|assistant|>\n"),
        col("response"),
        F.lit("</s>")
    )
)

# ── Step 4: Remove Nulls 
print(">>> Removing nulls...")
df = df.filter(col("text").isNotNull())
df = df.filter(F.trim(col("text")) != "")
print(f"After null removal: {df.count()}")

# ── Step 5: Remove Duplicates
print(">>> Removing duplicates...")
df = df.dropDuplicates(["text"])
print(f"After dedup: {df.count()}")

# ── Step 6: Select Final Columns 
df = df.select("text")

# ── Step 7: Train/Val Split 
print(">>> Splitting...")
train_df, val_df = df.randomSplit([0.9, 0.1], seed=42)
print(f"Train: {train_df.count()}")
print(f"Val:   {val_df.count()}")

# ── Step 8: Show Sample 
print("\nSample:")
train_df.show(1, truncate=300)

# ── Step 9: Save to S3 
train_df.write.mode("overwrite").parquet(S3_OUTPUT + "train/")
val_df.write.mode("overwrite").parquet(S3_OUTPUT + "validation/")

print(" Done!")
print(f"Train: {S3_OUTPUT}train/")
print(f"Val:   {S3_OUTPUT}validation/")
spark.stop()