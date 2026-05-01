from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("OpenMath-Preprocessing") \
    .config("spark.sql.shuffle.partitions", "400") \
    .getOrCreate()

# Paths
S3_INPUT  = "s3://cloud-project-time4/dataset/openmath_raw.jsonl"
S3_OUTPUT = "s3://cloud-project-time4/dataset/final/"

print("Loading data...")
df = spark.read.json(S3_INPUT)

# Rename columns
df = df.withColumnRenamed("problem", "instruction") \
       .withColumnRenamed("generated_solution", "response")

# Remove nulls first
df = df.filter(col("instruction").isNotNull()) \
       .filter(col("response").isNotNull())

# Build training text
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

# Remove blanks
df = df.filter(F.length(F.trim(col("text"))) > 0)

# Deduplicate
df = df.dropDuplicates(["text"])

# Keep only final column
df = df.select("text")

# Repartition for speed
df = df.repartition(200)

# Split
train_df, val_df = df.randomSplit([0.9, 0.1], seed=42)

# Save compressed parquet
train_df.write.mode("overwrite").option("compression", "snappy").parquet(S3_OUTPUT + "train/")
val_df.write.mode("overwrite").option("compression", "snappy").parquet(S3_OUTPUT + "validation/")

print("Done!")
spark.stop()