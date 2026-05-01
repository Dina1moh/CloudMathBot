import boto3
import json
from datasets import load_dataset
from tqdm import tqdm

BUCKET   = "cloud-project-time4"
S3_KEY   = "dataset/openmath_full.jsonl"
TMP_FILE = "/tmp/openmath_full.jsonl"

print("Downloading FULL dataset from HuggingFace...")

dataset = load_dataset(
    "nvidia/OpenMathInstruct-2",
    split="train",
    streaming=True
)

print("Saving to EC2...")

count = 0
with open(TMP_FILE, "w", encoding="utf-8") as f:
    for item in tqdm(dataset):
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        count += 1

print("Rows saved:", count)

print("Uploading to S3...")

s3 = boto3.client("s3")
s3.upload_file(TMP_FILE, BUCKET, S3_KEY)

print("Done!")
print(f"s3://{BUCKET}/{S3_KEY}")