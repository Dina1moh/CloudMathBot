import boto3
import json
from datasets import load_dataset
from tqdm import tqdm

BUCKET   = "cloud-project-time4"
S3_KEY   = "dataset/openmath_2M.jsonl"
TMP_FILE = "/tmp/openmath_2M.jsonl"

print("Downloading from HuggingFace...")
dataset = load_dataset(
    'nvidia/OpenMathInstruct-2',
    split='train_2M',
    streaming=True
)

print("Saving to EC2...")
with open(TMP_FILE, 'w') as f:
    for item in tqdm(dataset):
        f.write(json.dumps(item) + '\n')

print("Uploading to S3...")
s3 = boto3.client('s3')
s3.upload_file(TMP_FILE, BUCKET, S3_KEY)

print("Done!")
print(f"s3://{BUCKET}/{S3_KEY}")