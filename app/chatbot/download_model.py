import boto3
import os
from dotenv import load_dotenv

load_dotenv()

BUCKET    = os.getenv("BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX")
LOCAL_DIR = "/workspace/inference-model"

def download_model():
    model_file = f"{LOCAL_DIR}/adapter_model.safetensors"
    if os.path.exists(model_file):
        print("Model already exists ")
        return

    print("Downloading model from S3...")

    s3 = boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("AWS_DEFAULT_REGION")
    )

    os.makedirs(LOCAL_DIR, exist_ok=True)
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=S3_PREFIX)

    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith("/"):
            continue
        filename  = key.split("/")[-1]
        local_path = f"{LOCAL_DIR}/{filename}"
        s3.download_file(BUCKET, key, local_path)
        print(f"Downloaded: {filename}")

    print("Model downloaded ")