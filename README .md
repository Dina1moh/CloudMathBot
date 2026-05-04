# CloudMathBot 🤖➗

> **End-to-End Cloud-Based Mathematical Reasoning Chatbot**  
> CISC 886 – Cloud Computing | Queen's University, School of Computing

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Section 1 — VPC & Networking Setup](#section-1--vpc--networking-setup)
- [Section 2 — Dataset & S3 Setup](#section-2--dataset--s3-setup)
- [Section 3 — EMR Spark Preprocessing](#section-3--emr-spark-preprocessing)
- [Section 4 — Model Fine-Tuning](#section-4--model-fine-tuning)
- [Section 5 — EC2 Deployment](#section-5--ec2-deployment)
- [Section 6 — Web Interface](#section-6--web-interface)
- [Results](#results)
- [Cleanup](#cleanup)

---

## Project Overview

**CloudMathBot** is a cloud-native mathematical reasoning chatbot built on AWS. It fine-tunes `TinyLlama-1.1B-Chat-v1.0` using QLoRA on the `nvidia/OpenMathInstruct-2` dataset, preprocessed via Apache Spark on EMR, and served through a Dockerized FastAPI application on EC2.

| Property | Value |
|---|---|
| **Course** | CISC 886 – Cloud Computing |
| **Institution** | Queen's University, Kingston, Canada |
| **Base Model** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| **Dataset** | nvidia/OpenMathInstruct-2 |
| **Fine-tuning Method** | QLoRA (4-bit NF4 + LoRA adapters) |
| **AWS Region** | us-east-1 |
| **Application Port** | 8000 |
| **GitHub** | https://github.com/Dina1moh/CloudMathBot |

---

## Architecture

```
Hugging Face
(Dataset + Model)
       │
       ▼
  Amazon S3 (raw JSONL)
       │
       ▼
  EMR Cluster (Spark)
  ┌─────────────────┐
  │  Master Node    │──► Spark Preprocessing
  │  Core Node      │    (clean, deduplicate, split)
  └─────────────────┘
       │
       ▼
  Amazon S3 (Parquet train/ val/)
       │
       ▼
  EC2 GPU Instance
  ┌─────────────────────────┐
  │  LoRA Fine-tuning       │
  │  Docker Container       │──► FastAPI on port 8000
  │  (math-chatbot)         │
  └─────────────────────────┘
       │
       ▼
  Browser / User
```

**AWS Components:**
- **VPC:** `10.0.0.0/16` with public subnet `10.0.1.0/24`
- **Amazon S3:** Dataset storage + model adapter storage
- **Amazon EMR:** Spark preprocessing cluster (`emr-7.13.0`)
- **Amazon EC2:** GPU instance for fine-tuning + Docker deployment
- **Internet Gateway:** Public internet access
- **Security Groups:** Ports 22, 80, 443, 8000

---

## Repository Structure

```
CloudMathBot/
├── processing/
│   ├── spark_process.py          # PySpark EMR preprocessing script
│   └── upload_dataset.py         # HuggingFace → S3 ingestion script
├── fune-tuning/
│   ├── model1/
│   │   ├── Untitled8_llama.ipynb # Experiment 1 fine-tuning (4,000 records)
│   │   └── inference.ipynb       # Inference notebook – model 1
│   └── model2/
│       ├── llama_finetuning_final.(1).ipynb  # Experiment 2 fine-tuning (20,000 records)
│       └── inference_Final(1)(1).ipynb       # Final inference notebook
├── app/
│   ├── Dockerfile                # Docker build file
│   └── chatbot/
│       ├── main.py               # FastAPI app entry point
│       ├── router.py             # API routes
│       ├── model.py              # Model loading + LoRA + inference
│       └── requirements.txt      # Python dependencies
└── README.md
```

---

## Prerequisites

- AWS Account with appropriate IAM permissions
- AWS CLI configured (`aws configure`)
- Python 3.12+
- Docker installed
- SSH key pair for EC2 access

---

## Section 1 — VPC & Networking Setup

### 1.1 VPC Configuration

| Parameter | Value |
|---|---|
| VPC Name | `25vjy-vpc` |
| CIDR Block | `10.0.0.0/16` |
| Public Subnet | `25vjy-subnet-public1-us-east-1a` (`10.0.0.0/20`) |
| Internet Gateway | `25vjy-igw` |
| Region | `us-east-1` |

**Steps:**
1. Go to **VPC Console → Create VPC**
2. Set CIDR to `10.0.0.0/16`
3. Create public subnet `10.0.0.0/20` in `us-east-1a`
4. Attach Internet Gateway
5. Update route table: `0.0.0.0/0 → igw`

### 1.2 Security Group Rules

| Port | Protocol | Source | Purpose |
|---|---|---|---|
| `22` | TCP | `41.218.155.132/32` | SSH admin access |
| `80` | TCP | `0.0.0.0/0` | HTTP web traffic |
| `443` | TCP | `0.0.0.0/0` | HTTPS secure traffic |
| `8000` | TCP | `0.0.0.0/0` | ChatBot API (FastAPI) |

---

## Section 2 — Dataset & S3 Setup

### 2.1 Dataset Details

| Property | Value |
|---|---|
| Dataset | `nvidia/OpenMathInstruct-2` |
| Source | Hugging Face |
| License | CC-BY 4.0 |
| Raw Samples | ~2,000,000 |
| S3 Bucket | `s3://cloud-project-time4/dataset/` |
| Format | Snappy-compressed Parquet |

### 2.2 Upload Dataset to S3

```bash
# Install dependencies
pip install datasets boto3

# Run upload script
python processing/upload_dataset.py
```

The script downloads `nvidia/OpenMathInstruct-2` from Hugging Face and uploads to:
```
s3://cloud-project-time4/dataset/openmath_raw.jsonl
```

---

## Section 3 — EMR Spark Preprocessing

### 3.1 Cluster Configuration

| Parameter | Value |
|---|---|
| Cluster Name | `25vjy-master` |
| EMR Version | `emr-7.13.0` |
| Master Instance | `m5.xlarge` |
| Core Instance | `m5.xlarge` (1 node) |
| Applications | Spark, Hadoop |
| Output | Snappy Parquet → `s3://.../final/train/` and `.../validation/` |

### 3.2 Launch EMR Cluster

```bash
aws emr create-cluster \
  --name "25vjy-master" \
  --release-label emr-7.13.0 \
  --instance-groups \
    InstanceGroupType=MASTER,InstanceType=m5.xlarge,InstanceCount=1 \
    InstanceGroupType=CORE,InstanceType=m5.xlarge,InstanceCount=1 \
  --applications Name=Spark Name=Hadoop \
  --region us-east-1
```

### 3.3 PySpark Pipeline

The preprocessing script `processing/spark_process.py` runs the following 12-step pipeline:

| Step | Operation |
|---|---|
| 1 | Create SparkSession (`shuffle.partitions=400`) |
| 2 | Read raw JSONL from S3 |
| 3 | Rename: `problem→instruction`, `generated_solution→response` |
| 4 | Filter null rows |
| 5 | Build `text` field: `<\|user\|>\n{instruction}</s>\n<\|assistant\|>\n{response}</s>` |
| 6 | Filter blank rows |
| 7 | Deduplicate on `text` column |
| 8 | Select `text` column only |
| 9 | Repartition (200) |
| 10 | Split: `90% train / 10% validation` (seed=42) |
| 11 | Write train → Snappy Parquet to S3 |
| 12 | Write validation → Snappy Parquet to S3 |

### 3.4 Submit Spark Job

```bash
aws emr add-steps \
  --cluster-id <your-cluster-id> \
  --steps Type=Spark,Name="Preprocessing",\
ActionOnFailure=CONTINUE,\
Args=[s3://cloud-project-time4/processing/spark_process.py]
```

> ⚠️ **Terminate the EMR cluster after the job completes to avoid charges.**

---

## Section 4 — Model Fine-Tuning

### 4.1 Model Details

| Property | Value |
|---|---|
| Base Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Parameters | 1.1 Billion |
| Fine-tuning | QLoRA (4-bit NF4 + LoRA) |
| Hardware | NVIDIA RTX 2000 Ada GPU (CUDA) |
| Framework | PyTorch + HuggingFace Transformers + PEFT + TRL |

### 4.2 LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target Modules | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| Task Type | `CAUSAL_LM` |

### 4.3 Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 1 |
| Batch Size | 2 (effective: 16 via grad accumulation) |
| Learning Rate | `2e-4` |
| Max Sequence Length | 512 tokens |
| Precision | `bfloat16` |
| Eval Strategy | Every 300 steps |

### 4.4 Two Experiments

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Records | 4,000 | 20,000 |
| Training Samples | ~3,400 | 17,000 |
| Duration | ~35 min | ~2h 18min |
| Steps | 213 | 1,063 |
| Model Path | `models/llama-finetuned/` | `models/llama-finetuned-v4/` |

### 4.5 Run Fine-Tuning

Open and run the notebook:
```
fune-tuning/model2/llama_finetuning_final.(1).ipynb
```

### 4.6 Training Results (Experiment 2)

| Step | Train Loss | Val Loss | Token Accuracy |
|---|---|---|---|
| 300 | 0.7386 | 0.7340 | 79.03% |
| 600 | 0.7104 | 0.7092 | 79.59% |
| 900 | 0.7026 | 0.6976 | 79.87% |

**Test Set Results:**

| Metric | Value |
|---|---|
| Test Loss | 0.7097 |
| **Perplexity** | **2.03** |

---

## Section 5 — EC2 Deployment

### 5.1 Instance Details

| Parameter | Value |
|---|---|
| Instance Name | `application-ec2` |
| Instance Type | `m5.xlarge` |
| AMI | Ubuntu 22.04 |
| Region | `us-east-1` |
| Application Port | `8000` |

### 5.2 Deployment Steps

```bash
# 1. SSH into EC2
ssh -i "project-key.pem" ubuntu@ec2-54-90-230-30.compute-1.amazonaws.com

# 2. Install Docker
sudo apt update && sudo apt install docker.io -y
sudo systemctl start docker && sudo systemctl enable docker

# 3. Clone repository
git clone https://github.com/Dina1moh/CloudMathBot.git && cd CloudMathBot

# 4. Create model cache directory
mkdir -p ~/Documents/cloud-project/model-cache

# 5. Build Docker image
sudo docker build -t math-chatbot .

# 6. Run container (with auto-restart)
sudo docker run -d \
  --name chatbot \
  --restart=always \
  -p 8000:8000 \
  -v ~/Documents/cloud-project/model-cache:/workspace/inference-model \
  math-chatbot

# 7. Verify container is running
sudo docker ps
```

### 5.3 Test the API

```bash
curl -X POST http://54.90.230.30:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Solve for x: 2x + 4 = 10"}'
```

Expected response:
```json
{
  "response": "To solve for x: 2x + 4 = 10 → 2x = 6 → x = 3"
}
```

---

## Section 6 — Web Interface

Access the chatbot at:
```
http://54.90.230.30:8000/
```

The web interface is a browser-based HTML chat UI powered by FastAPI + Uvicorn running inside the Docker container. It displays the model name and allows users to send math questions and receive step-by-step solutions.

**Example interaction:**

| Role | Message |
|---|---|
| **User** | What is the derivative of sin(x)? |
| **Bot** | To find the derivative of sin(x), we differentiate... The derivative of sin(x) is cos(x). |

---

## Results

### Qualitative Comparison: Base vs. Fine-Tuned Model

| Prompt | Base Model | Fine-Tuned Model |
|---|---|---|
| *What is 25% of 200?* | ❌ "25% of 200 is 140" (wrong) | ✅ "25/100 × 200 = **50**" (structured) |
| *Train at 60 km/h for 2.5 hours?* | ❌ "14.5 miles (23 km)" (wrong) | ✅ "60 × 2.5 = **140 km**" (correct) |

### S3 Model Artifacts

```
s3://cloud-project-time4/models/llama-finetuned-v4/
├── adapter_model.safetensors
├── adapter_config.json
├── tokenizer.json
├── tokenizer_config.json
├── tokenizer.model
├── special_tokens_map.json
├── chat_template.jinja
├── training_args.bin
└── README.md
```

---

## Cleanup

> ⚠️ **Important:** Stop all AWS resources after use to avoid unexpected charges.

```bash
# 1. Stop EC2 instance
aws ec2 stop-instances --instance-ids i-09b7b55d503c9abc6

# 2. Terminate EMR cluster (if still running)
aws emr terminate-clusters --cluster-ids <cluster-id>

# 3. Delete S3 bucket contents (optional)
aws s3 rm s3://cloud-project-time4/ --recursive
```

---

## License

Dataset: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Base Model: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

*Queen's University • School of Computing • CISC 886 Cloud Computing*