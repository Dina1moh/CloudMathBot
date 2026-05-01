# CloudMathBot 🤖☁️

CloudMathBot is a **cloud-based conversational AI chatbot** built using AWS services and a fine-tuned **TinyLlama-1.1B** model.  
The project demonstrates an end-to-end machine learning pipeline including:

- Dataset ingestion from Hugging Face
- Distributed preprocessing using Apache Spark on AWS EMR
- Model fine-tuning using LoRA (PEFT)
- Containerized deployment using Docker on EC2
- Web-based chatbot interface

---

## 📁 Repository Structure

```
CloudMathBot/
│
├── dataset/              # Data download scripts
├── preprocessing/       # Apache Spark (EMR) jobs
├── training/            # Fine-tuning scripts (LoRA + TinyLlama)
├── deployment/          # Backend + Docker setup
├── frontend/            # Web UI for chatbot
├── inference-model/     # Mounted volume for model cache
└── README.md
```

---

## 🚀 Features

- Fine-tuned TinyLlama-1.1B model on math instruction dataset
- Scalable preprocessing using Apache Spark on EMR
- Efficient training using LoRA (PEFT)
- Fully containerized deployment using Docker
- Browser-accessible chatbot interface

---

## 🧠 Model

- Base Model: **TinyLlama-1.1B**
- Fine-tuning method: **LoRA (Low-Rank Adaptation)**
- Dataset: **nvidia/OpenMathInstruct-2**
- Task: Mathematical reasoning chatbot

---

## ⚙️ Prerequisites

- Docker installed
- AWS EC2 instance (GPU recommended)
- Open ports: 8000 (application), 22 (SSH)
- 5–10 GB storage for model cache

---

## 📦 Setup Instructions

### 1. Clone repository
```bash
git clone https://github.com/Dina1moh/CloudMathBot.git
cd CloudMathBot
```

---

### 2. Create model cache directory
```bash
mkdir -p ~/Documents/cloud-project/model-cache
```

---

### 3. Build Docker image
```bash
sudo docker build -t math-chatbot .
```

---

### 4. Run container locally / EC2
```bash
sudo docker run -d --name chatbot -p 8000:8000 -v ~/Documents/cloud-project/model-cache:/workspace/inference-model math-chatbot
```

---

## 🌐 Deploy on AWS EC2 (Public Access)

### Step 1: Connect to EC2
```bash
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 2: Install Docker (if not installed)
```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

### Step 3: Clone project
```bash
git clone https://github.com/Dina1moh/CloudMathBot.git
cd CloudMathBot
```

### Step 4: Create model cache
```bash
mkdir -p ~/Documents/cloud-project/model-cache
```

### Step 5: Build and run container
```bash
sudo docker build -t math-chatbot .

sudo docker run -d --name chatbot -p 8000:8000 -v ~/Documents/cloud-project/model-cache:/workspace/inference-model math-chatbot
```

### Step 6: Open in browser
```
http://<EC2-PUBLIC-IP>:8000
```

---

## 🧩 Environment Notes

- `/inference-model` is mounted for persistent storage
- Model weights are cached to avoid re-downloading
- Ensure Security Group allows port 8000

---

## 🐳 Docker Overview

The container:

- Loads TinyLlama fine-tuned model
- Starts backend server (Flask/FastAPI)
- Serves web UI chatbot
- Exposes API on port 8000

---

## 🧪 Example Usage

User:
```
Solve 3x + 7 = 22
```

Bot:
```
3x = 22 - 7 = 15
x = 5
```

---

## 📊 AWS Architecture

- S3 → Dataset + model storage  
- EMR (Spark) → Distributed preprocessing  
- EC2 → Training + inference  
- Docker → Deployment  
- VPC → Networking isolation  

---


