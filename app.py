from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import boto3
import os
import re

app = FastAPI()

# ── Settings ──────────────────────────────────
BUCKET        = "cloud-project-time4"
S3_PREFIX     = "models/llama-finetuned/"
LOCAL_DIR     = "./inference-model"
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── Download Adapter from S3 ───────────────────
def download_model():
    s3 = boto3.client('s3')
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print("Downloading model from S3...")
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=BUCKET, Prefix=S3_PREFIX):
        for obj in page.get('Contents', []):
            key      = obj['Key']
            filename = os.path.basename(key)
            if not filename:
                continue
            local_path = os.path.join(LOCAL_DIR, filename)
            s3.download_file(BUCKET, key, local_path)
            print(f"Downloaded: {filename}")

    print("Model downloaded ")

download_model()

# ── Load Tokenizer ─────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# ── Load Base Model + LoRA Adapter ────────────
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
    device_map='auto'
)

model = PeftModel.from_pretrained(base_model, LOCAL_DIR)
model.eval()
print("Model loaded ")

# ── Clean LaTeX response ───────────────────────
def clean_response(text: str) -> str:
    # \boxed{x} → x
    text = re.sub(r'\\boxed\{(.+?)\}', r'\1', text)
    # \[ ... \] → newline
    text = re.sub(r'\\\[', '\n', text)
    text = re.sub(r'\\\]', '\n', text)
    # \( ... \) → nothing
    text = re.sub(r'\\\(|\\\)', '', text)
    # \frac{a}{b} → a/b
    text = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'\1/\2', text)
    # \Rightarrow → →
    text = text.replace(r'\Rightarrow', '→')
    # \times → ×
    text = text.replace(r'\times', '×')
    # \cdot → ·
    text = text.replace(r'\cdot', '·')
    # remove remaining latex commands
    text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
    # remove { }
    text = re.sub(r'[{}]', '', text)
    # clean extra spaces/newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

# ── Generate Answer ────────────────────────────
def generate_answer(problem: str, max_new_tokens: int = 256) -> str:
    prompt = f"<|user|>\n{problem}</s>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs['input_ids'].shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return clean_response(raw)

# ── Request Schema ─────────────────────────────
class ChatRequest(BaseModel):
    message: str

# ── Routes ─────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Math Chatbot</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: #1a1a2e;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                padding: 20px;
            }

            .chat-box {
                background: #16213e;
                border-radius: 16px;
                width: 100%;
                max-width: 800px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                overflow: hidden;
            }

            .chat-header {
                background: #0f3460;
                padding: 20px;
                text-align: center;
                border-bottom: 2px solid #e94560;
            }

            .chat-header h1 {
                color: #ffffff;
                font-size: 24px;
                letter-spacing: 1px;
            }

            #messages {
                height: 460px;
                overflow-y: auto;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 12px;
                background: #16213e;
            }

            #messages::-webkit-scrollbar { width: 6px; }
            #messages::-webkit-scrollbar-track { background: #1a1a2e; }
            #messages::-webkit-scrollbar-thumb {
                background: #0f3460;
                border-radius: 3px;
            }

            .user-msg {
                background: #0f3460;
                color: #e0e0e0;
                padding: 12px 16px;
                border-radius: 16px 16px 4px 16px;
                align-self: flex-end;
                max-width: 75%;
                font-size: 15px;
                line-height: 1.5;
                border: 1px solid #1a4a7a;
            }

            .bot-msg {
                background: #1e1e2e;
                color: #d4d4d4;
                padding: 12px 16px;
                border-radius: 16px 16px 16px 4px;
                align-self: flex-start;
                max-width: 80%;
                font-size: 15px;
                line-height: 1.8;
                border: 1px solid #2a2a3e;
                white-space: pre-wrap;
                word-break: break-word;
            }

            .loading-msg {
                background: #1e1e2e;
                color: #888;
                padding: 12px 16px;
                border-radius: 16px 16px 16px 4px;
                align-self: flex-start;
                font-size: 14px;
                font-style: italic;
                border: 1px solid #2a2a3e;
                animation: pulse 1.5s infinite;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50%       { opacity: 0.4; }
            }

            #input-area {
                display: flex;
                gap: 10px;
                padding: 16px;
                background: #0f3460;
                border-top: 2px solid #e94560;
            }

            input {
                flex: 1;
                padding: 12px 16px;
                background: #1a1a2e;
                border: 1px solid #2a2a4e;
                border-radius: 25px;
                font-size: 15px;
                color: #e0e0e0;
                outline: none;
                transition: border 0.3s;
            }

            input:focus { border-color: #e94560; }
            input::placeholder { color: #555; }

            button {
                padding: 12px 24px;
                background: #e94560;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 15px;
                font-weight: bold;
                transition: background 0.3s;
            }

            button:hover    { background: #c73652; }
            button:disabled { background: #444; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <div class="chat-box">
            <div class="chat-header">
                <h1>🧮 Math Chatbot</h1>
            </div>
            <div id="messages"></div>
            <div id="input-area">
                <input
                    type="text"
                    id="userInput"
                    placeholder="Ask a math question..."
                    onkeypress="if(event.key==='Enter') sendMessage()"
                />
                <button id="sendBtn" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const input    = document.getElementById('userInput');
                const messages = document.getElementById('messages');
                const btn      = document.getElementById('sendBtn');
                const message  = input.value.trim();
                if (!message) return;

                messages.innerHTML +=
                    `<div class="user-msg">${message}</div>`;
                input.value  = '';
                btn.disabled = true;
                btn.textContent = '...';

                const loadingId = 'loading-' + Date.now();
                messages.innerHTML +=
                    `<div class="loading-msg" id="${loadingId}">🤔 Thinking...</div>`;
                messages.scrollTop = messages.scrollHeight;

                try {
                    const res  = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message })
                    });
                    const data = await res.json();

                    document.getElementById(loadingId).remove();
                    messages.innerHTML +=
                        `<div class="bot-msg">🧮 ${data.response}</div>`;
                } catch (e) {
                    document.getElementById(loadingId).remove();
                    messages.innerHTML +=
                        `<div class="bot-msg">❌ Error, please try again.</div>`;
                }

                btn.disabled    = false;
                btn.textContent = 'Send';
                messages.scrollTop = messages.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat")
def chat(request: ChatRequest):
    response = generate_answer(request.message)
    return {"response": response}

@app.get("/health")
def health():
    return {"status": "ok"}