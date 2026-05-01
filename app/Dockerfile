FROM python:3.10-slim

WORKDIR /workspace

COPY chatbot/requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY chatbot/ ./chatbot/

RUN mkdir -p /workspace/inference-model

EXPOSE 8000

CMD ["uvicorn", "chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]