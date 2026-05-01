from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from chatbot.model import generate_answer
import os

router = APIRouter()

def load_index_html() -> str:
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

INDEX_HTML = load_index_html()

class ChatRequest(BaseModel):
    message: str

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLResponse(content=INDEX_HTML, media_type="text/html")

@router.post("/chat")
def chat(request: ChatRequest):
    response = generate_answer(request.message)
    return {"response": response}

@router.get("/health")
def health():
    return {"status": "ok"}