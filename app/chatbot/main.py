from fastapi import FastAPI
from chatbot.router import router

app = FastAPI()
app.include_router(router)