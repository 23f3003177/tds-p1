from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Base64Str
from typing import Optional
from google import genai
import os
from app_utils import get_llm_response


class InputRequest(BaseModel):
    question: str
    image: Optional[Base64Str | list[Base64Str] | None] = None


class UrlSource(BaseModel):
    url: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    links: list[UrlSource]


app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.post("/api/v1/query")
def main(req: InputRequest) -> QueryResponse:
    try:
        return get_llm_response(req.question)
    except Exception as e:
        return { 'answer': '', 'links': '' }


@app.get("/")
def home():
    return {"message": "home"}
