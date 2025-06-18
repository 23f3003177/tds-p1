import requests
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from google import genai
from typing import Literal
import numpy as np

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBEDDING_MODEL_NAME = os.getenv("JINA_EMBEDDING_MODEL_NAME")
JINA_EMBEDDING_MODEL_DIMENSIONS = os.getenv("JINA_EMBEDDING_MODEL_DIMENSIONS")
JINA_API_ENDPOINT = os.getenv("JINA_API_ENDPOINT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

llm_client = genai.Client(api_key=GEMINI_API_KEY)


def get_prompt(context: str, question: str):
    return f"""
You are a helpful expert assistant for a university course community forum.
Answer the user's question based *only* on the provided context from the forum posts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def get_embeddings(inputs: list[dict[str, list[float]]], type: Literal['text', 'image']='text'):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    data = {
        "task": "retrieval.query",
        "input": inputs,
        "model": JINA_EMBEDDING_MODEL_NAME,
        "dimensions": JINA_EMBEDDING_MODEL_DIMENSIONS,
    }

    response = requests.post(JINA_API_ENDPOINT, headers=headers, json=data)
    if response.ok:
        return response.json()["data"]
    raise ValueError("Unable to get embeddings from Jina")


def get_context(vector: list[float]) -> list[str]:
    response = index.query(
        top_k=10, vector=vector, namespace=PINECONE_NAMESPACE, include_metadata=True
    )
    return response


def get_llm_response(question: str, image: str | list[str] | None=None):
    query = []
    query.append({ 'text': question })
    if image is not None:
        if isinstance(image, str):
            query.append({ 'image': image })
        if isinstance(image, list):
            query.extend({'image': x } for x in image)
    embeddings = get_embeddings(query)
    vector = None
    if len(embeddings) == 0:
        return { 'answer': '', 'links': [] }
    elif len(embeddings) == 1:
        vector = embeddings[0]['embedding']
    else:
        vector = np.array(embeddings[0]['embedding'])
        for emb in embeddings[1:]:
            vector += np.array(emb['embedding'])
        vector /= len(embeddings)
        vector = vector.tolist()
    # print(vector)
    # exit()
    context = get_context(vector)
    links = [{ 'url': x['metadata']['url'], 'text': x['metadata']['content'] } for x in context['matches']]
    prompt = get_prompt("\n\n".join(x['metadata']['content'] for x in context['matches']), question)
    response = llm_client.models.generate_content(
        model=GEMINI_MODEL_NAME, contents=[prompt]
    )
    return { 'answer': response.text, 'links': links }


if __name__ == "__main__":
    prompt = "When is the TDS Final Jan 2025 end-term exam?"
    img = "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%"
    # print(res)
    print(get_llm_response(prompt, img))
