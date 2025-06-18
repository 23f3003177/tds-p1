from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

pc = Pinecone(api_key="PINECONE_API_KEY")

JINA_API_KEY = os.getenv("JINA_API_KEY")
dimension = 1024

index_name = "jina-clip-v2"
namespace = "tds-p1"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

from typing import List
import requests


def get_embeddings(
    inputs: List[str],
    dimensions: int,
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    data = {
        "input": inputs,
        "model": "jina-clip-v2",
        "dimensions": dimensions,
    }

    response = requests.post(
        "https://api.jina.ai/v1/embeddings", headers=headers, json=data
    )
    return response.json()


data = [
    {"id": "img1", "modality": "image", "content": "<https://example.com/image1.jpg>"},
    {"id": "txt1", "modality": "text", "content": "A red apple on a table."},
    {"id": "img2", "modality": "image", "content": "<https://example.com/image2.png>"},
    {"id": "txt2", "modality": "text", "content": "A basket of green apples."},
]

vectors = []
for item in data:
    embeddings = get_embeddings([item["content"]], dimensions=dimension)
    embedding = embeddings["data"][0]["embedding"]
    vectors.append(
        {
            "id": item["id"],
            "values": embedding,
            "metadata": {"content": item["content"], "modality": item["modality"]},
        }
    )

index.upsert(vectors=vectors, namespace=namespace)
