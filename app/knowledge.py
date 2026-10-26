from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec, DenseVectorQuery
from app.config import settings
from langchain_groq import ChatGroq


@lru_cache()
def clients():
    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise ValueError("Missing Groq or Pinecone API key")

    return (
        ChatGroq(
            model=settings.groq_chat_model,
            api_key=settings.groq_api_key,
            temperature=0,
        ),
        Pinecone(api_key=settings.pinecone_api_key),
    )


def ensure_index():
    _, pc = clients()

    if not pc.has_index(settings.pinecone_index):
        raise ValueError(
            f"Pinecone index '{settings.pinecone_index}' does not exist."
        )

    return pc.index(name=settings.pinecone_index)


from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts):
    vectors = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query])[0]

    result = index.documents.search(
        namespace="medical",
        top_k=top_k,
        score_by=[
            DenseVectorQuery(
                field="chunk_text",
                values=vector,
            )
        ],
        include_fields=["*"],
    )

    results = []

    for doc in result.matches:
        data = doc.to_dict()
        score = data.get("_score", getattr(doc, "score", 0))

        text = (
            data.get("text")
            or data.get("content")
            or data.get("chunk_text")
            or data.get("title")
            or ""
        )

        if score is not None and score < 0.30:
            continue

        if not isinstance(text, str) or not text.strip():
            continue

        results.append({
            "page": data.get("page", data.get("page_number")),
            "text": text,
            "score": round(score, 3) if score is not None else 0,
            "source": "data/Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf",
        })

    return results
