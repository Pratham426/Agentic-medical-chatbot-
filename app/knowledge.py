from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
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
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{'page': m.metadata.get('page'), 'text': m.metadata.get('text',''), 'score': round(m.score,3), 'source': 'data/Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf'} for m in result.matches if m.score >= 0.30]