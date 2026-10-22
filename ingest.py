
import argparse
import hashlib
from pathlib import Path

import pymupdf

from app.knowledge import ensure_index, embed
from app.config import settings


NAMESPACE = "medical"
BATCH_SIZE = 50
SOURCE_TITLE = "Gale Encyclopedia of Medicine Vol. 1 (A-B)"


def chunks(text, size=1100, overlap=180):
    text = " ".join(text.split())
    step = size - overlap

    return [
        text[i:i + size]
        for i in range(0, len(text), step)
        if len(text[i:i + size]) > 80
    ]


def upload_batch(index, batch):
    if not batch:
        return

    # Generate local 384-dimensional embeddings.
    vectors = embed([item[1] for item in batch])

    documents = []

    for (key, content, metadata), vector in zip(batch, vectors):
        documents.append({
            "_id": key,

            # Must match the dense vector field in your Pinecone schema.
            "chunk_text": vector,

            # Must match your String field in the schema.
            "title": SOURCE_TITLE,

            # Additional fields are stored as document metadata.
            "text": content,
            "page": metadata["page"],
            "file": metadata["file"],
        })

    response = index.documents.upsert(
        namespace=NAMESPACE,
        documents=documents,
    )

    print(f"Uploaded {len(documents)} documents", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        default="data/Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf",
    )
    args = parser.parse_args()

    pdf = Path(args.pdf)

    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    index = ensure_index()

    print("Configured Pinecone index:", settings.pinecone_index)
    print("Index stats:", index.describe_index_stats())
    print("Using namespace:", NAMESPACE)

    total_chunks = 0

    with pymupdf.open(pdf) as doc:
        batch = []

        for page_number, page in enumerate(doc, start=1):
            page_chunks = chunks(page.get_text())

            for part, content in enumerate(page_chunks, start=1):
                key = hashlib.sha256(
                    f"{pdf.name}:{page_number}:{part}:{content}".encode("utf-8")
                ).hexdigest()

                metadata = {
                    "page": page_number,
                    "file": pdf.name,
                }

                batch.append((key, content, metadata))
                total_chunks += 1

                if len(batch) >= BATCH_SIZE:
                    upload_batch(index, batch)
                    batch = []

            if page_number % 50 == 0:
                print(
                    f"Processed pages: {page_number}/{len(doc)}",
                    flush=True,
                )

        if batch:
            upload_batch(index, batch)

    print(f"Ingestion complete. Processed {total_chunks} chunks.")


if __name__ == "__main__":
    main()
