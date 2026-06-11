"""Retrieval-Augmented Generation helpers for document intelligence."""

import re
from pathlib import Path
from uuid import uuid4


RAG_STORE_DIR = Path(".rag_store")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RAG_QUERIES = {
    "meeting_notes": "action items tasks owners deadlines decisions follow-ups commitments",
    "invoice": "invoice number vendor amount due due date payment terms approvals",
    "resume": "candidate name skills education experience projects certifications work history",
    "contract": "clauses obligations parties terms deadlines signatures compliance payment terms",
    "report": "summary findings recommendations risks next steps metrics conclusions",
    "email": "requests follow-ups sender recipient deadlines commitments replies needed",
    "research_paper": "title authors abstract findings methods conclusions experiments limitations",
    "generic_document": "important facts summary responsibilities tasks entities dates key points",
}


def split_into_semantic_units(text):
    """Split text into paragraph or sentence-like units without cutting words."""
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) > 1:
        return [" ".join(paragraph.split()) for paragraph in paragraphs]

    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_text(text, chunk_size=1200, overlap=2):
    """Split text into semantic chunks suitable for embedding."""
    units = split_into_semantic_units(text)
    if not units:
        return []

    chunks = []
    current_units = []
    current_length = 0

    for unit in units:
        unit_length = len(unit)

        if current_units and current_length + unit_length + 1 > chunk_size:
            chunks.append(" ".join(current_units))
            current_units = current_units[-overlap:] if overlap > 0 else []
            current_length = len(" ".join(current_units))

        current_units.append(unit)
        current_length += unit_length + 1

    if current_units:
        chunks.append(" ".join(current_units))

    return chunks


def preview_text(text, max_length=550):
    """Return a compact preview without exposing huge retrieved chunks."""
    cleaned_text = " ".join(text.split())
    if len(cleaned_text) <= max_length:
        return cleaned_text

    return cleaned_text[:max_length].rsplit(" ", 1)[0] + "..."


def build_rag_query(document_type):
    """Return a retrieval query tailored to the classified document type."""
    return RAG_QUERIES.get(document_type, RAG_QUERIES["generic_document"])


def _load_embedding_model():
    """Load the local sentence-transformer model lazily."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _load_chroma_collection():
    """Load the persistent Chroma collection lazily."""
    import chromadb

    RAG_STORE_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(RAG_STORE_DIR))
    return client.get_or_create_collection(name="document_chunks")


def create_document_index(document_id, text):
    """Create or replace a vector index for one document."""
    chunks = chunk_text(text)

    if not chunks:
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "message": "No text chunks available for RAG indexing.",
        }

    try:
        collection = _load_chroma_collection()
        model = _load_embedding_model()

        existing = collection.get(where={"document_id": document_id})
        existing_ids = existing.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)

        embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
        ids = [f"{document_id}-{uuid4()}" for _ in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "chunk_index": index,
                "source": f"Chunk {index + 1}",
            }
            for index, _ in enumerate(chunks)
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return {
            "success": True,
            "document_id": document_id,
            "chunk_count": len(chunks),
            "message": f"Indexed {len(chunks)} chunks.",
        }
    except Exception as exc:
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": len(chunks),
            "message": f"RAG indexing failed: {exc}",
        }


def retrieve_relevant_context(document_id, query, top_k=5):
    """Retrieve the most relevant chunks for a document and query."""
    try:
        collection = _load_chroma_collection()
        model = _load_embedding_model()
        query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"document_id": document_id},
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            chunks.append(
                {
                    "text": document,
                    "chunk_index": metadata.get("chunk_index"),
                    "source": metadata.get("source"),
                    "score": round(1 / (1 + distance), 4) if distance is not None else None,
                }
            )

        return {
            "success": True,
            "query": query,
            "chunks": chunks,
            "context": "\n\n".join(chunk["text"] for chunk in chunks),
        }
    except Exception as exc:
        return {
            "success": False,
            "query": query,
            "chunks": [],
            "context": "",
            "message": f"RAG retrieval failed: {exc}",
        }


def format_retrieved_evidence(retrieval_result):
    """Format retrieved chunks as Markdown for the Gradio UI."""
    if not retrieval_result.get("success"):
        return f"### Retrieved Context\n\n{retrieval_result.get('message', 'No context retrieved.')}"

    chunks = retrieval_result.get("chunks", [])
    if not chunks:
        return "### Retrieved Context\n\nNo relevant context was retrieved."

    evidence = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        source = chunk.get("source", "Unknown")
        evidence.append(f"**Source: {source}**\n\n{preview_text(text)}")

    return "### Retrieved Context Used\n\n" + "\n\n---\n\n".join(evidence)


def format_rag_status(index_result, retrieval_result):
    """Format RAG indexing and retrieval status for the UI."""
    indexed = index_result.get("chunk_count", 0)
    retrieved = len(retrieval_result.get("chunks", []))
    query = retrieval_result.get("query", "")

    if not index_result.get("success"):
        return f"### RAG Status\n\nIndexing failed: {index_result.get('message')}"

    if not retrieval_result.get("success"):
        return f"### RAG Status\n\nIndexed `{indexed}` chunks. Retrieval failed: {retrieval_result.get('message')}"

    return (
        "### RAG Status\n\n"
        f"- Indexed chunks: `{indexed}`\n"
        f"- Retrieved chunks used by AI: `{retrieved}`\n"
        f"- Retrieval focus: `{query}`"
    )
