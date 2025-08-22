import os
import sqlite3
from typing import List, Dict, Any

from fastapi import FastAPI, Query
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()


USE_GEMINI = True
try:
    import google.generativeai as genai
except Exception:
    USE_GEMINI = False

DB_PATH = os.getenv("DB_PATH", "data/mospi.db")
INDEX_PATH = os.getenv("INDEX_PATH", "data/processed/mospi.index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


app = FastAPI(title="MoSPI RAG API", version="1.0")

print(" Loading FAISS index + embeddings model...")
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Build it with: python -m pipeline.run")

index = faiss.read_index(INDEX_PATH)
model = SentenceTransformer(EMBED_MODEL)

llm = None
if USE_GEMINI and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        llm = genai.GenerativeModel(GEMINI_MODEL)
        print(f" Gemini model ready: {GEMINI_MODEL}")
    except Exception as e:
        print(f" Gemini init failed, falling back to extractive answers: {e}")
        llm = None
else:
    print("ℹ GEMINI_API_KEY not set or google-generativeai not installed; using extractive fallback.")

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_docs_meta(conn: sqlite3.Connection, doc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not doc_ids:
        return {}
    placeholders = ",".join(["?"] * len(doc_ids))
    rows = conn.execute(
        f"SELECT id, title, url FROM documents WHERE id IN ({placeholders})",
        doc_ids,
    ).fetchall()
    return {int(r["id"]): {"title": r["title"], "url": r["url"]} for r in rows}

def fetch_chunks_by_ids(conn: sqlite3.Connection, chunk_ids: List[int]) -> List[sqlite3.Row]:
    if not chunk_ids:
        return []
    placeholders = ",".join(["?"] * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT chunk_id, document_id, source_file_id, text
        FROM chunks
        WHERE chunk_id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    return rows

def embed_query(query: str) -> np.ndarray:
    q = model.encode([query], convert_to_numpy=True)
    return q.astype("float32") if q.dtype != np.float32 else q

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k chunks with metadata: chunk_id, dist, text, title, url."""
    q_vec = embed_query(query)
    D, I = index.search(q_vec, k)  # If you used IndexIDMap.add_with_ids, I[0] are chunk_ids
    ids = [int(i) for i in I[0] if i != -1]
    dists = [float(d) for d in D[0][: len(ids)]]

    if not ids:
        return []

    conn = get_conn()
    try:
        chunk_rows = fetch_chunks_by_ids(conn, ids)
        by_chunk = {int(r["chunk_id"]): r for r in chunk_rows}

        doc_ids = sorted({int(r["document_id"]) for r in chunk_rows})
        doc_meta = fetch_docs_meta(conn, doc_ids)

        results: List[Dict[str, Any]] = []
        for cid, dist in zip(ids, dists):
            row = by_chunk.get(cid)
            if not row:
                continue
            doc_id = int(row["document_id"])
            meta = doc_meta.get(doc_id, {"title": None, "url": None})
            text = (row["text"] or "").strip()
            snippet = (text[:500] + "…") if len(text) > 500 else text

            results.append(
                {
                    "chunk_id": cid,
                    "distance": dist,
                    "document_id": doc_id,
                    "source_file_id": int(row["source_file_id"]) if row["source_file_id"] is not None else None,
                    "title": meta["title"],
                    "url": meta["url"],
                    "text": snippet,
                }
            )
        return results
    finally:
        conn.close()

def build_prompt(query: str, chunks: List[Dict[str, Any]], history: List[Dict[str, str]] = None, max_chars: int = 8000) -> str:
    """Builds a chatbot-style prompt with optional conversation history."""
    context_parts = []
    used = 0
    for c in chunks:
        piece = f"[Source: {c['title'] or 'Unknown'} — {c['url'] or 'No URL'}]\n{c['text']}\n\n"
        if used + len(piece) > max_chars:
            break
        context_parts.append(piece)
        used += len(piece)

    context = "".join(context_parts).strip()

    history_text = ""
    if history:
        history_pairs = "\n".join([f"User: {h['user']}\nAssistant: {h['bot']}" for h in history[-3:]])
        history_text = f"Conversation so far:\n{history_pairs}\n\n"

    prompt = f"""
You are MoSPI Assistant  — a helpful AI trained only on documents from India's Ministry of Statistics and Programme Implementation.

{history_text}
User Question:
{query}

Relevant MoSPI Context:
{context}

Instructions:
- Answer naturally, like in a Q&A chatbot.
- Only use the provided MoSPI context. If unsure, say you don't have enough info.
- Keep answers clear and to the point.
- Cite sources inline like (Source: <title>, <url>).
"""
    return prompt.strip()


def generate_answer(query: str, chunks: List[Dict[str, Any]], history: List[Dict[str, str]] = None) -> str:
    """Generates a conversational answer using Gemini or fallback."""
    if llm is not None:
        try:
            prompt = build_prompt(query, chunks, history=history)
            resp = llm.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
        except Exception as e:
            print(f" Gemini error, using fallback: {e}")


    if not chunks:
        return "I don’t have enough information in the MoSPI corpus to answer that."

    bullet_lines = "\n".join([f"- {c['text']} (Source: {c['title']})" for c in chunks[:3]])
    return f"Here’s what I found in MoSPI documents:\n\n{bullet_lines}"

    if not chunks:
        return "I don't have enough information in the retrieved context to answer that."

    bullet_lines = [f"- {c['text']}" for c in chunks[:3]]
    return (
        "Here is what I found in the retrieved MoSPI context:\n\n"
        + "\n".join(bullet_lines)
        + "\n\n(Generated without generative model due to configuration or error.)"
    )

class SearchHit(BaseModel):
    chunk_id: str
    distance: float
    title: str | None
    url: str | None
    text: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]

class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchHit]

@app.get("/", tags=["meta"])
def root():
    return {"message": "Welcome to the MoSPI RAG API. Use /health, /search?query=..., or /ask?query=..."}

@app.get("/health", tags=["meta"])
def health_check():
    return {"status": "ok", "embed_model": EMBED_MODEL, "faiss_vectors": index.ntotal}

@app.get("/search", response_model=SearchResponse, tags=["retrieval"])
def search_endpoint(query: str = Query(..., description="User query text"), top_k: int = 5):
    hits = retrieve(query, k=top_k)
    return SearchResponse(
        query=query,
        results=[
            SearchHit(
                chunk_id=str(h["chunk_id"]),
                distance=h["distance"],
                title=h["title"],
                url=h["url"],
                text=h["text"],
            )
            for h in hits
        ],
    )

@app.get("/ask", response_model=AskResponse, tags=["qa"])
def ask_endpoint(query: str = Query(..., description="User question"), top_k: int = 5):
    hits = retrieve(query, k=top_k)
    answer = generate_answer(query, hits)
    return AskResponse(
        query=query,
        answer=answer,
        sources=[
            SearchHit(
                chunk_id=str(h["chunk_id"]),
                distance=h["distance"],
                title=h["title"],
                url=h["url"],
                text=h["text"],
            )
            for h in hits
        ],
    )