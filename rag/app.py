
import os
import sqlite3
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "data/mospi.db")
INDEX_PATH = os.getenv("INDEX_PATH", "data/processed/mospi.index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

st.sidebar.write("📥 Loading FAISS + embedding model...")
index = faiss.read_index(INDEX_PATH)
model = SentenceTransformer(EMBED_MODEL)


llm = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        llm = genai.GenerativeModel(GEMINI_MODEL)
        st.sidebar.success(f"🤖 Gemini ready: {GEMINI_MODEL}")
    except Exception as e:
        st.sidebar.warning(f"Gemini error, fallback to extractive: {e}")
else:
    st.sidebar.warning("⚠️ GEMINI_API_KEY not set, fallback to extractive answers")




def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_meta(conn, ids):
    if not ids:
        return {}
    q = ",".join(["?"] * len(ids))
    rows = conn.execute(f"SELECT id, title, url FROM documents WHERE id IN ({q})", ids).fetchall()
    return {int(r["id"]): {"title": r["title"], "url": r["url"]} for r in rows}

def fetch_chunks(conn, ids):
    if not ids:
        return []
    q = ",".join(["?"] * len(ids))
    return conn.execute(f"SELECT chunk_id, document_id, text FROM chunks WHERE chunk_id IN ({q})", ids).fetchall()

def embed_query(query: str) -> np.ndarray:
    v = model.encode([query], convert_to_numpy=True)
    return v.astype("float32") if v.dtype != np.float32 else v

def retrieve(query: str, k: int = 5):
    q_vec = embed_query(query)
    D, I = index.search(q_vec, k)
    ids = [int(i) for i in I[0] if i != -1]
    dists = [float(d) for d in D[0][: len(ids)]]
    if not ids: return []
    conn = get_conn()
    try:
        rows = fetch_chunks(conn, ids)
        meta = fetch_meta(conn, [int(r["document_id"]) for r in rows])
        results = []
        for cid, dist in zip(ids, dists):
            row = next((r for r in rows if int(r["chunk_id"]) == cid), None)
            if not row: continue
            m = meta.get(int(row["document_id"]), {"title": None, "url": None})
            results.append({
                "chunk_id": cid,
                "distance": dist,
                "title": m["title"],
                "url": m["url"],
                "text": row["text"][:500] + "…" if len(row["text"]) > 500 else row["text"]
            })
        return results
    finally:
        conn.close()

def build_prompt(query, chunks):
    context = "".join([f"[{c['title']}] {c['text']}\n\n" for c in chunks])
    return f"""You are a MoSPI Q&A assistant.
Answer ONLY from the context below. If not found, say you don't know.
Question: {query}

Context:
{context}
"""

def answer(query, chunks):
    if llm:
        try:
            prompt = build_prompt(query, chunks)
            resp = llm.generate_content(prompt)
            return resp.text.strip() if hasattr(resp, "text") else ""
        except:
            pass
    if not chunks:
        return "I don't have enough information in my data."
    return "Here’s what I found:\n" + "\n".join([f"- {c['text']}" for c in chunks[:3]])



st.set_page_config(page_title="📊 MoSPI RAG Assistant", layout="wide")
st.title(" MoSPI RAG Assistant")

tab1, tab2 = st.tabs(["💬 Ask", "🔎 Search"])

with tab1:
    q = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        if q.strip():
            with st.spinner("Retrieving..."):
                hits = retrieve(q, k=5)
                st.subheader("Answer")
                st.markdown(answer(q, hits))
                st.subheader("Sources")
                for h in hits:
                    st.markdown(f"- [{h['title']}]({h['url']}) (dist={h['distance']:.3f})")

with tab2:
    s = st.text_input("Search the index:")
    if st.button("Search"):
        if s.strip():
            with st.spinner("Searching..."):
                hits = retrieve(s, k=5)
                if not hits:
                    st.info("No results found.")
                for h in hits:
                    st.markdown(f"### {h['title'] or 'Untitled'}")
                    if h["url"]:
                        st.markdown(f"[Open Document]({h['url']})")
                    st.markdown(f"**Distance:** {h['distance']:.3f}")
                    st.markdown(f"**Snippet:** {h['text']}")
                    st.markdown("---")