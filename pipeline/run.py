import os, sqlite3, hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DB_PATH = "data/mospi.db"
INDEX_PATH = "data/processed/mospi.index"

def get_conn():
    return sqlite3.connect(DB_PATH)

def load_texts(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, document_id, text FROM files WHERE text IS NOT NULL")
    return cur.fetchall()   # [(file_id, doc_id, text), ...]

def chunk_text(doc_id, file_id, text, splitter):
    chunks = []
    for i, chunk in enumerate(splitter.split_text(text)):
        chunk_id = int(hashlib.sha256(f"{doc_id}-{file_id}-{i}".encode()).hexdigest(), 16) % (10**12)  # int64 ID
        chunks.append((chunk_id, doc_id, file_id, i, chunk))
    return chunks

def save_chunks(conn, chunks):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER,
            document_id INTEGER,
            source_file_id INTEGER,
            chunk_index INTEGER,
            text TEXT,
            UNIQUE(chunk_id),
            FOREIGN KEY (document_id) REFERENCES documents(id),
            FOREIGN KEY (source_file_id) REFERENCES files(id)
        )
    """)
    for c in chunks:
        cur.execute("""
            INSERT OR IGNORE INTO chunks (chunk_id, document_id, source_file_id, chunk_index, text)
            VALUES (?, ?, ?, ?, ?)
        """, (c[0], c[1], c[2], c[3], c[4]))
    conn.commit()

def build_faiss(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c[4] for c in chunks]
    ids = np.array([c[0] for c in chunks])  # chunk_id list
    
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    index.add_with_ids(embeddings, ids)
    
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    
    return len(chunks), embeddings.shape


def main():
    conn = get_conn()
    rows = load_texts(conn)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    all_chunks = []
    for file_id, doc_id, text in rows:
        chunks = chunk_text(doc_id, file_id, text, splitter)
        save_chunks(conn, chunks)   # save each file’s chunks
        all_chunks.extend(chunks)
    
    print(f" Total chunks saved: {len(all_chunks)}")
    n, shape = build_faiss(all_chunks)
    print(f" FAISS index built with {n} chunks, dim={shape[1]}")

    conn.close()

if __name__ == "__main__":
    main()