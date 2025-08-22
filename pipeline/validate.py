# pipeline/validate.py
import sqlite3, os
import faiss
import numpy as np

DB_PATH = "data/mospi.db"
INDEX_PATH = "data/processed/mospi.index"

def get_conn():
    return sqlite3.connect(DB_PATH)

def validate_chunks(conn):
    cur = conn.cursor()

    # Count docs, files, chunks
    cur.execute("SELECT COUNT(*) FROM documents")
    n_docs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM files")
    n_files = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM chunks")
    n_chunks = cur.fetchone()[0]

    print(f"📊 DB Summary:")
    print(f"   - Documents: {n_docs}")
    print(f"   - Files:     {n_files}")
    print(f"   - Chunks:    {n_chunks}")

    # Example: chunk stats
    cur.execute("""
        SELECT document_id, COUNT(*) as c
        FROM chunks
        GROUP BY document_id
        ORDER BY c DESC
        LIMIT 5
    """)
    top = cur.fetchall()
    print("\n🔝 Top docs by chunk count:")
    for d in top:
        print(f"   Doc {d[0]} → {d[1]} chunks")

def validate_faiss():
    if not os.path.exists(INDEX_PATH):
        print("❌ FAISS index not found. Run pipeline/run.py first.")
        return
    
    index = faiss.read_index(INDEX_PATH)
    n = index.ntotal
    d = index.d
    print(f"\n✅ FAISS index loaded: {n} vectors, dim={d}")

    # Run a dummy query
    xq = np.random.rand(1, d).astype("float32")
    D, I = index.search(xq, k=3)
    print("\n🔎 Sample query results:")
    print(f"   IDs: {I.tolist()}")
    print(f"   Distances: {D.tolist()}")

def main():
    conn = get_conn()
    validate_chunks(conn)
    conn.close()
    validate_faiss()

if __name__ == "__main__":
    main()
