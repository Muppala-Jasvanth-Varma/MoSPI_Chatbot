import os, sqlite3, textwrap
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH = os.getenv("DB_PATH", "data/mospi.db")
INDEX_PATH = os.getenv("INDEX_PATH", "data/processed/mospi.index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

def _snippet(s: str, n: int = 360) -> str:
    s = (s or "").strip().replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s

class Retriever:
    def __init__(self,
                 db_path: str = DB_PATH,
                 index_path: str = INDEX_PATH,
                 model_name: str = EMBED_MODEL):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run: python -m pipeline.run")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DB not found at {db_path}.")
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(model_name)

        try:
            dim = self.model.get_sentence_embedding_dimension()
            if hasattr(self.index, "d") and self.index.d != dim:
                print(f"⚠️ Embed dim ({dim}) != index dim ({self.index.d}). Ensure the same model was used.")
        except Exception:
            pass 

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True)
        if vec.dtype != np.float32:
            vec = vec.astype("float32")
        return vec  

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        qv = self._embed(query)  
        D, I = self.index.search(qv, k)
        ids = [int(i) for i in I[0] if i != -1]
        if not ids:
            return []


        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"""
            SELECT c.chunk_id, c.text, c.document_id, d.title, d.url
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.chunk_id IN ({placeholders})
            """,
            ids,
        ).fetchall()

        by_id = {int(r["chunk_id"]): r for r in rows}
        results = []
        for rank, (cid, dist) in enumerate(zip(I[0], D[0]), start=1):
            cid = int(cid)
            if cid in by_id:
                r = by_id[cid]
                results.append(
                    {
                        "rank": rank,
                        "chunk_id": cid,
                        "distance": float(dist),
                        "text": r["text"],
                        "snippet": _snippet(r["text"]),
                        "document_id": int(r["document_id"]),
                        "title": r["title"],
                        "url": r["url"],
                    }
                )
        return results

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Query MoSPI FAISS index")
    ap.add_argument("query", type=str, help="Your question / search query")
    ap.add_argument("--k", type=int, default=5, help="Top-K results")
    args = ap.parse_args()

    r = Retriever()
    out = r.search(args.query, k=args.k)
    if not out:
        print("No results.")
    else:
        for hit in out:
            print(f"\n#{hit['rank']}  (chunk_id={hit['chunk_id']}, dist={hit['distance']:.3f})")
            print(f"Title: {hit['title']}")
            print(f"URL  : {hit['url']}")
            print(f"Text : {hit['snippet']}")
    r.close()