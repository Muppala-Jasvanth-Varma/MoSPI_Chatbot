import sqlite3, os

DB_PATH = "data/mospi.db"

def init_db():
    os.makedirs("data/raw/pdf", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        url TEXT UNIQUE,
        date_published TEXT,           -- raw string
        date_published_norm TEXT,      -- YYYY-MM-DD
        summary TEXT,
        category TEXT,
        subject TEXT,
        hash TEXT,
        last_seen TEXT DEFAULT (datetime('now'))
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        file_url TEXT UNIQUE,
        file_path TEXT,
        file_hash TEXT,
        file_type TEXT,
        pages INTEGER,
        text TEXT,
        processed INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (document_id) REFERENCES documents(id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tables (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        source_file_id INTEGER,
        table_json TEXT,
        n_rows INTEGER,
        n_cols INTEGER,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (document_id) REFERENCES documents(id),
        FOREIGN KEY (source_file_id) REFERENCES files(id)
    )""")

    cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                source_file_id INTEGER,
                chunk_index INTEGER,
                text TEXT,
                embedding BLOB, -- optional for Part C
                FOREIGN KEY (document_id) REFERENCES documents(id),
                FOREIGN KEY (source_file_id) REFERENCES files(id))
                """)


    conn.commit(); conn.close()

def get_conn(): return sqlite3.connect(DB_PATH)

if __name__ == "__main__":
    init_db()
    print(f"✅ Database initialized at {DB_PATH}")
