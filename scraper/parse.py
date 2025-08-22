import os, hashlib, requests, tempfile, shutil, logging
import pdfplumber, camelot
from .models import get_conn
from .htt import get
from .logging_utils import setup_logging

PDF_DIR = "data/raw/pdf"
os.makedirs(PDF_DIR, exist_ok=True)
log = logging.getLogger("scraper.parse")

def download_pdf(file_url, file_path):
    try:
        r = get(file_url) 
        with open(file_path, "wb") as f: f.write(r.content)
        return True
    except Exception as e:
        log.error("download_fail", extra={"context":{"url":file_url,"err":str(e)}})
        return False

def extract_text_and_table(file_path):
    text, table_json, n_pages = "", None, 0
    try:
        with pdfplumber.open(file_path) as pdf:
            n_pages = len(pdf.pages)
            for p in pdf.pages: text += p.extract_text() or ""
        with tempfile.TemporaryDirectory() as t:
            tmp_pdf = os.path.join(t, "tmp.pdf"); shutil.copy(file_path, tmp_pdf)
            try:
                tables = camelot.read_pdf(tmp_pdf, pages="1", flavor="lattice", suppress_stdout=True)
                if tables and len(tables)>0: table_json = tables[0].df.values.tolist()
            except Exception: pass
        if not table_json:
            with pdfplumber.open(file_path) as pdf:
                ex = pdf.pages[0].extract_table()
                if ex: table_json = ex
    except Exception as e:
        log.error("parse_fail", extra={"context":{"path":file_path,"err":str(e)}})
    return text, table_json, n_pages

def save_file_metadata(file_id, file_path, n_pages, text, conn):
    file_hash = hashlib.sha256(open(file_path,"rb").read()).hexdigest()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(files)"); cols=[r[1] for r in cur.fetchall()]
    if "text" not in cols: cur.execute("ALTER TABLE files ADD COLUMN text TEXT")
    if "processed" not in cols: cur.execute("ALTER TABLE files ADD COLUMN processed INTEGER DEFAULT 0")
    cur.execute("""
        UPDATE files SET file_path=?, file_hash=?, pages=?, text=?, processed=1 WHERE id=?
    """, (file_path, file_hash, n_pages, text, file_id))
    conn.commit()

def save_table(document_id, file_id, table_json, conn):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tables (document_id, source_file_id, table_json, n_rows, n_cols)
        VALUES (?, ?, ?, ?, ?)
    """, (document_id, file_id, str(table_json), len(table_json), len(table_json[0]) if table_json else 0))
    conn.commit()

def parse_pdfs(limit=10):
    setup_logging()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, document_id, file_url FROM files WHERE (processed=0 OR processed IS NULL) LIMIT ?", (limit,))
        rows = cur.fetchall()
        if not rows:
            log.info("no_unprocessed"); return
        for file_id, doc_id, url in rows:
            file_path = os.path.join(PDF_DIR, f"{file_id}.pdf")
            log.info("download_start", extra={"context":{"file_id":file_id,"url":url}})
            if not download_pdf(url, file_path): continue
            text, table_json, n_pages = extract_text_and_table(file_path)
            save_file_metadata(file_id, file_path, n_pages, text, conn)
            if table_json: 
                save_table(doc_id, file_id, table_json, conn)
                log.info("table_saved", extra={"context":{"file_id":file_id,"rows":len(table_json)}})
            else:
                log.info("no_table", extra={"context":{"file_id":file_id}})
    log.info("parse_done")

if __name__ == "__main__":
    parse_pdfs()
