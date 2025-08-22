from .models import get_conn
from .logging_utils import setup_logging
import logging
log = logging.getLogger("scraper.report")

def report():
    setup_logging()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents"); docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM files"); files = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM files WHERE processed=1"); files_done = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM tables"); tables = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM documents WHERE date_published_norm IS NULL"); no_date = cur.fetchone()[0]
        log.info("report", extra={"context":{
            "documents": docs,
            "files_total": files,
            "files_processed": files_done,
            "tables": tables,
            "docs_missing_date": no_date
        }})
        # Top 5 recent docs
        cur.execute("SELECT id, title, date_published_norm FROM documents ORDER BY id DESC LIMIT 5")
        for r in cur.fetchall():
            log.info("recent_doc", extra={"context":{"id":r[0],"title":r[1],"date":r[2]}})

if __name__ == "__main__":
    report()