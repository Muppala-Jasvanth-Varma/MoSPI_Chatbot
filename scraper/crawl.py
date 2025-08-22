import hashlib, logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dateutil import parser as dtparse

from .config import SEED_URLS, MAX_PAGES, BASE_URL
from .htt import get
from .models import get_conn
from .logging_utils import setup_logging

log = logging.getLogger("scraper.crawl")

def _norm_date(s: str|None):
    if not s: return None
    try:
        return dtparse.parse(s, dayfirst=True).date().isoformat()
    except Exception:
        return None

def save_document(conn, title, url, date_raw, summary, category, subject):
    doc_hash = hashlib.sha256((url or "").encode()).hexdigest()
    cur = conn.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO documents
    (title, url, date_published, date_published_norm, summary, category, subject, hash)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (title, url, date_raw, _norm_date(date_raw), (summary or "").strip(),
          (category or "").strip(), (subject or "").strip(), doc_hash))
    conn.commit()
    cur.execute("SELECT id FROM documents WHERE url = ?", (url,))
    return cur.fetchone()[0]

def save_file_links(conn, doc_id, links):
    cur = conn.cursor()
    for href in links:
        cur.execute("""
            INSERT OR IGNORE INTO files (document_id, file_url, file_type)
            VALUES (?, ?, 'pdf')
        """, (doc_id, href))
    conn.commit()

def _extract_pdfs(soup, base):
    pdfs = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            pdfs.append(urljoin(base, href))
    return sorted(set(pdfs))

def _extract_detail_metadata(soup):
    title = None
    for sel in ["h1", "h2", ".page-title", ".title"]:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            title = el.get_text(strip=True); break
    summary = None
    for sel in [".field--name-body p", ".node__content p"]:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            summary = el.get_text(" ", strip=True); break
    title = soup.select_one("h1").get_text(strip=True) if soup.select_one("h1") else None

    date_text = None
    date_published = None

    date_tag = soup.find(string=lambda t: "Posted on" in t or "Release Date" in t)
    if date_tag:
        date_text = date_tag.strip().split(":")[-1].strip()

    if not date_text:
        tag = soup.select_one("span.date-display-single, .field--name-field-release-date")
        if tag:
            date_text = tag.get_text(strip=True)

    if date_text:
        try:
            date_published = dtparse.parse(date_text, dayfirst=True).date().isoformat()
        except Exception:
            pass

    summary_tag = soup.select_one(".field--name-body, .node__content p")
    summary = summary_tag.get_text(strip=True) if summary_tag else None

    subject = None
    cat = "Press Release"
    for sel in [".field--name-field-category a", ".taxonomy-term a"]:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            subject = el.get_text(strip=True); break
    return title, date_text, summary, cat, subject

def crawl():
    setup_logging()
    with get_conn() as conn:
        for seed in SEED_URLS:
            for page in range(MAX_PAGES):
                url = seed if page == 0 else f"{seed}?page={page}"
                try:
                    r = get(url)
                except Exception as e:
                    log.error("Failed listing", extra={"context":{"url":url,"err":str(e)}}); continue
                soup = BeautifulSoup(r.text, "html.parser")

                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    if not href: continue
                    full = urljoin(BASE_URL, href)
                    if href.lower().endswith(".pdf"):
                        title = a.get_text(strip=True) or "Untitled PDF"
                        doc_id = save_document(conn, title, full, None, a.get("title",""), "Press Release", None)
                        save_file_links(conn, doc_id, [full])
                        log.info("direct_pdf", extra={"context":{"doc_id":doc_id,"pdf":full}})
                
                for item in soup.select(".views-row, .node--type-publication, .node--type-press-release"):
                    al = item.select_one("a[href]")
                    if not al: continue
                    detail_url = urljoin(BASE_URL, al["href"])
                    try:
                        rr = get(detail_url)
                    except Exception as e:
                        log.error("detail_fetch_fail", extra={"context":{"url":detail_url,"err":str(e)}}); continue
                    dsoup = BeautifulSoup(rr.text, "html.parser")
                    title, date_txt, summary, cat, subject = _extract_detail_metadata(dsoup)
                    doc_id = save_document(conn, title or al.get_text(strip=True) or "Untitled",
                                            detail_url, date_txt, summary, cat, subject)
                    pdfs = _extract_pdfs(dsoup, BASE_URL)
                    if pdfs:
                        save_file_links(conn, doc_id, pdfs)
                        log.info("detail_pdfs", extra={"context":{"doc_id":doc_id,"count":len(pdfs)}})
                    else:
                        log.info("detail_no_pdfs", extra={"context":{"doc_id":doc_id}})
    log.info("crawl_done")
    
if __name__ == "__main__":
    crawl()