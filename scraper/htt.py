import time, urllib.robotparser as rp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import USER_AGENT, TIMEOUT, RETRY_TOTAL, RETRY_BACKOFF, RATE_LIMIT_SECONDS, RESPECT_ROBOTS

_last_request_ts = 0.0
_robots_cache = {}

def _session():
    s = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL, backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(["GET"])
    )
    s.headers.update({"User-Agent": USER_AGENT})
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

SESSION = _session()

def _respect_rate_limit():
    global _last_request_ts
    dt = time.time() - _last_request_ts
    if dt < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - dt)
    _last_request_ts = time.time()

def _robots_allowed(url):
    if not RESPECT_ROBOTS: return True
    from urllib.parse import urlparse, urljoin
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    if base not in _robots_cache:
        robots = rp.RobotFileParser()
        robots.set_url(urljoin(base, "/robots.txt"))
        try:
            robots.read()
            _robots_cache[base] = robots
        except Exception:
            return True
    return _robots_cache[base].can_fetch(USER_AGENT, url)

def get(url):
    if not _robots_allowed(url):
        raise PermissionError(f"Blocked by robots.txt: {url}")
    _respect_rate_limit()
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r