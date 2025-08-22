import os

def env_bool(name, default=False):
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y")

SEED_URLS = [u.strip() for u in os.getenv("SEED_URLS", "https://www.mospi.gov.in/press-release").split(",") if u.strip()]
MAX_PAGES = int(os.getenv("MAX_PAGES", "3"))
RATE_LIMIT_SECONDS = float(os.getenv("RATE_LIMIT_SECONDS", "1.0"))
USER_AGENT = os.getenv("USER_AGENT", "MoSPI-Scraper/1.0 (+https://example.org; contact: jasvanthvarmamuppala@gmail.com)")
RESPECT_ROBOTS = env_bool("RESPECT_ROBOTS", True)
TIMEOUT = int(os.getenv("TIMEOUT", "30"))
RETRY_TOTAL = int(os.getenv("RETRY_TOTAL", "3"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "0.6"))
BASE_URL = "https://www.mospi.gov.in"