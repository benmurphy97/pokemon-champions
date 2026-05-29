import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_last_request: float = 0.0
_RATE_LIMIT_SECS = 1.5


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


def polite_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    global _last_request
    elapsed = time.time() - _last_request
    if elapsed < _RATE_LIMIT_SECS:
        time.sleep(_RATE_LIMIT_SECS - elapsed)
    resp = session.get(url, timeout=15, **kwargs)
    _last_request = time.time()
    resp.raise_for_status()
    return resp
