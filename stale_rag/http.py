"""One cached, polite HTTP session for Wikidata and Wikipedia.

Network requests are spaced at least MIN_INTERVAL apart across threads, and
429s are retried after the server's Retry-After. Wikimedia rate-limits
anonymous API clients hard; set STALE_RAG_CONTACT to an email or URL to be
identified in the User-Agent, which earns higher limits.
"""
import os
import threading
import time

import requests
import requests_cache

from .config import CACHE, USER_AGENT

MIN_INTERVAL = 1.0
# Rebuild from the cache alone (e.g. after changing extraction rules while
# rate-limited); facts that would need a new request are dropped.
OFFLINE = os.environ.get("STALE_RAG_OFFLINE") == "1"

_session = None
_lock = threading.Lock()
_last = 0.0


def session():
    global _session
    if _session is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        _session = requests_cache.CachedSession(
            str(CACHE / "cache"), backend="sqlite", expire_after=-1,
            allowable_methods=("GET", "POST"))
        _session.headers["User-Agent"] = USER_AGENT
    return _session


def _get(url, params):
    """Serve from cache if possible; otherwise wait for our turn on the wire."""
    global _last
    s = session()
    if s.cache.contains(request=s.prepare_request(requests.Request("GET", url, params=params))):
        return s.get(url, params=params, timeout=90)
    if OFFLINE:
        raise LookupError(f"not cached and STALE_RAG_OFFLINE is set: {url}")
    with _lock:
        wait = _last + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return s.get(url, params=params, timeout=90)
        finally:
            _last = time.monotonic()


def get_json(url, params, retries=8):
    for attempt in range(retries):
        try:
            r = _get(url, params)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(min(2 ** attempt * 5, 120))
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:  # WDQS streams a stack trace into the body on timeout
                session().cache.delete(urls=[r.url])
                time.sleep(min(2 ** attempt * 5, 120))
                continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(float(r.headers.get("Retry-After", min(2 ** attempt * 5, 120))) + 1)
            continue
        r.raise_for_status()
    raise RuntimeError(f"gave up on {url} {params}")
