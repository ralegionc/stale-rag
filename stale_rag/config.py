"""Paths, relations and shared settings."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = DATA / "http_cache"

FACTS_FILE = DATA / "facts.jsonl"
TASKS_FILE = DATA / "tasks.jsonl"

_CONTACT = os.environ.get("STALE_RAG_CONTACT")
USER_AGENT = (f"stale-rag/0.1 ({_CONTACT}) python-requests" if _CONTACT
              else "stale-rag/0.1 (research script; python-requests)")
WDQS = "https://query.wikidata.org/sparql"
WD_API = "https://www.wikidata.org/w/api.php"
WP_API = "https://en.wikipedia.org/w/api.php"

TODAY = date.today()

# Facts that change hands. min_sitelinks keeps subjects prominent enough to
# have real English Wikipedia articles with prose about the holder.
RELATIONS = {
    "P6": {"name": "head of government", "min_sitelinks": 40,
           "question": "Who is the head of government of {s}?"},
    "P35": {"name": "head of state", "min_sitelinks": 40,
            "question": "Who is the head of state of {s}?"},
    "P169": {"name": "chief executive officer", "min_sitelinks": 15,
             "question": "Who is the chief executive officer of {s}?"},
    "P286": {"name": "head coach", "min_sitelinks": 15,
             "question": "Who is the head coach of {s}?"},
    "P488": {"name": "chairperson", "min_sitelinks": 20,
             "question": "Who is the chairperson of {s}?"},
}

# A change must fall in this window: old enough that the new holder shows up
# in the current article, recent enough that old revisions are well written.
CHANGE_FROM = date(2008, 1, 1)
CHANGE_MIN_AGE_DAYS = 90
# The stale snapshot is taken this long before the change.
SNAPSHOT_LEAD_DAYS = 120
