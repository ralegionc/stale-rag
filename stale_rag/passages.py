"""Cut a stale and a current passage for each fact from Wikipedia revisions.

Revisions are fetched as wikitext (rendering old revisions to HTML takes the
API seconds each). The stale passage comes from the subject's article as it stood shortly before
the change and names the old holder; the current passage comes from today's
article and names the new holder. Each passage must name only its own holder,
so the two really disagree.

    python -m stale_rag.passages [--limit N]
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from tqdm import tqdm

from .config import FACTS_FILE, SNAPSHOT_LEAD_DAYS, TASKS_FILE, WP_API
from .http import get_json

MIN_CHARS = 80
MAX_CHARS = 700


def _drop_nested(text, open_, close):
    """Remove balanced open_...close spans (templates, tables), nesting allowed."""
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith(open_, i):
            depth += 1
            i += len(open_)
        elif depth and text.startswith(close, i):
            depth -= 1
            i += len(close)
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return "".join(out)


_LINK = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]")


def wikitext_paragraphs(wt):
    """Plain prose paragraphs from raw wikitext.

    Crude on purpose: drops templates, tables, refs, files and list or heading
    lines, keeps link text. Good enough to find sentences that name a person.
    """
    wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    wt = re.sub(r"<ref[^>/]*/>", "", wt)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    # Unwrap the inline templates that carry prose before dropping the rest.
    wt = re.sub(r"\{\{(?:convert|cvt)\|([\d.,]+)\|([^|{}]+)[^{}]*\}\}", r"\1 \2", wt, flags=re.I)
    wt = re.sub(r"\{\{(?:lang|langx|transl|nowrap|small)\|(?:[^|{}]*\|)?([^|{}]+)\}\}", r"\1",
                wt, flags=re.I)
    wt = _drop_nested(wt, "{{", "}}")
    wt = _drop_nested(wt, "{|", "|}")
    wt = re.sub(r"\[\[(?:File|Image|Category):(?:[^\[\]]|\[\[[^\]]*\]\])*\]\]", "", wt,
                flags=re.I)
    for _ in range(2):
        wt = _LINK.sub(r"\1", wt)
    wt = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", wt)
    wt = re.sub(r"<[^>]+>", "", wt).replace("'''", "").replace("''", "")
    paras = []
    for block in re.split(r"\n\s*\n", wt):
        lines = [l for l in block.split("\n") if l.strip() and l.lstrip()[0] not in "=*#|!:;{}"]
        text = re.sub(r"\s+", " ", " ".join(lines)).strip()
        text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
        if len(text) > 40:
            paras.append(text)
    return paras


# Split after . ! ? unless the period ends an initial ("C. S. Venkatakrishnan")
# or a common abbreviation.
_SENT = re.compile(r"(?<=[.!?])(?<!\b[A-Z]\.)(?<!\bSt\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bJr\.)"
                   r"\s+(?=[A-Z0-9\"'(])")


def sentences(text):
    return [s for s in _SENT.split(text) if s]


def name_patterns(aliases):
    """Full names, plus the surname alone when it is distinctive enough."""
    full = [a for a in aliases if len(a) >= 4]
    surnames = {a.split()[-1] for a in aliases if len(a.split()) >= 2}
    short = [s for s in surnames if len(s) >= 5 and s[0].isupper()]
    def rx(words):
        return re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(words, key=len, reverse=True)))
                          + r")\b") if words else None
    return rx(full), rx(short)


def mentions(text, pats):
    full, short = pats
    if full and full.search(text):
        return "full"
    if short and short.search(text):
        return "surname"
    return None


# Sentences that state who holds the role now, rather than recount history.
_CUE = re.compile(r"\b(?:current(?:ly)?|incumbent|since|is (?:the|headed|led)|serves|"
                  r"has been|headed by|led by the|presides)\b", re.I)
_HATNOTE = re.compile(r"^(?:For|See)\b.*\bsee\b|^See also\b", re.I)


def find_passage(paras, want, avoid):
    """Best window of up to 3 sentences naming `want` and not `avoid`.

    Prefers a full-name match, then a present-tense role cue, then an earlier
    paragraph (leads summarise).
    """
    best = None
    for pi, para in enumerate(paras):
        sents = sentences(para)
        for si, sent in enumerate(sents):
            kind = mentions(sent, want)
            if not kind or _HATNOTE.match(sent):
                continue
            window = " ".join(sents[max(0, si - 1):si + 2])
            if len(window) > MAX_CHARS:
                window = sent
            if mentions(window, avoid) or not MIN_CHARS <= len(window) <= MAX_CHARS:
                continue
            rank = (kind != "full", not _CUE.search(sent), pi, si)
            if best is None or rank < best[0]:
                best = (rank, window, kind)
    return (best[1], best[2]) if best else (None, None)


def _rev_fields(rev):
    text = rev["slots"]["main"].get("content", "")
    return rev["revid"], rev["timestamp"][:10], wikitext_paragraphs(text)


def revision_at(title, when):
    """(revid, date, paragraphs) of the article as it stood on `when` (ISO date)."""
    page = get_json(WP_API, {
        "action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
        "rvprop": "ids|timestamp|content", "rvslots": "main", "redirects": 1,
        "rvstart": f"{when}T00:00:00Z", "rvdir": "older",
        "format": "json", "formatversion": 2})["query"]["pages"][0]
    revs = page.get("revisions")
    return _rev_fields(revs[0]) if revs else (None, None, [])


def current_revisions(titles, batch=20):
    """{title: (revid, date, paragraphs)} for today's revisions, many per request."""
    out = {}
    titles = sorted(set(titles))
    for i in tqdm(range(0, len(titles), batch), desc="current revisions"):
        chunk = titles[i:i + batch]
        params = {"action": "query", "prop": "revisions", "titles": "|".join(chunk),
                  "rvprop": "ids|timestamp|content", "rvslots": "main", "redirects": 1,
                  "format": "json", "formatversion": 2}
        alias = {t: t for t in chunk}
        while True:
            data = get_json(WP_API, params)
            q = data.get("query", {})
            for hop in q.get("normalized", []) + q.get("redirects", []):
                for src, dst in list(alias.items()):
                    if dst == hop["from"]:
                        alias[src] = hop["to"]
            pages = {p["title"]: p for p in q.get("pages", []) if p.get("revisions")}
            for src, dst in alias.items():
                if dst in pages and src not in out:
                    out[src] = _rev_fields(pages[dst]["revisions"][0])
            if "continue" not in data:
                break
            params = dict(params, **data["continue"])
    return out


def build(fact, current):
    snapshot = (date.fromisoformat(fact["change_date"])
                - timedelta(days=SNAPSHOT_LEAD_DAYS)).isoformat()
    old_pats = name_patterns(fact["old"]["aliases"])
    new_pats = name_patterns(fact["new"]["aliases"])
    out = dict(fact, snapshot=snapshot)
    if not current:
        return dict(out, status="no_revision")
    new_rev, new_ts, new_paras = current
    # Check the current article first: no point fetching history if it fails.
    new_text, new_kind = find_passage(new_paras, new_pats, old_pats)
    if not new_text:
        return dict(out, status="no_new_passage")
    try:
        old_rev, old_ts, old_paras = revision_at(fact["article"], snapshot)
    except Exception as e:  # keep going; the row records why it failed
        return dict(out, status=f"error: {e.__class__.__name__}")
    if not old_rev:
        return dict(out, status="no_revision")
    old_text, old_kind = find_passage(old_paras, old_pats, new_pats)
    if not old_text:
        return dict(out, status="no_old_passage")
    out["old"] = dict(fact["old"], passage=old_text, match=old_kind, revid=old_rev,
                      rev_date=old_ts, cue=bool(_CUE.search(old_text)))
    out["new"] = dict(fact["new"], passage=new_text, match=new_kind, revid=new_rev,
                      rev_date=new_ts, cue=bool(_CUE.search(new_text)))
    return dict(out, status="ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1,
                    help="threads; requests are throttled globally anyway")
    args = ap.parse_args()

    facts = [json.loads(l) for l in open(FACTS_FILE, encoding="utf-8")]
    if args.limit:
        facts = facts[:args.limit]
    current = current_revisions([f["article"] for f in facts])
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(tqdm(pool.map(lambda f: build(f, current.get(f["article"])), facts),
                         total=len(facts), desc="passages"))

    status = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    with open(TASKS_FILE, "w", encoding="utf-8") as fh:
        for r in rows:
            if r["status"] == "ok":
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"status counts: {status}")
    print(f"wrote {status.get('ok', 0)} tasks to {TASKS_FILE}")


if __name__ == "__main__":
    main()
