"""Pull facts that changed hands from Wikidata.

For each relation we fetch every dated statement on prominent subjects, then
keep subjects with exactly one current holder and a clear predecessor whose
term ended around when the current one started.

    python -m stale_rag.facts
"""
import json
from datetime import date, timedelta

from tqdm import tqdm

from .config import (CHANGE_FROM, CHANGE_MIN_AGE_DAYS, FACTS_FILE,
                     RELATIONS, SNAPSHOT_LEAD_DAYS, TODAY, WD_API, WDQS)
from .http import get_json

# Labels, sitelinks and articles are fetched afterwards via the Wikidata API:
# asking WDQS for them on every statement times out.
QUERY = """
SELECT ?s ?v ?start ?end WHERE {
  ?s p:%s ?st .
  ?st ps:%s ?v ; pq:P580 ?start .
  OPTIONAL { ?st pq:P582 ?end }
  ?st wikibase:rank ?rank . FILTER(?rank != wikibase:DeprecatedRank)
}"""

ENTITY = "http://www.wikidata.org/entity/"


def parse_day(value):
    """Wikidata xsd:dateTime -> date, or None for blank nodes and odd values."""
    if not value or not value[:1].isdigit():
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def qid(uri):
    return uri[len(ENTITY):] if uri.startswith(ENTITY) else None


def fetch_statements(pid):
    data = get_json(WDQS, {"query": QUERY % (pid, pid), "format": "json"})
    rows = []
    for b in data["results"]["bindings"]:
        s, v = qid(b["s"]["value"]), qid(b["v"]["value"])
        start = parse_day(b["start"]["value"])
        if not (s and v and start):
            continue
        end_raw = b.get("end", {}).get("value")
        end = parse_day(end_raw)
        if end_raw and not end:  # end exists but is unknown: not current
            end = date.min
        rows.append({"subject": s, "value": v, "start": start, "end": end})
    return rows


def pick_change(statements, today=TODAY):
    """From one subject's statements, return (current, predecessor) or None."""
    current = {s["value"]: s for s in statements if s["end"] is None}
    if len(current) != 1:
        return None
    cur = next(iter(current.values()))
    change = cur["start"]
    if not (CHANGE_FROM <= change <= today - timedelta(days=CHANGE_MIN_AGE_DAYS)):
        return None
    year = timedelta(days=365)
    prev = [s for s in statements
            if s["end"] not in (None, date.min) and s["value"] != cur["value"]
            and change - year <= s["end"] <= change + year]
    if not prev:
        return None
    old = max(prev, key=lambda s: s["end"])
    snapshot = change - timedelta(days=SNAPSHOT_LEAD_DAYS)
    if old["start"] > snapshot:  # predecessor's term too short to be written up
        return None
    return cur, old


def fetch_entities(qids, props):
    """wbgetentities in batches of 50; returns {qid: entity}."""
    out = {}
    qids = sorted(set(qids))
    for i in range(0, len(qids), 50):
        data = get_json(WD_API, {"action": "wbgetentities", "ids": "|".join(qids[i:i + 50]),
                                 "props": props, "languages": "en", "format": "json"})
        out.update(data.get("entities", {}))
    return out


def names(entity):
    """English label first, then aliases."""
    found = [entity.get("labels", {}).get("en", {}).get("value")]
    found += [a["value"] for a in entity.get("aliases", {}).get("en", [])]
    return [n for n in dict.fromkeys(found) if n]


def main():
    candidates = []
    for pid, spec in RELATIONS.items():
        rows = fetch_statements(pid)
        by_subject = {}
        for r in rows:
            by_subject.setdefault(r["subject"], []).append(r)
        picked = [(pid, pick_change(st)) for st in by_subject.values()]
        picked = [(p, c) for p, c in picked if c]
        print(f"{pid} {spec['name']}: {len(rows)} statements, "
              f"{len(by_subject)} subjects, {len(picked)} with a clean change")
        candidates += picked

    subjects = fetch_entities([c[0]["subject"] for _, c in candidates], "labels|sitelinks")
    holders = fetch_entities([c[i]["value"] for _, c in candidates for i in (0, 1)],
                             "labels|aliases")
    facts = []
    for pid, (cur, old) in tqdm(candidates, desc="filter"):
        spec = RELATIONS[pid]
        subj = subjects.get(cur["subject"], {})
        links = subj.get("sitelinks", {})
        label = subj.get("labels", {}).get("en", {}).get("value")
        new_names, old_names = (names(holders.get(c["value"], {})) for c in (cur, old))
        if (len(links) < spec["min_sitelinks"] or "enwiki" not in links or not label
                or not new_names or not old_names):
            continue
        facts.append({
            "fact_id": f"{pid}_{cur['subject']}", "relation": pid,
            "relation_name": spec["name"], "subject": cur["subject"],
            "subject_label": label, "article": links["enwiki"]["title"],
            "sitelinks": len(links),
            "question": spec["question"].format(s=label),
            "new": {"qid": cur["value"], "label": new_names[0], "aliases": new_names},
            "old": {"qid": old["value"], "label": old_names[0], "aliases": old_names,
                    "start": old["start"].isoformat(), "end": old["end"].isoformat()},
            "change_date": cur["start"].isoformat(),
        })

    FACTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FACTS_FILE, "w", encoding="utf-8") as fh:
        for f in facts:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    by_rel = {}
    for f in facts:
        by_rel[f["relation"]] = by_rel.get(f["relation"], 0) + 1
    print(f"wrote {len(facts)} facts to {FACTS_FILE}: {by_rel}")


if __name__ == "__main__":
    main()
