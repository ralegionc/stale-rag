"""Experimental conditions and prompt rendering.

Three factors, crossed:
  context   which passages the reader sees, and in what order
  dates     no date stamps, true revision dates, or the two dates swapped
  instruct  plain question, or today's date plus "prefer the newest source"

Swapped dates are the control that matters: if a model really reads dates,
labelling the stale passage as newer should pull it toward the old answer.
"""
from itertools import product

CONTEXTS = ["closed", "new", "old", "old_new", "new_old"]
DATES = ["none", "true", "swapped"]
INSTRUCTS = ["plain", "recency"]

SYSTEM = "Answer the question with just the person's full name."


def conditions():
    """All condition dicts; closed-book ignores the date factor."""
    out = []
    for ctx, dates, ins in product(CONTEXTS, DATES, INSTRUCTS):
        if ctx == "closed" and dates != "none":
            continue
        out.append({"context": ctx, "dates": dates, "instruct": ins})
    return out


def condition_key(c):
    return f"{c['context']}|{c['dates']}|{c['instruct']}"


def _stamp(task, which, dates):
    if dates == "none":
        return ""
    other = "old" if which == "new" else "new"
    src = which if dates == "true" else other
    return f" (Wikipedia, revision of {task[src]['rev_date']})"


def render(task, cond, today):
    """Return (system, user) strings for one task under one condition."""
    order = {"closed": [], "new": ["new"], "old": ["old"],
             "old_new": ["old", "new"], "new_old": ["new", "old"]}[cond["context"]]
    parts = []
    if cond["instruct"] == "recency":
        parts.append(f"Today's date is {today}. If the sources disagree, "
                     "trust the most recent one.")
    for i, which in enumerate(order, 1):
        stamp = _stamp(task, which, cond["dates"])
        parts.append(f"Source {i}{stamp}:\n{task[which]['passage']}")
    if order:
        parts.append("Using the sources above, answer the question.")
    parts.append(f"Question: {task['question']}")
    return SYSTEM, "\n\n".join(parts)
