"""Summarise a results file.

    python -m stale_rag.analyze results/hf__Qwen_Qwen2.5-0.5B-Instruct.jsonl --cutoff 2024-06-01

Headline numbers:
  p_new        share of answers that give the current holder
  margin       mean lp_new - lp_old (hf only)
  shift        margin minus the same fact's closed-book margin, which cancels
               how likely each name is a priori (hf only)

Contrasts, each a difference in p_new with a 95% bootstrap interval over facts:
  stale override   closed-book minus stale-only, on facts the model already
                   gets right closed-book: how much one stale passage undoes
  order effect     current-first minus stale-first (no dates)
  date effect      true dates minus swapped dates, mixed contexts; near zero
                   means dates are ignored
  last-source pull with the recency instruction, (stale-first minus
                   current-first) with no dates: positive means "most recent"
                   is read as "last"
"""
import argparse
import json

import numpy as np
import pandas as pd

from .config import TASKS_FILE

MIXED = ["old_new", "new_old"]


def load(path, tasks_file=TASKS_FILE):
    res = pd.read_json(path, lines=True)
    rows = [json.loads(l) for l in open(tasks_file, encoding="utf-8")]
    tasks = pd.DataFrame([{
        "fact_id": t["fact_id"], "relation": t["relation"], "change_date": t["change_date"],
        "cued": bool(t["old"].get("cue") and t["new"].get("cue")),
    } for t in rows])
    return res.merge(tasks, on="fact_id", how="inner")


def summarise(df):
    df = df.copy()
    df["is_new"] = (df["pick"] == "new").astype(float)
    g = ["context", "dates", "instruct"]
    cols = {"n": ("is_new", "size"), "p_new": ("is_new", "mean")}
    if "lp_new" in df:
        df["margin"] = df["lp_new"] - df["lp_old"]
        closed = (df[df.context == "closed"].groupby(["fact_id", "instruct"])["margin"]
                  .mean().rename("closed_margin"))
        df = df.join(closed, on=["fact_id", "instruct"])
        df["shift"] = df["margin"] - df["closed_margin"]
        cols.update(margin=("margin", "mean"), shift=("shift", "mean"))
    return df, df.groupby(g).agg(**cols).round(3)


def _cell(df, **where):
    """Per-fact mean of is_new over rows matching `where` (lists mean 'any of')."""
    m = pd.Series(True, index=df.index)
    for k, v in where.items():
        m &= df[k].isin(v) if isinstance(v, list) else df[k] == v
    return df[m].groupby("fact_id")["is_new"].mean()


def contrast(a, b, n_boot=2000, seed=0):
    """Mean of a - b over facts present in both, with a bootstrap 95% CI."""
    d = (a - b).dropna().to_numpy()
    if len(d) == 0:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "facts": 0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return {"diff": d.mean(), "lo": np.percentile(boots, 2.5),
            "hi": np.percentile(boots, 97.5), "facts": len(d)}


def contrasts(df):
    out = {}
    for ins in ["plain", "recency"]:
        closed = _cell(df, context="closed", instruct=ins)
        known = closed[closed == 1].index
        out[f"stale override ({ins})"] = contrast(
            closed.loc[known], _cell(df, context="old", instruct=ins).reindex(known))
        out[f"order effect ({ins})"] = contrast(
            _cell(df, context="new_old", dates="none", instruct=ins),
            _cell(df, context="old_new", dates="none", instruct=ins))
        out[f"date effect ({ins})"] = contrast(
            _cell(df, context=MIXED, dates="true", instruct=ins),
            _cell(df, context=MIXED, dates="swapped", instruct=ins))
    out["recency instruction, mixed, true dates"] = contrast(
        _cell(df, context=MIXED, dates="true", instruct="recency"),
        _cell(df, context=MIXED, dates="true", instruct="plain"))
    return pd.DataFrame(out).T.round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cutoff", help="reader's training cutoff, YYYY-MM-DD")
    ap.add_argument("--tasks", default=str(TASKS_FILE))
    args = ap.parse_args()

    df, table = summarise(load(args.path, args.tasks))
    pd.set_option("display.width", 160)
    print(f"{df.fact_id.nunique()} facts, {len(df)} rows\n")
    print(table, "\n")
    print("contrasts (difference in p_new, 95% bootstrap CI over facts):")
    print(contrasts(df), "\n")
    if args.cutoff:
        df["after_cutoff"] = df["change_date"] > args.cutoff
        print("p_new by context, split by whether the change postdates the cutoff:")
        print(df.pivot_table(index="context", columns="after_cutoff", values="is_new",
                             aggfunc="mean").round(3), "\n")
    m = df[df.context.isin(MIXED) & (df.dates == "true")]
    print("p_new in mixed contexts with true dates, by passage strength (both cued):")
    print(m.pivot_table(index="context", columns=["instruct", "cued"], values="is_new",
                        aggfunc="mean").round(3), "\n")
    print("p_new by relation (mixed contexts, true dates):")
    print(m.groupby("relation")["is_new"].agg(["size", "mean"]).round(3))
    table.to_csv(str(args.path).replace(".jsonl", "_summary.csv"))
    contrasts(df).to_csv(str(args.path).replace(".jsonl", "_contrasts.csv"))


if __name__ == "__main__":
    main()
