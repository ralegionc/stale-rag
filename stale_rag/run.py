"""Run one reader over every task x condition. Resumable.

    python -m stale_rag.run --reader hf --model Qwen/Qwen2.5-0.5B-Instruct --limit 100
    python -m stale_rag.run --reader claude --model claude-opus-5-5 --limit 100
"""
import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from .conditions import condition_key, conditions, render
from .config import RESULTS, TASKS_FILE, TODAY
from .readers import make_reader


def results_path(reader, model):
    return RESULTS / f"{reader}__{model.replace('/', '_')}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", choices=["hf", "claude"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--limit", type=int, help="number of tasks (random, seeded)")
    ap.add_argument("--threads", type=int, default=4, help="torch threads for hf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default=str(TASKS_FILE))
    ap.add_argument("--out", help="results file (default: results/<reader>__<model>.jsonl)")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8")]
    random.Random(args.seed).shuffle(tasks)
    if args.limit:
        tasks = tasks[:args.limit]

    out = Path(args.out) if args.out else results_path(args.reader, args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["fact_id"], r["cond"]))

    todo = [(t, c) for t in tasks for c in conditions()
            if (t["fact_id"], condition_key(c)) not in done]
    print(f"{len(tasks)} tasks, {len(conditions())} conditions, {len(todo)} to run")
    if not todo:
        return
    kw = {"threads": args.threads} if args.reader == "hf" else {}
    reader = make_reader(args.reader, args.model, **kw)
    with open(out, "a", encoding="utf-8") as fh:
        for task, cond in tqdm(todo):
            system, user = render(task, cond, TODAY.isoformat())
            res = reader.answer(system, user, task)
            fh.write(json.dumps({"fact_id": task["fact_id"], "cond": condition_key(cond),
                                 **cond, **res}, ensure_ascii=False) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
