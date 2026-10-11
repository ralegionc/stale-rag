# stale-rag

**Does retrieval-augmented generation notice when a fact is out of date?**

Real RAG corpora are full of passages that were true when written and are wrong now: an old revision, an archived press release, a cached page. This project measures what a reader model does when the retrieved context names a predecessor (the old prime minister, the previous CEO, last season's coach) instead of, or alongside, the person who holds the role today.

## Hypothesis

Readers treat retrieved text as authoritative and mostly ignore when it was written. Concretely:

1. Given only a stale passage, readers repeat the stale answer even when they know the current one from training.
2. Given a stale and a current passage, the answer depends on passage order more than on the dates attached to them.
3. Swapping the two passages' date stamps barely moves the answer. If a model really used dates, labelling the stale passage as newer would pull it toward the old holder.
4. Telling the model today's date and asking it to prefer the newest source helps, but less than it should.

## Data

Built automatically from public sources, no hand labelling:

- **Facts** (`stale_rag/facts.py`): every dated Wikidata statement for five roles that change hands: head of government (P6), head of state (P35), CEO (P169), head coach (P286) and chairperson (P488). A subject qualifies if it has exactly one current holder and a predecessor whose term ended within a year of the change, the predecessor was already in office on the snapshot date, and the subject is prominent (sitelink threshold per role). Changes run from 2008 to 90 days before the build date.
- **Passages** (`stale_rag/passages.py`): for each fact, the subject's English Wikipedia article is read at two points in time: the revision live 120 days before the change (stale), and today's revision (current). From each, the extractor takes the best window of up to three sentences that names that revision's holder and **does not** name the other one, so the two passages truly disagree. It prefers full-name matches and sentences with present-tense role cues ("incumbent", "currently", "is headed by").

A typical pair (Montana, head of government):

> **stale** (2020): The executive branch is headed by an elected governor. The governor is Steve Bullock, a Democrat elected in 2012.
>
> **current** (2026): The executive branch is headed by an elected governor. The governor is Greg Gianforte, a Republican elected in 2020.

## Design

Every task is crossed with three factors (`stale_rag/conditions.py`), 26 conditions in all:

| factor | levels |
|---|---|
| context | closed-book, current only, stale only, stale then current, current then stale |
| dates | no stamps, true revision dates, dates swapped between the passages |
| instruction | plain question, or "today is X; if sources disagree, trust the most recent" |

Two reader backends (`stale_rag/readers.py`):

- **Local scorer** (any Hugging Face chat model, CPU is fine): the log-probability of each holder's name as the complete answer. This gives a graded preference, not just a hit, and is deterministic.
- **Claude** (`--reader claude`): a short generated answer, matched against each holder's Wikidata name and aliases. Needs `ANTHROPIC_API_KEY`.

### Measures

- `p_new`: share of answers naming the current holder.
- `shift` (local scorer): `lp_new - lp_old` in a condition minus the same fact's closed-book value. This cancels how likely each name is a priori, which otherwise confounds name-scoring.
- **date effect**: `p_new` with true dates minus `p_new` with swapped dates, in the mixed contexts. Near zero means the model ignores dates.
- **cutoff split**: results split by whether the change happened before or after the reader's training cutoff. Before the cutoff the model may already know the answer; after it, the context is its only source.

## Run it

```bash
pip install -r requirements.txt
python -m stale_rag.facts                 # Wikidata -> data/facts.jsonl
python -m stale_rag.passages              # Wikipedia -> data/tasks.jsonl
python -m stale_rag.run --reader hf --model Qwen/Qwen2.5-0.5B-Instruct --limit 200
python -m stale_rag.analyze results/hf__Qwen_Qwen2.5-0.5B-Instruct.jsonl --cutoff 2024-06-01
python -m stale_rag.figures results/hf__Qwen_Qwen2.5-0.5B-Instruct.jsonl
python -m pytest
```

`run_all.sh` runs the full experiment (0.5B on every task, then 1.5B on a seeded 300-task subset). `run.py` appends to its results file and skips finished rows, so it can be stopped and resumed.

All HTTP responses are cached in `data/http_cache/`. Wikimedia rate-limits anonymous API clients, so requests are throttled to one per second; set `STALE_RAG_CONTACT` to an email or URL to identify yourself in the User-Agent. After changing extraction rules, `STALE_RAG_OFFLINE=1 python -m stale_rag.passages` rebuilds the tasks from the cache alone.

## Results

**Qwen2.5-0.5B-Instruct**, 725 facts × 26 conditions (18,850 scored prompts). Intervals are 95% bootstraps over facts. Qwen2.5-1.5B-Instruct on a 300-fact subset is in progress.

![Share of answers naming the current holder, by context and date condition](figures/conditions.png)

![Headline contrasts with 95% intervals](figures/contrasts.png)

1. **One stale passage overrides what the model knows.** Given only the stale passage, the model names the current holder 2% of the time. That includes the 174 facts where, with no passage, it prefers the current holder by at least about 150 to 1 (a log-probability margin above 5): a stale passage still flips 94% of those.
2. **With both passages and a plain question, the first passage wins.** Current-first gives the current holder 58%, stale-first 38%: an order effect of **+19 points** (16 to 23).
3. **Dates are ignored unless the model is told to use them.** With a plain question, swapping the two passages' revision dates moves the answer by **+1 point** (0.1 to 1.7). With "trust the most recent source", true dates beat swapped dates by **+12 points** (10 to 14).
4. **"Trust the most recent source" is partly read as "trust the last source."** With no dates at all, the instruction reverses the order effect to **-12 points** (-16 to -8): it pulls the answer toward whichever passage comes second, even though nothing says which one is newer.
5. **Even the best prompt leaves a lot of stale answers.** With both passages, true dates and the recency instruction, the model names the current holder 57% of the time (averaged over the two orders), 7 points better than the plain question (6 to 9).
6. **Stronger passages make the order effect bigger.** On the 273 facts where both passages state the role in the present tense ("the current governor is..."), the plain-question gap between current-first (67%) and stale-first (31%) is 36 points, against 20 on the rest.

A caveat on the training-cutoff split: at 0.5B, closed-book accuracy is flat at about 40% across every change year from 2014 to 2026, including changes after the model was trained. The model does not really know these facts, so its closed-book "knowledge" is mostly a name prior, and the before/after-cutoff comparison is not informative at this size. The 1.5B run is there to test whether a model that knows more resists stale passages better.

## Limits

- Wikidata succession data is incomplete and sometimes wrong; a fact whose "current" holder has already changed again will be scored against the wrong answer. The 90-day buffer and the one-current-holder rule reduce but do not remove this.
- The extractor works on raw wikitext with a regex stripper. It is deliberately crude, and some passages will read awkwardly.
- The "current" passage can name the new holder in a historical sentence rather than as the incumbent. Cue-word ranking makes this less common.
- Names that are only a short surname, or that share a surname with the other holder, are harder to match and are under-represented.

## License

MIT
