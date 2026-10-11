"""Readers: things that answer a rendered prompt.

HFScorer   local causal LM; scores both candidate names by log-probability.
           Cheap, deterministic, and gives a graded preference, not just a hit.
ClaudeReader  generates a short answer through the Anthropic API; the answer is
           matched against each holder's Wikidata names.
"""
import copy
import re
import unicodedata


def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def match_answer(text, task):
    """'new', 'old', 'both', or 'neither' for a free-text answer."""
    words = " " + " ".join(_norm(text)) + " "
    hit = {}
    for k in ("new", "old"):
        names = task[k]["aliases"]
        full = any(f" {' '.join(_norm(n))} " in words for n in names if _norm(n))
        surname = {_norm(n)[-1] for n in names if len(_norm(n)) >= 2}
        hit[k] = full or any(f" {s} " in words for s in surname if len(s) >= 4)
    if hit["new"] and hit["old"]:
        return "both"
    return "new" if hit["new"] else "old" if hit["old"] else "neither"


class HFScorer:
    kind = "hf"

    def __init__(self, model, threads=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.name = model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        self.model.eval()

    def _prompt_ids(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def answer(self, system, user, task):
        """Summed log-prob of each holder's label as the full reply."""
        torch = self.torch
        prompt = self._prompt_ids(system, user)
        eos = [self.tok.eos_token_id] if self.tok.eos_token_id is not None else []
        cands = {k: self.tok(task[k]["label"], add_special_tokens=False)["input_ids"] + eos
                 for k in ("new", "old")}
        out = {}
        with torch.no_grad():
            # Run the shared prompt once, then each candidate on a copy of its cache.
            first = self.model(input_ids=torch.tensor([prompt]), use_cache=True)
            for k, c in cands.items():
                cache = copy.deepcopy(first.past_key_values)
                rest = self.model(input_ids=torch.tensor([c]), past_key_values=cache).logits[0]
                logits = torch.cat([first.logits[0, -1:], rest[:-1]])
                logp = torch.log_softmax(logits.float(), dim=-1)
                out[f"lp_{k}"] = float(logp[torch.arange(len(c)), torch.tensor(c)].sum())
        out["pick"] = "new" if out["lp_new"] > out["lp_old"] else "old"
        return out


class ClaudeReader:
    kind = "claude"

    def __init__(self, model="claude-opus-5-5", effort="low"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.name = model
        self.effort = effort

    def answer(self, system, user, task):
        r = self.client.beta.messages.create(
            model=self.name,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if r.stop_reason == "refusal":
            return {"text": "", "pick": "refusal", "served_by": r.model}
        text = " ".join(b.text for b in r.content if b.type == "text").strip()
        return {"text": text, "pick": match_answer(text, task), "served_by": r.model}


def make_reader(kind, model, **kw):
    if kind == "hf":
        return HFScorer(model, **kw)
    if kind == "claude":
        return ClaudeReader(model, **kw)
    raise ValueError(kind)
