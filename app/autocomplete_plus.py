"""Autocomplete v2 for Travel NLP Lab Studio.

This module sits ON TOP of the original engine. It never edits `engine.next_word_suggestions` or
`engine.intent_hint` (those must stay identical to the notebook). It only reads the exported n-gram counts and
adds: confidence levels, "did you mean" typo help, safer behaviour on unknown words, a ghost-text suggestion that
keeps what the person typed, and an independent accuracy checker that recomputes the notebook's numbers.
"""
from __future__ import annotations

import math
from typing import Any

from . import engine as E

# ----------------------------------------------------------------------------- small helpers


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Damerau-Levenshtein (adjacent swaps count as 1), stops early above `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


def did_you_mean(model: dict, word: str, k: int = 3) -> list[dict]:
    """Known words that are 1-2 edits away from `word` (most frequent first). Empty if `word` is already known."""
    word = word.lower()
    wc = model["word_counts"]
    if not word or word in wc or len(word) < 3:
        return []
    cap = 1 if len(word) <= 4 else 2
    hits = []
    for w, c in wc.items():
        if abs(len(w) - len(word)) > cap or w == E.END_TOKEN:
            continue
        d = _edit_distance(word, w, cap)
        if d <= cap:
            hits.append((d, -c, w))
    hits.sort()
    return [{"word": w, "edits": d} for d, _, w in hits[:k]]


def confidence_label(share: float, context_hit: bool = True) -> str:
    """Plain-language confidence for one suggestion. `share` is its fraction of the candidates shown."""
    if not context_hit:
        return "low"
    return "high" if share >= 0.5 else "medium" if share >= 0.2 else "low"


def matched_context(model: dict, prefix_tokens) -> int:
    """How many previous words the model actually matched (0 = it only knows general word frequency)."""
    max_ctx = int(model["max_context_tokens"])
    seq = ["<s>"] + list(prefix_tokens)[-max_ctx:]
    for n in range(max_ctx, 0, -1):
        if n <= len(seq) and " ".join(seq[-n:]) in model["contexts"]:
            return n if seq[-n] != "<s>" else n - 1
    return 0


# ----------------------------------------------------------------------------- the live suggestion call


def suggest(model: dict, typed: str, n_words: int = 6, n_queries: int = 4) -> dict[str, Any]:
    """Everything the UI needs for what has been typed so far. Superset of `engine.autocomplete`."""
    typed = str(typed)[:300]
    base = E.autocomplete(model, typed, n_words=n_words, n_queries=n_queries)
    finished, partial = base["finished"], base["partial"]
    wc = model["word_counts"]

    # --- token view: which typed words does the model know?
    tokens = [{"text": w, "known": w in wc, "partial": False} for w in finished]
    if partial:
        has_prefix = any(w.startswith(partial) for w in wc)
        tokens.append({"text": partial, "known": has_prefix, "partial": True})
    unknown_final = [t["text"] for t in tokens if not t["known"] and not t["partial"]]
    unknown_partial = [t["text"] for t in tokens if t["partial"] and not t["known"]]

    # --- typo help for unknown finished words and for a half-typed word nothing starts with
    fixes = []
    for w in unknown_final + unknown_partial:
        sug = did_you_mean(model, w)
        if sug:
            fixes.append({"typed": w, "options": sug})

    # --- confidence per next word
    ctx_n = matched_context(model, finished + ([base["word_completions"][0]["word"]] if base["word_completions"] else []))
    for w in base["next_words"]:
        w["confidence"] = confidence_label(w["share"], ctx_n > 0)
    for w in base["word_completions"]:
        w["confidence"] = confidence_label(w["share"], w.get("fits_context", True))

    # --- how sure are we about the whole thing? (drives whether ghost text is shown at all)
    n_known = sum(1 for t in tokens if t["known"])
    coverage = n_known / len(tokens) if tokens else 1.0
    top_share = (base["word_completions"][0]["share"] if partial and base["word_completions"]
                 else base["next_words"][0]["share"] if base["next_words"] else 0.0)
    if tokens and coverage < 0.5:
        level = "unsure"
    elif unknown_final or unknown_partial:
        level = "careful"
    else:
        level = confidence_label(top_share, ctx_n > 0)
    base["level"] = level  # high | medium | low | careful | unsure
    base["context_words"] = ctx_n
    base["tokens"] = tokens
    base["fixes"] = fixes
    base["coverage"] = round(coverage, 3)

    # --- ghost text: keep EXACTLY what the person typed, only add the rest
    ghost = ""
    if level != "unsure" and typed.strip():
        lower = typed.lower()
        best_full = next((c for c in base["completions"] if c["text"].startswith(lower.rstrip()) and len(c["text"]) > len(lower.rstrip())), None)
        if partial and base["word_completions"] and base["word_completions"][0]["fits_context"]:
            w = base["word_completions"][0]["word"]
            ghost = w[len(partial):]
            if best_full and best_full["text"].startswith(" ".join(finished + [w])):
                rest = best_full["text"][len(" ".join(finished + [w])):]
                ghost += rest
        elif not partial and best_full is None and base["next_words"]:
            top = base["next_words"][0]
            if top["word"] != E.END_TOKEN and top["confidence"] != "low":
                ghost = ("" if typed.endswith(" ") else " ") + top["word"]
        elif not partial and best_full is not None:
            ghost = best_full["text"][len(lower.rstrip()):]
            if typed.endswith(" ") and ghost.startswith(" "):
                ghost = ghost[1:]
        if level in ("low", "careful") and len(ghost.split()) > 1:
            ghost = ghost.split(" ")[0] if not ghost.startswith(" ") else " " + ghost.split()[0]
    if level in ("unsure", "careful"):
        ghost = ""  # never guess after a word the model has not seen
    base["ghost"] = ghost
    base["can_finish"] = bool(typed.strip()) and any(w["word"] == E.END_TOKEN for w in base["next_words"][:1])
    # END_TOKEN is internal; show it as "end of sentence" in the UI.
    for w in base["next_words"]:
        w["is_end"] = w["word"] == E.END_TOKEN
    return base


# ----------------------------------------------------------------------------- independent accuracy checker


def _positions(model, queries):
    for q in queries:
        toks = E.next_word_tokens(q, limit=10**6)
        yield toks, toks + [E.END_TOKEN]


def evaluate(model: dict, queries: list[str], intents: list[str] | None = None, sample_for_slow: int = 300) -> dict[str, Any]:
    """Recompute next-word accuracy from raw held-out queries. Uses the SAME scoring function as the notebook.

    Fast part (every position): top-1/3/5, accuracy by position in the sentence, and by number of known context words.
    Slower part (a deterministic sample): half-typed-word completion and keystroke saving.
    """
    pos = h1 = h3 = h5 = 0
    by_pos: dict[int, list[int]] = {}
    for toks, seq in _positions(model, queries):
        for i, tgt in enumerate(seq):
            ranked = [w for w, _ in E.next_word_suggestions(model, toks[:i], 5)]
            pos += 1
            h1 += tgt in ranked[:1]
            h3 += tgt in ranked[:3]
            h5 += tgt in ranked[:5]
            b = by_pos.setdefault(min(i, 11), [0, 0])
            b[0] += 1
            b[1] += tgt in ranked[:3]
    out: dict[str, Any] = {
        "queries": len(queries), "positions": pos,
        "top1": h1 / pos if pos else None, "top3": h3 / pos if pos else None, "top5": h5 / pos if pos else None,
        "by_position": [{"position": k + 1, "label": f"word {k + 1}" if k < 11 else "word 12+", "positions": v[0], "top3": v[1] / v[0]}
                        for k, v in sorted(by_pos.items())],
    }
    # frequency baseline: always suggest the 3 most common "next words" overall (the model's empty-context table,
    # which includes the end-of-query marker, exactly like the notebook's unigram baseline)
    uni = (model["contexts"].get("") or [0, {}])[1]
    top_words = [w for w, _ in sorted(((w, c) for w, c in uni.items() if w != "<s>"), key=lambda kv: (-kv[1], kv[0]))[:3]]
    base_hits = sum(1 for _, seq in _positions(model, queries) for t in seq if t in top_words)
    out["baseline_top3"] = base_hits / pos if pos else None

    # --- slow part on a deterministic sample (every n-th query)
    step = max(1, len(queries) // max(1, sample_for_slow))
    sample = queries[::step][:sample_for_slow]
    typed_chars = saved_chars = total_chars = 0
    pw = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}  # chars typed -> [n, top1, top3]
    for toks, _ in _positions(model, sample):
        for i, word in enumerate(toks):
            ctx = toks[:i]
            total_chars += len(word) + 1
            typed_here = len(word) + 1
            for k in range(1, len(word)):
                cands = E.complete_partial_word(model, ctx, word[:k], 3)
                ws = [c["word"] for c in cands]
                if k in pw and len(word) > k:
                    pw[k][0] += 1
                    pw[k][1] += ws[:1] == [word]
                    pw[k][2] += word in ws
                if ws[:1] == [word]:
                    typed_here = k + 1  # k characters + one Tab
                    break
            typed_chars += typed_here
    out["word_completion"] = [{"typed_chars": k, "words": v[0], "top1": v[1] / v[0] if v[0] else None, "top3": v[2] / v[0] if v[0] else None}
                              for k, v in pw.items()]
    out["keystroke_saving"] = 1 - typed_chars / total_chars if total_chars else None
    out["sample_queries"] = len(sample)

    # --- intent estimate while typing (reported separately; computed here from the same intent_hint function)
    if intents is not None:
        n = ok1 = ok3 = 0
        for q, y in zip(queries, intents):
            t = E.next_word_tokens(q, limit=10**6)
            if len(t) < 2:
                continue
            ranked = [i for i, _ in E.intent_hint(model, t[: math.ceil(len(t) / 2)], 3)]
            n += 1
            ok1 += bool(ranked) and ranked[0] == y
            ok3 += y in ranked
        out["intent_half"] = {"queries": n, "top1": ok1 / n if n else None, "top3": ok3 / n if n else None}
    return out


def compare_with_stored(model: dict, fresh: dict) -> dict[str, Any]:
    """Line the fresh numbers up against what the notebook stored, with a plain verdict."""
    stored = model.get("evaluation") or {}
    rows, all_ok, any_checked = [], True, False
    for label, key_fresh, key_stored in (("Top-1 next word", "top1", "top1_accuracy"),
                                         ("Top-3 next word", "top3", "top3_accuracy"),
                                         ("Top-5 next word", "top5", "top5_accuracy"),
                                         ("Frequent-words baseline (top-3)", "baseline_top3", "frequent_words_baseline_top3")):
        a, b = fresh.get(key_fresh), stored.get(key_stored)
        if a is None or b is None:
            rows.append({"metric": label, "recomputed": a, "stored": b, "match": None})
            continue
        any_checked = True
        ok = abs(a - b) <= 5e-4
        all_ok &= ok
        rows.append({"metric": label, "recomputed": a, "stored": b, "match": ok})
    sp = stored.get("positions")
    pos_ok = sp is None or sp == fresh["positions"]
    rows.append({"metric": "Word positions scored", "recomputed": fresh["positions"], "stored": sp, "match": pos_ok if sp is not None else None})
    all_ok &= pos_ok
    status = "unchecked" if not any_checked else "verified" if all_ok else "mismatch"
    return {"status": status, "rows": rows}
