"""Travel NLP Lab Studio: extra analytics computed on the backend.

Rule: the browser never calculates ML statistics. Everything here is derived from the verified
`joined` predictions table (id, text, actual, predicted, correct, margin) or from the raw CSVs in data/raw.
Nothing is trained and nothing is invented; when data is missing the functions return None.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

TOKEN = re.compile(r"[a-z0-9']+")
# Project-level taxonomy (not a dataset label): which travel domain a raw file belongs to.
DOMAIN_FILES = {
    "flight": ("ATIS flights", ["train_atis.csv", "test_atis.csv"]),
    "hotel": ("Hotel queries", ["hotel_booking_queries.csv"]),
    "car": ("Car rental queries", ["car_rental_queries.csv"]),
    "rail": ("Rail queries", ["train_rail_queries.csv"]),
}
TEXT_COLS = ("text", "query", "user_query")
LABEL_COLS = ("intent", "label", "category", "intent_label")


def tokens(s: str) -> list[str]:
    return TOKEN.findall(str(s).lower())


def _pick(df: pd.DataFrame, names) -> str | None:
    low = {c.lower(): c for c in df.columns}
    return next((low[n] for n in names if n in low), None)


def spans_of(text: str, ner: str) -> list[dict]:
    """Entity spans (word offsets) from BIO tags; returns [] when tags and words do not line up."""
    w, tags = str(text).split(), str(ner).split()
    if len(w) != len(tags):
        return []
    out, cur = [], None
    for i, tg in enumerate(tags):
        if tg.startswith("B-"):
            cur = {"type": tg[2:], "start": i, "end": i + 1}; out.append(cur)
        elif tg.startswith("I-") and cur and cur["type"] == tg[2:]:
            cur["end"] = i + 1
        else:
            cur = None
    return out


def confusion(j: pd.DataFrame, top: int = 12) -> dict:
    labels = list(j["actual"].value_counts().index)
    for p in j["predicted"].unique():
        if p not in labels:
            labels.append(p)
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for a, p in zip(j["actual"], j["predicted"]):
        m[idx[a], idx[p]] += 1
    wrong = j[~j["correct"]]
    pairs = (wrong.groupby(["actual", "predicted"]).size().reset_index(name="count")
             .sort_values("count", ascending=False).head(top))
    # undirected view: "A <-> B" combines both directions, which is how a reader thinks about confusion
    both = Counter()
    for a, p, c in zip(wrong["actual"], wrong["predicted"], [1] * len(wrong)):
        both[tuple(sorted((a, p)))] += c
    return {
        "labels": labels, "matrix": m.tolist(), "row_totals": m.sum(axis=1).tolist(),
        "diagonal": int(np.trace(m)), "total": int(m.sum()), "errors": int(len(wrong)),
        "top_pairs": pairs.to_dict("records"),
        "mutual_pairs": [{"a": a, "b": b, "count": c} for (a, b), c in both.most_common(top)],
    }


def confidence(j: pd.DataFrame, bins: int = 10, queue: int = 12) -> dict | None:
    """Uses the classifier's decision margin. It is NOT a probability, so no 'percent sure' is claimed."""
    d = j[j["margin"].notna()].copy()
    if d.empty:
        return None
    d["margin"] = d["margin"].astype(float)
    lo, hi = float(d["margin"].min()), float(d["margin"].max())
    edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
    hist = []
    for k in range(bins):
        sel = d[(d["margin"] >= edges[k]) & ((d["margin"] < edges[k + 1]) if k < bins - 1 else (d["margin"] <= edges[k + 1]))]
        hist.append({"from": float(edges[k]), "to": float(edges[k + 1]), "correct": int(sel["correct"].sum()),
                     "wrong": int((~sel["correct"]).sum())})
    q = d["margin"].quantile([.2, .4, .6, .8]).tolist()
    cuts = [-np.inf] + q + [np.inf]
    names = ["very low", "low", "medium", "high", "very high"]
    bands = []
    for k, nme in enumerate(names):
        sel = d[(d["margin"] > cuts[k]) & (d["margin"] <= cuts[k + 1])] if k else d[d["margin"] <= cuts[1]]
        bands.append({"band": nme, "rows": int(len(sel)), "accuracy": float(sel["correct"].mean()) if len(sel) else None})
    # accept/flag curve for the threshold simulator (computed here, not in the browser)
    curve = []
    for t in np.unique(np.quantile(d["margin"], np.linspace(0, .95, 20))):
        acc_ = d[d["margin"] >= t]
        curve.append({"threshold": float(t), "accepted": int(len(acc_)), "flagged": int(len(d) - len(acc_)),
                      "coverage": len(acc_) / len(d), "accuracy_accepted": float(acc_["correct"].mean()) if len(acc_) else None})
    low = d.sort_values("margin").head(queue)[["id", "text", "actual", "predicted", "correct", "margin"]]
    return {"n": int(len(d)), "min": lo, "max": hi, "median_correct": _med(d[d["correct"]]), "median_wrong": _med(d[~d["correct"]]),
            "histogram": hist, "bands": bands, "curve": curve, "lowest": low.to_dict("records"),
            "note": "Decision margin from the classifier. Larger means further from the decision boundary. It is not a probability."}


def _med(x):
    return float(x["margin"].median()) if len(x) else None


def profile(j: pd.DataFrame) -> dict:
    t = j["text"].astype(str)
    words = t.map(lambda s: len(tokens(s)))
    chars = t.str.len()
    vocab = Counter(w for s in t for w in tokens(s))
    # error rate by query length (equal-count bins so every bin has data)
    lb = pd.qcut(words, q=min(5, words.nunique()), duplicates="drop")
    by_len = [{"label": f"{int(iv.left) + 1 if iv.left >= 0 else 0}–{int(iv.right)} words", "rows": int(len(g)),
               "error_rate": float((~g["correct"]).mean())}
              for iv, g in j.groupby(lb, observed=True)]
    dup = t.str.lower().str.strip().duplicated(keep=False)
    conflicts = j[dup].groupby(t.str.lower().str.strip())["actual"].nunique()
    per_intent = (j.assign(w=words).groupby("actual")
                  .agg(rows=("w", "size"), mean_words=("w", "mean"), error_rate=("correct", lambda s: float((~s).mean())))
                  .reset_index().sort_values("rows", ascending=False))
    return {
        "rows": int(len(j)), "unique_queries": int(t.str.lower().str.strip().nunique()),
        "duplicate_rows": int(dup.sum()), "label_conflicts": int((conflicts > 1).sum()),
        "words": {"min": int(words.min()), "max": int(words.max()), "mean": float(words.mean()), "median": float(words.median())},
        "chars": {"min": int(chars.min()), "max": int(chars.max()), "mean": float(chars.mean())},
        "vocab_size": len(vocab), "singleton_tokens": int(sum(1 for c in vocab.values() if c == 1)),
        "top_tokens": [{"token": w, "count": c} for w, c in vocab.most_common(15)],
        "word_hist": [{"words": int(k), "rows": int(v)} for k, v in words.clip(upper=30).value_counts().sort_index().items()],
        "error_by_length": by_len,
        "per_intent": per_intent.to_dict("records"),
    }


def domains(raw_dir: Path) -> list[dict]:
    """Coverage of the supplementary travel datasets found on disk. Reads whatever columns exist."""
    out = []
    for key, (title, files) in DOMAIN_FILES.items():
        parts = []
        for f in files:
            p = raw_dir / f
            if p.exists():
                try:
                    parts.append((f, pd.read_csv(p)))
                except Exception:
                    continue
        if not parts:
            continue
        info = {"domain": key, "title": title, "files": [], "rows": 0, "intents": set(), "has_ner": False, "words": []}
        for f, df in parts:
            tc, lc = _pick(df, TEXT_COLS), _pick(df, LABEL_COLS)
            if not tc:
                continue
            info["files"].append({"file": f, "rows": int(len(df))})
            info["rows"] += int(len(df))
            if lc:
                info["intents"] |= set(df[lc].dropna().astype(str))
            info["has_ner"] |= "ner" in {c.lower() for c in df.columns}
            info["words"] += [len(tokens(s)) for s in df[tc].dropna().astype(str)]
            info.setdefault("_vocab", Counter()).update(w for s in df[tc].dropna().astype(str) for w in tokens(s))
            if lc:
                info.setdefault("_dist", Counter()).update(df[lc].dropna().astype(str))
            if "samples" not in info:
                good = df[df[tc].astype(str).str.split().str.len().between(6, 14)].head(3)
                info["samples"] = [{"text": str(r[tc]), "label": str(r[lc]) if lc else None,
                                    "spans": spans_of(r[tc], r["ner"]) if "ner" in df.columns else []}
                                   for _, r in good.iterrows()]
        if not info["files"]:
            continue
        w = info.pop("words")
        v = info.pop("_vocab", Counter())
        dist = info.pop("_dist", Counter())
        info["intents"] = len(info["intents"])
        info["mean_words"] = float(np.mean(w)) if w else None
        info["vocab_size"] = len(v)
        info["top_intents"] = [{"intent": k, "rows": c} for k, c in dist.most_common(5)]
        out.append(info)
    return out


def entities_from_df(df: pd.DataFrame, source: str, top: int = 12, by_intent_top: int = 8) -> dict | None:
    """Slot annotations (dataset ground truth, not model predictions) from any frame with a text and a `ner` column.

    Works on whatever dataset the loaded run used, so a bundle without annotations simply returns None.
    Also reports an alignment check (words vs tags) and which entity types go with which intent.
    """
    tcol, ncol = _pick(df, TEXT_COLS), _pick(df, ("ner",))
    if not tcol or not ncol or df.empty:
        return None
    icol = _pick(df, LABEL_COLS)
    types, per_query = Counter(), []
    cities_from, cities_to, routes = Counter(), Counter(), Counter()
    by_int: dict[str, Counter] = {}
    by_int_q: Counter = Counter()
    by_int_n: Counter = Counter()
    mis = 0
    labels = df[icol].astype(str) if icol else [None] * len(df)
    for text, ner, lab in zip(df[tcol].astype(str), df[ncol].astype(str), labels):
        if len(text.split()) != len(ner.split()):
            mis += 1
            continue
        sp = spans_of(text, ner)
        w = text.split()
        spans = [(x["type"], w[x["start"]:x["end"]]) for x in sp]
        per_query.append(len(spans))
        for k, _ws in spans:
            types[k] += 1
        if lab is not None:
            by_int_q[lab] += 1
            by_int_n[lab] += len(spans)
            by_int.setdefault(lab, Counter()).update(k for k, _ in spans)
        o = [" ".join(ws) for k, ws in spans if k == "fromloc.city_name"]
        d = [" ".join(ws) for k, ws in spans if k == "toloc.city_name"]
        cities_from.update(o)
        cities_to.update(d)
        if o and d:
            routes[(o[0], d[0])] += 1
    if not per_query:
        return None
    dist = Counter(min(c, 5) for c in per_query)
    return {"source": source, "queries": len(per_query), "mean_entities": float(np.mean(per_query)),
            "alignment": {"checked": len(per_query) + mis, "misaligned": mis,
                          "rate": mis / (len(per_query) + mis),
                          "note": "Queries whose word count differs from their tag count are skipped, not repaired."},
            "types": [{"type": k, "count": c} for k, c in types.most_common(top)],
            "count_distribution": [{"entities": k, "queries": dist.get(k, 0), "label": f"{k}+" if k == 5 else str(k)} for k in range(0, 6)],
            "by_intent": [{"intent": k, "queries": int(by_int_q[k]), "mean_entities": by_int_n[k] / by_int_q[k],
                           "types": [{"type": t, "count": c} for t, c in by_int[k].most_common(4)]}
                          for k, _ in by_int_q.most_common(by_intent_top)] if icol else [],
            "origins": [{"city": k, "count": c} for k, c in cities_from.most_common(8)],
            "destinations": [{"city": k, "count": c} for k, c in cities_to.most_common(8)],
            "routes": [{"origin": a, "destination": b, "count": c} for (a, b), c in routes.most_common(8)]}


def entities(raw_dir: Path, top: int = 12) -> dict | None:
    """Studio-folder example data (used by the welcome page before any run is open)."""
    p = raw_dir / "train_atis.csv"
    if not p.exists():
        return None
    return entities_from_df(pd.read_csv(p), "train_atis.csv (annotation)", top)


def errors_by_entity_count(j: pd.DataFrame, raw_test: pd.DataFrame, fname: str, min_match: float = 0.99) -> dict | None:
    """Mistake rate by how many annotated entities a test query contains.

    Stored predictions are joined to the raw annotated test file by record id (`<file>::<id>`). The join is only
    used when at least `min_match` of the predictions find a row with identical text, so a wrong join never shows numbers.
    """
    tcol, ncol = _pick(raw_test, TEXT_COLS), _pick(raw_test, ("ner",))
    if j is None or not tcol or not ncol or "id" not in raw_test.columns:
        return None
    key = {f"{fname}::{i}": (str(t).strip(), str(n)) for i, t, n in zip(raw_test["id"], raw_test[tcol], raw_test[ncol])}
    rows, hit = [], 0
    for rid, text, ok in zip(j["id"].astype(str), j["text"].astype(str), j["correct"]):
        k = key.get(rid)
        if k is None or k[0] != text.strip():
            continue
        hit += 1
        w, tg = k[0].split(), k[1].split()
        if len(w) != len(tg):
            continue
        rows.append((min(len(spans_of(k[0], k[1])), 4), bool(ok)))
    if not len(j) or hit / len(j) < min_match or not rows:
        return None
    out = []
    for c in range(0, 5):
        sel = [ok for n, ok in rows if n == c]
        if sel:
            out.append({"entities": c, "label": f"{c}+" if c == 4 else str(c), "rows": len(sel),
                        "errors": sum(1 for ok in sel if not ok), "error_rate": sum(1 for ok in sel if not ok) / len(sel)})
    return {"basis": f"{hit} of {len(j)} stored predictions matched an annotated row in {fname}",
            "matched": hit, "rows": out,
            "note": "Annotations are dataset ground truth. A pattern here describes the data; it does not explain the model."}


def error_lab(j: pd.DataFrame, limit: int = 120) -> dict:
    """Group every mistake by simple, checkable properties and list heuristic contributing factors.

    Factors are HEURISTICS (shared wording, length, margin). They describe the data, they do not explain the model's reasoning.
    """
    t = j["text"].astype(str)
    toks = t.map(tokens)
    words = toks.map(len)
    norm = t.str.lower().str.strip()
    dup = norm.duplicated(keep=False)
    q_short, q_long = float(words.quantile(.2)), float(words.quantile(.8))
    m_low = float(j["margin"].astype(float).quantile(.2)) if j["margin"].notna().any() else None
    # document frequency of each token inside each intent (share of that intent's queries containing it)
    df_by: dict[str, Counter] = {}
    size = j["actual"].value_counts().to_dict()
    for a, tk in zip(j["actual"], toks):
        df_by.setdefault(a, Counter()).update(set(tk))
    stop = {w for w, c in Counter(w for tk in toks for w in set(tk)).items() if c / len(j) > .12}

    def share(intent, w):
        return df_by.get(intent, {}).get(w, 0) / max(size.get(intent, 1), 1)

    right = j[j["correct"]]
    right_tok = {i: set(tokens(x)) for i, x in zip(right.index, right["text"])}
    wrong = j[~j["correct"]]
    cats = Counter()
    out = []
    for i, r in wrong.iterrows():
        tk = set(toks[i])
        c = []
        if m_low is not None and pd.notna(r["margin"]) and float(r["margin"]) <= m_low:
            c.append("low_margin")
        if words[i] <= q_short:
            c.append("short_query")
        if words[i] >= q_long:
            c.append("long_query")
        if dup[i]:
            c.append("repeated_query")
        shared = sorted(w for w in tk if len(w) > 2 and w not in stop and share(r["actual"], w) >= .12 and share(r["predicted"], w) >= .12)[:6]
        if shared:
            c.append("shared_wording")
        for k in c:
            cats[k] += 1
        sims = sorted(((len(tk & s) / max(len(tk | s), 1), idx) for idx, s in right_tok.items() if j.at[idx, "actual"] == r["actual"]), reverse=True)[:2]
        out.append({"id": r["id"], "text": r["text"], "actual": r["actual"], "predicted": r["predicted"],
                    "margin": None if pd.isna(r["margin"]) else float(r["margin"]), "words": int(words[i]),
                    "categories": c, "shared_words": shared,
                    "similar_correct": [j.at[idx, "text"] for s_, idx in sims if s_ > 0]})
    out.sort(key=lambda e: (e["margin"] is None, e["margin"] if e["margin"] is not None else 0))
    return {"errors": int(len(wrong)), "categories": dict(cats), "thresholds": {"short_words_max": q_short, "long_words_min": q_long, "low_margin_max": m_low},
            "items": out[:limit], "shown": min(limit, len(out)),
            "note": "Categories and 'shared wording' are heuristics about the data. They are possible contributing factors, not the model's reasoning."}


def intent_map(clf: dict, per_intent: list | None, pairs: list | None, top_terms: int = 6) -> dict:
    """3D map of the intents, from the classifier's own weight vectors (nothing is retrained or invented).

    Each intent has one weight vector over the vocabulary. The vectors are centred across intents, so what is left is
    "how much more this word favours this intent than the average intent", which is comparable for SVM, logistic
    regression and Naive Bayes alike. PCA then gives the three directions that separate the intents most.
    Distances in 3D are an approximation: the share of variance each axis keeps is returned so it can be shown.
    """
    classes = list(clf["classes"])
    W = np.asarray(clf["_W"], dtype=float)
    k = len(classes)
    if k < 3:
        raise ValueError("The map needs at least 3 intents.")
    C = W - W.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(C, full_matrices=False)
    var = S ** 2
    total = float(var.sum()) or 1.0
    d = min(3, len(S))
    xyz = U[:, :d] * S[:d]
    xyz = np.pad(xyz, ((0, 0), (0, 3 - d)))
    # fix the sign of each axis so the same run always draws the same picture
    for a in range(3):
        if np.abs(xyz[:, a]).max() > 0 and xyz[np.abs(xyz[:, a]).argmax(), a] < 0:
            xyz[:, a] *= -1
    scale = float(np.abs(xyz).max()) or 1.0
    xyz = xyz / scale
    norms = np.linalg.norm(C, axis=1)
    norms[norms == 0] = 1.0
    cos = (C / norms[:, None]) @ (C / norms[:, None]).T
    vocab_inv = {i: t for t, i in clf["vocab"].items()}
    stats = {r["intent"]: r for r in (per_intent or [])}
    items = []
    for i, name in enumerate(classes):
        order = np.argsort(-C[i])[:top_terms]
        nn = [j for j in np.argsort(-cos[i]) if j != i][:3]
        st = stats.get(name, {})
        items.append({
            "intent": name, "x": float(xyz[i, 0]), "y": float(xyz[i, 1]), "z": float(xyz[i, 2]),
            "f1": None if st.get("f1") is None else float(st["f1"]),
            "support": None if st.get("support") is None else int(st["support"]),
            "top_terms": [vocab_inv.get(int(t), "?") for t in order],
            "nearest": [{"intent": classes[j], "similarity": float(cos[i, j])} for j in nn],
        })
    idx = {c: i for i, c in enumerate(classes)}
    edges = [{"a": p["a"], "b": p["b"], "count": int(p["count"]), "similarity": float(cos[idx[p["a"]], idx[p["b"]]])}
             for p in (pairs or []) if p["a"] in idx and p["b"] in idx]
    # does "the model gives these two intents similar word weights" go with "the model confuses them"?
    sim_check = None
    if pairs is not None:
        conf = {tuple(sorted((p["a"], p["b"]))) for p in pairs if p["a"] in idx and p["b"] in idx}
        iu = np.triu_indices(k, 1)
        flags = np.array([tuple(sorted((classes[a], classes[b]))) in conf for a, b in zip(*iu)])
        vals = cos[iu]
        if flags.any() and (~flags).any():
            sim_check = {"confused_pairs": int(flags.sum()), "other_pairs": int((~flags).sum()),
                         "mean_similarity_confused": float(vals[flags].mean()),
                         "mean_similarity_other": float(vals[~flags].mean())}
    return {"intents": items, "edges": edges,
            "explained": [float(v / total) for v in var[:d]] + [0.0] * (3 - d),
            "explained_total": float(var[:d].sum() / total),
            "model": clf.get("model_name"), "features": int(W.shape[1]), "similarity_check": sim_check}
