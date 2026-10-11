"""Travel NLP Lab Studio: analysis backend.

Everything runs on this computer. It only reads a finished run bundle (a ZIP made by the notebook), verifies it
with the original engine and computes every number the Streamlit app shows. The Streamlit app calls these
functions directly (in-process); the same functions are also exposed as a local JSON API for testing.
It never trains and never calls an external service.
"""
from __future__ import annotations

import io
import json
import math
import mimetypes
import random
import re
import threading
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from . import analytics as AN
from . import autocomplete_plus as AC
from . import engine as E

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"example": ROOT / "examples", "mine": ROOT / "output"}
BAD = {"mismatch", "missing", "ambiguous", "contract_mismatch", "unmanifested", "no_file_hashes"}
MAX_SESSIONS = 8
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()

APP_VERSION = "4.3.1"
_LOCK = threading.Lock()
app = FastAPI(title="Travel NLP Lab Studio", version=APP_VERSION, docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")


# ----------------------------------------------------------------------------- helpers
def num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def rows(df, limit=None):
    if not isinstance(df, pd.DataFrame):
        return []
    return json.loads((df.head(limit) if limit else df).to_json(orient="records"))


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, float):
        return num(v)
    if hasattr(v, "item"):
        return clean(v.item())
    return v


def find(files: dict, name: str):
    hits = [k for k in files if k == name or k.lower().endswith("/" + name)]
    return files[hits[0]] if len(hits) == 1 else None


def csv_of(files: dict, name: str):
    raw = find(files, name)
    if raw is None:
        return None
    try:
        return E.read_csv_bytes(raw, name)
    except Exception:
        return None


def json_of(files: dict, name: str):
    raw = find(files, name)
    try:
        return json.loads(raw.decode("utf-8")) if raw else None
    except Exception:
        return None


def get_session(sid: str) -> dict:
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "This session has expired (the server was restarted). Please open the bundle again.")
    SESSIONS.move_to_end(sid)
    return s


def split_frame(files: dict, word: str):
    """Find the train or test CSV in the bundle (processed copy preferred) -> (DataFrame, text_col, label_col, file name)."""
    cands = []
    for path in sorted(files):
        low = "/" + path.lower()
        base = low.rsplit("/", 1)[-1]
        if low.endswith(".csv") and word in base and ("/data/processed/" in low or "/data/raw/" in low):
            cands.append((0 if "/data/processed/" in low else 1, path))
    for _, path in sorted(cands):
        try:
            df = E.read_csv_bytes(files[path], path)
        except Exception:
            continue
        tcol = (E.detect_text_column(df) or (None,))[0]
        lcol = (E.detect_label_column(df, tcol) or (None,))[0]
        if tcol and lcol:
            return df, tcol, lcol, path.split("/")[-1]
    return None, None, None, None


def join_predictions(files: dict, pred, test_df, tcol):
    """Attach query text (and decision margin when the row order provably matches) to every prediction.

    Returns None unless every prediction row finds its text AND the labels agree, so a bad join can never show wrong text.
    """
    if not isinstance(pred, pd.DataFrame) or test_df is None or "record_id" not in test_df.columns:
        return None
    cols = {c.lower(): c for c in pred.columns}
    rid, ytrue, ypred = cols.get("record_id"), cols.get("true_intent"), cols.get("predicted_intent")
    if not (rid and ytrue and ypred) or pred[rid].duplicated().any():
        return None
    key = test_df.drop_duplicates("record_id").set_index("record_id")[tcol].astype(str)
    j = pd.DataFrame({"id": pred[rid].astype(str), "actual": pred[ytrue].astype(str), "predicted": pred[ypred].astype(str)})
    j["text"] = j["id"].map(key)
    if j["text"].isna().any():
        return None
    j["correct"] = j["actual"] == j["predicted"]
    unc = csv_of(files, "prediction_uncertainty.csv")
    j["margin"] = None
    # The notebook writes decision_margin when a margin classifier wins (SVM) and max_probability when a
    # probabilistic one wins (Logistic Regression, Naive Bayes). Both are stored under "margin"; the kind is
    # remembered so pages never call a probability a margin (or the other way round).
    kind = None
    if unc is not None and "correct" in unc.columns and len(unc) == len(j):
        same = (unc["correct"].astype(str).str.lower().isin(["true", "1"]).to_numpy() == j["correct"].to_numpy()).all()
        col = next((c_ for c_, k_ in (("decision_margin", "margin"), ("max_probability", "probability")) if c_ in unc.columns), None)
        if same and col:
            vals = pd.to_numeric(unc[col], errors="coerce").to_numpy()
            if pd.notna(vals).any():
                j["margin"] = vals
                kind = "margin" if col == "decision_margin" else "probability"
    j.attrs["score_kind"] = kind
    return j


# ----------------------------------------------------------------------------- building a session
def build_session(data: bytes, source: str) -> dict:
    tables, figures, files, warnings = E.parse_bundle(data)
    manifest = tables.get("manifest")
    if not isinstance(manifest, dict) or not manifest:
        raise E.BundleError("This ZIP has no run_manifest.json, so it is not a Travel NLP Lab run bundle. "
                            "Use the travel_nlp_run_bundle.zip made by the notebook, not a CSV or another ZIP.")
    checks = E.verify_manifest(manifest, files, tables["_bundle_member_names"])
    bad = [c for c in checks if c["status"] in BAD]
    counts = Counter(c["status"] for c in checks)
    integrity = {
        "status": "verified" if checks and not bad else "failed",
        "legacy": any(c["status"] == "legacy_contract" for c in checks),
        "counts": dict(counts),
        "problems": [{"file": c["file"], "status": c["status"]} for c in bad[:25]],
    }
    pred = tables.get("predictions")
    rc = E.recompute_prediction_metrics(pred) if isinstance(pred, pd.DataFrame) else None
    ci = E.bootstrap_metric_ci(pred) if rc else None
    if rc is not None:
        warnings = list(warnings)
        n_test, min_sup = int(len(pred)), int(rc["per_intent"]["support"].min())
        if n_test < 200:
            warnings.append(f"Small test set: only {n_test} test queries. A score such as 100% accuracy is weak evidence here, "
                            f"and a different split could move it a lot.")
        if min_sup < 5:
            few = int((rc["per_intent"]["support"] < 5).sum())
            warnings.append(f"{few} intent(s) have fewer than 5 test queries (as few as {min_sup}), so their per-intent scores are unreliable.")
    final = tables.get("final") if isinstance(tables.get("final"), dict) else {}
    ra, rf = num(final.get("accuracy")), num(final.get("macro_f1"))
    if rc is None:
        mstatus = "reported"
    elif (ra is None or abs(rc["accuracy"] - ra) <= 6e-4) and (rf is None or abs(rc["macro_f1"] - rf) <= 6e-4):
        mstatus = "recomputed"
    else:
        mstatus = "mismatch"
    nw, nw_err = None, None
    raw_nw = find(files, "next_word_model.json")
    if raw_nw:
        try:
            nw = E.parse_next_word_model(raw_nw)
        except E.BundleError as exc:
            nw_err = str(exc)
    else:
        nw_err = "This bundle has no autocomplete model. Re-run the notebook (section 13b) to create one."
    clf, clf_err = None, None
    raw_clf = find(files, "classifier_model.json")
    if raw_clf:
        try:
            clf = E.parse_classifier_model(raw_clf)
        except E.BundleError as exc:
            clf_err = str(exc)
    test_df, t_text, t_label, t_name = split_frame(files, "test")
    joined = join_predictions(files, pred, test_df, t_text)
    label = str(manifest.get("dataset_name", "")).lower()
    sid = uuid.uuid4().hex[:12]
    s = dict(id=sid, source=source, tables=tables, figures=figures, files=files, manifest=manifest, checks=checks,
             integrity=integrity, rc=rc, ci=ci, final=final, mstatus=mstatus, nw=nw, nw_err=nw_err, clf=clf, clf_err=clf_err,
             warnings=warnings, demo=any(w in label for w in ("demo", "synthetic", "sample")),
             joined=joined, score_kind=(joined.attrs.get("score_kind") if joined is not None else None),
             test=(test_df, t_text, t_label, t_name), ac_check=None)
    SESSIONS[sid] = s
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return s


def open_bytes(data: bytes, source: str):
    if len(data) > E.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than the 100 MB limit.")
    try:
        s = build_session(data, source)
    except E.BundleError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        raise HTTPException(400, "Could not read this file as a ZIP bundle.")
    return {"id": s["id"]}


# ----------------------------------------------------------------------------- loading
@app.get("/api/health")
def health():
    return {"ok": True, "contract": E.CONTRACT_VERSION, "pipeline": E.PIPELINE_VERSION, "sessions": len(SESSIONS)}


@app.get("/api/bundles")
def bundles():
    out = []
    for kind, folder in FOLDERS.items():
        for p in sorted(folder.glob("*.zip"), key=lambda q: q.stat().st_mtime, reverse=True) if folder.is_dir() else []:
            out.append({"key": f"{kind}:{p.name}", "name": p.name, "kind": kind, "size": p.stat().st_size,
                        "modified": int(p.stat().st_mtime)})
    return out


class LoadReq(BaseModel):
    key: str


@app.post("/api/load")
def load(req: LoadReq):
    kind, _, name = req.key.partition(":")
    folder = FOLDERS.get(kind)
    if not folder or not re.fullmatch(r"[\w .()-]+\.zip", name):
        raise HTTPException(400, "Unknown bundle.")
    path = folder / name
    if not path.is_file():
        raise HTTPException(404, "That bundle is no longer on disk.")
    return open_bytes(path.read_bytes(), name)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Bundle rejected: please choose the ZIP bundle, not a CSV or another file type.")
    # Read at most limit + 1 bytes so an oversized upload is rejected without loading its full body into RAM.
    limit = E.MAX_UPLOAD_BYTES
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, "File is larger than the 100 MB limit.")
    return open_bytes(data, file.filename)


# ----------------------------------------------------------------------------- pages
@app.get("/api/s/{sid}/overview")
def overview(sid: str):
    s = get_session(sid)
    m, rc, f, files = s["manifest"], s["rc"], s["final"], s["files"]
    summary = json_of(files, "experiment_summary.json") or {}
    gate = json_of(files, "quality_gate.json")
    n_ok = s["integrity"]["counts"].get("verified", 0)
    reconcile = [
        {"check": "Manifest hashes", "ok": s["integrity"]["status"] == "verified", "detail": f"{n_ok} files verified"},
        {"check": "Prediction metrics", "ok": s["mstatus"] == "recomputed",
         "detail": {"recomputed": "Recomputed from predictions and equal to reported",
                    "mismatch": "Recomputed values differ from reported",
                    "reported": "No predictions to recompute from"}[s["mstatus"]]},
        {"check": "Prediction row count", "ok": bool(rc and summary.get("test_rows") in (None, rc["rows"])),
         "detail": f"{rc['rows']:,} rows" if rc else "n/a"},
        {"check": "Reported final model", "ok": bool(f.get("final_model") or m.get("final_model")),
         "detail": str(f.get("final_model") or m.get("final_model") or "—")},
    ]
    weak = []
    if rc is not None:
        pi = rc["per_intent"]
        weak = rows(pi[pi["support"] >= 5].sort_values("f1").head(5))
    return clean({
        "id": sid, "source": s["source"], "demo": s["demo"], "warnings": s["warnings"],
        "integrity": s["integrity"], "metric_status": s["mstatus"], "reconcile": reconcile,
        "run": {"run_id": m.get("run_id"), "created": m.get("created_at_utc"), "seed": m.get("random_seed", m.get("seed")),
                "pipeline": m.get("pipeline_version"), "contract": m.get("contract_version"),
                "final_model": f.get("final_model") or m.get("final_model"),
                "selection": m.get("model_selection"), "fingerprint": m.get("dataset_fingerprint"),
                "note": m.get("verification_note")},
        "metrics": {"reported": {k: f.get(k) for k in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")},
                    "recomputed": {"accuracy": rc["accuracy"], "macro_f1": rc["macro_f1"], "rows": rc["rows"], "labels": rc["labels"]} if rc else None,
                    "ci": s["ci"]},
        "summary": summary, "quality_gate": gate, "weakest": weak,
        "has_autocomplete": s["nw"] is not None,
        "has_live_classifier": s["clf"] is not None,
        "has_predictions": s["joined"] is not None,
        "errors_exact": int((~s["joined"]["correct"]).sum()) if s["joined"] is not None else None,
        "version": APP_VERSION, "versions": _versions(s), "capabilities": _capabilities(s),
        "dataset_name": m.get("dataset_name"),
    })


@app.get("/api/s/{sid}/data")
def data_health(sid: str):
    s = get_session(sid)
    files, out = s["files"], []
    for path in sorted(files):
        low = "/" + path.lower()
        kind = "raw" if "/data/raw/" in low else "processed" if "/data/processed/" in low else None
        if not kind or not low.endswith(".csv"):
            continue
        try:
            df = E.read_csv_bytes(files[path], path)
        except Exception:
            continue
        item = {"name": path.split("/")[-1], "kind": kind, "rows": len(df), "columns": list(map(str, df.columns)),
                "missing": int(df.isna().sum().sum())}
        tcol = (E.detect_text_column(df) or (None,))[0]
        lcol = (E.detect_label_column(df, tcol) or (None,))[0]
        item["text_col"], item["label_col"] = tcol, lcol
        if tcol and kind == "raw":
            txt = df[tcol].astype(str)
            item["duplicates"] = int(txt.str.lower().str.strip().duplicated().sum())
            words = txt.str.split().str.len().clip(upper=30).value_counts().sort_index()
            item["length_hist"] = [{"words": int(k), "rows": int(v)} for k, v in words.items()]
        if lcol and kind == "raw":
            vc = df[lcol].astype(str).value_counts()
            item["intents"] = int(len(vc))
            item["distribution"] = [{"intent": k, "rows": int(v)} for k, v in vc.head(15).items()]
        out.append(item)
    leak = csv_of(files, "leakage_audit.csv")
    return clean({"datasets": out, "profile": json_of(files, "dataset_profile.json"), "leakage": rows(leak),
                  "quality_gate": json_of(files, "quality_gate.json"), "summary": json_of(files, "experiment_summary.json")})


@app.get("/api/s/{sid}/models")
def models(sid: str):
    s = get_session(sid)
    rc, files, t = s["rc"], s["files"], s["tables"]
    comp = csv_of(files, "model_comparison_with_baseline.csv")
    if comp is None:
        comp = t.get("model_comparison")
    out = {"comparison": rows(comp), "final": s["final"], "per_intent": [], "confusion": None, "pairs": [], "margin": None,
           "baseline_macro_f1": (json_of(files, "experiment_summary.json") or {}).get("majority_baseline_macro_f1")}
    if rc is not None:
        out["per_intent"] = rows(rc["per_intent"].sort_values("support", ascending=False))
        cm = rc["confusion"]
        out["confusion"] = {"actual": list(map(str, cm.index)), "predicted": list(map(str, cm.columns)), "matrix": cm.values.tolist()}
        pairs = [{"actual": str(a), "predicted": str(p), "errors": int(cm.loc[a, p])}
                 for a in cm.index for p in cm.columns if a != p and cm.loc[a, p] > 0]
        out["pairs"] = sorted(pairs, key=lambda x: -x["errors"])[:15]
    unc = csv_of(files, "prediction_uncertainty.csv")
    if unc is not None and {"decision_margin", "correct"} <= set(unc.columns):
        mg = pd.to_numeric(unc["decision_margin"], errors="coerce")
        ok = unc["correct"].astype(str).str.lower().isin(["true", "1"])
        if mg.notna().any():
            lo, hi = float(mg.min()), float(mg.max())
            edges = [lo + (hi - lo) * i / 12 for i in range(13)]
            hist = lambda m: [int(((m >= edges[i]) & (m <= edges[i + 1] if i == 11 else m < edges[i + 1])).sum()) for i in range(12)]
            out["margin"] = {"edges": edges, "correct": hist(mg[ok]), "wrong": hist(mg[~ok])}
    return clean(out)


@app.get("/api/s/{sid}/errors")
def errors(sid: str, q: str = "", actual: str = "", limit: int = Query(100, ge=1, le=500)):
    s = get_session(sid)
    df = filtered_errors(s, q, actual)
    t, j = s["tables"], s["joined"]
    src = t.get("errors_final") if isinstance(t.get("errors_final"), pd.DataFrame) else t.get("errors")
    if j is not None:
        classes = sorted(j.loc[~j["correct"], "actual"].unique())
    else:
        classes = sorted(map(str, src["Actual"].dropna().unique())) if isinstance(src, pd.DataFrame) and "Actual" in src else []
    rc = s["rc"]
    n_err = int((~j["correct"]).sum()) if j is not None else (int(round(rc["rows"] * (1 - rc["accuracy"]))) if rc else None)
    return clean({"total_errors": n_err, "rows_total": rc["rows"] if rc else None,
                  "complete": j is not None,
                  "sample_size": n_err if j is not None else (len(src) if isinstance(src, pd.DataFrame) else 0),
                  "matched": len(df), "rows": rows(df, limit), "classes": classes})


def filtered_errors(s: dict, q: str, actual: str):
    j = s["joined"]
    if j is not None:  # every wrong prediction, not just the notebook's saved sample
        df = j.loc[~j["correct"], ["text", "actual", "predicted"]].rename(columns={"text": "Query", "actual": "Actual", "predicted": "Predicted"})
        if q:
            df = df[df["Query"].str.contains(q, case=False, regex=False)]
        if actual:
            df = df[df["Actual"] == actual]
        return df
    t = s["tables"]
    df = t.get("errors_final") if isinstance(t.get("errors_final"), pd.DataFrame) else t.get("errors")
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if q and "Query" in df:
        df = df[df["Query"].astype(str).str.contains(q, case=False, regex=False)]
    if actual and "Actual" in df:
        df = df[df["Actual"].astype(str) == actual]
    return df


@app.get("/api/s/{sid}/errors.csv")
def errors_csv(sid: str, q: str = "", actual: str = ""):
    df = filtered_errors(get_session(sid), q, actual)
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="errors_filtered.csv"'})


def filtered_predictions(s: dict, q: str, actual: str, predicted: str, status: str):
    j = s["joined"]
    if j is None:
        raise HTTPException(409, "This bundle has no row-level predictions to browse.")
    df = j
    if q:
        df = df[df["text"].str.contains(q, case=False, regex=False)]
    if actual:
        df = df[df["actual"] == actual]
    if predicted:
        df = df[df["predicted"] == predicted]
    if status == "wrong":
        df = df[~df["correct"]]
    elif status == "right":
        df = df[df["correct"]]
    return df


@app.get("/api/s/{sid}/predictions")
def predictions(sid: str, q: str = "", actual: str = "", predicted: str = "", status: str = "all",
                limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    s = get_session(sid)
    df = filtered_predictions(s, q, actual, predicted, status)
    j = s["joined"]
    page = df.iloc[offset: offset + limit]
    return clean({"matched": len(df), "total": len(j), "wrong_total": int((~j["correct"]).sum()),
                  "offset": offset, "rows": rows(page), "classes": sorted(set(j["actual"]) | set(j["predicted"])),
                  "has_margin": bool(j["margin"].notna().any()), "score_kind": s.get("score_kind")})


@app.get("/api/s/{sid}/predictions.csv")
def predictions_csv(sid: str, q: str = "", actual: str = "", predicted: str = "", status: str = "all"):
    df = filtered_predictions(get_session(sid), q, actual, predicted, status)
    return Response(df.drop(columns=["margin"], errors="ignore").to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="predictions_filtered.csv"'})


@app.get("/api/s/{sid}/intents")
def intents(sid: str):
    """One card per intent: scores, wording the model leans on, what it is confused with, and real example queries."""
    s = get_session(sid)
    rc, files, j = s["rc"], s["files"], s["joined"]
    if rc is None:
        raise HTTPException(409, "This bundle has no predictions, so per-intent scores cannot be computed.")
    per = {r["intent"]: r for r in rows(rc["per_intent"])}
    cm = rc["confusion"]
    ngr = csv_of(files, "top_ngrams_by_intent.csv")
    terms = {}
    if isinstance(ngr, pd.DataFrame) and {"Intent", "Term", "Count"} <= set(ngr.columns):
        for name, g in ngr.groupby("Intent"):
            terms[str(name)] = [{"term": str(r.Term), "count": int(r.Count)} for r in g.sort_values("Count", ascending=False).head(8).itertuples()]
    feats = s["tables"].get("features") if isinstance(s["tables"].get("features"), dict) else {}
    tr_df, tr_text, tr_label, _ = split_frame(files, "train")
    train_rows = tr_df[tr_label].astype(str).value_counts().to_dict() if tr_df is not None else {}
    out = []
    for name in sorted(set(per) | set(train_rows)):
        item = {"intent": name, "train_rows": int(train_rows.get(name, 0)), "in_test": name in per}
        item.update({k: per[name].get(k) for k in ("precision", "recall", "f1", "support")} if name in per else {})
        item["terms"] = terms.get(name, [])
        item["features"] = list(feats.get(name, []))[:8]
        if name in cm.index:
            row = cm.loc[name]
            item["confused_with"] = [{"intent": str(c), "count": int(v)} for c, v in row.sort_values(ascending=False).items() if c != name and v > 0][:4]
        else:
            item["confused_with"] = []
        if name in cm.columns:
            col = cm[name]
            item["confused_as"] = [{"intent": str(r), "count": int(v)} for r, v in col.sort_values(ascending=False).items() if r != name and v > 0][:4]
        else:
            item["confused_as"] = []
        if j is not None:
            sub = j[j["actual"] == name].sort_values("correct")
            item["examples"] = [{"text": r.text, "predicted": r.predicted, "correct": bool(r.correct)} for r in sub.head(6).itertuples()]
        else:
            item["examples"] = []
        out.append(item)
    return clean({"intents": out, "has_examples": j is not None})


def headline(s: dict) -> dict:
    rc, f = s["rc"], s["final"]
    return {"id": s["id"], "source": s["source"], "model": f.get("final_model") or s["manifest"].get("final_model"),
            "accuracy": rc["accuracy"] if rc else num(f.get("accuracy")), "macro_f1": rc["macro_f1"] if rc else num(f.get("macro_f1")),
            "rows": rc["rows"] if rc else None, "labels": rc["labels"] if rc else None,
            "verified": s["integrity"]["status"] == "verified", "metric_status": s["mstatus"],
            "ci": s["ci"], "created": s["manifest"].get("created_at_utc"),
            "autocomplete_top3": ((s["nw"] or {}).get("evaluation") or {}).get("top3_accuracy")}


@app.get("/api/compare")
def compare(a: str, b: str):
    sa, sb = get_session(a), get_session(b)
    ha, hb = headline(sa), headline(sb)
    deltas, only_a, only_b = [], [], []
    if sa["rc"] is not None and sb["rc"] is not None:
        pa = sa["rc"]["per_intent"].set_index("intent")
        pb = sb["rc"]["per_intent"].set_index("intent")
        for name in sorted(set(pa.index) | set(pb.index)):
            if name in pa.index and name in pb.index:
                deltas.append({"intent": name, "a": num(pa.loc[name, "f1"]), "b": num(pb.loc[name, "f1"]),
                               "delta": num(pb.loc[name, "f1"] - pa.loc[name, "f1"]),
                               "support_a": int(pa.loc[name, "support"]), "support_b": int(pb.loc[name, "support"])})
            elif name in pa.index:
                only_a.append(name)
            else:
                only_b.append(name)
    deltas.sort(key=lambda d: -abs(d["delta"] or 0))
    same_test = bool(sa["rc"] and sb["rc"] and sa["rc"]["rows"] == sb["rc"]["rows"]
                     and sa["manifest"].get("dataset_fingerprint") == sb["manifest"].get("dataset_fingerprint"))
    big = [d for d in deltas if d["delta"] is not None and min(d["support_a"], d["support_b"]) >= 5]
    regress = sorted([d for d in big if d["delta"] <= -0.05], key=lambda d: d["delta"])
    improve = sorted([d for d in big if d["delta"] >= 0.05], key=lambda d: -d["delta"])
    dm = (hb.get("macro_f1") - ha.get("macro_f1")) if hb.get("macro_f1") is not None and ha.get("macro_f1") is not None else None
    verdict = None if dm is None else "regression" if dm <= -0.005 else "improvement" if dm >= 0.005 else "no_clear_change"
    return clean({"a": ha, "b": hb, "deltas": deltas[:40], "only_a": only_a, "only_b": only_b, "same_data": same_test,
                  "verdict": verdict, "regressions": regress[:8], "improvements": improve[:8],
                  "warning": None if same_test else "These runs used different data, so differences may come from the data and not the model."})


# ----------------------------------------------------------------------------- Studio 3 analytics
def _need_pred(s):
    if s["joined"] is None:
        raise HTTPException(409, "NO_PREDICTIONS: this run has no row-level predictions.")
    return s["joined"]


def _ner_frame(s, word: str):
    """Find the annotated (has a `ner` column) raw train/test CSV inside THIS bundle -> (DataFrame, file name) or (None, None).

    The Studio never looks at its own data folder for run analysis: whatever the loaded bundle contains is the source of truth.
    """
    cache = s.setdefault("_ner", {})
    if word in cache:
        return cache[word]
    found = (None, None)
    for path in sorted(s["files"]):
        low = "/" + path.lower()
        base = low.rsplit("/", 1)[-1]
        if not (low.endswith(".csv") and word in base and "/data/raw/" in low):
            continue
        try:
            df = E.read_csv_bytes(s["files"][path], path)
        except Exception:
            continue
        if AN._pick(df, AN.TEXT_COLS) and AN._pick(df, ("ner",)):
            found = (df, path.split("/")[-1])
            break
    cache[word] = found
    return found


def _capabilities(s) -> dict:
    """One honest capability map. Every flag is derived from what the loaded bundle actually contains."""
    j = s["joined"]
    margin = bool(j is not None and j["margin"].notna().any())
    ner_df, _ = _ner_frame(s, "train")
    dq = find(s["files"], "data_quality.json") is not None
    return {
        "benchmark": j is not None, "predictions": j is not None, "error_analysis": j is not None,
        "confidence_scores": margin, "autocomplete": s["nw"] is not None, "live_inference": s["clf"] is not None,
        "robustness": s["clf"] is not None, "ner_annotations": ner_df is not None, "leakage_check": dq,
        "figures": bool(s["figures"]), "run_comparison": True,
        # legacy names kept so older clients keep working
        "confidence": margin, "live_classifier": s["clf"] is not None, "entities": ner_df is not None,
    }


def _versions(s) -> dict:
    """Different things, deliberately kept apart: the Studio app, the notebook pipeline, the API contract and the bundle schema."""
    m = s["manifest"]
    return {"studio": APP_VERSION, "pipeline": m.get("pipeline_version"), "contract": m.get("contract_version"),
            "bundle_schema": m.get("schema_version")}


def _cached(s, key, fn):
    c = s.setdefault("_an", {})
    if key not in c:
        c[key] = clean(fn())
    return c[key]


@app.get("/api/domains")
def domains():
    """Travel-domain coverage read from data/raw (supplementary datasets are kept separate from the ATIS benchmark)."""
    return {"domains": clean(AN.domains(ROOT / "data" / "raw"))}


@app.get("/api/entities")
def entities():
    e = AN.entities(ROOT / "data" / "raw")
    if e is None:
        raise HTTPException(409, "NO_NER: no annotated dataset found in data/raw.")
    return clean(e)


@app.get("/api/s/{sid}/entities")
def entities_for_run(sid: str):
    """Entity annotations read from the loaded bundle's own raw data (never from the Studio folder)."""
    s = get_session(sid)

    def build():
        df, name = _ner_frame(s, "train")
        if df is None:
            return None
        out = AN.entities_from_df(df, f"{name} (annotation, from this bundle)")
        if out is None:
            return None
        tdf, tname = _ner_frame(s, "test")
        out["errors_by_entity_count"] = AN.errors_by_entity_count(s["joined"], tdf, tname) if (tdf is not None and s["joined"] is not None) else None
        return out
    r = _cached(s, "entities", build)
    if r is None:
        raise HTTPException(409, "NO_NER: this bundle has no annotated (ner column) dataset, so entity analysis is unavailable.")
    return r


@app.get("/api/s/{sid}/summary")
def summary(sid: str):
    s = get_session(sid)
    j = s["joined"]
    return clean({
        "run_id": s["manifest"].get("run_id"), "verified": s["integrity"]["status"] == "verified",
        "metric_status": s["mstatus"], "version": APP_VERSION,
        "capabilities": _capabilities(s), "versions": _versions(s),
        "provenance": {"predictions": "notebook (stored benchmark, not live inference)", "live_classifier": "bundled model, computed here from JSON weights (live inference)", "entities": "dataset annotation, not model prediction",
                       "confidence": ("classifier probability, not calibrated" if s.get("score_kind") == "probability" else "classifier decision margin, not a probability")},
        "rows": int(len(j)) if j is not None else None,
        "errors": int((~j["correct"]).sum()) if j is not None else None,
        "intents": int(j["actual"].nunique()) if j is not None else None,
    })


# ----------------------------------------------------------------------------- Query Lab (live inference)
class ClassifyReq(BaseModel):
    text: str = ""


def _need_clf(s):
    if s["clf"] is None:
        raise HTTPException(409, "NO_MODEL: " + (s["clf_err"] or "this bundle has no classifier_model.json. "
                                                  "Re-run the notebook (section 13c) to include the trained model."))
    return s["clf"]


def _margin_bands(s):
    """Low/high margin cut-offs taken from this run's own stored test margins (20th percentile = low)."""
    j = s["joined"]
    if j is None or not j["margin"].notna().any() or s.get("score_kind") != "margin":
        return None      # stored probabilities (0 to 1) are not on the same scale as a live decision margin
    m = j["margin"].dropna()
    return {"low_below": float(m.quantile(0.2)), "high_from": float(m.quantile(0.6)), "basis": f"{len(m)} stored test margins"}


def _band(margin, bands):
    if bands is None:
        return None
    return "low" if margin < bands["low_below"] else ("high" if margin >= bands["high_from"] else "medium")


def _stored_match(s, text):
    """If the typed text is one of the test queries, return the notebook's stored benchmark result for it."""
    j = s["joined"]
    if j is None:
        return None
    key = E.clf_preprocess(text)
    hit = j[j["text"].map(E.clf_preprocess) == key]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {"actual": r["actual"], "predicted": r["predicted"], "correct": bool(r["correct"])}


def _classify_one(s, text):
    r = E.clf_classify(_need_clf(s), text)
    r["band"] = _band(r["margin"], _margin_bands(s))
    return r


@app.post("/api/s/{sid}/classify")
def classify(sid: str, req: ClassifyReq):
    s = get_session(sid)
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "INVALID_QUERY: type a travel query first.")
    if len(text) > E.CLF_MAX_QUERY_CHARS:
        raise HTTPException(400, f"INVALID_QUERY: queries are limited to {E.CLF_MAX_QUERY_CHARS} characters.")
    r = _classify_one(s, text)
    return clean({
        "mode": "live_inference", "query": text, **r, "bands": _margin_bands(s),
        "stored_benchmark": _stored_match(s, text),
        "model": {"name": s["clf"].get("model_name"), "kind": s["clf"].get("kind"), "parity": s["clf"].get("parity"),
                  "trained_on": s["clf"].get("trained_on"), "features": len(s["clf"]["vocab"])},
        "notes": [s["clf"].get("score_note") or "Scores are decision values, not probabilities.",
                  "Live inference by the bundled model; this is not a stored notebook result."],
    })


def _typo(text):
    words = text.split()
    cand = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not cand:
        return None
    i = max(cand, key=lambda k: len(words[k]))
    w = words[i]
    words[i] = w[:2] + w[3] + w[2] + w[4:]
    return " ".join(words)


def _variants(text):
    out = [("Original", text, False),
           ("UPPERCASE", text.upper(), True),
           ("Extra spaces", "  ".join(text.split()) + "   ", True),
           ("Added ???", text.rstrip("?.! ") + " ???", False),
           ("Punctuation removed", re.sub(r"[^\w\s]", "", text), False)]
    ty = _typo(text)
    if ty and ty != text:
        out.append(("One typo", ty, False))
    w = text.split()
    if len(w) >= 4:
        out.append(("Last word dropped", " ".join(w[:-1]), False))
    return out


@app.post("/api/s/{sid}/robustness")
def robustness(sid: str, req: ClassifyReq):
    s = get_session(sid)
    text = " ".join((req.text or "").split())
    if not text or len(text) > E.CLF_MAX_QUERY_CHARS:
        raise HTTPException(400, "INVALID_QUERY: type a travel query (up to 500 characters).")
    _need_clf(s)
    rows_, base = [], None
    for name, v, neutral in _variants(text):
        r = _classify_one(s, v)
        base = base or r["intent"]
        rows_.append({"variant": name, "text": v, "intent": r["intent"], "margin": r["margin"], "band": r["band"],
                      "same_as_original": r["intent"] == base, "neutralised_by_preprocessing": neutral})
    changed = [r["variant"] for r in rows_ if not r["same_as_original"]]
    return clean({"mode": "live_inference", "query": text, "original_intent": base, "variants": rows_,
                  "changed": changed, "stable": not changed,
                  "note": "UPPERCASE and extra spaces are removed by the notebook's own preprocessing, so they cannot change the "
                          "prediction by design. Punctuation, typos and dropped words are the real tests."})


@app.get("/api/s/{sid}/leakage")
def leakage(sid: str):
    """Pre-training data-quality report from the notebook, plus accuracy on near-duplicate vs other test queries (computed here)."""
    s = get_session(sid)
    raw = find(s["files"], "data_quality.json")
    if not raw:
        raise HTTPException(409, "NO_DATA_QUALITY: this bundle has no data_quality.json. Re-run the notebook to create one.")
    try:
        dq = json.loads(raw.decode("utf-8"))
        nd = dq.get("near_duplicates") if isinstance(dq, dict) else None
    except Exception:
        raise HTTPException(409, "NO_DATA_QUALITY: data_quality.json could not be read.")
    out = {k: v for k, v in dq.items() if k != "near_duplicates"}
    out["near_duplicates"] = None
    if isinstance(nd, dict):
        out["near_duplicates"] = {k: v for k, v in nd.items() if k != "test_queries"}
        j = s["joined"]
        if j is not None and isinstance(nd.get("test_queries"), list):
            keys = {E.clf_preprocess(q) for q in nd["test_queries"] if isinstance(q, str)}
            m = j["text"].map(E.clf_preprocess).isin(keys)
            acc = lambda d: float(d["correct"].mean()) if len(d) else None
            out["near_duplicates"]["accuracy"] = {"near_duplicate_rows": int(m.sum()), "accuracy_near_duplicate": acc(j[m]),
                                                  "other_rows": int((~m).sum()), "accuracy_other": acc(j[~m]),
                                                  "accuracy_all": acc(j)}
    return clean(out)


@app.get("/api/s/{sid}/confusion")
def confusion(sid: str, top: int = Query(12, ge=1, le=50)):
    s = get_session(sid)
    j = _need_pred(s)
    return _cached(s, f"conf{top}", lambda: AN.confusion(j, top))


@app.get("/api/s/{sid}/intent-map")
def intent_map(sid: str):
    s = get_session(sid)
    clf = _need_clf(s)

    def build():
        rc, j = s["rc"], s["joined"]
        per = rows(rc["per_intent"]) if rc is not None else None
        pairs = AN.confusion(j, 60)["mutual_pairs"] if j is not None and len(j) else None
        try:
            return AN.intent_map(clf, per, pairs)
        except ValueError as e:
            raise HTTPException(409, "TOO_FEW_INTENTS: " + str(e))
    return _cached(s, "intentmap", build)


@app.get("/api/s/{sid}/confidence")
def confidence(sid: str, focus: str = ""):
    s = get_session(sid)
    j = _need_pred(s)
    focus = focus[:80]
    if focus:       # same rule as the other pages: keep rows where this intent was the true or the chosen one
        j = j[(j["actual"] == focus) | (j["predicted"] == focus)]
    r = _cached(s, "confidence|" + focus, lambda: AN.confidence(j) or {})
    if not r:
        raise HTTPException(409, "NO_MARGIN: this run has no stored decision-margin or probability file.")
    kind = s.get("score_kind") or "margin"
    if kind == "probability":
        r = {**r, "note": "Highest class probability from the classifier. It has not been calibrated, so 0.80 does not mean it is right 80% of the time."}
    return {**r, "kind": kind}


@app.get("/api/s/{sid}/error-lab")
def error_lab(sid: str):
    s = get_session(sid)
    j = _need_pred(s)
    return _cached(s, "errorlab", lambda: AN.error_lab(j))


@app.get("/api/s/{sid}/profile")
def profile(sid: str):
    s = get_session(sid)
    j = _need_pred(s)
    return _cached(s, "profile", lambda: AN.profile(j))


@app.get("/api/s/{sid}/explain")
def explain(sid: str):
    s = get_session(sid)
    feats = s["tables"].get("features")
    return clean({"features": feats if isinstance(feats, dict) else {}, "audit": rows(s["tables"].get("audit")),
                  "note": "Subgroups are query properties (length, style, source), not demographic fairness groups."})


class AutoReq(BaseModel):
    text: str = ""


@app.post("/api/s/{sid}/autocomplete")
def autocomplete(sid: str, req: AutoReq):
    s = get_session(sid)
    if s["nw"] is None:
        raise HTTPException(409, s["nw_err"] or "No autocomplete model in this bundle.")
    return clean(AC.suggest(s["nw"], req.text[:300]))


@app.get("/api/s/{sid}/autocomplete/check")
def autocomplete_check(sid: str):
    """Independent re-check of the autocomplete model on the held-out test queries (cached per session)."""
    s = get_session(sid)
    if s["nw"] is None:
        raise HTTPException(409, s["nw_err"] or "No autocomplete model in this bundle.")
    with _LOCK:
        if s["ac_check"] is None:
            df, tcol, lcol, name = s["test"]
            if df is None:
                s["ac_check"] = {"available": False, "reason": "This bundle has no test CSV to score the autocomplete model on."}
            else:
                texts = df[tcol].astype(str).tolist()
                fresh = AC.evaluate(s["nw"], texts, df[lcol].astype(str).tolist())
                cmp_ = AC.compare_with_stored(s["nw"], fresh)
                ev = s["nw"].get("evaluation") or {}
                s["ac_check"] = {
                    "available": True, "source": name, "fresh": fresh, "compare": cmp_,
                    "stored": {k: ev.get(k) for k in ("fit_verdict", "train_in_sample_top3", "train_minus_test_top3_gap", "cv_top3_mean",
                                                      "cv_top3_sd", "cv_folds", "unseen_target_word_rate", "intent_hint_queries",
                                                      "intent_hint_top1_at_half_query", "intent_hint_top3_at_half_query",
                                                      "intent_hint_majority_baseline_top1")},
                    "settings": s["nw"].get("selected_settings"), "settings_chosen_by": s["nw"].get("settings_chosen_by"),
                    "trained_on": s["nw"].get("trained_on"), "n_train_queries": s["nw"].get("n_train_queries"),
                    "vocab_size": s["nw"].get("vocab_size"), "limitations": s["nw"].get("limitations") or [],
                }
    return clean(s["ac_check"])


@app.get("/api/s/{sid}/autocomplete/samples")
def autocomplete_samples(sid: str, n: int = Query(6, ge=1, le=20), seed: int = 0):
    """Real held-out test queries (never training queries) for the 'try a real query' button."""
    s = get_session(sid)
    df, tcol, lcol, _ = s["test"]
    if df is None:
        return {"samples": []}
    pool = df[[tcol, lcol]].astype(str).drop_duplicates(tcol)
    pool = pool[pool[tcol].str.split().str.len() >= 4]
    picks = pool.sample(n=min(n, len(pool)), random_state=random.Random(seed).randint(0, 2**31 - 1)) if len(pool) else pool
    return {"samples": [{"text": r[tcol], "intent": r[lcol]} for _, r in picks.iterrows()]}


@app.get("/api/s/{sid}/files")
def file_list(sid: str):
    s = get_session(sid)
    status = {}
    for c in s["checks"]:
        for k in s["files"]:
            if k == c["file"] or k.endswith("/" + c["file"]):
                status[k] = c["status"]
    items = [{"path": k, "size": len(v), "status": status.get(k, "n/a")} for k, v in sorted(s["files"].items())]
    figs = [{"path": k, "name": k.split("/")[-1].rsplit(".", 1)[0].replace("_", " ")} for k in sorted(s["figures"])]
    return {"files": items, "figures": figs, "manifest_ok": s["integrity"]["status"] == "verified"}


@app.get("/api/s/{sid}/file")
def file_get(sid: str, path: str, download: bool = False):
    s = get_session(sid)
    blob = s["files"].get(path) or s["figures"].get(path)
    if blob is None:
        raise HTTPException(404, "File not in this bundle.")
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    if ctype.startswith("text/") or ctype == "application/json":
        ctype += "; charset=utf-8"
    disp = "attachment" if download or not ctype.startswith("image/") else "inline"
    return Response(blob, media_type=ctype, headers={"Content-Disposition": f'{disp}; filename="{path.split("/")[-1]}"',
                                                   "X-Content-Type-Options": "nosniff"})


@app.middleware("http")
async def security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp


@app.get("/")
def root():
    """The user interface is the Streamlit app (streamlit_app.py). This FastAPI module is the analysis backend and JSON API."""
    return {"app": "Travel NLP Lab Studio", "ui": "run: streamlit run streamlit_app.py", "api_docs": "/api/docs", "version": APP_VERSION}
