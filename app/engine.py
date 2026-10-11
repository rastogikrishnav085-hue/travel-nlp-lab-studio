from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

PIPELINE_VERSION = "8.2.0"
CONTRACT_VERSION = "5.1"
# Older contracts that this dashboard can still read (shown as a "legacy" note, not a failure).
LEGACY_CONTRACT_VERSIONS = ("5.0", "4.0")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_ZIP_FILES = 250
MAX_ZIP_MEMBER_BYTES = 100 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 500 * 1024 * 1024
MAX_ZIP_RATIO = 250
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_CSV_ROWS = 500_000
MAX_COLUMNS = 500
MAX_CELL_CHARS = 1_000_000
MAX_RENDER_ROWS = 100

TEXT_CANDIDATES = [
    "text", "query", "user_query", "utterance", "sentence", "question", "input",
    "prompt", "message", "request", "text_data", "clean_text", "processed_text",
    "normalized_text",
]
LABEL_CANDIDATES = [
    "intent", "label", "intent_label", "category", "class", "target", "y",
    "topic", "intent_name",
]
ID_CANDIDATES = [
    "id", "row_id", "record_id", "query_id", "sample_id", "example_id", "uid", "uuid",
]
TRAVEL_KEYWORDS = {
    "flight", "airfare", "airport", "ticket", "travel", "trip", "hotel", "train", "bus",
    "fare", "book", "booking", "reservation", "itinerary", "baggage", "layover",
    "departure", "arrival", "visa", "passport", "cruise", "rental", "boarding", "gate",
    "seat", "airline", "journey", "destination", "fly", "flying", "cab", "taxi", "ride",
    "hostel", "room", "car", "rent", "vehicle", "pickup", "insurance", "fuel", "agency",
    "drive", "driving", "suv", "sedan", "platform", "station", "rail", "luggage",
    "checkin", "checkout", "stay", "suite", "voyage", "excursion", "vacation", "resort",
    "lodging", "transit", "commute",
}
PROCESSED_HINTS = {
    "clean_text", "processed_text", "word_count", "length_group", "query_style",
    "normalized_text", "tokens", "lemmatized_text", "stemmed_text",
}
WANTED = {
    "model_comparison": "model_comparison.csv",
    "audit": "audit_results.csv",
    "per_intent": "per_intent_performance.csv",
    "errors": "error_analysis_sample.csv",
    "errors_final": "error_analysis_sample_final_model.csv",
    "profile": "dataset_profile.json",
    "final": "final_model_metrics.json",
    "features": "top_features_per_intent.json",
    "predictions": "predictions.csv",
    "manifest": "run_manifest.json",
}


class BundleError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def norm_name(name: str) -> str:
    name = str(name).replace("\\", "/")
    if "\x00" in name:
        raise BundleError("ZIP member name contains a NUL byte")
    while name.startswith("./"):
        name = name[2:]
    return name


def _is_texty(series) -> bool:
    """True for object / string / categorical columns (works on pandas 2 and 3, where text is dtype 'str')."""
    dt = series.dtype
    return bool(pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt) or isinstance(dt, pd.CategoricalDtype))


def _word_forms(token: str):
    """The token plus simple singular/gerund forms, so 'flights', 'fares' and 'booking' match 'flight', 'fare', 'book'."""
    forms = {token}
    if len(token) > 3 and token.endswith("s"):
        forms.add(token[:-1])
    if len(token) > 4 and token.endswith("es"):
        forms.add(token[:-2])
    if len(token) > 5 and token.endswith("ing"):
        forms.add(token[:-3])
    return forms


def normalized_columns(df: pd.DataFrame) -> dict[str, Any]:
    return {str(c).strip().lower(): c for c in df.columns}


def _score_named(cols, candidates, exclude=None):
    exclude = set(exclude or [])
    hits = []
    for i, c in enumerate(candidates):
        if c in cols and cols[c] not in exclude:
            hits.append((cols[c], max(0.70, 1.0 - i * 0.015), f"matched standard name '{c}'"))
    return hits


def text_column_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    cols = normalized_columns(df)
    out = []
    named = {id(x[0]): x for x in _score_named(cols, TEXT_CANDIDATES)}
    for c in df.columns:
        # Numeric/date-like columns are not useful NLP text candidates unless the
        # schema explicitly names them as text.
        if not _is_texty(df[c]):
            continue
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        avg = float(s.str.len().mean())
        uniq = float(s.nunique() / max(len(s), 1))
        spaces = float(s.str.contains(r"\s", regex=True).mean())
        score = min(0.88, 0.25 + min(avg / 80, 0.35) + 0.18 * uniq + 0.10 * spaces)
        why = f"heuristic: avg length {avg:.1f}, unique ratio {uniq:.2f}"
        for nc, nscore, nwhy in named.values():
            if nc == c:
                score, why = nscore, nwhy
                break
        out.append({"column": c, "confidence": round(float(score), 4), "reason": why})
    return sorted(out, key=lambda x: (-x["confidence"], str(x["column"])))


def label_column_candidates(df: pd.DataFrame, text_col=None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    cols = normalized_columns(df)
    out = []
    named = {x[0]: x for x in _score_named(cols, LABEL_CANDIDATES, [text_col])}
    for c in df.columns:
        if c == text_col:
            continue
        # Standard label names are trusted even when a tiny sample has one class.
        if c in named:
            _, score, why = named[c]
            out.append({"column": c, "confidence": round(float(score), 4), "reason": why})
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        n = int(s.nunique())
        ratio = n / max(len(df), 1)
        name = str(c).strip().lower()
        if name in ID_CANDIDATES or n < 2:
            continue
        if n <= min(1000, max(2, int(len(df) * 0.25))):
            score = max(0.25, min(0.75, 0.70 - abs(ratio - 0.05)))
            why = f"heuristic: {n} unique values ({ratio:.1%} of rows)"
            if c in named:
                _, score, why = named[c]
            out.append({"column": c, "confidence": round(float(score), 4), "reason": why})
    return sorted(out, key=lambda x: (-x["confidence"], str(x["column"])))


def detect_text_column(df):
    cands = text_column_candidates(df)
    if not cands:
        return None, 0.0, "no text-like column found" if df is not None and not df.empty else "empty dataset"
    c = cands[0]
    return c["column"], c["confidence"], c["reason"]


def detect_label_column(df, text_col=None):
    cands = label_column_candidates(df, text_col)
    if not cands:
        return None, 0.0, "no label-like column found" if df is not None and not df.empty else "empty dataset"
    c = cands[0]
    return c["column"], c["confidence"], c["reason"]


def detect_id_column(df):
    if df is None or df.empty:
        return None, 0.0, "empty dataset"
    cols = normalized_columns(df)
    for c in ID_CANDIDATES:
        if c in cols:
            col = cols[c]
            unique = bool(df[col].notna().all() and df[col].is_unique)
            if unique:
                return col, 0.95, f"unique standard identifier '{c}'"
            return col, 0.0, f"candidate '{c}' is NOT unique; unsafe as a stable key"
    for c in df.columns:
        lname = str(c).lower()
        if ("id" in lname or lname.endswith("_key")) and df[c].notna().all() and df[c].is_unique:
            return c, 0.82, "unique identifier-like column"
    return None, 0.0, "no stable unique ID detected"


def stable_id_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    if df is None or df.empty:
        return out
    for c in df.columns:
        lname = str(c).strip().lower()
        idish = lname in ID_CANDIDATES or "id" in lname or lname.endswith("_key")
        if not idish:
            continue
        nonnull = int(df[c].notna().sum())
        unique = bool(df[c].notna().all() and df[c].is_unique)
        out.append({
            "column": c,
            "non_null": nonnull,
            "unique": unique,
            "safe": unique and nonnull == len(df),
            "duplicate_count": int(df[c].duplicated(keep=False).sum()),
            "reason": "safe stable key" if unique and nonnull == len(df) else "UNSAFE: missing or duplicate values",
        })
    return out


def ambiguity_report(df: pd.DataFrame) -> dict[str, Any]:
    text = text_column_candidates(df)
    label = label_column_candidates(df, text[0]["column"] if text else None)
    ambiguous_text = len(text) > 1 and text[0]["confidence"] >= 0.70 and (text[1]["confidence"] >= text[0]["confidence"] - 0.08)
    ambiguous_label = len(label) > 1 and label[0]["confidence"] >= 0.70 and (label[1]["confidence"] >= label[0]["confidence"] - 0.08)
    return {
        "text_candidates": text[:8],
        "label_candidates": label[:8],
        "ambiguous_text": ambiguous_text,
        "ambiguous_label": ambiguous_label,
        "needs_confirmation": bool(ambiguous_text or ambiguous_label),
    }


def _check_df_shape(df, name):
    if df.shape[1] > MAX_COLUMNS:
        raise BundleError(f"{name} has too many columns ({df.shape[1]} > {MAX_COLUMNS})")
    for c in df.columns:
        if _is_texty(df[c]):
            max_value = df[c].dropna().astype(str).str.len().max()
            max_len = int(max_value) if pd.notna(max_value) else 0
            if max_len > MAX_CELL_CHARS:
                raise BundleError(f"{name} contains a cell larger than {MAX_CELL_CHARS:,} characters")


def read_csv_bytes(data: bytes, name="CSV"):
    if len(data) > MAX_UPLOAD_BYTES:
        raise BundleError(f"{name} exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB upload limit")
    try:
        df = pd.read_csv(io.BytesIO(data), nrows=MAX_CSV_ROWS + 1, low_memory=True)
    except Exception as e:
        raise BundleError(f"Could not read {name}: {e}") from e
    if len(df) > MAX_CSV_ROWS:
        raise BundleError(f"{name} exceeds the {MAX_CSV_ROWS:,}-row dashboard limit")
    _check_df_shape(df, name)
    return df


def validate_zip(data: bytes):
    if len(data) > MAX_UPLOAD_BYTES:
        raise BundleError(f"ZIP exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB upload limit")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise BundleError("Invalid ZIP file") from e
    infos = [i for i in z.infolist() if not i.is_dir()]
    if len(infos) > MAX_ZIP_FILES:
        raise BundleError(f"ZIP contains too many files ({len(infos)} > {MAX_ZIP_FILES})")
    total = 0
    seen = set()
    for i in infos:
        n = norm_name(i.filename)
        if "\x00" in i.filename or any(ord(ch) < 32 and ch not in "\t" for ch in i.filename):
            raise BundleError(f"Unsafe ZIP member name: {i.filename!r}")
        if i.flag_bits & 0x1:
            raise BundleError(f"Encrypted ZIP members are not supported: {i.filename}")
        if n in seen:
            raise BundleError(f"Duplicate ZIP member name: {i.filename}")
        seen.add(n)
        total += max(0, i.file_size)
        if total > MAX_ZIP_TOTAL_BYTES:
            raise BundleError("ZIP expands beyond the 500 MB safety limit")
        if i.file_size > MAX_ZIP_MEMBER_BYTES:
            raise BundleError(f"ZIP member is too large: {i.filename}")
        if i.compress_size and i.file_size / max(i.compress_size, 1) > MAX_ZIP_RATIO:
            raise BundleError(f"Suspicious compression ratio (possible ZIP bomb): {i.filename}")
        p = Path(n)
        if p.is_absolute() or ".." in p.parts:
            raise BundleError(f"Unsafe ZIP path: {i.filename}")
    return z


def _entry_candidates(z, suffix):
    return [n for n in z.namelist() if not n.endswith("/") and norm_name(n).lower().endswith(suffix.lower())]


def find_entry(z, suffix):
    hits = _entry_candidates(z, suffix)
    if not hits:
        return None
    ranked = sorted(hits, key=lambda x: (0 if "/results/" in "/" + norm_name(x).lower() else 1, len(Path(norm_name(x)).parts), len(x)))
    if len(ranked) > 1:
        a, b = ranked[0], ranked[1]
        ka = (0 if "/results/" in "/" + norm_name(a).lower() else 1, len(Path(norm_name(a)).parts), len(a))
        kb = (0 if "/results/" in "/" + norm_name(b).lower() else 1, len(Path(norm_name(b)).parts), len(b))
        if ka == kb:
            return None
    return ranked[0]



def parse_uploaded_zip(data: bytes):
    """Parse an uploaded ZIP. Only completed Colab run bundles are accepted.

    Accepted:
      * a canonical ``travel_nlp_run_bundle.zip`` (contains ``run_manifest.json``), or
      * a wrapper ZIP (for example a zipped project folder) that contains exactly one
        nested ``travel_nlp_run_bundle.zip``.

    Rejected with a clear message: ZIPs of raw/processed CSV files, or any ZIP without a
    run manifest. The dashboard has no CSV-only workflow; raw CSV files are uploaded and
    processed in Google Colab, which produces the run bundle.

    Returns ``(tables, figures, files, warnings, mode)`` where mode is ``"run_bundle"``
    or ``"nested_run_bundle"``.
    """
    outer = validate_zip(data)
    members = [i for i in outer.infolist() if not i.is_dir()]
    nested = [i for i in members if norm_name(i.filename).lower().endswith("travel_nlp_run_bundle.zip")]
    if len(nested) > 1:
        raise BundleError("This ZIP contains multiple run bundles; upload exactly one travel_nlp_run_bundle.zip.")
    if nested:
        info = nested[0]
        if info.file_size > MAX_UPLOAD_BYTES:
            raise BundleError("Nested run bundle exceeds the 100 MB upload limit")
        tables, figures, files, warnings = parse_bundle(outer.read(info))
        return tables, figures, files, [f"Wrapper ZIP: {w}" for w in warnings], "nested_run_bundle"
    if find_entry(outer, "run_manifest.json") is None:
        raise BundleError(
            "This ZIP is not a Travel NLP run bundle (run_manifest.json was not found). "
            "Upload the travel_nlp_run_bundle.zip created by the Colab notebook. "
            "Raw CSV files are uploaded and processed in Google Colab, not in this dashboard."
        )
    tables, figures, files, warnings = parse_bundle(data)
    return tables, figures, files, warnings, "run_bundle"


def parse_bundle(data: bytes):
    tables, figures, files, warnings = {}, {}, {}, []
    z = validate_zip(data)
    manifest_name = find_entry(z, "run_manifest.json")
    if manifest_name:
        raw = z.read(manifest_name)
        files[norm_name(manifest_name)] = raw
        try:
            tables["manifest"] = json.loads(raw.decode("utf-8"))
        except Exception as e:
            warnings.append(f"Invalid run_manifest.json: {e}")
    manifest = tables.get("manifest") if isinstance(tables.get("manifest"), dict) else {}
    declared = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    zip_names = {norm_name(n): n for n in z.namelist() if not n.endswith("/")}
    if isinstance(declared, dict):
        # A bundle made with shutil.make_archive nests everything in one top-level folder
        # (e.g. travel_nlp_run_bundle/results/x.csv) while the manifest lists results/x.csv.
        # Match by exact path or by "/<declared path>" suffix; keep the real ZIP path as the key.
        for declared_path in declared:
            dn = norm_name(declared_path)
            hits = [orig for key, orig in zip_names.items() if key == dn or key.endswith("/" + dn)]
            for actual_name in hits:
                files[norm_name(actual_name)] = z.read(actual_name)
    for key, suffix in WANTED.items():
        if key == "manifest":
            continue
        name = None
        for p in declared:
            if norm_name(p).lower().endswith(suffix.lower()) and norm_name(p) in zip_names:
                name = zip_names[norm_name(p)]
                break
        if name is None:
            name = find_entry(z, suffix)
        if not name:
            continue
        raw = z.read(name)
        files[norm_name(name)] = raw
        try:
            tables[key] = json.loads(raw.decode("utf-8")) if suffix.endswith(".json") else read_csv_bytes(raw, name)
        except Exception as e:
            warnings.append(f"{name}: {e}")
    for info in z.infolist():
        n = norm_name(info.filename)
        low = n.lower()
        if info.is_dir():
            continue
        if low.endswith(".csv") and (("/data/raw/" in "/" + low) or ("/data/processed/" in "/" + low)):
            try:
                files[n] = z.read(info)
            except Exception as e:
                warnings.append(f"Could not read {n}: {e}")
        if low.endswith((".png", ".jpg", ".jpeg", ".webp")):
            try:
                blob = z.read(info)
                if len(blob) > MAX_IMAGE_BYTES:
                    warnings.append(f"Skipped oversized image: {n}")
                    continue
                if Image is not None:
                    with Image.open(io.BytesIO(blob)) as im:
                        if im.width * im.height > MAX_IMAGE_PIXELS:
                            warnings.append(f"Skipped oversized image dimensions: {n}")
                            continue
                figures[n] = blob
            except Exception as e:
                warnings.append(f"Could not read image {n}: {e}")
    tables["_bundle_member_names"] = [norm_name(n) for n in z.namelist() if not n.endswith("/")]
    return tables, figures, files, warnings


def verify_manifest(manifest, files, member_names=None):
    checks = []
    if not isinstance(manifest, dict):
        return checks
    contract = str(manifest.get("contract_version", ""))
    if contract == CONTRACT_VERSION:
        pass
    elif contract in LEGACY_CONTRACT_VERSIONS:
        checks.append({"file": "<manifest>", "status": "legacy_contract", "expected": CONTRACT_VERSION, "actual": contract})
    else:
        checks.append({"file": "<manifest>", "status": "contract_mismatch", "expected": CONTRACT_VERSION, "actual": contract})
    expected = manifest.get("files") or {}
    if not isinstance(expected, dict) or not expected:
        checks.append({"file": "<manifest>", "status": "no_file_hashes", "expected": "non-empty files map", "actual": None})
        return checks
    normalized_expected = {norm_name(p) for p in expected}
    for path, meta in expected.items():
        path = norm_name(path)
        candidates = [k for k in files if k == path or k.endswith("/" + path)]
        if len(candidates) != 1:
            checks.append({"file": path, "status": "missing" if not candidates else "ambiguous", "expected": meta, "actual": None})
            continue
        actual = sha256_bytes(files[candidates[0]])
        exp = meta.get("sha256") if isinstance(meta, dict) else meta
        checks.append({"file": path, "status": "verified" if actual == exp else "mismatch", "expected": exp, "actual": actual})
    if member_names is not None:
        actual_members = {
            norm_name(x) for x in member_names
            if norm_name(x) != "run_manifest.json" and not norm_name(x).endswith("/run_manifest.json")
        }
        extras = sorted(
            m for m in actual_members
            if not any(m == e or m.endswith("/" + e) for e in normalized_expected)
        )
        for extra in extras:
            checks.append({"file": extra, "status": "unmanifested", "expected": "listed in manifest", "actual": "present in ZIP"})
    return checks


def recompute_prediction_metrics(pred):
    cols = normalized_columns(pred)
    true = next((cols[x] for x in ["true_intent", "y_true", "actual", "true_label", "intent"] if x in cols), None)
    predicted = next((cols[x] for x in ["predicted_intent", "y_pred", "prediction", "predicted_label"] if x in cols), None)
    if true is None or predicted is None:
        return None
    p = pred[[true, predicted]].dropna().astype(str)
    if p.empty:
        return None
    acc = float((p[true] == p[predicted]).mean())
    labels = sorted(set(p[true]) | set(p[predicted]))
    f1s = []
    per_class = []
    for lab in labels:
        tp = int(((p[true] == lab) & (p[predicted] == lab)).sum())
        fp = int(((p[true] != lab) & (p[predicted] == lab)).sum())
        fn = int(((p[true] == lab) & (p[predicted] != lab)).sum())
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        f1s.append(f1)
        per_class.append({"intent": lab, "support": tp + fn, "precision": precision, "recall": recall, "f1": f1})
    return {
        "accuracy": acc,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0,
        "rows": len(p),
        "labels": len(labels),
        "true_column": str(true),
        "predicted_column": str(predicted),
        "per_intent": pd.DataFrame(per_class),
        "confusion": pd.crosstab(p[true], p[predicted], dropna=False),
    }


def bootstrap_metric_ci(pred, n_boot: int = 1000, seed: int = 42, level: float = 0.95):
    """Percentile-bootstrap confidence intervals for Accuracy and Macro-F1 from row-level predictions.

    Resamples the TEST rows with replacement (seeded, so results are repeatable). It captures test-set
    sampling noise only: it does not include variation from retraining, different seeds or different splits.
    Returns None when predictions are missing/incompatible or too small to resample meaningfully.
    """
    import numpy as np
    cols = normalized_columns(pred)
    true = next((cols[x] for x in ["true_intent", "y_true", "actual", "true_label", "intent"] if x in cols), None)
    predicted = next((cols[x] for x in ["predicted_intent", "y_pred", "prediction", "predicted_label"] if x in cols), None)
    if true is None or predicted is None:
        return None
    p = pred[[true, predicted]].dropna().astype(str)
    n = len(p)
    if n < 20:
        return None
    labels = sorted(set(p[true]) | set(p[predicted]))
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    t = p[true].map(idx).to_numpy()
    y = p[predicted].map(idx).to_numpy()
    rng = np.random.default_rng(seed)
    accs, f1s = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        s_ = rng.integers(0, n, n)
        tt, yy = t[s_], y[s_]
        tp = np.bincount(tt[tt == yy], minlength=k).astype(float)
        support = np.bincount(tt, minlength=k).astype(float)
        pred_n = np.bincount(yy, minlength=k).astype(float)
        denom = support + pred_n
        f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)   # F1 = 2TP/(2TP+FP+FN)
        accs[b] = (tt == yy).mean()
        f1s[b] = f1.mean()                                                  # same label set as recompute_prediction_metrics
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {
        "level": level, "n_boot": n_boot, "seed": seed, "rows": n,
        "accuracy": (float(np.percentile(accs, lo)), float(np.percentile(accs, hi))),
        "macro_f1": (float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))),
    }


def normalized_text_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def dataset_profile(name, df, text_col=None, label_col=None):
    td = detect_text_column(df)
    text_col = text_col or td[0]
    ld = detect_label_column(df, text_col)
    label_col = label_col or ld[0]
    idc = detect_id_column(df)
    amb = ambiguity_report(df)
    out = {
        "file": name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "text_column": text_col,
        "text_confidence": round(td[1], 2),
        "label_column": label_col,
        "label_confidence": round(ld[1], 2),
        "id_column": idc[0],
        "id_reason": idc[2],
        "id_safe": bool(idc[0] and idc[1] > 0),
        "ambiguous_text": amb["ambiguous_text"],
        "ambiguous_label": amb["ambiguous_label"],
    }
    if text_col in df:
        s = df[text_col].dropna().astype(str)
        ns = normalized_text_series(df[text_col])
        out["unique_texts"] = int(s.nunique())
        out["duplicate_texts"] = int(s.duplicated().sum())
        out["normalized_duplicate_texts"] = int(ns.duplicated().sum())
        out["empty_texts"] = int((s.str.strip() == "").sum())
        out["avg_words"] = round(float(s.str.split().str.len().mean()), 2) if len(s) else 0
        out["median_words"] = round(float(s.str.split().str.len().median()), 2) if len(s) else 0
        out["travel_keyword_rate"] = round(float(ns.str.findall(r"[a-z]+\b").apply(lambda toks: bool({t for tok in toks for t in _word_forms(tok)} & TRAVEL_KEYWORDS)).mean()), 3) if len(s) else 0
        if label_col in df:
            conflict = pd.DataFrame({"text": ns, "label": df[label_col].astype(str)}).groupby("text")["label"].nunique()
            out["conflicting_normalized_texts"] = int((conflict > 1).sum())
    if label_col in df:
        counts = df[label_col].value_counts(dropna=False)
        out["unique_labels"] = int(len(counts))
        out["small_classes_lt5"] = int((counts < 5).sum())
        out["tiny_classes_lt2"] = int((counts < 2).sum())
        out["imbalance_ratio"] = round(float(counts.max() / counts.min()), 2) if len(counts) and counts.min() > 0 else None
        out["label_counts"] = counts.to_dict()
    return out


def guess_kind(name, df):
    n = name.lower()
    cols = {str(c).strip().lower() for c in df.columns}
    return "processed" if cols & PROCESSED_HINTS or re.search(r"(^|[_\-.])(processed|preprocessed|clean|cleaned)([_\-.]|$)", n) else "raw"


def pair_key(name):
    stem = Path(name).stem.lower()
    stem = re.sub(r"(?:^|[_\-.])(raw|processed|preprocessed|clean|cleaned)(?:$|[_\-.])", "_", stem)
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def split_key(name):
    m = re.search(r"(?:^|[_\-.])(train|test|validation|val|dev)(?:$|[_\-.])", Path(name).stem.lower())
    return m.group(1) if m else None


def schema_signature(df):
    return tuple(sorted(str(c).strip().lower() for c in df.columns))


def propose_pairs(raws, procs):
    proposals, used = [], set()
    bykey = defaultdict(list)
    for pn, pdf in procs:
        bykey[pair_key(pn)].append((pn, pdf))
    for rn, rdf in raws:
        exact = [x for x in bykey[pair_key(rn)] if x[0] not in used]
        if len(exact) == 1:
            pn, pdf = exact[0]
            proposals.append((rn, rdf, pn, pdf, "filename", 0.95))
            used.add(pn)
            continue
        rs = split_key(rn)
        split = [x for x in procs if x[0] not in used and rs and split_key(x[0]) == rs]
        if len(split) == 1:
            pn, pdf = split[0]
            proposals.append((rn, rdf, pn, pdf, "split name", 0.75))
            used.add(pn)
    return proposals


def _canon(s):
    return s.where(~s.isna(), "__<NA>__").astype(str)


def shared_safe_stable_keys(raw: pd.DataFrame, proc: pd.DataFrame) -> list[str]:
    """Return common identifier-like columns that are complete and unique in both frames."""
    out = []
    for c in raw.columns:
        if c not in proc.columns:
            continue
        lname = str(c).strip().lower()
        idish = lname in ID_CANDIDATES or "id" in lname or lname.endswith("_key")
        if not idish:
            continue
        if raw[c].notna().all() and proc[c].notna().all() and raw[c].is_unique and proc[c].is_unique:
            out.append(str(c))
    # Prefer canonical names, then deterministic lexical order.
    rank = {name: i for i, name in enumerate(ID_CANDIDATES)}
    return sorted(out, key=lambda c: (rank.get(c.lower(), 999), c.lower()))


def compare_frames(raw, proc, key_col=None, position_fallback=False):
    common = [c for c in raw.columns if c in proc.columns]
    added = [c for c in proc.columns if c not in raw.columns]
    removed = [c for c in raw.columns if c not in proc.columns]

    # Auto mode may use only a genuinely safe shared stable key. It MUST NOT
    # silently fall back to row position because preprocessing can reorder rows.
    if key_col is None:
        safe_keys = shared_safe_stable_keys(raw, proc)
        if safe_keys:
            key_col = safe_keys[0]

    if key_col == "__ROW_POSITION__":
        if len(raw) != len(proc):
            return {
                "mode": "unmatched", "key": None, "added": added, "removed": removed, "changed": {},
                "changed_cells": None, "changed_rows": None, "compared_rows": 0,
                "unmatched_raw": len(raw), "unmatched_processed": len(proc),
            }
        key_col = None
        mode = "position"
    else:
        mode = "unmatched"

    ra, pa = raw.copy(), proc.copy()
    unmatched_raw = unmatched_proc = 0
    if key_col and key_col in raw.columns and key_col in proc.columns:
        if not (raw[key_col].notna().all() and proc[key_col].notna().all() and raw[key_col].is_unique and proc[key_col].is_unique):
            return {
                "mode": "unsafe_key", "key": key_col, "added": added, "removed": removed, "changed": {},
                "changed_cells": None, "changed_rows": None, "compared_rows": 0,
                "unmatched_raw": len(raw), "unmatched_processed": len(proc),
            }
        mode = "key"
        ra = raw.set_index(key_col)
        pa = proc.set_index(key_col)
        idx = ra.index.intersection(pa.index)
        unmatched_raw = len(ra) - len(idx)
        unmatched_proc = len(pa) - len(idx)
        ra, pa = ra.loc[idx], pa.loc[idx]
    elif mode != "position":
        return {
            "mode": "unmatched", "key": None, "added": added, "removed": removed, "changed": {},
            "changed_cells": None, "changed_rows": None, "compared_rows": 0,
            "unmatched_raw": len(raw), "unmatched_processed": len(proc),
        }

    changed = {}
    rowmask = pd.Series(False, index=ra.index)
    for c in common:
        if c == key_col:
            continue
        a, b = _canon(ra[c]), _canon(pa[c])
        mask = a.ne(b)
        changed[str(c)] = int(mask.sum())
        rowmask = rowmask | mask
    return {
        "mode": mode, "key": key_col, "added": added, "removed": removed, "changed": changed,
        "changed_cells": sum(changed.values()), "changed_rows": int(rowmask.sum()), "compared_rows": len(ra),
        "unmatched_raw": unmatched_raw, "unmatched_processed": unmatched_proc,
    }


def csvs_from_bundle(files):
    raws, procs = [], []
    for n, b in files.items():
        if not n.lower().endswith(".csv"):
            continue
        try:
            if "/data/raw/" in "/" + n.lower():
                raws.append((n, read_csv_bytes(b, n)))
            elif "/data/processed/" in "/" + n.lower():
                procs.append((n, read_csv_bytes(b, n)))
        except BundleError:
            pass
    return raws, procs


def make_audit_csv(diff_rows):
    return pd.DataFrame(diff_rows).to_csv(index=False).encode("utf-8") if diff_rows else b""


def prediction_error_frame(pred: pd.DataFrame) -> pd.DataFrame | None:
    cols = normalized_columns(pred)
    t = next((cols[x] for x in ["true_intent", "y_true", "actual", "true_label", "intent"] if x in cols), None)
    p = next((cols[x] for x in ["predicted_intent", "y_pred", "prediction", "predicted_label"] if x in cols), None)
    if t is None or p is None:
        return None
    out = pred.copy()
    out["is_error"] = out[t].astype(str) != out[p].astype(str)
    return out


# ---------------------------------------------------------------------------
# Autocomplete / next-word suggestions (read-only lookup of the n-gram model exported by Colab).
# The dashboard never trains or updates this model; it only reads the exported counts.
# `next_word_suggestions` and `intent_hint` must stay identical to the notebook's versions.
# ---------------------------------------------------------------------------
NEXT_WORD_FORMAT = "travel_nlp_ngram_v2"
NEXT_WORD_MAX_BYTES = 24 * 1024 * 1024
NEXT_WORD_MAX_CONTEXTS = 600_000
_NW_TOKEN = re.compile(r"[a-z0-9_']+")
END_TOKEN = "</s>"


def _nw_nonneg_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**9


def parse_next_word_model(raw: bytes) -> dict:
    """Parse and strictly validate next_word_model.json. Untrusted input: JSON only, bounded size, typed values."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > NEXT_WORD_MAX_BYTES:
        raise BundleError("next_word_model.json is missing or too large")
    try:
        model = json.loads(bytes(raw).decode("utf-8"))
    except Exception:
        raise BundleError("next_word_model.json is not valid JSON")
    if not isinstance(model, dict) or model.get("format") != NEXT_WORD_FORMAT:
        raise BundleError("next_word_model.json has an unsupported format")
    contexts = model.get("contexts")
    if not isinstance(contexts, dict) or not contexts or len(contexts) > NEXT_WORD_MAX_CONTEXTS:
        raise BundleError("next_word_model.json has an invalid context table")
    for key, entry in contexts.items():
        if not (isinstance(key, str) and len(key) <= 400 and isinstance(entry, list) and len(entry) == 2
                and _nw_nonneg_int(entry[0]) and entry[0] > 0 and isinstance(entry[1], dict) and 0 < len(entry[1]) <= 100):
            raise BundleError("next_word_model.json has a malformed context entry")
        for word, count in entry[1].items():
            if not (isinstance(word, str) and len(word) <= 100 and _nw_nonneg_int(count)):
                raise BundleError("next_word_model.json has a malformed count")
    alpha = model.get("backoff_alpha")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise BundleError("next_word_model.json has an invalid backoff value")
    max_ctx = model.get("max_context_tokens")
    if not (isinstance(max_ctx, int) and not isinstance(max_ctx, bool) and 1 <= max_ctx <= 6):
        raise BundleError("next_word_model.json has an invalid context length")
    wc = model.get("word_counts")
    if not isinstance(wc, dict) or any(not isinstance(w, str) or len(w) > 100 or not _nw_nonneg_int(c) for w, c in wc.items()):
        raise BundleError("next_word_model.json has an invalid word_counts table")
    for name in ("intent_prior", "intent_tokens"):
        table = model.get(name)
        if not isinstance(table, dict) or any(not isinstance(k, str) or not _nw_nonneg_int(v) for k, v in table.items()):
            raise BundleError(f"next_word_model.json has an invalid {name}")
    ti = model.get("token_intent")
    if not isinstance(ti, dict) or any(
        not isinstance(w, str) or not isinstance(d, dict) or any(not isinstance(i, str) or not _nw_nonneg_int(c) for i, c in d.items())
        for w, d in ti.items()
    ):
        raise BundleError("next_word_model.json has an invalid token_intent table")
    if not _nw_nonneg_int(model.get("vocab_size")) or not _nw_nonneg_int(model.get("n_train_queries")):
        raise BundleError("next_word_model.json has invalid size fields")
    if not isinstance(model.get("evaluation", {}), dict):
        raise BundleError("next_word_model.json has an invalid evaluation block")
    return model


def next_word_tokens(text: str, limit: int = 40) -> list:
    """Tokenise typed text exactly like the notebook (lower-case words), capped for safety."""
    return _NW_TOKEN.findall(str(text)[:300].lower())[:limit]


def split_typed_text(text: str):
    """Split what the person typed into (finished words, half-typed word).

    'show me flights fro'  -> (['show','me','flights'], 'fro')   the last word is still being typed
    'show me flights '     -> (['show','me','flights'], '')      trailing space: the last word is finished
    """
    text = str(text)[:300]
    toks = next_word_tokens(text)
    if toks and text and not text[-1].isspace():
        return toks[:-1], toks[-1]
    return toks, ""


def next_word_suggestions(model: dict, prefix_tokens, k: int = 5):
    """Stupid-backoff ranking: [(word, score)], best first. END_TOKEN means 'the query ends here'."""
    max_ctx = int(model["max_context_tokens"])
    seq = ["<s>"] + list(prefix_tokens)[-max_ctx:]
    scores = {}
    for level, n in enumerate(range(max_ctx, -1, -1)):
        if n > len(seq):
            continue
        entry = model["contexts"].get(" ".join(seq[-n:]) if n else "")
        if not entry:
            continue
        total, nexts = entry
        for word, count in nexts.items():
            if word != "<s>" and word not in scores:
                scores[word] = (count / total) * (float(model["backoff_alpha"]) ** level)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def next_word_table(model: dict, prefix_tokens, k: int = 5):
    """Suggestions with a 'share' column: each candidate's score as a fraction of the candidates shown."""
    ranked = next_word_suggestions(model, prefix_tokens, k)
    total = sum(s for _, s in ranked) or 1.0
    return [{"word": w, "score": s, "share": s / total} for w, s in ranked]


def complete_partial_word(model: dict, finished_tokens, partial: str, k: int = 5):
    """Finish a half-typed word. Candidates are known words starting with `partial`, ranked by how well they
    fit the finished words before them (backoff score), falling back to overall word frequency."""
    partial = str(partial).lower()
    if not partial:
        return []
    pool = {w: c for w, c in model["word_counts"].items() if w.startswith(partial)}
    if not pool:
        return []
    wide = dict(next_word_suggestions(model, finished_tokens, k=len(model["word_counts"]) + 1))
    total_freq = float(sum(model["word_counts"].values())) or 1.0
    scored = []
    for word, count in pool.items():
        ctx = wide.get(word, 0.0)
        scored.append((word, ctx if ctx > 0 else 1e-6 * count / total_freq, ctx > 0))
    scored.sort(key=lambda r: (-r[1], r[0]))
    top = scored[:k]
    norm = sum(s for _, s, _ in top) or 1.0
    return [{"word": w, "share": s / norm, "fits_context": fit} for w, s, fit in top]


def complete_query(model: dict, prefix_tokens, n_results: int = 3, max_new: int = 12, branch: int = 4):
    """Beam search for likely full queries. Returns [{'text', 'finished', 'probability'}].

    `probability` is the product of the shares of the chosen words: a relative likelihood among the explored
    beams, not a calibrated probability of the whole sentence.
    """
    beams = [(list(prefix_tokens), 1.0, False)]
    for _ in range(max_new):
        grown = []
        for toks, p, done in beams:
            if done:
                grown.append((toks, p, True))
                continue
            for cand in next_word_table(model, toks, branch):
                if cand["word"] == END_TOKEN:
                    grown.append((toks, p * cand["share"], True))
                elif toks and cand["word"] == toks[-1]:
                    continue
                else:
                    grown.append((toks + [cand["word"]], p * cand["share"], False))
        if not grown:
            break
        beams = sorted(grown, key=lambda b: -b[1])[: max(n_results * 3, 6)]
        if all(done for _, _, done in beams):
            break
    seen, out = set(), []
    for toks, p, done in sorted(beams, key=lambda b: -b[1]):
        text = " ".join(toks)
        if text and text not in seen:
            seen.add(text)
            out.append({"text": text, "finished": done, "probability": p})
    return out[:n_results]


def autocomplete(model: dict, typed: str, n_words: int = 5, n_queries: int = 3):
    """One call for the UI: everything to show for whatever the person has typed so far."""
    finished, partial = split_typed_text(typed)
    result = {"finished": finished, "partial": partial, "word_completions": [], "next_words": [],
              "completions": [], "intents": [], "known_words": True}
    if partial:
        result["word_completions"] = complete_partial_word(model, finished, partial, n_words)
        if not result["word_completions"]:
            result["known_words"] = False
        best = result["word_completions"][0]["word"] if result["word_completions"] else partial
        base = finished + [best]
    else:
        base = finished
    result["next_words"] = next_word_table(model, base, n_words)
    result["completions"] = complete_query(model, base, n_results=n_queries)
    result["intents"] = intent_hint(model, base, 3) if base else []
    unknown = [t for t in base if t not in model["word_counts"]]
    if unknown:
        result["known_words"] = False
        result["unknown_words"] = unknown
    return result


def intent_hint(model: dict, prefix_tokens, k: int = 3):
    """Naive-Bayes style intent estimate from word counts exported by Colab: [(intent, probability)]."""
    prior = model["intent_prior"]
    total_q = sum(prior.values())
    if not total_q:
        return []
    vocab = max(int(model["vocab_size"]), 1)
    logp = {}
    for intent, count in prior.items():
        if count <= 0:
            continue
        lp = math.log(count / total_q)
        denom = model["intent_tokens"].get(intent, 0) + vocab
        for tok in prefix_tokens:
            if tok in model["token_intent"]:
                lp += math.log((model["token_intent"][tok].get(intent, 0) + 1) / denom)
        logp[intent] = lp
    if not logp:
        return []
    peak = max(logp.values())
    weights = {i: math.exp(v - peak) for i, v in logp.items()}
    norm = sum(weights.values())
    return sorted(((i, w / norm) for i, w in weights.items()), key=lambda kv: -kv[1])[:k]


# ----------------------------------------------------------------------------- portable classifier (Query Lab)
# The notebook exports the selected model as plain JSON (never a pickle, which could run code on load).
CLF_FORMAT = "travel_nlp_linear_v1"
CLF_MAX_BYTES = 40 * 1024 * 1024
CLF_MAX_FEATURES = 400_000
CLF_MAX_CLASSES = 500
CLF_MAX_QUERY_CHARS = 500
_CLF_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def parse_classifier_model(raw: bytes) -> dict:
    """Strictly validate classifier_model.json. Untrusted input: JSON only, bounded size, typed values."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > CLF_MAX_BYTES:
        raise BundleError("classifier_model.json is missing or too large")
    try:
        m = json.loads(raw.decode("utf-8"))
    except Exception:
        raise BundleError("classifier_model.json is not valid JSON")
    if not isinstance(m, dict) or m.get("format") != CLF_FORMAT:
        raise BundleError("classifier_model.json has an unsupported format")
    classes, vocab, idf, W, b = m.get("classes"), m.get("vocab"), m.get("idf"), m.get("weights"), m.get("bias")
    if not (isinstance(classes, list) and 2 <= len(classes) <= CLF_MAX_CLASSES and all(isinstance(c, str) for c in classes)):
        raise BundleError("classifier_model.json has an invalid class list")
    if not (isinstance(vocab, dict) and 0 < len(vocab) <= CLF_MAX_FEATURES):
        raise BundleError("classifier_model.json has an invalid vocabulary")
    n = len(vocab)
    if not all(isinstance(t, str) and isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n for t, i in vocab.items()) \
            or len(set(vocab.values())) != n:
        raise BundleError("classifier_model.json has an invalid vocabulary index")
    if not (isinstance(idf, list) and len(idf) == n):
        raise BundleError("classifier_model.json has an invalid idf table")
    if not (isinstance(W, list) and len(W) == len(classes) and all(isinstance(r, list) and len(r) == n for r in W)):
        raise BundleError("classifier_model.json has an invalid weight matrix")
    if not (isinstance(b, list) and len(b) == len(classes)):
        raise BundleError("classifier_model.json has an invalid bias vector")
    ng = m.get("ngram_range")
    if not (isinstance(ng, list) and len(ng) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in ng) and 1 <= ng[0] <= ng[1] <= 3):
        raise BundleError("classifier_model.json has an invalid n-gram range")
    try:
        idf_a = np.asarray(idf, dtype=np.float64)
        W_a = np.asarray(W, dtype=np.float64)
        b_a = np.asarray(b, dtype=np.float64)
    except (TypeError, ValueError):
        raise BundleError("classifier_model.json contains non-numeric values")
    if not (np.isfinite(idf_a).all() and np.isfinite(W_a).all() and np.isfinite(b_a).all()):
        raise BundleError("classifier_model.json contains non-finite numbers")
    m["_idf"], m["_W"], m["_b"] = idf_a, W_a, b_a
    for k in ("idf", "weights"):
        m.pop(k, None)
    return m


def clf_preprocess(text: str) -> str:
    return " ".join(str(text).lower().split())


def clf_scores(model: dict, text: str):
    """Decision scores for one query, computed from the JSON model alone. Returns (scores, n_known_features)."""
    toks = _CLF_TOKEN.findall(clf_preprocess(text)[:CLF_MAX_QUERY_CHARS])
    lo, hi = model["ngram_range"]
    terms = []
    for n in range(lo, hi + 1):
        terms += [" ".join(toks[k:k + n]) for k in range(len(toks) - n + 1)]
    counts = Counter(t for t in terms if t in model["vocab"])
    if not counts:
        return model["_b"].copy(), 0
    idx = np.fromiter((model["vocab"][t] for t in counts), dtype=np.int64, count=len(counts))
    c = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    tf = (1.0 + np.log(c)) if model.get("sublinear_tf") else c
    x = tf * model["_idf"][idx]
    nrm = np.linalg.norm(x)
    if nrm > 0:
        x = x / nrm
    return model["_W"][:, idx] @ x + model["_b"], len(counts)


def clf_classify(model: dict, text: str, k: int = 5) -> dict:
    s, known = clf_scores(model, text)
    order = np.argsort(-s)
    top = [{"intent": model["classes"][int(i)], "score": float(s[i])} for i in order[:max(1, k)]]
    return {"intent": top[0]["intent"], "margin": float(s[order[0]] - s[order[1]]), "top": top, "known_features": int(known)}
