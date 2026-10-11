"""Parity check between two run bundles (for example: one made in Google Colab, one made by tools/run_notebook_local.py).

    python tools/compare_bundles.py colab_bundle.zip local_bundle.zip [--tol 1e-6]

It compares what should NOT depend on where the notebook ran: final metrics, every stored prediction, per-intent scores,
confusion pairs, model comparison, the portable classifier weights, the data-quality counts and the manifest structure.
It does not compare timestamps, run ids, library versions or timings, which legitimately differ. Exit code 0 means identical
within tolerance. Library-version differences are reported as a note, since they are the usual reason for small drifts.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

VOLATILE = {"run_id", "created_at_utc", "python_version", "environment", "files", "artifact_provenance", "verification_note",
            "dataset_files", "timings", "duration_seconds", "elapsed_seconds"}


def read(zpath):
    z = zipfile.ZipFile(zpath)
    out = {}
    for n in z.namelist():
        if not n.endswith("/"):
            out[n.split("travel_nlp_run_bundle/", 1)[-1]] = z.read(n)
    return out


def jload(b):
    return json.loads(b.decode("utf-8"))


def num_close(a, b, tol):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(num_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(num_close(x, y, tol) for x, y in zip(a, b))
    return a == b


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if k not in VOLATILE and "time" not in k.lower() and "second" not in k.lower()}
    if isinstance(d, list):
        return [strip(x) for x in d]
    return d


def csv_of(files, name):
    return pd.read_csv(io.BytesIO(files[name])) if name in files else None


def compare(a_path, b_path, tol=1e-6):
    A, B = read(a_path), read(b_path)
    rows = []   # (check, ok, detail)

    def add(name, ok, detail=""):
        rows.append((name, bool(ok), detail))

    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    add("same set of files (ignoring figures)", not [x for x in only_a + only_b if "/figures/" not in "/" + x],
        f"only in first: {only_a[:4]}; only in second: {only_b[:4]}" if only_a or only_b else "")
    ma, mb = jload(A["run_manifest.json"]), jload(B["run_manifest.json"])
    add("manifest: same keys", set(ma) == set(mb), str(sorted(set(ma) ^ set(mb))))
    for k in ("schema_version", "pipeline_version", "contract_version", "random_seed", "final_model", "dataset_fingerprint"):
        add(f"manifest: {k}", ma.get(k) == mb.get(k), f"{ma.get(k)} vs {mb.get(k)}")
    add("manifest: model_selection", num_close(strip(ma.get("model_selection")), strip(mb.get("model_selection")), tol))
    add("manifest: same capabilities", ma.get("capabilities") == mb.get("capabilities"))
    for name in ("results/final_metrics.json", "results/data_quality.json", "results/dataset_profile.json", "results/quality_gate.json"):
        if name in A and name in B:
            add(name, num_close(strip(jload(A[name])), strip(jload(B[name])), tol))
    for name, keys in (("results/predictions.csv", None), ("results/per_intent_performance.csv", None),
                       ("results/confusion_pairs.csv", None), ("results/model_comparison.csv", None)):
        da, db = csv_of(A, name), csv_of(B, name)
        if da is None or db is None:
            add(name, da is None and db is None, "missing in one bundle")
            continue
        if da.shape != db.shape or list(da.columns) != list(db.columns):
            add(name, False, f"shape/columns differ {da.shape} vs {db.shape}")
            continue
        same = True
        for c in da.columns:
            if pd.api.types.is_numeric_dtype(da[c]) and pd.api.types.is_numeric_dtype(db[c]):
                same &= bool(np.allclose(da[c].to_numpy(float), db[c].to_numpy(float), rtol=tol, atol=tol, equal_nan=True))
            else:
                same &= bool((da[c].astype(str) == db[c].astype(str)).all())
        add(name, same, f"{len(da)} rows")
    if "results/classifier_model.json" in A and "results/classifier_model.json" in B:
        ca, cb = jload(A["results/classifier_model.json"]), jload(B["results/classifier_model.json"])
        ok = ca.get("classes") == cb.get("classes") and ca.get("vocab") == cb.get("vocab")
        ok = ok and np.allclose(np.asarray(ca["weights"], float), np.asarray(cb["weights"], float), rtol=tol, atol=tol)
        ok = ok and np.allclose(np.asarray(ca["bias"], float), np.asarray(cb["bias"], float), rtol=tol, atol=tol)
        add("classifier_model.json: classes, vocabulary, weights, bias", ok)
    note = ""
    if ma.get("environment") != mb.get("environment"):
        ea, eb = ma.get("environment") or {}, mb.get("environment") or {}
        diff = {k: (ea.get(k), eb.get(k)) for k in set(ea) | set(eb) if ea.get(k) != eb.get(k)}
        note = f"Library or platform versions differ: {diff}"
    return rows, note


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first"); ap.add_argument("second"); ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args(argv)
    rows, note = compare(Path(args.first), Path(args.second), args.tol)
    for name, ok, detail in rows:
        print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail and not ok else (f"   ({detail})" if detail else "")))
    bad = [r for r in rows if not r[1]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} checks identical within tolerance {args.tol:g}")
    if note:
        print("NOTE:", note)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
