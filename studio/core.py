"""Bridge between the Streamlit pages and the verified analysis backend (app/server.py).

The pages never compute metrics themselves. Every number comes from the same backend functions that the
automated tests check, called in-process (no second server, no network).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from fastapi import HTTPException

from app import server as SV

ROOT = SV.ROOT
RAW_DIR = ROOT / "data" / "raw"
NOTEBOOK = ROOT / "notebooks" / "ATIS_Complete_Pipeline.ipynb"
TITLE = "Travel Query Intent Classification Using NLP"
APP_NAME = "Travel NLP Lab Studio"
VERSION = SV.APP_VERSION


PAGES: dict = {}   # url_path -> st.Page, filled in by streamlit_app.py


def go(url_path: str) -> None:
    st.switch_page(PAGES[url_path])


class Unavailable(Exception):
    """The loaded run cannot answer this (for example: no classifier in the bundle)."""


# ----------------------------------------------------------------------------- runs
def bundles() -> list[dict]:
    return SV.bundles()


def _open(data: bytes, name: str) -> str:
    try:
        return SV.open_bytes(data, name)["id"]
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from None


def open_disk(key: str) -> None:
    kind, _, name = key.partition(":")
    path = SV.FOLDERS[kind] / name
    st.session_state.run = {"key": key, "name": name, "kind": kind, "sid": _open(path.read_bytes(), name), "data": None}
    _reset_page_state()


def open_upload(data: bytes, name: str) -> None:
    if not name.lower().endswith(".zip"):
        raise ValueError("Please choose the travel_nlp_run_bundle.zip made by the notebook, not a CSV or another file type.")
    sid = _open(data, name)
    key = f"upload:{name}"
    st.session_state.setdefault("uploads", {})[key] = (name, data)
    st.session_state.run = {"key": key, "name": name, "kind": "upload", "sid": sid, "data": data}
    _reset_page_state()


def open_key(key: str) -> None:
    """Open a run from the run list (a bundle on disk or one uploaded earlier in this session)."""
    if key.startswith("upload:"):
        name, data = st.session_state["uploads"][key]
        open_upload(data, name)
    else:
        open_disk(key)


def _reset_page_state() -> None:
    """A new run must never show the previous run's selections or cached results."""
    for k in list(st.session_state.keys()):
        if k.startswith(("p_", "cache_")):
            del st.session_state[k]


def ensure_run() -> dict | None:
    """Open the shipped example on first visit so every page has something real to show."""
    run = st.session_state.get("run")
    if run:
        return run
    ex = [b for b in bundles() if b["kind"] == "example"]
    if ex:
        try:
            open_disk(ex[0]["key"])
        except ValueError:
            return None
    return st.session_state.get("run")


def _reopen(run: dict) -> None:
    data = run["data"]
    if data is None:
        kind, _, name = run["key"].partition(":")
        data = (SV.FOLDERS[kind] / name).read_bytes()
    run["sid"] = _open(data, run["name"])


def call(fn, *args, **kwargs):
    """Call a backend function for the current run. Re-opens the bundle if its session was evicted."""
    run = ensure_run()
    if not run:
        raise Unavailable("No run is open. Open the example or upload a bundle from the sidebar.")
    for attempt in (0, 1):
        try:
            return fn(run["sid"], *args, **kwargs)
        except HTTPException as exc:
            if exc.status_code == 404 and attempt == 0 and "expired" in str(exc.detail):
                _reopen(run)
                continue
            if exc.status_code in (404, 409):
                raise Unavailable(str(exc.detail)) from None
            raise


def cached(name: str, fn, *args):
    """Per-run memo for heavier backend calls (cleared when another run is opened)."""
    key = "cache_" + name + "|" + "|".join(map(str, args))
    if key not in st.session_state:
        st.session_state[key] = call(fn, *args)
    return st.session_state[key]


def overview() -> dict:
    return cached("overview", SV.overview)


def sid() -> str:
    return st.session_state.run["sid"]


def session() -> dict:
    """The backend's in-memory session for the current run (re-opened if it was evicted)."""
    run = ensure_run()
    if not run:
        raise Unavailable("No run is open.")
    s = SV.SESSIONS.get(run["sid"])
    if s is None:
        _reopen(run)
        s = SV.SESSIONS[run["sid"]]
    return s


def file_bytes(path: str) -> bytes | None:
    s = session()
    return s["files"].get(path) or s["figures"].get(path)


def predictions_frame(q="", actual="", predicted="", status="all") -> pd.DataFrame:
    s = session()
    if s["joined"] is None:
        raise Unavailable("This bundle has no row-level predictions joined to the test queries.")
    return SV.filtered_predictions(s, q, actual, predicted, status)


# ----------------------------------------------------------------------------- raw datasets shipped with the project
ROLE = {"train_atis.csv": "ATIS · training split", "test_atis.csv": "ATIS · test split",
        "hotel_booking_queries.csv": "Hotels · supplementary (unsplit)", "car_rental_queries.csv": "Car rental · supplementary (unsplit)",
        "train_rail_queries.csv": "Rail · supplementary (unsplit)"}


@st.cache_data(show_spinner=False)
def raw_datasets() -> list[dict]:
    from app import engine as E
    out = []
    for p in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(p)
        tcol = (E.detect_text_column(df) or (None,))[0]
        lcol = (E.detect_label_column(df, tcol) or (None,))[0]
        out.append({"file": p.name, "role": ROLE.get(p.name, "dataset"), "rows": len(df), "columns": list(df.columns),
                    "text_col": tcol, "label_col": lcol, "intents": int(df[lcol].nunique()) if lcol else None,
                    "duplicates": int(df.duplicated(subset=[tcol]).sum()) if tcol else None,
                    "has_ner": "ner" in df.columns, "bytes": p.stat().st_size,
                    "top": df[lcol].value_counts().head(5).to_dict() if lcol else {}})
    return out


def raw_bytes(name: str) -> bytes:
    p = RAW_DIR / name
    if p.parent != RAW_DIR or not p.is_file():
        raise FileNotFoundError(name)
    return p.read_bytes()
