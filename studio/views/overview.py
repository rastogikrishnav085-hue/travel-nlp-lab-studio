import random

import pandas as pd
import streamlit as st

from app import server as SV
from studio import core, ui


@st.cache_data(show_spinner=False)
def _atis_test():
    p = core.RAW_DIR / "test_atis.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    return df[df["ner"].astype(str).str.split().str.len() == df["text"].astype(str).str.split().str.len()].reset_index(drop=True)


def _set_theme():
    name = st.session_state.get("theme_pick")
    if name in ui.THEMES:
        st.session_state.theme = name
        st.query_params["theme"] = name


def theme_picker():
    """Colour theme for the whole app. Only the accent changes; correct / wrong / warning colours keep their meaning."""
    cur = ui.theme_name()
    st.markdown(f'<div class="nf-k-l" style="margin:18px 0 8px">Colour theme</div>', unsafe_allow_html=True)
    st.segmented_control("Colour theme", list(ui.THEMES), default=cur, key="theme_pick", on_change=_set_theme,
                         label_visibility="collapsed", help="Changes the accent colours on every page and chart. "
                         "Teal = correct, red = wrong and amber = warning stay the same in every theme.")
    t = ui.THEMES[cur]
    st.caption(f"{cur}: {t['blurb']}. Applies to every page and chart, and is kept in the page address.")


def render():
    ov = core.overview()
    m, rep = ov["metrics"]["recomputed"] or {}, ov["metrics"]["reported"] or {}
    left, right = st.columns([1.08, 1], gap="large")
    with left:
        ui.header("Natural language processing · machine learning",
                  "Travel Query Intent <em>Classification</em> Using NLP", hero=True,
                  lead="A traveller types a sentence; the model decides what they want: a flight, a fare, an airline, ground transport and more. "
                       "This studio opens the notebook's finished run, re-checks every number, and shows where the classifier is right and wrong.")
        a, b = st.columns(2)
        if a.button("Explore the results", type="primary", width="stretch", icon=":material/insights:"):
            core.go("models")
        if b.button("Train in Google Colab", width="stretch", icon=":material/cloud_upload:"):
            core.go("data-colab")
    with right:
        df = _atis_test()
        if df is not None and len(df):
            if "p_home_ex" not in st.session_state:
                st.session_state.p_home_ex = 0
            row = df.iloc[st.session_state.p_home_ex % len(df)]
            live = None
            try:
                live = core.call(SV.classify, SV.ClassifyReq(text=str(row["text"])))
            except core.Unavailable:
                pass
            right_ = live and live["intent"] == row["intent"]
            ui.card(f'<div class="nf-k-l">Real ATIS test query · entity annotations from the dataset</div>'
                    f'<div class="nf-q" style="margin:12px 0 14px">{ui.annotated(row["text"], row["ner"])}</div>'
                    + ui.pill(f'dataset label: {row["intent"]}')
                    + (ui.pill(f'live model: {live["intent"]} {"✓" if right_ else "✗"}', "good" if right_ else "bad") if live else ui.pill("no bundled classifier", "warn"))
                    + '<div class="nf-k-s" style="margin-top:8px">The coloured words are ATIS annotations (place, time, other detail), not model output.</div>')
            if st.button("Another example", icon=":material/shuffle:"):
                st.session_state.p_home_ex = random.randrange(len(df)); st.rerun()

    st.write("")
    ci = ov["metrics"].get("ci") or {}
    acc_ci = ci.get("accuracy"); f1_ci = ci.get("macro_f1")
    ui.kpis([
        ("Accuracy", ui.pct(m.get("accuracy")), f"95% interval {ui.pct(acc_ci[0])} – {ui.pct(acc_ci[1])}" if acc_ci else "held-out test set", "good"),
        ("Macro-F1", ui.pct(m.get("macro_f1")), f"95% interval {ui.pct(f1_ci[0])} – {ui.pct(f1_ci[1])}" if f1_ci else "every intent weighted equally", "warn"),
        ("Test queries", ui.n0(m.get("rows")), f'{ui.n0(m.get("labels"))} intents in the test set', "brand"),
        ("Final model", ov["run"]["final_model"] or "—", f'chosen by {ov["run"]["selection"].get("folds", "?")}-fold cross-validation on training data', ""),
    ])
    if ov.get("warnings"):
        ui.note("<b>Read with care:</b> " + " ".join(ui.esc(w) for w in ov["warnings"]), "warn")
    ui.note(f"<b>Why Accuracy and Macro-F1 differ:</b> Accuracy counts every query, so the common <i>flight</i> intent dominates it. "
            f"Macro-F1 averages the intents equally, so rare intents with few test queries pull it down. Both are recomputed here from the "
            f"{ui.n0(m.get('rows'))} stored predictions and {'match' if ov['metric_status'] == 'recomputed' else 'are compared with'} what the notebook reported.")

    ui.section("How it works", "One notebook trains and evaluates; this studio verifies and explains the result.")
    st.markdown('<div class="nf-flow"><span>Raw CSV queries</span><i>→</i><span>Google Colab notebook</span><i>→</i><span>Clean text (lower-case, spaces)</span>'
                '<i>→</i><span>TF-IDF word 1–2 grams</span><i>→</i><span>LogReg · Linear SVM · Naive Bayes</span><i>→</i><span>5-fold CV picks the best</span>'
                '<i>→</i><span>One test on held-out queries</span><i>→</i><span>travel_nlp_run_bundle.zip</span><i>→</i><span>This studio</span></div>',
                unsafe_allow_html=True)

    ui.section("Datasets in this project", "Read from the data/raw folder. The supplementary sets are much smaller than ATIS and are never mixed into its benchmark.")
    raw = core.raw_datasets()
    groups = [("ATIS flights", ["train_atis.csv", "test_atis.csv"], "brand"), ("Hotels", ["hotel_booking_queries.csv"], ""),
              ("Car rental", ["car_rental_queries.csv"], "warn"), ("Rail", ["train_rail_queries.csv"], "good")]
    by = {r["file"]: r for r in raw}
    cols = st.columns(len(groups))
    for col, (name, fs, tone) in zip(cols, groups):
        rs = [by[f] for f in fs if f in by]
        if not rs:
            continue
        rows = sum(r["rows"] for r in rs)
        col.markdown(ui.kpi_html(name, f"{rows:,}", "queries · " + " + ".join(f'{r["file"]} ({r["rows"]:,})' for r in rs), tone), unsafe_allow_html=True)


    with st.expander("Experiment Details & Technical Information", expanded=False):
        run = ov.get("run", {})
        versions = ov.get("versions", {})
        integrity = ov.get("integrity", {})
        st.dataframe(pd.DataFrame([
            ("Experiment name", run.get("name", "—")),
            ("Run ID", run.get("run_id", "—")),
            ("Dataset fingerprint", run.get("fingerprint", "—")),
            ("Final model", run.get("final_model", "—")),
            ("Selection method", (run.get("selection") or {}).get("method", "—")),
            ("Random seed", run.get("seed", "—")),
            ("Studio version", versions.get("studio", "—")),
            ("Pipeline version", versions.get("pipeline", "—")),
            ("Bundle schema", versions.get("bundle_schema", "—")),
            ("Integrity status", integrity.get("status", "—")),
            ("Metric status", ov.get("metric_status", "—")),
        ], columns=["Field", "Value"]).astype(str), hide_index=True, width="stretch")
        st.caption("Technical values are read from the selected experiment bundle. Missing fields are shown as unavailable.")
