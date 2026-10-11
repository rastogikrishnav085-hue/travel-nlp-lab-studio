import json

import pandas as pd
import streamlit as st

from app import server as SV
from studio import core, ui

SECTIONS = [("1", "Environment setup", "installs pandas, numpy, matplotlib, scikit-learn"),
            ("2", "Upload the dataset", "a Choose files button appears: upload your CSV files here"),
            ("3", "Configuration", "finds the text and label columns automatically; stops if unsure"),
            ("4", "Load, clean, combine", "removes exact duplicate queries"),
            ("5", "Data profile + quality gate", "stops before training on empty splits, conflicting labels or train/test overlap"),
            ("6", "Text preprocessing", "lower-case and normalise spaces"),
            ("7", "Model search", "TF-IDF (word 1–2 grams) + Logistic Regression / Linear SVM / Naive Bayes, 5-fold CV Macro-F1"),
            ("9", "Final evaluation", "the chosen model is tested once on the held-out test set"),
            ("10–12", "Errors, subgroups, explanations", "mistakes, short vs long queries, top words per intent"),
            ("13–13c", "Try a query, autocomplete, model export", "exports the classifier as plain JSON for the Query lab"),
            ("14–15", "Figures + final quality gate", "must print PASS"),
            ("16", "Package the bundle", "writes travel_nlp_run_bundle.zip with SHA-256 fingerprints and downloads it")]


def render():
    ui.header("Start", "Data & Google Colab",
              "Training happens in Google Colab with the project notebook. You upload the <b>.ipynb</b> file to Colab yourself, upload the CSV "
              "files inside the notebook, and bring back one ZIP. This studio never trains a model and never accepts raw CSV files.")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        ui.section("Run the notebook in Google Colab")
        st.markdown("""<ol class="nf-steps">
<li><b>Download the notebook</b> with the button on the right (it is also in the project folder: <code>notebooks/ATIS_Complete_Pipeline.ipynb</code>).</li>
<li>Open <b>colab.research.google.com</b>, sign in, then <b>File → Upload notebook</b> and choose the <code>.ipynb</code> file.</li>
<li>Click <b>Runtime → Run all</b>. The first cell installs the libraries (about a minute the first time).</li>
<li>In section 2 a <b>Choose files</b> button appears. Upload the CSV files for this experiment. For the ATIS benchmark upload
<code>train_atis.csv</code> and <code>test_atis.csv</code> (download them from the table below). Only <code>.csv</code> files are accepted.</li>
<li>Let every cell finish. The <b>final quality gate</b> must print <b>PASS</b>; if a check fails, the notebook stops and says why.</li>
<li>The last cell <b>downloads <code>travel_nlp_run_bundle.zip</code></b> to your computer (allow the download if the browser asks).</li>
<li>Open it here: <b>sidebar → Upload travel_nlp_run_bundle.zip</b>, or copy it into the project's <code>output</code> folder so it appears in the run list.</li>
</ol>""", unsafe_allow_html=True)
        ui.note("<b>No Colab?</b> The same notebook runs on this computer: <code>START.bat</code> option <b>3</b> trains on the ATIS files "
                "(about 35 seconds), option <b>2</b> trains on the CSVs you put in <code>my_data</code>. The bundle appears in the run list automatically.")
    with right:
        ui.section("Files you need")
        nb = core.NOTEBOOK.read_bytes()
        cells = json.loads(nb)["cells"]
        st.download_button("Download the notebook (.ipynb)", nb, file_name=core.NOTEBOOK.name, mime="application/x-ipynb+json",
                           type="primary", width="stretch", icon=":material/download:")
        st.caption(f"{core.NOTEBOOK.name} · {len(cells)} cells · {len(nb) // 1024} KB · runs without any API key or paid service")
        st.write("")
        ui.card('<div class="nf-k-l">What the notebook does, section by section</div>' +
                "".join(f'<div style="display:flex;gap:10px;margin-top:9px"><span class="nf-pill brand" style="min-width:62px;text-align:center;align-self:flex-start;flex:none">{n}</span>'
                        f'<div><b style="color:var(--ink)">{ui.esc(t)}</b><div class="nf-k-s">{ui.esc(d)}</div></div></div>' for n, t, d in SECTIONS))

    ui.section("Raw datasets (data/raw)", "Exactly what is in the project folder. Download any file to upload it in Colab.")
    raw = core.raw_datasets()
    table = pd.DataFrame([{"File": r["file"], "Role": r["role"], "Rows": r["rows"], "Intents": r["intents"], "Text column": r["text_col"],
                           "Label column": r["label_col"], "Repeated queries": r["duplicates"], "Entity tags": "yes (BIO)" if r["has_ner"] else "no"}
                          for r in raw])
    st.dataframe(table, hide_index=True, width="stretch")
    cols = st.columns(len(raw))
    for col, r in zip(cols, raw):
        col.download_button(r["file"], core.raw_bytes(r["file"]), file_name=r["file"], mime="text/csv", width="stretch",
                            icon=":material/download:", key="dl_" + r["file"])

    by = {r["file"]: r for r in raw}
    if "train_atis.csv" in by and "test_atis.csv" in by:
        tr, te = by["train_atis.csv"], by["test_atis.csv"]
        ui.section("From raw files to the train/test split", "Why the example run trains on fewer rows than the CSV files contain.")
        try:
            prof = core.cached("data", SV.data_health)["profile"] or {}
        except core.Unavailable:
            prof = {}
        ui.kpis([
            ("train_atis.csv", f'{tr["rows"]:,}', f'raw rows · {tr["duplicates"]:,} repeat an earlier query', "brand"),
            ("Training rows used", ui.n0(prof.get("training_samples")), "after removing repeats and queries that also appear in the test file", "good"),
            ("test_atis.csv", f'{te["rows"]:,}', f'raw rows · {te["duplicates"]:,} repeat an earlier query', "brand"),
            ("Test rows used", ui.n0(prof.get("test_samples")), "held out; never seen while choosing the model", "good"),
        ])
        if prof:
            ui.note(f"Loaded run: the notebook removed <b>{ui.n0(prof.get('duplicate_rows_removed'))}</b> duplicate rows in total "
                    f"(<b>{ui.n0(prof.get('cross_source_duplicate_queries'))}</b> of them were queries present in both files), and the final "
                    f"train/test text overlap is <b>{ui.n0(prof.get('train_test_text_overlap'))}</b>. These numbers come from the run bundle, not from this page.")

    ui.section("Using your own CSV files", "")
    st.markdown("""- One labelled CSV, or a train/test pair. Column names do not need to match ATIS: the notebook looks for a text column
(`text`, `query`, `user_query`, `utterance`, `sentence`, …) and a label column (`intent`, `label`, `intent_label`, `category`, `class`, …).
- If two columns look equally likely, the notebook stops and asks you to set `COLUMN_OVERRIDES` in the configuration cell.
- A file without a split is divided inside the notebook: stratified by intent, seed 42, test share = intents ÷ rows, kept between 20% and 50%. Intents with only one row stay in training.
- Entity pages need an `ner` column with one BIO tag per word (ATIS has one; the supplementary sets do not).""")
