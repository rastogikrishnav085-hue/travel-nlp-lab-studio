"""Travel NLP Lab Studio (Streamlit).

Run with:  streamlit run streamlit_app.py      (START.bat option 1 does this for you)

The notebook (Google Colab) trains and evaluates the models and exports travel_nlp_run_bundle.zip.
This app opens that bundle, re-checks it, and shows the results. It never trains and never invents numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from studio import core, ui
from studio.views import (colab, compare, confidence, confusion, dataset, errors, files, home, intents,
                          models, model_lab, predictions)
from studio.views import overview

st.set_page_config(page_title=core.APP_NAME, page_icon="✈️", layout="wide", initial_sidebar_state="auto")
ui.css()
if st.query_params.get("theme") != ui.theme_name():
    st.query_params["theme"] = ui.theme_name()
if st.query_params.get("appearance") != ui.appearance_name():
    st.query_params["appearance"] = ui.appearance_name()


def guarded(fn):
    def run():
        try:
            fn()
        except core.Unavailable as exc:
            ui.unavailable(str(exc))
    run.__name__ = fn.__module__.rsplit(".", 1)[-1]
    return run


P = lambda mod, title, icon, url, default=False: st.Page(guarded(mod.render), title=title, icon=icon, url_path=url, default=default)
pages = {
    "Start": [P(home, "Home", ":material/home:", "home", True), P(overview, "Overview", ":material/dashboard:", "overview"), P(model_lab, "Model Lab", ":material/science:", "model-lab"), P(colab, "Data & Colab", ":material/cloud_upload:", "data-colab")],
    "Explore": [P(dataset, "Dataset", ":material/table_chart:", "dataset"), P(intents, "Intents", ":material/category:", "intents")],
    "Evaluate": [P(models, "Models", ":material/leaderboard:", "models"), P(confusion, "Confusion matrix", ":material/grid_on:", "confusion"),
                 P(predictions, "Predictions", ":material/fact_check:", "predictions"), P(errors, "Error analysis", ":material/troubleshoot:", "errors"),
                 P(confidence, "Confidence", ":material/speed:", "confidence")],
    
    "Evidence": [P(compare, "Compare runs", ":material/compare_arrows:", "compare"), P(files, "Files & checks", ":material/verified:", "files")],
}

core.PAGES = {pg.url_path: pg for group in pages.values() for pg in group}

_LOGO = {"Light": "logo_light.svg", "Dark": "logo.svg", "System": "logo_system.svg"}[ui.appearance_name()]
st.logo(str(Path(__file__).parent / "static" / _LOGO), size="large", icon_image=str(Path(__file__).parent / "static" / "icon.svg"))

nav = st.navigation(pages, position="sidebar", expanded=True)

# ----- run picker (sidebar, below the menu) -----
run = core.ensure_run()
with st.sidebar:
    st.divider()
    st.markdown('<div class="nf-k-l">Open run</div>', unsafe_allow_html=True)
    picker = st.container()   # the run list is drawn here (above the uploader) but filled in after the upload is handled
    # 1) an uploaded bundle is handled first, so the run list below already shows it as the selected run
    up = st.file_uploader("Upload travel_nlp_run_bundle.zip", type=["zip"], key="run_upload",
                          help="The ZIP made by the notebook's last cell. CSV files are uploaded in Colab, not here.")
    if up is not None and st.session_state.get("last_upload") != (up.name, up.size):
        st.session_state.last_upload = (up.name, up.size)
        try:
            with st.spinner("Verifying bundle…"):
                core.open_upload(up.getvalue(), up.name)
            run = st.session_state.run
            st.session_state.run_pick = run["key"]
        except ValueError as exc:
            st.error(str(exc))
    # 2) every run available in this session: examples, runs trained on this computer, and uploads
    labels = {b["key"]: f'{b["name"]}  ·  {"example" if b["kind"] == "example" else "trained here"}' for b in core.bundles()}
    for k, (name, _) in st.session_state.get("uploads", {}).items():
        labels[k] = f"{name}  ·  uploaded"
    keys = list(labels)
    if keys:
        if st.session_state.get("run_pick") not in keys:
            st.session_state.run_pick = run["key"] if run and run["key"] in keys else keys[0]
        pick = picker.selectbox("Run bundle", keys, format_func=labels.get, label_visibility="collapsed", key="run_pick")
        if run is None or pick != run["key"]:
            core.open_key(pick)
            run = st.session_state.run
    if run:
        try:
            ov = core.overview()
            ok = ov["integrity"]["status"] == "verified"
            # only speak up when something is wrong; a good bundle shows its status once (toast) and on the Files/Overview pages
            if not ok:
                st.markdown(ui.pill("Check failed", "bad"), unsafe_allow_html=True)
            elif st.session_state.get("notified_run") != run["key"]:
                st.session_state.notified_run = run["key"]
                st.toast(f'Bundle verified · {ov["run"]["final_model"] or "model"}', icon="✅")
        except core.Unavailable:
            pass
    st.divider()
    st.markdown('<div class="nf-k-l">Appearance</div>', unsafe_allow_html=True)
    selected_theme = st.selectbox("Colour theme", list(ui.THEMES), index=list(ui.THEMES).index(ui.theme_name()), key="global_theme")
    selected_appearance = st.segmented_control("Appearance mode", list(ui.APPEARANCES), default=ui.appearance_name(), key="global_appearance", label_visibility="collapsed")
    if selected_theme != ui.theme_name():
        st.session_state.theme = selected_theme
        st.query_params["theme"] = selected_theme
        st.rerun()
    if selected_appearance and selected_appearance != ui.appearance_name():
        st.session_state.appearance = selected_appearance
        st.query_params["appearance"] = selected_appearance
        st.rerun()
    st.caption(f"Studio {core.VERSION} · local processing · no external AI service")

nav.run()
