import pandas as pd
import streamlit as st

from app import server as SV
from studio import core, ui

STATUS = {"verified": "good", "n/a": "", "legacy_contract": "warn"}


def render():
    ui.header("Evidence", "Files & checks", "What is inside the run bundle, and the independent checks this app ran on it. "
              "SHA-256 fingerprints show the files were not changed after the notebook wrote them; they do not prove who made them.")
    ov = core.overview()
    ok = ov["integrity"]["status"] == "verified"
    v = ov["versions"]
    ui.kpis([("Bundle", "Verified" if ok else "Failed", f'{ov["integrity"]["counts"].get("verified", 0)} files match their fingerprints', "good" if ok else "bad"),
             ("Metrics", "Recomputed" if ov["metric_status"] == "recomputed" else ov["metric_status"].title(), "accuracy and Macro-F1 from the stored predictions", "good"),
             ("Quality gate", (ov.get("quality_gate") or {}).get("status", "—"), "the notebook's final checks", "good"),
             ("Versions", f'{v.get("studio")}', f'studio · notebook {v.get("pipeline")} · bundle schema {v.get("bundle_schema")} · contract {v.get("contract")}', "brand")])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section("Checks done here")
        st.dataframe(pd.DataFrame([{"Check": r["check"], "Result": "PASS" if r["ok"] else "FAIL", "Detail": r["detail"]} for r in ov["reconcile"]]),
                     hide_index=True, width="stretch")
        if ov["integrity"]["problems"]:
            ui.note("<b>Problems:</b> " + ", ".join(f'{ui.esc(p["file"])} ({ui.esc(p["status"])})' for p in ov["integrity"]["problems"]), "warn")
    with c2:
        ui.section("Notebook quality gate")
        qg = (ov.get("quality_gate") or {}).get("checks") or {}
        st.dataframe(pd.DataFrame([{"Check": k.replace("_", " "), "Result": "PASS" if val else "FAIL"} for k, val in qg.items()]), hide_index=True, width="stretch")

    ui.section("Run information")
    run = ov["run"]
    st.dataframe(pd.DataFrame([("Run ID", run["run_id"]), ("Created (UTC)", run["created"]), ("Final model", run["final_model"]),
                               ("Model selection", f'{run["selection"].get("method")} on {run["selection"].get("metric")}, {run["selection"].get("folds")} folds'),
                               ("Random seed", run["seed"]), ("Dataset fingerprint", run["fingerprint"])], columns=["Field", "Value"]).astype(str),
                 hide_index=True, width="stretch")

    fl = core.call(SV.file_list)
    figs = fl.get("figures") or []
    if figs:
        ui.section("Figures made by the notebook", "Saved by the notebook during training (PNG). Click a name to view it full width.")
        name = st.selectbox("Figure", [f["path"] for f in figs], format_func=lambda p: next(f["name"] for f in figs if f["path"] == p), key="p_fl_fig")
        st.image(core.file_bytes(name), width="stretch")

    ui.section("All files in the bundle")
    files = fl["files"]
    st.dataframe(pd.DataFrame([{"File": f["path"].split("/", 1)[-1], "Size (KB)": round(f["size"] / 1024, 1), "Fingerprint": f["status"]} for f in files]),
                 hide_index=True, width="stretch", height=320)
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    path = c1.selectbox("Download a file", [f["path"] for f in files], format_func=lambda p: p.split("/", 1)[-1], key="p_fl_dl")
    c2.download_button("Download", core.file_bytes(path), file_name=path.split("/")[-1], width="stretch", icon=":material/download:")
    card = next((f["path"] for f in files if f["path"].endswith("model_card.md")), None)
    if card:
        with st.expander("Model card (written by the notebook)"):
            st.markdown(core.file_bytes(card).decode("utf-8", "replace"))
