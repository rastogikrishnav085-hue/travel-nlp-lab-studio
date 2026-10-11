import streamlit as st

from studio import core, ui

SCORE = {"margin": "Decision margin (not a probability)", "probability": "Probability (not calibrated)"}


def render():
    ui.header("Evaluate", "Predictions", "Every held-out test query with the label from the dataset and the label the model chose. "
              "These are the notebook's stored benchmark predictions, not live re-runs.")
    s = core.session()
    full = core.predictions_frame()
    classes = sorted(set(full["actual"]) | set(full["predicted"]))
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1.1])
    q = c1.text_input("Search the query text", key="p_pr_q", placeholder="e.g. denver")
    actual = c2.selectbox("Really meant", [""] + classes, format_func=lambda v: v or "any intent", key="p_pr_a")
    predicted = c3.selectbox("Model chose", [""] + classes, format_func=lambda v: v or "any intent", key="p_pr_p")
    status = c4.segmented_control("Show", ["all", "right", "wrong"], default="all", key="p_pr_s") or "all"
    df = core.predictions_frame(q, actual, predicted, status)
    wrong = int((~full["correct"]).sum())
    st.markdown(ui.pill(f"{len(df):,} of {len(full):,} queries match", "brand") + ui.pill(f"{wrong} wrong in the whole test set", "bad"), unsafe_allow_html=True)
    kind = s.get("score_kind")
    view = df[["text", "actual", "predicted", "correct"] + (["margin"] if kind else [])].rename(
        columns={"text": "Query", "actual": "Really meant", "predicted": "Model chose", "correct": "Correct", "margin": SCORE.get(kind, "Score")})
    st.dataframe(view, hide_index=True, width="stretch", height=520,
                 column_config={"Correct": st.column_config.CheckboxColumn(), SCORE.get(kind, "Score"): st.column_config.NumberColumn(format="%.3f")})
    st.download_button("Download these rows as CSV", view.to_csv(index=False).encode("utf-8"), file_name="predictions_filtered.csv",
                       mime="text/csv", icon=":material/download:")
    if kind == "margin":
        ui.note("The margin is the gap between the model's top two scores. Larger means further from the decision boundary. It is not a probability; see the Confidence page.")
