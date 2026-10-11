"""Welcome page for Travel NLP Lab Studio."""
import streamlit as st
from studio import core, ui

def render():
    ui.header("NLP-34 · BBA (Artificial Intelligence)", "Understand what travellers <em>mean</em>.",
              "Travel NLP Lab Studio classifies travel-related queries, compares available machine-learning results and helps investigate model errors.", hero=True)
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        if st.button("Open Model Lab", type="primary", icon=":material/science:", width="stretch"):
            core.go("model-lab")
        if st.button("Explore Overview", icon=":material/dashboard:", width="stretch"):
            core.go("overview")
        ui.section("How it works", "Training and evaluation happen in the notebook; this studio verifies and explores the resulting bundle.")
        st.markdown('<div class="nf-flow"><span>Travel query</span><i>→</i><span>Preprocessing</span><i>→</i><span>TF-IDF</span><i>→</i><span>Classifier</span><i>→</i><span>Evaluation</span></div>', unsafe_allow_html=True)
    with b:
        ui.card('<div class="nf-k-l">Project</div><div class="nf-k-v" style="font-size:22px">Travel Query Intent Classification Using NLP</div><div class="nf-k-s">Team leader: Krishnav Rastogi<br>Team members: Vinayak Goyal and Sakaar Vir Singh</div>')
        ui.section("Quick actions")
        if st.button("Open Data & Colab", icon=":material/cloud_upload:", width="stretch"):
            core.go("data-colab")
        if st.button("Browse dataset", icon=":material/table_chart:", width="stretch"):
            core.go("dataset")
    st.write("")
    ui.note("<b>Results are experiment-specific.</b> Metrics and predictions come from the selected verified bundle. The app does not invent results or retrain models on startup.")
