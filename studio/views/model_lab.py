"""Unified home for the model's interactive tools."""
import streamlit as st
from studio import core, ui
from studio.views import querylab, autocomplete

def render():
    ui.header("Explore the model", "Model Lab",
              "Try the trained intent classifier or explore next-word suggestions. These tools use the selected run's bundled artefacts; they do not retrain the model.")
    predict_tab, complete_tab = st.tabs(["Query classifier", "Autocomplete"] )
    with predict_tab:
        querylab.render()
    with complete_tab:
        autocomplete.render()
