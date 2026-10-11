import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import server as SV
from studio import core, ui

BAND = {"high": ("good", "high"), "medium": ("warn", "medium"), "low": ("bad", "low")}


def _sample():
    smp = core.call(SV.autocomplete_samples, 1, random.randrange(10**6))["samples"]
    if smp:
        st.session_state.p_ql_text = smp[0]["text"]


def render():
    ui.header("Lab", "Query lab", "Type any travel question. The classifier bundled with this run scores it live, on this computer. "
              "This is live inference, separate from the stored benchmark results on the other pages.")
    core.call(SV.classify, SV.ClassifyReq(text="flight"))   # raises Unavailable if this run has no classifier
    if "p_ql_text" not in st.session_state:
        st.session_state.p_ql_text = "how much is a first class ticket from boston to dallas"
    c1, c2, c3 = st.columns([4, 1, 1.4], vertical_alignment="bottom")
    text = c1.text_input("Travel query", key="p_ql_text", max_chars=300)
    c2.button("Classify", type="primary", width="stretch", icon=":material/bolt:")
    c3.button("Use a real test query", width="stretch", icon=":material/shuffle:", on_click=_sample)
    text = (text or "").strip()
    if not text:
        ui.note("Type a travel query first.", "warn"); return

    r = core.call(SV.classify, SV.ClassifyReq(text=text))
    rb = core.call(SV.robustness, SV.ClassifyReq(text=text))
    tone, word = BAND.get(r.get("band"), ("", r.get("band") or "—"))
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        ui.card(f'<div class="nf-k-l">Live inference · {ui.esc(r["model"]["name"])}</div><div class="nf-k-v" style="font-size:40px">{ui.esc(r["intent"])}</div>'
                + (ui.pill(f"certainty: {word}", tone) if r.get("band") else ui.pill("no certainty band for this run", ""))
                + ui.pill(f'margin {r["margin"]:.2f}') + ui.pill(f'{r["known_features"]} known words/pairs', "brand")
                + (f'<div class="nf-k-s" style="margin-top:8px">Bands come from this run\'s {ui.esc(r["bands"]["basis"])}: low below {r["bands"]["low_below"]:.2f}, '
                   f'high from {r["bands"]["high_from"]:.2f}.</div>' if r.get("bands") else ""), tone or "brand")
        sb = r.get("stored_benchmark")
        if sb:
            ui.note(f'<b>This exact query is in the test set.</b> Dataset label: <b>{ui.esc(sb["actual"])}</b> · stored benchmark prediction: '
                    f'<b>{ui.esc(sb["predicted"])}</b> ({"correct" if sb["correct"] else "wrong"}).', "" if sb["correct"] else "warn")
        if not r.get("bands"):
            ui.note("This run stored probabilities, not decision margins, for its test predictions, so the live margin cannot be placed in low/medium/high bands (different scales).")
        if r["known_features"] == 0:
            ui.note("None of these words are in the model's vocabulary, so this answer only reflects the model's built-in bias towards common intents. Treat it as “unknown”.", "warn")
    with right:
        top = r["top"][::-1]
        fig = go.Figure(go.Bar(y=[t["intent"] for t in top], x=[t["score"] for t in top], orientation="h",
                               marker_color=[ui.C["teal"] if t["intent"] == r["intent"] else ui.C["brand"] for t in top],
                               text=[f'{t["score"]:+.2f}' for t in top], textposition="outside", textfont=dict(color=ui.C["muted"]),
                               hovertemplate="%{y}: score %{x:.3f}<extra></extra>"))
        lo, hi = min(0, min(t["score"] for t in top)), max(0, max(t["score"] for t in top))
        pad = .35 * (hi - lo or 1)
        fig.update_xaxes(title="classifier score for each intent (higher wins)", range=[lo - pad, hi + pad])
        fig.update_yaxes(ticksuffix="  ")
        ui.plot(ui.style(fig, 260, legend=False))
    for n in r.get("notes") or []:
        st.caption(n)

    ui.section("Does a small change flip the answer?", "The same query, slightly altered, scored again by the same model.")
    st.markdown(ui.pill("stable: no variant changed the intent", "good") if rb.get("stable") else ui.pill(f'{rb.get("changed")} variant(s) changed the intent', "bad"),
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Change": v["variant"], "Text": v["text"], "Intent": v["intent"], "Margin": v["margin"], "Same as original": v["same_as_original"],
                                "Removed by preprocessing": v["neutralised_by_preprocessing"]} for v in rb["variants"]]), hide_index=True, width="stretch",
                 column_config={"Margin": st.column_config.NumberColumn(format="%.2f"), "Same as original": st.column_config.CheckboxColumn(),
                                "Removed by preprocessing": st.column_config.CheckboxColumn()})
    if rb.get("note"):
        st.caption(rb["note"])
