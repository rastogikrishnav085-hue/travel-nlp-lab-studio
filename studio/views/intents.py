import math

import plotly.graph_objects as go
import streamlit as st

from app import server as SV
from studio import core, ui


def render():
    ui.header("Explore", "Intents", "One row per intent: how often it appears, how well the model recognises it, the words that point to it, and what it gets mixed up with.")
    data = core.cached("intents", SV.intents)
    items = data["intents"]
    scored = [i for i in items if i.get("f1") is not None and i.get("support")]

    ui.section("Score vs. number of test queries", "Each dot is an intent. Intents on the left have very few test queries, so their F1 can swing a lot from a single mistake.")
    if scored:
        fig = go.Figure(go.Scatter(
            x=[i["support"] for i in scored], y=[i["f1"] for i in scored], mode="markers+text",
            text=[i["intent"] if (i["train_rows"] and (i["f1"] < .9 or i["support"] >= 25)) else "" for i in scored], hovertext=[i["intent"] for i in scored],
            textposition="top center", textfont=dict(size=11.5, color=ui.C["muted"]),
            marker=dict(size=[10 + 4 * math.log2(1 + i["train_rows"]) for i in scored], color=[ui.TONE[ui.tone_f1(i["f1"])] for i in scored],
                        line=dict(color=ui.C["bg"], width=1), opacity=.92),
            customdata=[[i["train_rows"], i["precision"], i["recall"]] for i in scored],
            hovertemplate="<b>%{hovertext}</b><br>F1 %{y:.1%} · %{x} test queries<br>%{customdata[0]} training rows<br>precision %{customdata[1]:.1%} · recall %{customdata[2]:.1%}<extra></extra>"))
        fig.update_xaxes(type="log", title="test queries (log scale)")
        fig.update_yaxes(title="F1", tickformat=".0%", range=[-.06, 1.14])
        ui.plot(ui.style(fig, 420, legend=False))
        st.caption("Colour: teal F1 ≥ 85%, amber 60–85%, red under 60%. Dot size grows with the number of training rows. "
                   "Small red dots at 0% are intents that never appear in training. Hover any dot for its name and scores.")

    ui.section("Look at one intent")
    names = [i["intent"] for i in sorted(items, key=lambda i: (i.get("f1") is None, i.get("f1") or 0))]
    first_learnable = next((n for n in names if next(i for i in items if i["intent"] == n)["train_rows"]), names[0])
    default = st.session_state.get("focus_intent") if st.session_state.get("focus_intent") in names else first_learnable
    pick = st.selectbox("Intent (weakest first)", names, index=names.index(default), key="p_int_pick")
    st.session_state.focus_intent = pick
    x = next(i for i in items if i["intent"] == pick)
    ui.kpis([
        ("F1", ui.pct(x.get("f1")), "balance of precision and recall", ui.tone_f1(x.get("f1"))),
        ("Precision", ui.pct(x.get("precision")), "when the model says it, how often it is right", ""),
        ("Recall", ui.pct(x.get("recall")), "of the real ones, how many it found", ""),
        ("Rows", f'{ui.n0(x.get("train_rows"))} / {ui.n0(x.get("support"))}', "training / test", "brand"),
    ])
    if not x.get("in_test"):
        ui.note("This intent has no test queries, so it has no score. It can still be predicted.", "warn")
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        st.markdown("**Words and word pairs that push the model towards it** (largest classifier weights)")
        st.markdown("".join(ui.pill(f, "brand") for f in x.get("features") or []) or "—", unsafe_allow_html=True)
        st.markdown("**Most frequent words and word pairs in its training queries** (common English words removed)")
        st.markdown("".join(ui.pill(f'{t["term"]} · {t["count"]}') for t in x.get("terms") or []) or "—", unsafe_allow_html=True)
        st.markdown("**Mistakes**")
        cw = ", ".join(f'{c["intent"]} ({c["count"]})' for c in x.get("confused_with") or []) or "none"
        ca = ", ".join(f'{c["intent"]} ({c["count"]})' for c in x.get("confused_as") or []) or "none"
        st.markdown(f"- Really **{pick}**, model chose: {cw}\n- Model chose **{pick}**, really: {ca}")
    with c2:
        st.markdown("**Real test queries** (mistakes first)")
        ex = x.get("examples") or []
        if not ex:
            st.caption("No test queries for this intent.")
        for e in ex[:8]:
            ok = e["correct"]
            st.markdown(f'<div class="nf-note {"" if ok else "warn"}" style="margin:0 0 8px">{ui.esc(e["text"])}<br>'
                        f'{ui.pill("correct" if ok else "model chose " + e["predicted"], "good" if ok else "bad")}</div>', unsafe_allow_html=True)
