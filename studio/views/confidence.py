import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import server as SV
from studio import core, ui


def render():
    c = core.cached("confidence", SV.confidence, "")
    kind = c.get("kind")
    what = "decision margin (gap between the top two class scores; not a probability)" if kind == "margin" else "highest predicted probability (not calibrated)"
    ui.header("Evaluate", "Confidence", f"How sure was the model on each test query? The score here is the {what}. "
              "If wrong answers have lower scores, low-score queries can be flagged for a human to check.")
    ui.kpis([("Test queries", ui.n0(c["n"]), "with a stored score", "brand"),
             ("Median score · correct", f'{c["median_correct"]:.2f}', "right answers", "good"),
             ("Median score · wrong", f'{c["median_wrong"]:.2f}' if c.get("median_wrong") is not None else "—", "mistakes", "bad")])

    wrong_total = sum(x["wrong"] for x in c["histogram"])
    ui.section("Scores of right vs wrong answers", "Most mistakes sit at the low end, which is what makes the score useful." if wrong_total
               else "This run has no wrong answers on its test set, so the score has nothing to separate.")
    h = c["histogram"]
    mids = [(b["from"] + b["to"]) / 2 for b in h]
    fig = go.Figure()
    fig.add_bar(x=mids, y=[b["correct"] for b in h], name="correct", marker_color=ui.C["teal"], width=[(b["to"] - b["from"]) * .92 for b in h],
                customdata=[[b["from"], b["to"]] for b in h], hovertemplate="%{customdata[0]:.2f}–%{customdata[1]:.2f}: %{y} correct<extra></extra>")
    fig.add_bar(x=mids, y=[b["wrong"] for b in h], name="wrong", marker_color=ui.C["red"], width=[(b["to"] - b["from"]) * .92 for b in h],
                customdata=[[b["from"], b["to"]] for b in h], hovertemplate="%{customdata[0]:.2f}–%{customdata[1]:.2f}: %{y} wrong<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="score")
    fig.update_yaxes(title="test queries")
    ui.plot(ui.style(fig, 330))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section("Accuracy by score band", "Test queries split into five equal-sized groups, from least to most sure.")
        b = c["bands"]
        fig = go.Figure(go.Bar(x=[x["band"] for x in b], y=[x["accuracy"] for x in b], marker_color=[ui.TONE[ui.tone_f1(x["accuracy"])] for x in b],
                               text=[f'{x["accuracy"]:.1%}' for x in b], textposition="outside", textfont=dict(color=ui.C["ink"]),
                               customdata=[x["rows"] for x in b], hovertemplate="%{x}: %{y:.1%} correct of %{customdata} queries<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", range=[0, 1.12], title="accuracy")
        ui.plot(ui.style(fig, 320, legend=False))
    with c2:
        ui.section("Flag the least-sure queries", "Move the slider: queries below the threshold go to a person; the rest are answered automatically.")
        cur = c["curve"]
        i = st.select_slider("Share of queries flagged for review", options=list(range(len(cur))), value=min(4, len(cur) - 1),
                             format_func=lambda k: f'{cur[k]["flagged"] / c["n"]:.0%}', key="p_cf_t")
        p = cur[i]
        st.markdown(ui.kpi_html("Accuracy on the rest", ui.pct(p["accuracy_accepted"]),
                                f'{p["accepted"]:,} answered automatically · {p["flagged"]:,} flagged (score below {p["threshold"]:.2f}) · '
                                f'all queries: {ui.pct(cur[0]["accuracy_accepted"])}', "good"), unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(x=[x["coverage"] for x in cur], y=[x["accuracy_accepted"] for x in cur], mode="lines+markers",
                                   line=dict(color=ui.C["brand"], width=2.5), marker=dict(size=6),
                                   hovertemplate="answer %{x:.0%} automatically → %{y:.1%} correct<extra></extra>"))
        fig.add_scatter(x=[p["coverage"]], y=[p["accuracy_accepted"]], mode="markers", marker=dict(size=15, color=ui.C["teal"], line=dict(color="#04101F", width=2)),
                        hoverinfo="skip", showlegend=False)
        fig.update_xaxes(title="share answered automatically", tickformat=".0%", autorange="reversed")
        fig.update_yaxes(title="accuracy on those", tickformat=".0%")
        ui.plot(ui.style(fig, 240, legend=False))

    ui.section("Least-sure test queries")
    low = pd.DataFrame(c.get("lowest") or [])
    if len(low):
        st.dataframe(low[["text", "actual", "predicted", "correct", "margin"]].rename(columns={"text": "Query", "actual": "Really meant", "predicted": "Model chose",
                     "correct": "Correct", "margin": "Score"}), hide_index=True, width="stretch",
                     column_config={"Correct": st.column_config.CheckboxColumn(), "Score": st.column_config.NumberColumn(format="%.3f")})
