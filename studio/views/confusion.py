import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import server as SV
from studio import core, ui


def render():
    ui.header("Evaluate", "Confusion matrix", "Rows are what the traveller really meant; columns are what the model chose. "
              "The diagonal is correct. Every other filled cell is a group of mistakes you can open below.")
    cf = core.cached("confusion", SV.confusion, 50)
    labels, M = cf["labels"], np.array(cf["matrix"], dtype=float)
    pairs = cf.get("mutual_pairs") or []
    ui.kpis([
        ("Correct", ui.n0(cf["diagonal"]), f'of {ui.n0(cf["total"])} test queries', "good"),
        ("Mistakes", ui.n0(cf["errors"]), ui.pct(cf["errors"] / cf["total"]) + " of test queries" if cf["total"] else "", "bad"),
        ("Most confused pair", f'{pairs[0]["a"]} ↔ {pairs[0]["b"]}' if pairs else "none", f'{pairs[0]["count"]} mistakes in either direction' if pairs else "", "warn"),
    ])

    rows_tot = M.sum(axis=1, keepdims=True); rows_tot[rows_tot == 0] = 1
    share = M / rows_tot
    n = len(labels)
    diag = np.full_like(share, np.nan); off = np.full_like(share, np.nan)
    for i in range(n):
        for j in range(n):
            if M[i, j] > 0:
                (diag if i == j else off)[i, j] = share[i, j]
    text = [[("" if M[i, j] == 0 else str(int(M[i, j]))) for j in range(n)] for i in range(n)]
    hover = [[f"really <b>{labels[i]}</b> → model chose <b>{labels[j]}</b><br>{int(M[i, j])} queries ({share[i, j]:.0%} of this row)" for j in range(n)] for i in range(n)]
    fig = go.Figure()
    fig.add_heatmap(z=np.zeros_like(share), x=labels, y=labels, colorscale=[[0, ui.HEAT_EMPTY], [1, ui.HEAT_EMPTY]],
                    showscale=False, hoverinfo="skip", xgap=2, ygap=2)
    fig.add_heatmap(z=diag, x=labels, y=labels, colorscale=[[0, ui.HEAT_OK[0]], [1, ui.HEAT_OK[1]]], zmin=0, zmax=1, showscale=False,
                    text=text, texttemplate="%{text}", textfont=dict(size=12, color="#04101F"), hovertext=hover, hovertemplate="%{hovertext}<extra></extra>", xgap=2, ygap=2)
    fig.add_heatmap(z=off, x=labels, y=labels, colorscale=[[0, ui.HEAT_BAD[0]], [1, ui.HEAT_BAD[1]]], zmin=0, zmax=.6, showscale=False,
                    text=text, texttemplate="%{text}", textfont=dict(size=12, color="#04101F"), hovertext=hover, hovertemplate="%{hovertext}<extra></extra>", xgap=2, ygap=2)
    fig.update_yaxes(autorange="reversed", title="really meant", tickfont=dict(size=12))
    fig.update_xaxes(side="top", tickangle=-50, title="model chose", tickfont=dict(size=12))
    ui.plot(ui.style(fig, height=max(520, 30 * n + 160), legend=False))
    st.caption("Colour strength = share of that row, so mistakes on rare intents stay visible. Numbers are counts. Hover a cell for details.")

    ui.section("Open a group of mistakes")
    cells = [(labels[i], labels[j], int(M[i, j])) for i in range(n) for j in range(n) if i != j and M[i, j] > 0]
    cells.sort(key=lambda c: -c[2])
    if not cells:
        ui.note("This run has no mistakes on the test set.")
        return
    opts = [f"{a} → {b}  ({c})" for a, b, c in cells]
    pick = st.selectbox("Really meant → model chose", opts, key="p_cm_pick")
    a, b, _ = cells[opts.index(pick)]
    df = core.predictions_frame(actual=a, predicted=b)
    st.dataframe(df[["text", "actual", "predicted"]].rename(columns={"text": "Query", "actual": "Really meant", "predicted": "Model chose"}),
                 hide_index=True, width="stretch")
    ui.note("Words these queries share with the other intent are a likely reason, but the model's own reasoning is not shown here. "
            "The Query lab shows the model's scores for any query you type.")

    if pairs:
        ui.section("Pairs mixed up most (both directions)")
        st.dataframe(pd.DataFrame([{"Intent A": p["a"], "Intent B": p["b"], "Mistakes": p["count"]} for p in pairs[:12]]), hide_index=True, width="stretch")
