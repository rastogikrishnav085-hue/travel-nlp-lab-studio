import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import server as SV
from studio import core, ui

VERDICT = {"improvement": ("Improvement", "good"), "regression": ("Possible regression", "bad"), "no_clear_change": ("No clear change", "")}


def render():
    ui.header("Evidence", "Compare runs", "Put the open run (A) next to another run bundle (B): headline scores, and the intents that moved. "
              "Useful to compare a Colab run with the shipped example, or two experiments.")
    run = st.session_state.run
    others = [b for b in core.bundles() if b["key"] != run["key"]]
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    opts = {b["key"]: f'{b["name"]} · {"example" if b["kind"] == "example" else "trained here"}' for b in others}
    pick = c1.selectbox("Run B", list(opts), format_func=opts.get, key="p_cmp_b") if opts else None
    up = c2.file_uploader("…or upload run B", type=["zip"], key="p_cmp_up", label_visibility="collapsed")
    if up is None and pick is None:
        ui.note("Only one run bundle is available. Train another one (Colab or START.bat option 2/3) or upload a second bundle to compare.")
        return
    try:
        if up is not None:
            data, name = up.getvalue(), up.name
        else:
            kind, _, name = pick.partition(":"); data = (SV.FOLDERS[kind] / name).read_bytes()
        bkey = (name, len(data), up is not None)
        cached = st.session_state.get("p_cmp_sid")
        if not cached or cached[0] != bkey or cached[1] not in SV.SESSIONS:
            st.session_state.p_cmp_sid = (bkey, SV.open_bytes(data, name)["id"])
        bid = st.session_state.p_cmp_sid[1]
    except Exception as exc:
        st.error(getattr(exc, "detail", str(exc))); return
    r = core.call(lambda a: SV.compare(a, bid))
    A, B = r["a"], r["b"]
    v, tone = VERDICT.get(r.get("verdict"), ("—", ""))
    st.markdown(ui.pill(f"Macro-F1: {v}", tone) + ("" if r["same_data"] else ui.pill("different data", "warn")), unsafe_allow_html=True)
    if r.get("warning"):
        ui.note(ui.esc(r["warning"]), "warn")
    rows = [("Source file", A["source"], B["source"]), ("Model", A["model"], B["model"]), ("Accuracy", ui.pct(A["accuracy"]), ui.pct(B["accuracy"])),
            ("Macro-F1", ui.pct(A["macro_f1"]), ui.pct(B["macro_f1"])),
            ("Macro-F1 95% interval", *[f'{ui.pct(x["ci"]["macro_f1"][0])} – {ui.pct(x["ci"]["macro_f1"][1])}' if x.get("ci") else "—" for x in (A, B)]),
            ("Test queries", ui.n0(A["rows"]), ui.n0(B["rows"])), ("Intents in test", ui.n0(A["labels"]), ui.n0(B["labels"])),
            ("Autocomplete top-3", ui.pct(A.get("autocomplete_top3")), ui.pct(B.get("autocomplete_top3"))),
            ("Verified", "yes" if A["verified"] else "no", "yes" if B["verified"] else "no"), ("Created (UTC)", A["created"], B["created"])]
    st.dataframe(pd.DataFrame(rows, columns=["", "A (open run)", "B"]), hide_index=True, width="stretch")

    d = [x for x in r["deltas"] if x["delta"] is not None]
    if d:
        ui.section("Per-intent F1 change (B minus A)", "Intents present in both runs, biggest moves first. Moves on intents with fewer than 5 test queries are noisy.")
        d = d[:20][::-1]
        fig = go.Figure(go.Bar(y=[x["intent"] for x in d], x=[x["delta"] for x in d], orientation="h",
                               marker_color=[ui.C["teal"] if x["delta"] > 0 else ui.C["red"] for x in d],
                               customdata=[[x["a"], x["b"], x["support_a"], x["support_b"]] for x in d],
                               hovertemplate="<b>%{y}</b>: %{x:+.1%}<br>A %{customdata[0]:.1%} (%{customdata[2]} queries) → B %{customdata[1]:.1%} (%{customdata[3]})<extra></extra>"))
        fig.update_xaxes(tickformat="+.0%", title="change in F1", zeroline=True)
        ui.plot(ui.style(fig, max(260, 26 * len(d)), legend=False))
    if r["only_a"] or r["only_b"]:
        ui.note(f"Only in A: {', '.join(map(ui.esc, r['only_a'][:20])) or 'none'}<br>Only in B: {', '.join(map(ui.esc, r['only_b'][:20])) or 'none'}")
