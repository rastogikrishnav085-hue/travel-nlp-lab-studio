import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import server as SV
from studio import core, ui


def render():
    ui.header("Explore", "Dataset", "The training and test data inside the loaded run bundle, and the checks the notebook ran before any model was trained.")
    d = core.cached("data", SV.data_health)
    prof, lk = d.get("profile") or {}, core.cached("leakage", SV.leakage)
    ui.kpis([
        ("Training rows", ui.n0(lk.get("train_rows")), f'{ui.n0(lk.get("train_intents"))} intents', "brand"),
        ("Test rows", ui.n0(lk.get("test_rows")), f'{ui.n0(lk.get("test_intents"))} intents · held out', "brand"),
        ("Duplicates removed", ui.n0(prof.get("duplicate_rows_removed")),
         f'repeated queries, incl. {ui.n0(prof.get("cross_source_duplicate_queries"))} found in both train and test files' if prof.get("cross_source_duplicate_queries") else "repeated queries", ""),
        ("Imbalance", f'{lk["imbalance_ratio"]:,.0f}×' if lk.get("imbalance_ratio") else "—",
         f'largest intent ({lk["largest_class"]["intent"]}) vs smallest' if lk.get("largest_class") else "", "warn"),
    ])

    s = core.session()
    tr, ttext, tlab, tname = SV.split_frame(s["files"], "train")
    te, etext, elab, ename = s["test"]
    if tr is not None and te is not None:
        ui.section("How many examples each intent has", "Log scale. Intents with very few training rows are hard to learn, which is why Macro-F1 is lower than Accuracy.")
        a = tr[tlab].astype(str).value_counts(); b = te[elab].astype(str).value_counts()
        order = sorted(set(a.index) | set(b.index), key=lambda k: (a.get(k, 0), b.get(k, 0)))
        fig = go.Figure()
        fig.add_bar(y=order, x=[a.get(k, 0) for k in order], name="training", orientation="h", marker_color=ui.C["brand"],
                    hovertemplate="%{y}: %{x} training rows<extra></extra>")
        fig.add_bar(y=order, x=[b.get(k, 0) for k in order], name="test", orientation="h", marker_color=ui.C["teal"],
                    hovertemplate="%{y}: %{x} test rows<extra></extra>")
        fig.update_layout(barmode="group", bargap=.25)
        fig.update_xaxes(type="log", title="rows (log scale)")
        ui.plot(ui.style(fig, height=max(380, 24 * len(order))))
        c1, c2 = st.columns(2)
        with c1:
            if lk.get("test_only_intents"):
                ui.note("<b>In the test set but never in training</b> (the model cannot predict these): " +
                        "".join(ui.pill(x, "bad") for x in lk["test_only_intents"]))
        with c2:
            if lk.get("classes_under_5_train_rows"):
                ui.note("<b>Fewer than 5 training rows</b> (left out of cross-validation, kept in the final fit): " +
                        "".join(ui.pill(x, "warn") for x in lk["classes_under_5_train_rows"]), "warn")

    ui.section("Checks before training", "Run by the notebook's data-quality gate and re-read from the bundle.")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        rows = [{"Check": g["check"], "Result": g["status"].upper()} for g in lk.get("gate", [])]
        rows += [{"Check": r["Check"], "Result": f'{r["Status"]} ({r["Count"]})'} for r in d.get("leakage") or []]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with c2:
        nd = lk.get("near_duplicates") or {}
        if nd:
            ui.card(f'<div class="nf-k-l">Near-duplicate test queries</div><div class="nf-k-v">{ui.n0(nd.get("test_rows_with_near_train_match"))}</div>'
                    f'<div class="nf-k-s">test queries with a training query at cosine similarity ≥ {nd.get("threshold")} '
                    f'({ui.pct(nd.get("share_of_test"))} of the test set). Different labels among them: {ui.n0(nd.get("pairs_with_different_labels"))}. '
                    f'Exact overlap is zero, so these are look-alikes such as “what does q mean” vs “what does y mean”, not leaked rows.</div>', "brand")

    ent = None
    try:
        ent = core.cached("entities", SV.entities_for_run)
    except core.Unavailable as exc:
        ui.section("Travel entities")
        ui.unavailable(str(exc))
    if ent:
        ui.section("Travel entities in the queries", f"Dataset annotations from {ui.esc(ent['source'])}. These are labels written by the dataset authors, not model predictions.")
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            t = ent["types"][:12][::-1]
            fig = go.Figure(go.Bar(y=[ui.ent_label(x["type"]) for x in t], x=[x["count"] for x in t], orientation="h",
                                   marker_color=[ui.C["brand"] if ui.ent_kind(x["type"]) == "" else ui.C["amber"] if ui.ent_kind(x["type"]) == "t" else ui.C["violet"] for x in t],
                                   hovertemplate="%{y}: %{x} mentions<extra></extra>"))
            fig.update_xaxes(title="mentions in annotated queries")
            ui.plot(ui.style(fig, 400, legend=False))
        with c2:
            ui.kpis([("Annotated queries", ui.n0(ent["queries"]), f'{ent["mean_entities"]:.1f} entities per query on average', "brand")])
            st.write("")
            routes = ent.get("routes") or []
            if routes:
                st.dataframe(pd.DataFrame([{"From": r.get("origin"), "To": r.get("destination"), "Queries": r.get("count")} for r in routes[:8]]),
                             hide_index=True, width="stretch")

    ui.section("Browse the rows", "The processed files the model was trained and tested on.")
    which = st.segmented_control("File", ["Training", "Test"], default="Test", key="p_ds_which")
    frame, txt, lab = (tr, ttext, tlab) if which == "Training" else (te, etext, elab)
    if frame is not None:
        q = st.text_input("Search the text", key="p_ds_q", placeholder="e.g. denver")
        view = frame[[txt, lab]].rename(columns={txt: "query", lab: "intent"})
        if q:
            view = view[view["query"].astype(str).str.contains(q, case=False, regex=False)]
        st.caption(f"{len(view):,} of {len(frame):,} rows")
        st.dataframe(view.head(500), hide_index=True, width="stretch", height=360)
