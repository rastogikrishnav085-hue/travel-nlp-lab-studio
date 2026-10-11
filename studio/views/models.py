import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import server as SV
from studio import core, ui


def render():
    ui.header("Evaluate", "Models", "Three TF-IDF classifiers were compared with cross-validation on the training data only. "
              "The winner was then tested once on the held-out test set.")
    m = core.cached("models", SV.models)
    ov = core.overview()
    f, ci = m.get("final") or {}, ov["metrics"].get("ci") or {}
    rec = ov["metrics"]["recomputed"] or {}
    ui.kpis([
        ("Test accuracy", ui.pct(rec.get("accuracy", f.get("accuracy"))), "share of test queries classified correctly", "good"),
        ("Test Macro-F1", ui.pct(rec.get("macro_f1", f.get("macro_f1"))), "average over intents, each counts equally", "warn"),
        ("Weighted-F1", ui.pct(f.get("weighted_f1")), "average weighted by intent size", ""),
        ("Majority baseline", ui.pct(m.get("baseline_macro_f1")), "Macro-F1 if every query were called the most common intent", "bad"),
    ])

    comp = [r for r in m.get("comparison") or [] if r.get("CV_Macro_F1") is not None]
    if comp:
        ui.section("Model selection (5-fold cross-validation on training data)",
                   "Bars show mean Macro-F1 across folds; whiskers show ± one standard deviation. The highest mean wins.")
        comp = sorted(comp, key=lambda r: r["CV_Macro_F1"])
        win = f.get("final_model")
        fig = go.Figure(go.Bar(
            y=[r["Model"] for r in comp], x=[r["CV_Macro_F1"] for r in comp], orientation="h",
            error_x=dict(type="data", array=[r.get("CV_Macro_F1_SD") or 0 for r in comp], color=ui.C["muted"], thickness=1.5, width=6),
            marker_color=[ui.BAR_WIN if r["Model"] == win else ui.THEMES[ui.theme_name()]["brand"] for r in comp],
            text=[f'{r["CV_Macro_F1"]:.1%}' for r in comp], textposition="inside", insidetextanchor="start", textfont=dict(color="#04101F", size=13),
            customdata=[[r.get("CV_Accuracy")] for r in comp],
            hovertemplate="<b>%{y}</b><br>CV Macro-F1 %{x:.1%}<br>CV accuracy %{customdata[0]:.1%}<extra></extra>"))
        fig.update_xaxes(title="cross-validation Macro-F1", tickformat=".0%", range=[0, 1])
        ui.plot(ui.style(fig, 260, legend=False))
        excl = f.get("selection_cv_excluded_classes") or []
        ui.note(f"<b>{ui.esc(win)}</b> was selected on training data only; the test set was not used to choose it. "
                + (f"{len(excl)} intents with too few training rows ({ui.n0(f.get('selection_cv_excluded_rows'))} rows) were left out of the "
                   f"cross-validation folds only: {', '.join(map(ui.esc, excl))}." if excl else ""))
        table = pd.DataFrame([{"Model": r["Model"], "CV Macro-F1": r.get("CV_Macro_F1"), "± SD": r.get("CV_Macro_F1_SD"),
                               "CV accuracy": r.get("CV_Accuracy"), "Chosen": "✓" if r["Model"] == win else ""} for r in comp[::-1]])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in ("CV Macro-F1", "± SD", "CV accuracy")})

    ui.section("Final test result", f"One evaluation on the held-out test set. The 95% intervals come from {ui.n0(ci.get('n_boot'))} bootstrap resamples of the test predictions (seed {ci.get('seed')}).")
    acc_ci, f1_ci = ci.get("accuracy"), ci.get("macro_f1")
    rows = [("Accuracy", f.get("accuracy"), acc_ci), ("Macro precision", f.get("macro_precision"), None), ("Macro recall", f.get("macro_recall"), None),
            ("Macro-F1", f.get("macro_f1"), f1_ci), ("Weighted precision", f.get("weighted_precision"), None),
            ("Weighted recall", f.get("weighted_recall"), None), ("Weighted-F1", f.get("weighted_f1"), None)]
    st.dataframe(pd.DataFrame([{"Metric": a, "Value": ui.pct(v), "95% interval": f"{ui.pct(c[0])} – {ui.pct(c[1])}" if c else ""} for a, v, c in rows]),
                 hide_index=True, width="stretch")
    if f.get("test_only_intents"):
        ui.note("Intents that appear only in the test set count as zero in Macro-F1 because the model never saw them: "
                + "".join(ui.pill(x, "bad") for x in f["test_only_intents"]), "warn")

    ui.section("Per-intent scores", "Recomputed here from the stored predictions. Sort by any column.")
    per = pd.DataFrame(m.get("per_intent") or [])
    if len(per):
        st.dataframe(per[["intent", "support", "precision", "recall", "f1"]].rename(columns={"intent": "Intent", "support": "Test queries"}),
                     hide_index=True, width="stretch", height=420,
                     column_config={k: st.column_config.ProgressColumn(k.title(), min_value=0, max_value=1, format="percent") for k in ("precision", "recall", "f1")})


    # Keep provenance and dataset details available at the end of the main model dashboard,
    # rather than making users navigate to a separate visualisation page.
    with st.expander("Run, dataset and methodology details", expanded=False):
        run = ov.get("run") or {}
        integ = ov.get("integrity") or {}
        left, right = st.columns(2, gap="large")
        with left:
            ui.section("Selected experiment")
            st.write({
                "Run name": run.get("name", "—"),
                "Final model": run.get("final_model", "—"),
                "Model selection": run.get("selection", {}),
                "Bundle verification": integ.get("status", "unknown"),
                "Metric status": ov.get("metric_status", "unknown"),
            })
        with right:
            ui.section("Dataset in this run")
            st.write({
                "Test queries evaluated": rec.get("rows", "—"),
                "Intents represented in test": rec.get("labels", "—"),
                "Metric source": "Recomputed from stored predictions" if ov.get("metric_status") == "recomputed" else "Compared with notebook-reported values",
            })
        st.caption("Supplementary raw CSVs are available on Data & Colab. They are not automatically mixed into the selected ATIS benchmark.")
