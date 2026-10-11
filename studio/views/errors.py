import plotly.graph_objects as go
import streamlit as st

from app import server as SV
from studio import core, ui

NAMES = {"low_margin": "Low score", "short_query": "Short query", "long_query": "Long query", "shared_wording": "Shared wording", "repeated_query": "Repeated query"}


def render():
    ui.header("Evaluate", "Error analysis", "Every mistake on the test set, grouped by simple properties you can check. "
              "These are possible contributing factors that describe the data. They are not the model's own reasoning.")
    e = core.cached("errorlab", SV.error_lab)
    th = e.get("thresholds") or {}
    cats = e.get("categories") or {}
    defs = {"low_margin": f"score in the least-certain 20% of test queries (≤ {th['low_margin_max']:.2f})" if th.get("low_margin_max") is not None else "score in the least-certain 20%",
            "short_query": f"{th.get('short_words_max', 0):.0f} words or fewer (shortest 20%)", "long_query": f"{th.get('long_words_min', 0):.0f} words or more (longest 20%)",
            "shared_wording": "contains a word that is common in both the real and the chosen intent", "repeated_query": "the same text appears more than once in the test set"}
    ui.kpis([("Mistakes", ui.n0(e["errors"]), "wrong test predictions", "bad")] +
            [(NAMES.get(k, k), ui.n0(v), defs.get(k, ""), "warn") for k, v in sorted(cats.items(), key=lambda kv: -kv[1])[:3]])

    if cats:
        ui.section("What the mistakes have in common", "A mistake can have several factors, so the bars add up to more than the number of mistakes.")
        ks = sorted(cats, key=cats.get)
        fig = go.Figure(go.Bar(y=[NAMES.get(k, k) for k in ks], x=[cats[k] for k in ks], orientation="h", marker_color=ui.C["red"],
                               text=[f"{cats[k]} of {e['errors']}" for k in ks], textposition="outside", textfont=dict(color=ui.C["muted"]),
                               customdata=[defs.get(k, "") for k in ks], hovertemplate="<b>%{y}</b>: %{x} mistakes<br>%{customdata}<extra></extra>"))
        fig.update_xaxes(title="mistakes", range=[0, max(cats.values()) * 1.25])
        ui.plot(ui.style(fig, 90 + 46 * len(ks), legend=False))

    ui.section("The mistakes")
    pick = st.segmented_control("Factor", ["All"] + [NAMES.get(k, k) for k in cats], default="All", key="p_er_f") or "All"
    key = next((k for k in cats if NAMES.get(k, k) == pick), None)
    items = [i for i in e["items"] if key is None or key in i["categories"]]
    st.caption(f"{len(items)} shown" + (f" · the list is limited to {e['shown']}" if e.get("shown") and e["shown"] < e["errors"] else ""))
    for it in items:
        facts = "".join(ui.pill(NAMES.get(c, c), "warn") for c in it["categories"]) or ui.pill("no listed factor")
        extra = ""
        if it.get("shared_words"):
            extra += f'<div class="nf-k-s">Shared words: {ui.esc(", ".join(it["shared_words"]))}</div>'
        if it.get("similar_correct"):
            extra += f'<div class="nf-k-s">Similar queries the model got right: {ui.esc(" · ".join(it["similar_correct"][:2]))}</div>'
        score = f' · score {it["margin"]:.3f}' if it.get("margin") is not None else ""
        st.markdown(f'<div class="nf-note warn" style="margin-bottom:9px"><b>{ui.esc(it["text"])}</b><br>'
                    f'{ui.pill("really " + it["actual"], "good")}{ui.pill("model chose " + it["predicted"], "bad")}{facts}'
                    f'<div class="nf-k-s">{it["words"]} words{score}</div>{extra}</div>', unsafe_allow_html=True)
