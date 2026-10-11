import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import server as SV
from studio import core, ui


def _set(text):
    st.session_state.p_ac_text = text


def _complete(word):
    t = st.session_state.get("p_ac_text", "")
    head = t if t.endswith(" ") or not t else t.rsplit(" ", 1)[0] + " " if " " in t else ""
    st.session_state.p_ac_text = head + word + " "


def _append(word):
    t = st.session_state.get("p_ac_text", "")
    st.session_state.p_ac_text = (t if t.endswith(" ") or not t else t + " ") + word + " "


def render():
    ui.header("Lab", "Autocomplete", "A word-level n-gram model, trained on the training queries only, suggests how a travel query continues "
              "and which intent it is heading towards. Click a suggestion to add it.")
    core.call(SV.autocomplete, SV.AutoReq(text="show"))   # Unavailable if the bundle has no autocomplete model
    if "p_ac_text" not in st.session_state:
        st.session_state.p_ac_text = "show me flights fro"
    st.text_input("Start typing a travel query", key="p_ac_text", max_chars=300)
    text = st.session_state.p_ac_text
    r = core.call(SV.autocomplete, SV.AutoReq(text=text))

    best = (r.get("completions") or [{}])[0].get("text")
    if best and best.lower().startswith(text.lower().strip()) and text.strip():
        rest = best[len(text.strip()):]
        st.markdown(f'<div class="nf-card" style="font-size:20px"><span style="color:var(--ink)">{ui.esc(text.strip())}</span>'
                    f'<span style="color:var(--faint)">{ui.esc(rest)}</span></div>', unsafe_allow_html=True)
    if r.get("unknown_words"):
        ui.note(f'Never seen in training: <b>{ui.esc(", ".join(r["unknown_words"]))}</b>. Suggestions after an unknown word are less reliable.', "warn")

    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        if r.get("word_completions"):
            st.markdown(f'**Finish the word “{ui.esc(r["partial"])}”**')
            cols = st.columns(min(4, len(r["word_completions"])))
            for i, w in enumerate(r["word_completions"][:4]):
                cols[i].button(f'{w["word"]}', key=f"wc{i}", on_click=_complete, args=(w["word"],), width="stretch",
                               help=f'{w["share"]:.0%} of matching words · {w["confidence"]} confidence')
        nw = [w for w in r.get("next_words") or [] if not w.get("is_end")]
        if nw:
            st.markdown("**Next word**" + (" (after finishing the current word)" if r.get("partial") else ""))
            cols = st.columns(min(5, len(nw)))
            for i, w in enumerate(nw[:5]):
                cols[i].button(w["word"], key=f"nw{i}", on_click=_append, args=(w["word"],), width="stretch",
                               help=f'{w["share"]:.0%} of the model\'s next-word weight · {w["confidence"]} confidence')
        if r.get("completions"):
            st.markdown("**Whole queries the model would write**")
            for i, c in enumerate(r["completions"][:4]):
                st.button(c["text"], key=f"fc{i}", on_click=_set, args=(c["text"],), icon=":material/north_west:")
    with c2:
        it = r.get("intents") or []
        if it:
            fig = go.Figure(go.Bar(y=[x[0] for x in it[:5]][::-1], x=[x[1] for x in it[:5]][::-1], orientation="h", marker_color=ui.C["violet"],
                                   text=[f"{x[1]:.0%}" for x in it[:5]][::-1], textposition="outside", textfont=dict(color=ui.C["muted"]),
                                   hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig.update_xaxes(tickformat=".0%", range=[0, 1.15], title="where the query is heading (autocomplete model's intent hint)")
            ui.plot(ui.style(fig, 240, legend=False))
            st.caption("This hint comes from the autocomplete model, not from the main classifier. Use the Query lab for the classifier's answer.")

    ui.section("Is the autocomplete accurate?", "Re-scored here on every word position of the held-out test queries, then compared with the numbers the notebook stored.")
    ac = core.cached("accheck", SV.autocomplete_check)
    if not ac.get("available"):
        ui.unavailable(ac.get("reason", "")); return
    f = ac["fresh"]
    rows = {x["metric"]: x for x in ac["compare"]["rows"]}
    base = rows.get("Frequent-words baseline (top-3)", {}).get("recomputed")
    ui.kpis([("Top-1", ui.pct(f["top1"]), "the right next word is the first suggestion", "brand"),
             ("Top-3", ui.pct(f["top3"]), f"in the first three · baseline {ui.pct(base)}", "good"),
             ("Top-5", ui.pct(f["top5"]), "in the first five", ""),
             ("Positions scored", ui.n0(f["positions"]), f'{ui.n0(f["queries"])} test queries', "")])
    ok = ac["compare"]["status"] == "verified"
    st.markdown(ui.pill("matches the notebook's stored numbers" if ok else "differs from the stored numbers", "good" if ok else "bad"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Measure": x["metric"], "Recomputed here": x["recomputed"], "Stored by notebook": x["stored"], "Match": x["match"]}
                               for x in ac["compare"]["rows"]]), hide_index=True, width="stretch",
                 column_config={"Match": st.column_config.CheckboxColumn()})
    if (ac.get("stored") or {}).get("fit_verdict"):
        ui.note(f'<b>Over-fitting check (from the notebook):</b> {ui.esc(ac["stored"]["fit_verdict"])}.')
