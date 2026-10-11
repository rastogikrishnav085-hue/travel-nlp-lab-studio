"""Night Flight design system: colours, CSS, small HTML components and the Plotly theme.

One place for every colour so all pages and charts stay consistent.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

# Colour themes (accent palettes on the Night Flight dark base). Teal = correct, red = wrong and amber = warning
# never change, so a chart means the same thing in every theme; only the accent, background tint and glow change.
THEMES = {
    "Midnight Navy": dict(brand="#60A5FA", accent="#2DD4BF", glow="96,165,250", bg="#071421", side="#091B2B", s2="#10283B", dark_bg="#071421", dark_side="#091B2B", dark_s2="#10283B", blurb="Deep navy with blue and teal accents", light_brand="#1D4ED8", light_accent="#0F766E", light_bg="#F3F7FC", light_side="#EAF1F8", light_s2="#FFFFFF", light_ink="#15283D", light_muted="#43576C", light_faint="#536A82", light_line="#CBD8E6"),
    "Ocean Blue": dict(brand="#38BDF8", accent="#60A5FA", glow="56,189,248", bg="#061523", side="#092238", s2="#10314A", dark_bg="#061523", dark_side="#092238", dark_s2="#10314A", blurb="Ocean blue and clear cyan", light_brand="#0369A1", light_accent="#1D4ED8", light_bg="#F0F8FD", light_side="#E2F2FC", light_s2="#FFFFFF", light_ink="#10324B", light_muted="#36576F", light_faint="#4D6C82", light_line="#BFDDED"),
    "Forest Green": dict(brand="#34D399", accent="#A3E635", glow="52,211,153", bg="#071710", side="#0B2118", s2="#123425", dark_bg="#071710", dark_side="#0B2118", dark_s2="#123425", blurb="Forest green, emerald and mint", light_brand="#047857", light_accent="#3F6212", light_bg="#F1FAF4", light_side="#E3F3E8", light_s2="#FFFFFF", light_ink="#173B2A", light_muted="#426652", light_faint="#50735F", light_line="#C5E1CF"),
    "Royal Purple": dict(brand="#A78BFA", accent="#C4B5FD", glow="167,139,250", bg="#110D20", side="#1B1430", s2="#281D43", dark_bg="#110D20", dark_side="#1B1430", dark_s2="#281D43", blurb="Royal purple and lavender", light_brand="#6D28D9", light_accent="#5B21B6", light_bg="#F7F4FF", light_side="#EEE8FC", light_s2="#FFFFFF", light_ink="#2B2050", light_muted="#554875", light_faint="#655784", light_line="#D8CFF0"),
    "Sunset Orange": dict(brand="#FDBA74", accent="#F97316", glow="249,115,22", bg="#1A100C", side="#28170F", s2="#3A2115", dark_bg="#1A100C", dark_side="#28170F", dark_s2="#3A2115", blurb="Warm orange and amber highlights", light_brand="#9A3412", light_accent="#B53A0A", light_bg="#FFF7F0", light_side="#FCEBDD", light_s2="#FFFFFF", light_ink="#492711", light_muted="#714B31", light_faint="#80583C", light_line="#EBD0B8"),
    "Rose Pink": dict(brand="#F9A8D4", accent="#F472B6", glow="244,114,182", bg="#1A0C17", side="#281224", s2="#3B1A33", dark_bg="#1A0C17", dark_side="#281224", dark_s2="#3B1A33", blurb="Rose and magenta accents", light_brand="#9D174D", light_accent="#BE185D", light_bg="#FFF4FA", light_side="#FBE5F1", light_s2="#FFFFFF", light_ink="#481B37", light_muted="#70445E", light_faint="#80516B", light_line="#EAC7DB"),
    "Slate Teal": dict(brand="#93C5FD", accent="#5EEAD4", glow="37,99,235", bg="#101827", side="#121F31", s2="#1C2B40", dark_bg="#101827", dark_side="#121F31", dark_s2="#1C2B40", blurb="Clean slate, blue and teal", light_brand="#1D4ED8", light_accent="#0F766E", light_bg="#F5F7FA", light_side="#E9EEF5", light_s2="#FFFFFF", light_ink="#172338", light_muted="#46566D", light_faint="#53647B", light_line="#CBD5E1"),
}
DEFAULT_THEME = "Midnight Navy"
# Borders of form controls and buttons (WCAG 1.4.11: >= 3:1 against the page and the control fill).
CTL_DARK, CTL_LIGHT = "#6F8AA3", "#66798F"
APPEARANCES = ("Light", "Dark", "System")


# ----------------------------------------------------------------------------- colour maths
def _rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def luminance(h: str) -> float:
    def f(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = _rgb(h)
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def dual_safe(h: str, target: float = 0.19) -> str:
    """Same hue as `h`, lightness adjusted so it keeps >= 3:1 against both the dark and the light page backgrounds.

    Used for chart colours in System appearance, where the operating system (not the server) decides light or dark.
    """
    import colorsys
    r, g, b = (v / 255 for v in _rgb(h))
    hue, _, sat = colorsys.rgb_to_hls(r, g, b)
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        rr, gg, bb = colorsys.hls_to_rgb(hue, mid, sat)
        L = luminance("#%02X%02X%02X" % (round(rr * 255), round(gg * 255), round(bb * 255)))
        lo, hi = (mid, hi) if L < target else (lo, mid)
    rr, gg, bb = colorsys.hls_to_rgb(hue, (lo + hi) / 2, sat)
    return "#%02X%02X%02X" % (round(rr * 255), round(gg * 255), round(bb * 255))


def theme_name() -> str:
    try:
        name = st.session_state.get("theme")
        if name is None:
            q = st.query_params.get("theme")
            name = q if q in THEMES else DEFAULT_THEME
            st.session_state.theme = name
        return name if name in THEMES else DEFAULT_THEME
    except Exception:
        return DEFAULT_THEME


def appearance_name() -> str:
    try:
        value = st.session_state.get("appearance")
        if value in APPEARANCES:
            return value
        q = st.query_params.get("appearance")
        value = q if q in APPEARANCES else "System"
        st.session_state.appearance = value
        return value
    except Exception:
        return "System"


def theme() -> dict:
    base = dict(THEMES[theme_name()])
    appearance = appearance_name()
    # Streamlit's own widgets are always dark (see .streamlit/config.toml). Light re-skins them with CSS (LIGHT_CSS).
    # System cannot be known on the server, so the page variables follow the operating system in CSS, and charts use
    # colours that keep >= 3:1 on both a dark and a light page.
    light = appearance == "Light"
    system = appearance == "System"
    base.update({
        "bg": base["light_bg"] if light else base["dark_bg"],
        "side": base["light_side"] if light else base["dark_side"],
        "s2": base["light_s2"] if light else base["dark_s2"],
        "ink": base["light_ink"] if light else "#F1F5F9",
        "muted": base["light_muted"] if light else "#CBD5E1",
        "faint": base["light_faint"] if light else "#A8B7C8",
        "line": base["light_line"] if light else "#294055",
        "surface": base["light_s2"] if light else base["dark_s2"],
        "is_light": light,
        "is_system": system,
        "on_brand": "#FFFFFF" if light else "#071421",
        "ctl": CTL_LIGHT if light else CTL_DARK,
        "brand": base["light_brand"] if light else base["brand"],
        "accent": base["light_accent"] if light else base["accent"],
        "teal": "#0F766E" if light else "#2DD4BF",
        "red": "#B91C1C" if light else "#FF7A85",
        "amber": "#92400E" if light else "#F5B93E",
        "violet": "#6D28D9" if light else "#B39DFF",
    })
    return base


def chart_palette(t: dict) -> dict:
    """Colours for Plotly. In System appearance the browser, not the server, knows if the page is light or dark, so the
    series colours are mid-tones that keep >= 3:1 on both. Text colours are made to follow the page by CSS (NATIVE_CSS)."""
    if not t["is_system"]:
        return {k: t[k] for k in ("brand", "accent", "teal", "red", "amber", "violet")}
    pal = {k: dual_safe(t[k]) for k in ("brand", "accent")}
    pal.update({"teal": "#0D9488", "red": "#DC2626", "amber": "#B45309", "violet": "#7C3AED"})
    return pal


class _Colours(dict):
    """Fixed colours plus the accent of the current theme (looked up on every access)."""
    def __getitem__(self, k):
        t = theme()
        if k in ("brand", "accent", "teal", "red", "amber", "violet"):
            return chart_palette(t)[k]
        if k in ("bg", "ink", "muted", "faint", "line", "surface"):
            return t[k]
        if k == "surface2":
            return t["s2"]
        return dict.__getitem__(self, k)


C = _Colours(bg="#070D1A", surface="rgba(15,26,46,.72)", surface2="#0F1A2E", line="#1F2E4A", line_soft="rgba(120,160,220,.14)",
             ink="#EAF2FF", muted="#9DB0CC", faint="#7489A8", brand="#5EA8FF", accent="#2DD4BF", teal="#2DD4BF", amber="#F5B93E",
             red="#FF7A85", violet="#B39DFF", flight="#7DB0FF", hotel="#B9A0FF", car="#F2A65C", rail="#52D6A5")
PLOT_CFG = {"displayModeBar": False, "responsive": True}
# Fills that carry dark text (cell numbers, in-bar labels). They are opaque, so the text keeps >= 7:1 on a dark or a light page.
BAR_WIN = "#2DD4BF"
HEAT_OK = ("#A8EBE0", "#2DD4BF")
HEAT_BAD = ("#FFC9CD", "#FF7A85")
HEAT_EMPTY = "rgba(128,150,180,.10)"
BAR_TEXT = "#04101F"

esc = lambda v: html.escape(str(v if v is not None else ""))


def pct(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f}%"


def n0(v):
    return "—" if v is None else f"{int(v):,}"


def tone_f1(f):
    return "muted" if f is None else "good" if f >= .85 else "warn" if f >= .6 else "bad"


class _Tone(dict):
    def __getitem__(self, k):
        return {"good": C["teal"], "warn": C["amber"], "bad": C["red"], "muted": C["faint"], "brand": C["brand"]}[k]


TONE = _Tone()

CSS = """
<style>
:root {{ --bg:{bg}; --side:{side}; --surface:{surface}; --s2:{s2}; --line:{line}; --soft:rgba({glow},.18);
  --ink:{ink}; --muted:{muted}; --faint:{faint}; --brand:{brand}; --accent:{accent}; --glow:{glow};
  --light-bg:{light_bg}; --light-side:{light_side}; --light-s2:{light_s2}; --light-ink:{light_ink}; --light-muted:{light_muted}; --light-faint:{light_faint}; --light-line:{light_line}; --light-brand:{light_brand}; --light-accent:{light_accent};
  --teal:{teal}; --amber:{amber}; --red:{red}; --violet:{violet}; --on-brand:{on_brand}; --ctl:{ctl}; --page-border:var(--line); }}
[data-testid="stSidebar"] {{ background: var(--side) !important; }}
[data-testid="stAppViewContainer"] {{ color: var(--ink); }}
[data-testid="stMain"] {{ color: var(--ink); }}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] {{ border-color: var(--line) !important; }}
a {{ color: var(--brand); }}
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1100px 560px at 88% -12%, rgba(var(--glow),.17), transparent 60%),
              radial-gradient(900px 520px at -12% 112%, color-mix(in srgb, var(--accent) 9%, transparent), transparent 60%), var(--bg) !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1380px; }}
h1, h2, h3 {{ letter-spacing: -.03em; }}
p, li {{ line-height: 1.65; }}

/* sidebar navigation */
[data-testid="stSidebarHeader"] {{ padding-bottom: .25rem; }}
[data-testid="stSidebarNavSeparator"] {{ display:none; }}
[data-testid="stNavSectionHeader"] {{ color: var(--faint) !important; font-size: 12px !important; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 10px; transition: background .18s, transform .18s, box-shadow .18s; }}
[data-testid="stSidebarNavLink"]:hover {{ background: rgba(var(--glow),.08); transform: translateX(3px); }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: rgba(var(--glow),.13); box-shadow: inset 3px 0 0 var(--brand); }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: var(--ink) !important; font-weight: 700; }}

/* brand block */
.nf-brand {{ display:flex; gap:11px; align-items:center; margin: 2px 0 6px; }}
.nf-logo {{ width:38px; height:38px; border-radius:12px; display:grid; place-items:center; color:var(--on-brand); font-size:18px;
  background: linear-gradient(135deg, var(--brand), var(--accent)); box-shadow: 0 6px 20px rgba(var(--glow),.28); }}
.nf-brand b {{ font-family:"Space Grotesk",sans-serif; font-size:16px; color:var(--ink); display:block; line-height:1.2; }}
.nf-brand span {{ font-size:12px; color:var(--muted); }}

/* page header */
.nf-eyebrow {{ color: var(--brand); font-weight: 700; letter-spacing: .14em; font-size: 12px; text-transform: uppercase; }}
.nf-h1 {{ font-family:"Space Grotesk",sans-serif; font-size: clamp(30px,3.4vw,44px); line-height: 1.08; font-weight: 700; letter-spacing: -.035em; color: var(--ink); margin: 8px 0 10px; }}
.nf-hero {{ font-size: clamp(38px,5vw,60px); line-height: 1.02; }}
.nf-h1 em {{ font-style: normal; background: linear-gradient(90deg, var(--brand), var(--accent)); -webkit-background-clip: text; background-clip:text; color: transparent; }}
.nf-lead {{ color: var(--muted); font-size: 16px; line-height: 1.7; max-width: 72ch; margin-bottom: 18px; }}
.nf-sec {{ font-family:"Space Grotesk",sans-serif; font-size: 22px; font-weight: 700; letter-spacing: -.02em; color: var(--ink); margin: 30px 0 4px; }}
.nf-sub {{ color: var(--muted); font-size: 14px; margin: 0 0 12px; }}

/* cards */
.nf-card {{ background: var(--surface); border: 1px solid var(--soft); border-radius: 18px; padding: 18px 20px; backdrop-filter: blur(14px);
  transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease; animation: nf-rise .5s ease both; height: 100%; }}
.nf-card:hover {{ transform: translateY(-3px); border-color: rgba(var(--glow),.38); box-shadow: 0 14px 36px rgba(var(--glow),.14); }}
.nf-k-l {{ font-size: 12px; letter-spacing: .1em; color: var(--muted); font-weight: 700; text-transform: uppercase; }}
.nf-k-v {{ font-family:"Space Grotesk",sans-serif; font-size: 32px; font-weight: 700; letter-spacing: -.035em; color: var(--ink); margin: 6px 0 2px; line-height:1.15; }}
.nf-k-s {{ font-size: 13px; color: var(--muted); line-height: 1.5; }}
.nf-card.good {{ box-shadow: inset 0 2px 0 var(--teal); }} .nf-card.warn {{ box-shadow: inset 0 2px 0 var(--amber); }}
.nf-card.bad {{ box-shadow: inset 0 2px 0 var(--red); }} .nf-card.brand {{ box-shadow: inset 0 2px 0 var(--brand); }}
.nf-pill {{ display:inline-block; padding: 3px 11px; border-radius: 99px; font-size: 12.5px; font-weight: 700; margin: 0 6px 6px 0;
  background: rgba(var(--glow),.08); color: var(--ink); border: 1px solid var(--soft); }}
.nf-pill.good {{ color: var(--teal); border-color: color-mix(in srgb, var(--teal) 40%, transparent); background: color-mix(in srgb, var(--teal) 8%, transparent); }}
.nf-pill.warn {{ color: var(--amber); border-color: color-mix(in srgb, var(--amber) 40%, transparent); background: color-mix(in srgb, var(--amber) 8%, transparent); }}
.nf-pill.bad {{ color: var(--red); border-color: color-mix(in srgb, var(--red) 40%, transparent); background: color-mix(in srgb, var(--red) 8%, transparent); }}
.nf-pill.brand {{ color: var(--brand); border-color: rgba(var(--glow),.45); }}
.nf-note {{ border: 1px solid var(--soft); border-left: 3px solid var(--brand); background: rgba(var(--glow),.06); border-radius: 12px;
  padding: 12px 15px; color: var(--muted); font-size: 14px; margin: 6px 0 14px; }}
.nf-note b {{ color: var(--ink); }}
.nf-note.warn {{ border-left-color: var(--amber); background: color-mix(in srgb, var(--amber) 8%, transparent); }}
.nf-q {{ font-size: 19px; line-height: 1.95; color: var(--ink); }}
.nf-ent {{ padding: 2px 6px; border-radius: 6px; border-bottom: 2px solid var(--brand); background: rgba(var(--glow),.14); }}
.nf-ent small {{ font-size: 12px; color: var(--brand); font-weight: 700; margin-left: 5px; white-space: nowrap; }}
.nf-ent.t {{ border-color: var(--amber); background: color-mix(in srgb, var(--amber) 13%, transparent); }} .nf-ent.t small {{ color: var(--amber); }}
.nf-ent.o {{ border-color: var(--violet); background: color-mix(in srgb, var(--violet) 13%, transparent); }} .nf-ent.o small {{ color: var(--violet); }}
.nf-steps {{ counter-reset: s; list-style: none; padding: 0; margin: 0; }}
.nf-steps li {{ counter-increment: s; position: relative; padding: 2px 0 14px 44px; color: var(--muted); }}
.nf-steps li b {{ color: var(--ink); }}
.nf-steps li::before {{ content: counter(s); position: absolute; left: 0; top: 0; width: 30px; height: 30px; border-radius: 10px; display: grid; place-items: center;
  font-weight: 700; color: var(--on-brand); background: linear-gradient(135deg, var(--brand), var(--accent)); }}
.nf-flow {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
.nf-flow span {{ padding: 9px 14px; border-radius: 12px; border: 1px solid var(--soft); background: var(--surface); color: var(--ink); font-weight: 600; font-size: 14px; animation: nf-rise .5s ease both; }}
.nf-flow i {{ color: var(--brand); font-style: normal; }}
.nf-run {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; color: var(--muted); font-size: 13.5px; margin-bottom: 4px; }}
code {{ font-size: 13px !important; }}

/* buttons: lift on hover, press on click */
.stButton > button, .stDownloadButton > button, .stLinkButton > a, [data-testid="stPopover"] > div > button {{
  font-weight: 700; transition: transform .15s ease, box-shadow .2s ease, border-color .2s ease, background .2s ease; }}
.stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover, [data-testid="stPopover"] > div > button:hover {{
  transform: translateY(-2px); box-shadow: 0 10px 26px rgba(var(--glow),.20); border-color: var(--brand); }}
.stButton > button:active, .stDownloadButton > button:active {{ transform: translateY(0) scale(.97); }}
.stButton > button[kind="primary"]:hover {{ box-shadow: 0 10px 28px rgba(var(--glow),.32); }}

/* charts and tables fade in */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{ animation: nf-fade .6s ease both; }}
[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--soft); border-radius: 16px; padding: 14px 16px; }}
@keyframes nf-rise {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes nf-fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
/* System appearance: the variables follow the operating system. */
@media (prefers-color-scheme: light) {{
  :root:has(style[data-appearance="system"]) {{ --bg:var(--light-bg); --side:var(--light-side); --surface:var(--light-s2); --s2:var(--light-s2); --ink:var(--light-ink); --muted:var(--light-muted); --faint:var(--light-faint); --line:var(--light-line); --brand:var(--light-brand); --accent:var(--light-accent); --on-brand:#FFFFFF; --ctl:{ctl_light}; --teal:#0F766E; --amber:#92400E; --red:#B91C1C; --violet:#6D28D9; }}
}}
body:has([data-testid="stAppViewContainer"]) {{ color: var(--ink); }}
:focus-visible {{ outline: 3px solid var(--brand) !important; outline-offset: 2px !important; }}
@media (max-width: 760px) {{ .block-container {{ padding: 1.1rem .8rem 2rem; }} .nf-h1 {{ font-size: clamp(27px, 7vw, 38px); }} .nf-card {{ padding: 14px; }} }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation: none !important; transition: none !important; }} }}
</style>
"""


# Streamlit's own widgets are drawn with a fixed dark theme (config.toml). These rules re-skin them with the theme variables, so
# Light, Dark and System all stay readable. Only variables are used here, so one block serves every appearance.
NATIVE_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: var(--side) !important; }
[data-testid="stSidebarNavLink"] span, [data-testid="stSidebarNavLink"] p { color: var(--muted) !important; }
[data-testid="stSidebarNavLink"]:hover span, [data-testid="stSidebarNavLink"]:hover p,
[data-testid="stSidebarNavLink"][aria-current="page"] span, [data-testid="stSidebarNavLink"][aria-current="page"] p { color: var(--ink) !important; }
[data-testid="stSidebarCollapseButton"] *, [data-testid="stExpandSidebarButton"] *, [data-testid="stHeader"] button *, [data-testid="stHeader"] svg { color: var(--muted) !important; fill: currentColor; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; opacity: 1 !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { color: var(--muted) !important; }
[data-testid="stTooltipIcon"] *, [data-testid="stTooltipHoverTarget"] * { color: var(--muted) !important; }

/* inputs */
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"], [data-testid="stTextInputRootElement"],
[data-testid="stTextAreaRootElement"], [data-testid="stNumberInputContainer"], [data-testid="stFileUploaderDropzone"] {
  background: var(--s2) !important; border: 1px solid var(--ctl) !important; color: var(--ink) !important; }
[data-testid="stTextInputField"], [data-testid="stSelectbox"] input, textarea { background: transparent !important; color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
input::placeholder, textarea::placeholder { color: var(--faint) !important; -webkit-text-fill-color: var(--faint) !important; opacity: 1 !important; }
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg, [data-testid="stTextInputRootElement"] svg { color: var(--muted) !important; fill: currentColor; }
[data-testid="stFileUploaderDropzoneInstructions"] *, [data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color: var(--muted) !important; }
[data-testid="stSelectbox"] [role="group"]:focus-within, [data-testid="stTextInputRootElement"]:focus-within { border-color: var(--brand) !important; }

/* drop-down menus (rendered outside the app container) */
[data-testid="stSelectboxVirtualDropdown"], [role="listbox"], [data-testid="stPopoverBody"], [data-testid="stTooltipContent"] {
  background: var(--s2) !important; color: var(--ink) !important; border: 1px solid var(--line) !important; }
[role="option"], [role="option"] * { color: var(--ink) !important; }
[role="option"][aria-selected="true"], [role="option"]:hover { background: rgba(var(--glow),.16) !important; }

/* buttons */
button[data-testid^="stBaseButton-secondary"], a[data-testid^="stBaseLinkButton-secondary"] {
  background: var(--s2) !important; color: var(--ink) !important; border: 1px solid var(--ctl) !important; }
button[data-testid^="stBaseButton-secondary"] *, a[data-testid^="stBaseLinkButton-secondary"] * { color: var(--ink) !important; }
button[data-testid^="stBaseButton-secondary"]:hover, a[data-testid^="stBaseLinkButton-secondary"]:hover { border-color: var(--brand) !important; }
button[data-testid^="stBaseButton-primary"], a[data-testid^="stBaseLinkButton-primary"] {
  background: linear-gradient(135deg, var(--brand), var(--accent)) !important; color: var(--on-brand) !important; border: 0 !important; }
button[data-testid^="stBaseButton-primary"] *, a[data-testid^="stBaseLinkButton-primary"] * { color: var(--on-brand) !important; }
button[data-testid="stBaseButton-elementToolbar"], button[data-testid="stBaseButton-headerNoPadding"] { background: transparent !important; }
[data-testid="stButtonGroup"] button { background: var(--s2) !important; color: var(--ink) !important; border: 1px solid var(--ctl) !important; }
[data-testid="stButtonGroup"] button * { color: var(--ink) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] {
  border-color: var(--brand) !important; background: color-mix(in srgb, var(--brand) 16%, var(--s2)) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] *, [data-testid="stButtonGroup"] button[aria-pressed="true"] * { color: var(--ink) !important; font-weight: 700; }

/* tabs, expanders, code, alerts, sliders */
[data-testid="stTab"], [data-testid="stTab"] p { color: var(--muted) !important; }
[data-testid="stTab"][aria-selected="true"], [data-testid="stTab"][aria-selected="true"] p { color: var(--ink) !important; font-weight: 700; }
[data-testid="stTabs"] [role="tablist"] { border-bottom-color: var(--line) !important; }
[data-testid="stTab"] > div:not([data-testid]) { background: var(--brand) !important; }
[data-testid="stExpander"] details { background: var(--s2) !important; border-color: var(--ctl) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink) !important; }
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stMarkdownPre"], pre { background: var(--s2) !important; color: var(--ink) !important; }
code, pre code { color: var(--ink) !important; }
:not(pre) > code { background: color-mix(in srgb, var(--brand) 12%, var(--s2)) !important; border-radius: 6px; padding: 1px 5px; }
pre code { background: transparent !important; }
[data-testid="stAlert"], [data-testid="stAlertContainer"] { background: color-mix(in srgb, var(--brand) 10%, var(--s2)) !important; color: var(--ink) !important; }
[data-testid="stAlert"] *, [data-testid="stAlertContainer"] * { color: var(--ink) !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBar"] *, [data-testid="stSlider"] [data-testid="stMarkdownContainer"] * { color: var(--ink) !important; }
[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"]) { background: var(--brand) !important; box-shadow: 0 0 0 3px var(--s2); }
[data-testid="stElementToolbar"] { background: var(--s2) !important; border: 1px solid var(--line) !important; }
[data-testid="stElementToolbar"] * { color: var(--muted) !important; }

/* charts: axis, legend and title text follow the theme; annotation text inside heat-maps keeps its own colour */
[data-testid="stPlotlyChart"] .xtick text, [data-testid="stPlotlyChart"] .ytick text, [data-testid="stPlotlyChart"] .xtitle, [data-testid="stPlotlyChart"] .ytitle,
[data-testid="stPlotlyChart"] .legendtext, [data-testid="stPlotlyChart"] .gtitle, [data-testid="stPlotlyChart"] .bartext-outside,
[data-testid="stPlotlyChart"] .textpoint text { fill: var(--muted) !important; }
[data-testid="stPlotlyChart"] .barlayer .point path { stroke: var(--muted) !important; stroke-width: 1px !important; }
[data-testid="stPlotlyChart"] .hoverlayer .hovertext path { fill: var(--s2) !important; stroke: var(--line) !important; }
[data-testid="stPlotlyChart"] .hoverlayer .hovertext text { fill: var(--ink) !important; }
</style>
"""

# st.dataframe draws on a canvas that CSS cannot recolour; on a light page it is inverted (and hue-corrected) so it matches.
DF_LIGHT = '[data-testid="stDataFrameResizable"] { filter: invert(1) hue-rotate(180deg); }'


def css():
    t = theme()
    keys = ("bg", "side", "s2", "brand", "accent", "glow", "ink", "muted", "faint", "line", "surface", "teal", "amber", "red", "violet",
            "on_brand", "ctl", "light_bg", "light_side", "light_s2", "light_ink", "light_muted", "light_faint", "light_line", "light_brand", "light_accent")
    appearance = appearance_name()
    base = CSS.format(ctl_light=CTL_LIGHT, **{k: t[k] for k in keys}).replace("<style>", f'<style data-appearance="{appearance.lower()}">', 1)
    if appearance == "Light":
        extra = f"<style>{DF_LIGHT}</style>"
    elif appearance == "System":
        # System mode: the logo image is icon-only; its two text lines are drawn here so they always use the page colours (light or dark)
        brand_text = ('<style>'
                      '[data-testid="stSidebarHeader"] > :has(> img), [data-testid="stLogoLink"] { display: grid !important; grid-template-columns: auto 1fr; column-gap: 8px; align-items: center; }'
                      '[data-testid="stSidebarHeader"] > :has(> img) img, [data-testid="stLogoLink"] img { grid-column: 1; grid-row: 1 / span 2; }'
                      '[data-testid="stSidebarHeader"] > :has(> img)::before, [data-testid="stLogoLink"]::before { content: "Travel NLP Lab"; grid-column: 2; grid-row: 1; align-self: end; font: 700 13px/1.2 "Segoe UI", Inter, Arial, sans-serif; color: var(--ink); white-space: nowrap; }'
                      '[data-testid="stSidebarHeader"] > :has(> img)::after, [data-testid="stLogoLink"]::after { content: "Studio \u00b7 NLP \u00b7 NLU"; grid-column: 2; grid-row: 2; align-self: start; font: 400 9.5px/1.2 "Segoe UI", Inter, Arial, sans-serif; color: var(--muted); white-space: nowrap; }'
                      '</style>')
        extra = (f"<style>@media (prefers-color-scheme: light) {{ :root:has(style[data-appearance=\"system\"]) {DF_LIGHT} }}</style>" + brand_text)
    else:
        extra = ""
    st.markdown(base + NATIVE_CSS + extra, unsafe_allow_html=True)


def header(eyebrow: str, title: str, lead: str = "", hero: bool = False):
    st.markdown(f'<div class="nf-eyebrow">{esc(eyebrow)}</div><div class="nf-h1 {"nf-hero" if hero else ""}">{title}</div>'
                + (f'<div class="nf-lead">{lead}</div>' if lead else ""), unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="nf-sec">{esc(title)}</div>' + (f'<div class="nf-sub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


def pill(text, tone=""):
    return f'<span class="nf-pill {tone}">{esc(text)}</span>'


def note(text_html: str, tone=""):
    st.markdown(f'<div class="nf-note {tone}">{text_html}</div>', unsafe_allow_html=True)


def card(inner_html: str, tone=""):
    st.markdown(f'<div class="nf-card {tone}">{inner_html}</div>', unsafe_allow_html=True)


def kpi_html(label, value, sub="", tone=""):
    return f'<div class="nf-card {tone}"><div class="nf-k-l">{esc(label)}</div><div class="nf-k-v">{value}</div><div class="nf-k-s">{sub}</div></div>'


def kpis(items):
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(kpi_html(*it), unsafe_allow_html=True)


def unavailable(msg: str):
    note(f"<b>Not available for this run.</b> {esc(msg)}", "warn")


ENT_PRE = {"fromloc": "from", "toloc": "to", "stoploc": "stop", "depart_date": "departs", "depart_time": "departs",
           "arrive_date": "arrives", "arrive_time": "arrives", "return_date": "returns", "return_time": "returns"}


def ent_label(t: str) -> str:
    a, *r = str(t).split(".")
    rest = " ".join(r).removesuffix("_name").replace("_", " ")
    return f"{ENT_PRE.get(a, a.replace('_', ' '))} · {rest}" if r else a.replace("_", " ")


def ent_kind(t: str) -> str:
    return "" if "loc" in t or "city" in t or "airport" in t else "t" if ("date" in t or "time" in t) else "o"


def annotated(text: str, ner: str) -> str:
    """ATIS BIO tags -> highlighted query (only when word and tag counts match)."""
    words, tags = str(text).split(), str(ner).split()
    if len(words) != len(tags):
        return esc(text)
    out, i = [], 0
    while i < len(words):
        t = tags[i]
        if t.startswith("B-"):
            typ, j = t[2:], i + 1
            while j < len(words) and tags[j] == "I-" + typ:
                j += 1
            out.append(f'<span class="nf-ent {ent_kind(typ)}">{esc(" ".join(words[i:j]))}<small>{esc(ent_label(typ))}</small></span>')
            i = j
        else:
            out.append(esc(words[i])); i += 1
    return " ".join(out)


# ----------------------------------------------------------------------------- Plotly theme
def style(fig: go.Figure, height=360, legend=True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color=C["ink"]), showlegend=legend,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=13, color=C["muted"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=C["surface2"], bordercolor=C["line"], font=dict(color=C["ink"], size=13)),
        colorway=[C["brand"], C["teal"], C["amber"], C["red"], C["accent"]])
    fig.update_xaxes(gridcolor="rgba(120,160,220,.10)", zerolinecolor="rgba(120,160,220,.22)", linecolor=C["line"], tickfont=dict(color=C["muted"], size=12),
                     title_font=dict(color=C["muted"], size=13))
    fig.update_yaxes(gridcolor="rgba(120,160,220,.10)", zerolinecolor="rgba(120,160,220,.22)", linecolor=C["line"], tickfont=dict(color=C["muted"], size=12),
                     title_font=dict(color=C["muted"], size=13))
    return fig


def plot(fig, key=None):
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key=key)
