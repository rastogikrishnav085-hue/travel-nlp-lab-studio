"""Exact WCAG contrast for every theme and appearance, computed from the colour definitions in studio/ui.py.

The browser audit (tests/browser/theme_audit.py) measures what is really rendered; this test pins the colour definitions
themselves so a bad palette edit fails immediately and without a browser.
"""
import pytest

from studio import ui

THEMES = list(ui.THEMES)


def _modes(name):
    t = ui.THEMES[name]
    light = dict(bg=t["light_bg"], side=t["light_side"], s2=t["light_s2"], ink=t["light_ink"], muted=t["light_muted"],
                 faint=t["light_faint"], brand=t["light_brand"], accent=t["light_accent"], on_brand="#FFFFFF", ctl=ui.CTL_LIGHT)
    dark = dict(bg=t["dark_bg"], side=t["dark_side"], s2=t["dark_s2"], ink="#F1F5F9", muted="#CBD5E1", faint="#A8B7C8",
                brand=t["brand"], accent=t["accent"], on_brand="#071421", ctl=ui.CTL_DARK)
    return {"light": light, "dark": dark}


def test_there_are_seven_themes():
    assert THEMES == ["Midnight Navy", "Ocean Blue", "Forest Green", "Royal Purple", "Sunset Orange", "Rose Pink", "Slate Teal"]


@pytest.mark.parametrize("name", THEMES)
@pytest.mark.parametrize("mode", ["light", "dark"])
def test_text_colours_reach_4_5(name, mode):
    c = _modes(name)[mode]
    for surface in ("bg", "side", "s2"):
        for fg in ("ink", "muted", "faint", "brand", "accent"):
            assert ui.contrast(c[fg], c[surface]) >= 4.5, f"{name}/{mode}: {fg} on {surface} = {ui.contrast(c[fg], c[surface]):.2f}"


@pytest.mark.parametrize("name", THEMES)
@pytest.mark.parametrize("mode", ["light", "dark"])
def test_button_text_reaches_4_5_on_both_gradient_ends(name, mode):
    c = _modes(name)[mode]
    for end in ("brand", "accent"):
        assert ui.contrast(c["on_brand"], c[end]) >= 4.5, f"{name}/{mode}: button text on {end} = {ui.contrast(c['on_brand'], c[end]):.2f}"


@pytest.mark.parametrize("name", THEMES)
@pytest.mark.parametrize("mode", ["light", "dark"])
def test_control_borders_and_focus_ring_reach_3(name, mode):
    c = _modes(name)[mode]
    for surface in ("bg", "side", "s2"):
        assert ui.contrast(c["ctl"], c[surface]) >= 3, f"{name}/{mode}: control border on {surface} = {ui.contrast(c['ctl'], c[surface]):.2f}"
        assert ui.contrast(c["brand"], c[surface]) >= 3, f"{name}/{mode}: focus ring on {surface}"


@pytest.mark.parametrize("name", THEMES)
def test_system_chart_colours_work_on_a_dark_and_a_light_page(name):
    t = ui.THEMES[name]
    pal = {k: ui.dual_safe(t[k]) for k in ("brand", "accent")}
    pal.update({"teal": "#0D9488", "red": "#DC2626", "amber": "#B45309", "violet": "#7C3AED"})
    for k, col in pal.items():
        for bg in (t["dark_bg"], t["light_bg"]):
            assert ui.contrast(col, bg) >= 3, f"{name}: {k} {col} on {bg} = {ui.contrast(col, bg):.2f}"


@pytest.mark.parametrize("name", THEMES)
def test_dark_text_on_opaque_chart_fills_reaches_4_5(name):
    fills = [ui.THEMES[name]["brand"], ui.BAR_WIN, *ui.HEAT_OK, *ui.HEAT_BAD]
    for f in fills:
        assert ui.contrast(ui.BAR_TEXT, f) >= 4.5, f"{f}: {ui.contrast(ui.BAR_TEXT, f):.2f}"
