"""Real-browser check of the Streamlit app: clicks every page and the important controls, and checks what appears.

1. Start the app:   streamlit run streamlit_app.py --server.port 8501
2. Then run:        pip install playwright && python -m playwright install chromium
                    python tests/browser/ui_check.py [http://127.0.0.1:8501]
"""
import asyncio
import json
import pathlib
import os
import sys

from playwright.async_api import async_playwright

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8501").rstrip("/")
ROOT = pathlib.Path(__file__).resolve().parents[2]
HOTEL = ROOT / "tests" / "fixtures" / "hotel_run_bundle.zip"
PAGES = [("Home", "Understand what travellers"), ("Overview", "Travel Query Intent"), ("Model Lab", "Model Lab"), ("Data & Colab", "Data & Google Colab"),
         ("Dataset", "Dataset"), ("Intents", "Intents"), ("Models", "Models"), ("Confusion matrix", "Confusion matrix"), ("Predictions", "Predictions"),
         ("Error analysis", "Error analysis"), ("Confidence", "Confidence"), ("Compare runs", "Compare runs"), ("Files & checks", "Files & checks")]
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))


async def idle(pg, extra=600):
    await pg.wait_for_timeout(300)
    for _ in range(120):
        if await pg.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        await pg.wait_for_timeout(250)
    await pg.wait_for_timeout(extra)


async def main_text(pg):
    return await pg.locator('[data-testid="stMain"]').inner_text()


async def nav(pg, label):
    await pg.locator('[data-testid="stSidebarNav"] a', has_text=label).first.click()
    await idle(pg)


async def seg(pg, name):
    """Click an option of a Streamlit segmented control."""
    await pg.get_by_role("radio", name=name, exact=True).first.click()
    await idle(pg)


async def pick(pg, label, option):
    """Choose an option in a Streamlit selectbox identified by its label."""
    box = pg.get_by_role("combobox", name=label).first
    await box.click()
    await box.fill(option.split(" ")[0])
    await pg.get_by_role("option").filter(has_text=option).first.click()
    await idle(pg)


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None, args=["--no-sandbox"])
        ctx = await b.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        pg = await ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(BASE); await idle(pg, 1500)

        # ---- every page opens, shows its heading, no Python exception
        for label, heading in PAGES:
            await nav(pg, label)
            t = await main_text(pg)
            exc = await pg.locator('[data-testid="stException"]').count()
            cur = await pg.locator('[data-testid="stSidebarNav"] a[aria-current="page"]').inner_text()
            check(f"page '{label}' opens, is highlighted, no error", heading in t and exc == 0 and label in cur and "NaN" not in t, f"exc={exc} cur={cur}")

        # ---- sidebar colour theme + appearance: change every page, survive navigation and reload
        await nav(pg, "Overview")
        brand = lambda: pg.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--brand').trim()")
        appearance = lambda: pg.evaluate("document.querySelector('style[data-appearance]').dataset.appearance")
        b0 = await brand()
        await pick(pg, "Colour theme", "Forest Green")
        b1 = await brand()
        await nav(pg, "Models")
        b2 = await brand()
        await pg.reload(); await idle(pg, 1500)
        b3 = await brand()
        check("theme: the sidebar picker changes the colours", b0 != b1, f"{b0}->{b1}")
        check("theme: the choice applies on other pages and after reload", b1 == b2 == b3, f"{b1} {b2} {b3}")
        await seg(pg, "Light")
        check("appearance: Light applies and the page stays readable", await appearance() == "light" and "Models" in await main_text(pg))
        await seg(pg, "System")
        check("appearance: System applies", await appearance() == "system")
        await seg(pg, "Dark"); await pick(pg, "Colour theme", "Midnight Navy")
        check("theme: back to Midnight Navy / Dark", (await brand()).upper() == "#60A5FA" and await appearance() == "dark", await brand())

        # ---- home buttons
        await nav(pg, "Home")
        await pg.get_by_role("button", name="Open Model Lab").click(); await idle(pg)
        check("home: 'Open Model Lab' opens Model Lab", "Query classifier" in await main_text(pg))
        await nav(pg, "Home"); await pg.get_by_role("button", name="Explore Overview").click(); await idle(pg)
        check("home: 'Explore Overview' opens Overview", "Travel Query Intent" in await main_text(pg))
        await nav(pg, "Home"); await pg.get_by_role("button", name="Browse dataset").click(); await idle(pg)
        await pg.wait_for_function("document.querySelector('.nf-h1') && document.querySelector('.nf-h1').innerText.trim() === 'Dataset'", timeout=30000)
        check("home: 'Browse dataset' opens Dataset", "Dataset" in await pg.locator(".nf-h1").first.inner_text() and await pg.locator('[data-testid="stException"]').count() == 0)
        await nav(pg, "Home"); await pg.get_by_role("button", name="Open Data & Colab").click(); await idle(pg)
        check("home: 'Open Data & Colab' opens Data & Colab", "Run the notebook in Google Colab" in await main_text(pg))

        # ---- overview buttons
        await nav(pg, "Overview")
        q1 = await pg.locator(".nf-q").inner_text()
        await pg.get_by_role("button", name="Another example").click(); await idle(pg)
        check("overview: 'Another example' shows a different real query", (await pg.locator(".nf-q").inner_text()) != q1)
        live = await pg.locator(".nf-card").first.inner_text()
        check("overview: example shows dataset label and live model answer", "dataset label" in live and "live model" in live)
        await pg.get_by_role("button", name="Explore the results").click(); await idle(pg)
        check("overview: 'Explore the results' opens Models", "Model selection" in await main_text(pg))
        await nav(pg, "Overview")
        await pg.get_by_role("button", name="Train in Google Colab").click(); await idle(pg)
        check("overview: 'Train in Google Colab' opens Data & Colab", "Run the notebook in Google Colab" in await main_text(pg))
        await nav(pg, "Overview")
        await pg.get_by_text("Experiment Details & Technical Information").click(); await idle(pg)
        check("overview: technical details expand", "Technical values are read from the selected experiment bundle" in await main_text(pg))

        # ---- data & colab downloads
        await nav(pg, "Data & Colab")
        async with pg.expect_download() as d:
            await pg.get_by_role("button", name="Download the notebook (.ipynb)").click()
        dl = await d.value
        nb = json.loads(pathlib.Path(await dl.path()).read_text(encoding="utf-8"))
        check("colab: notebook download is the real .ipynb", dl.suggested_filename.endswith(".ipynb") and len(nb["cells"]) > 40)
        async with pg.expect_download() as d:
            await pg.get_by_role("button", name="train_atis.csv").click()
        dl = await d.value
        same = pathlib.Path(await dl.path()).read_bytes() == (ROOT / "data" / "raw" / "train_atis.csv").read_bytes()
        check("colab: train_atis.csv download matches data/raw exactly", same)
        t = await main_text(pg)
        check("colab: raw-to-split numbers shown (4,978 → 4,623 · 893 → 850)", all(x in t for x in ("4,978", "4,623", "893", "850")))

        # ---- dataset: toggle + search
        await nav(pg, "Dataset")
        await seg(pg, "Training")
        check("dataset: Training file selected", "of 4,623 rows" in await main_text(pg))
        await pg.get_by_label("Search the text").fill("denver"); await pg.keyboard.press("Enter"); await idle(pg)
        t = await main_text(pg)
        check("dataset: search filters rows", "of 4,623 rows" in t and not t.count("4,623 of 4,623"))

        # ---- intents: pick another intent
        await nav(pg, "Intents")
        await pick(pg, "Intent (weakest first)", "abbreviation")
        t = await main_text(pg)
        check("intents: choosing 'abbreviation' shows its 100% F1 and real queries", "100.0%" in t and "what is" in t.lower())

        # ---- confusion: open a group
        await nav(pg, "Confusion matrix")
        await pick(pg, "Really meant → model chose", "meal → flight")
        val = await pg.get_by_role("combobox", name="Really meant → model chose").input_value()
        tables = await pg.locator('[data-testid="stDataFrame"]').count()
        check("confusion: 'meal → flight' selected and its queries listed", val.startswith("meal → flight") and tables >= 1, val)

        # ---- predictions: filter + export
        await nav(pg, "Predictions")
        await seg(pg, "wrong")
        check("predictions: 'wrong' shows the 39 mistakes", "39 of 850 queries match" in await main_text(pg))
        await pg.get_by_label("Search the query text").fill("denver"); await pg.keyboard.press("Enter"); await idle(pg)
        async with pg.expect_download() as d:
            await pg.get_by_role("button", name="Download these rows as CSV").click()
        lines = pathlib.Path(await (await d.value).path()).read_text(encoding="utf-8").splitlines()[1:]
        check("predictions: CSV holds only the filtered rows (wrong + 'denver')", lines and all("denver" in l.lower() and ",False" in l for l in lines), f"{len(lines)} rows")

        # ---- error analysis filter
        await nav(pg, "Error analysis")
        await pg.wait_for_selector(".nf-note.warn", timeout=30000); await pg.wait_for_timeout(800)
        before = await pg.locator(".nf-note.warn").count()
        await seg(pg, "Short query")
        after = await pg.locator(".nf-note.warn").count()
        check("errors: factor filter narrows the list", 0 < after < before, f"{before}->{after}")

        # ---- confidence slider
        await nav(pg, "Confidence")
        k1 = await pg.locator(".nf-card", has_text="Accuracy on the rest").inner_text()
        sl = pg.locator('[data-testid="stSlider"] input[type="range"]').first
        await sl.focus(); await pg.keyboard.press("End"); await idle(pg)
        k2 = await pg.locator(".nf-card", has_text="Accuracy on the rest").inner_text()
        check("confidence: moving the slider updates accuracy and coverage", k1 != k2)

        # ---- Model Lab: query classifier tab, then autocomplete tab
        await nav(pg, "Model Lab")
        await pg.get_by_label("Travel query", exact=True).fill("what is the cheapest fare from boston to denver"); await pg.keyboard.press("Enter"); await idle(pg)
        await pg.get_by_role("button", name="Classify").click(); await idle(pg)
        t = await main_text(pg)
        check("model lab: live prediction for a fare question is 'airfare'", "airfare" in t and "Live inference" in t)
        old = await pg.get_by_label("Travel query", exact=True).input_value()
        await pg.get_by_role("button", name="Use a real test query").click(); await idle(pg)
        check("model lab: 'Use a real test query' loads another query", (await pg.get_by_label("Travel query", exact=True).input_value()) != old)
        await pg.get_by_role("tab", name="Autocomplete").click(); await idle(pg)
        await pg.get_by_label("Start typing a travel query", exact=True).fill("show me flights "); await pg.keyboard.press("Enter"); await idle(pg)
        await pg.get_by_role("button", name="from", exact=True).click(); await idle(pg)
        v = await pg.get_by_label("Start typing a travel query", exact=True).input_value()
        check("autocomplete: clicking 'from' finishes the word", v == "show me flights from ", repr(v))
        nxt = pg.locator('[data-testid="stButton"] button').filter(has_text="boston").first
        await nxt.click(); await idle(pg)
        v2 = await pg.get_by_label("Start typing a travel query", exact=True).input_value()
        check("autocomplete: clicking a next word appends it", v2 == "show me flights from boston ", repr(v2))
        check("autocomplete: accuracy matches stored numbers", "matches the notebook's stored numbers" in await main_text(pg))

        # ---- files
        await nav(pg, "Files & checks")
        t = await main_text(pg)
        check("files: bundle verified, metrics recomputed, check tables shown", "Verified" in t and "Recomputed" in t
              and await pg.locator('[data-testid="stDataFrame"]').count() >= 3)
        check("files: notebook figure shown", await pg.locator('[data-testid="stImage"] img').count() > 0)

        # ---- compare with the hotel run (upload B)
        await nav(pg, "Compare runs")
        inputs = pg.locator('[data-testid="stMain"] input[type="file"]')
        if await inputs.count():
            await inputs.first.set_input_files(str(HOTEL)); await idle(pg, 1500)
            t = await main_text(pg)
            check("compare: second run compared (different data flagged)", "different data" in t and "Macro-F1:" in t)

        # ---- open another run from the sidebar: pages adapt
        await pg.locator('[data-testid="stSidebar"] input[type="file"]').set_input_files(str(HOTEL)); await idle(pg, 1500)
        await nav(pg, "Dataset")
        t = await main_text(pg)
        side = await pg.locator('[data-testid="stSidebar"]').inner_text()
        check("upload: hotel run opens and pages follow it", "book_hotel" in t and "hotel_run_bundle.zip" in side and "4,623" not in t, side[-160:])
        check("upload: hotel run has no entity annotations (shown as not available)", "Not available for this run" in t)

        await pg.locator('[data-testid="stSidebar"] input[type="file"]').set_input_files(
            files=[{"name": "broken.zip", "mimeType": "application/zip", "buffer": b"not a zip"}]); await idle(pg, 1000)
        side = await pg.locator('[data-testid="stSidebar"]').inner_text()
        check("upload: a broken ZIP is refused with a readable message", "Invalid ZIP file" in side or "Could not read" in side, side[-200:])
        await nav(pg, "Dataset"); t2 = await main_text(pg)
        check("upload: the open run is unchanged after a refused file", "book_hotel" in t2 and "4,623" not in t2, side[-160:])

        check("no JavaScript errors", not errs, "; ".join(errs[:3]))
        await b.close()
    n = sum(ok for _, ok in results)
    print(f"\n{n}/{len(results)} browser checks passed")
    sys.exit(0 if n == len(results) else 1)


asyncio.run(main())
