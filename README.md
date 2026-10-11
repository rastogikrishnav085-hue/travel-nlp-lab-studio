# Travel Query Intent Classification Using NLP

**Travel NLP Lab Studio 4.3.1.** A Google Colab notebook trains and evaluates the intent classifier. A local Streamlit app opens the
finished run, re-checks it, and explains the results: data, models, confusion matrix, mistakes, confidence, a unified Model Lab with live query classification and autocomplete. Everything runs on your computer; no cloud or AI services are used.

**Full step-by-step guide: [HOW_IT_WORKS.md](HOW_IT_WORKS.md)**

## Quick start (Windows)
1. Extract this ZIP (right-click → Extract All).
2. Double-click `START.bat`, press `1`. The first run installs packages (2–5 minutes) and opens your browser.
3. The example ATIS run opens automatically. Use the sidebar to open another run.

Other systems: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`.

## Train in Google Colab (manual upload)
1. colab.research.google.com → **File → Upload notebook** → `notebooks/ATIS_Complete_Pipeline.ipynb`.
2. **Runtime → Run all**. In section 2 click **Choose files** and upload your CSVs (ATIS: `data/raw/train_atis.csv`, `data/raw/test_atis.csv`).
3. The last cell downloads `travel_nlp_run_bundle.zip`. In the Studio: sidebar → *Upload travel_nlp_run_bundle.zip*.

Without Colab: `START.bat` option 3 (ATIS) or 2 (your CSVs in `my_data`) runs the same notebook locally.
`python tools/compare_bundles.py a.zip b.zip` checks that two runs (e.g. Colab vs local) produced identical results.

## Method in one paragraph
Queries are lower-cased and whitespace-normalised, turned into TF-IDF features (word 1–2 grams), and classified with
Logistic Regression, Linear SVM and Multinomial Naive Bayes. Exact repeated queries and train/test overlap are removed first.
The model with the best 5-fold cross-validation Macro-F1 **on the training data only** is fitted on all training data and
tested once on the held-out test set. On the shipped ATIS example, TF-IDF + Linear SVM reaches 95.4% accuracy and 68.3% Macro-F1
on 850 test queries (20 intents). Macro-F1 is lower because several intents have only a handful of examples, and 4 test intents
never appear in training.

## Checks
- `pytest -q` (START.bat option 4): 151 tests: 94 covering bundle verification, recomputed metrics, ZIP safety (oversized and
  compression-bomb files), backend analytics, and every Streamlit page rendered on two different datasets, plus 57 exact
  colour-contrast tests (`tests/test_themes.py`) for all 7 themes in Light and Dark.
- `python tests/browser/ui_check.py` against the running app: 51 real-browser checks that click every page and control, including the colour theme picker.
- `.github/workflows/tests.yml` runs the tests, the notebook end to end, and the browser checks on every push.

## Run with Docker
```
docker build -t travel-nlp-studio .
docker run --rm -p 8501:8501 travel-nlp-studio          # open http://localhost:8501
# or, keeping your own bundles from ./output:
docker compose up --build
```
The image (`python:3.12-slim`) installs `requirements.txt`, runs as a non-root user and has a health check on `/_stcore/health`.
Everything it needs (example run, raw data, notebook, fonts, plotly.js) is inside the image, so it runs offline.

## Put it on GitHub
1. Create an empty repository on github.com (no README, no .gitignore), e.g. `travel-query-intent-nlp`.
2. In this folder:
   ```
   git init
   git add .
   git commit -m "Travel Query Intent Classification Using NLP"
   git branch -M main
   git remote add origin https://github.com/<your-user>/travel-query-intent-nlp.git
   git push -u origin main
   ```
3. GitHub Actions then runs two workflows automatically (tab **Actions**):
   - `tests.yml`: the 151 tests, the notebook end to end, and the 51 browser checks.
   - `docker.yml`: builds the Docker image, starts it once as a smoke test and, on `main`, publishes it to
     **GitHub Container Registry** as `ghcr.io/<your-user>/travel-query-intent-nlp:latest`.
     Anyone can then run `docker run -p 8501:8501 ghcr.io/<your-user>/travel-query-intent-nlp:latest`
     (make the package public under *Packages → Package settings* if the repository is private).
4. Optional: open the notebook straight from GitHub in Colab: colab.research.google.com → **GitHub** tab → paste the
   repository URL → pick `notebooks/ATIS_Complete_Pipeline.ipynb`.

`.gitignore` keeps `.venv`, caches and `output/` out of the repository; `.gitattributes` keeps `START.bat` in Windows line endings.

## Documents
- `docs/Project_Report_Travel_Query_Intent_Classification.docx`: the project report in the SAII NLP report format.
- `docs/Model_Card_Travel_Intent_Classifier.pdf` (and `.docx`): one model card in the course sample format. `MODEL_CARD.md` has the same content for GitHub.

## Version history
- **4.3.1**: the theme "Clean Light" is renamed "Slate Teal" (it is dark navy in Dark mode, so the old name was misleading). slider values and the System-mode logo text are now readable in every appearance. the sidebar no longer shows the "Verified" and model badges on every page (only a warning if a check fails; a brief toast when a bundle opens). fixed a crash on every page in 4.3.0 (theme CSS was missing two colour values). Light and System appearance now work for
  every native Streamlit widget (sidebar, buttons, selects, uploader, tabs, tables); stronger control borders; chart labels and
  heat-map numbers readable on any background; Sunset Orange light accent darkened to reach 4.5:1. New tests: `tests/test_themes.py`
  (exact contrast for every theme) and `tests/browser/theme_audit.py` (measures the rendered pages in a real browser).
- **4.3.0**: Added a separate Home welcome page and Overview dashboard; seven themes with Light/Dark/System controls; compact sidebar controls; upload read limit; technical details at the bottom of Overview; Intent Map removed from the UI.
- **4.1.0**: initial Streamlit conversion, earlier theme picker,
  Docker image + docker compose, GitHub Actions for tests and the Docker image (GHCR), project report and model card in `docs/`.
- **4.0.0**: the interface is now a Streamlit app in the "Midnight Navy" dark design. New **Data & Colab** page (raw CSVs, notebook
  download, Colab steps, raw → split numbers). The earlier release included an intent map. The old HTML front end, guided demo,
  presentation mode and command menu were removed. The analysis backend and every calculation are unchanged; every page is tested.
- 3.8.0: dark mode, palette and text-size fixes in the former HTML front end.
- 3.x: bundle verification, Query lab, leakage checks, entity analysis, error lab, confidence, run comparison, intent map.


## Studio 4.3.0 design and reliability update

- Added a dedicated Home welcome page and retained the previous experiment dashboard as Overview.
- Added seven colour themes and Light/Dark/System appearance controls in the sidebar. Theme colours are defined centrally in `studio/ui.py`; contrast is verified by the tests listed under Accessibility validation.
- Removed the Intent Map page from the user-facing application.
- Added a collapsible Experiment Details & Technical Information section at the bottom of Overview.
- Kept Google Colab as the training/evaluation pipeline and FastAPI as the local bundle analysis backend. Model calculations and preprocessing were not intentionally changed.

### Accessibility validation

The 4.3.0 release shipped without a browser run and crashed on start; 4.3.1 was checked in a real browser (see `tests/browser/theme_audit.py`).

Targets: WCAG AA, 4.5:1 for normal text, 3:1 for large text, control borders and the focus ring.

- `tests/test_themes.py` computes the exact contrast of every theme definition (no browser needed).
- `tests/browser/theme_audit.py` opens the pages in a real browser and measures every rendered text element and control border in Light, Dark and System (with the operating system set to light and to dark). Run it with the app running: `python tests/browser/theme_audit.py "Midnight Navy" Light,Dark,SysLight,SysDark all`.
- Limits: tables are drawn on a canvas, so their text is checked by eye (they are inverted for a light page), not measured. Gradient headings are not measured. System appearance follows the operating system.
