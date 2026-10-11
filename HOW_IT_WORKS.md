# How it works: Travel Query Intent Classification Using NLP

This project has two parts that work together:

| Part | Where it runs | What it does |
|---|---|---|
| **The notebook** `notebooks/ATIS_Complete_Pipeline.ipynb` | Google Colab (or your own computer) | Reads the CSV queries, cleans them, trains and compares the models, tests the winner once, and saves everything in **`travel_nlp_run_bundle.zip`** |
| **Travel NLP Lab Studio** `streamlit_app.py` | Your computer (Streamlit, in the browser) | Opens that ZIP, re-checks it, and shows the data, results, mistakes and live demos. It never trains a model and never invents a number. |

```
CSV files ──► Google Colab notebook ──► travel_nlp_run_bundle.zip ──► Studio (Streamlit)
            (you upload the .ipynb        (one file with data, metrics,      (verifies, explains,
             and the CSVs by hand)          predictions, model, figures)       lets you try queries)
```

---

## Part 1 · Train in Google Colab (manual upload)

1. **Get the notebook.** It is `notebooks/ATIS_Complete_Pipeline.ipynb` in this folder. You can also download it from the Studio page **Data & Colab** (button *Download the notebook (.ipynb)*).
2. **Open Colab.** Go to <https://colab.research.google.com>, sign in, then **File → Upload notebook** and choose the `.ipynb` file.
3. **Run it.** Click **Runtime → Run all**. Cell 1 installs pandas, numpy, matplotlib and scikit-learn (about a minute the first time).
4. **Upload the CSVs when asked.** In section 2 a **Choose files** button appears. Upload the CSV files for this experiment.
   - For the ATIS benchmark: `train_atis.csv` and `test_atis.csv` (from `data/raw/`, or download them on the Studio's **Data & Colab** page).
   - Only `.csv` files are accepted. Raw CSVs are uploaded **here in Colab**, never into the Studio.
5. **Let every cell finish.** The notebook stops on its own if something is wrong (empty split, conflicting labels, train/test overlap, ambiguous columns) and says why. The **final quality gate** must print **PASS**.
6. **Download the result.** The last cell downloads **`travel_nlp_run_bundle.zip`** (allow the download if the browser asks).
7. **Open it in the Studio.** Sidebar → *Upload travel_nlp_run_bundle.zip*, or copy the ZIP into the `output` folder so it appears in the run list.

No Colab? `START.bat` option **3** runs the same notebook on your computer with the ATIS files (about 35 seconds); option **2** uses the CSVs you put in `my_data`.

### What the notebook does (section by section)

| Section | What happens |
|---|---|
| 1 Environment setup | installs the libraries, creates `data/`, `results/`, `model_card/` |
| 2 Upload the dataset | *Choose files* button; CSVs go to `data/raw/` |
| 3 Configuration | finds the text and label columns automatically (`text`, `query`, `user_query`… / `intent`, `label`, `category`…); stops if two columns are equally likely (`COLUMN_OVERRIDES` fixes it) |
| 4 Load, clean, combine | removes exact repeated queries; a file without a split is split here (stratified by intent, seed 42, test share = intents ÷ rows, kept between 20% and 50%) |
| 5 Data profile + quality gate | class balance, duplicates, train/test overlap; stops before training if the data is unusable |
| 6 Text preprocessing | lower-case, normalise spaces |
| 7 Model search | TF-IDF (word 1–2 grams, sublinear tf) with **Logistic Regression**, **Linear SVM** and **Multinomial Naive Bayes**; 5-fold stratified cross-validation on the **training data only**; the best mean **Macro-F1** wins |
| 8 Optional Sentence-BERT benchmark | off by default; never used to choose the model |
| 9 Final evaluation | the chosen model is fitted on all training data and tested **once** on the held-out test set |
| 10–12 Errors, subgroups, explanation | mistakes, short vs long queries, top weighted words per intent |
| 13 · 13b · 13c | try your own query; autocomplete (n-gram) model; classifier exported as plain JSON for the Model Lab's query classifier (only if it reproduces scikit-learn exactly) |
| 14–15 Figures + final quality gate | about ten figures; the gate must say PASS |
| 16 Package | writes `travel_nlp_run_bundle.zip` with a SHA-256 fingerprint for every file, and downloads it |

---

## Part 2 · Open the Studio (Windows)

1. Extract the project ZIP (right-click → *Extract All*).
2. Double-click **`START.bat`** and press **1**. The first time it creates a private Python environment and installs the packages (2–5 minutes).
3. Your browser opens at `http://127.0.0.1:8501`. The shipped **example run** (ATIS) opens automatically.
4. To stop: click the black window and press **Ctrl + C**.

`START.bat` menu: **1** open Studio · **2** train on your CSVs in `my_data`, then open · **3** train on the ATIS sample, then open · **4** run the self-test · **5** quit.

Mac/Linux or by hand: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`.

### Choosing a run (sidebar)

- **Open run** list: the example in `examples/`, runs trained on this computer (`output/`), and runs you uploaded this session.
- **Upload travel_nlp_run_bundle.zip**: opens a bundle from Colab. A file that is not a run bundle (a CSV, the project ZIP, a broken ZIP) is refused with a message, and the open run stays as it was.
- Under the list: **Verified** (every file matches its fingerprint) and the final model's name.
- Switching runs resets every page's choices, so one run's selections never show on another run.

---

## Every page and what it shows

**Start**

- **Home**: welcome page with project purpose, workflow and quick actions to Model Lab, Overview, Dataset and Data & Colab.
- **Overview**: the selected experiment, actual metrics, model comparison, dataset summary and expandable technical details at the bottom.
  - Under the query: the dataset label next to the bundled model's **live** answer.
  - *Another example* picks a new query.

- **Data & Colab**: the Colab steps above, plus downloads:
  - *Download the notebook (.ipynb)* and one download button for each raw CSV.
  - A table of the raw files: role, rows, intents, text/label column, repeated queries, entity tags.
  - How the raw ATIS files become the split: 4,978 → 4,623 training rows and 893 → 850 test rows. The repeats are removed, and so are queries that appear in both files.

**Explore**

- **Dataset**: the training and test rows inside the loaded run.
  - Rows per intent, training vs test (log scale), so you can see the imbalance.
  - Intents that appear only in the test set, and intents with fewer than 5 training rows.
  - The data-quality checks, plus near-duplicate look-alikes (cosine ≥ 0.9).
  - ATIS entity types and the busiest routes. These are dataset annotations, not predictions.
  - A row browser with a Training/Test switch and search.
- **Intents**:
  - A scatter of F1 against the number of test queries for every intent. It shows that small intents have unstable scores.
  - Pick an intent to see its precision, recall and F1, and its training/test rows.
  - Also shown: the words with the largest classifier weights, its most frequent words, what it is confused with, and real test queries (mistakes first).

**Evaluate**

- **Models**:
  - Cross-validation Macro-F1 (± SD) for the three candidates; the winner is in teal.
  - The final test metrics with bootstrap intervals, and the majority-class baseline.
  - Per-intent precision/recall/F1, which you can sort.
- **Confusion matrix**:
  - Rows = really meant, columns = model chose. Teal = correct, red = mistakes; colour = share of the row, number = count.
  - Choose a group (e.g. *meal → flight*) to see the actual queries. The most-confused pairs are listed below.
- **Predictions**:
  - Every test query with its dataset label, the model's choice and its score.
  - Filter by text, real intent, chosen intent, and right/wrong.
  - *Download these rows as CSV* saves exactly the filtered rows.
- **Error analysis**:
  - Every mistake, tagged with checkable factors: low score (least-sure 20%), short query (≤ 20th percentile of words), long query (≥ 80th), and shared wording.
  - Filter by factor. These are descriptions of the data, not the model's reasoning.
- **Confidence**:
  - Score histogram for right vs wrong answers. For Linear SVM the score is the *decision margin* (top score minus second score; not a probability). For Logistic Regression / Naive Bayes runs it is the highest probability, which is not calibrated.
  - Accuracy in five equal-sized score bands.
  - A slider that flags the least-sure queries for a person, showing the accuracy on the rest.
  - The least-sure queries are listed.

**Lab**

- **Model Lab → Query classifier** (formerly the Query lab page):
  - Type any travel question and press *Classify* (or *Use a real test query*). The bundled classifier scores it **live** on your computer.
  - Shown: the predicted intent, the margin, a certainty band (bands come from this run's own test margins), the number of known words, and the top-5 intent scores.
  - If the query is a test query, its stored benchmark result is shown too. If no word is in the vocabulary, it warns that the answer means "unknown".
  - *Does a small change flip the answer?* re-scores 6 small changes: uppercase, extra spaces, `???`, no punctuation, one typo, last word dropped.
- **Autocomplete**:
  - Type the start of a query. Click a suggestion to *finish the word*, add the *next word*, or take a *whole query*.
  - A side chart shows which intent the query is heading towards (the autocomplete model's hint, not the classifier's).
  - Below, the Studio re-scores the autocomplete model on every word of the test queries and checks the result against the numbers the notebook stored (top-1/3/5 and baseline).
- **Intent Map**: removed from the user-facing navigation (since Studio 4.3.0; unchanged in 4.3.1) (the analysis code remains in the backend).

**Evidence**

- **Compare runs**:
  - Pick another bundle (or upload one) as run B. You see accuracy, Macro-F1 and its interval, test size, intents, autocomplete top-3, and the per-intent F1 change.
  - Verdict: *Improvement* / *Possible regression* (Macro-F1 moved ≥ 0.5 points) / *No clear change*.
  - If the two runs used different data, the page says so.
- **Files & checks**:
  - The fingerprint check for every file, metrics recomputed from the stored predictions, the notebook's quality gate, and run information (ID, date, seed, selection method, dataset fingerprint).
  - The notebook's own figures, every file in the bundle (with download), and the model card.

---

## Part 3 · Docker and GitHub

- **Docker:** `docker build -t travel-nlp-studio .` then `docker run --rm -p 8501:8501 travel-nlp-studio` and open <http://localhost:8501>. Or `docker compose up --build`, which also shares your `output` folder with the container.
- **GitHub:** create an empty repository, then run `git init`, `git add .`, `git commit -m "…"`, `git branch -M main`, `git remote add origin <url>` and `git push -u origin main`. Two workflows then run by themselves:
  - `tests.yml`: the tests, the notebook end to end, and the browser checks.
  - `docker.yml`: builds the image and smoke-tests it; on `main` it publishes the image to `ghcr.io/<user>/<repo>`.
- Full commands are in `README.md`.

## Documents

- `docs/Project_Report_Travel_Query_Intent_Classification.docx`: the project report in the SAII NLP format. Fill in the PRNs, the contributions and the GitHub link.
- `docs/Model_Card_Travel_Intent_Classifier.pdf` / `.docx` and `MODEL_CARD.md`: the model card.

## What the Studio guarantees (and what it does not)

- Every number comes from the loaded bundle and is recomputed where possible: accuracy and Macro-F1 from the stored predictions, and the autocomplete scores from the test queries. If a number cannot be checked, the page says so.
- Missing parts show **"Not available for this run"** instead of zeros. Examples: no entity tags in the hotel/car/rail sets, no classifier in an old bundle, no wrong answers.
- Stored benchmark results (the notebook's single test run) and **live inference** (Model Lab → Query classifier) are labelled separately.
- 100% local: no cloud, no AI service, no internet needed. Fonts and plotly.js are bundled in `static/`.
- SHA-256 fingerprints show the files were not changed after the notebook wrote them. They do not prove who made them.

## Folder layout

| Path | Contents |
|---|---|
| `streamlit_app.py` | the app: page list, sidebar, run picker |
| `studio/` | `core.py` (link to the backend), `ui.py` (theme design), `views/` (one file per page) |
| `app/` | analysis backend: `engine.py` (bundle verification), `analytics.py`, `autocomplete_plus.py`, `server.py` (functions + optional local JSON API) |
| `.streamlit/config.toml`, `static/` | theme, local fonts, plotly.js, logo |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/` | container image and GitHub automation |
| `docs/`, `MODEL_CARD.md` | project report and model card |
| `notebooks/` | the Colab notebook |
| `data/raw/` | `train_atis.csv`, `test_atis.csv`, `hotel_booking_queries.csv`, `car_rental_queries.csv`, `train_rail_queries.csv` |
| `examples/` | the example ATIS run bundle |
| `output/` | bundles trained on this computer appear here |
| `my_data/` | put your own CSVs here for START.bat option 2 |
| `tools/` | `run_notebook_local.py` (run the notebook without Colab), `compare_bundles.py` (check two bundles match) |
| `tests/` | `pytest` self-test (START.bat option 4) and `tests/browser/ui_check.py` (clicks through the running app) and `tests/browser/theme_audit.py` (measures colour contrast on the rendered pages) |


## Studio 4.3.1 navigation and themes

- Home is a welcome/workflow page; Overview is the experiment dashboard.
- Model Lab contains Query Classifier and Autocomplete.
- The compact sidebar includes seven themes: Midnight Navy (default), Ocean Blue, Forest Green, Royal Purple, Sunset Orange, Rose Pink and Slate Teal. Light, Dark and System appearance modes are available (System follows the operating system). Contrast is checked by `tests/test_themes.py` and `tests/browser/theme_audit.py`.
- The Intent Map is no longer a user-facing page.
- The overview includes a collapsed Experiment Details & Technical Information section at the bottom.
