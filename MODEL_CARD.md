# Model Card: Travel Query Intent Classifier (TravelIntent-SVM v1)

*Travel Query Intent Classification Using NLP · NLP-34 · SAII NLP course, 2026-27.* All numbers come from one verified run (run ID `87f82a74`, seed 42) of
`notebooks/ATIS_Complete_Pipeline.ipynb` on ATIS (the example bundle in `examples/`). Travel NLP Lab Studio recomputed them from the stored predictions.
PDF and Word versions: `docs/Model_Card_Travel_Intent_Classifier.pdf` / `.docx`.

## 1. Model details
| Field | Description |
|---|---|
| Model name | TravelIntent-SVM v1 |
| Task | Single-label intent classification of English travel queries (22 intents seen in training) |
| Model | TF-IDF (word 1–2 grams, sublinear tf, 6,772 features) + Linear SVM (LinearSVC, class_weight = balanced) |
| Model selection | Best of Logistic Regression / Linear SVM / Multinomial NB by 5-fold stratified CV Macro-F1 on training data only (Linear SVM 0.784 ± 0.033) |
| Training data | ATIS training file: 4,623 queries after removing repeats and train/test overlap (raw 4,978); “flight” = 74.1% |
| Test data | ATIS test file: 850 held-out queries (raw 893), 20 intents; evaluated once |
| Preprocessing | Lower-casing, whitespace normalisation, repeat removal, conflict and overlap checks; stop words kept; no stemming |
| Libraries | Python 3.12.3, scikit-learn 1.8.0, pandas 3.0.2, NumPy 2.4.4; app: Streamlit, Plotly |
| Developed by | NLP-34 team, SAII — NLP course, October 2026 |

## 2. Intended use
- **Intended:** routing short English travel questions (flights, fares, airlines, ground transport, airport/aircraft info) to the right service; teaching an explainable classical NLP pipeline.
- **Out of scope:** binding bookings or refunds without human confirmation; non-English or code-mixed text (e.g. Hinglish); hotel/rail/car questions; intents never seen in training; multi-turn context.

## 3. Overall performance
95.4% accuracy and 0.683 Macro-F1 (95% bootstrap interval 0.589–0.714) on 850 test queries.

| Model | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Majority-class baseline | 72.1% | — | — | 0.042 | — |
| TravelIntent-SVM v1 | 95.4% | 0.726 | 0.684 | 0.683 | 0.950 |

## 4. Performance across text subgroups
**Key finding:** on long queries (> 11 words) Macro-F1 falls from 0.849 to 0.438 and accuracy drops by 6.1 points.

| Subgroup | Definition | n | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Short queries | ≤ 11 words (training median) | 560 | 97.5% | 0.849 |
| Long queries | > 11 words | 290 | 91.4% | 0.438 |
| Questions | starts with what/which/how/can/does/is… | 255 | 93.3% | 0.807 |
| Request fragments | any other start | 595 | 96.3% | 0.608 |
| Near-duplicates of training | cosine ≥ 0.9 to a training query | 18 | 100.0% | — |
| All other test queries | no near-duplicate | 832 | 95.3% | — |

## 5. Explainability check
**Top Linear SVM weights:** airfare: fares (+3.14), fare (+2.93), fare from (+2.53), ticket (+2.33), airfare (+2.08). airline: airline (+3.24), airlines (+3.16), what airlines (+2.32), the airlines (+2.09), which airlines (+1.94). ground_service: ground (+2.70), transportation (+2.47), ground transportation (+1.79), car (+1.72), airport (+1.12).

**Exact word contributions on a mistake** (a linear model's score is the sum of TF-IDF value × weight plus a bias, so this is exact, not a LIME approximation):
“which different airlines go from las vegas to new york city” is really *airline* but was predicted *flight*, with the two scores only 0.0008 apart. The words favour airline
(airlines -0.72, which -0.36, airlines go -0.25), but the “flight” class bias is 0.51 higher, so imbalance tips the near-tie.

## 6. Limitations and ethical considerations
- Rare and unseen intents: 4 test intents never occur in training; meal recall 0.333; Macro-F1 far below accuracy.
- Data bias: early-1990s US domestic air travel; Indian, international, informal or code-mixed queries are likely to be handled worse.
- No word order, negation or context beyond bigrams; typos lower confidence.
- Small test counts for many intents; 18 near-duplicate test queries (accuracy on the rest 95.3%).
- No personal data in ATIS; a deployed system would need consent and retention rules.

## 7. Recommendations
- More or augmented data for rare and combined intents, or multi-label classification.
- Send low-confidence predictions to a person: flagging the least-certain 20% (margin < 1.12) leaves 99.0% accuracy on the rest.
- Compare a contextual model (DistilBERT / Sentence-BERT) and character n-grams on the same subgroup audit.
- Add an “unknown / other” intent.
