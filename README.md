# EntityGuard

**Scalable entity resolution and duplicate detection across bibliographic databases.**
EntityGuard finds which records in **DBLP** (2,616 papers) and **Google Scholar** (64,263 records) describe the same publication. Comparing every pair would mean 2,616 × 64,263 = **168,112,008** comparisons; multi-pass blocking cuts that to about 0.3 % while keeping 98 % of the true matches, and a weighted multi-attribute score (or a small ML model) decides which candidate pairs match. Everything runs on a CPU (free Google Colab is enough).

Dataset: DBLP–Scholar benchmark (Leipzig University) in `data/`. The gold mapping (5,347 pairs) is used **only for evaluation**, never to produce predictions. Weights and thresholds are tuned on a validation split; all reported numbers are on an untouched test split (60 / 20 / 20 by DBLP record, seed 42).

## Headline results (test split)

| Method | Pairs scored | Precision | Recall | F1 |
|---|---|---|---|---|
| B1 — exact normalized title | hash join | 0.749 | 0.551 | 0.635 |
| B2 — fuzzy title (edit similarity), no blocking | 168,112,008 | 0.721 | 0.838 | 0.775 |
| M3 — weighted multi-attribute, no blocking | 168,112,008 | 0.880 | 0.841 | 0.860 |
| **M4 — weighted + multi-pass blocking** | 462,813 | 0.873 | 0.894 | **0.883** |
| M5 — Logistic Regression + blocking | 462,813 | 0.878 | 0.911 | 0.894 |
| M5b — Decision Tree + blocking | 462,813 | 0.900 | 0.902 | **0.901** |

- **Blocking:** 462,813 candidate pairs, reduction ratio **99.72 %**, blocking recall **98.07 %**, ~6–10 s. Year blocking alone would give 4.3 M pairs with only 45 % recall (54 % of Scholar records have no year).
- **Chosen on validation:** S = 0.45·title + 0.30·author + 0.00·venue + 0.25·year, with S_title = 0.6·edit + 0.4·token Jaccard, author = average best-match name similarity, threshold **T = 0.69**. Missing fields are dropped and the remaining weights renormalized.
- **Confidence bands:** S ≥ 0.95 high confidence (94 % are true matches on test) · 0.49–0.95 manual review · S < 0.49 likely non-match (5 true matches among 90,980 test pairs).
- **Scalability:** on 10 → 100 % of Scholar, blocking recall stays at 98–99 % and the reduction ratio at 99.7 %; time grows linearly.
- **Human review:** 90 borderline pairs reviewed in the web app; 81.1 % agreement with the gold mapping.

All numbers come from `outputs/summary.json` and the CSV files in `outputs/`.

## Repository layout

```
data/                         DBLP.csv (Latin-1), Scholar.csv (UTF-8 with BOM), DBLP-Scholar_perfectMapping.csv
src/
  normalize.py, similarity.py   Student 1 — normalization, Levenshtein, Jaccard, n-grams, record score
  compare_similarity.py         Student 1 — sample-based comparison of the similarity measures
  data_loading.py               dataset loading (correct encodings), dataset health, exact-title baseline B1
  evaluation/                   Student 3 — precision/recall/F1, blocking metrics, threshold sweeps,
                                confidence bands, train/val/test split, Student 1's study as tables
  matching/                     Student 3 — fast feature engine (same values as Student 1's functions),
                                weighted matcher (reads outputs/summary.json), reference blocking
  ml/                           Student 3 — Logistic Regression & Decision Tree training, model save/load
  api/                          Student 3 — FastAPI backend + Gradio interface (standalone web app)
experiments/EntityGuard_Student3.ipynb   the full experiment pipeline, end to end
project_khushahal.ipynb       Student 2 — SQLite import, indexing, blocking experiments (Colab)
tests/                        automated tests (pytest)
outputs/                      results: CSV tables, plots, summary.json, lr_model.joblib,
                              entityguard_app.db (web-app database), human_reviews.csv,
                              error_analysis_40_cases.csv
docs/                         report sections and notes for teammates
```

## Setup

Python 3.11 or newer (3.11 is tested). `requirements.txt` pins the exact package versions that were tested together.

```bash
python -m pip install -r requirements.txt
```

**Quickest way to run the app on any laptop.** Clone or download the repository, then:

- **Windows:** double-click `run_app.bat`.
- **macOS / Linux:** run `bash run_app.sh`.

The first run creates a private `.venv` folder and installs the requirements (a few minutes, needs internet). Later runs start in seconds and open the app in the browser.

## How to run each part

**Tests** (from the repo root):

```bash
python -m pytest -q tests
```

**Experiment notebook** — `experiments/EntityGuard_Student3.ipynb` runs everything: data analysis, normalization and similarity (Student 1's code), blocking, measure comparison, weight and threshold tuning, confidence bands, Logistic Regression / Decision Tree, method comparison with brute force, error analysis, scalability, the SQLite database and the web app. A full run takes about 5 minutes.

- *Google Colab:* open the notebook, `Runtime → Run all`. It mounts Google Drive and works in `My Drive/EntityGuard/` (put `DBLP-Scholar.zip` there or upload it when asked). If that folder has no `src/`, it clones this repository to `/content`, which needs the repository to be public.
- *Locally:* open it in Jupyter from the repo root or from `experiments/`; it finds `data/` and writes to `outputs/` automatically. (`EG_BASE=/some/folder` overrides the folder.)

**Standalone Gradio app** (`app.py`, Windows / macOS / Linux, Python 3.11+; or use the launchers above):

```bash
python app.py            # http://127.0.0.1:7860   (--open, --port 8080, --share, --host 0.0.0.0)
```

It is built only on `src/pipeline.py` (the notebook's normalization, similarity, blocking, scoring and ML logic in one self-contained module, which does not use `src/normalize.py` or `src/similarity.py`) and `src/evaluation/`. Tabs:

1. **Pair Matching**: type two records or load them by DBLP / Scholar id. Shows S_title, S_author, S_venue, S_year, the composite `S = 0.45·S_title + 0.30·S_author + 0.00·S_venue + 0.25·S_year` (renormalized over the fields both records have), the Logistic Regression and Decision Tree probabilities, which blocking passes would generate the pair, the decision (**Auto-Match** S ≥ 0.95, **Manual Review** 0.49–0.95, **Non-Match** < 0.49) and a written explanation. The weights and band limits are not set by hand: the app reads them from `outputs/summary.json`, which holds the values chosen on the validation split, so they update whenever the experiments are re-run.
2. **Entity Resolution**: pick a record from `data/DBLP.csv` or `data/Scholar.csv`. Configurable multi-pass blocking returns candidates from the other source, ranked and classified. You can also run the blocking configuration on all records (candidates, reduction ratio, blocking recall).
3. **Human Review Queue**: unreviewed Manual Review pairs, most uncertain first. Match / Non-Match / Needs Review decisions are written to `outputs/human_reviews.csv` immediately.
4. **Evaluation Dashboard** and 5. **Dataset Statistics**: the precomputed results in `outputs/` (results summary, method comparison, blocking passes, ML models, thresholds, error analysis, scalability), shown without re-running anything.

The app needs `data/` (loaded in the background, about 10 s) and from `outputs/`: `summary.json`, `lr_model.json`, `dt_model.json`, `entityguard_app.db` and the CSV/PNG results. At start-up it prints a warning that lists any of these that are missing.

The two ML models are stored as plain JSON numbers (`outputs/*_model.json`), so they give identical results with any scikit-learn version. The `.joblib` files are the original scikit-learn objects, kept for training only. Maintenance commands:

```bash
python -m src.pipeline verify          # rebuild all 462,813 candidates; compare with summary.json / method_comparison.csv
python -m src.pipeline train           # same, then retrain LR + DT and save outputs/dt_model.joblib + .json (about 1 minute)
python -m src.pipeline export-models   # convert outputs/*_model.joblib to the portable outputs/*_model.json
```

**FastAPI web application** (human review interface + dashboard, Sections 40, 46, 47):

```bash
uvicorn src.api.app:api --reload
```

Open http://localhost:8000/ui for the interface and http://localhost:8000/docs for the API. The pages are *Dataset Statistics, Blocking Configuration, Run Matching, Candidate Pairs, Review Pair, Evaluation* and *Performance Dashboard*. *Review Pair* shows a DBLP and a Scholar record side by side with each field's similarity, the overall score, the confidence band and the Logistic Regression probability, and the buttons **Confirm Match / Reject Match / Needs Review**. Decisions are saved to `outputs/human_reviews.csv`.

The app needs only saved artifacts: `outputs/entityguard_app.db`, `outputs/summary.json`, `outputs/lr_model.joblib` and `outputs/human_reviews.csv`. The app database is a 17 MB copy of the full experiment database (which is ~180 MB and not in git): all records, the gold mapping, and the match results with score ≥ 0.40. Any other pair is scored on demand with Student 1's `record_similarity`.

API (Section 47):

| Route | What it does |
|---|---|
| `GET /records/dblp`, `GET /records/scholar` | records, with `offset` / `limit` |
| `POST /blocking/run?scholar_fraction=0.5&passes=title_prefix,rare_tokens` | re-runs blocking on a Scholar sample |
| `POST /matching/run?threshold=0.69` | predicted matches and test precision / recall / F1 at that threshold |
| `GET /matches?band_name=Manual review&limit=50` | stored match results |
| `GET /pair/{dblp_id}/{scholar_id}` or `GET /pair?dblp_id=…&scholar_id=…` | both records and all similarity scores |
| `POST /review` | `{"dblp_id", "scholar_id", "decision", "note", "reviewer"}` |
| `GET /evaluation` | the experiment summary and the human-review agreement |

**Smaller entry points**

```bash
python -m src.data_loading              # dataset health summary + exact-title baseline
python -m src.evaluation.student1_study # Student 1's measure study on train+val gold pairs -> outputs/student1_*.csv
python src/compare_similarity.py        # Student 1's original comparison script
```

## Team

| Student | Area |
|---|---|
| Student 1 — Hemant | normalization, edit distance, Jaccard, n-grams, attribute similarities, record-level scoring |
| Student 2 — Khushahal | database, dataset import, indexing, blocking, candidate generation, scalability |
| Student 3 — Shivam | threshold evaluation, precision / recall / F1, Logistic Regression / Decision Tree, error analysis, web interface and dashboard |

Blocking in the experiments and the app is a reference implementation (`src/matching/blocking_reference.py`) until Student 2's blocking is available as a module under `src/blocking/`.
