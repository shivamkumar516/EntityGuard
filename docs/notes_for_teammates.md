# Notes for teammates (from Student 3's integration work)

None of your files were changed. These are things found while plugging your code into the full pipeline.

## Hemant (Student 1) — `src/normalize.py`, `src/similarity.py`

1. **Accented letters are dropped, not folded.** `_basic_clean` lowercases and then replaces every character outside `[a-z0-9]` with a space, so `"J Wäsch"` becomes `"j w sch"` and `"Dalí"` becomes `"dal"`. This affects 7.5 % of DBLP author lists and 3.9 % of Scholar author lists. Suggested fix, before the regex in `_basic_clean`:

   ```python
   import unicodedata
   text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
   ```

   Garbled Scholar text such as `Floridaâ??s` could also be cleaned with `re.sub(r"â[^\w\s]*", " ", text)` first.

2. **Default weights and α are still the example values** (`WEIGHTS` = 0.50 / 0.30 / 0.10 / 0.10, `alpha=0.4`). The values chosen on the validation split are in `outputs/summary.json`: α = 0.6, weights (title, author, venue, year) = (0.45, 0.30, 0.00, 0.25), author method `best_match`. With your example settings the full pipeline reaches test F1 0.833; with the tuned ones 0.883 (`outputs/student1_settings_in_pipeline.csv`).

3. **Your comparison script reports much higher F1 than the real task** (≈ 0.95 vs ≈ 0.83 for the same settings), because its sample has one non-match per gold pair, while the real candidate set has about 100 non-matches per match. Its *ranking* of measures holds up (best-match authors beat token Jaccard, α ≈ 0.6–0.8 is best); its ranking of *weights* does not ("title heavy" is best on the sample but second-worst in the pipeline). It also uses all gold pairs, including the test split — `src/evaluation/student1_study.py` reruns it on train + validation pairs only.

4. Optional: a venue token-Jaccard without generic words (`proceedings`, `international`, …) plus an acronym token exists as a Student 3 extension in `src/matching/features.py` (`venue_tokens`). With the current normalization, validation gives venue weight 0 anyway, because Scholar venues are very noisy.

## Khushahal (Student 2) — `project_khushahal.ipynb`

1. **Cell 5 overwrites `candidate_pairs.csv`** with the year-prefix pairs only, so the de-duplication demo after it removes 0 duplicates. Write each pass to its own file, or de-duplicate the union of all passes.
2. **Hard-coded `/content/drive/MyDrive/...` paths** — the notebook only runs in one person's Drive. A `BASE` folder variable (as in the experiment notebook) would make it portable.
3. **No before/after timing for the indexes:** time the same lookup query with and without `CREATE INDEX`, so the report can show what indexing buys.
4. **Blocking module integration (implemented in this update).** `src/blocking/candidate_generation.py` is now the canonical blocking implementation imported by `src/pipeline.py`. The former in-pipeline `BlockingIndex` has been moved into the Student 2 module while preserving the query-time methods required by the app. The public `candidate_pairs(dblp, scholar)` interface remains available:

   ```python
   # src/blocking/__init__.py
   def candidate_pairs(dblp, scholar) -> set[tuple[str, str]]:   # {(dblp_id, scholar_id), ...}
   ```

   The module also exposes `run_blocking` and `generate_candidate_details` for experiments and per-rule diagnostics. The old `src/matching/blocking_reference.py` is retained only for regression comparison.
