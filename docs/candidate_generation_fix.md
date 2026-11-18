# Candidate-generation implementation update

## What was fixed

1. Added `src/blocking/` as a reusable package. Its public interface is
   `candidate_pairs(dblp, scholar)`, returning a set of `(dblp_id, scholar_id)` pairs.
2. Implemented four complementary passes: `title_prefix`, `rare_tokens`,
   `char_prefix`, and `rare_char6`.
3. Combined the passes into one candidate union. Each pair is stored once;
   `generate_candidate_details()` retains every rule that generated it.
4. Replaced the old notebook Cell 5 behaviour. It no longer reruns only
   year-prefix blocking or overwrites the multi-pass candidate file.
5. Changed notebook Cells 5–6 to write per-pass rows to
   `outputs/candidate_pairs_by_pass.csv`, then deduplicate them into
   `outputs/candidate_pairs.csv` while preserving rule labels.
6. Kept the previous `src.matching.blocking_reference` API as a compatibility
   adapter. The app's existing caller receives the same integer-key format,
   while the implementation now comes from `src.blocking`.

## Validation on the repository datasets

| Measure | Result |
|---|---:|
| DBLP records | 2,616 |
| Scholar records | 64,263 |
| Possible cross-source pairs | 168,112,008 |
| Candidate pairs after union | 462,813 |
| Gold-standard matches found | 5,244 / 5,347 |
| Blocking recall | 98.0737% |
| Candidate reduction ratio | 99.7247% |
| Duplicate rule hits merged | 49,402 |
| Oversized blocks skipped in this run | 0 |

The generated candidate set matched the repository's existing reference
implementation exactly on the full datasets. The figures above are measured
results from this validation run, not estimates.

## How to use the module

```python
from src.blocking import candidate_pairs, generate_candidate_details

pairs = candidate_pairs(dblp, scholar)
details = generate_candidate_details(dblp, scholar)

# Example:
# pairs = {("dblp-id-1", "scholar-id-9"), ...}
# details[("dblp-id-1", "scholar-id-9")] == {"title_prefix", "rare_tokens"}
```

The input DataFrames must contain `id` and `title`; `year` is optional. If the
normalized columns `t_tokens` and `t_squash` are absent, the module creates them.

## Tests

The focused candidate-generation and blocking tests pass:

```text
5 passed
```

The full repository test suite still has unrelated pre-existing/environment
failures involving Gradio's `Markdown(min_width=...)` argument and two pipeline/
ML expectations. Those failures are outside the candidate-generation changes
and should be reviewed separately.
