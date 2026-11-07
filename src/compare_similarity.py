"""Compare Student 1's similarity measures on DBLP-Scholar.

This file does not build blocking. That is a separate part of the project.
It only answers the string-matching questions:

- Section 20: which title mix (alpha) scores true matches well
- Section 21: token Jaccard authors versus best name match
- Section 23: a few record-level weight settings
- Section 44: exact, edit, token Jaccard, bigram, trigram, and combined

Gold pairs are used only as the answer key. They are not used to
compute a similarity score.

Run from the project folder:

    python src/compare_similarity.py
    python src/compare_similarity.py 300
"""

import os
import random
import sys
import time

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from normalize import normalize_title
from similarity import (
    WEIGHTS,
    author_similarity_best_match,
    author_similarity_jaccard,
    combine_scores,
    edit_similarity,
    jaccard_similarity,
    ngram_similarity,
    venue_similarity,
    year_similarity,
)

ROOT = os.path.dirname(_HERE)
ALL_PAIRS = 2616 * 64263
THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TITLE_MEASURES = ["exact", "edit", "jaccard", "bigram", "trigram", "fourgram", "combined"]

# The first setting is the one in the problem statement.
# These are compared. The default WEIGHTS in similarity.py are not changed here,
# because a final choice needs a held-out test split.
WEIGHT_TRIALS = [
    ("example 0.50/0.30/0.10/0.10", {"title": 0.50, "author": 0.30, "venue": 0.10, "year": 0.10}),
    ("title heavy", {"title": 0.70, "author": 0.20, "venue": 0.05, "year": 0.05}),
    ("author heavy", {"title": 0.35, "author": 0.45, "venue": 0.10, "year": 0.10}),
    ("year light", {"title": 0.55, "author": 0.30, "venue": 0.10, "year": 0.05}),
]


def find_file(name):
    data_dir = os.path.join(ROOT, "data")
    for folder, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.lower() == name.lower():
                return os.path.join(folder, file_name)
    raise FileNotFoundError("Could not find %s inside data/" % name)


def read_table(path):
    """DBLP is Latin-1. Scholar is UTF-8 and starts with a BOM."""
    file_name = os.path.basename(path).lower()
    if file_name.startswith("scholar"):
        encoding = "utf-8-sig"
    elif "mapping" in file_name:
        encoding = "utf-8"
    else:
        encoding = "latin1"

    table = pd.read_csv(path, encoding=encoding)
    cleaned = []
    for column in table.columns:
        name = str(column).replace("\ufeff", "").replace("ï»¿", "").replace('"', "").strip()
        cleaned.append(name)
    table.columns = cleaned
    return table


def load_records(table):
    records = {}
    for row in table.itertuples(index=False):
        records[str(row.id)] = {
            "title": getattr(row, "title"),
            "authors": getattr(row, "authors"),
            "venue": getattr(row, "venue"),
            "year": getattr(row, "year"),
            "norm_title": normalize_title(getattr(row, "title")),
        }
    return records


def first_token_index(records):
    """Map the first cleaned title word to record ids. Section 27 uses this idea."""
    index = {}
    for record_id, record in records.items():
        words = record["norm_title"].split()
        if not words:
            continue
        key = words[0]
        if key not in index:
            index[key] = []
        index[key].append(record_id)
    return index


def sample_pairs(gold_pairs, dblp, scholar, limit):
    """Gold matches, random non-matches, and same-first-word non-matches."""
    gold = list(gold_pairs)
    gold.sort()
    if limit is not None:
        gold = gold[:limit]

    gold_set = set(gold_pairs)
    dblp_ids = list(dblp.keys())
    scholar_ids = list(scholar.keys())
    picker = random.Random(42)

    random_pairs = []
    seen_random = set()
    while len(random_pairs) < len(gold):
        pair = (picker.choice(dblp_ids), picker.choice(scholar_ids))
        if pair not in gold_set and pair not in seen_random:
            seen_random.add(pair)
            random_pairs.append(pair)

    prefix_index = first_token_index(scholar)
    hard_pairs = []
    seen_hard = set()
    for left_id, _right_id in gold:
        words = dblp[left_id]["norm_title"].split()
        if not words:
            continue
        for other_id in prefix_index.get(words[0], []):
            pair = (left_id, other_id)
            if pair not in gold_set and pair not in seen_hard:
                seen_hard.add(pair)
                hard_pairs.append(pair)
                break

    return gold, random_pairs, hard_pairs


def title_scores(left, right, alpha=0.4):
    if left == "" or right == "":
        return {name: None for name in TITLE_MEASURES}, None, None

    edit_score = edit_similarity(left, right)
    token_score = jaccard_similarity(left.split(), right.split())
    scores = {
        "exact": 1.0 if left == right else 0.0,
        "edit": edit_score,
        "jaccard": token_score,
        "bigram": ngram_similarity(left, right, 2),
        "trigram": ngram_similarity(left, right, 3),
        "fourgram": ngram_similarity(left, right, 4),
        "combined": (alpha * edit_score) + ((1.0 - alpha) * token_score),
    }
    return scores, edit_score, token_score


def score_pairs(pairs, dblp, scholar):
    rows = []
    started = time.perf_counter()
    for number, (left_id, right_id) in enumerate(pairs, start=1):
        left = dblp[left_id]
        right = scholar[right_id]
        titles, edit_score, token_score = title_scores(left["norm_title"], right["norm_title"])
        author_jaccard = author_similarity_jaccard(left["authors"], right["authors"])
        author_names = author_similarity_best_match(left["authors"], right["authors"])
        venue = venue_similarity(left["venue"], right["venue"])
        year = year_similarity(left["year"], right["year"])
        rows.append({
            "titles": titles,
            "edit": edit_score,
            "jaccard": token_score,
            "author_jaccard": author_jaccard,
            "author_names": author_names,
            "venue": venue,
            "year": year,
        })
        if number % 500 == 0:
            print("  scored %d pairs" % number)
    seconds = time.perf_counter() - started
    return rows, seconds


def average(scores):
    kept = []
    for score in scores:
        if score is not None:
            kept.append(score)
    if not kept:
        return 0.0
    return sum(kept) / len(kept)


def precision_recall_f1(match_scores, other_scores, threshold):
    tp = 0
    for score in match_scores:
        if score is not None and score >= threshold:
            tp += 1
    fp = 0
    for score in other_scores:
        if score is not None and score >= threshold:
            fp += 1
    fn = len(match_scores) - tp
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def best_threshold(match_scores, other_scores):
    """Pick the threshold in the required list with the highest F1."""
    best_f1 = -1.0
    best = (THRESHOLDS[0], 0.0, 0.0, 0.0)
    for threshold in THRESHOLDS:
        precision, recall, f1 = precision_recall_f1(match_scores, other_scores, threshold)
        if f1 > best_f1:
            best_f1 = f1
            best = (threshold, precision, recall, f1)
    return best


def column(rows, key):
    return [row[key] for row in rows]


def print_title_table(gold_rows, other_rows, other_name):
    print()
    print("Title measures against %s" % other_name)
    print("%-12s %8s %8s %8s %8s %8s %8s" % (
        "measure", "gold avg", "other avg", "best T", "P", "R", "F1"
    ))
    for name in TITLE_MEASURES:
        match_scores = [row["titles"][name] for row in gold_rows]
        other_scores = [row["titles"][name] for row in other_rows]
        threshold, precision, recall, f1 = best_threshold(match_scores, other_scores)
        print("%-12s %8.3f %8.3f %8.2f %8.3f %8.3f %8.3f" % (
            name,
            average(match_scores),
            average(other_scores),
            threshold,
            precision,
            recall,
            f1,
        ))


def print_alpha_table(gold_rows, other_rows, other_name):
    print()
    print("Title alpha against %s" % other_name)
    print("S = alpha * edit + (1 - alpha) * token Jaccard")
    print("%-8s %8s %8s %8s %8s %8s" % ("alpha", "gold avg", "other avg", "best T", "recall", "F1"))
    for alpha in ALPHAS:
        match_scores = []
        other_scores = []
        for row in gold_rows:
            if row["edit"] is None:
                match_scores.append(None)
            else:
                match_scores.append((alpha * row["edit"]) + ((1.0 - alpha) * row["jaccard"]))
        for row in other_rows:
            if row["edit"] is None:
                other_scores.append(None)
            else:
                other_scores.append((alpha * row["edit"]) + ((1.0 - alpha) * row["jaccard"]))
        threshold, _precision, recall, f1 = best_threshold(match_scores, other_scores)
        print("%-8.1f %8.3f %8.3f %8.2f %8.3f %8.3f" % (
            alpha,
            average(match_scores),
            average(other_scores),
            threshold,
            recall,
            f1,
        ))


def print_author_table(gold_rows, other_rows, other_name):
    print()
    print("Author methods against %s" % other_name)
    print("%-16s %8s %8s %8s %8s" % ("method", "gold avg", "other avg", "best T", "F1"))
    methods = [
        ("token jaccard", "author_jaccard"),
        ("best name", "author_names"),
    ]
    for label, key in methods:
        match_scores = column(gold_rows, key)
        other_scores = column(other_rows, key)
        threshold, _precision, _recall, f1 = best_threshold(match_scores, other_scores)
        print("%-16s %8.3f %8.3f %8.2f %8.3f" % (
            label,
            average(match_scores),
            average(other_scores),
            threshold,
            f1,
        ))


def record_scores(rows, weights, author_key):
    scores = []
    for row in rows:
        if row["edit"] is None:
            title_score = None
        else:
            title_score = row["titles"]["combined"]
        parts = {
            "title": title_score,
            "author": row[author_key],
            "venue": row["venue"],
            "year": row["year"],
        }
        scores.append(combine_scores(parts, weights))
    return scores


def print_threshold_curve(label, match_scores, other_scores):
    print()
    print("Threshold curve: %s" % label)
    print("%8s %8s %8s %8s" % ("T", "P", "R", "F1"))
    for threshold in THRESHOLDS:
        precision, recall, f1 = precision_recall_f1(match_scores, other_scores, threshold)
        print("%8.2f %8.3f %8.3f %8.3f" % (threshold, precision, recall, f1))


def print_weight_table(gold_rows, other_rows, other_name):
    print()
    print("Record weights against %s, using token-Jaccard authors" % other_name)
    print("%-28s %8s %8s %8s %8s" % ("weights", "gold avg", "other avg", "best T", "F1"))
    for label, weights in WEIGHT_TRIALS:
        match_scores = record_scores(gold_rows, weights, "author_jaccard")
        other_scores = record_scores(other_rows, weights, "author_jaccard")
        threshold, _precision, _recall, f1 = best_threshold(match_scores, other_scores)
        print("%-28s %8.3f %8.3f %8.2f %8.3f" % (
            label,
            average(match_scores),
            average(other_scores),
            threshold,
            f1,
        ))


def main():
    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])

    dblp = load_records(read_table(find_file("DBLP.csv")))
    scholar = load_records(read_table(find_file("Scholar.csv")))
    gold_table = read_table(find_file("DBLP-Scholar_perfectMapping.csv"))

    gold_pairs = []
    seen = set()
    for row in gold_table.itertuples(index=False):
        pair = (str(row.idDBLP), str(row.idScholar))
        if pair not in seen and pair[0] in dblp and pair[1] in scholar:
            seen.add(pair)
            gold_pairs.append(pair)

    print("DBLP records:    %d" % len(dblp))
    print("Scholar records: %d" % len(scholar))
    print("Gold pairs:      %d" % len(gold_pairs))
    print("All possible pairs: %d" % ALL_PAIRS)
    print("Default record weights:", WEIGHTS)
    print()
    print("Precision and F1 in the tables are on the scored sample only.")
    print("Recall uses every gold pair in that sample.")
    print("This is not a run over all %d pairs." % ALL_PAIRS)

    gold, random_pairs, hard_pairs = sample_pairs(gold_pairs, dblp, scholar, limit)
    print()
    print("Scoring %d gold, %d random non-matches, %d same-first-word non-matches" % (
        len(gold), len(random_pairs), len(hard_pairs)
    ))

    gold_rows, gold_seconds = score_pairs(gold, dblp, scholar)
    random_rows, random_seconds = score_pairs(random_pairs, dblp, scholar)
    hard_rows, hard_seconds = score_pairs(hard_pairs, dblp, scholar)
    print("Scoring time: gold %.1fs, random %.1fs, same-first-word %.1fs" % (
        gold_seconds, random_seconds, hard_seconds
    ))

    print_title_table(gold_rows, random_rows, "random non-matches")
    print_title_table(gold_rows, hard_rows, "same-first-word non-matches")
    print_threshold_curve(
        "combined title vs same-first-word non-matches",
        [row["titles"]["combined"] for row in gold_rows],
        [row["titles"]["combined"] for row in hard_rows],
    )
    print_alpha_table(gold_rows, hard_rows, "same-first-word non-matches")
    print_author_table(gold_rows, hard_rows, "same-first-word non-matches")
    print_weight_table(gold_rows, hard_rows, "same-first-word non-matches")
    print_threshold_curve(
        "example record weights vs same-first-word non-matches",
        record_scores(gold_rows, WEIGHT_TRIALS[0][1], "author_jaccard"),
        record_scores(hard_rows, WEIGHT_TRIALS[0][1], "author_jaccard"),
    )
    print()
    print("Same-first-word pairs are harder than random pairs.")
    print("Use those rows when choosing a title measure, alpha, and weights.")
    print()
    print("On this run, exact title match stays precise but misses many gold pairs.")
    print("Edit similarity and character n-grams recover most of those pairs.")
    print("Best-name author similarity separates matches better than token Jaccard.")
    print("A higher threshold raises precision and lowers recall.")
    print("The starting alpha (0.4) and weights stay in similarity.py.")
    print("Change them only after a held-out split, so the test pairs stay untouched.")


if __name__ == "__main__":
    main()
