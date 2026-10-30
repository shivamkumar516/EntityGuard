import os
import sys


# Each function returns a score from 0 to 1, except Levenshtein, which returns a count.

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from normalize import (
    author_tokens,
    is_missing,
    normalize_title,
    normalize_venue,
    parse_year,
    split_author_names,
)

# Later experiments can change them. These are only a starting point.
WEIGHTS = {
    "title": 0.50,
    "author": 0.30,
    "venue": 0.10,
    "year": 0.10,
}


def levenshtein(a, b):
    """Section 15. Minimum insert, delete, and replace steps from a to b.

    D(i, j) is the cost of turning the first i characters of A
    into the first j characters of B.

        D(i, 0) = i
        D(0, j) = j
        D(i, j) = min(
            D(i - 1, j) + 1,          delete
            D(i, j - 1) + 1,          insert
            D(i - 1, j - 1) + cost    replace, cost is 0 when the characters match
        )
    """
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    rows = len(a) + 1
    cols = len(b) + 1
    table = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1
            delete = table[i - 1][j] + 1
            insert = table[i][j - 1] + 1
            replace = table[i - 1][j - 1] + cost
            table[i][j] = min(delete, insert, replace)

    return table[rows - 1][cols - 1]


def edit_similarity(a, b):
    """Section 16. Turn the edit count into a score between 0 and 1.

    S_edit = 1 - D(A, B) / max(length of A, length of B)
    """
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - (levenshtein(a, b) / longest)


def jaccard_similarity(tokens_a, tokens_b):
    """Section 18. Shared tokens divided by all unique tokens.

    J = |A intersection B| / |A union B|
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def char_ngrams(text, n):
    """Section 19. Overlapping character chunks of size n.

    "database" with n = 2 is da, at, ta, ab, ba, as, se.
    "database" with n = 3 is dat, ata, tab, aba, bas, ase.
    """
    text = "" if text is None else str(text)
    if n < 1 or text == "":
        return []
    if len(text) <= n:
        return [text]
    grams = []
    for i in range(len(text) - n + 1):
        grams.append(text[i:i + n])
    return grams


def ngram_similarity(text_a, text_b, n):
    """Jaccard similarity of character n-grams. n is 2, 3, or 4."""
    return jaccard_similarity(char_ngrams(text_a, n), char_ngrams(text_b, n))


def title_similarity(title_a, title_b, alpha=0.4):
    """Section 20. Mix edit similarity and token Jaccard.

    S_title = alpha * S_edit + (1 - alpha) * S_jaccard
    alpha = 0.4 is only the example starting value.
    """
    left = normalize_title(title_a)
    right = normalize_title(title_b)
    if left == "" or right == "":
        return None
    edit_score = edit_similarity(left, right)
    token_score = jaccard_similarity(left.split(), right.split())
    return (alpha * edit_score) + ((1.0 - alpha) * token_score)


def author_similarity_jaccard(authors_a, authors_b):
    """Section 21, method 1. Jaccard on author tokens."""
    left = author_tokens(authors_a)
    right = author_tokens(authors_b)
    if not left or not right:
        return None
    return jaccard_similarity(left, right)


def author_similarity_best_match(authors_a, authors_b):
    """Section 21, method 2. Average of the best name-to-name edit scores.

    Each name on one side is compared with every name on the other side.
    The best score for that name is kept. Then both directions are averaged
    so a longer author list does not decide the result by itself.
    """
    left = split_author_names(authors_a)
    right = split_author_names(authors_b)
    if not left or not right:
        return None

    def average_best(source, targets):
        total = 0.0
        for name in source:
            best = 0.0
            for other in targets:
                score = edit_similarity(name, other)
                if score > best:
                    best = score
            total += best
        return total / len(source)

    forward = average_best(left, right)
    backward = average_best(right, left)
    return (forward + backward) / 2.0


def venue_similarity(venue_a, venue_b, n=3):
    """Section 13. Compare venues with character n-grams.

    Exact equality is also returned by venue_exact. N-grams are the default
    because "ICDE" and "Proc. ICDE" are not the same string.
    """
    left = normalize_venue(venue_a)
    right = normalize_venue(venue_b)
    if left == "" or right == "":
        return None
    return ngram_similarity(left, right, n)


def venue_exact(venue_a, venue_b):
    """1 when the cleaned venue names are identical, otherwise 0."""
    left = normalize_venue(venue_a)
    right = normalize_venue(venue_b)
    if left == "" or right == "":
        return None
    if left == right:
        return 1.0
    return 0.0


def year_similarity(year_a, year_b):
    """Section 22.

    1.0 when the years are equal.
    0.5 when they differ by one year.
    0.0 otherwise.
    None when either year is missing, so that field is left out of the total.
    """
    left = parse_year(year_a)
    right = parse_year(year_b)
    if left is None or right is None:
        return None
    if left == right:
        return 1.0
    if abs(left - right) == 1:
        return 0.5
    return 0.0


def record_similarity(title_a, authors_a, venue_a, year_a,
                      title_b, authors_b, venue_b, year_b,
                      author_method="jaccard", alpha=0.4):
    """Sections 23 and 24. One score for two whole records.

    S = 0.50 * title + 0.30 * author + 0.10 * venue + 0.10 * year

    If a field is missing, its weight is removed and the other weights
    are divided by the remaining weight total.
    """
    if author_method == "jaccard":
        author_score = author_similarity_jaccard(authors_a, authors_b)
    elif author_method == "best_match":
        author_score = author_similarity_best_match(authors_a, authors_b)
    else:
        raise ValueError("author_method must be 'jaccard' or 'best_match'")

    scores = {
        "title": title_similarity(title_a, title_b, alpha),
        "author": author_score,
        "venue": venue_similarity(venue_a, venue_b),
        "year": year_similarity(year_a, year_b),
    }

    weight_total = 0.0
    weighted_sum = 0.0
    for field, score in scores.items():
        if score is None:
            continue
        weight_total += WEIGHTS[field]
        weighted_sum += WEIGHTS[field] * score

    if weight_total == 0:
        return 0.0, scores
    return weighted_sum / weight_total, scores


if __name__ == "__main__":
    print("EDIT DISTANCE")
    print("algorithm -> algorithms:", levenshtein("algorithm", "algorithms"))
    print("edit similarity:", round(edit_similarity("algorithm", "algorithms"), 4))
    print()

    print("JACCARD")
    words_a = ["deep", "neural", "network", "classification"]
    words_b = ["neural", "network", "classification"]
    print(jaccard_similarity(words_a, words_b), "(expected 0.75)")
    print()

    print("N-GRAMS OF database")
    print("bigrams ", char_ngrams("database", 2))
    print("trigrams", char_ngrams("database", 3))
    print()

    print("TITLE PAIR FROM THE PROBLEM STATEMENT")
    left_title = "An Efficient Algorithm for Entity Matching"
    right_title = "Efficient Algorithms for Entity Matching"
    print(normalize_title(left_title))
    print(normalize_title(right_title))
    print("title similarity", round(title_similarity(left_title, right_title), 4))
    print()

    print("TWO AUTHOR METHODS")
    left_authors = "A. Kumar, R. Singh"
    right_authors = "Anil Kumar; Raj Singh"
    print("jaccard   ", round(author_similarity_jaccard(left_authors, right_authors), 4))
    print("best match", round(author_similarity_best_match(left_authors, right_authors), 4))
    print()

    print("VENUE")
    print("exact  ", venue_exact("Proc. ICDE", "ICDE"))
    print("trigram", round(venue_similarity("Proc. ICDE", "ICDE"), 4))
    print()

    print("YEAR")
    print("same year ", year_similarity(2005, 2005))
    print("off by one", year_similarity(2005, 2006))
    print("far apart ", year_similarity(2005, 2008))
    print("missing   ", year_similarity(2005, ""))
    print()

    print("WHOLE RECORD, VENUE MISSING ON ONE SIDE")
    overall, parts = record_similarity(
        "An Efficient Algorithm for Entity Matching",
        "A. Kumar, R. Singh",
        "VLDB",
        2005,
        "Efficient Algorithms for Entity Matching",
        "Anil Kumar; Raj Singh",
        "",
        2005,
    )
    print("parts ", {name: None if value is None else round(value, 4) for name, value in parts.items()})
    print("overall", round(overall, 4))
