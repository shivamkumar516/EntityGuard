import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from normalize import (
    author_tokens,
    normalize_title,
    normalize_venue,
    parse_year,
    split_author_names,
)
from similarity import (
    author_similarity_best_match,
    author_similarity_jaccard,
    char_ngrams,
    combine_scores,
    edit_similarity,
    jaccard_similarity,
    levenshtein,
    levenshtein_two_rows,
    record_similarity,
    title_exact,
    title_length_ratio,
    title_similarity,
    venue_exact,
    venue_similarity,
    year_similarity,
)


class TestNormalization(unittest.TestCase):
    def test_title_example_from_the_problem(self):
        text = "An Efficient Method for Record-Linkage!"
        self.assertEqual(normalize_title(text), "efficient method record linkage")

    def test_author_tokens_keep_initials(self):
        tokens = author_tokens("Smith, J.; Kumar, A.")
        self.assertEqual(tokens, ["smith", "j", "kumar", "a"])

    def test_author_names_are_split_on_semicolon(self):
        names = split_author_names("Smith, J.; Kumar, A.")
        self.assertEqual(names, ["smith j", "kumar a"])

    def test_venue_and_year(self):
        self.assertEqual(normalize_venue("Proc. ICDE"), "proc icde")
        self.assertEqual(parse_year("2005"), 2005)
        self.assertIsNone(parse_year(""))
        self.assertIsNone(parse_year("see"))


class TestEditDistance(unittest.TestCase):
    def test_known_distance(self):
        # kitten -> sitting needs 3 edits.
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("algorithm", "algorithms"), 1)

    def test_base_cases(self):
        self.assertEqual(levenshtein("", ""), 0)
        self.assertEqual(levenshtein("abc", ""), 3)
        self.assertEqual(levenshtein("", "ab"), 2)

    def test_two_rows_match_the_full_table(self):
        samples = ["", "a", "algorithm", "algorithms", "database", "kitten", "sitting"]
        for left in samples:
            for right in samples:
                full = levenshtein(left, right)
                small = levenshtein_two_rows(left, right)
                self.assertEqual(full, small)

    def test_edit_similarity_bounds(self):
        score = edit_similarity("algorithm", "algorithms")
        self.assertAlmostEqual(score, 0.9)
        self.assertEqual(edit_similarity("", ""), 1.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestJaccardAndNgrams(unittest.TestCase):
    def test_problem_jaccard_example(self):
        words_a = ["deep", "neural", "network", "classification"]
        words_b = ["neural", "network", "classification"]
        self.assertAlmostEqual(jaccard_similarity(words_a, words_b), 0.75)

    def test_database_ngrams(self):
        self.assertEqual(char_ngrams("database", 2), ["da", "at", "ta", "ab", "ba", "as", "se"])
        self.assertEqual(char_ngrams("database", 3), ["dat", "ata", "tab", "aba", "bas", "ase"])
        self.assertEqual(len(char_ngrams("database", 4)), 5)


class TestAttributeScores(unittest.TestCase):
    def test_identical_titles_score_one(self):
        title = "An Efficient Method for Record-Linkage!"
        self.assertEqual(title_exact(title, title), 1.0)
        self.assertAlmostEqual(title_similarity(title, title), 1.0)
        self.assertAlmostEqual(title_length_ratio(title, title), 1.0)

    def test_missing_title_is_left_out(self):
        self.assertIsNone(title_similarity("A paper", ""))
        self.assertIsNone(title_exact("", "A paper"))

    def test_year_scores(self):
        self.assertEqual(year_similarity(2005, 2005), 1.0)
        self.assertEqual(year_similarity(2005, 2006), 0.5)
        self.assertEqual(year_similarity(2005, 2008), 0.0)
        self.assertIsNone(year_similarity(2005, ""))

    def test_venue_ngram_beats_exact_on_abbreviation(self):
        exact = venue_exact("Proc. ICDE", "ICDE")
        fuzzy = venue_similarity("Proc. ICDE", "ICDE")
        self.assertEqual(exact, 0.0)
        self.assertGreater(fuzzy, exact)

    def test_two_author_methods_both_run(self):
        left = "A. Kumar, R. Singh"
        right = "Anil Kumar; Raj Singh"
        token_score = author_similarity_jaccard(left, right)
        name_score = author_similarity_best_match(left, right)
        self.assertGreater(token_score, 0.0)
        self.assertGreater(name_score, 0.0)
        self.assertIsNone(author_similarity_jaccard("", right))

    def test_missing_field_is_dropped_from_the_total(self):
        # title, author, and year are present. venue is missing.
        # weights 0.50 + 0.30 + 0.10 = 0.90, and every present score is 1.
        overall, parts = record_similarity(
            "Database Systems",
            "Smith, J.",
            "VLDB",
            2005,
            "Database Systems",
            "Smith, J.",
            "",
            2005,
        )
        self.assertIsNone(parts["venue"])
        self.assertAlmostEqual(overall, 1.0)

    def test_combine_scores_renormalizes(self):
        scores = {"title": 1.0, "author": 1.0, "venue": None, "year": 0.0}
        # (0.50 + 0.30 + 0.10 * 0) / (0.50 + 0.30 + 0.10) = 0.8 / 0.9
        overall = combine_scores(scores)
        self.assertAlmostEqual(overall, 0.8 / 0.9)

    def test_custom_weights(self):
        weights = {"title": 1.0, "author": 0.0, "venue": 0.0, "year": 0.0}
        overall, parts = record_similarity(
            "Database Systems",
            "Smith, J.",
            "VLDB",
            2001,
            "Database Systems",
            "Someone Else",
            "Other Venue",
            1990,
            weights=weights,
        )
        self.assertAlmostEqual(parts["title"], 1.0)
        self.assertAlmostEqual(overall, 1.0)


if __name__ == "__main__":
    unittest.main()
