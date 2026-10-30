import re


# "An Efficient Method for Record-Linkage!" -> "efficient method record linkage"
TITLE_STOP_WORDS = {"a", "an", "the", "for", "of", "and", "in", "on", "to"}


def is_missing(value):
    """True when a field is empty or blank."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null"}


def _basic_clean(text):
    """Lowercase, turn punctuation into spaces, and squeeze extra spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def normalize_title(text):
    """Section 11. Clean a paper title."""
    if is_missing(text):
        return ""
    words = _basic_clean(text).split()
    kept = []
    for word in words:
        if word not in TITLE_STOP_WORDS:
            kept.append(word)
    return " ".join(kept)


def author_tokens(text):
    """Section 12. Split an author list into small tokens.

    "Smith, J.; Kumar, A." becomes ["smith", "j", "kumar", "a"].
    Order is kept here. Later matching uses a set, so order does not matter.
    Stop words are kept, because a single letter like "a" can be an initial.
    """
    if is_missing(text):
        return []
    return _basic_clean(text).split()


def split_author_names(text):
    """Section 12. Split an author list into whole names.

    A semicolon separates people. If there is no semicolon, a comma does.
    "Smith, J.; Kumar, A." becomes ["smith j", "kumar a"].
    """
    if is_missing(text):
        return []
    raw = str(text)
    if ";" in raw:
        parts = raw.split(";")
    else:
        parts = raw.split(",")
    names = []
    for part in parts:
        cleaned = _basic_clean(part)
        if cleaned:
            names.append(cleaned)
    return names


def normalize_venue(text):
    """Section 13. Clean a venue name.

    "Proc. ICDE" becomes "proc icde".
    "ICDE" becomes "icde".
    These two still differ. N-gram comparison in Week 2 is what measures them.
    """
    if is_missing(text):
        return ""
    return _basic_clean(text)


def parse_year(value):
    """Read a year as a 4-digit number. Missing or junk years become None."""
    if is_missing(value):
        return None
    digits = ""
    for ch in str(value).strip():
        if ch.isdigit():
            digits += ch
            if len(digits) == 4:
                break
        elif digits:
            break
    if len(digits) != 4:
        return None
    return int(digits)


if __name__ == "__main__":
    print("TITLE")
    sample = "An Efficient Method for Record-Linkage!"
    print(sample)
    print(normalize_title(sample))
    print()

    print("AUTHOR TOKENS")
    authors = "Smith, J.; Kumar, A."
    print(authors)
    print(author_tokens(authors))
    print(split_author_names(authors))
    print()

    print("VENUE")
    print(normalize_venue("Proc. ICDE"))
    print(normalize_venue("ICDE"))
    print(normalize_venue("International Conference on Data Engineering"))
    print()

    print("YEAR")
    print(parse_year("2005"))
    print(parse_year(""))
    print(parse_year("see"))
