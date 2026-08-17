"""
preprocessing.py
-----------------
This file is responsible for cleaning up raw text (both FAQ questions
and the user's typed question) so that the matching algorithm compares
"the meaning of the words" instead of getting confused by punctuation,
capitalization, or common filler words.

Why do we need this?
    "How do I reset my password?"
    "reset password"
    "How can I RESET my Password??"

To a computer, these three strings look completely different unless we
clean them first. After preprocessing, all three become something like:
    "reset password"

That makes it much easier for TF-IDF + cosine similarity (used in
chatbot.py) to see that they are asking the same thing.

We use NLTK for tokenization, stopword removal, and lemmatization.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------
# One-time NLTK setup
# ---------------------------------------------------------------------
# NLTK needs a few small data packages downloaded before it can tokenize,
# know stopwords, or lemmatize. We try to download them quietly the first
# time this module is imported. If they are already downloaded, this is
# a no-op and runs instantly.
_REQUIRED_NLTK_PACKAGES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

for _resource_path, _package_name in _REQUIRED_NLTK_PACKAGES:
    try:
        nltk.data.find(_resource_path)
    except (LookupError, OSError, Exception):
        try:
            nltk.download(_package_name, quiet=True)
        except Exception:
            # Fallback if quiet download fails (e.g. connection issue)
            pass

# Load English stopwords once (words like "is", "the", "a", "how" that
# carry little meaning on their own) and the lemmatizer (which reduces
# words to their base/dictionary form, e.g. "running" -> "run").
_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# NLTK's default stopword list already includes question words like
# "what", "how", "why" and "where". We deliberately keep "not" and "no"
# out of the removal set, because negation can matter for FAQ meaning
# (e.g. "payment not working" vs "payment working"). Everything else
# from NLTK's list is removed as-is.
_KEEP_WORDS = {"not", "no"}
_STOPWORDS = _STOPWORDS - _KEEP_WORDS


def clean_text(text: str) -> str:
    """
    Perform the full preprocessing pipeline on a single piece of text
    (either an FAQ question or a user's typed question) and return a
    cleaned string ready for TF-IDF vectorization.

    Steps:
        1. Lowercase the text.
        2. Remove punctuation and special characters.
        3. Tokenize (split into individual words).
        4. Remove stopwords (very common, low-meaning words).
        5. Lemmatize each remaining word (reduce to its base form).
        6. Join the cleaned tokens back into a single string.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # 1. Lowercase everything so "Password" and "password" are treated
    #    as the same word.
    text = text.lower()

    # 2. Remove punctuation (e.g. "?", "!", "'") and digits-attached
    #    symbols. We keep plain letters, numbers, and spaces.
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Tokenize: split the sentence into a list of individual words.
    tokens = word_tokenize(text)

    # 4 & 5. Remove stopwords, then lemmatize each remaining token to
    # its dictionary base form (e.g. "orders" -> "order",
    # "cancelled" -> "cancel").
    cleaned_tokens = [
        _LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in _STOPWORDS and len(token) > 1
    ]

    # 6. Join back into a single space-separated string, which is the
    # format scikit-learn's TfidfVectorizer expects.
    return " ".join(cleaned_tokens)


if __name__ == "__main__":
    # Small manual test you can run directly:
    #   python src/preprocessing.py
    samples = [
        "How do I reset my Password?!",
        "I forgot my password, what should I do??",
        "Where is my package right now?",
    ]
    for s in samples:
        print(f"{s!r}  ->  {clean_text(s)!r}")
