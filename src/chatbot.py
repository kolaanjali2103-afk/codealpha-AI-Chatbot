"""
chatbot.py
----------
This is the core "brain" of the FAQ chatbot. It:
    1. Loads the FAQ dataset from data/faqs.json
    2. Preprocesses every FAQ question (using preprocessing.py)
    3. Converts all FAQ questions into TF-IDF vectors
    4. Given a new user question, preprocesses it the same way,
       converts it to a TF-IDF vector, and compares it against every
       FAQ question using cosine similarity.
    5. Returns the best-matching answer if the similarity score is
       above a confidence threshold, otherwise returns a fallback
       message.

Why TF-IDF + cosine similarity?
    TF-IDF (Term Frequency - Inverse Document Frequency) turns each
    sentence into a vector of numbers, where common words across all
    FAQs get a lower weight and rarer, more distinctive words get a
    higher weight. Cosine similarity then measures the "angle" between
    two vectors -- the smaller the angle, the more similar the
    sentences are in meaning (based on word overlap and importance).
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text

# Default confidence threshold. If the best match's similarity score is
# below this value, we assume none of the FAQs are actually relevant
# and return a fallback response instead of a wrong answer.
#
# How to tune this:
#   - Too LOW (e.g. 0.1): the bot will confidently answer questions
#     that aren't really related, giving wrong answers.
#   - Too HIGH (e.g. 0.7): the bot will reject valid questions just
#     because the user phrased it differently from the FAQ.
#   - 0.25-0.35 is a good starting point for short FAQ-style sentences
#     with TF-IDF. Adjust based on your own testing (see the test
#     questions in README.md).
DEFAULT_THRESHOLD = 0.30

FALLBACK_RESPONSE = (
    "I'm sorry, I couldn't find a relevant answer to your question. "
    "Please try asking in a different way, or contact our support team "
    "for further help."
)


@dataclass
class MatchResult:
    """A simple container for what the chatbot found for a user question."""
    answer: str
    matched_question: Optional[str]
    category: Optional[str]
    similarity: float
    is_fallback: bool


class FAQChatbot:
    """
    Loads an FAQ dataset and answers user questions using TF-IDF +
    cosine similarity matching.
    """

    def __init__(self, faq_path: str, threshold: float = DEFAULT_THRESHOLD):
        self.faq_path = faq_path
        self.threshold = threshold

        self.faqs: List[dict] = self._load_faqs(faq_path)
        self.raw_questions: List[str] = [faq["question"] for faq in self.faqs]

        # Preprocess every FAQ question once, up front, so we don't
        # redo this work every time a user asks something.
        self.cleaned_questions: List[str] = [
            clean_text(q) for q in self.raw_questions
        ]

        # Fit a TF-IDF vectorizer on the cleaned FAQ questions. This
        # builds the vocabulary and the IDF (word importance) weights
        # from our FAQ dataset.
        self.vectorizer = TfidfVectorizer()
        self.faq_vectors = self.vectorizer.fit_transform(self.cleaned_questions)

    @staticmethod
    def _load_faqs(faq_path: str) -> List[dict]:
        if not os.path.exists(faq_path):
            raise FileNotFoundError(f"FAQ dataset not found at: {faq_path}")
        with open(faq_path, "r", encoding="utf-8") as f:
            faqs = json.load(f)
        if not faqs:
            raise ValueError("FAQ dataset is empty.")
        return faqs

    def get_response(self, user_question: str) -> MatchResult:
        """
        Given a raw user question (as typed in the UI), return the
        best-matching FAQ answer, or a fallback response if nothing
        matches confidently enough.
        """
        cleaned_user_question = clean_text(user_question)

        # If preprocessing wipes out the entire question (e.g. the user
        # only typed punctuation or stopwords), we can't meaningfully
        # compare it to anything.
        if cleaned_user_question == "":
            return MatchResult(
                answer=FALLBACK_RESPONSE,
                matched_question=None,
                category=None,
                similarity=0.0,
                is_fallback=True,
            )

        # Convert the user's cleaned question into the SAME TF-IDF
        # vector space that the FAQ questions were fitted on. Note we
        # use `.transform()` here, NOT `.fit_transform()`, because we
        # want to reuse the vocabulary learned from the FAQ dataset.
        user_vector = self.vectorizer.transform([cleaned_user_question])

        # Compare the user's vector against every FAQ vector at once.
        # This returns an array of similarity scores, one per FAQ.
        similarities = cosine_similarity(user_vector, self.faq_vectors)[0]

        best_index = similarities.argmax()
        best_score = float(similarities[best_index])

        if best_score < self.threshold:
            return MatchResult(
                answer=FALLBACK_RESPONSE,
                matched_question=self.raw_questions[best_index],
                category=self.faqs[best_index].get("category"),
                similarity=best_score,
                is_fallback=True,
            )

        best_faq = self.faqs[best_index]
        return MatchResult(
            answer=best_faq["answer"],
            matched_question=best_faq["question"],
            category=best_faq.get("category"),
            similarity=best_score,
            is_fallback=False,
        )


if __name__ == "__main__":
    # Quick manual test you can run directly from the command line:
    #   python -m src.chatbot
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faq_file = os.path.join(here, "data", "faqs.json")

    bot = FAQChatbot(faq_file)

    test_questions = [
        "I forgot my password. How can I get back into my account?",
        "Where is my package right now?",
        "Tell me about the weather tomorrow.",
    ]

    for q in test_questions:
        result = bot.get_response(q)
        print(f"\nUser: {q}")
        print(f"Bot:  {result.answer}")
        print(f"(similarity={result.similarity:.2f}, fallback={result.is_fallback})")
