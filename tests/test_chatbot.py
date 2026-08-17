"""
test_chatbot.py
----------------
A simple, dependency-light test script (not pytest-required, though it
works fine with pytest too) that runs a batch of sample questions
through the chatbot and prints whether each one matched an FAQ or
correctly triggered the fallback response.

Run it with:
    python tests/test_chatbot.py

or, if you have pytest installed:
    pytest tests/test_chatbot.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import FAQChatbot  # noqa: E402

FAQ_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "faqs.json"
)

# Each tuple: (question, expect_fallback)
# expect_fallback=True means we expect the bot to NOT find a confident match.
TEST_CASES = [
    ("How do I reset my password?", False),
    ("I forgot my password, what do I do?", False),
    ("What should I do if I can't remember my password?", False),
    ("Where is my package right now?", False),
    ("Can you tell me the status of my order?", False),
    ("How can I cancel my order?", False),
    ("What payment methods do you accept?", False),
    ("Do you accept cash on delivery?", False),
    ("How do I return a product I don't want?", False),
    ("My product arrived broken, what now?", False),
    ("How long will delivery take?", False),
    ("How do I apply a discount coupon?", False),
    ("Tell me a joke about cats.", True),
    ("What is the capital of France?", True),
    ("Can you help me book a flight ticket?", True),
]


def run_tests():
    bot = FAQChatbot(FAQ_PATH)
    passed = 0
    failed = 0

    for question, expect_fallback in TEST_CASES:
        result = bot.get_response(question)
        outcome_ok = result.is_fallback == expect_fallback
        status = "PASS" if outcome_ok else "FAIL"
        passed += outcome_ok
        failed += not outcome_ok

        print(f"[{status}] Q: {question}")
        print(f"       -> similarity={result.similarity:.2f} | "
              f"fallback={result.is_fallback} (expected {expect_fallback})")
        if not result.is_fallback:
            print(f"       -> matched FAQ: {result.matched_question!r}")
        print()

    print(f"Summary: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests.")


if __name__ == "__main__":
    run_tests()
