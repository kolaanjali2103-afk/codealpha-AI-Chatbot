# 💬 FAQ Chatbot using NLP and Similarity Matching

A beginner-friendly, fully local FAQ chatbot for an **e-commerce / online shopping** platform. It answers customer questions by matching them against a curated FAQ dataset using classic NLP techniques — **TF-IDF vectorization** and **cosine similarity** — with no external AI/LLM API involved.

## Description

Customers ask the same question in many different ways ("How do I reset my password?", "I forgot my password", "What should I do if I can't remember my password?"). This project builds a chatbot that understands the *underlying meaning* of a question well enough to route it to the correct FAQ answer, and that knows when to admit it doesn't know an answer rather than guessing.

## Features

- 35 realistic FAQ question–answer pairs across 8 categories (Orders, Payments, Returns, Shipping, Account, Discounts, Support, Products/Warranty)
- Text preprocessing pipeline: lowercasing, punctuation removal, tokenization, stopword removal, lemmatization (NLTK)
- FAQ matching via TF-IDF vectorization + cosine similarity (scikit-learn)
- Confidence threshold with a graceful fallback response for unrelated questions
- Simple, chat-style Streamlit web interface with optional similarity score display
- Modular, commented, beginner-readable codebase
- Standalone test script with 15 sample questions

## Technologies used

| Purpose            | Library         |
|--------------------|------------------|
| Text preprocessing | NLTK             |
| Vectorization & similarity | scikit-learn (TfidfVectorizer, cosine_similarity) |
| Web UI             | Streamlit        |
| Language           | Python 3.9+      |

## Project structure

```text
faq-chatbot/
│
├── data/
│   └── faqs.json              # FAQ dataset (35 Q&A pairs)
│
├── src/
│   ├── __init__.py            # Marks src as a Python package
│   ├── preprocessing.py       # Text cleaning: lowercase, tokenize, remove stopwords, lemmatize
│   ├── chatbot.py             # Core engine: TF-IDF + cosine similarity matching, threshold logic
│   └── app.py                 # Streamlit chat UI
│
├── tests/
│   └── test_chatbot.py        # 15 sample-question test script
│
├── requirements.txt
├── README.md
└── .gitignore
```

### What each file does

- **`data/faqs.json`** — The knowledge base. Each entry has an `id`, `category`, `question`, and `answer`.
- **`src/preprocessing.py`** — Cleans any piece of text (FAQ question or user question) the same way, so they can be fairly compared.
- **`src/chatbot.py`** — Loads the FAQs, preprocesses them, builds a TF-IDF matrix, and exposes `FAQChatbot.get_response(question)` which returns the best answer or a fallback.
- **`src/app.py`** — The Streamlit UI. Imports `FAQChatbot` from `chatbot.py` and wires it up to a chat interface. Contains no NLP logic of its own.
- **`tests/test_chatbot.py`** — Runs 15 sample questions through the chatbot and checks whether matches / fallbacks behave as expected.

## Installation

### 1. Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first time you run the app, NLTK will automatically download the small `punkt`, `stopwords`, and `wordnet` data packages it needs (this happens once and requires an internet connection).

## How to run

### Run the chat UI

```bash
streamlit run src/app.py
```

This opens the chatbot in your browser at `http://localhost:8501`.

### Run the test script

```bash
python tests/test_chatbot.py
```

### Run the chatbot from the command line (no UI)

```bash
python -m src.chatbot
```

## How the chatbot works

```text
FAQ Dataset (faqs.json)
        ↓
Text Preprocessing (lowercase, remove punctuation, tokenize,
                     remove stopwords, lemmatize)
        ↓
TF-IDF Vectorization of all FAQ questions
        ↓
User types a question
        ↓
Same preprocessing applied to the user's question
        ↓
User question converted into the same TF-IDF vector space
        ↓
Cosine similarity computed against every FAQ question
        ↓
Highest similarity score selected
        ↓
Is the score ≥ threshold (0.30)?
   ├── Yes → Return the matching FAQ's answer
   └── No  → Return a fallback ("I couldn't find a relevant answer...")
```

**In short:** the chatbot doesn't look for exact keyword matches. It turns every question into a vector of numbers that reflects which words matter most, then measures how "close" two vectors point in the same direction. Similar meaning → similar direction → high similarity score.

## Example questions to try

- "How do I reset my password?"
- "I forgot my password"
- "Where is my order?"
- "Can I cancel my order?"
- "What payment methods do you accept?"
- "How do I return a damaged product?"
- "Do you offer cash on delivery?"
- "How long does delivery take?"
- "Tell me about the weather tomorrow." *(should trigger the fallback)*

## Tuning the confidence threshold

The threshold lives in `src/chatbot.py` as `DEFAULT_THRESHOLD = 0.30`.

- **Lower it** (e.g. 0.20) if the bot is falling back too often on questions that should have matched.
- **Raise it** (e.g. 0.40) if the bot is confidently answering questions that aren't really related to any FAQ.

A good way to tune it: run `tests/test_chatbot.py`, look at the printed similarity scores, and find a value that separates your "should match" examples from your "should fallback" examples.

## Future improvements

- Use sentence embeddings (e.g. `sentence-transformers`) instead of TF-IDF for semantic matching, so paraphrases with completely different words (e.g. "ship internationally" vs "deliver outside the country") can still match.
- Add spelling correction for typos in user questions.
- Log unanswered questions so new FAQs can be added over time.
- Support multi-turn conversations (e.g. follow-up questions like "and how long does that take?").
- Add authentication and connect to a real order-tracking backend for account-specific answers.

## License

This project was built as an internship learning project and is free to use, modify, and extend.
