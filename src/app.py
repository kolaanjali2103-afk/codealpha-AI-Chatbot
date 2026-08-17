"""
app.py
------
This file is the Streamlit user interface for the FAQ chatbot.

It does NOT contain any NLP or matching logic itself -- it simply:
    1. Loads the FAQChatbot from chatbot.py (which loads the FAQ data,
       preprocesses it, and builds the TF-IDF matrix, once).
    2. Displays a simple chat-style interface.
    3. Sends whatever the user types to `chatbot.get_response(...)`.
    4. Displays the returned answer (and, optionally, the similarity
       score) back to the user.

Run this file with:
    streamlit run src/app.py
"""

import os
import sys

import streamlit as st

# Allow running `streamlit run src/app.py` directly from the project
# root by making sure the project root is on the Python path, so that
# `from src.chatbot import FAQChatbot` works no matter where the
# command is run from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.chatbot import FAQChatbot  # noqa: E402

FAQ_PATH = os.path.join(PROJECT_ROOT, "data", "faqs.json")

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="Loading FAQ knowledge base...")
def load_chatbot() -> FAQChatbot:
    """
    Load and prepare the chatbot exactly once per app session.
    Streamlit reruns this script on every user interaction, so caching
    the chatbot avoids re-reading the JSON file and rebuilding the
    TF-IDF matrix on every single message.
    """
    return FAQChatbot(FAQ_PATH)


chatbot = load_chatbot()

# ---------------------------------------------------------------------
# Sidebar: project info + settings
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "This chatbot answers common **e-commerce / online shopping** "
        "questions by comparing your question to a set of FAQs using "
        "**TF-IDF** and **cosine similarity** (classic NLP techniques)."
    )
    st.write("It does **not** use any external AI/LLM API for matching.")

    st.divider()
    show_debug = st.toggle("Show similarity score", value=True)
    st.caption(
        "The similarity score (0 to 1) shows how confident the bot is "
        "that it found the right FAQ. Below the threshold, it will say "
        "it doesn't know the answer instead of guessing."
    )

    st.divider()
    st.caption("Try asking about: orders, payments, returns, shipping, "
               "your account, coupons, or warranty.")

# ---------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------
st.title("💬 FAQ Chatbot")
st.caption("Ask me anything about your order, payments, returns, or account.")

# Keep the conversation history in Streamlit's session state so it
# persists across reruns (Streamlit reruns the whole script on every
# interaction, but session_state survives between reruns).
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 👋 Ask me anything about our online store "
                       "(orders, payments, returns, shipping, and more).",
        }
    ]

# Render the existing conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("similarity") is not None and show_debug:
            st.caption(f"Similarity: {message['similarity']:.2f}")

# Chat input box at the bottom of the page.
user_input = st.chat_input("Type your question here...")

if user_input:
    # Show the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get the chatbot's response using TF-IDF + cosine similarity.
    result = chatbot.get_response(user_input)

    with st.chat_message("assistant"):
        st.write(result.answer)
        if show_debug:
            st.caption(f"Similarity: {result.similarity:.2f}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "similarity": result.similarity,
        }
    )
