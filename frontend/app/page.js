"use client";

import { useState, useRef, useEffect } from "react";
import styles from "./page.module.css";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const WELCOME_MESSAGE = {
  id: "welcome",
  role: "assistant",
  text: "👋 Hello! I'm your e-commerce support assistant. Ask me anything about orders, payments, returns, shipping, your account, or discounts.",
  similarity: null,
  category: null,
  isFallback: false,
};

function SimilarityBadge({ score, isFallback }) {
  if (score === null) return null;
  const pct = Math.round(score * 100);
  const cls =
    isFallback || pct < 30
      ? styles.badgeLow
      : pct < 60
      ? styles.badgeMed
      : styles.badgeHigh;
  return (
    <span className={`${styles.badge} ${cls}`} title="Confidence score">
      {pct}% match
    </span>
  );
}

function ChatBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`${styles.bubbleRow} ${isUser ? styles.bubbleRowUser : styles.bubbleRowBot}`}>
      {!isUser && (
        <div className={styles.avatar} aria-label="Bot avatar">🤖</div>
      )}
      <div className={`${styles.bubble} ${isUser ? styles.bubbleUser : styles.bubbleBot}`}>
        <p className={styles.bubbleText}>{msg.text}</p>
        {!isUser && (
          <div className={styles.bubbleMeta}>
            {msg.category && (
              <span className={styles.category}>{msg.category}</span>
            )}
            <SimilarityBadge score={msg.similarity} isFallback={msg.isFallback} />
          </div>
        )}
      </div>
      {isUser && (
        <div className={`${styles.avatar} ${styles.avatarUser}`} aria-label="User avatar">👤</div>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className={`${styles.bubbleRow} ${styles.bubbleRowBot}`}>
      <div className={styles.avatar}>🤖</div>
      <div className={`${styles.bubble} ${styles.bubbleBot} ${styles.typing}`}>
        <span /><span /><span />
      </div>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async (e) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    const userMsg = { id: Date.now(), role: "user", text: question };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: data.answer,
          similarity: data.similarity,
          category: data.category,
          isFallback: data.is_fallback,
        },
      ]);
    } catch (err) {
      setError(
        err.message.includes("fetch")
          ? "⚠️ Cannot reach the chatbot API. Make sure the Python backend is running."
          : `⚠️ ${err.message}`
      );
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) sendMessage(e);
  };

  const suggestions = [
    "Where is my order?",
    "How do I return a product?",
    "What payment methods do you accept?",
    "How do I reset my password?",
  ];

  return (
    <main className={styles.main}>
      {/* ── Header ── */}
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.headerLogo}>💬</div>
          <div>
            <h1 className={styles.headerTitle}>FAQ Chatbot</h1>
            <p className={styles.headerSub}>E-Commerce Customer Support</p>
          </div>
          <div className={styles.statusDot} title="API connected" />
        </div>
      </header>

      {/* ── Chat Window ── */}
      <section className={styles.chatWindow} aria-label="Chat messages">
        {messages.map((msg) => (
          <ChatBubble key={msg.id} msg={msg} />
        ))}
        {loading && <TypingIndicator />}
        {error && (
          <div className={styles.errorBanner} role="alert">
            {error}
          </div>
        )}
        <div ref={bottomRef} />
      </section>

      {/* ── Quick Suggestions ── */}
      {messages.length <= 1 && (
        <div className={styles.suggestions}>
          {suggestions.map((s) => (
            <button
              key={s}
              className={styles.suggestionBtn}
              onClick={() => {
                setInput(s);
                inputRef.current?.focus();
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* ── Input Bar ── */}
      <form className={styles.inputBar} onSubmit={sendMessage} aria-label="Send a message">
        <textarea
          ref={inputRef}
          className={styles.textarea}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your question and press Enter…"
          rows={1}
          aria-label="Message input"
          disabled={loading}
        />
        <button
          type="submit"
          className={styles.sendBtn}
          disabled={!input.trim() || loading}
          aria-label="Send message"
        >
          {loading ? (
            <span className={styles.spinner} />
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </form>

      <footer className={styles.footer}>
        Powered by TF-IDF + Cosine Similarity · No external AI API
      </footer>
    </main>
  );
}
