"""Talk to the language model.

Uses Claude through the Anthropic API when ANTHROPIC_API_KEY is set.
Without a key it falls back to an "offline" mode that just returns the best
matching sentences from the sources, so the app still runs for anyone who clones it.
"""

from __future__ import annotations

import os
import re

from .embeddings import tokenize

DEFAULT_MODEL = os.getenv("RAG_MODEL", "claude-haiku-5-5")


def has_api_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def generate(system: str, user: str, model: str = DEFAULT_MODEL, max_tokens: int = 500) -> str:
    """Send one prompt to Claude and return the text answer."""
    import anthropic  # imported here so offline mode doesn't need the package

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.2,  # low = more factual, less creative
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


# Small "intent" hints so the offline mode picks the line that actually answers the question.
INTENT_HINTS = {
    ("much", "price", "cost", "costs"): re.compile(r"\$\d"),
    ("long", "days", "when", "until"): re.compile(r"\d+ (business )?(days|hours?|months?)", re.I),
    ("big", "size", "limit", "storage"): re.compile(r"\d+ ?(MB|GB)", re.I),
    ("many", "number"): re.compile(r"\b\d+\b"),
}


def extractive_answer(question: str, sources: list[tuple[int, str]], max_sentences: int = 3) -> str:
    """Offline fallback: pick the sentences that share the most words with the question.

    sources: list of (citation number, text)
    """
    q_words = set(re.findall(r"[a-z]+", question.lower()))
    q_terms = set(tokenize(question))
    hints = [pattern for words, pattern in INTENT_HINTS.items() if q_words & set(words)]

    scored = []
    for number, text in sources:
        _title, _, body = text.partition("\n")  # first line of every chunk is its title
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
            sentence = sentence.strip(" -")
            if len(sentence) < 10:
                continue
            score = len(q_terms & set(tokenize(sentence)))
            score += 2 * sum(1 for h in hints if h.search(sentence))
            if not score:
                continue
            if number == 1:
                score += 1  # the top-ranked source is usually the right one (eval: 72% -> 96%)
            scored.append((score, -number, sentence, number))  # earlier sources win ties
    if not scored:
        return ""
    scored.sort(reverse=True)
    picked = scored[:max_sentences]
    return " ".join(f"{s} [{n}]" for _, _, s, n in picked)
