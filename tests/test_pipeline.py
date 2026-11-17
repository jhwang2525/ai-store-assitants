"""Tests run fully offline (hashing embedder, no API key)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant import llm  # noqa: E402
from rag_assistant.chunking import chunk_documents  # noqa: E402
from rag_assistant.embeddings import HashingEmbedder, tokenize  # noqa: E402
from rag_assistant.evaluate import evaluate_answers, evaluate_retrieval, load_questions  # noqa: E402
from rag_assistant.loaders import Document, load_folder  # noqa: E402
from rag_assistant.rag import NOT_FOUND, RagAssistant, build_prompt  # noqa: E402
from rag_assistant.vectorstore import VectorStore  # noqa: E402

DOCS = ROOT / "data" / "docs"


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(scope="module")
def store():
    s = VectorStore(HashingEmbedder())
    s.add(chunk_documents(load_folder(DOCS)))
    return s


def test_load_folder_reads_markdown_and_text(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\n\n## Part\nHello world.", encoding="utf-8")
    (tmp_path / "notes_b.txt").write_text("Plain text file.", encoding="utf-8")
    (tmp_path / "skip.jpg").write_bytes(b"\xff\xd8")
    docs = load_folder(tmp_path)
    assert [d.title for d in docs] == ["Alpha", "Notes B"]
    assert docs[0].metadata["format"] == "markdown"


def test_markdown_split_by_heading():
    chunks = chunk_documents(load_folder(DOCS))
    titles = [c.title for c in chunks]
    assert "Plans & Pricing > Pro plan" in titles
    assert all(c.text.startswith(c.title) for c in chunks)


def test_long_text_is_split_with_overlap():
    long_text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_documents([Document("d", "Long", long_text, "x.txt", {"format": "txt"})], max_chars=200, overlap=40)
    assert len(chunks) > 5
    assert all(len(c.text) < 300 for c in chunks)


def test_tokenizer_stems_and_drops_stopwords():
    assert tokenize("The notes are syncing") == ["note", "sync"]


def test_embeddings_are_normalized():
    vectors = HashingEmbedder().embed(["end-to-end encryption", "student discount"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_search_finds_relevant_chunk(store):
    top = store.search("student discount", k=1)[0]
    assert top.chunk.doc_id == "doc-pricing"


def test_save_and_load_roundtrip(store, tmp_path):
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, HashingEmbedder())
    assert len(loaded.chunks) == len(store.chunks)
    assert loaded.search("API rate limit", k=1)[0].chunk.doc_id == "doc-integrations"


def test_load_rejects_other_embedder(store, tmp_path):
    store.save(tmp_path)
    with pytest.raises(ValueError):
        VectorStore.load(tmp_path, HashingEmbedder(dim=256))


def test_prompt_contains_numbered_sources(store):
    results = store.search("refund", k=2)
    prompt = build_prompt("Can I get a refund?", results)
    assert "[1]" in prompt and "[2]" in prompt
    assert "Question: Can I get a refund?" in prompt


def test_offline_answer_cites_sources(store):
    answer = RagAssistant(store).ask("How much does the Pro plan cost?")
    assert answer.mode == "offline"
    assert "$6" in answer.text and "[1]" in answer.text


def test_guardrail_for_unrelated_question(store):
    answer = RagAssistant(store).ask("Who won the 1998 World Cup?")
    assert answer.text == NOT_FOUND


def test_claude_path_is_used_when_key_present(store, monkeypatch):
    calls = {}

    def fake_generate(system, user, **_):
        calls["system"], calls["user"] = system, user
        return "Pro costs $6 per month [1]."

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(llm, "generate", fake_generate)
    answer = RagAssistant(store).ask("How much is Pro?")
    assert answer.mode == "claude"
    assert "Sources:" in calls["user"]
    assert "ONLY using the numbered sources" in calls["system"]


def test_eval_set_quality(store):
    questions = load_questions(ROOT / "eval" / "questions.jsonl")
    assistant = RagAssistant(store)
    retrieval = evaluate_retrieval(assistant, [q for q in questions if q["expected_doc"]])
    answers = evaluate_answers(assistant, questions)
    # Floors for the offline setup; a real embedding model + Claude should score higher.
    assert retrieval["hit_at_k"] >= 0.9, json.dumps(retrieval["misses"], indent=2)
    assert answers["answer_accuracy"] >= 0.85, json.dumps(answers["failures"], indent=2)
