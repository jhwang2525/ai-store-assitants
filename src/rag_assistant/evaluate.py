"""Measure how good the assistant is instead of guessing.

Each line in eval/questions.jsonl has a question, the doc it should be answered
from, and a few words the answer should contain. We report:

* hit@k  - % of questions where the right doc is in the top-k retrieved chunks
* MRR    - mean reciprocal rank: 1.0 if the right doc is always ranked first
* answer - % of answers that contain all the expected keywords
"""

from __future__ import annotations

import json
from pathlib import Path

from .rag import RagAssistant


def load_questions(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate_retrieval(assistant: RagAssistant, questions: list[dict]) -> dict:
    hits, rr = 0, 0.0
    misses = []
    for q in questions:
        results = assistant.retrieve(q["question"])
        doc_ids = [r.chunk.doc_id for r in results]
        if q["expected_doc"] in doc_ids:
            hits += 1
            rr += 1.0 / (doc_ids.index(q["expected_doc"]) + 1)
        else:
            misses.append({"question": q["question"], "expected": q["expected_doc"], "got": doc_ids[:2]})
    n = len(questions) or 1
    return {"hit_at_k": hits / n, "mrr": rr / n, "misses": misses}


def evaluate_answers(assistant: RagAssistant, questions: list[dict]) -> dict:
    passed = 0
    failures = []
    for q in questions:
        answer = assistant.ask(q["question"]).text.lower()
        missing = [kw for kw in q.get("must_include", []) if kw.lower() not in answer]
        if missing:
            failures.append({"question": q["question"], "missing": missing, "answer": answer[:160]})
        else:
            passed += 1
    n = len(questions) or 1
    return {"answer_accuracy": passed / n, "failures": failures}


def compare_search_modes(assistant: RagAssistant, questions: list[dict]) -> dict:
    original = assistant.search_mode
    table = {}
    for mode in ("vector", "keyword", "hybrid"):
        assistant.search_mode = mode
        r = evaluate_retrieval(assistant, questions)
        table[mode] = {"hit_at_k": r["hit_at_k"], "mrr": r["mrr"]}
    assistant.search_mode = original
    return table
