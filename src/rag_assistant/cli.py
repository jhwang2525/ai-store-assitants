"""Command line entry point.

    python -m rag_assistant ingest                      # index the demo docs in ./data/docs
    python -m rag_assistant ingest --docs ~/my-notes    # index your own folder (.md .txt .pdf)
    python -m rag_assistant ask "How much is the Pro plan?"
    python -m rag_assistant chat                        # interactive chat in the terminal
    python -m rag_assistant eval                        # run the evaluation set
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .chunking import chunk_documents
from .embeddings import get_embedder
from .evaluate import compare_search_modes, evaluate_answers, evaluate_retrieval, load_questions
from .loaders import load_folder
from .rag import RagAssistant
from .vectorstore import VectorStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = ROOT / "data" / "docs"
INDEX = ROOT / "index"


def build_index(docs_folder: str | Path | None, embedder_kind: str) -> VectorStore:
    embedder = get_embedder(embedder_kind)
    docs = load_folder(docs_folder or DEFAULT_DOCS)
    chunks = chunk_documents(docs)
    store = VectorStore(embedder)
    store.add(chunks)
    store.save(INDEX)
    print(f"Indexed {len(docs)} documents -> {len(chunks)} chunks using {embedder.name}")
    print(f"Saved to {INDEX}")
    return store


def load_assistant(embedder_kind: str, k: int = 4) -> RagAssistant:
    embedder = get_embedder(embedder_kind)
    if not (INDEX / "chunks.json").exists():
        print("No index yet, building it from the demo docs first...")
        store = build_index(None, embedder_kind)
    else:
        try:
            store = VectorStore.load(INDEX, embedder)
        except ValueError as exc:  # index was built with a different embedder
            print(f"{exc} Rebuilding...")
            store = build_index(None, embedder_kind)
    return RagAssistant(store, k=k)


def print_answer(answer) -> None:
    print(f"\n{answer.text}\n")
    for i, r in enumerate(answer.sources, 1):
        print(f"  [{i}] {r.chunk.title}  ({r.chunk.source}, similarity {r.score:.2f})")
    print(f"  mode: {answer.mode}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rag_assistant", description="Ask questions about your documents")
    parser.add_argument("--embedder", default=os.getenv("RAG_EMBEDDER", "auto"),
                        choices=["auto", "fastembed", "hashing"])
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="build the search index")
    p_ingest.add_argument("--docs", help="folder of .md / .txt / .pdf files (default: data/docs)")

    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=4, help="how many chunks to retrieve")

    sub.add_parser("chat", help="interactive chat")

    p_eval = sub.add_parser("eval", help="run the evaluation set")
    p_eval.add_argument("--file", default=str(ROOT / "eval" / "questions.jsonl"))

    args = parser.parse_args(argv)

    if args.command == "ingest":
        build_index(args.docs, args.embedder)

    elif args.command == "ask":
        print_answer(load_assistant(args.embedder, args.k).ask(args.question))

    elif args.command == "chat":
        assistant = load_assistant(args.embedder)
        print("Ask anything about your documents. Type 'quit' to exit.")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"quit", "exit", "q"}:
                break
            if q:
                print_answer(assistant.ask(q))

    elif args.command == "eval":
        assistant = load_assistant(args.embedder)
        questions = load_questions(args.file)
        answerable = [q for q in questions if q.get("expected_doc")]
        print(f"Evaluating {len(questions)} questions ({len(answerable)} answerable) with k={assistant.k}\n")

        print("Retrieval by search mode:")
        for mode, m in compare_search_modes(assistant, answerable).items():
            print(f"  {mode:8s}  hit@{assistant.k} = {m['hit_at_k']:.0%}   MRR = {m['mrr']:.2f}")

        retrieval = evaluate_retrieval(assistant, answerable)
        answers = evaluate_answers(assistant, questions)
        print(f"\nAnswer accuracy (keywords present): {answers['answer_accuracy']:.0%}")
        if retrieval["misses"]:
            print("\nRetrieval misses:")
            print(json.dumps(retrieval["misses"], indent=2))
        if answers["failures"]:
            print("\nAnswer failures:")
            print(json.dumps(answers["failures"], indent=2))


if __name__ == "__main__":
    main()
