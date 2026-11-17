"""Streamlit chat UI.  Run:  streamlit run app.py"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_assistant import llm  # noqa: E402
from rag_assistant.cli import load_assistant  # noqa: E402

st.set_page_config(page_title="RAG Assistant", page_icon="📚")


@st.cache_resource
def get_assistant():
    return load_assistant(os.getenv("RAG_EMBEDDER", "auto"))


assistant = get_assistant()

with st.sidebar:
    st.header("Settings")
    assistant.k = st.slider("Chunks to retrieve (k)", 1, 8, 4)
    assistant.search_mode = st.radio("Search mode", ["hybrid", "vector", "keyword"])
    st.caption(f"Embedder: `{assistant.store.embedder.name}`")
    st.caption(f"Indexed chunks: {len(assistant.store.chunks)}")
    st.caption("LLM: Claude" if llm.has_api_key() else "LLM: offline mode (set ANTHROPIC_API_KEY for Claude)")
    st.divider()
    st.markdown("**Try asking** (demo docs)")
    for example in [
        "How much does the Pro plan cost?",
        "Can I use the mobile app offline?",
        "I forgot my encryption password, what now?",
        "What is the API rate limit?",
    ]:
        st.markdown(f"- {example}")

st.title("📚 RAG Assistant")
st.caption("Answers come only from the indexed documents, with sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            answer = assistant.ask(question)
        st.markdown(answer.text)
        if answer.sources:
            with st.expander("Sources"):
                for i, r in enumerate(answer.sources, 1):
                    st.markdown(f"**[{i}] {r.chunk.title}** · `{r.chunk.source}` · similarity {r.score:.2f}")
                    st.text(r.chunk.text[:400])
    st.session_state.messages.append({"role": "assistant", "content": answer.text})
