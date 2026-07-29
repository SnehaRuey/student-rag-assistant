import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from src.document_loader import load_document
from src.embeddings import get_embeddings
from src.vector_store import build_vector_store
from src.rag_chain import get_llm, ask_question

load_dotenv()

st.set_page_config(page_title="Student RAG Assistant", page_icon="🎓")

# Keep data alive between clicks (this is the "backend memory" of the app)
if "store" not in st.session_state:
    st.session_state.store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- SIDEBAR (upload + process) ----------------
with st.sidebar:
    st.title("🎓 Student RAG Assistant")

    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        st.success("Groq API key: Configured ✅")
    else:
        st.error("Groq API key: Missing ❌")

    uploaded_file = st.file_uploader("Upload student CSV", type=["csv"])

    if st.button("Process Document"):
        if uploaded_file is None:
            st.error("Please upload a file first.")
        else:
            with st.spinner("Processing..."):
                # Save uploaded file temporarily so pandas can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                docs, df = load_document(tmp_path)
                embeddings = get_embeddings()
                store = build_vector_store(docs, embeddings)

                st.session_state.store = store
                st.success(f"Processed {len(docs)} student record(s)!")

    if st.button("🗑️ Clear / Reset"):
        st.session_state.store = None
        st.session_state.chat_history = []
        st.rerun()

# ---------------- MAIN PAGE (ask questions) ----------------
st.title("🎓 Student Information RAG Assistant")
st.caption("Upload student records and ask questions using natural language.")

if st.session_state.store is None:
    st.info("👈 Upload a CSV and click 'Process Document' to get started.")
else:
    question = st.text_input("Ask a question about a student:")

    if st.button("Ask"):
        if question.strip() == "":
            st.warning("Please type a question.")
        else:
            with st.spinner("Thinking..."):
                llm = get_llm()
                answer = ask_question(st.session_state.store, llm, question)
                st.session_state.chat_history.append((question, answer))

    st.divider()
    st.subheader("💬 Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)