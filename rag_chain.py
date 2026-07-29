import os
from langchain_groq import ChatGroq


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )


def ask_question(store, llm, question):
    docs = store.similarity_search(question, k=5)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a Student Information Assistant.

Answer the user's question using ONLY the student information
provided in the context below.

If the requested information is not available in the context,
say "Information not found in the uploaded student records."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content