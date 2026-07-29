from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_vector_store(text_list, embeddings):
    documents = [Document(page_content=text) for text in text_list]
    store = FAISS.from_documents(documents, embeddings)
    return store