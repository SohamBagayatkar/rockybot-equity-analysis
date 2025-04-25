import os
import shutil
import streamlit as st
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import SecretStr  # Import SecretStr for type safety

# Load API Key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("🚨 MISTRAL_API_KEY is missing! Set it in your environment.")
    raise ValueError("MISTRAL_API_KEY missing!")

mistral_api_key = SecretStr(api_key)  # Convert to SecretStr

FAISS_INDEX_PATH = "faiss_store_equity"

def build_faiss_index(articles):
    """
    Builds a FAISS index from extracted articles.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)  # Remove old index
        st.write("🗑 Cleared old FAISS index")

    if not articles:
        st.error("🚨 No articles to process!")
        return None

    st.write("🔄 Splitting and embedding text...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    docs = []

    for article in articles:
        split_docs = text_splitter.create_documents([article["content"]])
        for doc in split_docs:
            doc.metadata["source"] = article["source"]
        docs.extend(split_docs)

    if not docs:
        st.error("🚨 No text chunks generated!")
        return None

    st.write("🔄 Generating embeddings with Mistral AI...")
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key.get_secret_value())

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    st.write("✅ FAISS Index Saved!")

    return vectorstore

def load_faiss_index():
    """
    Loads the FAISS index if it exists.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error("🚨 FAISS index not found! Run analysis first.")
        return None

    st.write("✅ Loading FAISS index...")
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key.get_secret_value())
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
