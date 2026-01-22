import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama



# Page config
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Document Assistant")
st.write("Upload a **text-based PDF** and ask questions from it.")



# Helper functions
def load_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)


def build_prompt(context_docs, question):
    context_text = "\n\n".join(doc.page_content[:800] for doc in context_docs)


    return f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question:
{question}

Answer:
"""

# UI Logic
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Load & chunk document
        chunks = load_and_chunk(pdf_path)

    # Debug / transparency
    st.write(f"📄 Total chunks created: {len(chunks)}")

    # SAFETY CHECK (VERY IMPORTANT)
    if len(chunks) == 0:
        st.error("❌ No readable text found in this PDF. Please upload a text-based PDF.")
        st.stop()

    # Build vector store
    vectorstore = build_vectorstore(chunks)
    st.success("✅ Document processed successfully!")

    # Question input
    question = st.text_input("Ask a question from the document:")

    if question:
        with st.spinner("Thinking..."):
            retrieved_docs = vectorstore.similarity_search(question, k=2)
            prompt = build_prompt(retrieved_docs, question)

            llm = ChatOllama(model="tinyllama")
            response = llm.invoke(prompt)

        st.subheader("🤖 Answer")
        st.write(response.content)

        st.subheader("📚 Sources")
        for doc in retrieved_docs:
            st.write(doc.metadata)
