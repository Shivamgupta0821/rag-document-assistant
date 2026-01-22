from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables (.env)
load_dotenv()


def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vectorstore


if __name__ == "__main__":
    docs = load_documents("../data/sample.pdf")
    chunks = chunk_documents(docs)

    vectorstore = build_vectorstore(chunks)

    print("Vector store built successfully!")
    print(f"Total vectors stored: {vectorstore.index.ntotal}")
