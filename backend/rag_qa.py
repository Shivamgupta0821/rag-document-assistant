from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# Load & Prepare Documents
def load_and_chunk(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# Build Vector Store
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)
# Build RAG Prompt
def build_prompt(context_docs, question):
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question:
{question}

Answer:
"""
    return prompt

# MAIN RAG PIPELINE
if __name__ == "__main__":
    # 1. Load & chunk documents
    chunks = load_and_chunk("../data/sample.pdf")

    # 2. Build vector store
    vectorstore = build_vectorstore(chunks)

    # 3. User question
    question = input("Ask a question: ")

    # 4. Retrieve relevant chunks
    retrieved_docs = vectorstore.similarity_search(question, k=3)

    # 5. Build prompt
    prompt = build_prompt(retrieved_docs, question)

    # 6. Call local LLM
    llm = ChatOllama(model="phi")
    response = llm.invoke(prompt)

    # 7. Output answer
    print("\n🤖 Answer:\n")
    print(response.content)

    # 8. Show sources
    print("\n Sources:\n")
    for doc in retrieved_docs:
        print(doc.metadata)
