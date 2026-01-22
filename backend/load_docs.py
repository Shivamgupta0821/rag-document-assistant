from langchain_community.document_loaders import PyPDFLoader

def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents
if __name__ == "__main__":
    docs = load_documents("../data/sample.pdf")
    print(f"Total pages loaded: {len(docs)}\n")
    # Print first document for inspection
    print("FIRST DOCUMENT")
    print(docs[0].page_content[:500])  # first 500 characters
    print("\nMetadata:", docs[0].metadata)
