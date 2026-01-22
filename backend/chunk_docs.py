from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    docs = load_documents("../data/sample.pdf")
    chunks = chunk_documents(docs)

    print(f"Total documents: {len(docs)}")
    print(f"Total chunks created: {len(chunks)}\n")

    print("Sample chunk")
    print(chunks[0].page_content)
    print("\nMetadata:", chunks[0].metadata)
