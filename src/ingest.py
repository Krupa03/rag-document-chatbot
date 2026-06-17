import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"
DATA_PATH = "data"

def load_pdf(file_path):
    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def embed_and_store(chunks, collection_name):
    print(f"Generating embeddings with HuggingFace all-MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    vectorstore.add_documents(chunks)
    count = vectorstore._collection.count()
    print(f"Stored {count} chunks in ChromaDB collection '{collection_name}'")
    return count

def ingest(file_path):
    collection_name = os.path.basename(file_path).replace(".pdf", "").replace(" ", "_")
    documents = load_pdf(file_path)
    chunks = chunk_documents(documents)
    count = embed_and_store(chunks, collection_name)
    return collection_name, count

if __name__ == "__main__":
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            ingest(os.path.join(DATA_PATH, file))