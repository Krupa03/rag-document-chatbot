import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"
DATA_PATH = "data"

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def embed_and_store(chunks, file_name):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=file_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    vectorstore.add_texts(chunks)
    print(f"Stored {len(chunks)} chunks from {file_name}")

def ingest(file_path):
    file_name = os.path.basename(file_path).replace(".pdf", "")
    print(f"Loading {file_path}...")
    text = load_pdf(file_path)
    print(f"Chunking...")
    chunks = chunk_text(text)
    print(f"Embedding and storing {len(chunks)} chunks...")
    embed_and_store(chunks, file_name)
    print("Done!")

if __name__ == "__main__":
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            ingest(os.path.join(DATA_PATH, file))