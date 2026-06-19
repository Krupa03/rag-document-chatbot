from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"

def get_vectorstore(collection_name):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vectorstore

def retrieve(query, collection_name, k=3):
    vectorstore = get_vectorstore(collection_name)
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]