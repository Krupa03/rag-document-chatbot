from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"

def get_retriever(collection_name, k=3):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})

def retrieve(query, collection_name, k=3):
    retriever = get_retriever(collection_name, k)
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]