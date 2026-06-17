import streamlit as st
import ollama
import os
from ingest import ingest, CHROMA_PATH
from retriever import retrieve

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄")
st.title("📄 RAG Document Chatbot")
st.caption("Upload a PDF and ask questions about it")

# Session state for collection name
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing PDF — loading, chunking, embedding..."):
        collection_name, chunk_count = ingest(file_path)
        st.session_state.collection_name = collection_name

    st.success(f"✅ PDF processed — {chunk_count} chunks stored in ChromaDB!")

# Chat
if st.session_state.collection_name:
    st.divider()
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            chunks = retrieve(query, st.session_state.collection_name)
            context = "\n\n".join(chunks)

            prompt = f"""You are a helpful assistant. Use only the context below to answer the question. 
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {query}

Answer:"""

            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"]

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("View retrieved context chunks"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(chunk)