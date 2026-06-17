import streamlit as st
import ollama
import os
from ingest import ingest, CHROMA_PATH
from retriever import retrieve

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄")
st.title("📄 RAG Document Chatbot")
st.caption("Upload a PDF and ask questions about it")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    collection_name = uploaded_file.name.replace(".pdf", "")
    
    with st.spinner("Processing PDF..."):
        ingest(file_path)
    st.success(f"PDF processed and stored in ChromaDB!")

    # Chat
    st.divider()
    query = st.text_input("Ask a question about the document:")
    
    if query:
        with st.spinner("Retrieving and generating answer..."):
            # Retrieve relevant chunks
            chunks = retrieve(query, collection_name)
            context = "\n\n".join(chunks)
            
            # Generate answer with Ollama
            prompt = f"""Use the following context to answer the question.
            
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
        
        with st.expander("View retrieved context"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(chunk)