import streamlit as st
import ollama
import os
from ingest import ingest, CHROMA_PATH
from retriever import retrieve

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄")
st.title("📄 RAG Document Chatbot")
st.caption("Upload a PDF and chat with it")

# Session state
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    k_chunks = st.slider("Chunks to retrieve", min_value=1, max_value=6, value=3)
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()
    if st.session_state.collection_name:
        st.success(f"📂 Active: {st.session_state.collection_name}")
        st.info(f"📦 {st.session_state.chunk_count} chunks stored")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing PDF — loading, chunking, embedding..."):
        collection_name, chunk_count = ingest(file_path)
        st.session_state.collection_name = collection_name
        st.session_state.chunk_count = chunk_count
        st.session_state.chat_history = []

    st.success(f"✅ PDF processed — {chunk_count} chunks stored in ChromaDB!")

# Chat interface
if st.session_state.collection_name:
    st.divider()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    query = st.chat_input("Ask a question about the document...")

    if query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant chunks
                chunks = retrieve(query, st.session_state.collection_name, k=k_chunks)
                context = "\n\n".join(chunks)

                # Build conversation history string
                history_str = ""
                for msg in st.session_state.chat_history[-6:]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_str += f"{role}: {msg['content']}\n"

                prompt = f"""You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Conversation history:
{history_str}

Answer the latest user question:"""

                response = ollama.chat(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response["message"]["content"]

            st.write(answer)

            with st.expander("View retrieved chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk)

        # Add assistant message to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})