import streamlit as st
import ollama
import os
from ingest import ingest, CHROMA_PATH
from retriever import retrieve

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .status-box {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .chunk-info {
        font-size: 0.85rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    k_chunks = st.slider(
        "Chunks to retrieve",
        min_value=1, max_value=6, value=3,
        help="More chunks = more context but slower response"
    )

    chunk_size = st.slider(
        "Chunk size (tokens)",
        min_value=200, max_value=1000, value=500, step=50,
        help="Smaller chunks = more precise retrieval"
    )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Reset document", use_container_width=True):
        st.session_state.collection_name = None
        st.session_state.chat_history = []
        st.session_state.chunk_count = 0
        st.session_state.pdf_name = None
        st.rerun()

    st.divider()

    if st.session_state.collection_name:
        st.success(f"📂 **{st.session_state.pdf_name}**")
        st.caption(f"📦 {st.session_state.chunk_count} chunks stored")
    else:
        st.info("No document loaded yet")

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Ollama · Streamlit")

# Main content
st.markdown('<p class="main-header">📄 RAG Document Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload any PDF and ask questions about its content</p>', unsafe_allow_html=True)

# Upload section
if not st.session_state.collection_name:
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Max 200MB"
    )

    if uploaded_file:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"⏳ Processing **{uploaded_file.name}** — loading, chunking, embedding..."):
            collection_name, chunk_count = ingest(file_path, chunk_size=chunk_size)
            st.session_state.collection_name = collection_name
            st.session_state.chunk_count = chunk_count
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []

        st.success(f"✅ **{uploaded_file.name}** processed — {chunk_count} chunks stored in ChromaDB!")
        st.rerun()

# Chat section
if st.session_state.collection_name:
    st.markdown(f"### 💬 Chatting with: `{st.session_state.pdf_name}`")
    st.divider()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    query = st.chat_input("Ask a question about the document...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document and generating answer..."):
                chunks = retrieve(query, st.session_state.collection_name, k=k_chunks)
                context = "\n\n".join(chunks)

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

Answer the latest user question concisely and clearly:"""

                response = ollama.chat(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response["message"]["content"]

            st.write(answer)

            with st.expander("🔎 View retrieved chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk)
                    st.divider()

        st.session_state.chat_history.append({"role": "assistant", "content": answer})