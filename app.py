import streamlit as st
from dotenv import load_dotenv
import os
import time
from PyPDF2 import PdfReader

# --- LangChain & AI Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# These specific paths are required to avoid import errors in newer LangChain versions
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

# Load environme

# --- 1. UI CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(
    page_title="PDF AI Chatbot", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded" # This ensures it starts open
)

# --- 2. CLEAN CUSTOM CSS (Restores Native Sidebar Controls) ---
st.markdown("""
    <style>
    /* MAIN APP STYLES */
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* SIDEBAR STYLES */
    [data-testid="stSidebar"] { 
        background-image: linear-gradient(#2e3440, #1a1c23); 
        border-right: 1px solid #4c566a; 
    }

    /* TITLE STYLES */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; margin-bottom: 20px;
    }

    /* CHAT BUBBLE STYLES */
    .stChatMessage { 
        background-color: #1e2530; 
        border-radius: 15px; 
        border: 1px solid #3b4252; 
        margin-bottom: 10px; 
    }
    
    /* Ensure the sidebar toggle button is visible against the dark background */
    button[data-testid="stSidebarCollapseButton"] {
        color: #4facfe !important;
    }

    .stTab { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HELPERS & PDF PROCESSING ---

def get_pdf_text(pdf_docs):
    all_documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                all_documents.append(Document(
                    page_content=text, 
                    metadata={"source": pdf.name, "page": i + 1}
                ))
    return all_documents

def get_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Using a unique collection name helps prevent conflicts in shared memory
    return Chroma.from_documents(
        documents=text_chunks, 
        embedding=embeddings,
        collection_name="my_pdf_collection"
    )

def get_rag_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"), 
        model_name="llama-3.1-8b-instant", 
        temperature=0
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    context_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer, just reformulate."
    )
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    system_prompt = (
        "You are a professional technical assistant. Use the provided context to answer the question. "
        "If unsure, say you don't know. Always cite the Source File and Page Number at the end of your response.\n\n"
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 4. SESSION STATE INITIALIZATION ---
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- 5. SIDEBAR (PORTFOLIO BRANDING & CONTROLS) ---
with st.sidebar:
    st.markdown("### 👨‍💻 Portfolio Project")
    st.markdown("""
        [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=flat&logo=github)](https://github.com/JonHildreth)
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/jonathan-hildreth-linked)
    """)
    st.divider()
    
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
    
    if st.button("🚀 Process Documents"):
        if uploaded_files:
            with st.spinner("Analyzing and Indexing..."):
                raw_docs = get_pdf_text(uploaded_files)
                chunks = get_text_chunks(raw_docs)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.success(f"Successfully indexed {len(uploaded_files)} files!")
        else:
            st.error("Please upload at least one PDF.")

    if st.session_state.vectorstore:
        st.divider()
        st.subheader("💡 Need an idea?")
        if st.button("Suggest Questions"):
            with st.spinner("Brainstorming..."):
                temp_llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")
                sample = st.session_state.vectorstore.similarity_search("important information", k=1)
                res = temp_llm.invoke(f"Suggest 3 short, interesting questions based on this: {sample[0].page_content[:500]}")
                st.info(res.content)

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.chat_history:
        log = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
        st.download_button("💾 Download Chat Log", log, "chat_history.txt")

# --- 6. MAIN CONTENT AREA ---
st.markdown('<h1 class="main-title">📄 PDF AI Chatbot</h1>', unsafe_allow_html=True)

if st.session_state.vectorstore is None:
    # PRO LANDING PAGE
    st.markdown("""
        <div style="background-color: #1e2530; padding: 25px; border-radius: 12px; border: 1px solid #3b4252;">
            <h3>Welcome to your Technical Document Assistant</h3>
            <p>This is a professional RAG application designed to query complex multi-page PDF documents.</p>
            <hr style="border-color: #4c566a;">
            <b>🛠️ Project Architecture:</b>
            <ul>
                <li><b>Engine:</b> Llama 3.1-8B Instant (Groq Cloud)</li>
                <li><b>Database:</b> ChromaDB (Local Vector Store)</li>
                <li><b>Embeddings:</b> HuggingFace (MiniLM-L6-v2)</li>
                <li><b>Memory:</b> Full conversation history awareness</li>
            </ul>
            <p><i>👈 Upload your PDF files in the sidebar to begin.</i></p>
        </div>
    """, unsafe_allow_html=True)
else:
    tab1, tab2 = st.tabs(["💬 AI Chat Assistant", "📊 Pipeline Analytics"])

    with tab1:
        # Display Message History
        for message in st.session_state.chat_history:
            avatar = "👤" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    with tab2:
        st.subheader("🛠️ Vector Database Performance")
        c1, c2, c3 = st.columns(3)
        total_chunks = len(st.session_state.vectorstore.get()['ids'])
        c1.metric("Total Files", len(uploaded_files))
        c2.metric("Total Chunks", total_chunks)
        c3.metric("Embeddings", "MiniLM-L6-v2")
        
        st.write("**Processed Files Index:**")
        for f in uploaded_files:
            st.code(f"📄 {f.name} - {f.size//1024} KB")

    # --- CHAT INPUT (Anchored to bottom) ---
    if user_query := st.chat_input("Ask a question about your documents..."):
        with tab1:
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)
            
            with st.chat_message("assistant", avatar="🤖"):
                start_time = time.perf_counter()
                with st.spinner("Searching..."):
                    chain = get_rag_chain(st.session_state.vectorstore)
                    response = chain.invoke({
                        "input": user_query, 
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response["answer"]
                
                # Metrics
                st.caption(f"🚀 Response generated in {time.perf_counter() - start_time:.2f}s")
                
                # Typewriter Streaming
                def stream_text():
                    for word in answer.split(" "):
                        yield word + " "
                        time.sleep(0.04)
                st.write_stream(stream_text)

                # Source Transparency
                with st.expander("🔍 View Source Evidence"):
                    for doc in response["context"]:
                        st.write(f"**From: {doc.metadata['source']} (Page {doc.metadata['page']})**")
                        st.info(doc.page_content)
            
            # Save and Refresh
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()