import streamlit as st
from dotenv import load_dotenv
import os
import time
from PyPDF2 import PdfReader

# LangChain & AI Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

load_dotenv()

# --- 1. PROFESSIONAL UI CONFIGURATION ---
st.set_page_config(page_title="PDF AI Chatbot", page_icon="📄", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-image: linear-gradient(#2e3440, #1a1c23); border-right: 1px solid #4c566a; }
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; margin-bottom: 0px;
    }
    .stChatMessage { background-color: #1e2530; border-radius: 15px; border: 1px solid #3b4252; margin-bottom: 10px; }
    .stTab { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HELPERS & PDF PROCESSING ---

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
    return Chroma.from_documents(documents=text_chunks, embedding=embeddings)

def get_rag_chain(vectorstore):
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Contextualize question (Memory)
    context_q_system_prompt = "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer, just reformulate."
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    # Answer prompt
    system_prompt = (
        "You are a professional assistant. Use the provided context to answer the question. "
        "If unsure, say you don't know. Always cite the Source File and Page Number at the end.\n\n"
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    return create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))

# --- 3. SESSION STATE INITIALIZATION ---
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- 4. SIDEBAR (BRANDING & UPLOAD) ---
with st.sidebar:
    st.markdown("### 👨‍💻 Jon's Portfolio")
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
                st.success(f"Indexed {len(uploaded_files)} files!")
        else:
            st.error("Upload a PDF first.")

    # Suggested Questions Logic
    if st.session_state.vectorstore:
        st.divider()
        if st.button("💡 Suggest Questions"):
            temp_llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")
            sample = st.session_state.vectorstore.similarity_search("important", k=1)
            res = temp_llm.invoke(f"Suggest 3 short questions based on this: {sample[0].page_content[:500]}")
            st.info(res.content)

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    # Download Log
    if st.session_state.chat_history:
        log = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
        st.download_button("💾 Download Log", log, "chat_log.txt")

# --- 5. MAIN CONTENT AREA ---
st.markdown('<h1 class="main-title">📄 PDF AI Chatbot</h1>', unsafe_allow_html=True)

if st.session_state.vectorstore is None:
    st.markdown("""
        <div style="background-color: #1e2530; padding: 25px; border-radius: 12px; border: 1px solid #3b4252;">
            <h3>Welcome to your Technical Document Assistant</h3>
            <p>Upload multi-page PDFs to interact with them using Retrieval-Augmented Generation (RAG).</p>
            <hr style="border-color: #4c566a;">
            <b>🛠️ Technical Pipeline:</b>
            <ul>
                <li><b>Engine:</b> Llama 3.1-8B Instant (Groq Inference)</li>
                <li><b>Database:</b> ChromaDB (Vector Embeddings)</li>
                <li><b>Context Window:</b> Full chat history awareness</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
else:
    # 1. Create the Tabs
    tab1, tab2 = st.tabs(["💬 AI Chat Assistant", "📊 Pipeline Analytics"])

    # 2. Display the History inside Tab 1
    with tab1:
        for message in st.session_state.chat_history:
            avatar = "👤" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    # 3. Display the Stats inside Tab 2
    with tab2:
        st.subheader("🛠️ Vector Database Metrics")
        c1, c2, c3 = st.columns(3)
        total_chunks = len(st.session_state.vectorstore.get()['ids'])
        c1.metric("Total Files", len(uploaded_files))
        c2.metric("Total Vector Chunks", total_chunks)
        c3.metric("Embeddings", "HuggingFace/all-MiniLM")
        
        st.write("**Processed Files:**")
        for f in uploaded_files:
            st.code(f"📄 {f.name} - {f.size//1024} KB")

    # --- 4. INPUT LOGIC (MOVED OUTSIDE OF TABS) ---
    # By placing this here, it will always stay at the bottom of the screen.
    if user_query := st.chat_input("Ask a question about your documents..."):
        
        # We tell Streamlit to render the new messages inside Tab 1 specifically
        with tab1:
            # Show User Question
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)
            
            # Generate Assistant Response
            with st.chat_message("assistant", avatar="🤖"):
                start_time = time.perf_counter()
                with st.spinner("Consulting Vector Store..."):
                    chain = get_rag_chain(st.session_state.vectorstore)
                    response = chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
                    answer = response["answer"]
                
                st.caption(f"🚀 Response generated in {time.perf_counter() - start_time:.2f}s")
                
                # Typewriter streaming
                def stream_text():
                    for word in answer.split(" "):
                        yield word + " "
                        time.sleep(0.04)
                st.write_stream(stream_text)

                with st.expander("🔍 View Context Sources"):
                    for doc in response["context"]:
                        st.write(f"**From: {doc.metadata['source']} (Page {doc.metadata['page']})**")
                        st.info(doc.page_content)
        
        # 5. Save to history and force a rerun
        # Rerunning ensures the chat renders cleanly back in the history loop above
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()