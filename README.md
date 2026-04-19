# 📄 PDF Intelligence AI: Conversational RAG Assistant
A professional-grade **Retrieval-Augmented Generation (RAG)** application designed to query complex technical documents, including digital PDFs and legacy scanned images.

## 🚀 Live Demo
[**Visit JonathanHildreth.com**](https://jonathanhildreth.com)

## 🛠️ The Tech Stack
- **LLM & Inference:** [Groq Cloud](https://groq.com/) (Llama 3.1 8B Instant)
- **OCR Engine:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) & [Poppler](https://poppler.freedesktop.org/)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Telemetry & UI:** [Streamlit](https://streamlit.io/) & [Chart.js](https://www.chartjs.org/)

## ✨ Advanced Features
- **Hybrid OCR Processing:** Dynamically detects scanned PDFs/images and executes high-fidelity text extraction using the Tesseract engine.
- **Real-Time Telemetry:** Integrated Chart.js dashboard visualizing pipeline latency across Ingestion, OCR, Embedding, and Synthesis stages.
- **Conversational Awareness:** Full chat-history memory for context-aware follow-up questions.
- **Source Transparency:** Automated page-level citations with an "Evidence Expander" to view exact document fragments.
- **White-Label UI:** Custom-themed CSS interface with typewriter streaming and responsive sidebar navigation.

## 🧠 The 5-Step Pipeline
1. **Ingestion:** Extracts raw data from multi-page digital or scanned documents.
2. **Hybrid OCR:** Automatically processes non-digital formats into machine-readable text.
3. **Vectorization:** Maps text data into a high-dimensional vector space for semantic indexing.
4. **Semantic Search:** Real-time retrieval of relevant context from ChromaDB.
5. **Synthesis:** Groq-powered Llama 3.1 generates citation-backed responses in under 1 second.

## ⚙️ Setup & Deployment
### 1. System Dependencies (Windows)
To run OCR locally, you must install Tesseract and Poppler:
- [Tesseract OCR Binaries](https://github.com/UB-Mannheim/tesseract/wiki)
- [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)

### 2. Local Installation
```bash
git clone https://github.com/JonHildreth/PDF-AI-Chatbot.git
pip install -r requirements.txt
streamlit run app.py