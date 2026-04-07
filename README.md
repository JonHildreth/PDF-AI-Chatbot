# PDF Intelligence AI: Conversational RAG Assistant
A professional-grade Retrieval-Augmented Generation (RAG) application that allows users to have context-aware conversations with multiple PDF documents simultaneously.

## Live Demo
[JonathanHildreth.com]

## The Tech Stack
- **Inference Engine:** [Groq Cloud](https://groq.com/) (Llama 3.1 8B)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **UI Framework:** [Streamlit](https://streamlit.io/)

## Key Features
- **Multi-PDF Support:** Upload and index multiple technical documents at once.
- **Conversational Memory:** AI understands follow-up questions (e.g., "What about the tires?" after asking about oil).
- **Source Transparency:** Answers include precise page citations and a "View Context" expander to see original snippets.
- **High Performance:** Sub-1-second response times powered by Groq's LPU technology.
- **Professional UI:** Includes typewriter streaming effects, processing metrics, and a "Suggested Questions" brainstormer.