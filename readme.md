# AI-Powered Web Research & RAG System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL%20%26%20Agents-green.svg)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-blueviolet.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

An end-to-end, autonomous research assistant that automates web research, information retrieval, vector indexing, structured report generation, and automated critique evaluation.

By combining **LangChain Agents**, **Groq LLM Inference**, **Tavily Web Search**, **BeautifulSoup Scraping**, **Hugging Face Embeddings**, **FAISS Vector Store**, and **Streamlit**, this project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline tailored for dynamic real-time web research.

---

## 🌟 Features

* **Web Search Agent:** Uses LangChain Agents and Groq to dynamically query and retrieve current web information using Tavily.
* **Web Scraping Pipeline:** Extracts readable text content while sanitizing noisy HTML elements (scripts, styles, headers, sidebars, footers).
* **Document Chunking & Vectorization:** Splits scraped text into manageable semantic chunks and converts them into dense vector representations using Hugging Face models.
* **FAISS Vector Search:** Enables fast local semantic similarity search to ground report generation on relevant evidence.
* **Grounding via RAG:** Eliminates hallucination risks by anchoring the report writer in scraped evidence rather than internal static model weights.
* **Writer Pipeline (LCEL):** Generates structured research reports using LangChain Expression Language (LCEL).
* **Critic Pipeline:** Evaluates generated reports across 4 dimensions: factual grounding, completeness, relevance, and source usage.
* **Interactive UI:** Built with Streamlit for research execution, deep-dive inspections of evidence, and visual review of feedback.

---

## 🏗️ Architecture

                              +-------------------+
                              |    User Input     |
                              +---------+---------+
                                        |
                                        v
                             +----------+----------+
                             |    Search Agent     |
                             |  (LangChain + Groq) |
                             +----------+----------+
                                        |
                                        v
                             +----------+----------+
                             |  Tavily Web Search  |
                             +----------+----------+
                                        |
                                        v
                             +----------+----------+
                             |    Web Scraping     |
                             |   (BeautifulSoup)   |
                             +----------+----------+
                                        |
                                        v
                             +----------+----------+
                             | Text Chunking &     |
                             | HF Vector Store     |
                             |     (FAISS)         |
                             +----------+----------+
                                        |
                                        v
                             +----------+----------+
                             | Semantic Retrieval  |
                             +----------+----------+
                                        |
                                        v
                             +----------+----------+
                             |    Writer Chain     |
                             |    (LCEL + Groq)    |
                             +----------+----------+
                                        |
                                        v
                             +----------+----------+
                             |    Critic Chain     |
                             | (Evaluation & Recs) |
                             +----------+----------+
                                        |
                                        v
                             +----------+----------+
                             | Output: Report &    |
                             |   Critic Scores     |
                             +---------------------+

---

## ⚙️ How It Works

1. **Research Topic:** User submits a target topic (e.g., *"Impact of Generative AI on Software Development"*).
2. **Search Agent:** The agent uses Tavily to search for high-ranking recent URLs, snippets, and page titles.
3. **Web Scraping:** `BeautifulSoup` cleans DOM trees by stripping script, style, header, footer, and navigation elements, extracting raw text content.
4. **Chunking & Vector Store Indexing:** Document texts are split into overlapping character chunks and passed to Hugging Face `sentence-transformers` for embedding generation, stored temporarily in an in-memory FAISS vector index.
5. **RAG Retrieval:** The original query runs vector similarity searches against FAISS to collect relevant evidence.
6. **Writer Chain (LCEL):** Groq processes retrieved contexts to assemble a structured markdown document containing **Introduction**, **Key Findings**, **Conclusion**, and **Sources**.
7. **Critic Evaluation:** The draft report is evaluated by a secondary Critic Chain assessing factual precision, coverage, and context adherence.

---

## 💻 Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | `Python 3.10+` | Core programming language |
| **Framework** | `LangChain` | Agent orchestration & chain assembly |
| **Pipeline Logic**| `LCEL` | Composable, transparent execution chains |
| **LLM Inference** | `Groq` | High-throughput LLM runtime |
| **Search Engine** | `Tavily API` | Search tool optimized for RAG workflows |
| **Web Parser** | `BeautifulSoup4` | HTML parsing and boilerplate stripping |
| **Embeddings** | `Hugging Face` | Open-source sentence transformers |
| **Vector Index** | `FAISS` | Fast, lightweight local vector search |
| **User Interface**| `Streamlit` | Interactive Web Dashboard |

---

## 📁 Project Structure

```bash
multi-agent-rs/
│
├── app.py              # Streamlit dashboard interface
├── pipeline.py         # End-to-end research orchestrator execution flow
├── agents.py           # LLM configurations, Search Agent, Writer & Critic Chains
├── tools.py            # Web search and BeautifulSoup scraping utilities
├── rag.py              # Document chunking, HF embeddings, & FAISS vector store
├── requirements.txt    # Project dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Rules for Git exclusion
└── README.md           # Project documentation