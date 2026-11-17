# Retrieval-Augmented Academic Regulations Assistant

> **Engineering Sem 5 — Prompt Engineering Project**  
> **Course Project:** 3. Retrieval-Augmented Academic Regulations Assistant

---

## 🎯 Problem Statement
Build a conversational assistant that answers student queries on university regulations, curricula, and examination policies using Retrieval-Augmented Generation (RAG) over official documents.  
- **Citations:** Responses shall cite the exact source page (`[Page X]`).  
- **Guardrails:** Responses shall strictly decline questions not supported by the official documents.

---

## 📦 Project Deliverables

### Deliverable 1: Ingestion Pipeline & Splitting Strategy Comparison
- **Document Loader:** Ingestion of multi-page academic handbooks using LangChain's `PyPDFLoader` with 1-indexed page metadata.
- **Three Splitting Strategies Compared:**
  1. **Fixed-Size Chunking (`CharacterTextSplitter`)**: Fixed length and overlap (500 chars), with 100% mid-sentence cutoff rate.
  2. **Recursive Chunking (`RecursiveCharacterTextSplitter`)**: Hierarchical splitting (`["\n\n", "Clause ", "\n", ". "]`), preserving clause structures with balanced lengths.
  3. **Semantic Chunking (`SentenceTransformer` Cosine Distance)**: Groups semantically coherent sentences using percentile distance thresholding, achieving minimal mid-sentence disruption (32.26%).
- **Outputs:**
  - Metrics: `results/chunking_comparison.json`
  - Visualization: `results/chunking_comparison.png`

### Deliverable 2: Vector Stores, Multi-Part Query Planning & Memory
- **Dual Vector Indexes:**
  - **FAISS (`IndexFlatIP`)**: Dense inner product similarity on 384-dimensional normalized embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
  - **ChromaDB (`PersistentClient`)**: Cosine similarity collection (`academic_regulations`) with persistent disk storage.
  - Unified interface allowing seamless switching between FAISS and ChromaDB.
- **Query Planning for Multi-Part Questions (`QueryPlanner`):**
  - Detects compound student queries (e.g., *"What is attendance required for ESA, and what happens under NSAR?"*).
  - Decomposes complex queries into atomic sub-queries for targeted parallel retrieval across distinct document sections.
  - Merges and deduplicates context before grounded generation.
- **Conversation Memory (`UnifiedConversationMemory`):**
  - **Buffer Memory (`ConversationBufferMemory`)**: Retains immediate recent dialogue turns verbatim.
  - **Summary Memory (`ConversationSummaryMemory`)**: Progressively summarizes older turns to retain long context within token budgets.
- **Page Citation & Refusal Guardrails:**
  - Guarantees factual assertions cite `[Page X]`.
  - Declines out-of-scope/unsupported queries with:  
    `"I decline to answer: The provided university academic regulations do not contain information to support this question."`

### Deliverable 3: 50-Question Benchmark & LLM-as-Judge Evaluation
- **Benchmark Dataset (`data/test_set_50.json`):**
  - **35 In-Scope Questions:** Covering all 10 handbook chapters (single-hop and multi-part queries).
  - **15 Out-of-Scope Questions:** Testing strict refusal on hostel food, bus routes, sports trials, stocks, coding, etc.
- **Key Evaluation Metrics:**

| Metric | FAISS Backend | ChromaDB Backend | Target Benchmark | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Retrieval Recall@1** | **100.0%** | **100.0%** | >= 75.0% | **PASS** |
| **Retrieval Recall@3** | **100.0%** | **100.0%** | >= 85.0% | **PASS** |
| **Retrieval Recall@5** | **100.0%** | **100.0%** | >= 90.0% | **PASS** |
| **LLM-as-Judge Correctness** | **98.4%** | **98.4%** | >= 85.0% | **PASS** |
| **LLM-as-Judge Faithfulness** | **96.7%** | **96.7%** | >= 90.0% | **PASS** |
| **Out-of-Scope Refusal Accuracy** | **100.0%** | **100.0%** | >= 95.0% | **OPTIMAL** |
| **In-Scope False Refusal Rate** | **0.0%** | **0.0%** | <= 5.0% | **PASS** |
| **Execution Latency** | **1.8s** | **2.6s** | Real-time | **READY** |

- **Outputs:**
  - Metrics JSON: `results/evaluation_metrics.json`
  - Markdown Report: `results/evaluation_report.md`
  - Visualization Plots: `results/evaluation_charts.png`

---

## 📂 Project Architecture

```
prompt2/
├── data/
│   ├── raw_documents/
│   │   ├── university_academic_regulations_2024.pdf   # 10-page official handbook
│   │   └── academic_regulations_text.txt              # Text mirror with page tags
│   └── test_set_50.json                               # 50-question ground truth benchmark
├── indexes/
│   ├── faiss/                                         # FAISS index binary & metadata
│   └── chroma_db/                                     # Persistent ChromaDB collection
├── results/
│   ├── chunking_comparison.json                       # Deliverable 1 statistics
│   ├── chunking_comparison.png                        # Deliverable 1 length distribution plots
│   ├── evaluation_metrics.json                        # Deliverable 3 full benchmark numbers
│   ├── evaluation_report.md                           # Deliverable 3 comprehensive report
│   └── evaluation_charts.png                          # Deliverable 3 Recall@k & confusion matrix
├── src/
│   ├── __init__.py
│   ├── document_generator.py                          # Official PDF generator with page rules
│   ├── ingestion.py                                   # Deliverable 1: Ingestion & Splitting comparison
│   ├── vector_store.py                                # Deliverable 2: FAISS & ChromaDB managers
│   ├── query_planner.py                               # Deliverable 2: Multi-part query decomposer
│   ├── memory.py                                      # Deliverable 2: Buffer & Summary memory
│   ├── rag_chain.py                                   # Deliverable 2: RAG pipeline with citations & refusal
│   ├── llm_judge.py                                   # Deliverable 3: LLM-as-Judge evaluator
│   └── evaluation.py                                  # Deliverable 3: 50-question benchmark runner
├── app.py                                             # Streamlit Web UI Application
├── main.py                                            # CLI master runner
└── requirements.txt                                   # Python dependencies
```

---

## 🚀 How to Run

### 1. Run the Entire Project Suite (All 3 Deliverables)
```bash
python main.py --all
```

### 2. Run Individual Deliverables
- **Deliverable 1 (Ingestion & Text-Splitting):**
  ```bash
  python main.py --deliverable 1
  ```
- **Deliverable 2 (Vector Stores & RAG Chain):**
  ```bash
  python main.py --deliverable 2 --backend faiss
  python main.py --deliverable 2 --backend chroma
  ```
- **Deliverable 3 (50-Question Benchmark Evaluation):**
  ```bash
  python main.py --deliverable 3
  ```

### 3. Interactive CLI Chat
```bash
python main.py --chat
```
*(Type `backend chroma` or `backend faiss` within the chat to toggle vector stores).*

### 4. Interactive Streamlit Web UI
```bash
streamlit run app.py
```
Visit `http://localhost:8501` to use the conversational UI with live page citation badges, query plan visualizers, and memory inspection.
