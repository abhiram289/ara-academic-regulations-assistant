# Retrieval-Augmented Academic Regulations Assistant

### Prompt Engineering and Generative AI - Course Project
### 


## Problem Statement
Build a conversational assistant that answers student queries on university regulations, curricula, and examination policies using Retrieval-Augmented Generation (RAG) over official documents.  
- **Citations:** Responses shall cite the exact source page (`[Page X]`).  
- **Guardrails:** Responses shall strictly decline questions not supported by the official documents.


## Project Architecture

```
prompt2/
├── data/
│   ├── raw_documents/
│   │   ├── university_academic_regulations_2024.pdf   
│   │   └── academic_regulations_text.txt              
│   └── test_set_50.json                               
├── indexes/
│   ├── faiss/                                      
│   └── chroma_db/                                 
├── results/
│   ├── chunking_comparison.json                       
│   ├── chunking_comparison.png                       
│   ├── evaluation_metrics.json                       
│   ├── evaluation_report.md                           
│   └── evaluation_charts.png                          
├── src/
│   ├── __init__.py
│   ├── document_generator.py                         
│   ├── ingestion.py                                   
│   ├── vector_store.py                                
│   ├── query_planner.py                              
│   ├── memory.py                                     
│   ├── rag_chain.py                                   
│   ├── llm_judge.py                                  
│   └── evaluation.py                                 
├── app.py                                           
├── main.py                                          
└── requirements.txt                                   
```


## How to Run

### 1. Run the Entire Project Suite (All Deliverables)
```bash
python main.py --all
```

### 2. Interactive CLI Chat
```bash
python main.py --chat
```

### 3. Interactive Streamlit Web UI
```bash
streamlit run app.py
```
