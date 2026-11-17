"""
Deliverable 2: Vector Store Management with FAISS and ChromaDB.

Provides:
1. Unified embedding generation via SentenceTransformers (all-MiniLM-L6-v2).
2. FAISS vector store indexing and persistence (Cosine / Inner Product similarity).
3. ChromaDB vector store indexing and persistence (PersistentClient).
4. Standardized retrieval interface returning LangChain Document objects with page metadata.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import faiss
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

CACHE_MODEL_PATH = r"C:\Users\abhir\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_MODEL = CACHE_MODEL_PATH if os.path.exists(CACHE_MODEL_PATH) else "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingManager:
    """Singleton-style wrapper for SentenceTransformers embedding model."""
    _instance = None
    _model = None

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer(model_name)
        return cls._model

    @classmethod
    def embed_texts(cls, texts: List[str]) -> np.ndarray:
        model = cls.get_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    @classmethod
    def embed_query(cls, query: str) -> np.ndarray:
        model = cls.get_model()
        emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return np.array(emb[0], dtype=np.float32)


class FAISSVectorStore:
    """FAISS vector store with normalized inner product (cosine similarity) and persistence."""
    def __init__(self, index_dir: str = "indexes/faiss"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "index.faiss"
        self.meta_file = self.index_dir / "metadata.json"
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Document] = []
        self.dim = 384

    def build_from_documents(self, docs: List[Document]):
        """Builds and persists a new FAISS index from documents."""
        self.documents = docs
        texts = [d.page_content for d in docs]
        vectors = EmbeddingManager.embed_texts(texts)
        self.dim = vectors.shape[1]
        
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.save()
        print(f"[FAISS] Built index with {len(docs)} chunks (dim={self.dim}).")

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
            serializable_docs = [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self.documents
            ]
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(serializable_docs, f, indent=2)

    def load(self) -> bool:
        """Loads FAISS index and metadata if available."""
        if self.index_file.exists() and self.meta_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.meta_file, "r", encoding="utf-8") as f:
                raw_docs = json.load(f)
            self.documents = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in raw_docs
            ]
            print(f"[FAISS] Loaded {len(self.documents)} chunks from {self.index_file}")
            return True
        return False

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Searches top-k similar documents. Returns list of (Document, score)."""
        if self.index is None or len(self.documents) == 0:
            if not self.load():
                return []
        
        q_vec = EmbeddingManager.embed_query(query).reshape(1, -1)
        k = min(k, len(self.documents))
        scores, indices = self.index.search(q_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results


class ChromaVectorStore:
    """ChromaDB persistent vector store."""
    def __init__(self, persist_dir: str = "indexes/chroma_db", collection_name: str = "academic_regulations"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def build_from_documents(self, docs: List[Document]):
        """Embeds and indexes documents in ChromaDB."""
        # Clear existing entries in collection
        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)
        
        texts = [d.page_content for d in docs]
        embeddings = EmbeddingManager.embed_texts(texts).tolist()
        ids = [f"chunk_{i}" for i in range(len(docs))]
        metadatas = [d.metadata for d in docs]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"[ChromaDB] Built collection '{self.collection_name}' with {len(docs)} chunks.")

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Searches top-k similar documents in ChromaDB."""
        q_vec = EmbeddingManager.embed_query(query).tolist()
        total_count = self.collection.count()
        if total_count == 0:
            return []
        
        k = min(k, total_count)
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            for doc_text, meta, dist in zip(docs, metas, dists):
                # Cosine distance in Chroma: sim = 1 - dist
                score = 1.0 - float(dist)
                formatted_results.append((
                    Document(page_content=doc_text, metadata=meta),
                    score
                ))
        return formatted_results


class AcademicVectorIndexManager:
    """Unified manager that supports switching between FAISS and ChromaDB backends."""
    def __init__(self, backend: str = "faiss"):
        self.backend = backend.lower()
        self.faiss_store = FAISSVectorStore()
        self.chroma_store = ChromaVectorStore()

    def build_indexes(self, docs: List[Document]):
        """Builds both FAISS and ChromaDB indexes simultaneously."""
        self.faiss_store.build_from_documents(docs)
        self.chroma_store.build_from_documents(docs)

    def search(self, query: str, k: int = 4, backend: Optional[str] = None) -> List[Tuple[Document, float]]:
        active_backend = (backend or self.backend).lower()
        if active_backend == "faiss":
            return self.faiss_store.similarity_search(query, k=k)
        elif active_backend == "chroma" or active_backend == "chromadb":
            return self.chroma_store.similarity_search(query, k=k)
        else:
            raise ValueError(f"Unknown backend: {active_backend}. Choose 'faiss' or 'chroma'.")


if __name__ == "__main__":
    from src.ingestion import DocumentIngestionPipeline
    
    print("Testing Vector Store Indexing with FAISS & ChromaDB...")
    pipeline = DocumentIngestionPipeline("data/raw_documents/university_academic_regulations_2024.pdf")
    raw_docs = pipeline.load_pdf()
    recursive_chunks = pipeline.run_recursive_splitter(raw_docs)
    
    manager = AcademicVectorIndexManager()
    manager.build_indexes(recursive_chunks)
    
    test_query = "What is the minimum attendance required and what is the penalty for NSAR?"
    print(f"\nQuery: {test_query}")
    
    print("\n--- FAISS Results ---")
    faiss_res = manager.search(test_query, k=2, backend="faiss")
    for doc, score in faiss_res:
        print(f"Score: {score:.4f} | Page: {doc.metadata.get('page')} | Snippet: {doc.page_content[:120]}...")
        
    print("\n--- ChromaDB Results ---")
    chroma_res = manager.search(test_query, k=2, backend="chroma")
    for doc, score in chroma_res:
        print(f"Score: {score:.4f} | Page: {doc.metadata.get('page')} | Snippet: {doc.page_content[:120]}...")
