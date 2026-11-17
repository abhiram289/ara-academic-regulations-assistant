"""
Deliverable 1: Document Ingestion Pipeline & Text-Splitting Strategy Comparison.

Implements:
1. LangChain PyPDFLoader document ingestion with page-accurate metadata.
2. Comparison of three text-splitting strategies:
   - Fixed-size chunking (CharacterTextSplitter)
   - Recursive chunking (RecursiveCharacterTextSplitter)
   - Semantic text-splitting (Sentence-similarity distance thresholding)
3. Quantitative statistical analysis & visual comparison plots.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


CACHE_MODEL_PATH = r"C:\Users\abhir\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_MODEL = CACHE_MODEL_PATH if os.path.exists(CACHE_MODEL_PATH) else "sentence-transformers/all-MiniLM-L6-v2"

class SemanticTextSplitter:
    """
    Semantic text splitter that groups sentences by semantic similarity
    using sentence embedding cosine distances and percentile thresholding.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 threshold_percentile: float = 75.0, min_chunk_len: int = 150):
        self.model = SentenceTransformer(model_name)
        self.threshold_percentile = threshold_percentile
        self.min_chunk_len = min_chunk_len

    def _split_into_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries preserving abbreviations where feasible
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        split_docs = []
        for doc in documents:
            text = doc.page_content.strip()
            if not text:
                continue
            
            sentences = self._split_into_sentences(text)
            if len(sentences) <= 1:
                split_docs.append(Document(
                    page_content=text,
                    metadata=dict(doc.metadata, chunk_type="semantic", chunk_id=len(split_docs))
                ))
                continue

            # Embed sentences
            embeddings = self.model.encode(sentences, normalize_embeddings=True)
            
            # Compute cosine distances between consecutive sentences
            distances = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i+1])
                dist = 1.0 - float(sim)
                distances.append(dist)

            threshold = float(np.percentile(distances, self.threshold_percentile)) if distances else 0.5
            
            # Group into chunks
            current_chunk = [sentences[0]]
            for i, dist in enumerate(distances):
                if dist > threshold and len(" ".join(current_chunk)) >= self.min_chunk_len:
                    # Breakpoint triggered
                    chunk_text = " ".join(current_chunk)
                    split_docs.append(Document(
                        page_content=chunk_text,
                        metadata=dict(doc.metadata, chunk_type="semantic", chunk_id=len(split_docs))
                    ))
                    current_chunk = [sentences[i+1]]
                else:
                    current_chunk.append(sentences[i+1])

            if current_chunk:
                chunk_text = " ".join(current_chunk)
                split_docs.append(Document(
                    page_content=chunk_text,
                    metadata=dict(doc.metadata, chunk_type="semantic", chunk_id=len(split_docs))
                ))

        return split_docs


class DocumentIngestionPipeline:
    """Manages document loading and text-splitting comparisons."""
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

    def load_pdf(self) -> List[Document]:
        """Loads PDF using LangChain PyPDFLoader and ensures 1-indexed page metadata."""
        loader = PyPDFLoader(str(self.pdf_path))
        docs = loader.load()
        for idx, doc in enumerate(docs):
            # PyPDFLoader usually 0-indexes pages; ensure 1-based page numbers
            page_val = doc.metadata.get("page", idx)
            doc.metadata["page"] = int(page_val) + 1
            doc.metadata["source"] = self.pdf_path.name
        print(f"Loaded {len(docs)} pages from {self.pdf_path.name}")
        return docs

    def run_fixed_size_splitter(self, docs: List[Document],
                                chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = splitter.split_documents(docs)
        for i, c in enumerate(chunks):
            c.metadata["chunk_type"] = "fixed_size"
            c.metadata["chunk_id"] = i
        return chunks

    def run_recursive_splitter(self, docs: List[Document],
                               chunk_size: int = 700, chunk_overlap: int = 100) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "Clause ", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = splitter.split_documents(docs)
        for i, c in enumerate(chunks):
            c.metadata["chunk_type"] = "recursive"
            c.metadata["chunk_id"] = i
        return chunks

    def run_semantic_splitter(self, docs: List[Document],
                              threshold_percentile: float = 75.0) -> List[Document]:
        splitter = SemanticTextSplitter(threshold_percentile=threshold_percentile)
        return splitter.split_documents(docs)

    def evaluate_strategies(self, docs: List[Document], output_dir: str = "results") -> Dict[str, Any]:
        """Runs and benchmarks all three text splitting strategies."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fixed_chunks = self.run_fixed_size_splitter(docs)
        recursive_chunks = self.run_recursive_splitter(docs)
        semantic_chunks = self.run_semantic_splitter(docs)

        def analyze_chunks(chunks: List[Document], name: str) -> Dict[str, Any]:
            lengths = [len(c.page_content) for c in chunks]
            
            # Check clause boundary preservation (how many chunks begin with "Clause" or standard title)
            clause_starts = sum(1 for c in chunks if re.match(r'^(Clause|CHAPTER|\b[A-Z][a-z]+)', c.page_content.strip()))
            mid_sentence_cuts = sum(1 for c in chunks if not c.page_content.strip().endswith(('.', ':', '?', '!')))
            
            return {
                "strategy": name,
                "total_chunks": len(chunks),
                "mean_char_length": float(np.mean(lengths)) if lengths else 0,
                "std_char_length": float(np.std(lengths)) if lengths else 0,
                "min_char_length": int(np.min(lengths)) if lengths else 0,
                "max_char_length": int(np.max(lengths)) if lengths else 0,
                "clause_start_alignment": round(clause_starts / len(chunks) * 100, 2) if chunks else 0,
                "mid_sentence_cut_percentage": round(mid_sentence_cuts / len(chunks) * 100, 2) if chunks else 0
            }

        stats = {
            "fixed_size": analyze_chunks(fixed_chunks, "Fixed-Size (Character)"),
            "recursive": analyze_chunks(recursive_chunks, "Recursive Character"),
            "semantic": analyze_chunks(semantic_chunks, "Semantic Similarity")
        }

        # Save stats JSON
        json_path = Path(output_dir) / "chunking_comparison.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved chunking comparison metrics to {json_path}")

        # Generate comparative visualization plot
        self._plot_comparison(fixed_chunks, recursive_chunks, semantic_chunks, output_dir)

        return {
            "stats": stats,
            "fixed_chunks": fixed_chunks,
            "recursive_chunks": recursive_chunks,
            "semantic_chunks": semantic_chunks
        }

    def _plot_comparison(self, fixed_chunks, recursive_chunks, semantic_chunks, output_dir: str):
        """Generates distribution histograms and boxplots for chunk sizes."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        lengths_fixed = [len(c.page_content) for c in fixed_chunks]
        lengths_rec = [len(c.page_content) for c in recursive_chunks]
        lengths_sem = [len(c.page_content) for c in semantic_chunks]

        # Subplot 1: Histograms
        axes[0].hist(lengths_fixed, bins=12, alpha=0.5, label='Fixed-Size', color='#EF4444', edgecolor='black')
        axes[0].hist(lengths_rec, bins=12, alpha=0.5, label='Recursive', color='#3B82F6', edgecolor='black')
        axes[0].hist(lengths_sem, bins=12, alpha=0.5, label='Semantic', color='#10B981', edgecolor='black')
        axes[0].set_title("Chunk Character Length Distribution", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Number of Characters")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Subplot 2: Boxplot
        axes[1].boxplot([lengths_fixed, lengths_rec, lengths_sem], tick_labels=['Fixed-Size', 'Recursive', 'Semantic'], patch_artist=True,
                        boxprops=dict(facecolor='#E2E8F0', color='#1E293B'))
        axes[1].set_title("Chunk Size Variance Comparison", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Characters")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = Path(output_dir) / "chunking_comparison.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved chunking comparison chart to {plot_path}")


if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline("data/raw_documents/university_academic_regulations_2024.pdf")
    docs = pipeline.load_pdf()
    results = pipeline.evaluate_strategies(docs)
    print("\n--- TEXT SPLITTING COMPARISON SUMMARY ---")
    for key, data in results["stats"].items():
        print(f"\nStrategy: {data['strategy']}")
        print(f"  Chunks: {data['total_chunks']}")
        print(f"  Mean Length: {data['mean_char_length']:.1f} ± {data['std_char_length']:.1f} chars")
        print(f"  Range: [{data['min_char_length']} - {data['max_char_length']}] chars")
        print(f"  Mid-sentence cuts: {data['mid_sentence_cut_percentage']}%")
