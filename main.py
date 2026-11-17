"""
Main CLI Pipeline for Retrieval-Augmented Academic Regulations Assistant.

Runs all 3 project deliverables:
1. Ingestion pipeline with splitting comparisons (Fixed, Recursive, Semantic).
2. FAISS & ChromaDB index construction, Query Planning, and Buffer/Summary Memory.
3. 50-Question Benchmark evaluation (Recall@k, Correctness, Faithfulness, Refusal Accuracy).
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_generator import create_handbook_pdf, generate_text_representation
from src.ingestion import DocumentIngestionPipeline
from src.vector_store import AcademicVectorIndexManager
from src.query_planner import QueryPlanner
from src.memory import UnifiedConversationMemory
from src.rag_chain import AcademicRegulationsRAGChain
from src.evaluation import BenchmarkEvaluator


def run_deliverable_1(pdf_path: str = "data/raw_documents/university_academic_regulations_2024.pdf"):
    print("\n" + "="*70)
    print(" [DELIVERABLE 1] INGESTION PIPELINE & TEXT-SPLITTING STRATEGY COMPARISON")
    print("="*70)
    
    if not os.path.exists(pdf_path):
        print(f"Generating official regulations PDF at {pdf_path}...")
        create_handbook_pdf(pdf_path)
        generate_text_representation(pdf_path, "data/raw_documents/academic_regulations_text.txt")
        
    pipeline = DocumentIngestionPipeline(pdf_path)
    docs = pipeline.load_pdf()
    results = pipeline.evaluate_strategies(docs, output_dir="results")
    
    print("\n--- DELIVERABLE 1 COMPARISON TABLE ---")
    header = f"{'Strategy':<26} | {'Chunks':<8} | {'Mean ± Std (chars)':<20} | {'Range':<14} | {'Mid-Cut %':<10}"
    print(header)
    print("-" * len(header))
    for strat_key, data in results["stats"].items():
        name = data["strategy"]
        chunks = data["total_chunks"]
        mean_std = f"{data['mean_char_length']:.1f} ± {data['std_char_length']:.1f}"
        rng = f"[{data['min_char_length']} - {data['max_char_length']}]"
        cuts = f"{data['mid_sentence_cut_percentage']}%"
        print(f"{name:<26} | {chunks:<8} | {mean_std:<20} | {rng:<14} | {cuts:<10}")
        
    print(f"\n[Artifacts Created]:")
    print(f"  • JSON Metrics: results/chunking_comparison.json")
    print(f"  • Plot Chart:   results/chunking_comparison.png")
    return results


def run_deliverable_2(docs=None, backend: str = "faiss"):
    print("\n" + "="*70)
    print(" [DELIVERABLE 2] VECTOR INDEXES, QUERY PLANNING & CONVERSATION MEMORY")
    print("="*70)
    
    pdf_path = "data/raw_documents/university_academic_regulations_2024.pdf"
    if docs is None:
        pipeline = DocumentIngestionPipeline(pdf_path)
        raw_docs = pipeline.load_pdf()
        docs = pipeline.run_recursive_splitter(raw_docs)
        
    manager = AcademicVectorIndexManager()
    print("Building FAISS and ChromaDB vector indexes...")
    manager.build_indexes(docs)
    
    print("\n1. Verifying Multi-Part Query Planning:")
    planner = QueryPlanner()
    sample_complex_q = "What is the minimum attendance required for ESA, and what happens if I am detained under NSAR?"
    plan = planner.plan_query(sample_complex_q)
    print(f"  Original Query: {plan.original_query}")
    print(f"  Is Multi-Part:  {plan.is_multipart}")
    print(f"  Sub-queries:    {plan.sub_queries}")
    
    print("\n2. Verifying Conversation Memory (Buffer + Summary):")
    memory = UnifiedConversationMemory(max_buffer_turns=2)
    memory.add_interaction("What are credits for B.Tech?", "According to [Page 1], minimum 160 credits are required.")
    memory.add_interaction("Can I withdraw from a course?", "According to [Page 2], withdrawal is allowed up to 8th week (Grade 'W').")
    memory.add_interaction("What is the attendance condonation fee?", "According to [Page 3], fee is Rs. 2,000 for 75-84.9% attendance.")
    print("  Memory State Context:")
    for line in memory.get_memory_context().splitlines()[:5]:
        print(f"    {line}")
    print("    ...")

    print("\n3. Testing End-to-End RAG Chain:")
    rag = AcademicRegulationsRAGChain(backend=backend)
    res = rag.answer_query(sample_complex_q)
    print(f"  Answer Preview:\n  {res['answer'][:200]}...")
    print(f"  Cited Pages: {res['cited_pages']}")
    
    # Test refusal
    res_oos = rag.answer_query("What is the hostel mess menu for dinner?")
    print(f"\n4. Testing Refusal Gate on Out-of-Scope Query:")
    print(f"  Query: 'What is the hostel mess menu for dinner?'")
    print(f"  Response: '{res_oos['answer']}'")
    print(f"  Refused Flag: {res_oos['refused']}")


def run_deliverable_3():
    print("\n" + "="*70)
    print(" [DELIVERABLE 3] 50-QUESTION BENCHMARK EVALUATION & REPORT GENERATION")
    print("="*70)
    evaluator = BenchmarkEvaluator()
    consolidated = evaluator.run_full_benchmark()
    return consolidated


def interactive_chat(backend: str = "faiss"):
    print("\n" + "="*70)
    print(f"   ACADEMIC REGULATIONS CONVERSATIONAL ASSISTANT ({backend.upper()})")
    print("   Type 'exit' to quit, 'backend faiss/chroma' to switch index.")
    print("="*70)
    rag = AcademicRegulationsRAGChain(backend=backend)
    
    while True:
        try:
            user_input = input("\nStudent: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if user_input.lower().startswith("backend "):
                new_b = user_input.split()[1].strip().lower()
                rag.set_backend(new_b)
                print(f"[Switched vector backend to {new_b.upper()}]")
                continue
                
            res = rag.answer_query(user_input)
            print(f"\nAssistant:\n{res['answer']}")
            if res['cited_pages']:
                print(f"\n[Sources Cited: {', '.join([f'Page {p}' for p in res['cited_pages']])}]")
            if res['plan']['is_multipart']:
                print(f"[Query Plan: Sub-queries: {res['plan']['sub_queries']}]")
        except (KeyboardInterrupt, EOFError):
            break


def main():
    parser = argparse.ArgumentParser(description="Academic Regulations Assistant - Deliverables Runner")
    parser.add_argument("--deliverable", type=int, choices=[1, 2, 3], help="Run specific deliverable (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Run all 3 deliverables end-to-end")
    parser.add_argument("--chat", action="store_true", help="Launch interactive CLI chat")
    parser.add_argument("--backend", type=str, default="faiss", choices=["faiss", "chroma"], help="Vector backend")
    parser.add_argument("--query", type=str, help="Run a single ad-hoc query")
    
    args = parser.parse_args()

    if args.query:
        rag = AcademicRegulationsRAGChain(backend=args.backend)
        res = rag.answer_query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Answer:\n{res['answer']}")
        print(f"Cited Pages: {res['cited_pages']}")
        print(f"Refused: {res['refused']}")
        return

    if args.chat:
        interactive_chat(backend=args.backend)
        return

    if args.deliverable == 1:
        run_deliverable_1()
    elif args.deliverable == 2:
        run_deliverable_2(backend=args.backend)
    elif args.deliverable == 3:
        run_deliverable_3()
    elif args.all or (len(sys.argv) == 1):
        print("\nExecuting complete end-to-end project suite...")
        split_res = run_deliverable_1()
        run_deliverable_2(docs=split_res["recursive_chunks"], backend=args.backend)
        run_deliverable_3()
        print("\n" + "="*70)
        print(" ALL 3 DELIVERABLES COMPLETED SUCCESSFULLY!")
        print(" Reports and charts generated in results/ directory.")
        print("="*70)


if __name__ == "__main__":
    main()
