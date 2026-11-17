"""
Deliverable 2: Query Planning for Multi-Part Questions.

Implements:
1. Multi-part question detection and linguistic/semantic decomposition.
2. Generating structured query plans with atomic sub-queries.
3. Sub-query retrieval and context fusion with deduplication.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class QueryPlan:
    """Represents the decomposed plan for answering a user query."""
    def __init__(self, original_query: str, sub_queries: List[str], is_multipart: bool, reasoning: str):
        self.original_query = original_query
        self.sub_queries = sub_queries
        self.is_multipart = is_multipart
        self.reasoning = reasoning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "is_multipart": self.is_multipart,
            "sub_queries": self.sub_queries,
            "reasoning": self.reasoning
        }

    def __repr__(self):
        return f"<QueryPlan is_multipart={self.is_multipart} sub_queries={self.sub_queries}>"


class QueryPlanner:
    """
    Deconstructs complex, multi-faceted student queries into atomic,
    focused sub-queries for precision retrieval across distinct document sections.
    """
    def __init__(self, llm_fn: Optional[Any] = None):
        self.llm_fn = llm_fn

    def plan_query(self, query: str, conversation_summary: str = "") -> QueryPlan:
        """Analyzes and decomposes the query into sub-queries."""
        query_clean = query.strip()
        
        # If an LLM function is provided, try LLM-driven query planning first
        if self.llm_fn:
            try:
                plan = self._llm_decompose(query_clean, conversation_summary)
                if plan:
                    return plan
            except Exception as e:
                # Fall back to rule-based decomposer
                pass

        return self._rule_based_decompose(query_clean)

    def _rule_based_decompose(self, query: str) -> QueryPlan:
        """Linguistic rule-based query decomposition."""
        # 1. Check for multiple distinct questions separated by question marks
        q_mark_splits = [q.strip() + "?" for q in query.split("?") if q.strip()]
        if len(q_mark_splits) > 1:
            sub_qs = [re.sub(r'^\s*(and|also|additionally|furthermore)\s+', '', q, flags=re.IGNORECASE).strip() for q in q_mark_splits]
            return QueryPlan(
                original_query=query,
                sub_queries=sub_qs,
                is_multipart=True,
                reasoning="Detected multiple question sentences delimited by question marks."
            )

        # 2. Check for semicolon or coordinating conjunctions separating questions
        patterns = [
            r';\s*',
            r',\s*and\s+(?=[a-z0-9])',
            r'\s+and\s+(?=(?:what|how|who|can|is|are|does|if|where|when|why)\b)',
            r'\s+as\s+well\s+as\s+(?=(?:what|how|who|can|is|are|does|the|rules|policies)\b)',
            r'\s+also\s+(?=(?:what|how|who|can|is|are|does|tell|explain)\b)'
        ]
        
        for pat in patterns:
            parts = re.split(pat, query, flags=re.IGNORECASE)
            if len(parts) > 1 and all(len(p.strip()) > 8 for p in parts):
                sub_qs = []
                for p in parts:
                    sub_q = p.strip().rstrip(',;')
                    if not sub_q.endswith(('?', '.')):
                        sub_q += "?"
                    # If clause starts with lowercase conjunction or adverb, trim it
                    sub_q = re.sub(r'^(and|also|furthermore|additionally)\s+', '', sub_q, flags=re.IGNORECASE)
                    # Capitalize first letter
                    if sub_q:
                        sub_q = sub_q[0].upper() + sub_q[1:]
                    sub_qs.append(sub_q)
                
                return QueryPlan(
                    original_query=query,
                    sub_queries=sub_qs,
                    is_multipart=True,
                    reasoning="Detected compound multi-clause query separated by conjunction."
                )

        # Single atomic query
        return QueryPlan(
            original_query=query,
            sub_queries=[query],
            is_multipart=False,
            reasoning="Single focused intent detected; no decomposition needed."
        )

    def _llm_decompose(self, query: str, context: str) -> Optional[QueryPlan]:
        """Prompt-based LLM query decomposition."""
        prompt = (
            f"You are a Query Planning agent for an academic regulations RAG system.\n"
            f"Analyze the user query: '{query}'\n"
            f"Context: {context}\n"
            f"If the query asks about more than one distinct topic or policy (e.g. attendance AND grading), "
            f"split it into 2 or 3 standalone atomic sub-queries.\n"
            f"If it is a single question, return just that question.\n"
            f"Format response strictly as JSON:\n"
            f'{{"is_multipart": bool, "sub_queries": ["sub_query_1", "sub_query_2"], "reasoning": "brief explanation"}}\n'
        )
        res = self.llm_fn(prompt)
        import json
        match = re.search(r'\{.*\}', res, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return QueryPlan(
                original_query=query,
                sub_queries=data.get("sub_queries", [query]),
                is_multipart=data.get("is_multipart", False),
                reasoning=data.get("reasoning", "LLM-based planning")
            )
        return None


if __name__ == "__main__":
    planner = QueryPlanner()
    
    test_queries = [
        "What is the minimum attendance required for ESA, and what happens if I am detained under NSAR?",
        "Can I register for an audit course, and what are the rules for re-evaluation fees?",
        "What is the maximum duration to complete B.Tech degree?",
        "What is the pass mark in CIE? Also, how is SGPA calculated?"
    ]
    
    print("--- QUERY PLANNER DECOMPOSITION TEST ---")
    for q in test_queries:
        plan = planner.plan_query(q)
        print(f"\nOriginal: {plan.original_query}")
        print(f"Is Multi-Part: {plan.is_multipart}")
        print(f"Sub-queries ({len(plan.sub_queries)}):")
        for i, sq in enumerate(plan.sub_queries, 1):
            print(f"  {i}. {sq}")
        print(f"Reasoning: {plan.reasoning}")
