"""
Deliverable 2: Conversation Memory Architectures (Buffer & Summary Memory).

Implements:
1. ConversationBufferMemory: Sliding window buffer retaining raw dialogue turns.
2. ConversationSummaryMemory: Progressive condensed abstraction of conversation history.
3. CombinedConversationMemory: Dual memory blending high-level summary with recent dialogue turns.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ConversationTurn:
    """Represents a single message exchange."""
    def __init__(self, role: str, content: str):
        self.role = role  # "user" or "assistant"
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationBufferMemory:
    """Sliding-window buffer memory storing recent dialogue turns verbatim."""
    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []

    def add_turn(self, role: str, content: str):
        self.turns.append(ConversationTurn(role, content))
        if len(self.turns) > self.max_turns * 2:  # 2 messages per turn (user + assistant)
            self.turns = self.turns[-self.max_turns * 2:]

    def get_history_string(self) -> str:
        """Formats the buffer as a readable conversation transcript."""
        if not self.turns:
            return ""
        lines = []
        for t in self.turns:
            label = "Student" if t.role == "user" else "Assistant"
            lines.append(f"{label}: {t.content}")
        return "\n".join(lines)

    def clear(self):
        self.turns = []


class ConversationSummaryMemory:
    """Progressive summary memory that condenses dialogue history."""
    def __init__(self, llm_fn: Optional[Any] = None):
        self.llm_fn = llm_fn
        self.summary: str = ""
        self.evicted_history: List[ConversationTurn] = []

    def update_summary(self, new_turns: List[ConversationTurn]):
        """Updates the running summary with new dialogue turns."""
        if not new_turns:
            return
        
        turn_text = "\n".join([f"{'Student' if t.role == 'user' else 'Assistant'}: {t.content}" for t in new_turns])
        
        if self.llm_fn:
            try:
                prompt = (
                    f"Current summary of discussion:\n{self.summary or 'None'}\n\n"
                    f"New conversation turns:\n{turn_text}\n\n"
                    f"Update the summary concisely, preserving student regulations, clauses, and key policies mentioned."
                )
                self.summary = self.llm_fn(prompt).strip()
                return
            except Exception:
                pass
        
        # Heuristic fallback summary
        topics = []
        for t in new_turns:
            if t.role == "user":
                topics.append(f"Student asked about: '{t.content[:80]}'")
            else:
                citations = [w for w in t.content.split() if "Page" in w or "Clause" in w]
                c_str = f" (cited {', '.join(set(citations[:3]))})" if citations else ""
                topics.append(f"Assistant provided regulations guidance{c_str}.")
        
        addition = " | ".join(topics)
        if self.summary:
            self.summary += f" | {addition}"
        else:
            self.summary = addition

    def clear(self):
        self.summary = ""
        self.evicted_history = []


class UnifiedConversationMemory:
    """
    Combines Buffer Memory and Summary Memory for context-aware multi-turn dialogue.
    Maintains the immediate buffer window while rolling older turns into a running summary.
    """
    def __init__(self, max_buffer_turns: int = 3, llm_fn: Optional[Any] = None):
        self.max_buffer_turns = max_buffer_turns
        self.buffer = ConversationBufferMemory(max_turns=max_buffer_turns)
        self.summarizer = ConversationSummaryMemory(llm_fn=llm_fn)
        self.all_turns: List[ConversationTurn] = []

    def add_interaction(self, user_msg: str, assistant_msg: str):
        u_turn = ConversationTurn("user", user_msg)
        a_turn = ConversationTurn("assistant", assistant_msg)
        self.all_turns.extend([u_turn, a_turn])
        self.buffer.add_turn("user", user_msg)
        self.buffer.add_turn("assistant", assistant_msg)

        # If total turns exceed buffer limit, update summary
        buffer_limit = self.max_buffer_turns * 2
        if len(self.all_turns) > buffer_limit:
            overflow = self.all_turns[:-buffer_limit]
            self.summarizer.update_summary(overflow)

    def get_memory_context(self) -> str:
        """Returns the formatted memory block for injection into RAG prompt."""
        parts = []
        if self.summarizer.summary:
            parts.append(f"--- PREVIOUS CONVERSATION SUMMARY ---\n{self.summarizer.summary}\n")
        
        buffer_str = self.buffer.get_history_string()
        if buffer_str:
            parts.append(f"--- RECENT CONVERSATION HISTORY ---\n{buffer_str}\n")
            
        return "\n".join(parts) if parts else "No previous conversation context."

    def clear(self):
        self.buffer.clear()
        self.summarizer.clear()
        self.all_turns = []


if __name__ == "__main__":
    memory = UnifiedConversationMemory(max_buffer_turns=2)
    
    # Simulate multi-turn dialogue
    memory.add_interaction(
        "What is the minimum attendance required for ESA?",
        "According to [Page 3, Clause 3.1], students must maintain at least 85% attendance."
    )
    memory.add_interaction(
        "Can it be condoned if I was sick?",
        "According to [Page 3, Clause 3.2], condonation is allowed between 75% and 84.9% on medical grounds with Rs. 2000 fee."
    )
    memory.add_interaction(
        "What if my attendance is 72%?",
        "According to [Page 3, Clause 3.3], attendance below 75% results in NSAR detention and cannot be condoned."
    )
    
    print("--- UNIFIED MEMORY INSPECTION ---")
    print(memory.get_memory_context())
