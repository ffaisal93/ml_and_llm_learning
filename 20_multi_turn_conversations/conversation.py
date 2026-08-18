"""
Multi-Turn Conversations & Long Context
Interview question: "How to design multi-turn conversations?"
"""
import numpy as np

class ConversationManager:
    """
    Manages multi-turn conversation history
    """
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history = []
    
    def add_message(self, role: str, content: str):
        """
        Add message to conversation
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.conversation_history.append({
            'role': role,
            'content': content
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def get_context(self) -> str:
        """Format conversation for model input"""
        context = ""
        for msg in self.conversation_history:
            context += f"{msg['role']}: {msg['content']}\n"
        return context
    
    def clear(self):
        """Clear conversation history"""
        self.conversation_history = []


class LongContextHandler:
    """
    Handle long context (millions of tokens)
    Strategies: Chunking, RAG, Summarization
    """
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> list:
        """Split text into chunks"""
        return [text[i:i+chunk_size] 
                for i in range(0, len(text), chunk_size)]
    
    def retrieval_based(self, query: str, documents: list, 
                       top_k: int = 5) -> list:
        """
        RAG: Retrieve relevant chunks
        
        For millions of tokens:
        1. Split into chunks
        2. Embed chunks
        3. Retrieve top-k relevant
        4. Use only retrieved for generation
        """
        # Simplified: Just return top-k by length similarity
        # In practice, use embeddings and cosine similarity
        doc_scores = []
        for doc in documents:
            # Simple scoring (in practice, use embeddings)
            score = len(set(query.split()) & set(doc.split()))
            doc_scores.append((score, doc))
        
        doc_scores.sort(reverse=True)
        return [doc for _, doc in doc_scores[:top_k]]
    
    def summarize_context(self, text: str, max_length: int = 500) -> str:
        """
        Summarize long context
        
        Keep only important parts
        """
        # Simplified: Just truncate
        # In practice, use summarization model
        if len(text) <= max_length:
            return text
        
        # Keep beginning and end
        half = max_length // 2
        return text[:half] + "...[truncated]..." + text[-half:]


def how_to_increase_context_length():
    """
    Techniques to handle millions of tokens:
    
    1. RAG (Retrieval-Augmented Generation)
       - Store chunks in vector DB
       - Retrieve only relevant chunks
       - Can handle unlimited tokens
    
    2. Summarization
       - Summarize old context
       - Keep summaries + recent
       - Compress history
    
    3. Hierarchical Processing
       - Process at multiple levels
       - Coarse summaries, then details
    
    4. Sparse Attention
       - Only attend to important parts
       - Longformer, BigBird attention
    
    5. External Memory
       - Store context externally
       - Retrieve when needed
    """
    strategies = [
        "RAG: Retrieve only relevant chunks",
        "Summarization: Compress old context",
        "Hierarchical: Multi-level processing",
        "Sparse Attention: Efficient attention",
        "External Memory: Store externally"
    ]
    
    print("Strategies for Long Context (Millions of Tokens):")
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy}")


# Usage Example
if __name__ == "__main__":
    print("Multi-Turn Conversations & Long Context")
    print("=" * 60)
    
    # Multi-turn conversation
    conv = ConversationManager(max_history=5)
    conv.add_message("user", "Hello, what's the weather?")
    conv.add_message("assistant", "I don't have access to weather data.")
    conv.add_message("user", "Can you help me with Python?")
    
    print("Conversation Context:")
    print(conv.get_context())
    print()
    
    # Long context handling
    long_text = "This is a very long text. " * 1000
    handler = LongContextHandler()
    
    chunks = handler.chunk_text(long_text, chunk_size=100)
    print(f"Long text split into {len(chunks)} chunks")
    
    how_to_increase_context_length()

