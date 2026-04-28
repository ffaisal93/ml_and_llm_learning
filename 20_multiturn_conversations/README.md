# Topic 20: Multi-Turn Conversations & Long Context

## What You'll Learn

This topic teaches you:
- How to design multi-turn conversations
- Managing conversation history
- Increasing context length to millions
- Techniques for long context (chunking, retrieval, etc.)

## Why We Need This

### Interview Importance
- **Common question**: "How to handle multi-turn conversations?"
- **Practical knowledge**: Essential for chatbots
- **Scalability**: Long context is challenging

### Real-World Application
- **Chatbots**: ChatGPT, Claude
- **Long documents**: Code review, analysis
- **Conversation memory**: Maintain context

## Industry Use Cases

### 1. **Multi-Turn Conversations**
**Use Case**: Chatbots, assistants
- Maintain conversation history
- Reference previous messages
- Context-aware responses

### 2. **Long Context (Millions of Tokens)**
**Use Case**: Document analysis, codebases
- Process entire codebases
- Analyze long documents
- Multi-document reasoning

## Core Intuition

Multi-turn systems are not just long prompts.

They are memory-management systems.

The hard part is not storing everything. It is deciding:
- what to keep verbatim
- what to summarize
- what to retrieve when needed

### Multi-Turn Conversations

The model needs to track:
- user goals
- entities and references
- unresolved questions
- prior commitments or constraints

### Long Context

Long context helps only when the extra tokens are actually relevant.

More raw history can also create:
- distraction
- higher memory cost
- worse focus

## Technical Details Interviewers Often Want

### History Management Is a Selection Problem

A good system may keep:
- recent turns verbatim
- older turns summarized
- key facts in structured memory

### Retrieval vs Raw Inclusion

For large histories or corpora, retrieval-based selection is often better than forcing the model to read everything every time.

### Summarization Trade-Off

Summaries save tokens but can:
- omit detail
- distort commitments
- lose exact wording needed later

## Common Failure Modes

- appending everything forever without prioritization
- summarizing too aggressively and losing important commitments
- retrieving poorly and missing the truly relevant context
- assuming longer history always improves answer quality

## Edge Cases and Follow-Up Questions

1. Why is raw long context not enough for good memory?
2. Why can summarization hurt even when it saves tokens?
3. How do you decide what to keep verbatim vs summarize?
4. Why can retrieval be better than full-history inclusion?
5. Why is multi-turn conversation partly a memory-management problem?

## What to Practice Saying Out Loud

1. Why multi-turn systems need selection, not just storage
2. Why long context and conversation memory are related but not identical
3. Why good context management is about relevance and fidelity

## Industry-Standard Boilerplate Code

### Multi-Turn Conversation Design

```python
"""
Multi-Turn Conversation System
"""
class ConversationManager:
    """
    Manages conversation history and context
    """
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history = []
    
    def add_message(self, role: str, content: str):
        """Add message to history"""
        self.conversation_history.append({
            'role': role,  # 'user' or 'assistant'
            'content': content
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def get_context(self) -> str:
        """Format conversation for model"""
        context = ""
        for msg in self.conversation_history:
            context += f"{msg['role']}: {msg['content']}\n"
        return context
    
    def clear(self):
        """Clear conversation history"""
        self.conversation_history = []
```

### Long Context Solutions

```python
"""
Long Context: Handle millions of tokens
"""
class LongContextHandler:
    """
    Strategies for long context:
    1. Chunking + Summarization
    2. Retrieval-based (RAG)
    3. Hierarchical attention
    4. Sliding window
    """
    
    def chunk_and_summarize(self, text: str, chunk_size: int = 1000) -> str:
        """
        Chunk long text and summarize each chunk
        Then combine summaries
        """
        chunks = [text[i:i+chunk_size] 
                 for i in range(0, len(text), chunk_size)]
        
        summaries = []
        for chunk in chunks:
            # Summarize chunk (simplified)
            summary = self.summarize(chunk)
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def retrieval_based(self, query: str, documents: list, 
                      top_k: int = 5) -> str:
        """
        RAG: Retrieve relevant chunks, then generate
        
        For millions of tokens:
        1. Split into chunks
        2. Embed chunks
        3. Retrieve top-k relevant chunks
        4. Use only retrieved chunks for generation
        """
        # Embed query
        query_embedding = self.embed(query)
        
        # Find most relevant chunks
        similarities = []
        for doc in documents:
            doc_embedding = self.embed(doc)
            sim = np.dot(query_embedding, doc_embedding)
            similarities.append((sim, doc))
        
        # Get top-k
        similarities.sort(reverse=True)
        relevant_chunks = [doc for _, doc in similarities[:top_k]]
        
        # Generate with only relevant chunks
        context = "\n".join(relevant_chunks)
        return self.generate(query, context)
    
    def hierarchical_attention(self, text: str) -> str:
        """
        Hierarchical: Attend at multiple levels
        1. Coarse level: Summaries
        2. Fine level: Relevant details
        """
        # Level 1: Summarize sections
        sections = self.split_into_sections(text)
        section_summaries = [self.summarize(s) for s in sections]
        
        # Level 2: For relevant sections, use details
        relevant_sections = self.select_relevant(section_summaries)
        detailed_context = "\n".join([sections[i] 
                                     for i in relevant_sections])
        
        return detailed_context
```

### Increasing Context Length

```python
"""
Techniques to increase context length:
"""
def increase_context_length():
    """
    Strategies:
    
    1. Chunking + Retrieval (RAG)
       - Store chunks in vector DB
       - Retrieve only relevant chunks
       - Can handle millions of tokens
    
    2. Summarization
       - Summarize old context
       - Keep summaries + recent context
       - Compress history
    
    3. Hierarchical Processing
       - Process at multiple levels
       - Coarse then fine
    
    4. Sparse Attention
       - Only attend to important parts
       - Longformer, BigBird
    
    5. External Memory
       - Store context externally
       - Retrieve when needed
    """
    pass
```

## Theory

### Multi-Turn Design

**Key Components:**
1. **History Management**: Store conversation
2. **Context Window**: Limit history length
3. **Summarization**: Compress old messages
4. **Role Tracking**: User vs assistant

### Long Context Strategies

**Problem**: Models can't handle millions of tokens directly

**Solutions:**
1. **RAG**: Retrieve only relevant parts
2. **Summarization**: Compress context
3. **Hierarchical**: Multi-level processing
4. **Sparse Attention**: Efficient attention

## Exercises

1. Implement conversation manager
2. Design RAG system
3. Implement chunking
4. Test with long documents

## Next Steps

- **Topic 21**: Dimensionality reduction
- **Topic 22**: Recommendation systems
