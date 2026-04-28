# Topic 39: RAG (Retrieval-Augmented Generation)

> 🔥 **For interviews, read these first:**
> - **`RAG_DEEP_DIVE.md`** — frontier-lab interview deep dive: indexing/retrieval/rerank/generate pipeline, chunking strategies, BM25 vs dense vs hybrid, HNSW/IVF/PQ vector indexing, embedding models (BGE/E5), HyDE, lost-in-the-middle, RAGAS evaluation, Self-RAG/GraphRAG.
> - **`INTERVIEW_GRILL.md`** — 50 active-recall questions.

## What You'll Learn

This topic covers RAG from an industry perspective:
- RAG architecture and components
- Real-world challenges and solutions
- Industry-standard implementations
- Production-ready code
- Evaluation and monitoring
- Common problems and fixes
- Advanced techniques

## Why We Need This

### Interview Importance
- **Common questions**: "Design a RAG system", "How do you improve RAG?", "RAG evaluation"
- **Industry standard**: RAG is widely used in production
- **Practical knowledge**: Real-world implementation details

### Real-World Application
- **Enterprise search**: Document Q&A systems
- **Customer support**: Knowledge base systems
- **Research tools**: Academic paper Q&A
- **Internal tools**: Company knowledge bases

## Overview

**RAG Components:**
- Document ingestion and chunking
- Embedding generation
- Vector database
- Retrieval strategies
- Re-ranking
- Generation with context

**Industry Challenges:**
- Chunking strategies
- Embedding quality
- Retrieval accuracy
- Context window limits
- Hallucination prevention
- Evaluation metrics

**Key Topics:**
- **Architecture**: Complete RAG pipeline and components
- **Chunking Strategies**: 10+ strategies with use cases and code
- **Retrieval Methods**: BM25, TF-IDF, Dense, Hybrid search
- **Challenges & Solutions**: Real-world problems and industry solutions
- **Evaluation**: Comprehensive metrics and frameworks
- **Implementation**: Production-ready code

**Retrieval Methods:**
- **BM25**: Industry-standard sparse retrieval
- **TF-IDF**: Simple keyword-based retrieval
- **Dense**: Semantic retrieval with embeddings
- **Hybrid**: Combining sparse + dense (best practice)

See detailed files for industry-standard implementations!

## Core Intuition

RAG exists because parametric knowledge inside a language model is not always enough.

You may need:
- fresher information
- domain-specific documents
- grounded answers with evidence

RAG adds a retrieval system so the model can answer using external context instead of relying only on memorized weights.

### The Core Pipeline

A simple way to explain RAG is:
1. break documents into retrievable units
2. retrieve the most relevant chunks for a query
3. pass those chunks into the generator
4. produce an answer grounded in retrieved context

### Why RAG Is Harder Than It Looks

RAG is not just "search plus LLM."

The system can fail at multiple stages:
- chunking
- embedding quality
- retrieval
- reranking
- context packing
- generation groundedness

That is why good interview answers about RAG are pipeline-aware.

## Technical Details Interviewers Often Want

### Chunking Trade-Off

Chunks that are too small:
- may lose necessary context

Chunks that are too large:
- may waste context window space
- may retrieve a lot of irrelevant text

### Retrieval Metric vs End-to-End Quality

Improving Recall@k does not automatically improve final answer quality.

Why?
- retrieved context may be noisy
- ordering may be poor
- generator may ignore relevant evidence
- context packing may truncate the best chunk

This is one of the most important RAG interview points.

### Hybrid Retrieval

Hybrid retrieval matters because sparse and dense methods fail differently:
- sparse search is strong on exact terms
- dense retrieval is strong on semantic similarity

Combining them often gives more robust behavior.

## Common Failure Modes

- poor chunking boundaries
- retrieving semantically related but answer-irrelevant text
- too much context causing distraction or truncation
- hallucinating despite retrieval because the generator ignores evidence
- evaluating only retrieval or only generation instead of the full pipeline

## Edge Cases and Follow-Up Questions

1. Why can better retrieval metrics still produce worse answers?
2. How do you tell whether a failure is retrieval-side or generation-side?
3. Why is chunk size such a high-leverage choice?
4. When is sparse retrieval better than dense retrieval?
5. Why is hybrid retrieval often a strong default?

## What to Practice Saying Out Loud

1. The RAG pipeline from ingestion to answer generation
2. Why RAG failures should be diagnosed stage by stage
3. Why grounding quality depends on more than retrieval alone
