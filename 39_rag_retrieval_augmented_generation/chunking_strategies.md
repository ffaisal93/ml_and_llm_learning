# Chunking Strategies: Complete Guide with Use Cases

## Overview

Chunking is one of the most critical components of RAG systems. The way you split documents directly impacts retrieval accuracy and generation quality. This document covers all chunking strategies used in production systems.

## Why Chunking Matters

**Problems with Poor Chunking:**
- Breaking sentences/paragraphs → Loss of context
- Too small chunks → Missing information
- Too large chunks → Exceeding context window, noise
- Wrong boundaries → Semantic mismatch

**Impact:**
- **Retrieval accuracy**: Bad chunks → Bad retrieval → Bad answers
- **Generation quality**: Incomplete context → Incomplete answers
- **Cost**: Too many small chunks → More API calls
- **Latency**: Too many chunks → Slower retrieval

## Chunking Strategies

### 1. Fixed-Size Chunking

**How it works:**
- Split text into fixed-size chunks (e.g., 512 characters)
- Overlap between chunks (e.g., 50 characters)
- Simple and fast

**Implementation:**
```python
def fixed_size_chunk(text, chunk_size=512, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

**Use Cases:**
- ✅ **Uniform documents**: Similar structure throughout
- ✅ **Code documentation**: Consistent formatting
- ✅ **Simple Q&A**: Basic retrieval needs
- ✅ **Fast prototyping**: Quick to implement

**Pros:**
- Simple to implement
- Fast processing
- Predictable chunk sizes
- Easy to manage

**Cons:**
- ❌ Breaks sentences/paragraphs
- ❌ No semantic awareness
- ❌ May split important information
- ❌ Not optimal for most use cases

**When to Use:**
- Simple documents
- Fast prototyping
- Uniform content
- When speed > accuracy

**Parameters:**
- **Chunk size**: 256-1024 characters (or tokens)
- **Overlap**: 10-20% of chunk size
- **Common**: 512 chars with 50 char overlap

---

### 2. Sentence-Based Chunking

**How it works:**
- Split on sentence boundaries
- Group sentences into chunks
- Respects semantic units

**Implementation:**
```python
def sentence_based_chunk(text, max_chunk_size=512, min_chunk_size=100):
    sentences = split_sentences(text)  # Use NLTK, spaCy
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        if current_size + len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = len(sentence)
        else:
            current_chunk.append(sentence)
            current_size += len(sentence) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
```

**Use Cases:**
- ✅ **Narrative text**: Stories, articles, books
- ✅ **General documents**: Most text documents
- ✅ **Better than fixed-size**: When you need semantic boundaries
- ✅ **Production systems**: Common default choice

**Pros:**
- Respects sentence boundaries
- Better semantic coherence
- More natural chunks
- Good default choice

**Cons:**
- ❌ Sentence splitting can be imperfect
- ❌ May create very small/large chunks
- ❌ Doesn't understand paragraph structure

**When to Use:**
- Most text documents
- Narrative content
- When you want better than fixed-size
- Production systems (common default)

**Parameters:**
- **Max chunk size**: 400-800 tokens
- **Min chunk size**: 50-200 tokens
- **Common**: 512 tokens max, 100 tokens min

---

### 3. Paragraph-Based Chunking

**How it works:**
- Split on paragraph boundaries
- Each paragraph (or group) becomes a chunk
- Respects document structure

**Implementation:**
```python
def paragraph_based_chunk(text, max_paragraphs_per_chunk=3):
    paragraphs = text.split('\n\n')  # Simple split
    chunks = []
    current_chunk = []
    
    for para in paragraphs:
        if len(current_chunk) >= max_paragraphs_per_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
        else:
            current_chunk.append(para)
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    return chunks
```

**Use Cases:**
- ✅ **Structured documents**: Articles, reports, papers
- ✅ **Paragraph-level Q&A**: Questions about paragraphs
- ✅ **Academic papers**: Well-structured content
- ✅ **Long-form content**: Books, manuals

**Pros:**
- Respects document structure
- Natural semantic units
- Good for structured content
- Preserves context

**Cons:**
- ❌ Paragraphs vary greatly in size
- ❌ May exceed context window
- ❌ May be too small for some queries

**When to Use:**
- Structured documents
- Academic papers
- Long-form content
- When paragraphs are meaningful units

**Parameters:**
- **Max paragraphs**: 2-5 paragraphs per chunk
- **Min size**: Ensure minimum chunk size
- **Common**: 3 paragraphs per chunk

---

### 4. Semantic Chunking

**How it works:**
- Use embeddings to find semantic boundaries
- Group similar sentences together
- Split when semantic shift detected

**Implementation:**
```python
def semantic_chunk(text, embedding_model, similarity_threshold=0.7):
    sentences = split_sentences(text)
    embeddings = [embedding_model.encode(s) for s in sentences]
    
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = embeddings[0]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(current_embedding, embeddings[i])
        
        if similarity < similarity_threshold:
            # Semantic shift - start new chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
            current_embedding = embeddings[i]
        else:
            # Similar - add to current chunk
            current_chunk.append(sentences[i])
            # Update embedding (average or weighted)
            current_embedding = update_embedding(current_embedding, embeddings[i])
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
```

**Use Cases:**
- ✅ **Topic-based documents**: Multiple topics per document
- ✅ **High accuracy needs**: When retrieval accuracy critical
- ✅ **Complex documents**: Mixed content types
- ✅ **Production systems**: When quality > speed

**Pros:**
- Best semantic coherence
- Respects topic boundaries
- Optimal for retrieval
- High quality chunks

**Cons:**
- ❌ Slower (needs embeddings)
- ❌ More complex
- ❌ Requires embedding model
- ❌ Higher cost

**When to Use:**
- High accuracy requirements
- Complex documents
- Topic-based content
- When quality is priority

**Parameters:**
- **Similarity threshold**: 0.6-0.8 (lower = more chunks)
- **Min chunk size**: Ensure minimum
- **Max chunk size**: Prevent too large
- **Common**: 0.7 threshold, 100-512 token chunks

---

### 5. Recursive Chunking

**How it works:**
- Hierarchical splitting: Try large chunks first, recursively split if too large
- Respects document structure (sentences → paragraphs → sections)
- Multi-level approach

**Implementation:**
```python
def recursive_chunk(text, chunk_size=512, separators=['\n\n', '\n', '. ', ' ']):
    # Try largest separator first
    for separator in separators:
        if separator in text:
            splits = text.split(separator)
            if all(len(s) <= chunk_size for s in splits):
                # All splits fit, return them
                return [s for s in splits if s.strip()]
            else:
                # Some too large, recurse
                chunks = []
                for split in splits:
                    if len(split) <= chunk_size:
                        chunks.append(split)
                    else:
                        # Recurse with next separator
                        chunks.extend(recursive_chunk(split, chunk_size, separators[1:]))
                return chunks
    
    # No separator found, use fixed-size
    return fixed_size_chunk(text, chunk_size)
```

**Use Cases:**
- ✅ **Variable structure**: Documents with different structures
- ✅ **Robust systems**: Need to handle any document
- ✅ **General-purpose**: Works for most documents
- ✅ **Production systems**: LangChain uses this

**Pros:**
- Handles variable structures
- Respects hierarchy
- Robust
- Good default

**Cons:**
- ❌ More complex
- ❌ Can be slow
- ❌ May create inconsistent sizes

**When to Use:**
- General-purpose systems
- Variable document types
- When you need robustness
- Production systems (LangChain default)

**Parameters:**
- **Separators**: ['\n\n', '\n', '. ', ' '] (hierarchical)
- **Chunk size**: 400-1000 tokens
- **Common**: 512 tokens, standard separators

---

### 6. Sliding Window Chunking

**How it works:**
- Fixed-size chunks with overlap
- Sliding window moves through text
- Ensures context preservation

**Implementation:**
```python
def sliding_window_chunk(text, chunk_size=512, stride=256):
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += stride  # Move by stride (not full overlap)
    
    return chunks
```

**Use Cases:**
- ✅ **Context preservation**: Need overlapping context
- ✅ **Sequential information**: When order matters
- ✅ **Code**: When context spans chunks
- ✅ **Long documents**: Preserve relationships

**Pros:**
- Preserves context
- Good for sequential data
- Simple to implement
- Predictable

**Cons:**
- ❌ Creates many chunks
- ❌ Redundancy
- ❌ Higher storage cost
- ❌ May retrieve duplicates

**When to Use:**
- Sequential information
- Code documentation
- When context matters
- Long documents

**Parameters:**
- **Chunk size**: 256-1024 tokens
- **Stride**: 50-75% of chunk size
- **Common**: 512 tokens, 256 stride (50% overlap)

---

### 7. Token-Based Chunking

**How it works:**
- Split by token count (not character count)
- More accurate for LLMs
- Respects token boundaries

**Implementation:**
```python
def token_based_chunk(text, tokenizer, max_tokens=512, overlap_tokens=50):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap_tokens
    
    return chunks
```

**Use Cases:**
- ✅ **LLM integration**: When using token-based models
- ✅ **Accurate sizing**: Precise context window control
- ✅ **Production systems**: When token limits matter
- ✅ **Cost optimization**: Accurate token counting

**Pros:**
- Accurate for LLMs
- Respects token boundaries
- Precise size control
- Better cost estimation

**Cons:**
- ❌ Requires tokenizer
- ❌ Slower than character-based
- ❌ Model-specific

**When to Use:**
- LLM-based systems
- When token limits critical
- Cost optimization
- Production systems

**Parameters:**
- **Max tokens**: 256-1024 (model-dependent)
- **Overlap tokens**: 10-20% of max
- **Common**: 512 tokens, 50 token overlap

---

### 8. Hierarchical Chunking

**How it works:**
- Multi-level chunking (document → section → paragraph → sentence)
- Store chunks at multiple levels
- Retrieve at appropriate level

**Implementation:**
```python
def hierarchical_chunk(document):
    hierarchy = {
        'document': document,
        'sections': [],
        'paragraphs': [],
        'sentences': []
    }
    
    # Extract sections (by headings)
    sections = extract_sections(document)
    hierarchy['sections'] = sections
    
    for section in sections:
        paragraphs = extract_paragraphs(section)
        hierarchy['paragraphs'].extend(paragraphs)
        
        for para in paragraphs:
            sentences = split_sentences(para)
            hierarchy['sentences'].extend(sentences)
    
    return hierarchy
```

**Use Cases:**
- ✅ **Complex documents**: Multiple levels of structure
- ✅ **Multi-level retrieval**: Retrieve at different granularities
- ✅ **Academic papers**: Sections, subsections, paragraphs
- ✅ **Long documents**: Books, manuals

**Pros:**
- Preserves structure
- Multi-level retrieval
- Better context
- Flexible

**Cons:**
- ❌ Complex to implement
- ❌ More storage
- ❌ More complex retrieval

**When to Use:**
- Complex structured documents
- Academic papers
- When structure matters
- Multi-level Q&A

**Parameters:**
- **Levels**: Document → Section → Paragraph → Sentence
- **Store all levels**: For flexible retrieval
- **Common**: 3-4 levels

---

### 9. Content-Aware Chunking

**How it works:**
- Different strategies for different content types
- Code: Function/class boundaries
- Tables: Row/column boundaries
- Lists: Item boundaries

**Implementation:**
```python
def content_aware_chunk(document):
    chunks = []
    
    # Split by content type
    code_blocks = extract_code(document)
    tables = extract_tables(document)
    text_blocks = extract_text(document)
    
    # Chunk each type appropriately
    for code in code_blocks:
        chunks.extend(chunk_code(code))  # By function/class
    
    for table in tables:
        chunks.append(chunk_table(table))  # Keep table together
    
    for text in text_blocks:
        chunks.extend(sentence_based_chunk(text))  # Sentence-based
    
    return chunks
```

**Use Cases:**
- ✅ **Mixed content**: Documents with code, tables, text
- ✅ **Technical documentation**: Code + explanations
- ✅ **Research papers**: Text + tables + figures
- ✅ **Complex documents**: Multiple content types

**Pros:**
- Optimal for each content type
- Preserves structure
- Better retrieval
- Handles complexity

**Cons:**
- ❌ Very complex
- ❌ Requires content detection
- ❌ More processing

**When to Use:**
- Mixed content documents
- Technical documentation
- Research papers
- When content types vary

**Parameters:**
- **Content types**: Text, code, tables, lists, etc.
- **Strategy per type**: Optimize for each
- **Common**: Detect type, apply appropriate strategy

---

### 10. Metadata-Enriched Chunking

**How it works:**
- Chunk with rich metadata
- Include section titles, page numbers, etc.
- Better context for retrieval

**Implementation:**
```python
def metadata_enriched_chunk(document):
    chunks = []
    current_section = None
    current_page = 1
    
    for paragraph in document.paragraphs:
        # Extract metadata
        if is_heading(paragraph):
            current_section = paragraph.text
        
        metadata = {
            'section': current_section,
            'page': current_page,
            'paragraph_index': paragraph.index,
            'document_id': document.id
        }
        
        chunk = Chunk(
            content=paragraph.text,
            metadata=metadata
        )
        chunks.append(chunk)
    
    return chunks
```

**Use Cases:**
- ✅ **Structured documents**: With sections, pages
- ✅ **Citation needs**: Need to cite sources
- ✅ **Filtering**: Filter by section, page, etc.
- ✅ **Production systems**: When metadata matters

**Pros:**
- Rich context
- Better filtering
- Citation support
- Better retrieval

**Cons:**
- ❌ More storage
- ❌ More complex
- ❌ Requires metadata extraction

**When to Use:**
- Structured documents
- When citations needed
- When filtering needed
- Production systems

**Parameters:**
- **Metadata fields**: Section, page, author, date, etc.
- **Extract automatically**: From document structure
- **Common**: Section, page, document ID

---

## Comparison Table

| Strategy | Complexity | Speed | Quality | Use Case |
|----------|-----------|-------|--------|----------|
| **Fixed-Size** | Low | Fast | Low | Simple, prototyping |
| **Sentence-Based** | Low | Fast | Medium | General documents |
| **Paragraph-Based** | Low | Fast | Medium | Structured documents |
| **Semantic** | High | Slow | High | High accuracy needs |
| **Recursive** | Medium | Medium | Medium-High | General-purpose |
| **Sliding Window** | Low | Fast | Medium | Sequential data |
| **Token-Based** | Medium | Medium | High | LLM systems |
| **Hierarchical** | High | Medium | High | Complex documents |
| **Content-Aware** | Very High | Slow | Very High | Mixed content |
| **Metadata-Enriched** | Medium | Medium | High | Production systems |

---

## Best Practices

### 1. Choose Based on Document Type

**Text Documents:**
- Start with sentence-based or recursive
- Use semantic if accuracy critical

**Code Documentation:**
- Content-aware (function/class boundaries)
- Or fixed-size with overlap

**Academic Papers:**
- Hierarchical (section → paragraph)
- Or paragraph-based

**Mixed Content:**
- Content-aware chunking
- Different strategy per type

### 2. Overlap is Critical

**Why overlap:**
- Preserves context across boundaries
- Prevents information loss
- Better retrieval

**How much:**
- 10-20% of chunk size
- 50-100 tokens common
- More for sequential data

### 3. Size Considerations

**Too small:**
- Missing context
- Incomplete information
- More chunks to manage

**Too large:**
- Exceeds context window
- Noise in retrieval
- Higher cost

**Optimal:**
- 256-1024 tokens
- 512 tokens common default
- Adjust based on model context window

### 4. Test and Iterate

**Evaluate:**
- Retrieval accuracy
- Generation quality
- User satisfaction

**Iterate:**
- Try different strategies
- Adjust parameters
- Measure impact

### 5. Production Considerations

**Scalability:**
- Fast chunking for large documents
- Efficient storage
- Incremental updates

**Quality:**
- Semantic boundaries
- Rich metadata
- Proper overlap

**Cost:**
- Balance chunk size
- Minimize redundant chunks
- Efficient processing

---

## Industry Examples

### LangChain

**Default:** Recursive chunking
- Hierarchical separators
- 1000 char chunks, 200 char overlap
- Robust for most documents

### LlamaIndex

**Default:** Sentence-based
- Smart chunking
- Metadata enrichment
- Good for structured documents

### Pinecone

**Recommendation:** 
- 500-1000 tokens
- 10-20% overlap
- Sentence or paragraph boundaries

### OpenAI RAG

**Recommendation:**
- 500-1000 tokens
- Semantic boundaries
- Rich metadata

---

## Summary

**Key Strategies:**
1. **Fixed-size**: Simple, fast, low quality
2. **Sentence-based**: Good default, medium quality
3. **Semantic**: Best quality, slower
4. **Recursive**: Robust, general-purpose
5. **Token-based**: Accurate for LLMs

**Best Practices:**
- Choose based on document type
- Use overlap (10-20%)
- Optimal size (256-1024 tokens)
- Test and iterate
- Consider production needs

**Recommendation:**
- **Start**: Sentence-based or recursive
- **Upgrade**: Semantic if accuracy critical
- **Production**: Content-aware + metadata-enriched

