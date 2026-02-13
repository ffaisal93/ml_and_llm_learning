# Retrieval Methods: BM25, TF-IDF, and Hybrid Search

## Overview

RAG systems use various retrieval methods to find relevant documents. This document covers sparse retrieval (BM25, TF-IDF) and how to combine them with dense retrieval (hybrid search).

## Retrieval Methods Comparison

| Method | Type | How it Works | Pros | Cons |
|--------|------|-------------|------|------|
| **TF-IDF** | Sparse | Term frequency × inverse document frequency | Simple, interpretable | No semantic understanding |
| **BM25** | Sparse | Advanced TF-IDF with saturation | Better than TF-IDF, handles term frequency better | Still no semantics |
| **Dense (Embeddings)** | Dense | Vector similarity (cosine, dot product) | Semantic understanding | Needs good embeddings |
| **Hybrid** | Both | Combine sparse + dense | Best of both worlds | More complex, higher cost |

---

## 1. TF-IDF Retrieval

### What is TF-IDF?

**TF-IDF** (Term Frequency-Inverse Document Frequency) is a statistical measure that reflects how important a word is to a document in a collection.

**Components:**

**Term Frequency (TF):**
```
TF(t, d) = count(t in d) / total_terms_in_d
```
- How often word appears in document
- Higher TF = more important to document

**Inverse Document Frequency (IDF):**
```
IDF(t, D) = log(N / documents_containing_t)
```
- How rare word is across documents
- Common words → low IDF
- Rare words → high IDF

**TF-IDF:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t, D)
```

### How to Use for Retrieval

**Step 1: Build TF-IDF Index**
- Compute TF-IDF for all terms in all documents
- Store in sparse matrix

**Step 2: Query Processing**
- Compute TF-IDF for query terms
- Match with document TF-IDF scores

**Step 3: Scoring**
```
Score(query, document) = Σ TF-IDF(term, document) for term in query
```

**Step 4: Ranking**
- Sort documents by score
- Return top-K

### Use Cases

**When to use:**
- ✅ Keyword-based queries
- ✅ Exact term matching important
- ✅ Fast retrieval needed
- ✅ Interpretable results

**Limitations:**
- ❌ No semantic understanding
- ❌ Can't handle synonyms
- ❌ Misses related concepts

---

## 2. BM25 (Best Matching 25)

### What is BM25?

**BM25** is an advanced ranking function that improves upon TF-IDF. It's the de facto standard for sparse retrieval in search engines.

**Key Improvements over TF-IDF:**
1. **Term frequency saturation**: Prevents very frequent terms from dominating
2. **Document length normalization**: Handles variable document lengths
3. **Tunable parameters**: Can adjust for different domains

### BM25 Formula

**Score for a single term:**
```
BM25(t, d) = IDF(t) × (f(t, d) × (k₁ + 1)) / (f(t, d) + k₁ × (1 - b + b × |d|/avgdl))

Where:
- f(t, d): Term frequency in document
- |d|: Document length
- avgdl: Average document length
- k₁: Term frequency saturation parameter (usually 1.2-2.0)
- b: Length normalization parameter (usually 0.75)
- IDF(t): Inverse document frequency
```

**IDF in BM25:**
```
IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))

Where:
- N: Total number of documents
- df(t): Number of documents containing term t
```

**Total score:**
```
BM25(query, document) = Σ BM25(term, document) for term in query
```

### Why BM25 is Better

**1. Term Frequency Saturation:**
- TF-IDF: Linear with frequency (can be dominated by one term)
- BM25: Saturates (bounded growth)
- Example: Term appears 10x vs 20x → Similar scores (not 2x)

**2. Document Length Normalization:**
- Long documents naturally have more terms
- BM25 normalizes by document length
- Prevents bias toward long documents

**3. Tunable Parameters:**
- k₁: Controls term frequency saturation
  - Higher k₁: More weight to frequency
  - Lower k₁: Less weight to frequency
- b: Controls length normalization
  - b = 0: No normalization
  - b = 1: Full normalization
  - Usually b = 0.75

### Intuitive Explanation

**Term Frequency Saturation:**
```
TF-IDF: 10 occurrences → score = 10
        20 occurrences → score = 20 (2x)

BM25:   10 occurrences → score = 8.5
        20 occurrences → score = 9.2 (only slightly higher)
```

**Why this helps:**
- Prevents one very frequent term from dominating
- More balanced scoring
- Better ranking

**Document Length Normalization:**
```
Short doc (100 words): "machine learning" appears 2x
Long doc (1000 words): "machine learning" appears 2x

TF-IDF: Same score (doesn't account for length)
BM25: Short doc gets higher score (more relevant per word)
```

### Use Cases

**When to use:**
- ✅ Keyword-based search
- ✅ Exact term matching
- ✅ Production search systems
- ✅ When you need interpretable results

**Industry Usage:**
- Elasticsearch (default ranking)
- Lucene (Apache)
- Most search engines

**Limitations:**
- ❌ Still no semantic understanding
- ❌ Can't handle synonyms well
- ❌ Misses related concepts

---

## 3. Dense Retrieval (Embeddings)

### What is Dense Retrieval?

**Dense retrieval** uses vector embeddings to find semantically similar documents.

**How it works:**
1. Encode documents to embeddings
2. Encode query to embedding
3. Compute similarity (cosine, dot product)
4. Rank by similarity

**Advantages:**
- ✅ Semantic understanding
- ✅ Handles synonyms
- ✅ Finds related concepts
- ✅ Works with modern LLMs

**Disadvantages:**
- ❌ Requires good embeddings
- ❌ Can miss exact matches
- ❌ More expensive

### Similarity Metrics

**Cosine Similarity:**
```
similarity = (A · B) / (||A|| × ||B||)
```
- Measures angle between vectors
- Range: [-1, 1] (usually [0, 1] for normalized)
- Most common for embeddings

**Dot Product:**
```
similarity = A · B
```
- Simpler computation
- Depends on vector magnitude
- Use when embeddings not normalized

**Euclidean Distance:**
```
distance = ||A - B||
```
- Measures distance
- Lower = more similar
- Less common for embeddings

---

## 4. Hybrid Search (Sparse + Dense)

### Why Hybrid Search?

**Problem:**
- **Sparse (BM25)**: Good for exact matches, keywords
- **Dense (Embeddings)**: Good for semantic similarity
- **Neither alone is perfect**

**Solution:**
- Combine both!
- Get benefits of both methods
- Better overall retrieval

### How Hybrid Search Works

**Step 1: Retrieve from Both**
- Sparse retrieval: Top-K from BM25
- Dense retrieval: Top-K from embeddings

**Step 2: Combine Scores**
```
Final_Score = α × BM25_Score + (1 - α) × Dense_Score

Where:
- α: Weight for sparse (usually 0.3-0.7)
- (1 - α): Weight for dense
```

**Step 3: Re-rank**
- Sort by combined score
- Return top-K

### Score Normalization

**Problem:** BM25 and dense scores have different scales

**Solution: Normalize before combining**

**Min-Max Normalization:**
```
normalized_score = (score - min) / (max - min)
```

**Z-score Normalization:**
```
normalized_score = (score - mean) / std
```

**Common Approach:**
- Normalize both to [0, 1]
- Then combine with weights

### Weight Selection

**How to choose α:**

**More sparse (α = 0.7):**
- Keyword-heavy queries
- Exact matching important
- Domain-specific terms

**More dense (α = 0.3):**
- Semantic queries
- Synonyms important
- General understanding

**Balanced (α = 0.5):**
- General use case
- Good default
- Works for most queries

**Tuning:**
- Test on validation set
- Measure retrieval accuracy
- Choose α that maximizes performance

### Industry Examples

**Pinecone:**
- Hybrid search with BM25 + dense
- Automatic score normalization
- Tunable weights

**Weaviate:**
- Hybrid search support
- BM25 + vector search
- Configurable weights

**Elasticsearch:**
- BM25 (default) + vector search
- Can combine in queries

---

## 5. Other Retrieval Methods

### 1. Query Expansion

**What it is:**
- Add related terms to query
- Synonyms, related concepts
- Improve recall

**Methods:**
- **Thesaurus**: Use synonym dictionary
- **Word embeddings**: Find similar words
- **LLM-based**: Generate related terms

**Example:**
```
Original: "machine learning"
Expanded: "machine learning, AI, artificial intelligence, ML, deep learning"
```

### 2. Re-ranking

**What it is:**
- Initial retrieval: Fast, approximate (BM25, dense)
- Re-ranking: Slow, accurate (cross-encoder)

**Two-stage retrieval:**
```
Stage 1: Retrieve top-100 (fast, approximate)
Stage 2: Re-rank top-100 → top-10 (slow, accurate)
```

**Re-ranking models:**
- **Cross-encoder**: More accurate than bi-encoder
- **Learning-to-rank**: Trained models
- **LLM-based**: Use LLM to score

### 3. Multi-Vector Retrieval

**What it is:**
- Multiple embeddings per document
- Different granularities (sentence, paragraph, document)
- Better coverage

**How it works:**
1. Generate embeddings at multiple levels
2. Query matches against all
3. Aggregate scores

### 4. Learned Sparse Retrieval

**What it is:**
- Learn sparse representations (like BM25 but learned)
- SPLADE, ColBERT
- Better than BM25, interpretable

**Advantages:**
- Learned (optimized for task)
- Still interpretable (sparse)
- Better than BM25

---

## When to Use Each Method

### Use BM25 When:
- ✅ Keyword-based queries
- ✅ Exact term matching important
- ✅ Fast retrieval needed
- ✅ Interpretable results
- ✅ Domain-specific terms

### Use Dense Retrieval When:
- ✅ Semantic queries
- ✅ Synonyms important
- ✅ Related concepts matter
- ✅ General understanding needed
- ✅ Good embeddings available

### Use Hybrid Search When:
- ✅ Production systems
- ✅ Want best of both
- ✅ Can afford complexity
- ✅ Mixed query types
- ✅ High accuracy needs

---

## Best Practices

### 1. Start with BM25
- Simple, fast, interpretable
- Good baseline
- Works for keyword queries

### 2. Add Dense if Needed
- If BM25 insufficient
- Semantic understanding needed
- Good embeddings available

### 3. Use Hybrid for Production
- Best overall performance
- Handles both keyword and semantic
- Industry standard

### 4. Tune Parameters
- BM25: k₁, b parameters
- Hybrid: α weight
- Test on validation set

### 5. Normalize Scores
- Before combining
- Min-max or z-score
- Ensure fair combination

---

## Summary

**Key Methods:**
1. **TF-IDF**: Simple, interpretable, keyword-based
2. **BM25**: Advanced TF-IDF, industry standard
3. **Dense**: Semantic, embeddings-based
4. **Hybrid**: Best of both (BM25 + Dense)

**Recommendation:**
- **Start**: BM25 (simple, fast)
- **Upgrade**: Add dense retrieval
- **Production**: Hybrid search (BM25 + Dense)

**Key Insight:**
- Sparse (BM25): Exact matches, keywords
- Dense: Semantic similarity
- Hybrid: Combines both for best results

