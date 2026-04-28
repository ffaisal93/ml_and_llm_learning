# RAG (Retrieval-Augmented Generation): A Frontier-Lab Interview Deep Dive

> **Why this exists.** RAG is now the default architecture for any LLM application that needs current or proprietary knowledge. Interviewers probe: chunking strategy, embedding choice, retrieval metric, reranking, evaluation. Strong candidates have an opinion on each component, not just "use a vector DB."

---

## 1. The big picture

A RAG system has two phases:

**Indexing (offline):**
1. Collect documents.
2. Chunk them into passages.
3. Embed each chunk with a retrieval model.
4. Store vectors in a vector database (or dense + sparse hybrid index).

**Query time (online):**
1. User query → retrieval (vector search, BM25, or hybrid).
2. (Optional) rerank top-N candidates with a cross-encoder.
3. Build prompt: system instruction + retrieved chunks + user query.
4. LLM generates a grounded answer.

Almost every interview question lives in *one* of these steps. The hardest question is always: "Why does your retrieval pipeline retrieve junk for ~30% of queries?" — and there's never one fix.

---

## 2. Why RAG

LLMs have three problems that RAG addresses:

1. **Knowledge cutoff.** Training data has a fixed date; RAG provides current context.
2. **Hallucination.** Grounding the answer in retrieved evidence reduces fabrication.
3. **Proprietary knowledge.** Internal docs, customer data, codebases — can't be in pretraining.

RAG is also cheaper than fine-tuning for most knowledge-injection use cases. Fine-tune for *style*; RAG for *facts*.

### When RAG is the wrong tool

- **Reasoning over the whole knowledge.** RAG retrieves a few chunks; if the answer requires synthesizing across thousands of docs, retrieval misses.
- **Ambiguous queries with no obvious anchor.** "Tell me about our company" — retrieval returns whatever was indexed first; not contextualized.
- **Tasks where style or behavior matters more than facts.** Fine-tuning wins.

---

## 3. Chunking — the underestimated component

How you chunk dominates retrieval quality more than embedding choice. Common strategies:

### Fixed-size chunking
Cut documents every N tokens (e.g., 512). Simple, fast. **Bad for documents with structure** because chunks may split a list, table, or paragraph mid-sentence.

### Recursive character-based
Try to split on paragraphs first; if too large, split on sentences; if still too large, on words. Used by LangChain's default. Better than naive fixed-size for unstructured text.

### Semantic chunking
Use embedding similarity to determine chunk boundaries: split where adjacent sentences have low cosine similarity. Better preserves semantic units. More expensive.

### Document-structure-aware
For markdown/HTML/code: split on natural boundaries (headers, code blocks, function definitions). Preserves structure in metadata. Critical for code RAG.

### Hierarchical chunking
Index multiple granularities: chunks, paragraphs, sections, full doc. At query time, retrieve at the right level. Enables both fine-grained matching and broad-context answers.

### Chunk size trade-offs

- **Too small (~100 tokens):** chunks lack context; the LLM can't connect retrieved info.
- **Too large (~2000 tokens):** retrieval imprecise (one chunk covers many topics; embedding is muddled).
- **Sweet spot:** 256–512 tokens for most use cases.

### Chunk overlap
Common: 10-20% overlap between adjacent chunks. Captures information that crosses boundaries. Adds storage cost; mild retrieval improvement.

### Failure modes from chunking
- Tables split across chunks → unparsable.
- Code split mid-function → useless retrieval.
- Lists separated from intro → context lost.
- Multilingual text with mixed structure → unreliable splitting.

### Interview takeaway
"Tell me about your chunking strategy" is now a serious interview question. Saying "I just use 512 tokens" is a red flag. Saying "I evaluated semantic chunking against fixed-size on our domain and chose X for these reasons" is a good signal.

---

## 4. Retrieval methods

### Sparse: BM25
Classic IR scoring:

$$
\operatorname{BM25}(q, d) = \sum_{w \in q} \operatorname{IDF}(w) \cdot \frac{\operatorname{tf}(w, d) \cdot (k+1)}{\operatorname{tf}(w, d) + k \cdot (1 - b + b \cdot |d| / \overline{|d|})}
$$

Term frequency × inverse document frequency, with normalizations. **Lexical match** — works for keywords, exact phrases, codes, IDs.

**Strengths:** Strong baseline. Great for queries with rare/specific keywords. No training needed.

**Weaknesses:** No semantic understanding. "What's a transformer?" doesn't match "self-attention is the foundation of..."

### Dense: vector retrieval
Embed query and documents into a shared vector space; retrieve by cosine similarity (or dot product, or Euclidean). Bi-encoder architecture: one encoder embeds the query, another (often shared) embeds documents.

**Strengths:** Semantic match. Handles paraphrasing, synonyms, multilingual.

**Weaknesses:** Can miss exact-keyword queries (rare names, IDs). Embedding quality is critical. Vector index size grows with corpus.

### Hybrid: BM25 + dense
Combine scores:

$$
\text{score} = \alpha \cdot \text{BM25\_score} + (1 - \alpha) \cdot \text{dense\_score}
$$

or use Reciprocal Rank Fusion (RRF):

$$
\operatorname{RRF}(d) = \sum_i \frac{1}{k + \operatorname{rank}_i(d)}
$$

**Empirically dominant** for production RAG. Captures both lexical and semantic signals.

### Cross-encoder reranking
After retrieving top-N candidates, rerank with a cross-encoder: a transformer that takes (query, document) as input and outputs a relevance score. Far more accurate than bi-encoder retrieval but slower (one forward pass per (q, d) pair).

**Pipeline:**
1. Retrieve top-100 with bi-encoder + BM25.
2. Rerank to top-10 with cross-encoder.
3. Pass top-10 to LLM.

This two-stage retrieval is the modern production default.

---

## 5. Embedding models

### What makes a good retrieval embedding
- **Semantic similarity** in the vector space matches task-relevant similarity.
- **Asymmetric encoding:** queries and documents may need different encoders or instructions.
- **Length handling:** can encode chunks of various sizes.
- **Domain match:** general-purpose embeddings struggle on specialized domains (legal, medical, code).

### Common embedding models

- **OpenAI `text-embedding-3` family** — strong general-purpose, multilingual.
- **BGE (BAAI/bge-large-en, bge-m3)** — open-source, often near-SOTA, trained with contrastive learning.
- **E5 (intfloat/e5-large)** — open-source, instruction-tuned, asymmetric for query/document.
- **Cohere Embed v3** — strong API option, multilingual.
- **Voyage AI** — domain-specialized embedding models (legal, code, finance).

### Embedding training: contrastive learning
Train so positives (relevant pairs) have high similarity; negatives (irrelevant) have low.

$$
\mathcal{L} = -\log \frac{\exp(\operatorname{sim}(q, d_+))}{\sum_i \exp(\operatorname{sim}(q, d_i))}
$$

This is **InfoNCE** loss. Standard for retrieval embeddings. Negative mining matters: hard negatives (almost-relevant docs) train better than random negatives.

### Embedding dimension trade-offs
- **Small (64-128 dim):** fast retrieval, smaller index, may lose fidelity.
- **Medium (384-768 dim):** sweet spot for most production.
- **Large (1024-3072 dim):** best quality but storage and search cost grows linearly.
- **Matryoshka (recent):** train embeddings hierarchically so you can truncate to smaller dimensions for cheap retrieval, full dimension for reranking.

---

## 6. Vector databases

### Approximate nearest neighbor (ANN) algorithms

**HNSW (Hierarchical Navigable Small World).** Graph-based. Each node connects to nearby nodes; multiple layers for fast traversal. **Modern default** — used by FAISS, Weaviate, Qdrant, Milvus. Sub-linear search time.

**IVF (Inverted File).** Cluster vectors into K groups (k-means); search only the closest few clusters. Older approach; still in some FAISS configurations.

**LSH (Locality-Sensitive Hashing).** Hash similar vectors to the same buckets. Less popular than HNSW; some tradeoffs in tuning.

**Flat / brute force.** Compare query to every vector. Exact but $O(N)$. Used for small corpora or as ground truth for evaluating ANN.

### Quantization

**Product Quantization (PQ).** Split vectors into sub-vectors; quantize each sub-vector to a code. Compresses memory ~10x with minimal retrieval quality loss.

**Scalar Quantization.** Reduce float precision (fp32 → int8 or int4). Simple; good in practice.

### Production vector DBs

- **FAISS** — Meta's library, many algorithms, embeddings only.
- **Pinecone** — managed cloud service.
- **Weaviate** — open-source, hybrid search.
- **Qdrant** — open-source, Rust-based, fast.
- **Milvus** — open-source, scales to billions.
- **pgvector** — Postgres extension; great if you already use Postgres.

### Hybrid index architectures

Many production systems combine:
- An inverted index for BM25 (Elasticsearch, OpenSearch, Tantivy).
- A vector index for dense retrieval (FAISS, HNSW).
- Metadata filtering on top (date, source, user permissions).

The fusion at query time is critical and often custom.

---

## 7. Query rewriting and expansion

The user's literal query often isn't optimal for retrieval. Improvements:

### Hypothetical Document Embeddings (HyDE)
Have the LLM write a *hypothetical* document that would answer the query; embed that for retrieval. Empirically improves retrieval on diverse queries.

### Query expansion / rewriting
Rewrite the query into multiple variations; retrieve with each; union or rerank. Captures different angles.

### Multi-step / iterative retrieval
For complex queries: retrieve → answer partial → identify gaps → retrieve more. Used in agent-style RAG (FLARE, IRCoT).

### Decomposition
Break the query into sub-queries; retrieve for each; combine. Common for multi-hop questions.

### Why this matters
A user asks "What's the impact of OAuth on session handling?" — direct retrieval might find generic OAuth docs but miss session-specific implications. A rewritten query "OAuth session token lifetime concurrent device handling" retrieves different (better) chunks.

---

## 8. Reranking deeper

Reranking is the highest-leverage step in production RAG. Strong rerankers move a 60% accuracy retriever to 90%.

### Cross-encoder rerankers
- **Cohere Rerank** — API, strong quality.
- **bge-reranker-large, bge-reranker-v2** — open-source.
- **MS MARCO models** — general baselines.

### Listwise rerankers (recent)
Send the top-N to an LLM with the query; have it order them. Higher quality, much slower. RankGPT, RankLlama.

### Why rerank instead of better retrieval?
Bi-encoder retrieval (one forward pass per doc, offline-indexed) doesn't see the query and document together. Cross-encoder (one forward pass per pair) does — much richer feature interaction. The two-stage architecture trades off recall vs precision.

---

## 9. Prompt construction

After retrieval, you have N chunks. The prompt design choices:

### Chunk ordering
Place most relevant chunks first or last? **"Lost in the middle" (Liu et al. 2023):** LLMs pay less attention to mid-context tokens. Recommendation: place the most relevant retrieved chunks at the start or end of the context.

### Chunk metadata
Including chunk source, date, author, etc., as part of the prompt helps the model contextualize. Without it, all chunks look equally authoritative.

### Citation requests
Ask the LLM to cite which chunk it used: `[doc 3]`, `[chunk: ...]`. Helps with hallucination detection and user trust.

### Context window budget
Prompt format eats tokens: system prompt + N chunks (each ~512 tokens) + user query + response. Stay within the model's context window with margin.

### Instruction prompting
"Use only the provided sources to answer. If the answer isn't in the sources, say so." Reduces hallucination at the cost of sometimes refusing answerable questions.

---

## 10. Evaluation of RAG

This is hard. Many metrics; no perfect one.

### Retrieval metrics (offline)
- **Recall@K:** of all relevant docs, what fraction in top-K?
- **MRR:** mean reciprocal rank of first relevant doc.
- **NDCG:** position-discounted with graded relevance.

Need labeled relevance: hard to get at scale. Often: synthetic queries from documents, human-annotated subsets.

### Answer quality (end-to-end)
- **Faithfulness:** does the answer match the retrieved sources? Not just "is the answer right?" but "is it grounded?"
- **Answer relevance:** does the answer address the question?
- **Context precision:** are retrieved chunks actually used in the answer?
- **Context recall:** are all relevant chunks retrieved?

### LLM-as-judge for RAG
RAGAS, Trulens, ARES — frameworks for LLM-judged faithfulness and relevance. Standard in production.

### Failure mode taxonomy
- **Retrieval miss:** relevant docs not retrieved.
- **Reranking miss:** retrieved but ranked low.
- **Generation hallucination:** LLM ignores retrieved context.
- **Generation contradiction:** LLM contradicts retrieved sources.
- **Refusal failure:** LLM says "I don't know" when answer is in context.

Each requires different fixes.

---

## 11. Advanced patterns

### Self-RAG (Asai et al. 2023)
The model decides when to retrieve, retrieves multiple times, and verifies its own outputs. Combines retrieval with self-reflection.

### CRAG (Corrective RAG)
After retrieval, verify quality. If retrieved docs are weak, expand search or fall back to no-RAG generation.

### GraphRAG (Microsoft)
Build a knowledge graph from documents at indexing time. At query time, traverse the graph for richer context. Better for cross-document synthesis.

### Long-context vs RAG
GPT-4 with 128K context vs RAG: which wins? Empirically, RAG often wins on cost and quality even with long-context models, because long context dilutes attention. But the answer is task-dependent.

### Agentic RAG
Treat retrieval as a tool an agent can call multiple times, with reflection. Used in advanced research and code-assistant systems.

### ColBERT / Late-interaction retrieval

Instead of embedding each document as one vector, embed each *token* and compute query-document similarity as a max-over-tokens of dot products: $s(q,d) = \sum_i \max_j q_i^\top d_j$. Captures fine-grained term-level matches that single-vector retrieval misses. Trade-off: $O(\mathrm{tokens})$ storage instead of $O(\mathrm{docs})$. ColBERTv2 + PLAID make it tractable for production (compress + ANN-prune the per-token vectors).

When to consider: retrieval where rare entities / exact phrases matter (legal, scientific, code).

### Late chunking (Jina, 2024)

Embed the full document with a long-context encoder, *then* slice the resulting per-token embeddings into chunks. Each chunk's embedding now incorporates surrounding context. Beats naïve chunk-then-embed when chunks need to "know" their document context.

### Contextual retrieval (Anthropic, 2024)

Before chunking, prepend each chunk with an LLM-generated 1-paragraph summary of its surrounding context. Reduces "this chunk talks about an unnamed `it`" failures. ~35% lower retrieval failure rate at minor pre-processing cost.

### HyDE refinements: hypothetical question generation

Reverse-HyDE: instead of generating a hypothetical *answer*, generate hypothetical *questions* that each chunk could answer; embed those alongside the chunks. Boosts retrieval of "this answers Q" patterns.

### RAG with re-ranking models (modern)

Strong cross-encoder rerankers as of 2024-2025: BGE-reranker (BAAI), Cohere Rerank 3, Jina ColBERT v2, FlashRank. All accept (query, doc) and output relevance score. ~10-50× more accurate than dense bi-encoder ranking on top-K. Standard production stack: dense retriever → top-100 → reranker → top-10 → generator.

### Multi-vector / multi-representation indexing

Index a document under *multiple* representations: original text, summary, keywords, hypothetical questions. Retrieve against each independently; merge. Handles queries that match different aspects of the same document.

### Recursive retrieval / iterative RAG

Single-shot RAG: 1 query → 1 retrieval → 1 generation. Recursive RAG: LLM may issue follow-up queries based on what it learned from initial retrieval. Composes well with agent loops. Good for complex multi-hop questions ("Who was the third president of country X's neighbor?").

### Knowledge graph RAG (revisited)

Beyond Microsoft's GraphRAG: extract entities and relations at indexing time → build a graph → at query time, walk the graph from query-mentioned entities. Excels at multi-hop reasoning and cross-document synthesis. Costs: graph construction is expensive; graph schemas need maintenance.

### RAG evaluation frameworks

- **RAGAS** (current standard): faithfulness, answer relevance, context precision, context recall.
- **TruLens**: similar metrics with finer-grained breakdowns.
- **DeepEval**: open-source, supports RAGAS metrics + custom.
- **Per-component metrics**: retrieval Recall@K and MRR for the retrieval stage; faithfulness for generation.

Frontier-lab interview probe: "How would you measure if RAG is helping?" Don't say "AB test." Say: faithfulness ↑ and end-to-end task accuracy ↑ on a held-out eval set; latency and cost trade-off; iterate per-stage with RAGAS.

### Hybrid scoring conventions

Reciprocal Rank Fusion (RRF): given ranked lists from multiple retrievers, score each doc as $\sum_r 1/(k + \mathrm{rank}_r)$ with $k \approx 60$. Robust, parameter-free, dominant in production. Beats fancy weighted combinations in most empirical studies.

---

## 12. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "Why not just use long-context LLMs instead of RAG?" | RAG is cheaper, more current, easier to update; long-context models suffer "lost in the middle" and dilute attention; cost scales with context. |
| "How do you choose chunk size?" | Domain-specific. 256–512 tokens for most. Smaller for dense factual retrieval; larger for narrative content. Always evaluate. |
| "Why do you need reranking?" | Bi-encoder retrieval is fast but imprecise. Cross-encoder reranks see (q, d) jointly — much higher precision at small N. |
| "BM25 vs dense retrieval?" | BM25: lexical, great for rare terms. Dense: semantic, great for paraphrasing. Hybrid wins almost always. |
| "What's HyDE?" | Have LLM generate hypothetical answer; embed it for retrieval. Captures query intent better than literal embedding. |
| "Why does RAG sometimes hallucinate?" | LLM ignores retrieved context, or context is incomplete/contradictory, or prompt didn't enforce grounding. |
| "How do you evaluate RAG?" | Retrieval (Recall@K, MRR), answer (faithfulness, relevance), end-to-end (LLM-judge or human). RAGAS framework. |
| "What's the lost-in-the-middle problem?" | LLMs underweight mid-context tokens. Place most relevant content at start or end of prompt. |

---

## 13. The 10 most-asked RAG interview questions

1. **Walk me through the RAG pipeline.** Index (chunk + embed + store) → retrieve (vector + BM25 + filter) → rerank → generate with grounded prompt.
2. **How do you chunk documents?** Recursive character / semantic / structure-aware. 256-512 tokens typical. Critical: preserve structure.
3. **Sparse vs dense retrieval?** BM25 (lexical, exact-match) vs dense (semantic). Hybrid wins.
4. **What's reranking for?** After bi-encoder retrieval, cross-encoder reranks for higher precision.
5. **Common embedding models?** BGE, E5, OpenAI text-embedding-3, Cohere v3.
6. **Vector DB algorithms?** HNSW (modern default), IVF, LSH. Quantization (PQ, scalar) for memory.
7. **What's HyDE?** Generate hypothetical document; embed it for retrieval.
8. **How do you evaluate retrieval?** Recall@K, MRR, NDCG with labeled relevance.
9. **Lost-in-the-middle?** LLMs ignore middle of context. Put critical info at start/end.
10. **When does RAG fail?** Retrieval miss, ranking miss, hallucination, refusal — each needs different fix.

---

## 14. Drill plan

1. Master the indexing → retrieval → rerank → generate pipeline.
2. Know chunking strategies and trade-offs.
3. BM25 + dense + reranking; explain why.
4. RAGAS-style faithfulness/relevance evaluation.
5. Drill `INTERVIEW_GRILL.md`.

---

## 15. Further reading

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) — original RAG paper.
- Karpukhin et al., "Dense Passage Retrieval for Open-Domain QA" (DPR, 2020).
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023).
- Gao et al., "HyDE" (2022).
- Asai et al., "Self-RAG" (2023).
- Shi et al., "ReplugRetrieval-Augmented Black-Box Language Models" (2023).
- Yang et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023).
- Microsoft, "GraphRAG" (2024).
