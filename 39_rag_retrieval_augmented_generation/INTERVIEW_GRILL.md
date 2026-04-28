# RAG — Interview Grill

> 45 questions on RAG systems. Drill until you can answer 30+ cold.

---

## A. Architecture and motivation

**1. Walk me through the RAG pipeline end-to-end.**
Indexing (offline): collect docs → chunk → embed → store in vector + sparse index. Query time: query → retrieve top-N (BM25 + dense + filters) → optionally rerank → build prompt with retrieved chunks → LLM generates grounded answer.

**2. Why use RAG instead of fine-tuning?**
Fine-tuning is for style/behavior; RAG is for facts. RAG is cheaper to update, handles current/proprietary knowledge, gives citations. Fine-tuning bakes facts in but: stale, expensive to update, harder to attribute.

**3. Why not just use a long-context LLM?**
Cost (context tokens are expensive). "Lost in the middle" effect — LLMs underweight middle-context content. Updating the entire context per query vs cached embeddings. RAG often wins on quality/cost even with long-context models.

**4. When does RAG fail?**
(a) Retrieval miss — relevant docs not retrieved. (b) Reranking miss — retrieved but ranked low. (c) Generation hallucination — LLM ignores retrieved context. (d) Generation contradiction — LLM contradicts sources. (e) Refusal failure — LLM declines despite answer being in context.

---

## B. Chunking

**5. Why is chunking strategy critical?**
Chunking dominates retrieval quality more than embedding choice. Bad chunks → embeddings encode unrelated content → retrieval fails systematically.

**6. What's a typical chunk size?**
256–512 tokens for most use cases. Smaller for dense factual retrieval; larger for narrative content. Domain-dependent — always evaluate.

**7. Fixed-size vs recursive vs semantic chunking?**
Fixed: simple, naive, breaks structure. Recursive: tries paragraph → sentence → word boundaries; LangChain default. Semantic: splits where adjacent sentence embeddings disagree; preserves semantic units; expensive.

**8. What's chunk overlap and why?**
10-20% overlap between adjacent chunks. Captures information crossing chunk boundaries. Increases storage; usually worth it.

**9. How do you handle structured docs (markdown, code, HTML)?**
Structure-aware splitting on natural boundaries (headers, code blocks, function definitions). Preserve metadata (file, line numbers, section). Critical for code RAG.

**10. What's hierarchical chunking?**
Index multiple granularities — chunks, paragraphs, sections, full doc. Retrieve at the right level for the query. Enables fine-grained matching plus broad context.

**11. How would you debug "retrieval looks OK but answers are bad"?**
Likely chunking — retrieval finds the right chunk but it's missing context (intro paragraph, table headers, etc.). Try larger chunks, more overlap, structure-aware splitting.

---

## C. Retrieval methods

**12. What's BM25?**
Classic IR scoring:

$$
\sum_{w \in q} \mathrm{IDF}(w) \cdot \frac{\mathrm{tf}(w, d) \cdot (k + 1)}{\mathrm{tf}(w, d) + k \cdot (1 - b + b \cdot |d| / \overline{|d|})}
$$

Term frequency × IDF with length normalization. Lexical match — strong on rare keywords, IDs, exact phrases.

**13. Dense retrieval — how does it work?**
Encode query and documents into shared vector space; retrieve by cosine similarity (or dot product). Bi-encoder: query encoder + document encoder, often shared weights. Trained with contrastive (InfoNCE) loss.

**14. Why use hybrid (BM25 + dense)?**
BM25 catches lexical/keyword matches dense misses (rare names, IDs, exact phrases). Dense catches semantic matches BM25 misses (paraphrases, synonyms). Combined: best of both.

**15. How do you fuse BM25 and dense scores?**
Weighted sum ($\alpha \cdot \text{BM25} + (1 - \alpha) \cdot \text{dense}$) requires score normalization. Reciprocal Rank Fusion (RRF): $\sum 1 / (k + \text{rank}_i)$ — rank-based, no normalization needed. RRF is the modern default.

**16. Why is reranking necessary?**
Bi-encoder retrieval is fast but coarse — embeds query and doc independently, no joint reasoning. Cross-encoder (one forward pass per (q, d) pair) sees them together — much higher precision. Two-stage architecture: bi-encoder for recall, cross-encoder for precision.

**17. Walk through a reranking workflow.**
Retrieve top-100 with bi-encoder + BM25. Rerank with cross-encoder to top-10. Pass to LLM. Bi-encoder is $O(N)$ similarity computes; cross-encoder is $O(K)$ forward passes ($K = 100$). Trade-off: latency for quality.

**18. What's a listwise reranker?**
Send top-N to an LLM; have it order them. Even higher quality than cross-encoder. Slow (one LLM call). RankGPT, RankLlama. Used when reranking quality dominates latency.

---

## D. Embedding models

**19. What makes a good retrieval embedding?**
Semantic similarity in vector space matches task-relevant similarity. Asymmetric encoding (query/doc may differ). Length handling. Domain match. Trained with contrastive loss on relevant/irrelevant pairs.

**20. Common embedding models?**
OpenAI text-embedding-3 (paid, multilingual). BGE (BAAI/bge-large-en, bge-m3) — open-source, near-SOTA. E5 — open-source, instruction-tuned. Cohere Embed v3 — strong API. Voyage AI — domain specialized.

**21. What's contrastive learning for embeddings?**
Train so positives have high similarity, negatives have low: $\mathcal{L} = -\log[\exp(\mathrm{sim}(q, d_+)) / \sum \exp(\mathrm{sim}(q, d_i))]$. InfoNCE loss. Hard negatives (almost-relevant) train better than random negatives.

**22. Embedding dimension trade-offs?**
Larger: more expressive but slower retrieval, more memory. Smaller: faster, smaller index. 384-768 typical sweet spot. Matryoshka embeddings (recent): hierarchical so you can truncate.

**23. What are Matryoshka embeddings?**
Trained so that the first $k$ dimensions form a meaningful sub-embedding for any $k$. Truncate to small dim for cheap retrieval, full dim for reranking. Used in OpenAI text-embedding-3.

---

## E. Vector databases

**24. What's HNSW?**
Hierarchical Navigable Small World. Graph-based ANN: nodes connect to nearby nodes; multiple layers for fast traversal. Sub-linear search time. Modern default in FAISS, Weaviate, Qdrant, Milvus.

**25. What's IVF?**
Inverted File. K-means cluster vectors into K groups; search only the closest few clusters. Older approach; still useful in FAISS for very large indexes.

**26. What's product quantization?**
Split vectors into sub-vectors; quantize each sub-vector to a code (k-means cluster ID). Compresses memory ~10x with minimal retrieval quality loss. Standard memory optimization.

**27. Common production vector DBs?**
FAISS (library, embeddings only). Pinecone (managed). Weaviate (open-source, hybrid). Qdrant (Rust). Milvus (large-scale). pgvector (Postgres).

**28. When would you use pgvector vs Pinecone?**
pgvector: already use Postgres, want one DB, moderate scale. Pinecone: managed, fast scaling, don't want to operate. FAISS: in-process embeddings only, full control.

---

## F. Query handling

**29. What's HyDE?**
Hypothetical Document Embeddings. Have LLM generate a hypothetical answer to the query; embed that for retrieval. Captures query intent better than literal embedding. Improves retrieval on diverse queries.

**30. What's query rewriting?**
Rewrite the user query into multiple variations; retrieve with each; union or rerank. Captures different angles. Useful when literal queries are short or under-specified.

**31. What's query decomposition?**
Break complex queries into sub-queries; retrieve for each. Multi-hop or multi-faceted questions need this — single retrieval misses parts.

**32. What's iterative / agentic RAG?**
Retrieve → answer partial → identify gaps → retrieve more. Used in FLARE, Self-RAG. The model controls the retrieval loop.

**33. Why does query rewriting matter?**
User queries are often short, ambiguous, or use different vocabulary than the docs. Rewriting bridges the gap. "OAuth impact?" → "OAuth session token security implications" → much better retrieval.

---

## G. Prompt construction

**34. How should you order retrieved chunks in the prompt?**
Place most relevant at start or end of context. "Lost in the middle" (Liu et al. 2023): LLMs underweight mid-context. Putting critical info at extremes mitigates.

**35. What metadata should you include with chunks?**
Source document, date, author, section. Helps the LLM contextualize and cite. Without metadata, all chunks look equally authoritative.

**36. Why request citations in the prompt?**
"Cite which chunk you used: [doc 3]". Helps detect hallucination, builds user trust, enables filtering or source-clicking in UI.

**37. How do you reduce hallucination in RAG generation?**
Strong instructions ("Use only provided sources; if not in sources, say so"). Cite sources. Lower temperature. Smaller, more focused chunks. Reranking quality. Calibrated refusal.

---

## H. Evaluation

**38. How do you evaluate retrieval?**
Recall@K (fraction of relevant docs in top-K). MRR (mean reciprocal rank of first relevant). NDCG (position-discounted, graded relevance). Need labeled relevance — hard at scale.

**39. How do you evaluate end-to-end RAG?**
Faithfulness: does the answer match retrieved sources? Answer relevance: does it address the query? Context precision/recall: are retrieved chunks actually useful? Frameworks: RAGAS, Trulens.

**40. What's the difference between accuracy and faithfulness in RAG?**
Accuracy: is the final answer correct? Faithfulness: is it grounded in the retrieved context? An answer can be accurate (from model's parametric knowledge) but unfaithful (not grounded), which is essentially hallucination.

**41. How do you label retrieval relevance at scale?**
Synthetic: generate queries from documents (LLM produces "what question would this answer?"). Human-annotated subsets for ground truth. Click data from production. Always validate synthetic against human.

---

## I. Advanced patterns

**42. What's Self-RAG?**
Asai et al. 2023. Model decides when to retrieve, retrieves multiple times, verifies own outputs with reflection tokens. Combines retrieval with self-evaluation.

**43. What's GraphRAG?**
Microsoft 2024. Build a knowledge graph from documents at indexing. At query time, traverse the graph for richer cross-document context. Better for synthesis questions.

**44. What's CRAG (Corrective RAG)?**
After retrieval, verify quality. If retrieved docs are weak, expand search or fall back to non-RAG generation. Adds robustness.

**45. When does long-context beat RAG?**
Tasks requiring synthesis across many documents simultaneously. When the relevant context can fit in the model's window and is self-contained. Coding contexts where the entire repo is relevant.

---

## J. Quick fire

**46.** *Default chunk size?* 256-512 tokens.
**47.** *Default overlap?* 10-20%.
**48.** *Modern hybrid retrieval default?* BM25 + dense + RRF.
**49.** *Default ANN algorithm?* HNSW.
**50.** *Common open-source embedding?* BGE family.

---

## Self-grading

If you can't answer 1-15, you don't know RAG. If you can't answer 16-30, you'll struggle on RAG-focused interviews. If you can't answer 31-50, frontier-lab interviews will go past you.

Aim for 30+/50 cold.
