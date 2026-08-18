# RAG — Interview Grill

> 45 questions on RAG systems. Drill until you can answer 30+ cold.

---

## A. Architecture and motivation

**1. Walk me through the RAG pipeline end-to-end.**
Indexing (offline): collect docs → chunk → embed → store in vector + sparse index. Query time: query → retrieve top-N (BM25 + dense + filters) → optionally rerank → build prompt with retrieved chunks → LLM generates grounded answer.

> **Saying it out loud.** RAG is two pipelines, one offline and one online. Offline you take your documents, cut them into chunks, embed each chunk, and put them in a vector index plus a keyword index. Online, a question comes in, you search both indexes, take maybe the top hundred, rerank them down to the best handful, paste those into the prompt, and ask the model to answer using only what you gave it. That's the whole thing. The part people underinvest in is the offline half — chunking and indexing decide your ceiling, and no amount of prompt engineering recovers a document you never retrieved.

**2. Why use RAG instead of fine-tuning?**
Fine-tuning is for style/behavior; RAG is for facts. RAG is cheaper to update, handles current/proprietary knowledge, gives citations. Fine-tuning bakes facts in but: stale, expensive to update, harder to attribute.

> **Saying it out loud.** The one-liner is: fine-tuning teaches behavior, RAG supplies facts. If you want the model to write in your company's voice or follow a specific output format, fine-tune. If you want it to know what your policy document said as of this morning, retrieve. The practical arguments all favor RAG for knowledge — updating means re-indexing one document instead of re-running a training job, and you get citations, so a user can check the source. The failure mode of baking facts into weights is that they go stale silently and you can't tell where an answer came from.

**3. Why not just use a long-context LLM?**
Cost (context tokens are expensive). "Lost in the middle" effect — LLMs underweight middle-context content. Updating the entire context per query vs cached embeddings. RAG often wins on quality/cost even with long-context models.

> **Saying it out loud.** Because stuffing everything into context is expensive and it doesn't even work that well. Cost is linear in tokens on every single query, so pushing a million tokens per question is wildly more expensive than retrieving five thousand. And quality actually degrades — the "lost in the middle" result showed models attend well to the beginning and end of a long context and reliably underweight the middle, so burying the answer at position 400 of 800 makes it worse than not having it. Long context and RAG aren't rivals, though. The right framing is that long context raises the ceiling on how much retrieved material you can use, it doesn't remove the need to choose what to retrieve.

**4. When does RAG fail?**
(a) Retrieval miss — relevant docs not retrieved. (b) Reranking miss — retrieved but ranked low. (c) Generation hallucination — LLM ignores retrieved context. (d) Generation contradiction — LLM contradicts sources. (e) Refusal failure — LLM declines despite answer being in context.

> **Saying it out loud.** RAG fails in stages, and knowing which stage is most of debugging. The retriever can miss the document entirely, which is fatal and invisible downstream. It can retrieve the document but rank it below your cutoff. Or the model can have the right context in front of it and still answer from its own parametric memory, contradict the source, or refuse even though the answer is right there. The reason to enumerate them is that they have completely different fixes — a retrieval miss needs better chunking or hybrid search, while a grounded-but-ignored context needs prompt instructions and lower temperature. Always measure retrieval recall separately from answer quality, or you'll tune the wrong stage.

---

## B. Chunking

**5. Why is chunking strategy critical?**
Chunking dominates retrieval quality more than embedding choice. Bad chunks → embeddings encode unrelated content → retrieval fails systematically.

> **Saying it out loud.** Chunking matters more than which embedding model you pick, and that surprises people. The reason is that an embedding is a single vector summarizing whatever text you handed it — so if a chunk splices together the end of one topic and the start of another, that vector represents neither, and it will never be the nearest neighbor for either question. Worse, the failure is systematic rather than random: every query about that topic fails the same way. Swapping from a decent embedder to a great one might buy you a couple of recall points; fixing chunking on a badly split corpus can buy you twenty.

**6. What's a typical chunk size?**
256–512 tokens for most use cases. Smaller for dense factual retrieval; larger for narrative content. Domain-dependent — always evaluate.

> **Saying it out loud.** The default answer is 256 to 512 tokens, and I'd give that number but immediately say what drives it. Small chunks give precise embeddings — one idea per vector, so retrieval is sharp — but they can arrive at the model missing the context needed to interpret them. Big chunks carry their context but their embeddings get muddy, since one vector is now averaging several topics. So you go smaller for dense factual lookup like API docs and FAQs, and larger for narrative or argumentative text where meaning spans paragraphs. And the honest ending is that it's corpus-dependent and you should measure recall at a few sizes rather than trusting the default.

**7. Fixed-size vs recursive vs semantic chunking?**
Fixed: simple, naive, breaks structure. Recursive: tries paragraph → sentence → word boundaries; LangChain default. Semantic: splits where adjacent sentence embeddings disagree; preserves semantic units; expensive.

> **Saying it out loud.** Fixed-size just counts tokens and cuts, which is trivial and will happily slice a sentence or a table in half. Recursive splitting tries a hierarchy of separators — paragraph breaks first, then sentences, then words — so it only makes an ugly cut when it has to, and it's the sensible default. Semantic chunking embeds each sentence and cuts where consecutive sentences stop being similar, so boundaries land on real topic shifts. The tradeoff is cost: semantic chunking requires embedding every sentence at index time, and in most evaluations it beats recursive by a modest margin that often doesn't justify the pipeline complexity.

**8. What's chunk overlap and why?**
10-20% overlap between adjacent chunks. Captures information crossing chunk boundaries. Increases storage; usually worth it.

> **Saying it out loud.** Overlap means consecutive chunks share some text at the seam, typically 10 to 20 percent. The point is that a fact can straddle a boundary — a definition in one sentence and its consequence in the next — and without overlap neither chunk contains the complete statement, so neither one retrieves well. Overlap gives every span a chance to appear intact in at least one chunk. The cost is straightforward: 20 percent overlap means 20 percent more vectors, more storage, more search cost, and duplicate results you may want to deduplicate before building the prompt. It's almost always worth it.

**9. How do you handle structured docs (markdown, code, HTML)?**
Structure-aware splitting on natural boundaries (headers, code blocks, function definitions). Preserve metadata (file, line numbers, section). Critical for code RAG.

> **Saying it out loud.** For anything with real structure, you split on the structure instead of on token counts. Markdown splits on headers, code splits on function and class boundaries, HTML on the DOM. That way a chunk is a semantically complete unit rather than an arbitrary window. The second half of the answer is metadata: carry the file path, the section heading, line numbers, and the date along with the chunk, both because the model can use them to contextualize and because you need them to produce citations. For code specifically, the killer detail is that a function body without its imports and class context is often uninterpretable, so you prepend the enclosing context to every chunk.

**10. What's hierarchical chunking?**
Index multiple granularities — chunks, paragraphs, sections, full doc. Retrieve at the right level for the query. Enables fine-grained matching plus broad context.

> **Saying it out loud.** Hierarchical chunking is the answer to the small-versus-large chunk dilemma: index both. You embed small chunks so matching is precise, but you store a pointer from each small chunk up to its parent section, and when a small chunk wins you send the parent to the LLM. That's often called small-to-big or parent-document retrieval, and it gets you precise search with complete context. There's also the summary variant, where you index an LLM-generated summary of each section and retrieve the full section behind it. The tradeoff is index size and pipeline complexity — you're maintaining multiple representations of the same document and they have to stay in sync.

**11. How would you debug "retrieval looks OK but answers are bad"?**
Likely chunking — retrieval finds the right chunk but it's missing context (intro paragraph, table headers, etc.). Try larger chunks, more overlap, structure-aware splitting.

> **Saying it out loud.** My first hypothesis is that the chunks are right but incomplete. Classic case: the retrieved chunk is row 40 of a table and the column headers were in a different chunk, so the model is looking at numbers with no idea what they mean. Same with a paragraph whose subject was named only in the section intro. The tell is to actually read the retrieved chunks — not the scores, the text — and ask whether you personally could answer from them. If you can't, it's a chunking problem, and the fixes are bigger chunks, more overlap, or parent-document retrieval. If you *could* answer and the model didn't, then it's a generation problem and you go work on the prompt instead.

---

## C. Retrieval methods

**12. What's BM25?**
Classic IR scoring:

$$
\sum_{w \in q} \mathrm{IDF}(w) \cdot \frac{\mathrm{tf}(w, d) \cdot (k + 1)}{\mathrm{tf}(w, d) + k \cdot (1 - b + b \cdot |d| / \overline{|d|})}
$$

Term frequency × IDF with length normalization. Lexical match — strong on rare keywords, IDs, exact phrases.

> **Saying it out loud.** BM25 is the workhorse keyword scorer and it's basically TF-IDF with two corrections. Term frequency saturates, so a document mentioning your term fifty times isn't fifty times better than one mentioning it once — the $k_1$ parameter bends that curve flat around 1.2 to 2.0. And it normalizes for document length with $b$, usually 0.75, so long documents don't win by sheer volume. Multiply by IDF so rare terms count more, and you have a ranking function that needs no training at all. That's why it's still in every production stack — it's free, it's fast, and it beats embeddings outright on exact identifiers.

**13. Dense retrieval — how does it work?**
Encode query and documents into shared vector space; retrieve by cosine similarity (or dot product). Bi-encoder: query encoder + document encoder, often shared weights. Trained with contrastive (InfoNCE) loss.

> **Saying it out loud.** Dense retrieval embeds the query and every document into the same vector space and returns the nearest neighbors. The key architectural word is bi-encoder: query and document are encoded completely independently, which is what lets you precompute every document vector offline and reduce query time to one embedding plus a nearest-neighbor search. Training is contrastive — pull the query toward its correct passage and away from others, and the quality hinges on mining hard negatives rather than random ones. The tradeoff to name is that because the two sides never interact, all the matching has to survive being compressed into a single dot product, which is exactly what a cross-encoder reranker fixes afterward.

**14. Why use hybrid (BM25 + dense)?**
BM25 catches lexical/keyword matches dense misses (rare names, IDs, exact phrases). Dense catches semantic matches BM25 misses (paraphrases, synonyms). Combined: best of both.

> **Saying it out loud.** Because they fail in opposite places. Dense retrieval shines when the user's words don't appear in the document at all — "my laptop won't boot" finding a page about POST failures. It's weak on exact strings: a part number, an error code, a person's surname, because those get shattered by the tokenizer and their meaning isn't compositional. BM25 nails exactly those and is useless on paraphrase. Since the errors are close to uncorrelated, taking the union recovers substantially more than either alone, usually several points of recall, and it's cheap. That's why "hybrid by default" is the standard recommendation.

**15. How do you fuse BM25 and dense scores?**
Weighted sum ($\alpha \cdot \text{BM25} + (1 - \alpha) \cdot \text{dense}$) requires score normalization. Reciprocal Rank Fusion (RRF): $\sum 1 / (k + \text{rank}_i)$ — rank-based, no normalization needed. RRF is the modern default.

> **Saying it out loud.** This is where naive implementations go wrong. BM25 scores are unbounded and query-dependent, cosine similarities live between minus one and one, so you cannot just add them — and normalizing per query is fragile because the score distributions shift with query length and rarity. Reciprocal rank fusion sidesteps the whole problem by throwing the scores away and using only rank position: each document scores one over sixty plus its rank in each list, and you sum. It has one parameter, it needs no calibration, and it consistently holds up against carefully tuned weighted blends. That robustness is why it's the default.

**16. Why is reranking necessary?**
Bi-encoder retrieval is fast but coarse — embeds query and doc independently, no joint reasoning. Cross-encoder (one forward pass per (q, d) pair) sees them together — much higher precision. Two-stage architecture: bi-encoder for recall, cross-encoder for precision.

> **Saying it out loud.** A bi-encoder has to compress an entire document into one vector before it has any idea what the question will be. That's a hard constraint, and it means subtle relevance distinctions get lost. A cross-encoder takes the query and the document together in one forward pass, so every query token can attend to every document token, and it's dramatically more precise. The catch is obvious: you can't precompute anything, so it's one full transformer pass per candidate. Hence the two-stage design — the bi-encoder's job is recall over millions, the cross-encoder's job is precision over a hundred. Typical gains are five to ten points of NDCG for maybe fifty to two hundred milliseconds.

**17. Walk through a reranking workflow.**
Retrieve top-100 with bi-encoder + BM25. Rerank with cross-encoder to top-10. Pass to LLM. Bi-encoder is $O(N)$ similarity computes; cross-encoder is $O(K)$ forward passes ($K = 100$). Trade-off: latency for quality.

> **Saying it out loud.** The standard shape is a funnel. Hybrid retrieval pulls maybe a hundred candidates out of millions in a few milliseconds, the cross-encoder scores those hundred and keeps the top five to ten, and those go into the prompt. The number that matters is where you set the first-stage cutoff, because reranking can only reorder what retrieval already found — if the right document is at rank 150 and you took the top 100, no reranker saves you. So you measure recall at your cutoff separately, and if it's below about 90 percent you widen the funnel rather than buying a better reranker. The cost is latency, linear in candidates, which is why 100 is the usual compromise.

**18. What's a listwise reranker?**
Send top-N to an LLM; have it order them. Even higher quality than cross-encoder. Slow (one LLM call). RankGPT, RankLlama. Used when reranking quality dominates latency.

> **Saying it out loud.** A cross-encoder scores each document on its own, so it never compares candidates directly. A listwise reranker hands the whole shortlist to an LLM and asks it to put them in order, which lets it reason about relative merit — this one is redundant with that one, this one is more specific. Quality is the best of the three approaches. The costs are real: it's a full LLM call, so it's hundreds of milliseconds to seconds, it's expensive per query, and it hits context limits so you have to slide a window over the candidates. You use it for offline evaluation, for generating training labels for a cheaper reranker, or in products where a second of latency is acceptable.

---

## D. Embedding models

**19. What makes a good retrieval embedding?**
Semantic similarity in vector space matches task-relevant similarity. Asymmetric encoding (query/doc may differ). Length handling. Domain match. Trained with contrastive loss on relevant/irrelevant pairs.

> **Saying it out loud.** A good retrieval embedding is one where geometric closeness means what your task needs it to mean, which is not the same as general similarity. Retrieval is asymmetric — a short question and a long passage that answers it aren't similar in any ordinary sense, they're *relevant* to each other — which is why modern embedders use instruction prefixes like "query:" and "passage:" and why forgetting them quietly destroys your recall. Beyond that: it has to handle your document lengths, it has to have seen your domain, and it should be trained contrastively with hard negatives. The thing to say last is that MTEB leaderboard position is weak evidence; you evaluate on your own corpus.

**20. Common embedding models?**
OpenAI text-embedding-3 (paid, multilingual). BGE (BAAI/bge-large-en, bge-m3) — open-source, near-SOTA. E5 — open-source, instruction-tuned. Cohere Embed v3 — strong API. Voyage AI — domain specialized.

> **Saying it out loud.** On the hosted side it's OpenAI's text-embedding-3, Cohere Embed, and Voyage for domain-specialized work like code and legal. On the open side, BGE and E5 are the families people actually deploy, and bge-m3 is the go-to when you need multilingual and long documents. The decision usually isn't about leaderboard rank, it's about operational shape: a hosted API means no GPU to run but a per-token bill and your documents leaving your network, while an open model means you own the inference and can fine-tune on your domain. And the migration cost is the thing to flag — changing embedding models means re-embedding your entire corpus, so it's not a decision you revisit casually.

**21. What's contrastive learning for embeddings?**
Train so positives have high similarity, negatives have low: $\mathcal{L} = -\log[\exp(\mathrm{sim}(q, d_+)) / \sum \exp(\mathrm{sim}(q, d_i))]$. InfoNCE loss. Hard negatives (almost-relevant) train better than random negatives.

> **Saying it out loud.** Contrastive training is multiple choice. Give the model a query, one correct passage, and a batch of wrong ones, and use cross-entropy to make the correct one score highest. That's InfoNCE. The entire art is in choosing the negatives. Random negatives from the corpus are trivially wrong — different topic, different vocabulary — so the model learns coarse topic matching and then plateaus. Hard negatives are passages that look right and aren't, usually mined by running BM25 or an earlier checkpoint and taking high-ranked non-answers, and they're what teach fine distinctions. The failure mode to name is false negatives: mine too aggressively and you'll label genuinely correct passages as negatives and actively train the model to be wrong.

**22. Embedding dimension trade-offs?**
Larger: more expressive but slower retrieval, more memory. Smaller: faster, smaller index. 384-768 typical sweet spot. Matryoshka embeddings (recent): hierarchical so you can truncate.

> **Saying it out loud.** Dimension is a straight cost-versus-quality dial, and the cost side is bigger than people expect. Memory and search time scale linearly with dimension, so going from 384 to 1536 quadruples your index footprint — and at ten million chunks that's the difference between roughly fifteen gigabytes and sixty. Quality gains are real but sublinear and flatten out fast; 384 to 768 is where most corpora sit. What makes this less painful now is Matryoshka training, where the first $k$ dimensions are themselves a usable embedding, so you can retrieve cheaply on a truncated vector and rescore the top candidates at full dimension.

**23. What are Matryoshka embeddings?**
Trained so that the first $k$ dimensions form a meaningful sub-embedding for any $k$. Truncate to small dim for cheap retrieval, full dim for reranking. Used in OpenAI text-embedding-3.

> **Saying it out loud.** Matryoshka embeddings are trained so that the information is front-loaded: the first 64 dimensions are a complete little embedding, the first 256 are a better one, and so on out to the full width. Normally truncating a vector destroys it, because the information is spread evenly; here the loss explicitly optimizes every prefix length simultaneously. The payoff is a two-stage search where you scan the whole index at 256 dimensions and rescore the top thousand at 1536, which cuts memory and search cost several-fold for a point or two of recall. It's in OpenAI's text-embedding-3, which is why you can pass a dimensions parameter and still get a sensible vector.

---

## E. Vector databases

**24. What's HNSW?**
Hierarchical Navigable Small World. Graph-based ANN: nodes connect to nearby nodes; multiple layers for fast traversal. Sub-linear search time. Modern default in FAISS, Weaviate, Qdrant, Milvus.

> **Saying it out loud.** HNSW builds a graph where each vector points to its near neighbors, stacked in layers, with the top layers being sparse long-range links. Searching is like zooming in on a map: you start at the top, greedily walk toward the query, drop a layer, refine, repeat. That gives roughly logarithmic search instead of linear. It's the default in basically every vector database because the recall-versus-latency curve is excellent and you tune it with two knobs, `M` for graph degree and `efSearch` at query time. The costs to name are memory — the graph plus full-precision vectors is often several times the raw data — and that deletes are awkward, so heavily churning indexes need periodic rebuilds.

**25. What's IVF?**
Inverted File. K-means cluster vectors into K groups; search only the closest few clusters. Older approach; still useful in FAISS for very large indexes.

> **Saying it out loud.** IVF is the simpler idea: k-means your vectors into buckets, store a centroid per bucket, and at query time only search the handful of buckets whose centroids are closest. If you have a thousand clusters and probe ten of them, you've cut the work by a hundredfold. It's cheaper to build than HNSW and much cheaper in memory, which is why it's still the base layer for billion-scale indexes. The failure mode is boundary effects — if the true nearest neighbor sits just across a cluster border and you didn't probe that cluster, you simply never see it, so `nprobe` is a direct recall knob.

**26. What's product quantization?**
Split vectors into sub-vectors; quantize each sub-vector to a code (k-means cluster ID). Compresses memory ~10x with minimal retrieval quality loss. Standard memory optimization.

> **Saying it out loud.** Product quantization chops each vector into segments, runs k-means on each segment independently, and stores just the cluster ID per segment. So a 768-dimensional float vector, which is about three kilobytes, becomes something like 64 or 96 bytes — call it a thirty- to fiftyfold reduction. Distances get computed against the codebooks with precomputed lookup tables, so it's fast as well as small. It's lossy, obviously, so the standard pattern is PQ for the coarse scan and then re-ranking the top few hundred against full-precision vectors kept on disk. That's how a billion vectors fits in memory at all.

**27. Common production vector DBs?**
FAISS (library, embeddings only). Pinecone (managed). Weaviate (open-source, hybrid). Qdrant (Rust). Milvus (large-scale). pgvector (Postgres).

> **Saying it out loud.** They split into three shapes. FAISS is a library, not a database — enormously fast, no persistence, no filtering, you build the service around it. Pinecone is fully managed, so you trade money and data residency for never operating anything. Qdrant, Weaviate, and Milvus are self-hostable services with metadata filtering, hybrid search, and replication built in. And pgvector isn't really in the same category — it's an extension that lets your existing Postgres do vector search. The honest advice is that the index algorithm is largely commoditized and the real differentiators are filtered search performance, how updates and deletes are handled, and operational burden.

**28. When would you use pgvector vs Pinecone?**
pgvector: already use Postgres, want one DB, moderate scale. Pinecone: managed, fast scaling, don't want to operate. FAISS: in-process embeddings only, full control.

> **Saying it out loud.** If you're already running Postgres and you're under a few million vectors, pgvector is almost always the right call, because the biggest win in RAG is having your documents, metadata, permissions, and embeddings in one transactional store. No sync job, no two-systems-drift bug, and you can filter on ordinary SQL columns in the same query. You move to a dedicated vector database when scale or query volume outgrows that — roughly tens of millions of vectors, or when you need aggressive filtered search and per-tenant isolation. The tradeoff is operational: a second data store means a second consistency problem, and re-embedding a corpus across two systems is where most production RAG bugs actually live.

---

## F. Query handling

**29. What's HyDE?**
Hypothetical Document Embeddings. Have LLM generate a hypothetical answer to the query; embed that for retrieval. Captures query intent better than literal embedding. Improves retrieval on diverse queries.

> **Saying it out loud.** HyDE fixes a mismatch: questions and answers don't look alike, so embedding a short question and comparing it to long passages is comparing apples to oranges. So instead you ask an LLM to hallucinate a plausible answer — it doesn't have to be factually right — and you embed *that*, because a fake answer is stylistically and lexically much closer to the real passage than the question was. It helps most on short or jargon-light queries. The costs are the ones you'd guess: an extra LLM call in the critical path, so hundreds of milliseconds and real money per query, and if the model hallucinates in a wildly wrong direction you retrieve confidently irrelevant documents.

**30. What's query rewriting?**
Rewrite the user query into multiple variations; retrieve with each; union or rerank. Captures different angles. Useful when literal queries are short or under-specified.

> **Saying it out loud.** Query rewriting means you don't take the user's words at face value. You generate a few paraphrases with different vocabulary, run retrieval on each, and merge the results — usually with reciprocal rank fusion. The reason it works is that a single embedding is a single point in space, and one bad phrasing lands you in the wrong neighborhood with no recovery, whereas three phrasings give you three shots. It's also where you resolve conversational references, turning "what about the second one" into a standalone query. The cost is one extra LLM call plus N times the retrieval work, and the failure mode is drift — rewrites that quietly change what the user asked.

**31. What's query decomposition?**
Break complex queries into sub-queries; retrieve for each. Multi-hop or multi-faceted questions need this — single retrieval misses parts.

> **Saying it out loud.** Decomposition is for questions that no single document answers. "How does our refund policy compare to our competitor's?" needs two different documents, and one embedding of that whole sentence sits somewhere between them and retrieves neither well. So you have the LLM break it into sub-questions, retrieve for each independently, and assemble. Same for multi-hop, where you need the answer to the first part in order to even form the second query, which means retrieval becomes sequential rather than parallel. The tradeoff is latency and error compounding: every hop is another LLM call and another chance to go off the rails, so most production systems cap it at two or three.

**32. What's iterative / agentic RAG?**
Retrieve → answer partial → identify gaps → retrieve more. Used in FLARE, Self-RAG. The model controls the retrieval loop.

> **Saying it out loud.** Agentic RAG hands the retrieval loop to the model instead of running it once up front. The model starts answering, notices it's missing something, issues another retrieval, and continues — so retrieval becomes a tool call rather than a fixed preprocessing step. FLARE triggers a new lookup when the model's next-token confidence drops; Self-RAG trains special reflection tokens so the model explicitly decides whether to retrieve and whether what came back was useful. It's clearly better on multi-hop questions. The tradeoffs are that latency becomes unbounded and unpredictable, cost multiplies by the number of loops, and you need a hard iteration cap because models will happily search forever.

**33. Why does query rewriting matter?**
User queries are often short, ambiguous, or use different vocabulary than the docs. Rewriting bridges the gap. "OAuth impact?" → "OAuth session token security implications" → much better retrieval.

> **Saying it out loud.** Real user queries are terrible retrieval inputs. They're three words long, they use internal slang, they rely on the previous turn for context, and they use the vocabulary of someone who doesn't know the answer, while the documents use the vocabulary of someone who does. That vocabulary gap is the core problem — the user says "site is down," the runbook says "503 upstream connect failure." Rewriting bridges it by expanding the query into the language the corpus actually uses. Concretely, expanding a three-word query into a full specific sentence routinely moves recall at ten by double digits, and it's the cheapest quality win in the whole pipeline.

---

## G. Prompt construction

**34. How should you order retrieved chunks in the prompt?**
Place most relevant at start or end of context. "Lost in the middle" (Liu et al. 2023): LLMs underweight mid-context. Putting critical info at extremes mitigates.

> **Saying it out loud.** Order matters more than it should. The "lost in the middle" paper showed a U-shaped accuracy curve: models use information at the start and the end of a long context well and systematically underweight what's in the middle, and the drop can exceed twenty points on retrieval-style tasks. So you don't just dump the reranked list in rank order. The standard trick is to put your best chunk first, your second best last, and bury the weaker ones in the middle. It's a workaround for an attention pathology rather than a principle, but it's free and it measurably helps.

**35. What metadata should you include with chunks?**
Source document, date, author, section. Helps the LLM contextualize and cite. Without metadata, all chunks look equally authoritative.

> **Saying it out loud.** At minimum: source document, section heading, and date. Date is the one people forget and it's the one that causes real damage, because without it a chunk from a deprecated 2019 policy looks exactly as authoritative as this quarter's, and the model has no way to prefer the current one. Section headings matter because they restore context the chunker stripped away. And you need a stable identifier per chunk so the model can cite it and your UI can link back to the source. The general principle is that everything the model needs to judge or attribute a claim has to be inside the text you hand it — it can't see your database.

**36. Why request citations in the prompt?**
"Cite which chunk you used: [doc 3]". Helps detect hallucination, builds user trust, enables filtering or source-clicking in UI.

> **Saying it out loud.** Citations do three jobs. They let the user verify, which is the whole reason to use RAG over a bare model. They give you an automated faithfulness check, because you can programmatically test whether the cited chunk actually supports the sentence. And they change the model's behavior — being required to point at a source makes it noticeably less likely to invent one, since a claim with nothing to cite is visibly unsupported. The failure mode to name is citation hallucination: models will cite a chunk that doesn't contain the claim, so you have to verify the citations rather than trusting their presence as proof of grounding.

**37. How do you reduce hallucination in RAG generation?**
Strong instructions ("Use only provided sources; if not in sources, say so"). Cite sources. Lower temperature. Smaller, more focused chunks. Reranking quality. Calibrated refusal.

> **Saying it out loud.** It's layered, and no single layer is sufficient. Instruction level: tell the model explicitly to answer only from the provided context and to say it doesn't know otherwise — and give it explicit permission to refuse, because a model that feels obligated to answer will invent something. Structural level: require per-claim citations, which makes unsupported statements visible. Decoding level: low temperature, since sampling diversity is hallucination surface area. And upstream: better retrieval, because most "hallucination" is actually the model papering over context that didn't contain the answer. Then verify — run a faithfulness check that each claim is entailed by a cited chunk. The tradeoff worth naming is that hard grounding raises refusal rate, so you're trading recall for precision on purpose.

---

## H. Evaluation

**38. How do you evaluate retrieval?**
Recall@K (fraction of relevant docs in top-K). MRR (mean reciprocal rank of first relevant). NDCG (position-discounted, graded relevance). Need labeled relevance — hard at scale.

> **Saying it out loud.** Retrieval gets evaluated separately from generation, and the metric depends on what happens downstream. Recall at K is the one that matters most in RAG, because a document you didn't retrieve is unrecoverable no matter how good the model is — so measure recall at your reranker's input cutoff. MRR tells you how high the first relevant result lands, which matters when there's one right answer. NDCG handles graded relevance and position discounting, which is what you want when several documents are partially useful. The practical difficulty is labels: you need query-document relevance judgments, and generating them is the actual work.

**39. How do you evaluate end-to-end RAG?**
Faithfulness: does the answer match retrieved sources? Answer relevance: does it address the query? Context precision/recall: are retrieved chunks actually useful? Frameworks: RAGAS, Trulens.

> **Saying it out loud.** End-to-end evaluation has to decompose, or you can't act on the result. Context recall asks whether retrieval found what was needed. Context precision asks how much of what you retrieved was junk padding the prompt. Faithfulness asks whether every claim in the answer is supported by the retrieved text. And answer relevance asks whether it actually addressed the question. Frameworks like RAGAS compute these with an LLM judge. The caveat to state out loud is that LLM judges are noisy and biased toward verbose, confident answers, so you calibrate them against a few hundred human labels before you trust the numbers to make decisions.

**40. What's the difference between accuracy and faithfulness in RAG?**
Accuracy: is the final answer correct? Faithfulness: is it grounded in the retrieved context? An answer can be accurate (from model's parametric knowledge) but unfaithful (not grounded), which is essentially hallucination.

> **Saying it out loud.** These come apart in both directions, which is why you measure both. An answer can be correct but unfaithful — the model knew it from pretraining and the retrieved context didn't actually support it. That looks like a win on your accuracy metric and it's a landmine, because the same behavior on a question where the model's parametric knowledge is stale or wrong produces a confident error with a fake citation. And an answer can be faithful but wrong, if your source document is wrong; the RAG system did its job and the corpus failed. Faithfulness is the metric that's actually under your system's control, which is why it's the one to optimize.

**41. How do you label retrieval relevance at scale?**
Synthetic: generate queries from documents (LLM produces "what question would this answer?"). Human-annotated subsets for ground truth. Click data from production. Always validate synthetic against human.

> **Saying it out loud.** The scalable trick is to run it backwards: take a chunk, ask an LLM "what question does this answer?", and now you have a query with a known correct document, for free, at whatever volume you want. That gets you a few thousand evaluation pairs in an afternoon. Two things keep it honest. Synthetic queries are cleaner and better-phrased than real ones, so they systematically overstate your recall — validate against a few hundred human-labeled real queries to measure the gap. And once you're in production, click and thumbs data is the best signal you'll get, with the standard caveat that it's biased toward whatever you already ranked highly.

---

## I. Advanced patterns

**42. What's Self-RAG?**
Asai et al. 2023. Model decides when to retrieve, retrieves multiple times, verifies own outputs with reflection tokens. Combines retrieval with self-evaluation.

> **Saying it out loud.** Self-RAG trains the model to run its own retrieval loop using special tokens it emits during generation. One token decides whether retrieval is even needed for this query — a lot of questions don't need it, and retrieving anyway just injects noise. Others critique whether a retrieved passage is relevant and whether the generated sentence is actually supported by it. So the quality control is inside the model rather than bolted on around it. The cost is that it requires training with those reflection tokens, so you can't just prompt your way into it with an off-the-shelf API model, and inference is slower because you're generating and critiquing multiple candidates.

**43. What's GraphRAG?**
Microsoft 2024. Build a knowledge graph from documents at indexing. At query time, traverse the graph for richer cross-document context. Better for synthesis questions.

> **Saying it out loud.** GraphRAG exists because chunk-level retrieval can't answer global questions. Ask "what are the main themes across these ten thousand documents" and there is no top-five chunk that contains the answer — the answer is a property of the whole corpus. So at index time you extract entities and relationships into a knowledge graph, cluster it into communities, and pre-generate summaries at each level. Then a global question gets answered from community summaries rather than raw chunks. It genuinely beats vector RAG on synthesis and multi-hop. The tradeoff is indexing cost — you're running an LLM over your entire corpus to extract the graph, which can be orders of magnitude more expensive than embedding it, and the graph has to be rebuilt as documents change.

**44. What's CRAG (Corrective RAG)?**
After retrieval, verify quality. If retrieved docs are weak, expand search or fall back to non-RAG generation. Adds robustness.

> **Saying it out loud.** Corrective RAG adds a checkpoint that plain RAG is missing: before generating, judge whether what you retrieved is actually any good. A lightweight evaluator grades the retrieved set as correct, ambiguous, or wrong. If it's correct you proceed as normal, if it's ambiguous you refine, and if it's wrong you fall back — typically to a web search or to answering without retrieval while flagging lower confidence. The value is that it stops the specific failure where retrieval returns confident garbage and the model dutifully grounds an answer in it. The cost is an extra evaluation step per query, and the whole thing lives or dies on how well-calibrated that grader is.

**45. When does long-context beat RAG?**
Tasks requiring synthesis across many documents simultaneously. When the relevant context can fit in the model's window and is self-contained. Coding contexts where the entire repo is relevant.

> **Saying it out loud.** Long context wins when the task needs everything at once and chunking destroys the thing you need. Summarizing a whole contract, tracing a bug across a codebase, or answering "how did this argument evolve over the document" — those are global properties, and top-k retrieval returns fragments that individually look relevant and collectively miss the structure. It also wins when the corpus is simply small enough to fit, in which case retrieval is pure added risk for no benefit. RAG wins on scale, cost, and freshness: you can't fit ten million documents in any context window, and you're paying per token on every query. The honest answer is that the modern pattern is both — retrieve aggressively, then use a long window so you can afford to be generous about what you pass in.

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
