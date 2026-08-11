# From RAG to context engineering: where retrieval actually stands

> **Scope.** This chapter is the frontier layer. It assumes you already know chunking,
> embeddings, hybrid search, reranking, and RAG evaluation — those live in the sibling
> files in this folder:
> - `RAG_DEEP_DIVE.md` — the full pipeline, end to end
> - `chunking_strategies.md` — fixed, recursive, semantic, and document-aware chunking
> - `retrieval_methods.md` — BM25, dense, hybrid, fusion
> - `rag_architecture.md` — system layout and components
> - `rag_challenges_solutions.md` — the standard failure modes
>
> Here we cover what changed in 2025–2026: whether long context killed RAG (it did not,
> and the reason is interesting), what the modern ingestion pipeline looks like,
> structure-aware retrieval, multimodal late interaction, and the reframe of retrieval
> as "context engineering" for agents.

---

## 0. A note on the source, and on trust

This chapter started from a widely circulated piece: RAGFlow's
[*From RAG to Context — A 2025 Year-End Review of RAG*](https://ragflow.io/blog/rag-review-2025-from-rag-to-context)
(published 22 December 2025). It is a genuinely good map of the landscape and I have used
its structure.

It is also published by a vendor. RAGFlow is an
[open-source RAG engine](https://github.com/infiniflow/ragflow) (Apache-2.0, ~80k GitHub
stars) maintained by InfiniFlow, which also sells a hosted service. Several of the
article's named concepts — **DeepDoc** (its document parser), **TreeRAG** (its
hierarchical indexing feature), and the "**Context Engine / Context Platform**" framing —
are its own product vocabulary, not industry terms or peer-reviewed contributions. That
does not make them wrong. It does mean you should not cite them in an interview as if
they were established art.

Throughout I separate three tiers: **established** (peer-reviewed or widely reproduced —
stated plainly with a citation), **active research** (real papers, real numbers, not yet
settled practice — hedged), and **vendor positioning** (one company's framing or
self-reported benchmark — explicitly labelled).

A short list of things the source asserts that I **could not verify** appears at the
end of §8. Being willing to say "I couldn't check this" is the difference between a
technical chapter and a press release.

---

## 1. Is RAG dead? — answering the question honestly

### 1.1 The argument for "yes", stated at its strongest

Do not strawman this. The steelman goes:

1. **Context windows got huge.** 1M tokens is roughly 2,000 pages — bigger than most
   corporate knowledge bases anyone actually queries.
2. **Retrieval is a lossy bottleneck you inserted yourself.** Every chunking decision,
   every embedding, every top-$k$ cutoff throws information away *before* the model gets
   to reason. If the model reads everything, a whole class of bugs disappears.
3. **RAG is a lot of machinery.** Parsers, embedding models, a vector store, hybrid
   fusion, a reranker, a prompt assembler, an eval harness — each with its own failure
   modes and its own on-call pages.
4. **The frontier keeps eating scaffolding.** Chain-of-thought, tool-use scaffolds, and
   hand-built agent loops were all partially absorbed into models. Retrieval scaffolding
   might be next.
5. **Caching removes the cost objection.** Prompt caching makes the second and subsequent
   queries against the same 1M-token document set an order of magnitude cheaper.

Point 5 is the strongest and the one most people miss. Take it seriously.

### 1.2 What is actually true

**Cost and latency still scale with what you put in the window.**

Real, verifiable 2026 list prices:

| Model | Input $/M tok | Output $/M tok | Long-context premium |
|---|---|---|---|
| [Gemini 3.1 Pro](https://ai.google.dev/gemini-api/docs/pricing) (≤200K prompt) | $2.00 | $12.00 | — |
| Gemini 3.1 Pro (>200K prompt) | $4.00 | $18.00 | **2× input, 1.5× output** |
| Gemini 3.6 Flash | $1.50 | $7.50 | none listed |
| [Claude Opus 5](https://platform.claude.com/docs/en/about-claude/pricing) | $5.00 | $25.00 | none — flat to 1M |
| Claude Sonnet 5 | $2.00 | $10.00 | none — flat to 1M |
| Claude Haiku 4.5 | $1.00 | $5.00 | none — flat to 1M |

Note the divergence: Google charges a **2× input premium above 200K tokens**; Anthropic
explicitly does not, billing a 900K-token request at the same per-token rate as a 9K one.
So "does long context cost extra per token" already depends on your vendor. What does not
depend on your vendor is that you are paying for *every token you send, every time*.

Worked example. Corpus of 1M tokens (~2,000 pages). Assume 20 retrieved chunks of 600
tokens = 12K tokens of context for the RAG path.

| Approach | Input tokens/query | Cost/query (Opus 5) | Notes |
|---|---|---|---|
| Stuff whole corpus, no cache | 1,000,000 | **$5.00** | plus a large prefill latency |
| Whole corpus, 1-hour prompt cache | 1,000,000 (cached) | **$0.50** | + one-time $10.00 cache write (2× multiplier) |
| RAG, top-20 chunks | ~12,500 | **$0.0625** | + ~$0.00001 query embedding, + vector search |

That is **~80×** cheaper uncached and **~8×** cheaper even against a warm cache. The
source article's claim of "roughly a two-order-of-magnitude gap" is directionally right
for the uncached case and overstated for the cached case. Cite the arithmetic, not the
slogan.

Caching also has real constraints that get glossed over: it is prefix-based, so it only
helps if the big block sits at the *front* of the prompt and is byte-identical across
requests. Add one user-specific document at position 3 and you have invalidated
everything after it. Caches also expire (5-minute and 1-hour tiers are typical), so a
low-QPS workload pays the write premium repeatedly. Anthropic's own break-even note is
that a 5-minute cache pays for itself after one read — which is a *low* bar, but it
assumes you get that read inside five minutes.

Latency scales too. Attention cost is quadratic in sequence length ($O(n^2 d)$ for the
attention matmuls, though with flash-attention style kernels the wall-clock is dominated
by memory bandwidth and is closer to linear in practice for prefill). Either way, prefill
over 1M tokens is seconds to tens of seconds of time-to-first-token. A vector search over
the same corpus is single-digit milliseconds and the subsequent prefill is over 12K
tokens. For an interactive product, that difference is the product.

**Long context degrades in the middle.** The canonical result is
[*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172),
Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang — published in
[TACL 2024](https://aclanthology.org/2024.tacl-1.9/). The finding: performance is highest
when the relevant information sits at the **beginning or end** of the context and
"significantly degrades when models must access relevant information in the middle,"
producing a U-shaped position curve. Crucially, they found this held **even for models
explicitly built for extended contexts**.

Two caveats an interviewer will respect you for adding. First, the paper is from 2023 and
frontier models have improved on positional robustness since; treat it as "the effect is
real and recurs" rather than "the 2023 magnitudes still hold." Second, the *mechanism*
matters: the problem is not literally position, it is that adding irrelevant text adds
distractors. Which leads to the more useful result.

**More retrieved context is not monotonically better.**
[*In Defense of RAG in the Era of Long-Context Language Models*](https://arxiv.org/abs/2409.01666)
(Yu, Xu, Akkiraju; NVIDIA, Sept 2024) proposes *order-preserve RAG* (OP-RAG) — keep
retrieved chunks in their original document order rather than in relevance-rank order —
and shows answer quality follows an **inverted U** as you add chunks: it rises, peaks,
then falls. Their headline numbers on the EN.QA split of ∞Bench:

| Setup | Input tokens | F1 |
|---|---|---|
| **OP-RAG, Llama 3.1-70B** | **48K** | **47.25** |
| Llama 3.1-70B, full context, no RAG | 117K | 34.26 |
| GPT-4o, full context, no RAG | 117K | 32.36 |
| Gemini-1.5-Pro, full context, no RAG | 196K | 43.08 |

Better answers from ~40% of the tokens. This is the single most useful empirical citation
in the whole debate, and the order-preserving trick is a free win most people don't
implement.

**The counter-evidence, which you should also know.** Google's
[LOFT benchmark](https://arxiv.org/abs/2406.13121) (Lee et al., 2024, *Can Long-Context
Language Models Subsume Retrieval, RAG, SQL, and More?*) found long-context models
"rival state-of-the-art retrieval and RAG systems, despite never having been explicitly
trained for these tasks" at the 128K scale — while still failing on compositional
reasoning of the sort SQL requires, and degrading as you push toward 1M. So: at moderate
scale with a well-behaved corpus, long context genuinely is competitive on *quality*. The
argument for RAG at that scale is economics and operations, not accuracy. Say that
plainly; it is more credible than pretending retrieval always wins.

**Three things long context structurally cannot do.** These are not about quality.

1. **Freshness.** A cached context is a snapshot. If your corpus changes hourly — tickets,
   inventory, prices, incident state — you either rebuild the cache constantly (paying
   the write premium every time) or you serve stale answers. Retrieval reads current
   state at query time.
2. **Per-user access control.** If Alice may see 3 documents and Bob 3,000, a shared
   stuffed context is either a permissions leak or a per-user cache — and per-user caches
   destroy the cache economics that made long context affordable. Retrieval applies an
   ACL filter as a metadata predicate on the query. This is, in my experience, the single
   most common reason enterprises keep RAG regardless of context length.
3. **Auditability.** In regulated settings you must show *which source sentence* produced
   *which claim*. With retrieval you have a document ID and offset for every chunk in the
   prompt — provenance is a property of the architecture. With a stuffed context, the
   model's citation is itself a generated token and can be hallucinated. You can mitigate
   with post-hoc attribution, but you are now bolting on a retrieval-shaped verification
   step. You have reinvented RAG at the back end.

### 1.3 The actual answer

**They compose; they do not compete.** Retrieve to decide *what* enters the window;
use long context to hold *more* of it, and more coherently, than you could before.

Concretely, long context changes RAG's parameters rather than its existence:

- You can afford larger chunks and larger $k$, so the aggressive small-chunk regime that
  existed to fit 4K windows is obsolete. Retrieve at fine granularity, *return* coarse
  (see §4).
- You can pass whole parent documents instead of fragments, which kills a huge class of
  "correct chunk, missing context" failures.
- Multi-hop agents can accumulate several rounds of retrieved evidence in one window.
- You should still not fill the window. OP-RAG's inverted U says the optimum is well
  short of the limit.

The interview-ready sentence: *"Long context didn't kill RAG; it killed aggressive
chunking. Retrieval is now a filter that decides what deserves a place in a large window,
and the remaining hard reasons for retrieval — freshness, per-user ACLs, provenance, and
cost at scale — are architectural, not capability gaps that a bigger window closes."*

---

## 2. What RAG is actually for — the use cases

### 2.1 The alternatives table

Most "RAG problems" are really "wrong tool" problems. Before building anything:

| Approach | What it's for | What it costs | Failure mode if you choose it wrongly |
|---|---|---|---|
| **Plain prompt** | Knowledge the model already has; reasoning, formatting, style | Cheapest; nothing to build | Confident hallucination on anything proprietary or post-cutoff; no provenance |
| **Tool / API call** | Anything with a **precise, authoritative** answer: balances, inventory, "how many open tickets in EMEA", current price | Small eng cost; needs an API to exist | People use RAG over exported reports and get stale, approximate, un-aggregatable answers. **If a SQL query or an API answers it exactly, do not embed it.** |
| **Long context** | Small or bounded corpus; one-off deep analysis; document you already have in hand | High per-query token cost; latency; ACL and freshness problems | Cost blows up at volume; middle-of-context degradation; no per-user isolation |
| **RAG** | Large / private / changing corpora where you need grounding and citations | Ingestion pipeline, index, eval harness, ongoing ops | Over-applied to problems that were really aggregation or tool-use problems |
| **Fine-tuning** | Teaching *form*: output schema, tone, domain vocabulary, task-specific behaviour, latency/cost reduction via smaller models | Training data curation, training runs, versioning, re-do on base-model upgrade | Used to teach *facts*. Fine-tuning is a bad database: no updates without retraining, no citations, and it does not reliably suppress the pretrained prior |

The clearest heuristic I know: **fine-tuning changes behaviour, retrieval changes
knowledge, tools change ground truth.** If you find yourself fine-tuning to inject facts,
or embedding a table to compute a sum, stop.

These compose, and the strongest production systems use three of them at once: a
fine-tuned or prompted router, tool calls for anything numeric or transactional, and RAG
for the prose.

### 2.2 Where RAG is genuinely the right tool

- **Private / proprietary corpora.** The model has never seen your contracts, runbooks,
  design docs, or support history. There is no substitute.
- **Freshness.** Anything that changes faster than you can retrain or re-cache. Retrieval
  reads live state.
- **Per-user access control.** Retrieval-time filtering is the only clean way to give
  10,000 users different views of one corpus with one model deployment.
- **Citation and auditability.** Legal, medical, financial, and compliance workloads where
  "the model said so" is not an acceptable answer and every claim needs a source anchor.
- **Cost control at corpus scale.** Beyond a few million tokens, stuffing is not an option
  at any price; retrieval is the only way to get from a 10 GB corpus to a 12 KB prompt.
- **Corpora larger than any window.** 10M documents is not a context-window problem, it is
  an information-retrieval problem, and it always will be.

### 2.3 Where RAG is the wrong tool

- **Aggregation and analytics.** "What was total Q3 revenue by region?" Retrieval finds
  *some* chunks mentioning revenue. It cannot guarantee it found *all* of them, and the
  model cannot reliably sum them. Use SQL. (GraphRAG's global search partly addresses the
  corpus-sensemaking version of this — §4.3 — but is still not an arithmetic engine.)
- **A single well-known document**, which should just sit in the window, cached.
- **Behaviour and formatting problems.** "It doesn't follow our JSON schema" is not a
  retrieval problem.
- **Real-time transactional state.** Read the system of record.
- **Reasoning-only tasks.** Math, code transformation, planning — retrieval adds latency
  and distractors.

---

## 3. The modern ingestion pipeline

### 3.1 Parse → Transform → Index

The source article proposes **PTI (Parse–Transform–Index)** as the unstructured-data
analogue of **ETL/ELT** for structured data — the claim being that just as dbt/Fivetran/
Airbyte industrialised the structured pipeline, unstructured ingestion is standardising
into a named three-stage shape. *That framing is RAGFlow's, and I am attributing it as
theirs*, but it is a fair and useful analogy and I will use it.

| Stage | ETL/ELT (structured) | PTI (unstructured) |
|---|---|---|
| **Extract / Parse** | DBs, APIs, logs, CDC streams | PDFs, Office docs, HTML, scans, images; format parsers, OCR, layout models, VLMs |
| **Transform** | SQL/dbt: clean, join, aggregate, conform | **LLM-driven semantic enrichment**: summaries, entity/relation extraction, synthetic questions, keywords, structure inference |
| **Load / Index** | Warehouse tables, partitions, materialised views | Vector index + inverted index + metadata filters; optionally tree and graph indexes |

The analogy earns its keep because it imports a discipline the RAG world lacks:
versioning, idempotent re-runs, incremental updates, lineage, tests, and backfills. Most
RAG ingestion pipelines are a notebook that someone ran once. Treating ingestion as a
data-engineering artifact — with a schema, a run history, and the ability to reprocess a
subset when the parser improves — is the actual advice hiding inside the acronym.

### 3.2 Parsing is the underrated bottleneck

**Garbage in dominates.** This is the most robust practical claim in the whole chapter and
it is under-taught because it is unglamorous. Swapping embedding models moves retrieval
metrics by a few points. Fixing a parser that silently mangles two-column layouts,
drops table structure, or interleaves footnotes mid-sentence can move them by tens of
points, because the *text you embedded was never right*.

What actually goes wrong: multi-column PDFs read in the wrong order, splicing unrelated
sentences together; tables flattened to whitespace so a chunk reads `Revenue 412 388 401`
— confidently retrievable and semantically empty; headers, footers and page numbers
injected into every chunk, adding identical noise to every embedding and compressing the
distances between them; scanned documents with no text layer, yielding OCR garbage;
figures where the answer is in the image and the text says "see Figure 4"; and lost
section context, producing chunks like "This is prohibited under the above clause" with
no indication of which clause.

Diagnostic discipline: **read your chunks.** Sample 50 random chunks from your index and
read them as a human. If you cannot answer a question from a chunk, neither can the model.
This costs an hour and routinely finds the actual bug. Track a parse-quality metric
(tables preserved, reading-order accuracy, % chunks below a minimum information density)
as a first-class number, separate from retrieval metrics.

Tooling here is a genuine competitive area: layout-aware models, vision-language models
used as OCR, and commercial document-intelligence APIs. RAGFlow's **DeepDoc** is one such
parser and is genuinely open source in their repo; treat it as one option among several
rather than a category-defining technology, which is how the article positions it.

### 3.3 Transform is where value is created

This is the real shift of the last two years, and it is not vendor-specific. Historically
"transform" meant *chunk and embed*. Now it means: **spend LLM tokens at index time to
create information that did not exist in the raw document**, so that retrieval time is
cheap and accurate.

What that looks like in practice:

- **Contextual summaries per chunk.** Prepend a sentence or two on what the chunk is and
  where it sits ("From the 2024 10-K, Item 7 MD&A, discussing segment revenue…"). By far
  the highest-ROI enrichment: it fixes anaphora, disambiguates near-duplicate chunks
  across documents, and restores the document-level context flat chunking destroys.
- **Extracted entities and structured metadata.** Dates, parties, product names,
  jurisdictions, document type, version. These become **filter predicates**, worth more
  than better embeddings for a large class of queries — "the 2023 policy, not the 2021
  one" is a filter, not a semantic problem.
- **Synthetic questions.** Generate the questions each chunk answers and embed *those*
  alongside it. This attacks the query-document asymmetry — users write questions,
  documents contain statements. (HyDE attacks the same asymmetry from the query side at
  run time; doing it at index time costs more up front and nothing per query.)
- **Hierarchical summaries** as their own retrievable nodes (§4.1), and **VLM-generated
  descriptions** for every figure and table so visual content is reachable from a text
  index.

The trade is explicit: **you are moving cost from query time to index time.** For anything
with a read/write ratio above ~10:1 — which is nearly every knowledge base — that is
obviously correct. For a corpus that turns over daily and is queried rarely, it is not.
Compute your ratio before committing.

The honest caveat: LLM enrichment introduces LLM errors into your index, permanently and
invisibly. A hallucinated entity in metadata is a filter that silently excludes correct
documents. Sample and check enrichment output the way you would check a parser.

---

## 4. Structure-aware retrieval — beyond flat chunks

### 4.1 The problem flat chunking cannot solve

The core tension, which the source article names well: a **single chunk size cannot
simultaneously optimise recall and utility.**

- Small chunks embed cleanly. One idea, one vector, high precision, high recall — and then
  the model receives a fragment with no surrounding context and answers badly.
- Large chunks carry context but their embedding is an average of several topics, so they
  match everything weakly and nothing strongly.

The general fix decouples the two: **search at fine granularity, return at coarse
granularity.** Match against small, semantically pure units; then expand to the parent,
siblings, or neighbours before handing anything to the model. The naive version of this
(small-to-big / parent-document retrieval) is already standard and covered in
`chunking_strategies.md`. The 2025 development is doing the expansion over an
*explicitly built hierarchy* rather than raw adjacency.

### 4.2 Hierarchical / tree approaches

**RAPTOR** is the well-established academic version:
[*RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*](https://arxiv.org/abs/2401.18059),
Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning — **ICLR 2024**. The method: embed chunks,
cluster them, summarise each cluster with an LLM, embed the summaries, and recurse
bottom-up. You end up with a tree whose leaves are raw chunks and whose interior nodes are
progressively more abstract summaries. At query time you retrieve across *all* levels, so
a question needing a broad thematic answer can match a high-level summary node while a
detail question matches a leaf.

Headline result: coupling RAPTOR retrieval with GPT-4 improved the best performance on the
**QuALITY** benchmark by **20% in absolute accuracy**. That is a large, real, peer-reviewed
number on long-document QA.

**TreeRAG** is RAGFlow's implementation of this family — a document-derived hierarchy
(chapter → section → subsection → paragraph summary) built at ingest, with online search
against the finest nodes and automatic aggregation upward into "logically complete large
fragments." *This is a product feature, not a paper.* I could find no TreeRAG publication
and it is not described in the RAGFlow repo README. The idea is sound and is essentially
RAPTOR-with-document-structure; cite RAPTOR.

**PageIndex** takes the complementary route: instead of *inferring* a hierarchy by
clustering, use the document's *existing* one — the table of contents, heading levels,
section numbering — and have an LLM **navigate** that tree by reasoning, the way a human
flips to a chapter. It is [open source](https://github.com/VectifyAI/PageIndex) (MIT,
~35k stars) from VectifyAI, and markets itself as "vectorless" RAG.

Two honest notes. Its headline claim, **98.7% accuracy on FinanceBench**, is
**self-reported by the vendor** without an independent replication I could find, and the
baseline it is compared against ("traditional vector-based RAG") is not precisely
specified. Treat it as a promising direction, not a settled result. And structurally: the
approach inherits the source document's quality entirely. A well-structured 10-K works
beautifully; a scanned memo with no headings gives you nothing to navigate.

**Costs and when it is not worth it:**

| | Build cost | Staleness | Latency | Skip it when |
|---|---|---|---|---|
| **RAPTOR-style summary tree** | High — LLM summarisation over the whole corpus, recursively | Bad — editing one leaf invalidates every ancestor summary | Low at query time | Documents are short, self-contained, or heterogeneous (clustering unrelated docs produces meaningless summaries) |
| **PageIndex-style ToC tree** | Low — parse structure, summarise nodes | Good — local edits touch one node | **Higher** — LLM reasoning in the retrieval loop, multiple hops | Documents lack real structure; latency budget is tight |

RAPTOR's staleness problem is the one people under-plan for. If your corpus is
append-mostly (research papers, published reports), a summary tree is great. If it is
edit-heavy (a wiki, a spec repo), you are re-summarising constantly.

### 4.3 GraphRAG

[*From Local to Global: A Graph RAG Approach to Query-Focused Summarization*](https://arxiv.org/abs/2404.16130) —
Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt, Metropolitansky, Ness, Larson
(Microsoft Research, April 2024). Note it is an arXiv report, not a peer-reviewed
conference paper.

The method: at index time, an LLM extracts entities and relationships from every chunk,
builds a knowledge graph, detects communities in that graph (Leiden clustering), and
writes an LLM summary of each community at several hierarchy levels. At query time,
**local search** traverses from matched entities to their neighbourhoods; **global search**
map-reduces over community summaries to answer questions no single chunk contains — the
"what are the main themes in this corpus?" class.

**What it solves that flat chunking cannot.** Two things, genuinely:
1. **Multi-hop** questions where the connection between A and C is only visible through B,
   and A and C never co-occur in any chunk.
2. **Global sensemaking** — questions about the corpus as a whole, where the answer is
   nowhere in the text and must be synthesised across all of it. Flat top-$k$ retrieval
   is structurally incapable of this; it returns $k$ chunks and $k$ chunks is not the
   corpus.

The reported quality gains are real: against a naive-RAG baseline, comprehensiveness win
rates of **72–83%** (podcast dataset) and **72–80%** (news), diversity win rates of
**75–82%** and **62–71%** respectively, with $p<.01$ or better.

**What it costs — and this is the part that gets skipped.** From the paper's own tables,
on the podcast dataset, context tokens consumed at query time:

| Mode | Context tokens | % of max |
|---|---|---|
| C0 (root-level community summaries) | 26,657 | 2.6% |
| C1 | 225,756 | 22.2% |
| C2 | 565,720 | 55.8% |
| C3 | 746,100 | 73.5% |
| TS (all source texts) | 1,014,611 | 100% |

Global search at deeper community levels reads *hundreds of thousands to a million tokens
per query*. That is the same cost profile as stuffing the whole corpus into the context —
which is precisely the thing RAG was supposed to avoid. The economical mode is C0/C1, and
that is where you should start.

Index-time cost is worse. The paper reports graph indexing of the podcast dataset took
**281 minutes** with a 600-token extraction window on GPT-4-turbo. Entity extraction runs
an LLM over every chunk, often several times (extract, then "gleanings" passes to catch
misses), then deduplication, then community summarisation at every level. Multiples of the
corpus size in LLM tokens is the normal outcome.

**GraphRAG is over-recommended.** I will say that flatly. It became a 2024–25 buzzword and
gets proposed for problems that are ordinary single-hop lookups. Three further real
problems: **extraction noise** (duplicates like "Acme"/"Acme Corp."/"ACME Corporation",
wrong types, hallucinated relations — graph quality degrades quietly with no alarm);
**staleness** (adding documents can change community structure, invalidating summaries;
incremental update is possible but is real engineering); and **domain fit** — it works
where entities and relations are the semantic backbone (org charts, drug interactions,
supply chains, legal parties, incident causality) and poorly over narrative prose or
how-to documentation where the useful units are not entities.

**Use GraphRAG when** you have verified — with an eval set — that a meaningful share of
your queries are multi-hop or global, *and* your domain is entity-dense, *and* you can
afford the index. **Otherwise don't.** A well-tuned hybrid retriever with a good reranker
plus query decomposition handles a surprising fraction of "multi-hop" questions at a tiny
fraction of the cost, and you should prove that it doesn't before building a graph.

### 4.4 Combining them

Tree and graph structures are orthogonal and compose: the tree fixes *local* semantic
breaks caused by physical chunking, while the graph finds content that is semantically
related but *physically distant*, using traversal (e.g. Personalized PageRank) rather than
vector similarity. Systems that do both retrieve a tree-expanded fragment for each graph
node they land on. This is real and sensible, though at that point you are maintaining
three indexes (vector, tree, graph) and the operational burden is substantial.

---

## 5. Multimodal RAG and late interaction

### 5.1 Two paths

**Path 1 — modality conversion.** Run OCR or a vision-language model over images, charts,
and tables; produce text; index the text with your existing stack. Pros: works today,
compatible with everything, one index, cheap to search. Cons: lossy in exactly the way you
would expect — layout, spatial relationships, colour encoding in charts, and fine visual
detail are compressed into whatever the VLM chose to mention. If the caption doesn't
mention the outlier in the top-right of the scatter plot, the outlier is unretrievable.

**Path 2 — native multi-vector late interaction.** Skip text entirely. Encode the *page
image* into many vectors and match at that granularity.

### 5.2 What late interaction actually is

Standard dense retrieval is **single-vector / bi-encoder**: one vector per chunk, one per
query, score by cosine. Everything the chunk says is averaged into one point.

**Late interaction** (from ColBERT, extended to images by ColPali) keeps **one vector per
token or per image patch** and defers the interaction to scoring time. Given query vectors
$\{q_1,\dots,q_n\}$ and document vectors $\{d_1,\dots,d_m\}$, the score is **MaxSim**:

$$
S(Q, D) \;=\; \sum_{i=1}^{n} \max_{j=1..m} \; q_i^\top d_j
$$

In words: for each query token, find the single document token it matches best; sum those
best matches. Each query term gets to find its own evidence anywhere in the document,
instead of competing to influence one averaged vector.

Why it retrieves better: it preserves term-level and region-level signal. A query
mentioning three specific things can match three different regions of a page. It also
degrades gracefully on rare terms and proper nouns, which single-vector embeddings
notoriously smear. And it keeps the *offline-precomputable* property that cross-encoders
lack — document vectors are computed once at index time; only the cheap MaxSim runs at
query time. That is the whole trick: cross-encoder-like quality at bi-encoder-like serving
cost, paid for in storage.

**ColPali** —
[*ColPali: Efficient Document Retrieval with Vision Language Models*](https://arxiv.org/abs/2407.01449),
Faysse, Sibille, Wu, Omrani, Viaud, Hudelot, Colombo — **ICLR 2025**. It feeds a page
*image* to a VLM (PaliGemma), projects the patch representations to 128 dimensions each,
and scores with MaxSim. It also introduced **ViDoRe**, the visual document retrieval
benchmark. The claim, which held up in independent reimplementations, is that it "largely
outperforms modern document retrieval pipelines while being drastically simpler, faster
and end-to-end trainable" — notably, it needs no OCR, no layout detection, and no chunking
at all. For scanned and layout-heavy documents that is a step change.

### 5.3 The engineering wall: index size

This is where the article is straightforwardly correct, and the arithmetic is worth
internalising.

Per [Vespa's implementation notes](https://blog.vespa.ai/scaling-colpali-to-billions/),
ColPali emits **1,030 vectors per PDF page** — a $32 \times 32 = 1024$ patch grid plus
6 instruction tokens — each **128-dimensional**.

$$
\text{bytes/page (float32)} \;=\; 1030 \times 128 \times 4 \;=\; 527{,}360 \;\approx\; 515\ \text{KB}
$$

Compare a single-vector baseline: one 768-dim float32 vector per page is 3 KB. That is a
**~170× blowup**. Scaled up:

| Corpus | Single vector (768-d, fp32) | ColPali multi-vector (fp32) | ColPali (binary, 1 bit/dim) |
|---|---|---|---|
| 10K pages | 30 MB | ~5 GB | ~165 MB |
| 1M pages | 3 GB | **~515 GB** | ~16 GB |
| 100M pages | 300 GB | ~50 TB | ~1.6 TB |

At 1M pages you have crossed from "fits in RAM on one box" to "this is a storage
engineering project." Scoring cost rises too: MaxSim over 1,030 document vectors × ~20
query vectors is ~20,000 dot products *per candidate document*, so you cannot brute-force
a large collection and you need a two-phase pipeline (ANN retrieve on something cheap,
MaxSim rerank on the survivors).

### 5.4 Mitigations, with evidence

**Binary quantization.** Store each dimension as one bit and replace dot products with
Hamming distance. Vespa reports a **32× storage reduction** and **3.5× lower latency**,
with nDCG@5 falling only from **52.4** (float–float) to **51.6** (binary–binary with a
float rerank stage). That is a very good trade and it is available in production systems
today. It is the first thing to do.

**Token merging beats token pruning.**
[*Towards Storage-Efficient Visual Document Retrieval*](https://arxiv.org/html/2506.04997v1)
(Ma et al., NTU + Shanghai AI Lab, 2025) studies this empirically and the result is
counterintuitive and important. **Pruning** patch embeddings works badly — so badly that
"a simple random strategy outperforms other sophisticated pruning methods," because at
index time you do not know which patches a future query will need. **Merging** similar
patches by clustering works well: their Light-ColPali/ColQwen2 "maintains 98.2% of
retrieval performance with only 11.8% of original memory," and still holds **94.6%
effectiveness at 2.8% memory footprint.** Cost: ~72 A100-GPU-hours of fine-tuning to
recover the quality lost to merging.

Worth flagging: the source article lists "model-side token pruning" as "the most
fundamental method." The best empirical evidence I found points the other way — merging,
not pruning. That is a place where the vendor's roadmap and the literature disagree.

**MUVERA** —
[*MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings*](https://arxiv.org/abs/2405.19504),
Dhulipala, Hadian, Jayaram, Lee, Mirrokni (Google Research / DeepMind / UMD),
**NeurIPS 2024**. The idea: transform a multi-vector *set* into a **single fixed-dimensional
vector (FDE)** such that the plain dot product of two FDEs approximates the true Chamfer/
MaxSim similarity. It does this via space partitioning with locality-sensitive hashing —
so it is a principled approximation with guarantees, not just a projection heuristic. The
payoff is that you can then use ordinary single-vector ANN infrastructure for candidate
generation and only run exact MaxSim on the shortlist.

Reported results: **10% higher recall@k with 90% lower latency** than PLAID on average
across BEIR, up to **5.7× lower latency**; FDEs retrieve **2–5× fewer candidates** for the
same recall (95% recall from 75 candidates on MS MARCO with 5120-dim FDEs); and FDEs
compress **32×** with product quantization — a 10,240-dim FDE stored in 1,280 bytes with
negligible quality loss.

**Benchmarks.** [*M3Retrieve: Benchmarking Multimodal Retrieval for Medicine*](https://arxiv.org/abs/2510.06888)
(Acharya, Ghosh et al., **EMNLP 2025 main**,
[ACL Anthology](https://aclanthology.org/2025.emnlp-main.771/)) is real and is the largest
medical retrieval benchmark: **5 domains, 16 medical fields, 4 tasks, 1.2M text documents,
164K multimodal queries.** Its stated purpose is to expose how specialty-specific
challenges affect retrieval. *The specific finding the source article attributes to it —
that multimodal models excel on text-image tasks while mature single-modal RAG retains an
efficiency advantage on text-only tasks — I could not confirm from the paper's abstract or
landing page.* It is a plausible reading; verify against the PDF before repeating it.

### 5.5 Honest verdict on production readiness

| Situation | Verdict |
|---|---|
| Scanned / image-heavy documents, corpus under ~100K pages | **Ready.** ColPali-style retrieval with binary quantization is deployable now and is meaningfully better than OCR pipelines |
| Layout-heavy documents (forms, financial filings, slides), moderate scale | **Ready with care.** Budget for storage; use two-phase ANN → MaxSim |
| Millions of pages | **Not yet, for most teams.** Storage and serving are a dedicated project. Use OCR/VLM-to-text as the primary index and reserve late interaction for a reranking stage over a shortlist |
| Pure text corpora | **No.** Late interaction over text (ColBERT-style) is worth evaluating for quality, but the multimodal machinery buys nothing |

The pragmatic pattern for 2026 is **late interaction as a reranker, not as the primary
index**: cheap single-vector or BM25 retrieval to 100 candidates, then MaxSim rescoring.
You get most of the quality with none of the index blowup.

### 5.6 The theoretical backdrop: the DeepMind embedding-limits paper

The article gestures at a "Google DeepMind September 2025 article" arguing single global
vectors have inherent semantic loss. **This paper is real and it matters**, and it is worth
getting the details right because it is the strongest theoretical argument for
multi-vector retrieval that exists.

[*On the Theoretical Limitations of Embedding-Based Retrieval*](https://arxiv.org/abs/2508.21038) —
Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee (Google DeepMind), arXiv
**28 August 2025** (the coverage wave was September, which is presumably where the article's
date comes from). Code and data at
[github.com/google-deepmind/limit](https://github.com/google-deepmind/limit).

**The claim, precisely.** The number of distinct top-$k$ document subsets that a
$d$-dimensional single-vector embedding model can return, over any query, is *bounded by
$d$*. Their Theorem 1 gives, for embedding dimension $d$ to realise every $k$-subset of
$n$ documents with margin $\gamma$:

$$
d \;\ge\; \frac{\log \binom{n}{k}}{\log\!\left(1 + \tfrac{1}{\gamma}\right)}
$$

The result comes from sign-rank / communication-complexity arguments and holds **even
under free optimisation directly on the test set with unconstrained parameters**. It is
not a training or data problem. It is a representational ceiling.

**LIMIT**, the dataset they built to expose it, is deliberately trivial — queries like
"who likes Hawaiian pizza?" over 50K documents (a 46-document small version), 1,000
queries, $k=2$. State-of-the-art embedders fail badly on it despite the task's simplicity,
and performance scales with embedding dimension, exactly as the theory predicts.

**What survives it.** BM25 does well, because a sparse lexical model is effectively
extremely high-dimensional. Multi-vector models (they test gte-ModernColBERT) improve over
single-vector ones. Cross-encoders, which never compress a document to a fixed vector at
all, are outside the bound.

**How to state it without overclaiming.** This is *not* "embeddings don't work" — real
queries are not adversarially chosen to require arbitrary top-$k$ subsets, and dense
retrieval works fine on most workloads. It *is* a rigorous statement that there exist
simple, natural retrieval tasks no single-vector model of a given dimension can solve, and
therefore that hybrid retrieval (dense + lexical) and multi-vector/late-interaction
methods are not merely empirical hacks — they are addressing a proven representational
limit. That is a genuinely strong point to make in an interview, and it is one of the few
places where a theory result has immediate architectural consequences.

---

## 6. From RAG to context engineering

### 6.1 The reframe

The term **context engineering** is real and not a RAGFlow coinage — Anthropic published
[*Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
in September 2025, and the term circulated widely through late 2025. The core observation:
as systems became agentic, the interesting problem stopped being "write a good prompt" and
became "decide, dynamically, what should occupy the model's limited attention at this
step."

Framed that way, an agent needs **three kinds of context**, and only one of them is what
we have historically called RAG:

| Kind | What it is | Data character | Retrieval machinery |
|---|---|---|---|
| **Domain knowledge** | Documents, policies, code, tickets | Mostly static, curated | Classic RAG (§3–§5) |
| **Tool metadata** | Which tools exist, what they do, how they've been used | Semi-static, small-to-medium | Semantic + lexical search over tool descriptions |
| **Memory** | Conversation history, user preferences, agent state, learned procedures | **Dynamic, written by the system itself** | Same retrieval machinery, different write path |

The genuine architectural insight here — and I think it *is* genuine, independent of who
is selling it — is that these three are the **same problem**. Storage, indexing, ranking,
freshness, and eviction. Building three unrelated subsystems for them is a mistake teams
make routinely.

### 6.2 Tool selection is a retrieval problem

MCP (Model Context Protocol) standardises **how** to call a tool. It says nothing about
**which** tool to call. That is fine with 10 tools — put them all in the system prompt.
It breaks at 500.

Two things break at once. First, prompt bloat: tool schemas are verbose, and hundreds of
them consume the window you wanted for actual content. Second, and worse, **selection
accuracy degrades as the candidate set grows** — more near-duplicate tools means more
opportunities to pick the wrong one.

Evidence: [*RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented
Generation*](https://arxiv.org/abs/2505.03275) (Gan & Sun, May 2025) retrieves a small
relevant subset of tool descriptions before the model chooses, and reports tool-selection
accuracy of **43.13% vs 13.62%** for the all-tools-in-prompt baseline, with prompt tokens
cut by **over 50%**. Read that carefully: the absolute number is *43%*. Even with
retrieval, tool selection at scale is not solved. It is barely working. (This is a single
arXiv preprint, not a peer-reviewed result — treat the exact numbers as indicative.)

Practical notes that generalise: BM25 over tool names and descriptions is a strong
baseline, because tool queries are short and keyword-heavy — do this before anything
fancy. Log usage and retrieve on it ("which tools were used together for tasks like
this?" often beats description similarity). Retrieve a *toolset* (namespace/server) first,
then tools within it. Rewrite descriptions: they are retrieval documents, and most are
written as human API docs and are terrible retrieval targets. And store retrievable
**playbooks** — "for a refund request, first call X, then Y" — which converts a
closed-book reasoning problem into an open-book one. That is the same insight as ACE below.

### 6.3 Memory

Memory is retrieval over data the system generates about itself. The machinery is
identical; the *write path* is what's new — you must decide when to write, what to
summarise, when to consolidate, and what to forget. That last one has no good answer yet.

The cognitive-science taxonomy (working / episodic / semantic / procedural memory) is a
useful vocabulary and is widely borrowed, but be careful: it is an *analogy*, not an
architecture, and mapping it onto a system does not by itself tell you what to build.

The most substantive recent work here is
[*Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models*](https://arxiv.org/abs/2510.04618)
(Zhang, Hu, Upasani et al., October 2025). ACE treats the context itself as an evolving
artifact maintained by three roles — a **Generator** that produces trajectories, a
**Reflector** that extracts lessons from them, and a **Curator** that folds those lessons
into the context as structured incremental updates. It names two failure modes precisely,
and the names are worth stealing:

- **Brevity bias** — optimisation pressure toward concise instructions strips out the
  domain detail that actually made them work.
- **Context collapse** — iteratively rewriting a context with an LLM erodes detail over
  time, like repeated lossy re-encoding.

Reported gains: **+10.6%** on agent benchmarks and **+8.6%** on finance tasks over
baselines, with reduced adaptation latency and cost, using execution feedback rather than
labelled supervision. Again: October 2025 preprint, promising, not yet settled practice.

The related thread is prompt/context *optimisation*:
[**GEPA**](https://arxiv.org/abs/2507.19457) (*Genetic-Pareto*; Agrawal, Tan, Soylu et al.,
**ICLR 2026 oral**, integrated into DSPy) evolves prompts using natural-language reflection
on execution traces rather than scalar rewards, and reports beating GRPO by **6% on average
(up to 20%) with up to 35× fewer rollouts**, and MIPROv2 by **over 10%**. The source article
describes GEPA as "genetic-algorithm-based prompt evolution," which is half right — the
Pareto-frontier selection is evolutionary, but the paper's actual contribution is that
*reflection in natural language* is a much richer learning signal than a scalar reward.
For RAG specifically, GEPA-style optimisation is how you tune the query-rewriting and
answer-synthesis prompts without hand-fiddling.

### 6.4 Why agent retrieval is different from human search

This is a real and under-appreciated observation, and it is the part of the "Context
Engine" argument I find most convincing.

| | Human search | Agent retrieval |
|---|---|---|
| **Who queries** | A person, deliberately | An LLM, automatically, inside a loop |
| **Volume** | One query, then reading | Many queries per task — decomposition, verification, re-retrieval on failure |
| **Latency budget** | Seconds are fine | Every retrieval sits on the critical path of a multi-step loop; 500 ms × 10 steps is a 5-second stall |
| **Error recovery** | Human notices bad results, rephrases | Agent may not notice; bad context propagates silently into the next step |
| **Query style** | Keywords, sometimes vague | Often precise, sometimes bizarre — LLMs generate queries no human would write |
| **Consumer** | A human who filters and judges | A model that will treat retrieved text as true |

The consequences for design are concrete:

latency matters more than one more point of nDCG (a reranker that adds 400 ms is cheap in
a search box and expensive in a 15-step loop); precision matters more than recall, since a
human ignores result seven but an agent may build on it; retrieval must be cacheable,
because agents re-issue near-identical queries constantly; and evaluation gets much harder,
because offline metrics measure one query while agent performance depends on a sequence
where errors compound (see §8).

### 6.5 What's real and what's packaging

**Real architectural observations.** Domain knowledge, tool metadata and memory are the
same retrieval problem and benefit from shared infrastructure (indexing, ranking,
freshness, governance, access control). Tool selection at scale is an unsolved retrieval
problem that MCP does not address. Agent-driven retrieval has genuinely different
requirements from human search. And a unified serving layer that fuses, deduplicates and
formats results from several sources is worth building deliberately — most teams build it
accidentally and badly.

**Vendor packaging.** "Context Engine" and "Context Platform" as named product categories:
there is no standard, no benchmark, no shared definition, and the trajectory "Context
Engineering → Context Engine → Context Platform" is a marketing arc. The implicit claim
that you should buy this as one integrated platform: the pieces being related does not
imply they must ship together, and coupling document ingestion to agent memory is a real
architectural commitment. And the prediction that enterprises will build an "AI middle
platform" centred on a RAG engine — possible, and also exactly what a RAG-engine vendor
would predict.

Use the observations. Don't adopt the nouns.

---

## 7. What to do when — the decision guide

### 7.1 The order to try things

The single most common mistake: reaching for GraphRAG or agentic retrieval before fixing
chunking. Work down this list and stop when the numbers are good enough.

1. **Read 50 random chunks from your index.** Not a metric — actually read them. Most
   serious RAG bugs are visible here in ten minutes.
2. **Build an eval set.** 50–200 (query, relevant-document-ids, good-answer) triples.
   Without this everything below is guesswork. Yes, hand-label them.
3. **Measure retrieval and generation separately.** Recall@k and nDCG@k for retrieval;
   faithfulness and answer correctness for generation, *conditioned on correct retrieval*.
   You cannot fix what you cannot localise.
4. **Fix parsing.** Tables, reading order, headers, OCR. Highest ROI, least glamorous.
5. **Fix chunking.** Respect document structure. Add overlap. Prepend section headers to
   every chunk. See `chunking_strategies.md`.
6. **Add hybrid retrieval.** BM25 + dense with reciprocal rank fusion. Nearly always a
   win, especially on identifiers, error codes, product names, and rare terms — the exact
   cases the LIMIT paper predicts dense retrieval will miss.
7. **Add a reranker.** A cross-encoder over the top 50–100. Typically the largest single
   quality jump available for a fixed amount of work.
8. **Add metadata filters.** Date, type, source, version, permissions. Converts many
   "semantic" failures into exact filtering.
9. **Enrich at index time.** Contextual summaries per chunk; synthetic questions.
10. **Fix the generation prompt.** Order-preserve the chunks (OP-RAG). Reduce $k$ — the
    inverted U is real. Require citations.
11. **Query transformation.** Rewriting, decomposition for multi-hop, HyDE.
12. **Only now:** structure-aware retrieval (RAPTOR/PageIndex), GraphRAG, late-interaction
    multimodal, agentic retrieval loops.

Steps 1–8 fix the large majority of production RAG problems. Steps 9–12 are where the
conference talks live.

### 7.2 Symptom → likely cause → first fix

| Symptom | Likely cause | First fix |
|---|---|---|
| **Retrieval returns nothing relevant** | Parsing destroyed the text; or vocabulary mismatch between query and documents | Read the chunks. Then add BM25 hybrid — dense models fail on exact identifiers and rare terms |
| **Retrieves relevant chunks, answer still wrong** | Chunks lack surrounding context; or too many chunks so the answer is buried; or the prompt lets the model use its prior | Parent-document retrieval; reduce $k$; order-preserve; instruct "answer only from context, else say you don't know" |
| **Works on simple questions, fails on multi-hop** | Single-shot retrieval can't find the bridge entity | Query decomposition first (cheap). Iterative retrieve-read loop next. GraphRAG only if you've measured that a real share of queries need it |
| **Correct at chunk level, misses document-level context** | Flat chunking with no hierarchy | Prepend section/document summaries to chunks; then small-to-big; then a summary tree (RAPTOR) |
| **Right document, wrong version / wrong date** | No metadata filtering | Extract and index date, version, status at ingest; filter at query time. This is not a semantic problem |
| **Numbers and totals are wrong** | You are using retrieval for aggregation | Stop. Route to SQL or an API. Retrieval cannot guarantee completeness |
| **Good offline, bad in production** | Eval set doesn't match real queries; or a distribution the index doesn't cover; or ACL filters shrinking the candidate pool at runtime | Log production queries, sample the failures, rebuild the eval set from them. Check how many candidates survive ACL filtering |
| **Retrieval is slow** | Reranking too many candidates; or ANN parameters tuned for recall over latency; or embedding the query with a large model | Cut rerank candidates to 50; tune HNSW `ef_search`; cache query embeddings; use a smaller query encoder |
| **Costs too much** | Too many tokens per query; or over-enriching an index that is rarely read | Reduce $k$ (check the inverted U — quality may improve); use a cheaper model for synthesis with a strong reranker in front; prompt-cache stable prefixes |
| **Hallucinated citations** | Model generating source IDs rather than copying them | Post-hoc verification: check every cited ID exists in the passed context and that the claim is entailed by it. Cheap, catches a lot |
| **Quality degraded after adding documents** | Near-duplicates crowding the top-$k$; or index drift | Deduplicate at ingest; add diversity (MMR) to result selection |
| **Agent retrieves the wrong tool** | Too many tools in the prompt; poor descriptions | Retrieve tool subsets (RAG-MCP style); rewrite descriptions as retrieval documents |

### 7.3 Situation → architecture

| Situation | Recommended architecture |
|---|---|
| **Small static corpus (< ~200K tokens)** | Don't build RAG. Long context with prompt caching. Revisit if it grows |
| **Medium corpus (~1M–10M tokens), stable** | Standard hybrid RAG: parse well, structure-aware chunking, BM25 + dense, cross-encoder rerank, metadata filters |
| **Large corpus (10M+ documents)** | Hierarchical: route to a collection/namespace first, then retrieve within it. Sharded ANN index. Aggressive metadata filtering to shrink the candidate pool. Two-stage retrieve → rerank. Budget for index build and re-index cost as a first-class concern |
| **Frequently updated data** | Incremental ingestion with document-level versioning and tombstones. Avoid summary trees (expensive to invalidate). Prefer BM25-heavy hybrid — no re-embedding on edit. Consider retrieving from the system of record via API rather than a copy |
| **Multi-hop analytical questions** | Query decomposition + iterative retrieval first. GraphRAG only after measuring that a meaningful query share needs it and the domain is entity-dense |
| **Global "what are the themes" questions** | GraphRAG global search at low community levels (C0/C1), or map-reduce summarisation over the corpus. Flat top-$k$ cannot do this at all |
| **Scanned documents, tables, charts** | VLM/OCR-to-text as the primary index for scale, plus late-interaction (ColPali-style) reranking over a shortlist. Extract tables into structured form and query them as data, not prose |
| **Per-user permissions** | ACLs as indexed metadata, applied as a pre-filter (not post-filter — post-filtering silently shrinks your top-$k$). Verify the filter is enforced in the vector store, not in application code after the fact. Test with a user who can see almost nothing |
| **Strict citation requirements** | Small chunks with stable IDs and character offsets; pass IDs into the prompt; require inline citations; verify every cited ID post-hoc against the context. Never rely on the model to reconstruct a source |
| **Agent with hundreds of tools** | Tool retrieval layer: BM25 + embeddings over descriptions, hierarchical namespace-then-tool selection, usage-history signals |
| **Multilingual corpus** | Multilingual embedding model; keep BM25 per-language with correct analysers; test cross-lingual retrieval explicitly — it is usually much worse than the model card implies |

---

## 8. What's next

### 8.1 Nearly certain

- **Ingestion quality becomes the differentiator.** Parsing, layout understanding, and
  table extraction determine retrieval quality more than embedding choice does, and the
  field has internalised this. Expect parsers to be benchmarked the way retrievers are.
- **Index-time LLM enrichment becomes standard.** Contextual chunk summaries, extracted
  metadata, synthetic questions. The read/write economics are overwhelming for typical
  knowledge bases, and the technique is simple.
- **Structure-aware retrieval becomes default rather than exotic.** Some form of hierarchy
  — parent documents, section trees, ToC navigation — in most serious systems.
- **Hybrid retrieval is non-negotiable.** The LIMIT result gave a theoretical spine to what
  practitioners already knew: pure dense retrieval has a representational ceiling, and
  lexical matching covers cases it provably cannot.
- **Retrieval serves agents, not humans.** Latency budgets, precision-over-recall, and
  retrieval-inside-a-loop become the default design assumptions.

### 8.2 Plausible

- **Late-interaction multimodal goes mainstream — if index size is solved.** The quality
  case is made; the storage case is not. Binary quantization (32×, ~1.5% nDCG loss) plus
  token merging (98.2% quality at 11.8% memory) plus MUVERA-style FDEs are collectively
  close to closing the gap. If those land in mainstream vector databases as first-class
  features rather than research code, this flips within a year or two.
- **Native tensor/multi-vector support in vector databases** as a standard feature rather
  than a Vespa/Qdrant specialty, alongside **quantization-aware multimodal encoders**
  trained so that binarising their outputs costs almost nothing.
- **Memory systems consolidating on a shared design.** Right now every agent framework has
  its own incompatible memory abstraction. Something will win.

### 8.3 Genuinely unsolved

- **Tool selection at scale.** 43% accuracy with retrieval assistance is not a solved
  problem, it is a warning. Nobody has a good answer for thousands of tools, and MCP's
  success makes the problem worse by making tools easy to add.
- **Evaluating retrieval inside agent loops.** Offline nDCG measures a single query. Agent
  outcomes depend on a *sequence* of retrievals with compounding errors, where a bad
  retrieval at step 3 may only manifest as a wrong answer at step 11. Credit assignment
  across the loop is an open problem, and it is the thing most likely to be silently
  wrong in your system.
- **Memory consolidation and forgetting.** When to summarise, what to discard, how to
  avoid ACE's "context collapse," how to keep a memory store from degenerating into noise
  after 10,000 interactions. There are proposals; there is no consensus and no benchmark.
- **Retrieval quality on generated content.** As corpora fill with LLM-written documents,
  retrieval over them has feedback dynamics nobody has characterised.
- **Multi-vector at web scale.** 50 TB for 100M pages is not a solved storage problem.

### 8.4 Where the source article's predictions are self-serving

Flagging these is not an accusation of bad faith; it is normal reading hygiene.

- "**RAG becomes the foundational data layer / AI middle platform**" — predicted by a
  company selling a RAG engine. It may well be right. Note the alignment.
- "**Context Platform**" as the natural next category — the category is defined such that
  the author's product is in it.
- "**This evolutionary trend is now irreversible**" — no technology trend is, and claims
  of irreversibility are a rhetorical move, not an analysis.
- Emphasis on **document parsing depth** as the key differentiator — true and also
  precisely where DeepDoc is the product.
- **Model-side token pruning** as "the most fundamental" mitigation for index size — the
  strongest empirical study I found concludes that pruning underperforms *merging*, and
  that random pruning beats sophisticated pruning.

### 8.5 Claims from the source article I could NOT verify

Stated plainly, because this is the part with the most value:

1. **"Roughly a two-order-of-magnitude cost gap" between long context and RAG.** No source
   given. My own arithmetic (§1.2) gives ~80× uncached and ~8× against a warm cache. The
   figure is defensible for the uncached case and overstated for the cached one.
2. **"KV-cache-based approaches cost at least an order of magnitude more than RAG."** No
   citation; plausible but unquantified in the article and I found no study establishing it.
3. **The M3Retrieve finding** that multimodal models excel on text-image tasks while
   single-modal RAG retains an efficiency advantage on text-only tasks. The benchmark is
   real (EMNLP 2025); this specific conclusion I could not confirm from the paper's
   abstract or landing page.
4. **TreeRAG** as a described architecture. No paper, no independent evaluation, not
   mentioned in the RAGFlow repository README. RAPTOR (ICLR 2024) is the citable version
   of the same idea.
5. **PageIndex's 98.7% FinanceBench accuracy.** Vendor self-reported, no independent
   replication found, baseline unspecified.
6. **"Agent retrieval request frequency is one to two orders of magnitude higher than
   traditional search."** Directionally obviously true; the specific magnitude is
   unsourced.
7. **A "late interaction workshop in early 2026."** Could not confirm such an event exists.
8. **The DeepMind paper's date as "September 2025."** The arXiv preprint is 28 August 2025;
   the media coverage was September. Minor, but if you cite it, cite August.
9. **GEPA described as "genetic-algorithm-based prompt evolution."** Partially right —
   GEPA is *Genetic-Pareto* — but the paper's central claim is about reflective
   natural-language feedback outperforming scalar RL rewards, not about genetic search.
10. **"Context Engine" / "Context Platform"** as recognised industry categories. Context
    *engineering* is a real, widely used term (Anthropic, September 2025). The productised
    nouns are RAGFlow's.

---

## 9. Interview angle

**1. "Is RAG dead now that we have million-token context windows?"**
No, but the reasons have changed. Long context is genuinely competitive on *quality* at
moderate scale — Google's LOFT benchmark showed long-context models rivalling dedicated
retrieval pipelines at 128K. The remaining case for RAG is architectural, not capability:
cost and latency scale with every token you send (roughly 80× cheaper to retrieve 12K
tokens than to stuff 1M, ~8× even against a warm prompt cache); caches are prefix-based so
they break under per-user or freshly-updated content; per-user access control needs
query-time filtering; and auditability requires that provenance be a property of the
architecture rather than a generated token. What long context *did* kill is aggressive
small-chunking. The modern shape is retrieve-then-fill: retrieval decides what deserves
the window, long context holds more of it, more coherently.

**2. "When would you not use RAG?"**
When something else answers the question exactly. Aggregation and analytics ("total Q3
revenue by region") should be SQL — retrieval cannot guarantee it found every relevant
chunk and the model cannot reliably sum them. Transactional state should be an API call.
Behaviour and format problems ("it won't emit our JSON schema") are fine-tuning or
prompting problems. A single document you always use in full should just be in the
context, cached. The rule I use: fine-tuning changes behaviour, retrieval changes
knowledge, tools change ground truth. Injecting facts by fine-tuning is the classic
expensive mistake — no updates without retraining, and no citations.

**3. "How would you handle a 10-million-document corpus?"**
Hierarchical routing before retrieval: classify the query into a collection or namespace,
then retrieve within it, so ANN search never touches the whole space. Aggressive metadata
pre-filtering (date, type, source, permissions) to shrink the candidate pool — and
pre-filter inside the vector store, not post-filter in application code, or you silently
lose your top-$k$. Two-stage retrieval: cheap hybrid recall to ~100 candidates, then a
cross-encoder rerank to ~10–20. Sharded index with replicas sized for QPS. Then the
operational half that people forget: incremental ingestion with document versioning and
tombstones, a re-index strategy that doesn't require full rebuild when the parser improves,
and monitoring on candidate-pool size after filtering. At that scale the index build and
update pipeline is more of the engineering than the retrieval.

**4. "How do you evaluate retrieval separately from generation?"**
Two independent harnesses. For retrieval: a labelled set of (query → relevant document
IDs), scored with recall@k, nDCG@k, and MRR. This needs no LLM and runs in seconds, so you
can iterate on chunking and ranking quickly. For generation: run the generator against
*known-correct* retrieved context and measure faithfulness (is every claim entailed by the
context?) and answer correctness. That conditioning is the whole point — it localises the
failure. If retrieval recall is 60% and generation-given-good-context is 95%, you have a
retrieval problem and tuning prompts is wasted effort. Then measure end-to-end for the
number you actually ship on. In production, log retrieved IDs with every response so you
can replay failures, and rebuild your eval set from real failing queries rather than
imagined ones.

**5. "What is late interaction and why does it help?"**
A bi-encoder compresses a whole document into one vector, so everything it says is
averaged into one point. Late interaction keeps one vector per token (ColBERT) or per image
patch (ColPali) and defers scoring to query time via MaxSim: for each query vector, take
its maximum dot product against any document vector, and sum those. Each query term finds
its own best evidence anywhere in the document instead of competing to influence one
average. It helps most on rare terms, proper nouns, and multi-aspect queries, and it
preserves the crucial property that document vectors are computed offline — so you get
close to cross-encoder quality at bi-encoder serving cost. The bill comes as storage:
ColPali emits 1,030 vectors × 128 dims per page, about 515 KB in fp32, versus ~3 KB for a
single vector. That's ~515 GB per million pages.

**6. "So how would you make late interaction affordable?"**
Three layers. Binary quantization first — Vespa reports 32× storage reduction and 3.5×
lower latency with nDCG@5 going only from 52.4 to 51.6 with a float rerank stage. Then
token *merging*, not pruning: the empirical work shows pruning is bad enough that random
pruning beats sophisticated pruning, because at index time you don't know which patches a
future query needs, whereas clustering-based merging holds 98.2% of quality at 11.8% of
memory. Then MUVERA-style fixed-dimensional encodings, which map a multi-vector set to a
single vector whose dot product approximates Chamfer similarity, so you can use ordinary
ANN infrastructure for candidate generation. Honestly, though: for most teams in 2026 the
right answer is late interaction as a *reranker* over a shortlist, not as the primary
index.

**7. "What's the theoretical argument that single-vector embeddings are insufficient?"**
Weller, Boratko, Naim and Lee at Google DeepMind, *On the Theoretical Limitations of
Embedding-Based Retrieval* (August 2025). The number of distinct top-$k$ document subsets
a $d$-dimensional embedding model can return is bounded by $d$ — via sign-rank arguments,
and it holds even when you optimise directly on the test set with unlimited parameters. So
it's a representational ceiling, not a training problem. They built LIMIT, a deliberately
trivial dataset ("who likes Hawaiian pizza?", 50K docs, k=2), and SOTA embedders fail on
it, with performance scaling in embedding dimension exactly as predicted. BM25 does well
because sparse lexical retrieval is effectively very high-dimensional; multi-vector models
do better than single-vector. The practical upshot: hybrid retrieval and late interaction
aren't empirical hacks, they address a proven limit. The upshot to *not* overclaim: real
queries aren't adversarial, and dense retrieval works fine on most workloads.

**8. "When is GraphRAG worth it?"**
Rarely, and less often than it's proposed. It genuinely solves two things flat retrieval
cannot: multi-hop questions where A and C connect only through B and never co-occur in a
chunk, and global sensemaking where the answer isn't in any chunk. Microsoft's paper
reports 72–83% comprehensiveness win rates over naive RAG, which is real. But global search
at deeper community levels consumes 200K–1M context tokens per query — the same profile as
stuffing the corpus, which is what you were avoiding — and indexing the podcast dataset
took 281 minutes on GPT-4-turbo, since you run an LLM over every chunk for extraction plus
deduplication plus community summarisation. Add extraction noise ("Acme" vs "Acme Corp.")
and staleness, since adding documents can invalidate community summaries. I'd require
three things first: an eval set proving a meaningful share of queries are multi-hop or
global, an entity-dense domain, and evidence that query decomposition plus a good reranker
doesn't already handle it.

**9. "Your RAG system works in testing but users say it's bad. Walk me through debugging."**
First, get the real queries — the offline set almost never matches production. Sample 50
failures and classify them, because "it's bad" is at least six different bugs. Then
localise: for each failure, was the right document in the top-$k$? If no, it's retrieval;
if yes, it's generation. For retrieval failures, read the chunks — parsing damage and lost
section context account for a startling share. Check whether ACL filtering is shrinking
the candidate pool before ranking. Check whether the failing queries contain identifiers or
rare terms that dense retrieval misses, which points at adding BM25. For generation
failures, check $k$ — the OP-RAG result shows quality follows an inverted U, so more
context can be actively worse — and check chunk ordering, since order-preserving beats
relevance-ordering. Only then start changing models.

**10. "Where do you spend index-time compute for the biggest retrieval win?"**
Contextual chunk summaries. Prepending one or two sentences describing what the chunk is
and where it sits in its parent document fixes anaphora, disambiguates near-duplicate
chunks across documents, and restores the document-level context that flat chunking
destroys — all in one cheap enrichment. Second is structured metadata extraction, because
"the 2023 policy, not the 2021 one" is a filter predicate, not a semantic problem, and
filters are more reliable than similarity. Third is synthetic questions embedded alongside
chunks, which attacks the query-document asymmetry at index time rather than paying HyDE's
cost on every query. All three trade index-time cost for query-time quality, which is
correct whenever your read/write ratio exceeds roughly 10:1 — so, nearly always. The
caveat is that LLM enrichment bakes LLM errors permanently into your index, so sample and
verify it like you'd verify a parser.

**11. "How does retrieval for agents differ from retrieval for humans?"**
Volume, latency, and error propagation. An agent issues many queries per task —
decomposition, verification, retry — so QPS is far higher and each retrieval sits on the
critical path of a loop where 500 ms × 10 steps is a five-second stall. Precision matters
more than recall, inverting the usual search trade-off: a human ignores result seven, but
an agent may build on it, so false positives are actively harmful. And errors compound
invisibly — a bad retrieval at step 3 can surface as a wrong answer at step 11, which
makes evaluation genuinely hard, because offline nDCG measures one query and agent outcomes
depend on a sequence. Design consequences: cache aggressively since agents re-issue similar
queries, prefer a fast reranker over a marginally better slow one, and log the full
retrieval trace per task, not per query.

**12. "You have 800 tools registered via MCP. How does the agent pick one?"**
It's a retrieval problem, and MCP explicitly doesn't solve it — it standardises how to call
a tool, not which. Putting 800 schemas in the prompt blows the window and, worse, degrades
selection accuracy because near-duplicate tools multiply. So retrieve a small candidate set
before the model chooses. Start with BM25 over tool names and descriptions — tool queries
are short and keyword-heavy, and lexical matching is a strong baseline. Add hierarchy:
pick a namespace or server first, then a tool within it. Add usage history as a signal —
"which tools were used together for tasks like this" often beats description similarity.
And rewrite the descriptions, because they're retrieval documents and most are written as
human API docs. The honest caveat is that the published numbers here are bad: RAG-MCP
reports 43% selection accuracy with retrieval versus 14% without. It's better, and it's
still not a solved problem.

---

## Further reading

**Long context vs retrieval**
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024 — https://aclanthology.org/2024.tacl-1.9/ (arXiv: https://arxiv.org/abs/2307.03172)
- Yu, Xu, Akkiraju, *In Defense of RAG in the Era of Long-Context Language Models* (OP-RAG) — https://arxiv.org/abs/2409.01666
- Lee et al., *Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?* (LOFT) — https://arxiv.org/abs/2406.13121

**Theory of retrieval**
- Weller, Boratko, Naim, Lee, *On the Theoretical Limitations of Embedding-Based Retrieval* — https://arxiv.org/abs/2508.21038
- LIMIT dataset and code — https://github.com/google-deepmind/limit

**Structure-aware retrieval**
- Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*, ICLR 2024 — https://arxiv.org/abs/2401.18059 (code: https://github.com/parthsarthi03/raptor)
- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — https://arxiv.org/abs/2404.16130
- PageIndex (VectifyAI) — https://github.com/VectifyAI/PageIndex

**Multimodal and late interaction**
- Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models*, ICLR 2025 — https://arxiv.org/abs/2407.01449
- Dhulipala et al., *MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings*, NeurIPS 2024 — https://arxiv.org/abs/2405.19504
- Ma et al., *Towards Storage-Efficient Visual Document Retrieval* (token merging vs pruning) — https://arxiv.org/abs/2506.04997
- Acharya, Ghosh et al., *M3Retrieve: Benchmarking Multimodal Retrieval for Medicine*, EMNLP 2025 — https://aclanthology.org/2025.emnlp-main.771/
- Vespa, *Scaling ColPali to billions of PDFs* (concrete storage/latency engineering) — https://blog.vespa.ai/scaling-colpali-to-billions/

**Context engineering, memory, and optimisation**
- Anthropic, *Effective context engineering for AI agents* — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Zhang, Hu, Upasani et al., *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models* — https://arxiv.org/abs/2510.04618
- Agrawal et al., *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*, ICLR 2026 — https://arxiv.org/abs/2507.19457 (code: https://github.com/gepa-ai/gepa)
- Gan & Sun, *RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection* — https://arxiv.org/abs/2505.03275

**Pricing references used in §1**
- Gemini API pricing — https://ai.google.dev/gemini-api/docs/pricing
- Claude API pricing — https://platform.claude.com/docs/en/about-claude/pricing

**The source article (read critically)**
- RAGFlow, *From RAG to Context — A 2025 Year-End Review of RAG* — https://ragflow.io/blog/rag-review-2025-from-rag-to-context
- RAGFlow source, for what DeepDoc actually is — https://github.com/infiniflow/ragflow

---

*Fundamentals are in `RAG_DEEP_DIVE.md`, `chunking_strategies.md`, `retrieval_methods.md`,
`rag_architecture.md`, and `rag_challenges_solutions.md`. Prices and model names in §1
were verified in August 2026 and will drift; the ratios will outlast the numbers.*
