# Latency engineering for production RAG

There is a version of the RAG interview that is about quality — chunking, hybrid search, reranking, why
your answers are wrong. This chapter is about the other one, which is about time. It gets asked because
it is the question that most cleanly separates people who have shipped a RAG system from people who
have built one, and the tell is immediate: ask someone how they would make a RAG system faster and the
weak answer starts naming components ("switch to a faster vector DB, use a smaller embedding model")
while the strong answer starts by refusing to answer until it has profiled something.

The frame worth adopting, and worth saying out loud in an interview, is that the goal is not

> "make every component as fast as possible"

but

> "meet the product's latency SLO while preserving answer quality and controlling cost."

Those are different objectives and the second one is the engineering job. It implies you know the SLO,
which implies you asked what the product is; it implies you know where your time goes, which implies you
instrumented; and it implies you are tracking what the speedups cost you, which is the part almost
everyone omits.

---

## 1. Where the time actually goes

### The warning that has to come first

Search for "RAG latency breakdown" and you will find a dozen pages with confident millisecond tables.
Most of them are fabricated. While researching this chapter I fetched several — one publishes a full
stack decomposition ("network transit 20–80 ms, tokenization 1–5 ms, model inference 5–15 ms") with no
hardware, no region, no concurrency, no trial count, and no date; another surfaced in search results
with a headline benchmark figure that does not appear anywhere on the actual page; a third was a 404 on
a staffing site. The numbers are not implausible, which is exactly the problem — they are not
measurements, and you cannot tell whether they describe your concurrency regime, your region, or
anything at all.

So the numbers below are limited to sources that publish a measurement setup, and each is marked. Where
no credible figure exists I say so rather than filling the gap, because "nobody publishes this, you have
to measure it yourself" is itself a useful thing to know and a strong thing to say in an interview.

### The four scaling classes

The useful decomposition is not "which stage is slow" but **which stages are fixed, which scale with
corpus size, which scale with candidate count, and which scale with context length.** These behave
completely differently under growth, and a budget that does not separate them misleads you the moment
anything grows.

**Query rewriting** is a model call and should be budgeted as one. People assume it is cheap because the
prompt is short, but the cost is dominated by *decode*, not prefill, so it is proportional to how many
tokens the rewriter emits. At the 200–500 ms TTFT and 10–50 ms per output token that ClickHouse gives as
well-tuned operating ranges [vendor-published], a rewriter emitting 30 tokens costs roughly 0.5–1.9
seconds — very often more than the entire retrieval stack that follows it. Scaling class: **fixed with
respect to corpus and k, linear in the rewriter's output length.** Constrain the output format (a JSON
array of three strings, not prose) and run it concurrently with a first-pass retrieval of the raw query.

**Query embedding** has two regimes an order of magnitude apart, and conflating them is the most common
budgeting error in the pipeline. The *compute* is one forward pass over a few dozen tokens — genuinely
small. The *hosted API call* is DNS, TLS, pool wait, serialization, provider-side queueing, and the round
trip. Milvus benchmarked 20+ embedding APIs and found their own database contributed under 5% of total
operation time (20–40 ms) while embedding API calls took "hundreds of milliseconds to several seconds"
[operator blog]. Their sharpest finding: a client in Asia calling a North American endpoint saw 3–4×
higher latency, and for one provider nearly 100×. **The dominant term in hosted embedding latency is
geography, not model size.** Scaling class: **fixed** — a constant tax, removable by co-locating or
self-hosting the encoder.

**ANN search** is the stage everyone budgets for and almost nobody is bottlenecked by. Weaviate's
benchmark is one of the few that publishes hardware, dataset, HNSW parameters, recall, and percentiles
together [vendor-published]:

| Dataset | Vectors × dim | ef | Recall@10 | Mean | p99 |
|---|---|---|---|---|---|
| SIFT1M | 1M × 128 | 64 | 98.35% | 1.44 ms | 3.13 ms |
| DBPedia OpenAI | 1M × 1536 | 96 | 97.24% | 2.80 ms | 4.43 ms |
| MSMARCO | 8.8M × 768 | 48 | 97.36% | 2.15 ms | 3.69 ms |
| Sphere DPR | 10M × 768 | 96 | 96.06% | 4.49 ms | 7.73 ms |

Read rows three and four against each other: both around 9–10M vectors at 768 dimensions, differing 2.1×
in mean latency purely because one runs at ef=48 and the other at ef=96. Meanwhile going from 1M to 8.8M
vectors *decreased* mean latency, because the dimension dropped from 1536 to 768. **Dimension and
ef_search dominate; corpus size is close to irrelevant over this range**, which is what you expect from
HNSW's roughly logarithmic traversal. Scaling class: **logarithmic in corpus, linear in ef_search,
linear in dimension.**

Two important asterisks. These are warm, single-node, RAM-resident numbers; disk-backed and serverless
vector search behaves completely differently, and that difference is the actual reason people see slow
retrieval. And **filtered** search is the exception — a selective filter can force the graph traversal
far past ef to find k surviving candidates, which is the most common cause of pathological retrieval
latency in production. No vendor publishes a filtered-search latency curve, which is a real gap.

**Reranking** is the stage whose cost model people get structurally wrong, and the one that actually
scales. A bi-encoder embeds the query once and compares against precomputed document vectors — the
document side is amortized forever. A cross-encoder amortizes nothing: it concatenates (query, document)
and runs a full forward pass, once per pair. Reranking 100 candidates is 100 forward passes, and no
batching trick changes that; batching improves utilization, not work performed. Decagon, which operates
latency-sensitive customer service agents, states plainly that "aside from final response generation,
reranking typically incurs the largest latency cost" [operator blog]. Scaling class: **linear in
candidate count, superlinear in chunk length** — the one retrieval stage where turning a knob buys a
proportional, predictable reduction.

Worth knowing for the interview: **no hosted reranking vendor publishes absolute latency.** Cohere
publishes only capacity limits; Voyage publishes only ratios against LLM baselines. The most
consequential stage in retrieval has the emptiest public record, and the honest answer to "how long does
reranking take" is "I would measure it against my own candidate count and chunk length, because there is
no number to borrow."

**Hydration** is the stage that appears in no architecture diagram and every real trace. Many pipelines
store only IDs and vectors in the index and must fetch the actual text from a document store before
assembling the prompt. Issued as k individual gets rather than one batch get, k=50 at even 5 ms each is
250 ms of pure latency for zero computation. Scaling class: **linear in k if unbatched, fixed if
batched.** Check this first when a trace surprises you.

**Prefill** processes the entire assembled prompt before a single token comes out, so TTFT scales
directly with total input tokens. This is where retrieval quality converts into latency: every extra
chunk is paid at the prefill rate on every query, and going from k=5 to k=20 does not merely cost more
reranking, it roughly quadruples prefill work. Scaling class: **linear-to-superlinear in context
length** — the attention term is quadratic, so very long contexts degrade worse than proportionally.

**Decode** is memory-bandwidth bound, one forward pass per token, essentially independent of context
length to first order. At 10–50 ms per token, a 300-token answer is 3 to 14.5 seconds. **For any answer
of substance this is the single largest term in end-to-end latency**, frequently exceeding the entire
retrieval stack by an order of magnitude. Scaling class: **linear in output token count.** The
highest-leverage optimization in most RAG systems is telling the model to be shorter, which is also free.

### The three facts that reorganize the problem

**ANN search is not your bottleneck and it is not close.** Roughly 97% recall over 10M vectors at 4.49 ms
mean, against a decode phase measured in seconds. The entire vector-database performance discourse —
HNSW versus IVF, the benchmark leaderboards, the competitive claims — optimizes a term that is typically
under 2% of user-visible latency. A 4× improvement in a 5 ms stage buys you 3.75 ms. If you are tuning
ef_search before you have profiled generation, you are optimizing 2 ms out of a 2,000 ms budget.

**Network round trips rival or exceed compute.** The cleanest evidence is accidental: Weaviate's
in-process 2–8 ms against Pinecone's over-the-wire 23–44 ms p90 for a comparable workload isolates the
network hop at roughly 20–40 ms — **the hop costs more than the search.** Same-AZ round trips are
sub-millisecond, cross-region within a continent is tens of milliseconds, transcontinental is 80–200 ms.
A pipeline with embedding, vector DB, reranker, and generation as four separately-hosted services in
four regions pays four independent taxes before any compute happens, and that is usually a larger effect
than every index-tuning technique combined. It is also free to fix.

**Everything before the first token is TTFT.** This is the structural insight. In a streaming
application, rewriting, embedding, search, fusion, reranking, hydration, assembly, and prefill all
complete before the user sees anything. They are not "retrieval latency" sitting alongside "generation
latency" — they are all TTFT. **Retrieval optimization and TTFT optimization are the same activity.**

---

## 2. Define the budget before you optimize anything

Before touching code, write down the SLO. Something like p50 under 1.5 s, p95 under 3 s, p99 under 5 s —
the actual numbers come from the product, and a real-time voice assistant and a compliance research tool
have no business sharing them.

Then allocate it per stage, which turns an argument into an engineering target:

```
API / network        100 ms
Query embedding      100 ms
Retrieval            150 ms
Reranking            200 ms
Prompt preparation   100 ms
LLM TTFT             500 ms
Generation           850 ms
                   --------
Total               2000 ms
```

That is a *budget*, not a measurement — an allocation you then hold each stage against. The value is
that it makes overspend visible and forces the conversation about which stage should get the next 100 ms.

### TTFT and total latency are different problems

A user tolerates a 5-second response far better when the first useful text appears at 500 ms than at 4
seconds. So measure `TTFT`, `token generation rate`, and `total generation time` separately, and
decompose end-to-end as *time before the LLM* + *TTFT* + *time to generate the remaining tokens*.

Streaming is what exploits this, and the clarification that gets missed in interviews is that
**streaming does not reduce total generation time.** If the model needs 4 seconds it still needs 4
seconds. Streaming reduces *perceived* latency and improves time-to-useful-output. Saying that
distinction unprompted is a small, cheap credibility signal.

There is a ceiling worth knowing: adults read at roughly 5–6 tokens per second, so beyond about 20
tokens/second further decode speedups are imperceptible [vendor-published]. Once TPOT is under ~50 ms,
stop optimizing inter-token latency and spend everything on TTFT — which for RAG *is* the retrieval
stack.

And the rule inverts completely for agentic consumers. When the RAG call feeds another program rather
than a human, nothing is consumed incrementally and only end-to-end latency exists. Streaming buys
nothing, and decode length becomes dominant again. A system serving both audiences needs both metrics on
the dashboard, and they will disagree about what to fix.

### Percentiles, and why the tail is the number

Never optimize the mean. A pipeline with 90% cache hits at 200 ms and 10% misses at 5 s has a 680 ms
mean, which describes a request that never happens. RAG pipelines are bimodal by construction — hit
versus miss, short context versus long, warm connection versus cold — and averages hide exactly that.
Pinecone's own published sample run makes the point without argument: **p90 of 44 ms against a max of
4,602 ms**, in a controlled same-region benchmark [vendor-published].

The tail math is worth being able to do out loud, because it is where most candidates hand-wave.

**Fan-out.** When a stage issues n calls in parallel and needs all of them — sharded index, hybrid
retrieval, batched reranking — its latency is the *maximum*, and `P(max > t) = 1 − (1−p)^n`. Tail
probability multiplies by fan-out width. Dean and Barroso's canonical example: a server with a 1-second
p99 means 1 request in 100 is slow; fan out to 100 such servers and **63% of user requests take over a
second.**

**Sequential.** A RAG pipeline is mostly serial, and here the math differs. Means add unconditionally.
Quantiles do *not* add — the sum of per-stage p99s is a conservative overestimate, not the end-to-end
p99. The useful inversion is the union bound: to hit an end-to-end p99 across N stages, each stage must
meet its budget with probability 0.01/N — **p99.83 for a six-stage pipeline.** Per-stage tails have to
be tighter than the percentile you are selling, and adding a seventh stage does not just add its mean, it
tightens the requirement on all the others.

One honest caveat that will impress anyone who has actually done capacity work: variances only add
cleanly under independence, and stages in a RAG pipeline share a GPU, a network, a node, and an
autoscaler. Under load they slow down *together*, all covariances go positive, and the real tail is
fatter than the independent model predicts. **The independence math is a lower bound on your tail, not
an estimate of it** — and it understates exactly when p99 matters most.

---

## 3. How to measure it

Emit one span per stage on a single trace: rewrite (with output token count), embedding, ANN search,
lexical search, fusion, rerank (with candidate count and chunk length), hydration, assembly (with final
token count), and generation (with TTFT as a span event, plus input and output token counts). **The
attributes matter as much as the durations** — a duration without the k and token counts that produced it
cannot be interpreted and cannot predict what happens when you move the knob.

Use the OpenTelemetry GenAI semantic conventions rather than inventing names, so traces stay portable:
`gen_ai.client.operation.duration`, `gen_ai.client.operation.time_to_first_chunk`,
`gen_ai.server.time_to_first_token`, `gen_ai.execute_tool.duration` and friends.

The output is a per-request breakdown you can act on:

```
request_id: 12345
query_embedding:    73 ms
dense_search:       42 ms
bm25_search:        31 ms
fusion:              4 ms
reranker:          118 ms
document_fetch:     21 ms
prompt_build:        7 ms
LLM TTFT:          480 ms
LLM generation:   1100 ms
                 --------
total:            1876 ms
```

Without this you get the standard incident conversation — "maybe the vector DB? maybe the LLM? maybe
embeddings?" — which is guessing. With it you know the reranker is 6% and generation is 84%, and you
know where not to work.

### Four measurement traps

**Server-side spans exclude the things users feel.** A server span starts when your handler is entered.
It misses DNS, TLS, connection-pool wait, load-balancer and ingress queueing, and last-mile network. For
a mobile or cross-region client those can exceed everything the span contains. Healthy server-side p99
while users complain is the standard symptom, and it means you need client-side measurement too. For
hosted third-party APIs you fundamentally cannot separate provider queueing from network transit — you
observe only the sum, which means a regression in a hosted dependency is detectable but not diagnosable
from your side.

**Attempt latency and call latency are different numbers.** Attempt covers one try; call includes retries
and backoff. A pipeline with a retry on a flaky embedding call can have excellent attempt percentiles and
catastrophic call percentiles. Useful heuristic: if your p99 looks like `p50 × 2 + 1000 ms`, look for a
retry before you look for a slow stage.

**Once anything runs concurrently, sum-of-spans exceeds wall clock.** A "time by stage" bar chart built by
summing durations then apportions more than 100% of the request. The correct analysis is **critical
path** — for parallel dense and sparse retrieval only the slower branch counts, and the faster
contributes zero marginal latency. Report the distribution of *which stage was on the critical path*
alongside durations. That stage changes identity across the distribution — search might be on the
critical path for 2% of requests at p50 and 30% at p99 — and that shift is the most actionable thing a
trace can tell you.

**Never build a p99 waterfall from per-stage p99s.** This is the subtlest one. Stack each stage's
independently-computed p99 into a bar chart and you have described a request that does not exist; no
single request was simultaneously at the 99th percentile of every stage. The correct method is
exemplar-driven: select traces whose *end-to-end* latency lands in the p99 bucket and look at the stage
breakdown of *those specific traces*. This routinely shows that slow requests are slow for a reason that
never appears in the aggregate — a cold connection, a retry, a selective filter forcing deep traversal,
an unusually long document blowing up reranking — rather than being uniformly stretched versions of the
median. Relatedly: **averaging percentiles is arithmetically invalid.** The mean of per-host p95s is not
the p95. Merge histograms, not summary statistics.

---

## 4. The optimization hierarchy

This is the part to memorize, because it is the shape of a good answer and it is ordered by leverage:

```
LEVEL 1  Eliminate work    cache, skip RAG, skip reranking, avoid unnecessary calls
LEVEL 2  Parallelize work  dense + BM25, metadata + retrieval, independent tool calls
LEVEL 3  Reduce work       smaller context, smaller top-k, smaller models, shorter prompts
LEVEL 4  Make work faster  ANN tuning, GPU, quantization, batching, optimized serving
LEVEL 5  Perceived latency streaming, progressive UI, early feedback
```

Most candidates start at level 4 because it is the level with the most vocabulary attached. Level 1 is
where the wins are, because you are not making an expensive operation faster — you are removing it.

### Level 1 — eliminate

**Conditional execution / early exit.** Not every query needs the full pipeline. "Hello" does not need
embedding, vector search, reranking, and a large model. Route: cache hit returns immediately, a
recognized FAQ intent returns a canned answer, a simple query skips retrieval entirely, and only genuine
knowledge questions go down the expensive path. This is often the single highest-impact change available.

**Skip reranking when the retriever is already confident.** If the top 3 results have very high
similarity and a wide margin over rank 4, a heavyweight cross-encoder adds latency and almost nothing
else. Gate it on a confidence threshold: high confidence takes the fast path, low confidence pays for
the reranker. Fast path plus accurate fallback is a strong production pattern and a strong interview
answer, because it shows you think of the reranker as a cost rather than a component.

**Adaptive top-k.** `top_k = 50` for every query pays maximum cost for the queries that needed 5.
Different questions have different information requirements; make retrieval depth a function of query
type rather than a constant.

**Move everything possible offline.** Document parsing, chunking, metadata extraction, embedding, index
construction, deduplication, and summarization all belong before the user's query arrives, not during
it. The online path should be as close to *cache → embed → retrieve → rerank → generate* as you can get
it. The more work you move from online to offline, the better latency gets, and this is the least clever
and most effective idea in the chapter.

### Level 2 — parallelize

Dense search, BM25, metadata lookup, and cache lookup have no data dependencies on each other. Run them
sequentially at 100 + 80 + 50 ms and you pay 230 ms; run them concurrently and you pay max(100, 80, 50) =
100 ms. In Python that means actually using async rather than awaiting independent network calls in
sequence — a surprisingly common bug, since `await` in a row *looks* concurrent.

The senior framing, and the sentence worth saying in an interview, is: **which operations have
dependencies and which can safely run concurrently?** That is a more interesting question than "use
async," and it has real answers — query rewriting blocks retrieval only if you throw away the raw query;
hydration can be prefetched during reranking for the candidates likely to survive; the cache lookup can
race the whole pipeline.

The cost, per the tail math above, is that fan-out multiplies tail probability. Parallelism improves p50
more than p99, and adding a fifth parallel branch to shave the mean can widen the tail.

### Level 3 — reduce

**Cut candidates before reranking.** Reranking is linear in candidate count, so retrieve 50 → rerank 20 →
keep 5 often produces nearly the same answer quality as retrieve 1000 → rerank 1000 at a fraction of the
cost. The exact numbers have to come from your evaluation, but the shape is reliable.

**Cut what reaches the generator.** 20 chunks × 500 tokens is 10,000 tokens of prefill on every query,
and the goal is not "give the model as much information as possible" but **"give the model the smallest
amount of high-quality information required to answer."**

This is the one family where latency and quality may move *together*, and the literature is worth knowing
precisely because it is contested. *Lost in the Middle* (Liu et al.) shows a U-shaped curve — performance
is highest when relevant information sits at the beginning or end of context and degrades significantly
in the middle — so at large k you are paying full prefill cost for positions the model can barely use.
Anthropic's contextual retrieval work reports the opposite direction, that top-20 outperformed top-10 and
top-5 on their internal suite [vendor-published], though note those are *retrieval* failure rates and the
stack included reranking. And the most-cited justification for keeping k high — *The Power of Noise*,
which reported that random documents improved accuracy by up to 35% — was reproduced in 2026 by *The
Powerless Noise*, which attributes the effect to three experimental artifacts (4-bit quantization, no
chat template on chat-tuned models, and a 15-token output cap colliding with the instruction) and reports
that with those fixed "the effect vanishes entirely for several models" [benchmark]. **Do not cite Power
of Noise as a reason to keep k high.**

The defensible synthesis: reducing k after a good reranker is much safer than reducing k without one,
because the reranker concentrates signal into the top positions. Cutting k blind, with no reranker, is
cutting blind.

**Control conversation history.** Turn 1 is 1,000 tokens, turn 5 is 10,000, and if you resend everything
every turn the prompt grows without bound. Use recent messages plus a summary plus retrieved relevant
history rather than the full transcript.

**Don't return data you don't need.** If your application uses IDs and metadata, do not ask the vector
store to return the raw vectors — Pinecone specifically recommends against it because it adds latency,
particularly on larger result sets [vendor-published]. It looks like a trivial optimization until you
notice it is 1536 floats × k on every request.

### Level 4 — make it faster

This is the level with the most vocabulary and the least leverage, which is precisely why it is where
weak answers start.

**ef_search is the real dial** and it has a sharply diminishing-returns shape. The best public
recall-latency curve I found, with full methodology (6.7M Wikipedia articles, 633 queries, MiniLM-384,
M=16, OpenSearch/Lucene, M1 Max) [benchmark]:

| ef_search | Latency | Recall@10 |
|---|---|---|
| 10 | 5.31 ms | 0.51 |
| 240 | 34.96 ms | 0.90 |
| 640 | 68.28 ms | 0.95 |

**6.6× latency to buy the first 39 points of recall, then 2× more for the last 5.** The last few points
are the most expensive you will ever buy. The honest gap, worth naming: nobody has published the mapping
from ANN recall to *end-answer* quality, so every "target 0.95 recall" recommendation is unjustified in
both directions. There is good reason to think it is heavily sublinear in a reranked pipeline — a missing
#7 neighbor replaced by a #12 that is nearly as good gets sorted out by the reranker — and much less
forgiving in a no-rerank k=3 pipeline. **The right recall target is a function of how much downstream
slack you have, not a universal number.**

**Quantization** is a memory optimization first and a latency optimization second. Qdrant reports scalar
quantization at 4× compression with under 1% accuracy loss, and binary at up to 32× with "up to 40×
speedup" [vendor-published] — but the recall figures and the speedup are *not the same configuration*.
Every published recall number requires 3–4× oversampling plus rescoring against full-precision vectors,
which consumes an unpublished share of the speedup. And the caveat dropped from every summary: binary
quantization "gives poorer results for small embeddings, i.e. less than 1024 dimensions" — the 768-dim
models in their own table land at 0.944–0.956 recall, a 4–6 point loss.

**Dimension reduction** via Matryoshka embeddings speeds distance computation roughly linearly in d and
shrinks the index so more stays in RAM, which is often the larger effect. It does **not** speed up the
embedding forward pass at all, since truncation happens after inference. To reduce embedding latency you
need a genuinely smaller model, which is a real quality trade — and before you make it, check you are not
simply calling your current model across an ocean.

**Batching helps throughput and hurts interactive latency.** Milvus measured batch=1 → batch=10
*increasing* per-request latency 2–5× [operator blog]. Decagon found batch size 2 optimal for their
reranker [operator blog], which cuts against GPU-utilization intuition. Continuous batching in LLM
serving is a genuine improvement because requests enter and leave dynamically, but "batching is good" is
a throughput statement being smuggled into a latency conversation. And in vLLM the tradeoff is more
specific than latency-versus-throughput: chunked prefill trades **TTFT against inter-token latency**, with
the default `max_num_batched_tokens` of 2048 tuned for ITL and larger values improving TTFT
[vendor-published].

**GPU is not automatically faster.** A remote GPU reranker adds network, serialization, queueing, cold
starts, and batching delay. A small CPU cross-encoder running in-process can beat a remote GPU endpoint
outright. Benchmark the complete request path, not the kernel.

**Kill network hops.** Same region as your index, connection pooling and keep-alive, persistent
connections, gRPC where appropriate, pre-warmed models and connection pools so cold starts never land on
a user. This is the cheapest item at this level and usually the biggest.

### Level 5 — perceived latency

Streaming, progressive UI, early feedback. It does not make anything faster; it changes what the wait
feels like, and for human-facing products that is often worth more than 200 ms of real improvement.

---

## 5. Caching, in full

Caching deserves its own section because it spans the whole hierarchy, and because "explain semantic
caching" is now a standard interview question in its own right.

### Exact-match response cache

Hash the normalized query and return the stored answer. Zero false positives by construction. Hit rate
depends entirely on the query distribution — high for FAQ and support surfaces, near zero for open-ended
assistants.

### Query embedding cache

Cheap, safe, underused. Exact-match on normalized query text → embedding vector. Zero quality risk
because it caches a pure function, and it removes a network round trip to the embedding provider from the
critical path. Given the 3–4× cross-region penalty on embedding APIs, this is a real win in a head-heavy
query distribution.

### Retrieved-results cache

Cache query → top-k document IDs, skipping embedding, the vector DB request, and candidate processing.
Freshness is the constraint: if the knowledge base changes, this returns stale document sets silently.

### Prompt / prefix caching

The highest-leverage caching technique for RAG, because it is **correctness-preserving by construction** —
it caches computation (KV state), not answers. Anthropic prices cache writes at 1.25× base input (5-minute
TTL) or 2× (1-hour) and **reads at 0.1× input** [vendor-published]. Their published latency table (−79%
for a 100k-token cached prompt, −31% for 10k, −75% for a 10-turn conversation) is attributed to "early
customers" with no methodology, so treat the specific percentages as illustrative — though the *shape* is
mechanistically credible, since cache hits skip prefill and prefill dominates TTFT, so the reduction
should scale with cached fraction of input.

Two traps worth knowing. **Below the minimum cacheable prefix, nothing is cached and no error is
returned** — you have to check the token counts in the response to confirm, and this silently costs people
money for months. And the canonical mistake is placing the cache breakpoint *after* a block containing a
timestamp or per-request ID, which makes the prefix hash differ every request: 100% miss rate while
paying 1.25× on writes.

The structural awkwardness for RAG is that prefix caching rewards a stable prefix and RAG injects fresh
per-query chunks — the least cacheable content possible. You can cache the system prompt, tools, and
few-shot examples; you cannot cache retrieval results. The mitigation is to deliberately order the prompt
so all volatile content sits *after* all stable content. Note also that the biggest published win
(−79%, "chat with a book") is a *different architecture* that skips retrieval entirely, not a tuned RAG
pipeline.

### Semantic caching — the full answer

**The mechanism.** Embed the incoming query. ANN-search it against the embeddings of previously cached
queries. If the nearest cached query exceeds a similarity threshold, return its stored response. On a
miss, run the full pipeline and write the result back. Apply TTL and invalidation policies, and monitor
hit rate.

**Why it saves money and latency.** "What is the leave policy?" and "Tell me about our leave rules" are
different strings with the same meaning. Exact-match caching sees two distinct queries; semantic caching
serves the second from the first. The savings are real and compound: on a hit you skip retrieval,
reranking, prefill, and decode — the entire cost of the request, not a fraction of it.

**The tradeoff, stated honestly, which is what the interview is actually testing.** A higher similarity
threshold means fewer hits and higher precision; a lower threshold means more hits and more incorrect
reuse. And this is not a tuning problem you can solve, it is structural. vCache (arXiv 2502.03771)
measured the underlying issue precisely: correct and incorrect cache hits have **highly overlapping
similarity distributions**, and the optimal per-prompt threshold spans a range from 0.71 to 1.0. No
single global threshold works for all queries. Their measured baseline is worth memorizing: **GPTCache at
threshold 0.99 produced 2.5% error at a 37% hit rate; at 0.97, 5.2% error at a 67% hit rate**
[benchmark].

Sit with that. A semantic cache tuned to a useful hit rate returned a wrong answer on roughly **1 in 20
requests**. That is not a latency optimization; it is a quality regression sold as one. Saying this in an
interview — with the number — is the difference between reciting the mechanism and understanding it.

**Where embedding similarity fails specifically.** Negation, dates, entity names, version numbers.
"Revenue in Q3 2024" and "revenue in Q3 2025" are extremely close in embedding space and have completely
different correct answers. Any domain where a small token carries the semantic load is a domain where
semantic caching is dangerous.

**How to deploy it responsibly.** Start conservative (0.9+) and tune down with measurement, never
intuition. Build a labeled set of query pairs from your own distribution and measure precision and recall
separately — target high precision before production. Use a confidence buffer: if your validated
threshold is 0.90, serve only above 0.92. Layer an exact-match cache in front so obvious repeats never
reach the semantic path. Partition by domain and tenant. And consider using a near-match as a *routing
hint* — telling you which documents to retrieve — rather than as an answer source, which captures much of
the latency win with none of the correctness risk.

**Where it is right:** high-volume narrow domains, support deflection, product FAQ — places where you can
afford to build the labeled set and actually measure the false-positive rate. **Where it is wrong:**
anything where a wrong answer is expensive, anything with per-user data, anything where queries differ by
small load-bearing tokens.

### Invalidation, which is the hard part

A correct cache key is a version vector, not a query hash:

```
query_norm + tenant/permission_scope + filters + index_version
           + embedding_model_version + reranker_version
           + generator_model_version + prompt_template_version
```

Omit `index_version` and a document update does not invalidate answers derived from its old content.
Omit `prompt_template_version` and your prompt change silently does not take effect for cached queries —
and you will conclude the change did not help. **Omit `permission_scope` and your cache is a data-leak
primitive**, which is the failure nobody catches in staging.

TTL bounds staleness but does not eliminate it, and it is not a substitute for event-driven invalidation
on document update when correctness matters. "What is the company's leave policy?" can be cached for
hours; "what is today's stock price?" cannot be cached at all. That distinction is a property of the
question, not of the system, which means routing has to know about it.

---

## 6. Reliability is part of latency

Two things that belong in a latency answer and are usually missing.

**Retries can make latency worse.** A vector DB request that times out at 1 second and retries costs the
user 2+ seconds before the LLM has even started. Retries need bounded counts, exponential backoff,
circuit breakers, and — most importantly — a total time budget, so an unhealthy dependency cannot consume
the entire request. Retry storms also correlate across stages under load, which is exactly when your tail
math stops holding.

**Timeouts need fallbacks, not failures.** Reranker times out → use the original retrieval ranking and
continue. Large model unavailable → fall back to the smaller one. Vector DB unavailable → serve from
cached retrieval. A production AI system should degrade gracefully, and "one dependency fails, the whole
application fails" is a design choice you made by not making one. Graceful degradation is also a latency
strategy: a slightly worse answer in 2 seconds usually beats a better answer in 12.

---

## 7. A worked example

The article this chapter draws on has a good end-to-end example. A system at roughly 5 seconds:

```
Query embedding    150 ms
Vector search      200 ms
Query rewriting    500 ms
BM25               150 ms
Reranking          700 ms
Prompt processing  150 ms
LLM TTFT           900 ms
Generation        2250 ms
                 --------
Total             5000 ms
```

Optimize in hierarchy order. **Remove unnecessary query rewriting** where the query is already
well-formed: −500 ms, and note this is level 1, eliminating work, not making it faster. **Parallelize
vector search and BM25**: 350 ms becomes max(200, 150) = 200 ms, −150 ms. **Reduce reranking candidates**:
700 → 250 ms. **Reduce context** so prefill shrinks: TTFT 900 → 550 ms. **Route simple queries to a faster
model**: generation 2250 → 900 ms.

Roughly 5 seconds becomes roughly 2, without changing the architecture. And the lesson to state out loud:
**the biggest gains came from eliminating work and reducing LLM computation — not from micro-optimizing
the vector database.** Nobody touched the index.

Two caveats I would add to that example if asked, because they matter. The numbers are illustrative
rather than measured, which is fine for a worked example and not fine for a budget. And the last
optimization — routing simple queries to a faster model — is the only one that trades quality, so it is
the only one that needs an eval gate before it ships.

---

## 8. The interview answer

Asked "how would you optimize a RAG system for latency in production," the answer that lands starts with
measurement and ends with re-measurement:

> "First I'd instrument the complete pipeline and establish a p95/p99 latency budget — measuring query
> embedding, retrieval, reranking, document fetch, prompt construction, TTFT and generation separately,
> because 'retrieval is slow' is not actionable and the stage everyone suspects is usually not the one.
>
> Then I'd find the largest contributor, which in most RAG systems is generation, and after that
> reranking. Vector search is typically single-digit milliseconds and not worth touching first.
>
> Then I'd work in order of leverage: eliminate work before parallelizing it, parallelize before
> reducing it, reduce before making it faster. Can I skip retrieval for this query class? Can I skip
> reranking when the retriever is confident? Can I cache? Can I cut top-k? Then run dense and lexical
> retrieval concurrently. Then cut context, which reduces prefill and often improves quality. Only then
> would I touch ANN parameters or model serving.
>
> Streaming last, to improve perceived latency once the real numbers are as good as they're going to get.
>
> And after every change I'd re-measure four things, not one: did latency improve, did quality change,
> did cost change, did reliability change. A latency win that quietly costs three points of answer
> accuracy is not a win, it's a trade someone needs to approve."

The single most important rule, compressed: **measure → identify bottleneck → eliminate unnecessary work
→ parallelize → reduce workload → optimize infrastructure → re-measure.**

Do not start with "which vector database is fastest?" Start with "where exactly is my latency budget
being spent?"

---

## 9. The mistakes that cost people the round

**Optimizing before profiling.** "Vector search is probably slow." It probably isn't.

**Raising top-k to improve quality, then being surprised latency rose.** Retrieval, reranking, and prefill
all just got more work, and per §4 the quality gain may not even exist.

**Sending everything to the model.** 50 chunks plus 10,000 tokens of history plus a 5,000-token system
prompt is expensive, slow, and often *worse* — the middle of that context is barely read.

**Adding agents everywhere.** Every reasoning step is another model call plus tool call plus retrieval
plus another model call. Agentic RAG is powerful and it multiplies latency; it should be a routing
decision, not a default.

**Using the largest model for everything.** A simple FAQ lookup does not need a frontier reasoning model.

**Assuming streaming means faster inference.** It improves perceived latency, not computation time.

**Ignoring cache invalidation.** A stale cache is a correctness bug wearing a performance costume.

**Optimizing p50 only.** Your p99 users may be waiting 15 seconds while your dashboard looks healthy.

**Treating latency and quality as separate projects.** System A at 500 ms with 70% recall and System B at
800 ms with 95% recall — neither is better in the abstract. A real-time assistant wants A; a compliance
research tool wants B. **You cannot answer the latency question without the product requirement**, and
saying so is not a dodge, it is the answer.

---

## 10. If you remember one analogy

A pizza restaurant that makes dough from scratch, cuts vegetables, makes sauce, and heats the oven on
every order is your unoptimized RAG system. The good restaurant prepared the dough earlier, keeps the
oven hot, handles orders in parallel, uses only the ingredients this pizza needs, and serves slices as
they come out.

Precomputed embeddings are the pre-made dough. The pre-warmed model is the hot oven. Parallel retrieval
is multiple cooks. Context selection is only the toppings you need. Streaming is serving slices
immediately. The cache is remembering the regular's usual order.

The best kitchen does not make every step faster. It **avoids unnecessary work, prepares in advance, runs
independent work in parallel, and gets something in front of the customer as soon as possible.** That is
the whole chapter.

---

## Sources and evidence

Numbers in this chapter carry markers: **[vendor-published]** for a figure published by the operator of
the system with at least a partial measurement setup, **[operator blog]** for an engineering post from a
company running it in production, **[benchmark]** for a reproducible harness with published
configuration, **[derived]** for arithmetic I performed on a cited figure.

Where a number is widely circulated but has no published methodology, it is excluded rather than
repeated. Notable gaps that you should not try to fill from the internet: **absolute latency for any
hosted reranking API** (neither Cohere nor Voyage publishes one), **standalone BM25 latency at a stated
corpus size**, **a formula relating prompt tokens to prefill milliseconds** (both ClickHouse and Anyscale
explicitly decline to give one), **filtered ANN search latency as a function of selectivity**, and **the
mapping from ANN recall to end-answer quality**, which is the single most useful missing number in the
field.

- [Weaviate ANN benchmarks](https://docs.weaviate.io/weaviate/benchmarks/ann) — the ANN latency table
- [Pinecone: test at scale](https://docs.pinecone.io/guides/get-started/test-at-scale) and their latency guidance
- [Milvus: we benchmarked 20 embedding APIs](https://milvus.io/blog/we-benchmarked-20-embedding-apis-with-milvus-7-insights-that-will-surprise-you.md) — geography, batching, DB overhead
- [Decagon: designing low-latency AI agents through reranker optimization](https://decagon.ai/blog/designing-low-latency-ai-agents-through-reranker-optimization) — stage ranking, batch size 2
- [Qdrant quantization](https://qdrant.tech/documentation/guides/quantization/) — compression and recall
- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [vLLM optimization](https://docs.vllm.ai/en/v0.8.2/performance/optimization.html) — chunked prefill, TTFT vs ITL
- [ClickHouse: LLM inference latency](https://clickhouse.com/resources/engineering/llm-inference-latency) and [tail latency](https://clickhouse.com/resources/engineering/tail-latency)
- [Anyscale: LLM serving metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics)
- [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md)
- vCache, Schroeder et al., [arXiv 2502.03771](https://arxiv.org/abs/2502.03771) — semantic cache error rates
- *Lost in the Middle*, Liu et al., [arXiv 2307.03172](https://arxiv.org/abs/2307.03172)
- *The Power of Noise*, Cuconasu et al., [arXiv 2401.14887](https://arxiv.org/abs/2401.14887), and its 2026 reproduction *The Powerless Noise*, [arXiv 2607.03615](https://arxiv.org/abs/2607.03615)
- Dean & Barroso, *The Tail at Scale*, [summary](https://blog.acolyer.org/2015/01/15/the-tail-at-scale/)
- Ayush Singh, [How to Optimize RAG for Latency in Production](https://www.linkedin.com/pulse/how-optimize-rag-latency-production-ayush-singh-vrezc/) — the structure of this chapter, the optimization hierarchy, the worked example, and the pizza analogy come from this article; the measured figures and the evidence discipline are added here.
