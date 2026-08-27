# The applied GenAI one-pager

The rest of this folder compresses the ML material — losses, gradients, attention, the code you must be
able to write cold. This page compresses the *applied* layer: RAG, agents, evaluation, latency, security.
It exists because an AI engineer loop tests different things than an ML loop, and the night before is a
bad time to discover which one you're in.

For staring at, not reading. Depth is in [`74_ai_engineer_interview_prep`](../74_ai_engineer_interview_prep/README.md) and
[`39_rag_retrieval_augmented_generation`](../39_rag_retrieval_augmented_generation/README.md).

---

## The five numbers to have in your mouth

| Thing | Number | Why it matters |
|---|---|---|
| ANN search, 10M vectors, ~96% recall@10 | **4.5 ms mean, 7.7 ms p99** | Vector search is not your bottleneck. Not close. |
| Decode | **10–50 ms/token** | 300 tokens = 3–15 s. This is your latency. |
| TTFT, well-tuned | **200–500 ms** | Everything before the first token lives here. |
| Semantic cache at 67% hit rate (GPTCache) | **5.2% wrong answers** | It's a quality regression sold as a latency win. |
| RRF constant | **k = 60** | Convention, not a tuned value. |

## The three shapes

**Latency budget.** `total ≈ request overhead + retrieval + reranking + prefill + generation`, except many
stages run in parallel so it isn't a strict sum. Generation dominates, reranking is second, vector search
is a rounding error, and network hops often cost more than the compute they connect.

**Optimization hierarchy.** Eliminate → parallelize → reduce → accelerate → perceived. Weak answers start
at "accelerate" because that's where the vocabulary is. The wins are at "eliminate."

**Tail math.** Fan-out to n: `P(max > t) = 1 − (1−p)^n` — tail probability multiplies. Sequential across N
stages: to hit end-to-end p99, each stage needs **p(1 − 0.01/N)** — p99.83 for six stages. Quantiles do
not add; means do.

## Diagnostic reflexes

Given a symptom, the five moves: **what changed** → **bisect, don't scan** → **falsifiable hypothesis** →
**how you'd prove it** → **stop-the-bleeding fix vs. real fix.** The fourth is the one candidates skip and
the one being graded.

The two highest-value experiments in RAG debugging:

- **Replay** old-correct queries and compare *retrieved document IDs*, not answers. Splits the pipeline in
  half in ten minutes.
- **Oracle context** — hand-assemble the perfect context and ask the model. Still wrong? Retrieval work
  won't help you.

Sudden wrong answers, in order of likelihood: model version moved under you, index rebuilt (worst case:
new embedding model on the docs, old one on the queries — retrieval goes random, dashboards stay green),
corpus got a bulk ingest of garbage, someone changed a number in a config file.

## Between retrieval and generation

Retrieval looks fine, answers are bad. In order: **position** (middle of long context is where information
goes to die), **too much context**, **conflicting sources** (usually one is outdated — surface the
conflict, don't let the model blend it), **incoherent chunks** (starts at step 4, refers to "the above
table"), **silent truncation**, **the prompt** (no grounding instruction, no "I don't know" path),
**formatting** (no delimiters, no source labels).

## What breaks at scale

10k → 1M documents. **Precision breaks first, not latency.** Every query now has hundreds of plausible
near-misses; nothing errors and quality quietly collapses. Underneath: near-duplicates, ANN recall
degrading at fixed parameters, embedding-space crowding (which is why hybrid stops being optional).

Redesign order: shrink the search space (partition and route) → multi-stage retrieve-then-rerank →
dedup and freshness as real pipeline stages → cache the head → re-tune the index *and verify recall
against exact search*.

## Security, compressed

OWASP LLM Top 10 (2025): **prompt injection, sensitive info disclosure, supply chain, data/model
poisoning, improper output handling, excessive agency, system prompt leakage, vector and embedding
weaknesses, misinformation, unbounded consumption.**

For RAG the load-bearing ones are **indirect prompt injection** (ingestion is the attack surface),
**access control at retrieval time** (a filter against the requesting principal — not after generation,
not a prompt instruction), and **vector/embedding weaknesses** (index poisoning, embedding inversion,
cross-tenant leakage).

**Lethal trifecta:** private data + untrusted content + external communication. RAG hands you the first
two by definition. Watch the third.

The one-liner: **the controls that survive an attacker are enforced in code, not in the prompt.**

## If the domain is high-stakes

**Agents can own workflows. They should not own truth.** The model plans, drafts, searches, proposes;
deterministic software decides what is valid, what gets recorded, and what happens next.

Three lines that carry the architecture: **validators run before execution, not after** (and asking a
second model whether the first looked right is critique, not validation); **budgets are a correctness
control** (unbounded search plus selective reporting is p-hacking with extra steps); **agents sharing a
model are not independent verifiers** (three agents agreeing is one opinion stated three times).

Place approval gates by **consequence if wrong × difficulty of reversal**. And the gate everyone builds
wrong: if reviewers get more cases than they can inspect, it is a rubber stamp.

Retrieval is **four** things, not one — structured DB query, lexical, vector, knowledge-graph traversal.
Saying that out loud is worth points in any design round with real structured data behind it.

## Things people get backwards

- Streaming reduces *perceived* latency, not computation time.
- Batching helps throughput and **hurts** interactive latency.
- More context is not better — cutting k often improves quality *and* latency. Do not cite *The Power of
  Noise* for the opposite; the 2026 reproduction attributes it to experimental artifacts.
- GPU is not automatically faster — a remote GPU reranker adds network, queueing, and cold starts.
- Groundedness ≠ correctness. Perfectly grounded in a wrong document is faithfully wrong.
- Temperature 0 is not determinism (batching, GPU non-associativity, MoE routing).
- Recall@k is your ceiling; precision decides whether you reach it.
- An agent's biggest practical failure is context accumulation, not reasoning.

## If they ask the open question

"How would you optimize this for latency" → *instrument first, budget in percentiles, find the largest
contributor (it's generation), then eliminate before parallelizing before reducing before accelerating,
and re-measure latency, quality, cost, and reliability — not just latency.*

"Your RAG gives wrong answers" → *when did it start, and is it everything or a subset.*

"Which is better, 500 ms at 70% recall or 800 ms at 95%" → *you cannot answer without the product
requirement.* That is the answer, not a dodge.
