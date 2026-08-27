# Diagnosing a RAG system: five questions that separate builders from engineers

These questions give you a symptom and ask what you would do. Building teaches the happy path — chunk,
embed, index, retrieve, rerank, generate. Operating teaches the failure surface, which is larger and
shaped differently, so people answer a symptom question with a parts list. That is not a diagnosis.

The five below cover five failure surfaces: sudden regression, the retrieval-generation gap, measuring a
change, multi-document synthesis, and scale.

---

## The shape of the answer

Five moves, and the moves matter more than the cause, because you cannot see the interviewer's system.

1. **Ask what changed.** Deploy, index rebuild, model version, document set, config, traffic mix,
   upstream API. "When did it start, and what shipped near that time."
2. **Bisect, do not scan.** The highest-value cut is between retrieval and generation: was the right
   content in the context? That splits the failure space in half.
3. **State a falsifiable hypothesis.** Not "it's the chunking" but "the reindex split policy sections
   across chunks, which would show as recall@10 dropping on multi-sentence policy questions and flat on
   definitional ones" — because that names the experiment.
4. **Say how you would prove it.** Design the measurement that kills four of five causes in an afternoon.
5. **Separate the stop-the-bleeding fix from the real fix.** Roll back the index; then stabilise the
   reindex job and add an eval gate.

Underneath all five: you cannot debug a system you cannot see. If no trace exists, "first I'd make this
debuggable" is an answer, not a dodge.

---

## 1. "Your RAG system suddenly starts giving incorrect answers. What's the first thing you investigate — and how would you prove that's the root cause?"

### What is being tested

**Suddenly** means regression: find the delta, do not audit the system. Correlation with a deploy is a
prior, not proof.

### The answer

Two questions first: when did it start, and is it everything or a subset? For that window I pull the
deploy log, index job history, model pinning, and config history. Four things move.

1. **Model version**, under an aliased endpoint: a prompt tuned on one version degrades on the next with
   your repository unchanged. Pin versions.
2. **Index** — new chunking parameters, new embedding model, or a dropped shard. Rebuilt on a new
   embedding model while the query encoder is old, the vector spaces no longer match, retrieval goes
   approximately random, and dashboards stay green.
3. **Corpus** — a bulk ingest of low-quality or near-duplicate documents crowds out the good ones.
   Retrieval works perfectly on garbage.
4. **Config** — top-k, similarity threshold, temperature, max context length, truncation.

Uniform degradation points at generation. Degradation in one class points at retrieval: acronym and
exact-identifier queries mean the lexical arm or hybrid weighting, conversational follow-ups mean the
query rewriter, one product area means the corpus.

### How I'd prove it

**Replay.** Rerun fifty queries correct last week and compare retrieved document IDs, not just answers.
Different IDs means the fault is at or before retrieval; identical IDs with worse answers means the
model, prompt, or assembly.

**Pin and swap, one variable at a time.** Previous model version, previous index snapshot, diffed prompt.
Snapshots mean treating the index as a deployable artifact, not a mutable thing rebuilt in place.

**Check invariants.** Index document count versus source of truth; embedding dimension and model identity
in index metadata versus the query path; mean top-1 similarity, where 0.82 to 0.61 overnight means the
vector space changed. Monitor that one continuously.

Proof standard: I can turn the failure off and on. Reverting restores the replay set, reapplying breaks
it; anything less is a correlation I chose to believe.

### The follow-ups

**Follow-up.** *"No replay set."* Rebuild from logged queries; if retrieved IDs were never logged, that
is this incident's real finding.

**Follow-up.** *"Dashboards are green."* They measure whether it ran, not whether it was right. Add
thumbs, escalation rate, or abandonment, plus a canary eval set.

**Follow-up.** *"Catch it earlier?"* Eval gated on every prompt, model, retrieval-config and index
change, plus a nightly frozen set — which stops describing the system if the corpus moves and the set
does not.

> **Say it.** Suddenly means regression, so I look for the delta rather than auditing the system. I ask
> when it started and what shipped near then — deploys, index jobs, model pins, config. Then whether it
> is everything or one query class, because uniform failure points at generation and concentrated failure
> points at retrieval. Then I replay fifty known-good queries and compare retrieved document IDs, which
> tells me which half broke. I only call it root cause when I can turn the failure off and on by
> reverting and reapplying.

---

## 2. "Your retriever returns relevant documents, but answer quality is still poor. What could be going wrong between retrieval and generation?"

### What is being tested

Whether you know the pipeline has a middle. Most mental models are retrieve → generate, one arrow; there
are at least six things on that arrow.

### The answer

I'd look at the assembled prompt string for a failing case first. Then, in order of frequency:

1. **Position.** Models attend unevenly across long context; the middle is where information goes to die.
   Test by reordering top-ranked chunks to the beginning and end, or cutting k — quality *improving* on
   less context is the signature.
2. **Too much context.** Twenty chunks where five would do dilutes the signal, and one plausible-wrong
   chunk can dominate. Recall sets the ceiling; precision decides whether you reach it.
3. **Conflicting sources.** Usually one is outdated. The model blends or picks arbitrarily, and blended
   answers are individually sourceable and jointly wrong. Fix with effective date, version and status
   surfaced in context and used in ranking, plus deprecating old documents.
4. **Individually retrievable, jointly incoherent chunks.** The chunk starts at step 4 and refers to a
   table in another chunk: relevant by every metric, useless to the model. Parent-document and contextual
   retrieval fix this — search small units, pass the coherent larger one.
5. **Truncation.** Silent, cuts the end of the context where the instructions usually are, and degrades
   long queries only.
6. **The prompt.** No grounding instruction, no "say you don't know" path, or a conflict such as "be
   concise" against a question needing a full procedure.
7. **Formatting.** Chunks concatenated without delimiters or source labels leave no way to tell documents
   apart and no way to cite. Separators and metadata headers cost nothing.

### How I'd prove which one

An **oracle context test**: hand-assemble the perfect context for failing questions — right passages,
correctly ordered, nothing else. Correct answers mean the fault is entirely in assembly, a bounded
problem. Still-wrong answers mean the model or the prompt, and no retrieval work will help.

Then ablate against a fixed set, one variable each: k=20 versus k=5, reordered versus original, with and
without source labels.

### The follow-ups

**Follow-up.** *"Retrieval metrics look great, users are unhappy."* The metric measures the wrong unit:
recall@10 on document-level labels can be 95% while the passage the user needed sits in a chunk you never
retrieved. Or the labels are stale, or retrieval is fine and the problem is downstream.

**Follow-up.** *"Detect this automatically?"* Groundedness scoring. Ungrounded claims mean the model went
outside the context; grounded-but-wrong points back at retrieval or conflicting sources, and separating
those two is what makes the signal actionable.

> **Say it.** First I print the actual string we sent the model, because most of these failures are
> visible in it. The usual causes are position, too much context, conflicting sources, chunks that are
> individually relevant but incoherent alone, silent truncation, a prompt with no grounding instruction,
> and missing source delimiters. To separate them I run an oracle context test: I hand-build the perfect
> context and ask the model. If it answers correctly the bug is in assembly. If it still fails it is the
> model or the prompt, and no retrieval work will help.

---

## 3. "How would you know whether improving embeddings actually improved the system? What would you measure before and after?"

### What is being tested

Whether you can evaluate a change rather than believe in it, and whether you know a component improvement
is not a system improvement.

### The answer

Three levels, because an embedding change can improve one and not the others.

1. **Retrieval in isolation.** Against labeled query-document pairs: recall@k at the k you actually use,
   nDCG@10 for ranking, MRR if there is one right answer. Lead with recall@k — it is the ceiling, so a
   change that does not move it is unlikely to move anything.
2. **End to end.** Answer accuracy, or LLM-judged faithfulness and helpfulness, on a fixed question set.
   Routinely smaller: 6 points of recall might buy 2 points of answer accuracy, because questions
   retrieval already handled do not improve.
3. **Operational.** Index build time, index storage, query latency at p50 and p95, cost per query. Three
   points of retrieval for 200ms and double the index size is a tradeoff, not a free win.

Hold everything else fixed — same chunking, k, reranker, prompt, questions. Embedding and chunking
changes ship together constantly and then nobody knows which did the work.

Look at the distribution, not the mean: 5 points of average recall while regressing your highest-traffic
category is a bad change wearing a good number. Segment by query type — definitional, procedural,
exact-identifier, conversational follow-up — and check exact-match cases, because stronger semantic
models trade lexical precision for semantic breadth.

### The offline-to-online step

Offline evaluation earns the right to an online test, no more. Online: thumbs, escalation-to-human rate,
follow-up-question rate as a dissatisfaction proxy, session abandonment, task completion. Run an A/B with
a pre-registered primary metric, enough traffic for the effect size you care about, and the rollback rule
written down first.

Two traps: the eval set the model was selected on cannot also validate it, or you are measuring your own
selection process; and re-embedding compares against a rebuilt index, so a failed rebuild looks exactly
like a bad embedding model.

### The follow-ups

**Follow-up.** *"Offline improved, online did not."* Retrieval was not the bottleneck for real traffic
though it was for your eval set, so the eval set does not reflect production. Or it is stale, or the gain
sat in a rare query type, or the online metric is underpowered and you observed "cannot tell," not "no
change."

**Follow-up.** *"How do you build the labeled set?"* Mine logs and treat documents behind accepted
answers as weak positives; better, SME labels on a few hundred pairs, a day of work that pays back
permanently. Synthetic questions are a bootstrap and a trap, because they share the document's
vocabulary, which is what retrieval is already good at.

> **Say it.** I measure at three levels and hold everything else fixed. Retrieval in isolation with
> recall@k, nDCG and MRR, because recall is the ceiling. End to end on a frozen question set, where the
> gain is usually much smaller — six points of recall might buy two points of answer accuracy.
> Operationally, p95 latency, index size, build time and cost per query. Then I look at the distribution
> rather than the mean and check exact-identifier queries. Offline wins only earn an A/B with a
> pre-registered metric and a rollback rule written down first.

---

## 4. "A user asks a question that requires information from 5 different documents. How would you design retrieval and context construction for that?"

### What is being tested

Whether you distinguish two problems that look identical. Five documents one query will find is
coverage; needing document A to know which document B to fetch is multi-hop and needs iteration.
"Increase k to 20" collapses the second into the first.

### The answer

First I'd ask which shape it is.

**Independently retrievable** — comparing five products, summarising five incidents. This is coverage and
diversity, not reasoning, and the failure mode is all five top slots going to near-duplicates of one
document, because similarity search is not built for variety. Retrieve wider, enforce diversity with MMR
or another redundancy-penalizing selection, deduplicate aggressively, and where the query decomposes,
decompose it explicitly, retrieve per subquery and merge — which guarantees per-subquestion coverage.

**Sequentially dependent** — you need document A to learn the identifier that finds document B. No single
pass works at any k, and saying that clearly is most of the answer. It needs an agentic loop: retrieve,
let the model reason about what it knows and still needs, retrieve again, with an iteration cap and a
budget. Because the cost is latency and unpredictability, route: detect multi-hop queries and send only
those down the expensive path.

**Context construction** matters as much. Group chunks by source document rather than interleaving by
score. Label each with source and date, because five documents on one topic frequently disagree and
recency is the tiebreaker. Put the most important material at the beginning and end. If 5 documents means
40 chunks, rerank hard down to the 10 that carry the answer rather than pass all 40 and hope. Then tell
the model the task is synthesis, ask for per-source attribution, and instruct it to state conflicts.
Silent reconciliation is the dangerous failure, because the output is fluent, sourced, and wrong.

### How I'd evaluate it

Recall@k is the wrong metric. What matters is whether all five required documents reached the context —
a per-question set-coverage measure. Build a labeled set of multi-document questions with their full
required sets and measure complete-coverage rate. A system at 90% recall@10 might be at 40% complete
coverage.

### The follow-ups

**Follow-up.** *"How do you know it's multi-hop first?"* Classify up front, cheaply and imperfectly, or
run one pass and escalate when the model reports insufficient context — more robust, slower. It is a
routing decision, so measure how often you route wrong and what each error costs.

**Follow-up.** *"What if they contradict?"* The correct answer is that they contradict, with dates and
sources. Resolving it invisibly removes the user's ability to notice.

**Follow-up.** *"Cost and latency of iteration?"* Multiple retrieval rounds plus multiple model calls,
several times a single-pass query on both — hence routing, caps, and a budget the model can see so it
wraps up rather than being cut off mid-reasoning.

> **Say it.** First I ask whether the five documents are independently retrievable or sequentially
> dependent, because they need different designs. If independent it is a coverage problem: retrieve
> wider, enforce diversity with MMR, deduplicate, and decompose into subqueries. If sequential, no single
> pass works at any k, so it needs an iterative loop with a cap, and I route only multi-hop queries there
> because it is slow. For context I group chunks by source, label dates, rerank hard, and tell the model
> to flag conflicts. I measure complete set coverage, not recall@k.

---

## 5. "It works perfectly with 10,000 documents. Now it has 1 million. What breaks first — and how do you redesign?"

### What is being tested

Whether you know scale changes the *kind* of problem. The naive answer is "latency" and it is wrong:
vector search is sublinear. It also matters whether growth is 100× more of the same or 100× more diverse.

### The answer

**Retrieval precision breaks first, not latency.** At 10,000 documents there are few plausible
distractors, so a mediocre retriever looks excellent. At a million, every query has hundreds of
topically adjacent and substantively wrong documents, so the top-5 fills with plausible near-misses and
the model answers from them. Nothing errors, nothing slows, quality collapses quietly.

Three mechanisms underneath. **Near-duplicates**: corpora accumulate versions, translations and copies,
so your top-10 can be ten copies of one document. **ANN recall degradation**: the approximate-versus-exact
gap widens with corpus size at fixed parameters, so an HNSW `ef_search` giving 98% recall at 10k gives
less at 1M, silently, because scores look confident either way. **Embedding space crowding**:
neighborhoods densify, distances between right and plausible-wrong compress, and dense retrieval loses
discriminative power — which is why hybrid search stops being optional, since lexical signal on
identifiers, error codes and rare entities does not degrade that way.

Then the operational things. Index build goes from minutes to hours, making reindexing a scheduled event
with a rollout plan. An HNSW index over a million chunks is a serious RAM number, and past one machine
you are distributed. Incremental updates degrade graph quality, so rebuild periodically with
snapshot-and-swap. And cost: embedding a million documents, storing them, re-embedding to change models.

### The redesign

Cheapest and highest-leverage first.

1. **Stop searching everything.** Metadata filtering, partitioning by tenant, product or document type,
   routing to the right partition. Searching 50,000 relevant documents beats 1,000,000 on precision,
   latency and cost at once; most large-corpus systems that work are several medium-corpus systems behind
   a router.
2. **Go multi-stage.** A cheap wide hybrid pass of a few hundred candidates, then an aggressive rerank to
   the handful you pass — reranking separates true matches from plausible near-misses, the problem scale
   created.
3. **Make deduplication and freshness pipeline stages.** Near-duplicate detection at ingest, lifecycle
   with supersession, reconciliation with source of truth so deletes propagate.
4. **Cache the head.** Exact-match query caching and document embedding caching; semantic caching only
   with a careful threshold and a per-tenant key, since a loose one serves confidently wrong answers.
5. **Re-tune and verify the index.** `ef_search`, `M`, probe counts and quantization right at 10k are not
   right at 1M, and product quantization starts to matter for memory. Measure recall against exact search
   on a sample rather than assuming.

### The follow-ups

**Follow-up.** *"How would you know, given nothing errors?"* You would not, without an eval set that
grows with the corpus; when the corpus 100×s and the set does not, it measures a system that no longer
exists. Add questions targeting newly added regions, and track score distribution and result diversity as
leading indicators.

**Follow-up.** *"Where does the money go?"* Embedding the corpus once, storing the index continuously,
generation per query, reranking small per query but adding up. The expensive surprise is re-embedding,
which makes the initial embedding choice an architectural commitment disguised as a config value.

**Follow-up.** *"When is a vector database wrong?"* When exact search is trivially fast, when queries
belong in SQL, when the problem is a knowledge graph of relationships rather than similarity, or when the
corpus fits in a context window and the task needs global understanding.

> **Say it.** What breaks first is retrieval precision, not latency — vector search is sublinear, but at a
> million documents every query has hundreds of plausible near-misses, so the top-5 fills with them and
> nothing errors. Underneath are near-duplicates, silent ANN recall loss at fixed ef_search, and a crowded
> embedding space, which is why hybrid search stops being optional. The redesign is to stop searching
> everything: partition and route, go multi-stage with a wide hybrid pass and a hard rerank, take
> deduplication seriously, cache the head, and re-tune the index against exact search.

---

## What all five have in common

The five questions are one question asked five ways: can you reason about a system you cannot fully see?
Operating a system teaches you that failures are rarely where you look first, that the middle of a
pipeline is where things hide, and that the difference between a guess and a diagnosis is an experiment
you can name.

Memorizing five answers fails, because the interviewer changes one detail. The transferable thing is the
shape: what changed, bisect rather than scan, make the hypothesis falsifiable, say how you would prove
it, and separate the fix that stops the bleeding from the fix that stops the recurrence.

Mechanics are in [`39_rag_retrieval_augmented_generation`](../39_rag_retrieval_augmented_generation/README.md); the component-level questions these build on
are in [`MODERN_QUESTION_BANK.md`](MODERN_QUESTION_BANK.md) §2 and §5.
