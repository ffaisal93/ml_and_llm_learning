# Diagnosing a RAG system: five questions that separate builders from engineers

There is a category of interview question that sounds like it is about RAG and is not. It gives you a
symptom and asks what you would do. Something has broken, or something is about to break, and the
interviewer wants to watch you think.

Candidates who have built RAG systems answer these badly, and the reason is specific and worth naming.
Building teaches you the happy path — chunk, embed, index, retrieve, rerank, generate. Operating teaches
you the failure surface, which is much larger and shaped completely differently. If you have only built,
you have a vocabulary for components and no vocabulary for symptoms, so you answer a symptom question by
listing components. "I'd check the chunking, the embeddings, the reranker, the prompt." That is a parts
list, not a diagnosis, and the interviewer hears it as *this person has never had one of these break at
2am*.

This chapter works five such questions in full. They are the ones I see asked most, and they cover the
five distinct failure surfaces: sudden regression, the retrieval-generation gap, measuring a change,
multi-document synthesis, and scale.

---

## The shape of the answer

Before the questions, the shape. Every diagnostic answer has the same five moves, and getting the moves
right matters more than getting the specific cause right — the interviewer has a system in their head
that you cannot see, so you will often guess the cause wrong. What they are grading is whether your
process would have found it.

**One: ask what changed.** Systems that worked and then did not are almost never mysteries. Something
moved. Deployment, index rebuild, model version, a document set, a config value, traffic mix, an
upstream API. The first question is always "when did it start, and what shipped near that time," and a
candidate who starts anywhere else has told you they would spend the first day of an incident guessing.

**Two: bisect the pipeline, do not scan it.** A RAG pipeline is a chain, and a chain is a binary search
problem, not a checklist. The single highest-value cut is between retrieval and generation: was the
right content in the context or not? That one question splits the entire failure space in half, and
everything downstream of it is a different investigation. Candidates who list components are doing a
linear scan of a space you can bisect.

**Three: state the hypothesis as something falsifiable.** "I think it's the chunking" is not a
hypothesis. "I think the reindex changed chunk boundaries so that policy sections are now split across
chunks, which would show up as recall@10 dropping specifically on multi-sentence policy questions while
staying flat on definitional ones" is a hypothesis. The difference is that the second one tells you what
experiment to run.

**Four: say how you would prove it.** This is the move nearly everyone skips, and it is the one being
graded. Any plausible-sounding person can generate causes. The engineer is the person who can design the
measurement that kills four of the five candidate causes in an afternoon. Whenever an interviewer adds
"and how would you prove that's the root cause," they are telling you outright that this is the part
they care about. Answer it before they ask.

**Five: separate the stop-the-bleeding fix from the real fix.** In production those are different
actions on different clocks. Rolling back the index is the first; fixing the reindex job so boundaries
are stable and adding an eval gate that would have caught it is the second. Saying both, in that order,
is what an on-call engineer sounds like.

One more thing, underneath all five: **you cannot debug a system you cannot see.** Several of the
answers below reduce to "I would look at the trace," and if the honest answer is that no trace exists,
say so — "the first thing I'd actually do is make this debuggable, because right now I'd be guessing"
is a strong answer, not a dodge. Interviewers who have run these systems will agree with you
immediately.

---

## 1. "Your RAG system suddenly starts giving incorrect answers. What's the first thing you investigate — and how would you prove that's the root cause?"

### What is being tested

The word doing the work is **suddenly**. This is a regression, not a quality problem, and regressions
have a discipline: find the delta, do not audit the system. A candidate who begins redesigning chunking
has misread the question. Nothing about the chunking changed — it worked yesterday.

The second half of the question — *prove it* — is testing whether you know the difference between a
story and a diagnosis. Correlation with a deploy is a strong prior, not proof.

### The answer

I'd start with two questions before touching anything: **when did it start**, and **is it everything or
a subset?**

The timing question is the cheap one and it usually ends the investigation. If quality fell off a cliff
at 14:00 Tuesday, I want the deploy log, the index job history, the model version pinning, and the
config change history for that window. Four things move in a RAG system and they cover most sudden
regressions:

- **A model version changed under you.** Provider-side updates to an aliased endpoint are the classic. A
  prompt tuned against one model version can degrade meaningfully against the next, and nothing in your
  repository changed, which is exactly why it is confusing. Pin versions; this failure is why.
- **The index changed.** A reindex ran with different chunking parameters, a different embedding model,
  or a partial failure that dropped a shard. Embedding model changes are the brutal one — if the index
  was rebuilt with a new embedding model but the query encoder is still the old one, the vector space no
  longer matches and retrieval becomes approximately random while every dashboard stays green.
- **The corpus changed.** A bulk ingest added a large volume of low-quality or near-duplicate documents
  that now crowd out the good ones. This is the one people never think of, and it does not look like a
  bug anywhere — retrieval is working perfectly, it is just retrieving the new garbage.
- **A config value moved.** Top-k, similarity threshold, temperature, max context length, a truncation
  limit. Someone changed a number in a YAML file.

The "everything or a subset" question is diagnostically richer than it sounds. Uniform degradation across
all query types points at generation — model version, prompt template, a truncation bug. Degradation
concentrated in one query class points at retrieval, and *which* class tells you a lot: if it is
acronym-heavy and exact-identifier queries, your lexical arm is broken or your hybrid weighting shifted;
if it is conversational follow-ups, your query rewriter is failing; if it is one product area, the
corpus changed there.

### How I'd prove it

Three things, roughly in order of cost.

**Replay.** This is the highest-value tool and it is only available if you built it. Take fifty queries
that were correct last week, run them through the current system, and compare not just the answers but
**the retrieved document IDs**. If the same queries now retrieve different documents, the fault is at or
before retrieval and I have cut the problem in half in ten minutes. If they retrieve identical documents
and produce worse answers, retrieval is exonerated and it is the model, the prompt, or the assembly step
between them.

**Pin and swap, one variable at a time.** Having narrowed to a stage, I bisect within it. Suspect the
model? Pin to the previous version and rerun the same fifty. Suspect the index? Point at the previous
index snapshot — which is a strong argument for keeping index snapshots and for making the index version
a deployable artifact rather than a mutable thing you rebuild in place. Suspect the prompt? Diff it. Each
of these is a single-variable experiment against a fixed query set, and a single-variable experiment is
what turns a hypothesis into a cause.

**Check the invariants.** Some things should be true regardless. Document count in the index versus
source of truth. Embedding dimension and model identity recorded in the index metadata versus what the
query path is using. Distribution of similarity scores — if mean top-1 similarity dropped from 0.82 to
0.61 overnight, the vector space changed and you know it in one query. That last one is worth building
as a standing monitor precisely because it catches the embedding-mismatch failure that is otherwise
invisible.

The proof standard I'd hold myself to: **I can turn the failure off and on.** Reverting the suspected
change restores the correct answers on the replay set, reapplying it breaks them again. Anything short
of that is a correlation I am choosing to believe, and I would say so out loud rather than declaring
victory.

### The follow-ups

*"You don't have a replay set. Now what?"* Then I build one from production logs — if I have logged
queries and retrieved IDs, I can reconstruct it. If I have not logged retrieved IDs, that is the actual
finding of this incident, and I would say so: a RAG system that does not log what it retrieved is not
debuggable, and every future incident costs what this one is costing.

*"Users report it, dashboards are green. Why?"* Because the dashboards measure whether the system ran,
not whether it was right. Latency, error rate, and throughput are all perfect during a total quality
collapse. This is the argument for an online quality signal — thumbs, escalation rate, follow-up-question
rate, abandonment — and for a small canary eval set that runs continuously against production.

*"How would you have caught this before users did?"* An eval suite gated on every change to prompt,
model, retrieval config, or index, plus a nightly run against a frozen question set. Note the trap that
the frozen set introduces: if the eval set is frozen and the corpus is not, the eval slowly stops
describing the system. Freshness of the eval set is itself a maintained property.

---

## 2. "Your retriever returns relevant documents, but answer quality is still poor. What could be going wrong between retrieval and generation?"

### What is being tested

Whether you know that a RAG pipeline has a middle. Most people's mental model is retrieve → generate,
one arrow. There are at least six things on that arrow, and each one has a characteristic failure. This
question is a direct probe for whether you have ever looked at the actual string you sent the model,
which is the single most clarifying thing you can do in RAG debugging and which surprisingly few people
have done.

### The answer

I'd want to see the assembled prompt for a failing case before saying anything else. Most of what goes
wrong here is visible in it. Assuming I have that, the candidates, roughly in order of how often I have
seen them:

**Position.** Models attend unevenly across a long context — the middle of a long context is where
information goes to die. Retrieval put the right document at rank 7, the assembler put it in the middle
of twelve chunks, and the model effectively did not read it. Test: reorder so the top-ranked chunks sit
at the beginning and the end, or simply cut k and see if quality *improves*. Quality improving when you
give the model less is the signature of this failure and it feels wrong the first time you see it.

**Too much context.** Related but distinct. More retrieved context is not better; it is more surface for
the model to be distracted by. Passing 20 chunks when 5 would do dilutes the signal, and a single highly
plausible but wrong chunk can dominate the answer. This is why precision matters in RAG and not just
recall — recall determines your ceiling, precision determines whether you reach it.

**Conflicting sources.** Two retrieved documents say different things, usually because one is outdated.
The model has no basis for adjudicating and will typically blend them or pick arbitrarily, and blended
answers are the worst kind because they are individually sourceable and jointly wrong. The fix is not in
the prompt — it is metadata (effective dates, version, status) surfaced in the context and used in
ranking, plus deprecating old documents rather than letting them accumulate.

**Chunks that are individually retrievable and jointly incoherent.** The retriever found the right
chunk. The chunk is the middle of a procedure, starting at step 4, referring to "the above table" that
lives in a different chunk. Relevant by every retrieval metric, useless to the model. This is what
parent-document retrieval and contextual retrieval exist to fix — search over small precise units, pass
the coherent larger unit to the model.

**Truncation.** Silent, and it is always the end of the context that gets cut, which is often where you
put your instructions. Worth checking explicitly because it produces the confusing pattern where quality
degrades only on long queries.

**The prompt itself.** The instruction may not tell the model to ground its answer in the provided
context, may not tell it what to do when the context is insufficient, or may conflict with the retrieved
content — a system prompt saying "be concise" against a question that needs the full procedure. And if
there is no explicit "say you don't know" path, the model will always produce something, because that is
what it was asked to do.

**Formatting.** Underrated. Chunks concatenated with no delimiters or source labels give the model no
way to tell where one document ends and the next begins, and no way to cite. Clear separators, source
identifiers, and metadata headers measurably help, and they cost nothing.

### How I'd prove which one

The clean experiment is an **oracle context test**. Take failing questions, hand-assemble the perfect
context — exactly the right passages, correctly ordered, nothing else — and ask the model. If it now
answers correctly, the fault is entirely in assembly and I have a bounded problem. If it still fails
with perfect context, the fault is the model or the prompt, and no amount of retrieval work will help.
That single test is the highest-information hour available in this investigation and it is worth
proposing by name.

After that, ablate: same questions with k=20 versus k=5, reordered versus original, with and without
source labels. Each is one variable against a fixed set.

### The follow-ups

*"Your retrieval metrics look great but users are unhappy. Reconcile that."* Usually the retrieval metric
is measuring the wrong unit. Recall@10 against document-level labels can be 95% while the specific
passage the user needed sits in a chunk you did not retrieve. Or the labels themselves are stale.
Or — most commonly — retrieval is genuinely fine and the problem is downstream, which is exactly the
scenario this question describes.

*"How would you detect this class of failure automatically?"* Groundedness scoring: is every claim in the
answer supported by the provided context? Ungrounded claims mean the model went outside the context.
Grounded-but-wrong means the context was wrong, which points back at retrieval or at conflicting
sources. Splitting those two apart in monitoring is what makes the signal actionable rather than just
alarming.

---

## 3. "How would you know whether improving embeddings actually improved the system? What would you measure before and after?"

### What is being tested

Whether you can evaluate a change rather than believe in it. This is the question that most cleanly
separates people who have shipped from people who have demoed, because it is impossible to answer well
without having thought about attribution. It is also, quietly, a question about whether you understand
that a component improvement is not a system improvement.

### The answer

I'd measure at three levels, because a change to embeddings can improve one and not the others, and
knowing which is the whole point.

**Retrieval, in isolation.** This is where an embedding change should show up first and most cleanly.
Against a labeled set of query-to-relevant-document pairs: recall@k at the k you actually use, nDCG@10
for ranking quality, MRR if there is typically one right answer. Recall@k is the one I'd lead with,
because it is the system's ceiling — if the right document is not in the context, nothing downstream can
recover, so a change that does not move recall is unlikely to move anything.

**End to end.** Answer accuracy or an LLM-judged faithfulness and helpfulness score on a fixed question
set, held constant so the only variable is the embedding model. This is the number that matters, and it
is routinely smaller than the retrieval gain — a 6-point recall improvement might buy 2 points of answer
accuracy, because the questions where retrieval was already succeeding do not improve and some of the
newly retrieved documents were not the bottleneck.

**Operational.** New embedding models are frequently larger and slower. Index build time, index storage
footprint, query latency at p50 and p95, and cost per query. A 3-point retrieval gain for 200ms of added
latency and double the index size is a real tradeoff, not a free win, and saying that unprompted is a
strong signal.

The thing I would insist on: **hold everything else fixed**. Same chunking, same k, same reranker, same
prompt, same question set. Embedding changes get evaluated alongside chunking changes constantly, and
then nobody knows which one did the work. One variable.

And I'd look at the **distribution of the change, not just the mean**. An embedding model that improves
average recall by 5 points while regressing your highest-traffic query category is a bad change wearing
a good number. I'd segment by query type — definitional, procedural, exact-identifier, conversational
follow-up — and specifically check whether exact-match cases got worse, because stronger semantic models
sometimes trade lexical precision for semantic breadth, and identifier lookups are where that hurts.

### The offline-to-online step

Offline evaluation earns you the right to run an online test, and no more than that. The online measures:
answer thumbs or explicit feedback, escalation-to-human rate, follow-up-question rate as a
dissatisfaction proxy, session abandonment, and task completion if the product has a completable task.
I'd run it as an A/B with a pre-registered primary metric and enough traffic to detect the effect size I
care about, and I'd write down what result would make me roll back before I started — because the
temptation to reinterpret a flat result as a win is strong and universal.

Two traps worth naming out loud, because interviewers like hearing them: the eval set the embedding
model was selected on cannot also be the eval set that validates it, or you are measuring your own
selection process; and re-embedding the corpus means the comparison is against a rebuilt index, so a
partial or failed rebuild will look exactly like a bad embedding model.

### The follow-ups

*"Your offline metrics improved and online metrics did not. What happened?"* Several honest
possibilities. Retrieval was not the bottleneck for real traffic even though it was for your eval set —
which means the eval set does not reflect production, and that is now the finding. Or the eval set is
stale. Or the improvement was concentrated in a query type that is rare in production. Or the online
metric is noisy and underpowered and you have not actually observed "no change," you have observed
"cannot tell" — which is a different thing and gets treated differently.

*"How do you build the labeled retrieval set in the first place?"* Cheapest path: mine production logs
for questions, and use the documents that produced answers users accepted as weak positives. Better:
have subject matter experts label a few hundred query-document pairs, which is a day of work and pays
back permanently. Synthetic generation — have a model write questions from each document — is a
legitimate bootstrap and a trap, because it produces questions whose vocabulary matches the document,
which is precisely the case retrieval is already good at. Real queries are messier and that mess is the
difficulty.

---

## 4. "A user asks a question that requires information from 5 different documents. How would you design retrieval and context construction for that?"

### What is being tested

Whether you can distinguish two problems that look identical and are not. There is a large difference
between *the answer is spread across five documents that a single query will find* and *you cannot know
which the fourth document is until you have read the third*. The first is a coverage problem. The second
is multi-hop, and it needs iteration.

Candidates who answer "I'd increase k to 20" have collapsed both into the first, and the interviewer will
usually follow up until they hit the second.

### The answer

First I'd ask which shape it is, because it changes the design.

**If the five are independently retrievable** — a comparison across five products, a summary of five
incidents, a policy that happens to be documented in five places — the problem is coverage and diversity,
not reasoning. The failure mode is that all five of your top slots go to near-duplicates of the same
document, because similarity search is not built to give you variety. So: retrieve wider and enforce
diversity. MMR or a similar redundancy-penalizing selection so the k you keep are different from each
other rather than merely individually good. Deduplicate aggressively — near-duplicate chunks are a
silent killer of coverage. If the query decomposes cleanly into subqueries, decompose it explicitly,
retrieve for each, and merge, which guarantees per-subquestion coverage instead of hoping one query
surfaces everything.

**If they are sequentially dependent** — you need document A to learn the identifier that lets you find
document B — no single retrieval pass will work, at any k, and saying that clearly is most of the answer.
This needs an agentic loop: retrieve, let the model reason about what it now knows and what it still
needs, retrieve again, with a cap on iterations and a budget. The cost is latency and unpredictability,
which is why I would not build it for a corpus where most questions are single-hop — I would route,
detecting multi-hop queries and sending only those down the expensive path.

**Context construction** matters as much as retrieval here and is usually undercooked. Five documents'
worth of chunks is a lot of context, and everything from question 2 applies with more force. Concretely:
group chunks by source document rather than interleaving them by score, so the model sees coherent units
instead of fragments from five places alternating. Label each with its source and any relevant metadata
— date especially, because five documents on one topic frequently disagree and recency is often the
tiebreaker. Put the most important material at the beginning and end. And keep it as small as it can be:
if 5 documents means 40 chunks, I would rather rerank hard down to the 10 chunks that carry the answer
than pass all 40 and hope.

Then the honest part: **tell the model what the task is.** Synthesizing across sources is a different
instruction from answering from a passage. If the answer requires comparison or aggregation, the prompt
should say so, should ask for per-source attribution, and should explicitly instruct the model to state
when sources conflict rather than silently reconciling them. Silent reconciliation of conflicting sources
is one of the most dangerous failure modes in RAG, because the output is fluent, sourced, and wrong.

### How I'd evaluate it

Standard recall@k is the wrong metric here and noticing that is worth points. What I care about is
**whether all five required documents made it into the context**, which is a per-question set-coverage
measure — did we retrieve the complete required set, not did we retrieve something relevant. I would
build a small labeled set of genuinely multi-document questions with their full required document sets,
and measure complete-coverage rate. A system at 90% recall@10 might be at 40% complete coverage, and
those two numbers describe very different products.

### The follow-ups

*"How do you know it's a multi-hop question before you answer it?"* You can classify it up front, cheaply
and imperfectly. Or you can let retrieval tell you: run a pass, and if the model reports the context is
insufficient, escalate to the iterative path. The second is more robust and slower. Either way it is a
routing decision, and routing decisions should be measured — how often do you route wrong, and what does
each error cost.

*"What if the five documents contradict each other?"* Then the correct answer is that they contradict,
presented as such with dates and sources. A system that resolves the contradiction invisibly is worse
than one that surfaces it, because it removes the user's ability to notice. This is a place where the
right product behavior is to expose uncertainty rather than hide it.

*"Cost and latency of the iterative approach?"* Multiple retrieval rounds plus multiple model calls, so
several times a single-pass query on both. Hence routing, hence caps, hence a budget the model can see
so it wraps up rather than being cut off mid-reasoning.

---

## 5. "It works perfectly with 10,000 documents. Now it has 1 million. What breaks first — and how do you redesign?"

### What is being tested

Whether you know that scale changes the *kind* of problem, not just its size. The naive answer is
"latency" and it is usually wrong. Vector search is sublinear; going 100× on corpus size is not what
kills you.

This is also a question where saying "it depends on the shape of the growth" is genuinely correct and
sounds sophisticated, because 100× more of the same documents and 100× more diverse documents break
different things.

### The answer

**What breaks first is retrieval precision, not latency.** With 10,000 documents there are few plausible
distractors for any query, so a mediocre retriever looks excellent — there is nothing to confuse it. At a
million, every query has hundreds of documents that are topically adjacent and substantively wrong. Your
top-5 fills with plausible near-misses, and the model dutifully answers from them. Nothing errors,
nothing slows down, and quality quietly collapses. This is the answer to "what breaks first" and it is
the one most candidates miss because it is not an infrastructure failure.

Underneath it are three specific mechanisms worth naming:

**Near-duplicates.** Large corpora accumulate versions, translations, templated documents, and copies. At
10k you have a handful; at 1M your top-10 can be ten copies of the same document and your effective
context is one document. Deduplication moves from housekeeping to load-bearing.

**Recall degradation in the ANN index.** Approximate search is approximate, and the gap between
approximate and exact widens with corpus size at fixed parameters. Your HNSW `ef_search` that gave 98%
recall at 10k gives noticeably less at 1M, and this is silent — the index returns results with confident
scores either way. Recall against an exact-search baseline on a sample is a thing you should measure
periodically and almost nobody does.

**Embedding space crowding.** In a small corpus the semantic neighborhoods are sparse. In a large one
they are dense, distances between the right answer and the plausible-wrong one compress, and pure dense
retrieval loses discriminative power exactly where it matters. This is why hybrid search stops being
optional at scale — lexical signal on identifiers, error codes, product names, and rare entities does not
degrade the way semantic similarity does.

**Then, second, the operational things.** Index build time goes from minutes to hours, which changes
reindexing from a routine operation to a scheduled event with a rollout plan. Memory: an HNSW index over
a million chunks with large embeddings is a serious RAM number, and once it exceeds one machine you are
in a distributed system with all that implies. Incremental updates that were fine at small scale start
degrading graph quality over time, so you need periodic rebuilds with snapshot-and-swap rather than
in-place mutation. And cost — of embedding a million documents, of storage, of re-embedding all of them
every time you want to change embedding models, which is the decision that becomes expensive to reverse.

### The redesign

Roughly in the order I would do it, cheapest and highest-leverage first:

**Stop searching everything.** The largest win at scale is usually not a better index, it is a smaller
search space. Metadata filtering, partitioning by tenant or product or document type, routing queries to
the relevant partition. Searching 50,000 relevant documents beats searching 1,000,000 on precision, on
latency, and on cost simultaneously. Most large-corpus RAG systems that work well are really several
medium-corpus systems with a router in front.

**Make the pipeline multi-stage.** At 10k you can retrieve 10 and generate. At 1M you want a cheap wide
first pass — hybrid, a few hundred candidates — then an aggressive rerank down to the handful you
actually pass. Reranking is optional at small scale and close to mandatory at large scale, because its
whole job is separating true matches from plausible near-misses and that is precisely the problem scale
created.

**Take deduplication and freshness seriously as pipeline stages.** Near-duplicate detection at ingest,
explicit document lifecycle with supersession, and reconciliation between source of truth and index so
deletes actually propagate. Stale documents in a 10k corpus are noise; in a 1M corpus they are a
substantial fraction of your retrieval results.

**Add caching where the distribution is skewed.** Query distributions are usually long-tailed with a
heavy head, so an exact-match cache on common queries and an embedding cache on documents pays for
itself. Semantic caching too, if you are careful with the threshold and scope the key per tenant — a
loose semantic cache serves confidently wrong answers to different questions, which is worse than a cache
miss.

**Re-tune the index for the new size, and verify it.** `ef_search`, `M`, probe counts, and quantization
choices that were right at 10k are not right at 1M. Product quantization or another compression scheme
starts to matter for memory. And I would measure recall against exact search on a sample rather than
assuming the parameters still hold, because that measurement is the only thing standing between you and
silent degradation.

### The follow-ups

*"How would you know precision degraded, given nothing errors?"* You would not, without an eval set that
grows with the corpus. This is the argument for eval sets being a maintained asset — when the corpus 100×s
and the eval set does not, the eval set is now measuring a system that no longer exists. I would add
questions specifically targeting the newly added regions of the corpus, and I would track score
distribution and result diversity in production as leading indicators.

*"Where does the money go at this scale?"* Usually embedding the corpus once, storing the index
continuously, and generation per query. Reranking is small per query and adds up. The expensive surprise
is re-embedding: changing embedding models means recomputing everything, which makes the initial choice
much stickier than it feels when you make it. Worth saying that out loud — it is a real architectural
commitment disguised as a config value.

*"When is a vector database the wrong answer entirely?"* When the corpus is small enough that exact
search is trivially fast, when the queries are structured and belong in SQL, when the real problem is a
knowledge graph shaped like relationships rather than similarity, or when the corpus fits in a context
window and the task needs global understanding of it. Naming the case where you would not build this is a
strong signal in either direction of scale.

---

## What all five have in common

Read together, the five questions are one question asked five ways: **can you reason about a system you
cannot fully see?**

Every one of them is unanswerable by someone who only knows the components. Every one of them is
straightforward for someone who has operated a system, because operating one teaches you that failures
are rarely where you would look first, that the middle of a pipeline is where things hide, and that the
difference between a guess and a diagnosis is an experiment you can name.

There is a version of preparation that treats this as five answers to memorize. That version fails,
because the interviewer will change one detail and watch it collapse. The transferable thing is the
shape: what changed, bisect rather than scan, make the hypothesis falsifiable, say how you would prove
it, and separate the fix that stops the bleeding from the fix that stops the recurrence.

The rest of it — the specific failure modes, the specific metrics — is in `39_rag_retrieval_augmented_generation`
for the mechanics and in `MODERN_QUESTION_BANK.md` §2 and §5 for the component-level questions these
build on.
