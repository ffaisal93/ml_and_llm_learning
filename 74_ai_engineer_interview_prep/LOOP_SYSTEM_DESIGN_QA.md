# ML / AI System Design Round — Question & Answer Bank

Fourteen design prompts with complete worked answers. These are the prompts that get asked in the ML/AI system design loop. Each answer is written so you can study it once and then reproduce its shape under pressure.

The most common failure in this round is spending forty-five minutes on model selection. In every one of these systems, about **20% of the work is AI and 80% is ordinary engineering**. That 80% is ingestion, chunking, permissions, idempotency, retries, queues, caches, schema migrations, integrations, monitoring, and on-call. Interviewers hire the person who can ship the whole thing. So a candidate who says "we'll use a good embedding model" and then spends thirty minutes on the ingestion pipeline, ACL propagation, and eval harness scores far higher than one who compares six embedding models and never says how a document gets into the index.

Every number below is either sourced, with links at the end, or marked as an assumption. In an interview, say "I'd assume about 200ms p50 for the reranker, and I'd measure it." That is better than stating a benchmark you invented.

---

## The framework (read once, then spend your time on the problems)

Six moves, in order. The bracket after each move is its share of a 45-minute round.

**1. Clarify, then commit [5 min].** Ask four to six *specific* questions. Do not ask "what's the scale." Ask "are permissions per-document or per-folder, and do they change hourly or monthly?" Then state your assumptions out loud and move on. Never ask more than six questions, because interviewers read endless clarification as stalling.

**2. Sketch the data flow before the model [5 min].** Say where data enters, where it rests, what shape it is in when the model sees it, where the output goes, and who consumes it. Draw the boxes. The model is one box.

**3. Name the AI surface precisely, and its boundaries [5 min].** Say exactly which decision the model makes. Then say out loud what you would *not* use a model for. "Permissions are enforced in the query filter, never by the LLM" is the highest-signal sentence available to you in this round.

**4. Build the offline path and the online path [10 min].** Ingestion and training form a batch system, so it needs backfills, versioning, and idempotency. Serving is a low-latency system, so it needs caching, timeouts, and fallbacks. The two paths must share feature and embedding logic, because otherwise you get skew.

**5. Evaluation before scale [8 min].** You need a golden set, an offline metric, an online metric, and a guardrail metric. If you cannot say how you would know the system got worse, you have not designed it.

**6. Failure modes and the one hard tradeoff [7 min].** Name the failures before the interviewer does. Then pick the real fork in the road, argue both branches, and say what evidence would flip you.

Two habits carry every answer. First, **quantify**: give QPS, document count, p95, and dollars per 1k requests. Second, **degrade gracefully**: say what serves traffic when the model is down, which is usually BM25, a cached result, or a rules baseline.

---

## 1. RAG for enterprise document search

### Q: "Design a RAG system for enterprise document search. It's multi-tenant, and permissions matter."

**Clarifying questions to ask first**

- **Are permissions per-document, per-folder, or per-field, and how often do they change?** Per-folder ACLs that change monthly let me denormalize labels onto chunks and refresh nightly. Per-document ACLs that change hourly force a live authorization check at query time against the source system, and that check adds a network hop into my p95.
- **Is tenant isolation a compliance requirement (separate indexes) or a logical one (filter on tenant_id)?** Compliance means one index per tenant. That changes my cost model from "one big index" to "10,000 small indexes", which makes small tenants expensive.
- **What's the document mix — clean HTML/Confluence, or scanned PDFs and spreadsheets?** Scanned PDFs mean an OCR stage and a whole class of extraction failures. Spreadsheets mean chunking by row groups instead of by tokens.
- **Do users expect answers with citations, or a ranked list of documents?** Generated answers need hallucination guardrails and a much heavier eval harness. A ranked list is a search problem, and I might not need an LLM in the response path at all.
- **What's the freshness SLA — is a document searchable within seconds of upload, or is nightly fine?** Seconds means a streaming ingestion path with a write-ahead queue. Nightly means a much simpler batch job.
- **Corpus size and QPS?** 1M chunks and 5 QPS is a single Postgres box with pgvector. 5B chunks and 5k QPS is a sharded distributed index with a very different cost story.

**Assume:** 5,000 tenants; largest has 20M documents, median has 3,000. ~500M chunks total. Per-document ACLs sourced from the customer's IdP, changing on the order of hours. Mixed content, 30% scanned. Users want cited answers. Freshness SLA of 5 minutes. Peak 300 QPS. Target p95 of 3 seconds to first token.

**The design.**

There are two systems here, and they share nothing but a schema: an **ingestion plane** and a **query plane**.

Ingestion starts at connectors for SharePoint, Google Drive, Confluence, and S3. Each connector runs a cursor-based incremental sync. It publishes one change event per document to Kafka, partitioned by tenant, so one huge tenant cannot starve the others. Candidates skip this part, but it is most of the work. Connectors need OAuth token refresh, rate-limit backoff, deleted-document tombstones, and resumable cursors so a crash does not force a full re-crawl.

A parsing worker consumes the event, fetches the bytes, and routes by file type. Native PDFs go through a text extractor. Scanned PDFs go to OCR. I'd emit a `parse_confidence` score and quarantine anything below the threshold to a dead-letter queue with a human review UI, because silently indexing garbage OCR is how you poison a corpus.

Then chunking. I'd chunk on structural boundaries such as headings and sections, with a target of about 500 tokens and about 15% overlap. Before embedding, I'd prepend the document title and the heading path to each chunk. That header injection is cheap and it beats raw chunk text on retrieval quality. The reason is simple: a chunk that says "the limit is 30 days" means nothing without "Refund Policy > Enterprise Tier" attached to it.

Each chunk is embedded and written to the index with a metadata payload of `tenant_id`, `doc_id`, `acl_hash`, `source`, `updated_at`, and `chunk_ordinal`. I'd use a hybrid index, which means dense vectors plus BM25. Real enterprise corpora are full of product codes, error strings, and acronyms, so lexical matching catches things that dense embeddings miss entirely. Published hybrid comparisons show that fusing BM25 and dense results with reciprocal rank fusion beats either one alone by roughly 1–7% nDCG, depending on the domain, and the largest gains come on jargon-heavy corpora.

**Embedding is content-addressed**, and this matters. The cache key is `hash(chunk_text + model_version)`. Re-syncing an unchanged document therefore costs zero embedding dollars. At 500M chunks, that is the difference between a \$50k re-index and a \$0 one.

```
CONNECTORS ──> Kafka (part. by tenant) ──> PARSE/OCR ──> CHUNK ──> EMBED (cached)
                                               │                        │
                                          quarantine DLQ                v
                                                              ┌──────────────────┐
                                                              │ Hybrid index     │
                                                              │ HNSW + BM25      │
                                                              │ + metadata filter│
                                                              └──────────────────┘
                                                                       ^
 USER ──> auth ──> ACL resolve (cached) ──> hybrid retrieve (filtered) ─┘
                                               │
                                          rerank top-100 -> top-8
                                               │
                                          LLM synth (stream, cited) ──> USER
```

Now the query plane. A request arrives with a user identity. First I resolve that user's authorization set, which is their group memberships and their accessible ACL hashes. I read it from a cache keyed by user with a short TTL of about 5 minutes, refreshed asynchronously from the IdP. **The ACL filter is applied inside the vector search as a pre-filter, not as a post-filter, and never by the LLM.** Post-filtering is a correctness bug, because you ask for 100 candidates and get 3 back after filtering, and it is also a latency bug. However, pre-filtering with a selective predicate degrades HNSW recall. So for tenants with very restrictive ACLs I'd over-fetch, requesting $k = 500$ in order to return 100, and I'd monitor a filtered-recall metric.

Retrieval works like this. Hybrid search returns about 100 candidates, fused with RRF. A cross-encoder reranker then cuts those 100 down to the top 8. Reranking is the highest-leverage quality knob in RAG, and I'd budget 150–250ms for it. That budget is an assumption, and I'd measure it on my own shortlist size. Then synthesis: an LLM receives the 8 chunks with their IDs, and it is instructed to answer only from the context and to cite chunk IDs. A post-processor then validates that every cited ID actually appeared in the context, and it drops uncited claims where it can.

Caching sits at three layers: the ACL sets, the embedding of the query string, and a full response cache keyed by `(normalized_query, acl_hash)`. The ACL hash in that key is what makes response caching safe in a multi-tenant permissioned system.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is in exactly three places: the embedding model, the reranker, and the answer synthesizer. That is all. I would **not** use a model for permissions, because permissions are always a filter predicate. I would not use a model to route queries to tenants, because that is a lookup. I would not use a model to decide document freshness, because that is a timestamp. And I would not use a model for query understanding in v1. No LLM query rewriting until I have data showing it helps, because rewriting adds a serial LLM call to every request and it often hurts precision on short keyword queries. I'd also not use an LLM to decide *whether* to retrieve. A cheap classifier is fine, and always retrieving is also fine.

**The hard tradeoff** — *Denormalized ACL labels on chunks vs. live authorization at query time.*

Denormalizing means every chunk carries an `acl_hash`. Retrieval is then one fast filtered search and p95 stays low. However, a permission revocation is not effective until the reindex lands, and that gap is a real data-leak window. Live authorization means calling the source system to check access on the retrieved results. It is always correct, but it adds 50–300ms, it creates a hard dependency on a system you do not control, and it breaks pre-filtering entirely.

I'd take denormalized labels plus a fast revocation path. Revocations go to a high-priority Kafka topic processed within seconds, separate from the normal sync lane. What changes my mind is the regulatory posture. If this serves healthcare or defense customers, where a five-minute leak window is unacceptable, I flip to live checks and I pay the latency. Probably I'd run a hybrid: denormalized filtering narrows the set, and a live check validates the final 8.

**How I'd evaluate it.**

Offline, I'd build a golden set of 300–500 real queries per major tenant with labeled relevant chunks. I'd measure **recall@100** for the retrieval stage, because recall@100 is the ceiling and nothing downstream recovers a missed chunk. I'd measure **nDCG@8** after reranking. Separately I'd keep a permissions test suite. For each of N user/document pairs with known access, it asserts that the document never appears for unauthorized users. That suite runs on every deploy, and a single failure blocks the release.

For end-to-end answer quality I'd use an LLM-as-judge rubric. It scores groundedness, meaning is every claim supported by a cited chunk, plus completeness and citation validity. I'd calibrate the judge against about 200 human-labeled examples, so I know its agreement rate before I trust it.

Online, I'd track click-through on citations, the session-level reformulation rate, explicit thumbs, deflection rate if this feeds support, and p50/p95 time-to-first-token. Reformulation is a strong negative signal, because a user who rephrases did not get the answer. My guardrail metrics are the rate of answers with zero citations and the rate of "I don't know." A spike in either one means retrieval broke.

**Failure modes I name before the interviewer does.**

- **Stale ACLs after a revocation** — the leak window above; mitigated by the priority lane and a nightly full reconciliation.
- **Pre-filter recall collapse** — a user with access to 0.01% of a tenant's corpus gets almost nothing relevant back from HNSW, because the graph traversal wanders into inaccessible regions. Mitigation: route highly selective queries to an exact or IVF path, or to a per-user partition.
- **OCR garbage silently indexed** — quarantine on low parse confidence, and monitor the share of chunks with unusual character distributions.
- **Noisy-neighbor tenant** — one customer syncing 20M documents starves everyone. Use per-tenant partitions and rate quotas on the ingestion lane.
- **Embedding model upgrade** — you cannot mix embedding versions in one index. It needs a dual-write, a shadow index, and a cutover. For 500M chunks that is a multi-day, five-figure operation. Design for it on day one by putting `model_version` in the index name.
- **Chunk-boundary answer loss** — the answer straddles two chunks and neither one scores. Mitigated by overlap plus a step that fetches the neighbors of the top hits.
- **Prompt injection from documents** — a document that says "ignore previous instructions." Treat retrieved text as untrusted data, never as instructions, and keep tool access out of this path entirely.

**Follow-ups they will ask.**

*"How do you handle a document that's 400 pages?"*
Chunk it structurally and index every chunk. Also generate a document-level summary embedding, so that "what is the master services agreement about" retrieves the document rather than a random clause. At query time, if several top hits come from one document, I collapse them and expand the context inside that document instead of showing eight near-duplicate chunks. For very long documents I'd add a hierarchical layer: section summaries indexed alongside the leaf chunks, so retrieval can land on a section and then drill down.

*"A user says the answer is wrong. How do you debug it?"*
The trace is the product. Every request logs the query, the resolved ACL hash, the retrieval candidates with scores, the rerank scores, the final chunks, the prompt, and the completion, all under a request ID the user can quote. Debugging is then a decision tree. Was the correct chunk in the index at all, which is an ingestion bug? Was it retrieved in the top 100, which is a retrieval bug? Did it survive reranking, which is a rerank bug? Or did the LLM ignore it, which is a synthesis bug? Each branch has a different owner and a different fix. Without this trace you are guessing, and most reports of "the LLM hallucinated" turn out to be ingestion failures.

*"How do you keep costs down at 300 QPS?"*
The embedding cache keyed by content hash kills re-embedding costs. The response cache on `(query, acl_hash)` typically absorbs 20–40% of enterprise search traffic, because people ask the same policy questions. That absorption rate is an assumption, so measure it. Use a small model for synthesis and reserve the large model for queries the reranker flags as low-confidence. Cap the retrieved context at 8 chunks, because the marginal quality of chunks 9–20 is near zero while the token cost grows linearly. Track dollars per resolved query as a first-class dashboard metric, not as a quarterly finance exercise.

*"Why not just put everything in a 1M-token context window?"*
Cost and latency both scale with context length, and long-context recall degrades in the middle of the window. But the real reason is permissions. You cannot put a tenant's whole corpus into a prompt when different users see different subsets. You would need a per-user assembly step, and that step is retrieval with extra work attached. Long context is a good *complement* instead: retrieve at the document level, then put a handful of whole documents in the window instead of fragments.

*"How do you support 'find me the latest version of X'?"*
That is a metadata query wearing a semantic query's clothes. I'd detect version and recency intent, then apply a sort or a recency boost on `updated_at`. I'd also keep a document-lineage graph so superseded versions are demoted. Pure vector search has no notion of "latest", because the embeddings of v3 and v7 of a policy are nearly identical, so the model cannot help here. This is a good example of the 80%: the fix is metadata plumbing, not a better model.

*"How would you add a new connector?"*
Put it behind a stable interface of `list_changes(cursor)`, `fetch(doc_id)`, `get_acl(doc_id)`, and `resolve_user_groups(user)`. Everything downstream is shared. The work per connector is auth, pagination, rate limits, and ACL model translation. That last one is the hard part, because every system models permissions differently: SharePoint uses inheritance, Drive uses sharing links, Confluence uses spaces. I'd normalize all of them to a principal-set model and write conformance tests per connector.

*"What if a tenant has 20M documents and 3 users?"*
The index is oversized for the traffic. I'd tier the storage. Hot tenants sit on memory-resident HNSW. Cold tenants sit on disk-based indexes in the DiskANN family, or they are built lazily on the first query with a warmup penalty. Cost per tenant should be roughly proportional to corpus size for storage and to QPS for compute. Conflating those two is how RAG products lose money on enterprise accounts.

---

## 2. Customer support agent that takes actions

### Q: "Design a customer support agent that can actually take actions — issue refunds, change account settings, cancel subscriptions."

**Clarifying questions to ask first**

- **What's the maximum blast radius of a single action — a \$20 refund or a \$20,000 wire?** Small and reversible means the agent can act on its own with a post-hoc audit. Large or irreversible means every action goes through human approval or a hard policy engine, and the agent becomes a drafting tool.
- **Do we have an existing rules-based refund policy, or is it tribal knowledge in agents' heads?** A documented policy becomes a deterministic policy engine, and then the LLM only extracts arguments. If the policy is tribal, my first project is codifying it. I'd say that out loud, because that is the actual work.
- **Is the agent user-facing or agent-facing (a copilot for human reps)?** Copilot-first is much lower risk. It also gets you the training data and the eval set you need to justify going autonomous later.
- **What does the downstream billing system look like — does it have idempotency keys?** If it does not, my retry story is broken from the start, so I need an idempotency layer before I write a line of agent code.
- **What's the regulatory surface — chargebacks, GDPR deletion, financial disclosure requirements?** Regulated actions need mandatory disclosure text and immutable audit logs with retention.
- **What fraction of contacts do we want to deflect, and what's the cost of a wrong action vs. an escalation?** This is the objective function. If a bad refund costs 50x an unnecessary escalation, I tune the confidence threshold hard toward escalation.

**Assume:** consumer subscription business, 40k contacts/week. Actions: refund (up to \$200), cancel subscription, change plan, update email/address, resend receipt. A documented refund policy exists. User-facing chat, with human escalation available. Billing is Stripe-like with idempotency key support. Goal: deflect 50% of contacts with a wrong-action rate under 0.1%.

**The design.**

Here is the core architectural claim. **The LLM decides *what* to do and extracts *arguments*. A deterministic policy engine decides whether the action is *allowed*. A durable workflow engine *executes* it.** Those are three separate components with three separate failure modes. Collapsing them into "the LLM calls the Stripe API" is the design that puts you in a headline.

Now the flow. A message arrives over chat. Before any model runs, we load context deterministically: user ID, subscription state, payment history, prior refunds in the last 90 days, open tickets, and account flags. This is a plain service call, because the agent should never have to figure out who the user is.

Then the orchestrator runs. I'd use a tool-calling loop with a strong model and a small set of tools, split into two classes. **Read tools** such as `get_order`, `get_subscription`, and `search_help_center` are free to call, unlimited, with no approval. **Write tools** such as `issue_refund`, `cancel_subscription`, and `change_email` are proposals, not executions. The model calls `propose_refund(order_id, amount, reason_code)`, and what comes back is a decision from the policy engine, not a completed refund.

The policy engine is ordinary code. It checks that the refund amount is ≤ \$200, that the order is within 30 days, that there are fewer than 3 refunds in 90 days, that the subscription is not already cancelled, and that the user is not flagged for abuse. It returns `ALLOW`, `DENY(reason)`, or `NEEDS_APPROVAL`. It must be code and not a prompt, for three reasons: it is auditable, it is testable with unit tests, and it does not change behavior when someone tweaks a system prompt. When the policy engine returns `DENY`, that reason goes back to the model so the model can explain it to the user. The model handles the conversation. The engine handles the decision.

Execution goes through a durable workflow such as Temporal, not an inline HTTP call. Every action gets an idempotency key derived from `hash(conversation_id, action_type, args)`. This matters more than anything model-related, because without it a timeout plus a retry equals a double refund, and double refunds are how these projects get cancelled. The workflow handles retries with backoff, compensating actions on partial failure, and a terminal state written to an immutable audit log.

```
USER ──> chat gateway ──> context loader (deterministic)
                                 │
                                 v
                        ┌──────────────────┐   read tools (free)
                        │ LLM tool loop    │<─────────────────────> order/sub/KB APIs
                        └──────────────────┘
                                 │ propose_action(args)
                                 v
                        ┌──────────────────┐
                        │ POLICY ENGINE    │  deterministic rules, unit tested
                        │ ALLOW/DENY/APPR  │
                        └──────────────────┘
                        ALLOW │        │ NEEDS_APPROVAL
                              v        v
                   ┌───────────────┐  human queue ──> rep approves
                   │ Temporal wf   │
                   │ idempotent    │──> Stripe / account svc ──> AUDIT LOG (immutable)
                   └───────────────┘
```

Escalation is a first-class path, not an error path. It triggers on a policy `DENY` for something the user clearly wants, on sentiment and frustration signals, after three turns without progress, on any mention of legal action, chargebacks, or a regulator, and on model confidence below the threshold. Escalation hands the human a *summary plus the full trace*, so the rep does not ask the user to repeat themselves. Repeating yourself to a human after talking to a bot is the biggest driver of CSAT damage from bad handoffs.

The conversational part is grounded by retrieval over the help center and the policy docs, so answers cite real policy instead of the model's memory of it. The RAG hygiene is the same as in problem 1, but the corpus is much smaller.

Rollout is staged, and I'd volunteer this without being asked. Stage one is shadow mode: the agent proposes, humans act, and we measure agreement. Stage two is copilot: the agent drafts, the rep clicks approve. Stage three is autonomous for the narrowest and most reversible action only, which is resend receipt. Stage four widens action by action, as long as the wrong-action rate holds. Each stage gates on measured numbers, not on impressions.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI does three things here. It understands the user's intent and messy phrasing, it extracts structured arguments from the conversation, and it writes the reply. That work is hard and valuable.

I would **not** use a model to decide policy eligibility. I would not use it to compute refund amounts, because that is arithmetic on order lines and belongs in a function. I would not use it to authenticate the user, to decide escalation thresholds, which is a rules layer over signals, or to write to any system directly. I would also give it no free-form SQL or shell tools. The tool surface is a fixed, small, typed set. In every one of these places a model's 99% accuracy becomes a 1% incident rate on money.

**The hard tradeoff** — *autonomous action vs. human-in-the-loop approval.*

Autonomous action gets you real deflection, such as a refund in 30 seconds at 2am, and that is the ROI that funds the project. However, at 40k contacts/week, even a 0.5% wrong-action rate is 200 bad refunds a week, plus the trust damage. Human approval drives the error rate near zero, but it caps throughput at rep capacity, which means you have built an expensive autocomplete.

I'd go autonomous only for actions that are **cheap and reversible**, and I'd gate on a measured per-action error rate from shadow mode. Refunds under \$50 with a clean account history are autonomous. Everything else needs approval. What changes my mind is the shadow-mode data. If agreement with human reps is above about 98% on a class of action, and the residual errors are recoverable, I widen. If the errors cluster in a way I cannot predict from features, I do not widen, no matter how good the average looks.

**How I'd evaluate it.**

Offline, I'd replay a regression suite of a few hundred real conversations. I'd measure **action accuracy**, meaning did the agent propose the same action a senior rep would. I'd measure **argument accuracy**, meaning the right order ID and the right amount. I'd measure **policy adherence**, meaning did the agent ever propose something the engine rejected, because a high rate there means a prompt problem. And I'd measure **escalation precision and recall**. I'd also keep an adversarial set: users trying to talk the agent into refunds, prompt injection attempts, and ambiguous multi-order cases.

Online, I'd track **deflection rate**, meaning resolved without a human. The guardrail is the **wrong-action rate**, measured by sampling audited actions and by tracking reversals and chargebacks. I'd also track CSAT split by bot-resolved versus escalated, handle time for escalated conversations, and repeat-contact rate within 7 days, because a resolution that generates a second contact is not a resolution. For dollar metrics I'd watch refund dollars per contact against a pre-launch baseline. A model that resolves everything by refunding everything looks great on CSAT and destroys margin.

**Failure modes I name before the interviewer does.**

- **Double execution** on a retry or a duplicate message — solved by idempotency keys, and this must be tested with deliberate fault injection.
- **Social engineering** — "I'm the account owner's spouse, just change the email." Fix: identity verification is a hard precondition on write tools, enforced outside the model.
- **Prompt injection via user-supplied content** — an order note containing instructions. Treat all retrieved and user text as data, and never let it expand the tool set.
- **Policy drift between the engine and the prompt** — the prompt says 30 days, the engine says 14, so users get told yes and then denied. Fix: generate the policy summary in the prompt *from* the engine's config, so there is a single source of truth.
- **Refund-maximizing behavior** — the model is rewarded on CSAT, so it learns that yes is always easier. Watch refund dollars per contact as a guardrail metric.
- **Silent partial failure** — the subscription is cancelled but the refund failed. Use compensating transactions in the workflow, plus alerting on any workflow stuck in a non-terminal state.
- **Escalation black hole** — a handoff at 3am with no rep online. This needs explicit expectation-setting and a callback commitment.

**Follow-ups they will ask.**

*"How do you stop it from being talked into a refund it shouldn't give?"*
Two layers. First, the policy engine makes it structurally impossible to exceed the limits, whatever the model believes. Persuasion cannot change a rule evaluated in code against the account's real history. Second, I detect manipulation patterns as features, such as repeated reformulation after a denial, escalating emotional language, and claims that contradict account data, and I route those conversations to humans. Prompt-level defenses like "do not be persuaded" are the weakest layer, so I would not rely on them at all. I mention them last on purpose.

*"How do you handle a user with three orders who says 'refund the broken one'?"*
Disambiguation is a conversation design problem. The agent should present the candidates with distinguishing details such as date, item, and amount, and then ask. It should not guess. I'd hard-block the write path when argument confidence is low. The `propose_refund` tool requires an explicit `order_id`, so a resolver that returns multiple matches forces a clarifying turn. Guessing wrong here is the most common source of wrong-action incidents, and it is cheap to prevent.

*"What's your latency budget?"*
For chat, the first token should arrive within about 1.5s and the full reply under 5s, which keeps the conversation feeling live. Those numbers are an assumption, so tune them to your CSAT data. The tool loop is the risk, because each read tool call adds a round trip. So I'd parallelize independent reads, prefetch the obvious context before the model's first turn, and cap the loop at about 5 iterations before forcing either a reply or an escalation. Actions execute asynchronously. The user gets "processing your refund now" immediately and a confirmation when the workflow lands, instead of staring at a spinner through a Stripe retry.

*"How do you version and roll out prompt changes?"*
Prompts are code. They live in the repo, they are code-reviewed, versioned, and pinned per deployment. Every change runs the offline regression suite, and the diff report is part of the PR. Rollout is a canary: 5% of traffic, then watch wrong-action and escalation rates for 24 hours, then widen. Every conversation logs the prompt version and the model version, so an incident can be traced to a specific change. Untracked prompt edits in a web console are the most common cause of "it worked last week."

*"What if the LLM provider has an outage?"*
Degrade in tiers. First, fail over to a secondary provider or a smaller self-hosted model and accept reduced quality. The abstraction layer that makes this possible is worth building early. Second, if all models are down, fall back to a deterministic intent classifier plus templated flows for the top intents, which covers maybe 30% of volume. Third, queue the rest and route to humans with an honest wait-time message. What I would *not* do is retry a hanging provider until the queue backs up. Circuit-break at the client with a hard timeout.

*"How would you measure whether it's actually saving money?"*
Not by deflection rate alone, because that is the metric that gets gamed. I'd run a holdout: a randomized 5% of contacts routed to humans only, permanently. Then I'd compare fully loaded cost per resolved contact against that control. Fully loaded means model tokens, infra, the human time on escalations, and the cost of wrong actions and reversals. I'd compare 7-day repeat-contact rate and CSAT too. A holdout costs a little revenue and it is the only way to get a causal number, so I'd fight to keep it.

*"Where does the training data for improvements come from?"*
The escalation queue is the best source. Every escalation is a labeled example of "the agent could not do this", and the rep's resolution is the target. I'd instrument the rep tool to capture the action taken and a short reason code. Then I'd mine the disagreements between agent proposals and rep actions in shadow and copilot mode. That dataset drives prompt improvements, few-shot example selection, and eventually fine-tuning of a smaller, cheaper model on the high-volume intents.

---

## 3. LLM-powered code review assistant

### Q: "Design an LLM-powered code review assistant for our monorepo."

**Clarifying questions to ask first**

- **Is this replacing human review or augmenting it?** Augmenting means I optimize for precision, so a few high-value comments. Replacing means blocking gates and a much higher bar, and I'd argue against that for v1.
- **Monorepo size and PR volume?** 500 PRs/day at 2k lines each is a very different cost problem than 30 PRs/day. The answer decides whether I can afford a large model on every diff.
- **What do reviewers actually complain about — style, bugs, missing tests, or architectural drift?** Style is a linter's job, not a model's. If the complaint is "we miss null-deref bugs", the answer is a static analysis and LLM hybrid. If it is "people don't follow our internal patterns", the answer is retrieval over the codebase.
- **Can the assistant see the whole repo, or just the diff?** Diff-only reviews produce confidently wrong comments about functions the model cannot see. Repo access means building a code index and a symbol resolver, and that is most of the work.
- **What's the tolerance for false positives?** This is the make-or-break number. Developers abandon a bot after roughly two or three bad comments, so I'd target ≥70% of comments being actioned.
- **Are there existing CI signals (tests, coverage, linters, SAST) I can condition on?** A failing test in the diff's blast radius is a much better trigger for a comment than a model's suspicion.

**Assume:** 4,000-engineer monorepo, ~800 PRs/day, median diff 180 lines, p95 2,000 lines. Go/TypeScript/Python. Existing CI with linters, type checkers, and a SAST tool. Goal: augment, never block. Target ≥70% of posted comments marked useful or acted on.

**The design.**

Here is the framing that wins this question. **A code review assistant is a precision problem, not a capability problem.** The model can find plenty of issues. The product dies if 40% of its comments are noise. So the architecture is mostly about *suppression*.

The trigger is a webhook on PR open and on each push. First comes a cheap deterministic gate, before any model runs. It skips generated files, lockfiles, vendored code, pure-formatting diffs, and PRs over a size threshold. An oversized PR gets "this PR is too large for useful automated review" instead of a hallucinated summary. This gate alone removes a large share of the volume and the cost.

Context assembly is the real engineering, and it is where I'd spend most of the design time. For each changed hunk I want the full changed function, not just the diff lines. I also want the definitions of the symbols the hunk calls, the callers of the changed function, the file's tests, and any repo conventions relevant to the touched area. That requires a code index. I'd build one with tree-sitter parsing plus a language server or a `scip`/LSIF-style symbol index, updated incrementally on merge to main. Retrieval is symbol-graph traversal first and embeddings second. For code, "find the definition of `chargeCard`" is an exact lookup, so a vector search for it is strictly worse. Embeddings are for the fuzzy query, such as "how do we usually handle idempotency here", retrieved from a corpus of internal design docs, ADRs, and exemplar code.

Then review generation. I'd fan out per hunk instead of sending the whole diff. A single 2,000-line prompt produces vague comments, while per-hunk prompts produce specific ones. Per-hunk also parallelizes, which keeps wall-clock under a couple of minutes. Each hunk prompt gets the assembled context plus a rubric restricted to the categories where models are good: logic errors and edge cases, missing error handling, concurrency and resource-leak issues, security-relevant patterns, missing test coverage for new branches, and violations of retrieved repo conventions. The rubric explicitly excludes formatting, naming preferences, and anything a linter already checks.

The model must emit structured output: `{file, line, category, severity, claim, suggested_fix, confidence, evidence_symbols}`. Free-form prose comments cannot be filtered.

Then comes the suppression pipeline, which is what makes this shippable:
1. **Deduplicate** against comments already posted on this PR and against unresolved comments on prior pushes.
2. **Cross-check with deterministic tools** — if the model claims a null deref and the type checker says the value is non-nullable, drop the comment. Static analysis is a much better arbiter than a second LLM call.
3. **Self-verification pass** — a second model call sees the claim plus the cited evidence and answers "is this definitely true given only this evidence?" It is cheap, and it kills a large share of confident nonsense.
4. **Confidence and severity threshold**, tuned per repo from the feedback loop.
5. **Volume cap** — at most 5 comments per PR, ranked by severity. A bot that leaves 30 comments is ignored whatever its quality.

```
PR webhook ──> cheap gate (size, generated, lockfiles) ──> hunk splitter
                                                              │ per-hunk (parallel)
                 code index (tree-sitter + symbol graph) ──> context assembly
                 conventions/ADR vector index ────────────────┘
                                                              v
                                                      LLM review per hunk
                                                              │ structured findings
                        SUPPRESSION: dedupe -> tool cross-check -> self-verify
                                    -> threshold -> cap at 5 -> rank
                                                              v
                                          post as PR review comments (+ 👍/👎)
                                                              │
                                                     feedback store ──> threshold tuning, evals
```

Every comment carries a reaction affordance. We log resolved and unresolved state, and whether the author changed the code in response. That feedback store is the asset, because it is how thresholds get tuned per team and how you build the eval set.

Now cost control. At 800 PRs/day with about 6 hunks each, that is roughly 5k model calls/day. I'd route by risk. A cheap model does a first pass, and only the hunks it flags get escalated to the expensive model. Hunks touching sensitive paths such as auth, payments, and migrations get escalated too. I'd cache by `hash(hunk + context)`, so re-pushes do not re-review unchanged hunks.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

The AI reads code and reasons about intent versus implementation. That is the one thing that was previously impossible.

I would **not** use a model for formatting, because gofmt and prettier do that. Not for lint rules, because existing linters do that deterministically. Not for type errors, because the type checker does that. Not for known CVEs in dependencies, because a scanner does that. Not for test execution or coverage measurement either. I'd also not let the model *block* a merge, because a nondeterministic gate on the critical path of 4,000 engineers is an availability and morale disaster. And I would not have it auto-apply fixes without an author click. Suggested diffs yes, auto-commit no.

**The hard tradeoff** — *review every PR shallowly vs. review a subset deeply.*

Reviewing every PR with a cheap model and tight context gives broad coverage and low cost. However, the comments are generic and precision suffers, and poor precision is exactly what kills adoption. Deep review, with full symbol context, an expensive model, and self-verification, produces genuinely valuable catches. However, it costs maybe 10x per PR, so it cannot cover everything.

I'd go deep on a risk-scored subset. That subset is PRs touching payments, auth, and data-migration paths, PRs from engineers with under 90 days' tenure, PRs with no tests added despite new branches, and PRs where CI is already unhappy. Everything else gets the cheap pass. What flips me is measured precision. If the cheap pass reaches a 60% or higher actioned-comment rate, universal coverage is worth more than depth, because then the value is in the volume of small catches.

**How I'd evaluate it.**

Offline, I'd build a benchmark from your own history. Take merged PRs that were later reverted or followed by a hotfix within 7 days, and check whether the assistant flags the offending hunk. That is a real "did it catch the bug" metric grounded in your codebase, and it is far more convincing than a public benchmark. I'd complement it with a labeled set of about 200 hunks with known injected bugs for regression testing. I'd also keep a **false-positive set** of clean, idiomatic code where the correct output is zero comments. Precision on that set is the number I'd watch.

Online, I'd track the **actioned-comment rate**, meaning the author edited the code near the comment or resolved it as useful, with a target of ≥70%. I'd also track the 👎 rate, comments per PR, and time-to-first-human-review, which tells me whether the bot's pass makes humans faster or slower. For the counterfactual I'd measure escaped-defect rate as reverts and hotfixes per 100 PRs, compared against a holdout set of repos where the bot is off. I'd keep that holdout for at least a quarter.

**Failure modes I name before the interviewer does.**

- **Confidently wrong comments about unseen code** — "this can be null" when the caller guarantees otherwise. Mitigation: symbol context, and dropping any claim whose evidence symbols were not in the prompt.
- **Nitpick flood** — the model comments on style because it always has something to say. Mitigation: rubric restriction and the volume cap.
- **Duplicate comments across pushes** — infuriating for the author. This requires stable comment identity across rebases, which is fiddly. Anchor on the hunk content hash, not on the line number.
- **Leaking secrets or proprietary code to a third-party API** — this needs a self-hosted model or a vendor with contractual guarantees, plus secret scanning before the prompt is built.
- **Gaming** — engineers split PRs to slip under the size gate, or ignore the bot entirely. Watch adoption per team. A team with a 0% action rate is a signal to investigate, not to nag.
- **Index staleness** — the symbol index lags main, so the context is wrong after a big refactor. Monitor index lag, and degrade to diff-only review with a visible caveat when the index is stale.
- **Cost blowup on a mega-refactor PR** — a 50k-line rename generates thousands of calls. Use a hard per-PR spend cap.

**Follow-ups they will ask.**

*"How do you handle a 3,000-line PR?"*
I do not review it hunk by hunk exhaustively, because that is expensive and it produces noise. I'd triage instead. Classify hunks as mechanical, meaning renames, generated code, and import churn, versus substantive, using cheap heuristics plus a small model. Review only the substantive hunks. Then post a top-level comment saying which files got real review and which were skipped. Being honest about coverage preserves trust. For genuinely large refactors the more useful output is a structural summary, such as "this changes the retry semantics in 12 call sites, 3 of which lack tests", rather than line comments.

*"How do you teach it our internal conventions?"*
Retrieval, not fine-tuning, at least at first. I'd index ADRs, style guides, and past human review comments. Past review comments are the most valuable, because they are the conventions as actually enforced. When reviewing a hunk, retrieve the most similar past review comments on similar code and include them as few-shot examples. This adapts per team automatically, and it updates the moment a new convention doc lands. Fine-tuning is worthwhile only once you have tens of thousands of accepted comments and want to shrink the model for cost.

*"Won't developers just ignore it?"*
Yes, if precision is bad. That is why precision is the primary metric rather than recall. Beyond quality there are three adoption levers. Post comments before human reviewers are assigned, so the author fixes issues privately, which is much less socially costly than a public nit. Let teams opt into categories. Never block merges. I'd also publish per-team dashboards of the bot's action rate, so teams can see whether it helps them, and let a team turn off any category that is not earning its keep.

*"How does this interact with security review?"*
It complements SAST, and it does not replace it. A model should not be your control for OWASP categories that a deterministic tool covers with better recall. The model adds value on context-dependent security, for example "this endpoint reads `org_id` from the request body rather than the session, so it's an IDOR." Catching that requires understanding the auth pattern used elsewhere in the repo. I'd route sensitive-path diffs to a dedicated security rubric with a lower confidence threshold, which buys higher recall and accepts more false positives. I'd route those comments to the security team's queue rather than to the author, because the false-positive tolerance there is different.

*"What about test generation?"*
It is adjacent to review and, in my experience, higher value than review comments. Given a hunk with new branches and no test coverage delta, generate a test, run it in a CI sandbox, and post it only if it passes and actually increases coverage. That execution loop is the whole trick, because an unexecuted generated test is a liability. It also gives you a clean, objective reward signal. That makes it a much better candidate for automated improvement than review comments, where "correct" is subjective.

*"How do you keep the code index fresh in a monorepo with 500 merges a day?"*
Index incrementally on merge, keyed by file, and update the symbol graph only for the affected files and their reverse dependencies. Run full rebuilds nightly as a correctness backstop. I'd track index lag as a p95 metric and alert above a few minutes. For a monorepo this size the index is a real distributed system: sharded by directory, with a serving layer that answers "definition of symbol X" in single-digit milliseconds. This is the 80% again. The reviewer's quality is bounded by index quality, and index quality is plain infrastructure work.

---

## 4. Semantic search / recommendations for a marketplace

### Q: "Design semantic search and recommendations for a marketplace — think eBay or Etsy scale."

**Clarifying questions to ask first**

- **Is inventory unique (one-of-a-kind listings) or replenishable (many identical SKUs)?** Unique inventory means brutal cold-start on every item and a churning index. Replenishable inventory means stable item embeddings and rich per-item interaction history.
- **What are we optimizing — GMV, conversion rate, or buyer retention?** Optimizing clicks gets you clickbait listings. Optimizing GMV biases toward expensive items. The honest objective is usually a weighted blend with a return-rate penalty.
- **Two-sided constraints — do we need seller fairness or new-seller exposure guarantees?** If yes, that is an explicit constraint in the ranker or in a re-ranking step. It is not something you hope will emerge.
- **How much of search traffic is navigational ("nike air max 270 size 10") vs. exploratory ("gift for someone who likes camping")?** Navigational traffic is a lexical and faceted problem, where semantic search can actively hurt. Exploratory traffic is where embeddings earn their keep.
- **Latency budget and QPS?** Search at p95 200ms with 10k QPS constrains the whole architecture. It forces a two-stage retrieve-and-rank design with a strict per-stage budget.
- **Is there a paid-placement/ads business?** Ads turn the ranker into a joint auction-and-relevance problem, which is a materially different design.

**Assume:** 200M active listings, 60% unique/one-of-a-kind. 8k search QPS peak, 40M DAU sessions/week. Objective: GMV per session, with a relevance guardrail. New-seller exposure is a stated business goal. p95 budget 250ms. Ads exist but are a separate slot allocation.

**The design.**

The standard pipeline is correct here: **candidate generation → filtering → ranking → policy re-ranking.** The mistake is trying to do all of it in one stage with one clever model.

*Indexing.* Every listing produces a document with lexical fields, which are title, description, structured attributes, and category path, plus a multimodal embedding. Image matters enormously in a marketplace, because half of the "does this look like what I want" signal is visual. So I'd embed title, attributes, and the primary image into a shared space with a CLIP-style dual encoder, ideally fine-tuned on your own click and purchase pairs. That fine-tuning is the highest-ROI model work in this whole system. An off-the-shelf embedder does not know that in your marketplace "vintage" means pre-1990 and "boho" is a real category.

Ingestion has to handle 200M listings with high churn. New listings must be searchable within a minute, because sellers watch this obsessively, and sold-out listings must vanish immediately. So the index is a streaming write path with a fast delete lane, plus nightly reconciliation against the source of truth to catch drift.

*Candidate generation* runs several retrievers in parallel, and each one returns about 500 candidates:
- **BM25/lexical** over title and attributes — non-negotiable for navigational queries, brand names, and model numbers.
- **Dense ANN** over the multimodal embedding — HNSW for the hot set. This is what handles "gift for someone who likes camping."
- **Collaborative/behavioral** — item-to-item co-purchase and co-view neighbors, plus a two-tower user-embedding retriever for recommendations. This surfaces things text and images cannot.
- **Business retrievers** — trending in category, and the new-seller exposure pool.

The results are fused with RRF or a learned blend into about 1,000 candidates. On real hybrid comparisons, fusing lexical and dense beats either one alone by a few points of nDCG. One published e-commerce comparison shows RRF at 0.7068 nDCG against 0.6983 for BM25 alone and 0.6953 for dense alone, and it rises to 0.7497 with field boosting. The lesson is not the exact number. The lesson is that domain-specific field boosting beat the fancy part.

*Filtering* is hard business logic, and it happens after retrieval. It covers shipping availability to the buyer's country, price and size facets, prohibited items, blocked sellers, and out-of-stock items. It is deterministic, and getting it wrong is worse than any ranking error, because showing an item that cannot ship is a guaranteed bad session.

*Ranking* is a gradient-boosted tree or a small DNN over about 200 features, scoring the roughly 1,000 candidates in under 40ms. The features are the query-item relevance scores from each retriever, item quality signals such as seller rating, photo quality, return rate, and dispute rate, price relative to the category median, recency, personalization from the user's category affinities, price band, and past sellers, and context such as device and session position. I'd use a multi-objective model that predicts $P(\text{click})$, $P(\text{purchase} \mid \text{click})$, and expected value, combined as roughly

$$\text{score} = P(\text{click}) \cdot P(\text{purchase} \mid \text{click}) \cdot f(\text{price}) - \lambda \cdot P(\text{return})$$

In words: the score is the chance the user clicks, times the chance a click turns into a purchase, times a function of the price that captures the value of that purchase, minus a penalty for the chance the item comes back. I tune $\lambda$ so we do not optimize our way into a returns problem.

*Policy re-ranking* is the last stage, and it is pure code. It handles diversity, meaning no more than 3 listings per seller in the top 20, the new-seller exposure quota, ads slot interleaving, and dedup of near-identical listings. Business constraints belong here, explicitly and tunably, not buried inside a loss function.

```
QUERY ─┬─> BM25            ─┐
       ├─> dense ANN (HNSW) ─┤
       ├─> behavioral i2i    ├─> fuse (RRF) ~1000 ─> FILTER (ship/stock/policy)
       └─> business pools   ─┘                            │
                                                          v
                        feature store (online) ──> RANKER (GBDT, ~40ms)
                                                          │
                                       policy re-rank (diversity, quota, ads)
                                                          v
                                                        RESULTS ──> logging
```

Logging is the product's future. Log every impression with its position, every click, and every purchase, together with the exact feature vector used at scoring time. Without logged features you cannot train an unbiased next-generation ranker, and reconstructing them later is impossible.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI sits in the embedding model, the behavioral retrievers, and the ranker. Note that **there is no LLM in the request path.** An LLM at 8k QPS and 250ms p95 is neither affordable nor fast enough, and it does not beat a well-tuned GBDT on ranking.

I'd use LLMs offline, where they are excellent. They generate structured attributes from messy seller descriptions, which materially improves faceting and lexical matching. They produce synthetic query-item training pairs for the embedder, normalize category taxonomies, and write listing quality assessments. I would **not** use a model for stock, shipping eligibility, or price. For prohibited-item enforcement a model can assist, but the block itself must be a rule plus human review. And I would not replace BM25 with pure vector search. That is the classic marketplace regression, where someone searches an exact part number and gets vibes.

**The hard tradeoff** — *personalization depth vs. result consistency and cold start.*

Heavy personalization lifts engagement for known users. However, it wrecks the experience for the roughly 40% of sessions that are logged-out or new. It also makes results unreproducible, so support cannot debug "I saw it yesterday." And it creates filter bubbles that suppress inventory discovery, which in a two-sided marketplace hurts seller supply.

I'd personalize the *ranker* moderately, using session context and coarse affinities, but I'd keep candidate generation mostly query-driven. Then the same query returns a broadly similar set for everyone, with reordering on top. What changes my mind is the A/B data. If deep personalization lifts GMV per session by more than a couple of percent *without* degrading new-seller exposure or long-tail impression share, I'd push further. I'd want that seller-side metric in the readout, not just the buyer-side one.

**How I'd evaluate it.**

Offline, from logged interactions, I'd measure **recall@1000** for candidate generation, because that is the ceiling, and **nDCG@10 and MRR** for the ranker. Purchases are the strongest relevance label and clicks are weak labels. I'd correct for position bias with inverse propensity weighting, because naive offline evaluation on logged clicks reliably overstates gains. The reason is that the old ranker chose what got seen. I'd add a human-rated relevance set of a few thousand query-item pairs as a guardrail, because a ranker can improve GMV while getting less relevant.

Online, I'd start with interleaving experiments, because they give fast and sensitive relevance comparisons and need far less traffic than an A/B test. Then I'd A/B on the real objective: **GMV per session**, conversion rate, and search-abandonment rate. My guardrails are relevance rating, return rate, new-seller impression share, the seller Gini coefficient, and p95 latency. I'd run for at least two weeks to catch novelty effects.

**Failure modes I name before the interviewer does.**

- **Popularity feedback loop** — the ranker promotes what is popular, that makes it more popular, and the long tail dies. Mitigation: exploration slots, propensity-weighted training, and monitoring the impression-share distribution.
- **Cold-start listings** — a unique item with no interaction history has no behavioral signal. Mitigation: content-based embeddings carry it at first, plus an explicit exploration budget for new listings.
- **Query drift on seasonal terms** — "boots" means something different in December. Embeddings are static, so the fix is recency features and periodic retraining, not a better encoder.
- **Stale index showing sold items** — the worst buyer experience. Use the fast delete lane plus a final availability check before render.
- **Ads cannibalizing relevance** — organic quality degrades quietly as ad load rises. This needs a fixed relevance floor for ad slots and a monitored organic-relevance metric.
- **Training-serving skew in features** — the offline pipeline computes "seller rating" over a different window than the online one. This is the most common source of "great offline, flat online." Solved by a shared feature definition; see problem 12.
- **Multi-lingual and misspelled queries** — a huge share of real marketplace traffic. This needs spell correction and query translation before retrieval, because embeddings alone do not fix "addidas."

**Follow-ups they will ask.**

*"How do you handle 'gift for my dad who likes fishing'?"*
That is an exploratory query. Lexical retrieval fails completely on it, so dense retrieval does most of the work. I'd add an offline-built query understanding layer. For the head and torso of exploratory queries, which are surprisingly repetitive, I'd precompute an expansion of LLM-generated category and attribute hints and cache it. Then at serve time it is a lookup, not a generation. For the true tail I'd rely on dense retrieval alone. I'd also detect exploratory intent with a cheap classifier and shift the fusion weights toward the dense retriever and toward diversity, because for a gift query the user wants breadth, not the 20 closest matches.

*"How do you evaluate the embedding model in isolation?"*
Build query-item pairs from purchase logs. A query that led to a purchase is a positive, and sampled non-purchased impressions are hard negatives. Then measure recall@k of the embedder alone on held-out purchases from a later time window. Use a temporal split, never a random one, or you leak. I'd also keep a small human-labeled set for semantic categories the logs will not cover. And I'd always compare against BM25 as a baseline, because an embedder that does not beat BM25 on your domain is not worth its serving cost, and that outcome is more common than people admit.

*"What's your reindexing story when you change the embedding model?"*
Use a dual index. Build the new index in a shadow cluster while the old one serves, backfilling 200M embeddings at whatever throughput the budget allows, which is probably a couple of days. Then shadow-evaluate: replay live traffic against both indexes and compare recall and ranker features offline. Cut over per region behind a flag, with instant rollback. One point matters most here. The ranker consumes retriever scores as features, so a new embedder shifts the feature distribution. Therefore the ranker needs retraining on data from the new retriever. Otherwise you see an unexplained ranking regression that has nothing to do with the ranker itself.

*"How do you get recommendations for a brand-new user?"*
Context is all you have, and it is more than people think. It includes the entry point, meaning which category page or referral, plus geography, device, time of day, and the first click in the session. I'd serve a popularity-by-context baseline and personalize it within the session from the first one or two interactions, using a lightweight session-based model such as a GRU or a transformer over the click sequence. Session-based recommenders are the right tool here precisely because they need no user history. I'd also not be too clever. For a logged-out first-time visitor, "best-selling in this category, high seller rating" is a strong baseline that many personalized models fail to beat.

*"How do you prevent the ranker from just learning 'cheap items get clicked'?"*
That is the multi-objective problem. A click-only objective does find price as the dominant feature. I'd train against purchase and post-purchase outcomes, meaning delivered, not returned, and not disputed. I'd weight by margin or by GMV, depending on the business objective, and add the return-rate penalty. Then I'd monitor the price distribution of served results against the inventory distribution. If the served median drifts well below the inventory median, the ranker has found the cheap-item shortcut. I'd also examine feature importances, and if needed apply monotonicity constraints in the GBDT so price cannot dominate arbitrarily.

*"Do you need a GPU to serve this?"*
Not for the ranker. A GBDT over 1,000 candidates runs on CPU in tens of milliseconds, and it is cheaper and more debuggable there. GPUs are needed for the embedding model, but only for indexing, which is batch and offline, and for encoding the query at serve time. Query encoding is a single short forward pass. I'd either run it on a small GPU pool with batching or use a distilled query encoder on CPU, and I'd cache the embeddings for head queries, which covers a large share of marketplace volume. Reserving GPUs for the ranker would spend the expensive resource on the part that needs it least.

---

## 5. Evaluation system for an LLM product

### Q: "Design an evaluation system for an LLM product. How do you know when it got worse?"

**Clarifying questions to ask first**

- **What's the unit of success — a single response, or a whole task/session?** Per-response scoring misses agents that take ten correct steps and then fail. Session-level evaluation needs trajectory scoring, and it is much harder to build.
- **Is there a ground-truth signal in the product (a resolved ticket, an accepted suggestion, a completed purchase), or is quality purely subjective?** A real outcome signal changes everything, because it lets you evaluate continuously on production traffic instead of on a frozen test set.
- **What changes underneath us — our prompts, our retrieval corpus, or the vendor's model?** Vendor model updates are the sneaky one. Nothing in your repo changed and behavior moved. That needs a continuous canary, not just pre-deploy gates.
- **What's the cost tolerance per eval run?** If a full run costs \$500 and takes 3 hours, it cannot gate every PR, so I'd need a fast tier and a slow tier.
- **Who adjudicates disagreement — is there a domain expert available, and how many hours per week?** Human labeling capacity is the real constraint on eval quality. I'd rather design around 4 hours a week of an expert than pretend I have unlimited labeling.
- **Do we need per-segment guarantees (by language, customer tier, topic)?** Aggregate metrics hide segment regressions, and a big customer's segment breaking is what actually causes escalations.

**Assume:** an LLM support/answering product, ~50k requests/day. Weak outcome signal available (ticket resolved without escalation, thumbs). We control prompts and retrieval; the model is a vendor API that updates. One domain expert, ~5 hours/week. Need per-language and per-top-customer segment views.

**The design.**

Four layers, cheapest and fastest first. The organizing idea is this: **most regressions should be caught by assertions, not by judges.** Judges are for the fuzzy residue.

**Layer 1 — Assertions (deterministic, milliseconds, free).** For every response, run mechanical checks. Is the JSON valid when structured output is expected? Does every citation ID exist in the retrieved context? Is there PII in the output? Are the required disclaimers present when the topic triggers them? Is the length within bounds? Is there a refusal on benign input, or leaked prompt text? These checks run in CI and *on every production response* as an online guardrail. In my experience they catch a large majority of real regressions, because most regressions are structural. A prompt edit breaks the output format, or a retrieval change drops citations. Assertions are also the only layer you can afford to run on 100% of traffic.

**Layer 2 — Golden set with reference answers (minutes, cheap).** This is 200–500 curated examples with expert-written reference answers, stratified across intents, languages, difficulty, and known past failures. Every production incident becomes a permanent golden case. That is the ratchet that stops regressions from recurring. Scoring uses exact or fuzzy match where possible, and an LLM judge with a reference where not. This tier gates every PR.

**Layer 3 — LLM-as-judge on production samples (hourly, moderate cost).** Sample a few hundred production interactions per hour. Sample them stratified, not uniformly, because uniform sampling drowns you in easy cases. Score them with a rubric judge on groundedness, helpfulness, tone, and task completion. The judge must be **calibrated**. I hold out about 200 human-labeled examples and measure the judge's agreement with the expert using Cohen's kappa. If kappa is below about 0.6, the judge is not trustworthy, so I fix the rubric before I trust its trend. I'd also use pairwise comparison rather than absolute scoring where possible. Judges are much more reliable at "is A better than B" than at "rate this 1–5", and absolute scores drift over time and across judge model versions.

**The judge model and its prompt are pinned and versioned.** This matters, because if the judge changes then all historical scores are incomparable. When I upgrade the judge, I re-score a historical window with both judges to establish a conversion.

**Layer 4 — Human review (weekly, expensive).** The expert's 5 hours go to the highest-information cases: judge–assertion disagreements, low-confidence outputs, user-reported failures, and a small random sample for unbiased calibration. Their labels feed back into judge calibration and into the golden set.

```
                 ┌──────────────── production traffic (50k/day) ───────────┐
                 │                                                          │
        L1 ASSERTIONS (100% of traffic, ms) ──> alert on rate change        │
                 │                                                          │
        stratified sampler ──> L3 JUDGE (hourly, ~300/hr) ──> metric TSDB   │
                 │                                    │                     │
                 └──> L4 HUMAN QUEUE (5 hr/wk) <──────┘ disagreements       │
                              │                                             │
                              └──> golden set / judge calibration <─────────┘
                                              │
   PR ──> CI: L1 + L2 GOLDEN (gate, ~5 min) ──┘        CANARY: 5% traffic, 24h
```

**Detecting "it got worse" is a change-detection problem, not a threshold problem.** Each metric is a time series segmented by version, language, intent, and customer tier. I'd apply CUSUM or a simple sequential test per segment. The comparison is *this version against the previous version on the same traffic mix*, because a shift in traffic mix alone will move aggregate metrics. Alerts fire on statistically significant degradation, not on crossing an arbitrary line.

There are two things I'd build that people forget. First, a **frozen canary set replayed against the vendor's model daily**: the same 100 prompts at the same temperature-0 settings, diffed against yesterday's outputs. When the vendor silently updates, this is the only thing that tells you. Second, **shadow evaluation**: new prompt versions run against a mirror of live traffic without serving, so you get a full distribution comparison before a single user is exposed.

The release process works like this. A PR gates on L1 and L2. A merge deploys behind a flag. Then a canary runs at 5% for 24 hours, with automated comparison of L1 rates, L3 scores, and product metrics. Rollback is automatic on an assertion-rate regression, and it is a human decision on a judge-score regression, because judge noise is real.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is the judge in layer 3, and nothing else. Layer 1, layer 2's matching logic, the sampling, the statistics, the alerting, and the rollback are all ordinary software.

I would **not** use an LLM to decide whether to roll back, because that is a statistical test on metrics. I would not use one to generate the golden set unsupervised, because LLM-generated test cases are systematically easier than reality and give false confidence. I'd use them only as drafts for an expert to edit. I would not use a model to score anything an assertion can check deterministically. And I would not use one to summarize the eval results into a single "quality score", because a single number hides segment regressions. I'd also never let the same model family judge its own output as the sole signal without human calibration.

**The hard tradeoff** — *fast, cheap gates on every PR vs. thorough evaluation that's too slow to gate.*

A 5-minute gate keeps velocity, but it only catches structural breaks. It will pass a prompt change that makes answers subtly less helpful. A 3-hour, \$500 full evaluation catches quality drift, but it cannot run per PR, so regressions land and are found later.

I'd do both, deliberately. Cheap gates block merges, and the thorough run happens nightly and on the canary. The regression window is then up to 24 hours. That is acceptable, because the canary is only 5% of traffic and rollback is one click. What changes my mind is severity. For a product where a bad answer is a compliance event, I'd move the thorough evaluation pre-merge and pay the velocity cost, probably by shrinking the thorough set with stratified sampling until it fits in 20 minutes.

**How I'd evaluate it.** (Yes — the eval system needs evaluation.)

I'd track five meta-metrics. First, **judge–human agreement**, measured as kappa and tracked over time per segment. Second, **regression detection recall**: inject known-bad prompt variants and measure what fraction the system catches before the canary ends. Third, **false alarm rate**, meaning how often it flags a regression that a human then confirms is noise. Above about 20% and engineers start ignoring alerts. Fourth, **time-to-detection** for the vendor-update canary. Fifth, **golden set coverage**: what share of production incidents in the last quarter were already represented by an existing golden case before they happened. That last number is the honest measure of whether your eval set reflects reality.

**Failure modes I name before the interviewer does.**

- **Golden set overfitting** — everyone tunes prompts against the same 300 cases until the scores are meaningless. Mitigation: a held-out set nobody sees, rotated quarterly, plus continuous addition from production.
- **Judge drift** — the vendor updates the judge model and every historical score shifts. Pin versions, and re-score overlap windows.
- **Aggregate metrics hiding segment collapse** — the overall score is flat while Japanese-language quality falls off a cliff. Require per-segment breakdowns with minimum sample sizes.
- **Survivorship bias in feedback** — thumbs-down comes disproportionately from users who cared enough to click. Never treat thumbs as the primary metric. Use the stratified sample for unbiased estimates.
- **Eval cost exceeding inference cost** — easy to hit if you judge everything with a large model. Sample, and use a smaller judge for high-volume tiers.
- **Slow-boiling drift** — quality declines 0.5% a week and no single comparison is significant. Mitigation: compare against a fixed baseline version from N months ago, not only against the previous release.
- **Nobody looks at it** — the most common failure. The eval dashboard has to sit in the release process and in the on-call runbook, or it rots.

**Follow-ups they will ask.**

*"How big should the golden set be?"*
Big enough that a change you care about is detectable. If the metric is a proportion around 0.8, and you want to detect a 5-point drop with reasonable power, you need on the order of a few hundred examples per segment. The binomial math sets that floor. So 300 total is fine for one aggregate number and completely inadequate for ten segments. I'd rather have 200 well-curated, hard, expert-labeled cases per key segment than 5,000 auto-generated easy ones. Composition matters more than size, so stratify by intent and difficulty and deliberately over-sample the hard tail.

*"How do you handle the case where there's no single right answer?"*
Use pairwise preference instead of absolute scoring. Generate responses from the candidate and the incumbent, present them blind to a judge or a human in randomized order, and compute a win rate. This sidesteps rubric calibration entirely, and it is far more stable across judge versions. I'd also decompose subjective quality into checkable sub-properties: factually grounded, addresses the question asked, right length, right tone. Each of those is much less ambiguous than "good."

*"The vendor pushed a model update and things feel off, but metrics look flat. What now?"*
Trust the report, then find the segment. Flat aggregates plus real complaints almost always means a segment or behavior-class regression. I'd diff the canary set outputs token by token to characterize *how* behavior changed: longer, more hedging, a different refusal boundary. Then I'd slice production metrics by every dimension I have. I'd also check metrics the eval does not cover, such as response length, latency, refusal rate, and tool-call frequency, because a behavior shift often shows there before it shows in quality scores. If the complaint is real and unmeasured, that becomes a new assertion and new golden cases.

*"How do you evaluate multi-turn conversations?"*
Evaluate at the session level with trajectory metrics: task completion, judged from the full transcript plus the outcome signal, turns-to-resolution, and recovery rate after an error. I'd also build the eval as a replay harness with a simulated user, meaning an LLM playing the user role from a persona and a goal. That lets me test multi-turn paths deterministically. However, simulated users are noticeably more cooperative than real ones. So I'd calibrate against real transcripts and treat simulated results as a relative signal between versions, not as an absolute quality estimate.

*"How much should you spend on evals?"*
As a rule of thumb I'd budget 10–20% of inference spend on evaluation, rising during major changes. That range is an assumption and a planning heuristic, not a measured law. The framing I'd use with a skeptical exec is this: the cost of one undetected regression running for a week across 350k requests dwarfs the eval bill. But I'd also keep eval spend visible and tiered, so the expensive layers can be dialed down in steady state and up during a migration.

*"What do you do on day one when you have no data?"*
Write 50 test cases by hand from the product spec, before writing the prompt. That forces you to specify what "correct" means, and it catches the worst failures immediately. Add assertions right away, because they need no labels. Then instrument production heavily from launch, review every failure manually for the first weeks, and let the golden set grow from real traffic. The trap to avoid is generating a synthetic eval set with an LLM and declaring victory at 95%. That number is meaningless, and it buys false confidence at exactly the moment you most need honest signal.

---

## 6. Real-time content moderation

### Q: "Design a real-time content moderation system for a large user-generated content platform."

**Clarifying questions to ask first**

- **Which modalities — text, images, video, audio, or all?** Video is a different cost and latency problem, because of frame sampling and transcoding, and it dominates the budget if it is present.
- **Is moderation pre-publish (blocking) or post-publish (takedown)?** Pre-publish puts the model on the critical path with a hard latency budget. Post-publish lets me use slow, expensive, accurate models, but it accepts exposure time.
- **What's the harm taxonomy and its legal structure?** CSAM and terrorist content have mandatory reporting and zero-tolerance handling. Spam is a nuisance. Harassment is contextual. These need entirely different pipelines, not one classifier with many labels.
- **What are the regulatory obligations — DSA, appeals, transparency reporting?** If there are any, I need per-decision audit records, an appeals workflow, and statistics reporting as core features, not as add-ons.
- **What's the relative cost of a false positive vs. a false negative, per category?** Wrongly removing a benign post is a user-trust cost. Missing CSAM is unbounded. Thresholds must be per category and derived from this answer.
- **Volume and peak burstiness?** Moderation load spikes hard around events, and a coordinated brigading attack is a 100x spike on one topic.

**Assume:** text and images, 50M posts/day (~600/s average, 3k/s peak). Post-publish for most categories with a pre-publish block for the highest-severity ones. Categories: CSAM/terror (zero tolerance), violence/gore, hate, harassment, adult, spam/scams, self-harm. DSA-style appeal obligations. Human review capacity ~2,000 items/hour.

**The design.**

The design is a **cascade**, ordered by cost. Human review is the scarce resource, and everything else exists to protect it.

**Stage 0 — Hash and rule matching (sub-millisecond).** Every image is hashed with PhotoDNA or PDQ-style perceptual hashing and checked against known-bad databases such as NCMEC and GIFCT. Text runs through exact and fuzzy matching against known scam templates and banned strings. This is not machine learning, and it catches a large share of the worst content instantly and deterministically. A candidate who jumps straight to an LLM classifier has missed the most important stage in the system. Hash matches on the zero-tolerance categories go straight to block-plus-report, with no model involved.

**Stage 1 — Cheap classifiers (~5–20ms, on everything).** These are small fine-tuned transformer classifiers: a distilled BERT-class model for text and an image classifier. Both are multi-label over the taxonomy, and both run on every item. They output calibrated probabilities per category. Most content is clearly benign and exits here. In parallel I'd compute non-content signals: account age, prior violations, posting velocity, device and IP reputation, and network features such as who shares this content. These behavioral features are often more predictive of spam and coordinated abuse than the content itself, because a brand-new account posting 200 identical links is spam whatever the text says.

**Stage 2 — Expensive model on the uncertain band (~200ms–2s, ~2–5% of items).** An item goes here when the cheap classifier lands in the ambiguous range, or when behavioral risk is elevated. Stage 2 is a large multimodal model with the full context: the post, the image, the parent thread, the community's norms, and the relevant policy text. This is where an LLM genuinely earns its cost, because the hard cases are contextual: reclaimed slurs, satire, news reporting of violence, medical discussion of self-harm. A classifier cannot read context. A model with the policy in its prompt can, and it can produce a rationale citing the specific policy clause. That rationale is exactly what a human reviewer and an appeals process need.

**Stage 3 — Human review (the scarce resource).** The queue is ordered by expected harm × uncertainty × reach. Reach matters enormously, because a borderline post with 2 views and one with 200k views are not the same priority. Reviewers see the item, the model's rationale, the policy clause, and the account history. Their decisions are the training data for everything upstream.

```
POST ──> S0 hash/rules (µs) ──match──> BLOCK + report (zero tolerance)
            │ no match
            v
    S1 cheap classifiers + behavioral features (~10ms, 100% of items)
            │                    │                       │
       clear benign          uncertain 2-5%         clear violation
         PUBLISH                  │                  auto-enforce
                                  v                       │
              S2 multimodal LLM + policy context (~1s)     │
                                  │                       │
                    ┌─────────────┴──────────┐             │
                 auto-enforce           HUMAN QUEUE <──────┘ (sampled)
                    │                     (2k/hr, priority = harm×uncertainty×reach)
                    v                        │
              enforcement svc <──────────────┘
                    │
        ┌───────────┴──────────┐
     user notice + APPEAL   audit log (immutable, DSA reporting)
```

The **enforcement service** is deliberately separate from detection. Detection says "this is 0.93 hate speech." Enforcement decides the action, which can be remove, demote, age-gate, label, warn, or suspend, based on category, confidence, account history, and jurisdiction. The same content can warrant different actions for a first-time user and for a repeat offender. Geography matters too, because content that is legal in one country is not legal in another. Enforcement is a rules engine, versioned, with every decision logged immutably.

**Appeals are a first-class pipeline**, not an afterthought. An appeal re-runs the item through the expensive model with the user's stated reason, then routes it to a human who did not make the original decision. Appeal outcomes are gold-standard labels. A high overturn rate on a category means your threshold is wrong, and that is the fastest quality signal in the system.

For the pre-publish block on zero-tolerance categories the budget is tight: a hash lookup plus the cheap classifier, under 50ms. On timeout, most categories fail open to review, and the worst categories fail closed.

Throughput engineering matters as much as the models. That means Kafka ingest, autoscaled classifier workers with GPU batching, and backpressure that degrades gracefully. Under a 3k/s spike I'd keep stages 0 and 1 at full coverage and sample stage 2, rather than letting the queue grow without bound.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is the cheap classifiers and the contextual multimodal model. That is real value, because nothing else scales to 50M items/day.

I would **not** use a model for known-bad content, because hashing is exact, instant, and legally defensible. I would not use one for the final enforcement decision on high-severity categories, because human confirmation is required there and the legal and ethical cost of an automated error is too high. I would not use one for jurisdiction rules, because that is a lookup table. And I would not use one for the appeals decision. I'd also resist a single LLM doing everything. At 50M items/day, even at \$0.10 per thousand items that is meaningful money, and it would be strictly worse than a distilled classifier on the easy 95%.

**The hard tradeoff** — *aggressive automated enforcement vs. human-confirmed enforcement.*

Aggressive automation removes harmful content in seconds and scales without headcount. However, every false positive is a user wrongly punished, and at 50M items/day a 0.1% false positive rate is 50,000 wrongful removals daily. Human confirmation makes errors rare, but it caps enforcement at about 2,000 items an hour and leaves harmful content live for hours.

I'd set thresholds per category from the actual harm asymmetry. Auto-enforce aggressively on spam and on hash-matched CSAM, because those errors are cheap or nonexistent. Require human confirmation on harassment and hate, because those are contextual and carry a high false-positive cost and high user-trust stakes. In the uncertain middle, use reversible actions such as demotion and age-gating rather than removal. Reversible enforcement is the underrated move, because it reduces harm exposure without the trust cost of a wrongful takedown. What changes my mind is the measured appeal overturn rate. If overturns on a category exceed a few percent, I raise the threshold or move that category to human-confirmed.

**How I'd evaluate it.**

Offline, I'd measure per-category **precision and recall at the operating threshold** on a stratified labeled set, including an adversarial slice of deliberately evasive content such as leetspeak, text inside images, and cropped memes. I'd report precision-recall curves per category, not one aggregate F1, because the operating points differ wildly. I'd also measure **calibration**: if the model says 0.9, is it right 90% of the time? Calibration matters more than raw accuracy here, because thresholds are the control surface.

Recall is the hard problem, because you cannot measure what you did not catch. The solution is a **random audit sample**: a few thousand items a day pulled uniformly from published content and reviewed by humans regardless of model score. That gives an unbiased prevalence estimate, meaning what fraction of live content violates policy. Prevalence, not classifier recall, is the metric leadership should care about, and it is the one used in public transparency reports.

Online, I'd track prevalence by category, time-to-action as p50 and p95 from post to enforcement, and views-before-removal, which is the actual harm exposure metric. I'd also track appeal rate and overturn rate, reviewer throughput and agreement rate, and user reports per 1k posts. My guardrails are the false-positive rate estimated from appeals, and the enforcement rate by demographic segment, which detects bias.

**Failure modes I name before the interviewer does.**

- **Adversarial evasion** — users adapt within hours of a new rule, using character substitution, text in images, and coded language. This needs continuous retraining and a fast-response path for new patterns, because a static classifier decays measurably within weeks.
- **Context collapse** — removing a news organization's report of violence, or a survivor's account of harassment. This is the biggest source of PR damage. Mitigated by context in stage 2 and by trusted-publisher allowlists.
- **Coordinated brigading spike** — 100x volume on one topic overwhelms both queues. This needs burst detection, correlated-content clustering so a reviewer sees one representative and the decision applies to the cluster, and preemptive rate limits.
- **Reviewer trauma and turnover** — a real operational failure mode with quality consequences. Design for it with rotation, blurring, and wellness limits on exposure to the worst queues.
- **Feedback loop bias** — the model is trained on reviewed items, and reviewed items are the ones the model flagged, so it never learns what it systematically misses. The only fix is putting the random audit sample into the training set.
- **Language and cultural coverage gaps** — performance is far worse outside English, and that is where the worst real-world harms have historically occurred. This needs per-language metrics with minimum sample sizes and explicit investment, not a multilingual model and hope.
- **Over-blocking a legitimate community** — a term that is a slur in one context and an in-group identifier in another. Use per-community threshold tuning and appeal monitoring by community.

**Follow-ups they will ask.**

*"How do you handle a brand-new harm type that appears overnight?"*
I'd use a fast path that needs no retraining. Rules and hash-based blocking are deployable within minutes by the policy team. Alongside that runs a nearest-neighbor lane: embed a handful of confirmed examples, then queue anything similar in embedding space for review. That gets meaningful coverage within hours from about 50 examples. Meanwhile human reviewers label aggressively, and a classifier retrain lands in days. The key architectural property is that policy changes are configuration, not deploys. If adding a new rule requires a model retrain, you will always be a week behind.

*"How do you decide the threshold for each category?"*
From the cost asymmetry, made explicit. I'd write down the cost of a false positive and of a false negative per category in comparable units: user-trust damage, harm exposure, and legal risk. Then I'd pick the threshold that minimizes expected cost on the calibrated probability. In practice that is a conversation with policy and legal. My job is to hand them the precision-recall curve and say "at this threshold you get 92% precision and 60% recall, at that one 75% and 85%; which errors do you want?" Then revisit quarterly, using appeal overturn rates as ground truth.

*"What about video?"*
Sample frames using scene-change detection plus a fixed rate, run image classification on those frames, transcribe the audio and run text classification on the transcript, and check the video's perceptual hash against known-bad. The expensive part is transcode and frame extraction, not inference. Live video is a different design. Sample more aggressively at the start, because most policy-violating streams declare themselves early. Escalate viewer-reported streams immediately. Weight by concurrent viewers, because a stream with 50k viewers needs sub-minute action.

*"How do you keep the human reviewers' labels consistent?"*
Treat labeling as a measured process. Use a shared rubric with worked examples, run regular calibration tests where every reviewer labels the same gold items, and track inter-annotator agreement per reviewer and per category. Disagreement above the threshold means the policy is ambiguous, not that the reviewers are bad, so it is a signal to rewrite the policy. I'd also route a fraction of items to multiple reviewers for ongoing agreement measurement, and never train on single-reviewer labels for categories with low agreement.

*"What's the cost structure?"*
The cost is dominated by stage 1 running on 100% of traffic and by human review, not by the fancy model. At 600/s average, cheap classifier inference on batched GPU is a modest fleet. Stage 2 at 3% of 50M/day is 1.5M expensive calls/day. That is the line item to watch, and it is controlled entirely by how wide the uncertainty band is. Human review at 2,000 items an hour with round-the-clock coverage is likely the largest single cost in the system. So the highest-leverage optimization is not a better model. It is narrowing the uncertainty band and clustering duplicate items, so reviewers see each distinct piece of content once.

*"How do you handle a regulator asking why a specific post was removed?"*
Keep immutable per-decision records: content hash, timestamp, model versions and scores, the policy clause cited, the enforcement rule applied, the reviewer ID if a human was involved, and the appeal history. Retention is set per jurisdiction. This is a straightforward but non-trivial data engineering requirement, because it means 50M structured audit records a day, queryable by content ID, kept for years. That requirement quietly dominates storage design. I'd build it from day one, because retrofitting audit logging onto an existing pipeline is painful and you never fully recover the historical records.

---

## 7. Personalized news / feed ranking

### Q: "Design a personalized news feed ranking system."

**Clarifying questions to ask first**

- **Is the content pool from followed sources only, or open (anything on the platform)?** A follow-graph feed has a small candidate set, so the problem is ordering. An open feed needs real retrieval over millions of items, and cold-start dominates.
- **What's the objective — time spent, or something like "meaningful interactions" / long-term retention?** Time spent is easy to optimize, and it is well documented to produce outcomes nobody wants. If the objective is retention, I need long-horizon evaluation and surrogate metrics.
- **Content lifetime?** News decays in hours, and evergreen content decays in weeks. That sets the retraining cadence and how heavily recency features weigh.
- **Are there editorial or integrity constraints — misinformation demotion, source diversity, political balance?** These are hard constraints in a re-ranking layer, and they need to be designed in, not bolted on.
- **How often does a user visit, and how much do they consume per session?** Someone opening the app 20 times a day needs dedup and freshness guarantees across sessions. A daily visitor needs a digest.
- **Latency and scale?** Feed loads are the highest-QPS surface on most platforms, and the p95 budget shapes everything.

**Assume:** open feed, 50M DAU, ~10M new items/day, average 8 sessions/user/day, 25 items consumed per session. Objective is 28-day retention, with weekly active engagement as the surrogate. Integrity constraints: demote low-quality/misinfo sources, cap source concentration. p95 budget 300ms for a feed load.

**The design.**

The skeleton is the same as the marketplace: retrieve, rank, re-rank. The interesting differences are **recency, repetition, and long-horizon objectives**.

*Candidate generation.* From a pool of about 10M fresh items I need about 1,000 candidates per request in tens of milliseconds. Several sources run in parallel:
- **Follow-graph / subscribed sources** — recent items from what the user follows.
- **Two-tower embedding retrieval** — a user tower over recent interaction history and a content tower over item text and media, searched with ANN. Item embeddings are computed once at publish. User embeddings are recomputed on interaction, or approximated by pooling recent item embeddings, which is much cheaper and nearly as good.
- **Collaborative signals** — items engaged with by users similar to this one, and items trending inside the user's topic clusters.
- **Fresh/exploration pool** — items too new to have signal, injected deliberately.

Pre-computation is essential at 50M DAU. For most users I'd precompute a candidate pool asynchronously, say every 15 minutes and on any significant interaction. The request path is then "fetch the precomputed pool, merge in the freshest items, rank." Doing full retrieval synchronously at feed-load QPS is the wrong cost curve.

*Filtering* is deterministic and it does heavy work here. It removes already-seen items, blocked sources, muted topics, region-restricted content, and integrity blocklists. The seen-store, which is a per-user Bloom filter or a dedicated store, is critical and unglamorous.

*Ranking.* I'd use a multi-task neural ranker that predicts several outcomes per item: $P(\text{click})$, $P(\text{dwell} > 30s)$, $P(\text{share})$, $P(\text{hide/report})$, and $P(\text{follow source})$. These combine into a single score with tuned weights:

$$\text{score} = \sum_i w_i \cdot P(\text{action}_i) \; - \; \sum_j v_j \cdot P(\text{negative}_j)$$

In words: add up the probability of each good action, weighted by how much we value it, then subtract the probability of each bad action, weighted by how much we dislike it. The negative terms carry real weight, because hides and reports are the strongest available proxy for "this feed is getting worse." A ranker without them optimizes straight into rage-bait. The features are user-item affinity from the towers, the user's topic and source affinities over 1-day, 7-day, and 90-day windows, item quality and source reputation, item age with an explicit decay, engagement velocity, and session context such as position in session, time of day, and device. Engagement velocity means how fast an item accrues engagement relative to its age, and it is the key freshness signal.

Recency needs care. A raw "engagement count" feature guarantees a rich-get-richer loop and buries new items. So I'd use velocity normalized by age and impressions, plus an explicit decay term $e^{-\lambda t}$. That term shrinks an item's score as it gets older, and I'd tune $\lambda$ per content type: hours for news, days for evergreen.

*Re-ranking* is where the constraints live, and it operates on the whole slate, not on one item at a time. It enforces source diversity, meaning no more than 2 items from one source in the top 10, plus topic diversity, integrity demotion multipliers for low-quality sources, ad interleaving, and an exploration quota of about 10% of slots for items the model is uncertain about. Slate-level optimization matters, because the marginal value of the fifth article on the same story is near zero even when each one scores well on its own. I'd also cluster near-duplicate stories, meaning the same news event covered by 40 outlets, and show one representative with a "more coverage" affordance.

```
publish ──> item embed + quality score ──> fresh item store (10M/day)
                                                  │
       ASYNC (every 15 min per user): retrieve ~2000 candidates ──> candidate cache
                                                  │
REQUEST ──> fetch cache + merge freshest ──> FILTER (seen store, blocks, region)
                                                  │
                       online feature store ──> MULTI-TASK RANKER (~1000 items)
                                                  │
                    SLATE RE-RANK: dedup stories, source cap, integrity demote,
                                   explore quota, ad interleave
                                                  v
                                         FEED ──> impression + interaction logs
```

Log every impression with its position and dwell time, every interaction, and the feature vector used. Position-bias correction is mandatory, because the top slot gets clicked whatever its quality. Training on raw clicks therefore teaches the model to predict position.

The retraining cadence is fast. User interests and news cycles move daily, so I'd retrain the ranker daily and update user embeddings near real time. Item embeddings are static per item.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is the two-tower retrievers, the item content embeddings, the multi-task ranker, and offline content understanding, which covers topic classification, quality scoring, and near-duplicate clustering.

I would **not** put an LLM in the request path. At this QPS it is economically impossible, and it adds nothing the ranker does not do better. LLMs go offline, where they generate item topic labels and summaries, cluster stories, and assess content quality signals. I would **not** use a model for the seen-store, the source diversity caps, the integrity blocklists, or the region restrictions, because all four are deterministic. And I'd resist letting the model implicitly set the objective. The weights $w_i$ are a product decision, made by humans and written down. They are not something the optimizer discovers.

**The hard tradeoff** — *optimizing engagement vs. optimizing long-term retention.*

Engagement metrics are measurable in hours and they move reliably. However, they are well documented to drift toward outrage, clickbait, and content users regret consuming. Retention is the objective you actually want, but it takes 28 days to measure, it has terrible statistical power, and it cannot gate weekly experiments.

I'd use engagement as the training signal and constrain it heavily. That means strong negative weights on hide and report, a survey-based "was this worth your time" signal collected from a small panel and used to train a quality model that feeds the ranker, and holdout experiments measuring long-horizon retention on a slower cadence. What changes my mind is the holdout data. If long-horizon holdouts show engagement-optimized variants winning on retention too, I'd loosen the constraints. In practice the two usually diverge. When they do, I'd trust the 28-day number and take the short-term hit. However, that is a leadership decision, and I'd want it made explicitly rather than by a loss function.

**How I'd evaluate it.**

Offline, I'd replay logged sessions with IPS or doubly-robust off-policy estimation, which corrects for the fact that the old ranker chose what was shown. I'd measure nDCG on engagement labels plus AUC per task head. Offline gains here are notoriously weakly correlated with online gains, so I'd treat offline results as a *filter* for obviously bad candidates, not as a decision.

Online, I'd run an A/B for at least two weeks to clear novelty effects. The primary metrics are sessions per user per day and 7-day retention. Secondary metrics are dwell time and meaningful interactions. The guardrails matter more than the primary metrics here: hide and report rate, source diversity measured as a Gini coefficient over sources shown, share of impressions to low-quality sources, share of feed from the top-10 sources, and the fraction of users whose topic diversity narrowed. I'd also keep a long-horizon holdout, meaning a persistent small cell on the old ranker, to measure cumulative effects that week-long A/Bs structurally cannot see.

**Failure modes I name before the interviewer does.**

- **Filter bubble narrowing** — the ranker exploits known interests and the user's topic diversity collapses over months. This is invisible in a two-week A/B and visible in a long-horizon holdout. Mitigation: exploration quota and diversity constraints.
- **Clickbait and rage-bait ascendance** — a direct consequence of a click-weighted objective. Mitigation: negative signals, dwell quality, and the quality model.
- **Stale feed for heavy users** — 8 sessions a day exhausts the candidate pool, so the user sees repeats. Mitigation: a robust seen-store, freshness quotas, and pool refresh triggered by consumption depth.
- **Cold-start for new users** — there is no history, so the two-tower user embedding is garbage. Mitigation: onboarding topic selection, a geo and context popularity baseline, and fast session-based adaptation.
- **Breaking news latency** — a major event happens and the feed is 20 minutes stale, because candidate pools are precomputed. This needs an event-detection bypass that injects high-velocity items into feeds immediately.
- **Coordinated manipulation** — actors game engagement velocity to get amplified. This needs an integrity layer that scores the *audience* of the engagement, not just its volume.
- **Position-bias feedback loop** — training on uncorrected clicks makes the model progressively more confident about whatever it already ranked highly.

**Follow-ups they will ask.**

*"How do you handle breaking news?"*
Use a separate real-time lane. An event detector watches for abnormal velocity in story clusters, looking at posting rate, engagement rate, and geographic spread. When it fires, it injects the representative item into the merge step of every relevant user's feed request, bypassing the precomputed pool. Relevance is determined by geography and topic affinity. I'd keep this lane deliberately conservative and rate-limited, because it is an obvious manipulation target. The entry criteria should include source credibility, and I'd want human editorial override available for the highest-reach injections.

*"How do you prevent the same story from 40 outlets filling the feed?"*
Cluster stories offline and continuously. Embed items and cluster them by content similarity within a time window, which produces a story ID. Then re-ranking treats the story as the unit. It picks the best single item per story for that user, meaning the best source for their preferences or the highest quality, and suppresses the rest behind a "more coverage" expansion. This is a good example of an offline LLM and embedding pipeline creating a lot of value invisibly. The user just experiences a feed that is not repetitive.

*"How do you serve 50M DAU with 300ms p95?"*
Precomputation and caching do the heavy lifting. Candidate pools are built asynchronously and cached. User features are precomputed in an online feature store with single-digit-millisecond reads. Item features are cached in memory on the ranking service, because 10M items a day is small enough to keep hot. The synchronous path is then fetch the pool at about 10ms, fetch user features at about 5ms, score 1,000 items with a small model at about 30ms on CPU with batching, re-rank at about 5ms, and assemble. The ranker must be small. This is a case where a 10x bigger model that is 2% better is not worth 10x the fleet.

*"What if a user says 'my feed got worse'?"*
An individual complaint usually has a specific mechanism behind it, so I'd instrument for that. I'd track per-user feed composition over time, meaning topic mix, source mix, and share of followed versus recommended items, and I'd build a diff view showing what changed. The common real causes are a single viral interaction pulling the user's embedding toward a topic they do not actually want, a seen-store failure causing repeats, or a source they liked being demoted by integrity. I'd also give users direct controls such as "show less of this", topic mutes, and "why am I seeing this." Those are good product, and they generate high-quality negative labels that are otherwise very hard to collect.

*"How do you weight the multi-task heads?"*
Not by intuition, and not by the optimizer. I'd calibrate the weights against the long-horizon objective. Run experiments with different weight vectors, measure 28-day retention, and fit the relationship. In practice you only get a handful of experiments' worth of signal, so the estimate is coarse. A useful discipline is to express the weights in interpretable units, such as "one share is worth as much as eight clicks", so product leadership can argue about them directly. Weights should be reviewed on a schedule, because the right tradeoff shifts as the platform and the content mix change.

*"How do you deal with the exploration cost?"*
Exploration slots have measurably worse immediate engagement, so the short-term cost is real and there is constant pressure to cut them. I'd defend the budget by measuring what exploration buys: the rate at which explored items become long-term engaged interests, and the effect on catalog coverage. I'd also make exploration smarter than random. Thompson sampling or an upper-confidence-bound rule over the ranker's uncertainty concentrates the budget where the information value is highest, which typically cuts the cost of exploration substantially compared with uniform random injection.

---

## 8. Internal LLM serving platform

### Q: "Design an LLM serving platform for internal teams — multiple models, multiple teams, and we need cost control."

**Clarifying questions to ask first**

- **Self-hosted open models, vendor APIs, or both?** Both means the platform's main job is abstraction and routing. Self-hosted only means it is a GPU capacity and scheduling problem, which is a very different system.
- **What's the workload mix — interactive chat, batch processing, or agentic loops?** Batch can be scheduled off-peak on cheap capacity. Interactive needs headroom and low latency. Agentic loops are bursty and can blow budgets fast, because one user request becomes fifty model calls.
- **Is cost control advisory (show teams their spend) or enforcing (hard quotas that cut them off)?** Hard quotas need a real-time accounting path and a well-designed failure mode, because cutting off a production service to save money is usually the wrong call.
- **What are the data-governance requirements — can any team send any data to a vendor?** If some data cannot leave, I need routing policies enforced at the gateway using data classification, not left to team discretion.
- **How many teams, and how sophisticated are they?** Twenty ML teams want raw access and control. Two hundred product teams want a simple endpoint and sane defaults.
- **Is there an SLA, and who's on call?** A platform without a defined SLA becomes everyone's scapegoat during incidents.

**Assume:** 60 internal teams, both vendor APIs and 4 self-hosted open models on a shared 64-GPU cluster. Mix of interactive and batch. Cost control is enforcing with per-team quotas and a break-glass override. Some data classifications cannot leave the network. ~5M requests/day.

**The design.**

The platform is a **gateway plus a scheduler plus an accounting system**. The models are the easy part.

*Gateway.* There is one OpenAI-compatible API surface for everything, vendor models and self-hosted models alike. Compatibility matters practically, because teams can then use existing SDKs and switching a model becomes a config change. The gateway handles authentication with per-team service tokens, request validation, data-classification tagging, routing, rate limiting, retries, and logging.

Routing is policy-driven. Each request is evaluated against `(team, model_alias, data_class, priority)`. Teams request **aliases**, not model versions, such as `fast-chat`, `deep-reason`, and `cheap-classify`. That is the most important design decision in the whole system. Aliases let the platform migrate everyone off a deprecated model, or shift traffic to a cheaper equivalent, without 60 teams changing code. A team that needs a pinned version can ask for one, and it pays for the privilege of being on the migration hook.

Data classification is enforced at the gateway. A request tagged `confidential` cannot route to a vendor endpoint, full stop. That is a gateway policy check, not a guideline in a wiki.

*Self-hosted serving.* I'd run vLLM or an equivalent behind the gateway, with continuous batching and paged KV-cache management. The throughput difference between continuous batching and naive request-level batching is large, with published measurements up to 23x on realistic mixed-length workloads. Each model gets a deployment sized to its demand. Small models are co-located, and large ones get dedicated tensor-parallel groups. I'd keep the number of distinct self-hosted models deliberately small, because every additional model fragments GPU memory and hurts batch efficiency. Four well-utilized models beat twelve half-idle ones.

Capacity is the hard scheduling problem. I'd run three priority classes. **Interactive** is latency-sensitive and gets guaranteed headroom. **Standard** is the default. **Batch** is preemptible and runs on whatever is free. Batch jobs go to a separate queue with a completion SLA in hours rather than a latency SLA, and they get preempted when interactive load spikes. This is what lets a 64-GPU cluster serve a workload that would otherwise need 100 GPUs, because you are filling the troughs.

*Caching.* An exact-match cache on `(model, prompt, params)` with a TTL catches a surprising amount of internal traffic: teams re-running the same evaluation, retry storms, and duplicated pipeline stages. Prefix caching in the serving layer is worth more still. Internal workloads have enormous shared prefixes, such as the same system prompt across a million classification calls, so reusing that KV cache cuts both latency and cost materially. I'd expose an explicit "this is my stable prefix" hint in the API.

*Accounting.* Every request logs the team, the alias, the resolved model, input and output token counts, cached-token counts, latency, and computed cost. Vendor cost comes from the price sheet, and self-hosted cost comes from an amortized GPU-second rate. This flows to a real-time counter per team in Redis for quota enforcement, and to a warehouse for reporting. Quota enforcement is soft, then hard. At 80% the team gets an alert. At 100%, non-production traffic is rejected while production traffic continues with escalating alerts. Break-glass requires a manager approval, and the approval is logged. Cutting off production to enforce a budget is almost always worse than the overspend.

```
TEAMS ──> GATEWAY (auth, data-class check, alias resolve, rate limit, cache)
             │                          │
             │                    quota counter (Redis, realtime)
             v
       ROUTER ──┬──> vendor APIs (multi-provider, failover)
                └──> self-hosted: vLLM fleet on 64 GPUs
                          │  priority classes: interactive | standard | batch(preempt)
                          v
             usage log ──> warehouse ──> per-team dashboards, chargeback, anomaly alerts
```

*Reliability.* Each alias has multi-provider failover behind it. Add circuit breakers on failing endpoints, per-team rate limits so one team's runaway loop does not degrade everyone, and hard request timeouts. A shared platform's dominant failure mode is noisy neighbors, and per-team isolation is the answer.

*Developer experience* is most of the adoption battle. That means a playground UI, prompt and version management, one-line SDK setup, sane defaults, and a dashboard where a team sees its own spend broken down by endpoint. If the platform is harder to use than calling the vendor directly, teams route around it, and then you lose both cost control and governance.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

The models are the payload. The platform contains almost no AI of its own. That is the point, and saying it plainly is good signal.

There are two places I'd consider a model. One is **semantic caching**, which returns a cached response for a semantically similar prompt. The other is **routing by difficulty**, where a small classifier decides whether a query needs the expensive model. I'd treat both as opt-in and prove them per workload. Semantic cache hits on subtly different prompts produce wrong answers, and difficulty routing that mispredicts silently degrades quality.

I would **not** use a model for quota decisions, data-classification enforcement, scheduling, retries, or cost attribution. And I'd resist building "automatic prompt optimization" into the platform. That belongs to teams who can evaluate it, not to shared infrastructure that cannot.

**The hard tradeoff** — *self-hosting vs. vendor APIs.*

Self-hosting is cheaper at high sustained utilization. It keeps data in-network, and it gives you fixed capacity and no rate limits. However, it means owning GPU capacity planning, model upgrades, inference-stack bugs, and 3am pages. It is also badly economic at low utilization, because a mostly-idle 8-GPU deployment costs more than the API calls it replaces. Vendor APIs have no ops burden and frontier quality. However, their cost scales linearly forever, their rate limits are outside your control, and data leaves the boundary.

I'd run both, with a clear rule. **Self-host the high-volume, stable, small-model workloads** such as classification, extraction, and embeddings, because utilization is high there and open models meet the quality bar. **Use vendor APIs for low-volume frontier-quality work** such as complex reasoning and agents. The crossover is a utilization calculation I'd do explicitly per workload. What changes my mind is sustained utilization. Below roughly 40% on a dedicated deployment, self-hosting is losing money, so I'd move that workload to the API.

**How I'd evaluate it.**

I'd track platform metrics, not model metrics. That means **GPU utilization**, which is the number that justifies the cluster, tokens per second per GPU by model, p50/p95/p99 latency and time-to-first-token per alias and priority class, queue depth and wait time by class, cache hit rate, and error rate broken down by cause such as rate limit, timeout, and provider error. It also means **cost per million tokens by alias**, compared against the vendor equivalent. That comparison is the platform's ROI statement.

Adoption metrics matter as much. Track the share of company LLM spend flowing through the gateway, because traffic routing around the gateway is a failure. Track the number of teams onboarded and the time-to-first-successful-call for a new team. Also track the number of teams pinned to a deprecated model, which is the platform's technical-debt gauge.

I'd also run a continuous quality canary per alias, using the frozen-prompt daily replay from problem 5. Then when the platform silently reroutes an alias to a cheaper model, the quality change is visible.

**Failure modes I name before the interviewer does.**

- **Noisy neighbor** — one team's agentic loop consumes the whole interactive pool. Use per-team rate limits and priority isolation.
- **Silent quality change from alias rerouting** — the platform swaps the model behind `fast-chat` and three teams regress. This needs notice, canary evals per alias, and opt-out pinning.
- **Retry storms** — a provider slows down, every client retries, load triples, and everything collapses. This needs client-side jittered backoff, circuit breakers, and load shedding at the gateway.
- **Cost attribution gaps** — a shared service calls the platform on behalf of many teams, so everything lands on one cost center. This needs propagated request attribution through internal service calls.
- **Vendor rate limits during a spike** — you do not control them. Use multi-provider failover and a queue for non-interactive traffic.
- **GPU fragmentation** — too many model variants leaves memory stranded and batch sizes small. Enforce a small supported model set.
- **Prompt/PII leakage into logs** — logging full prompts is invaluable for debugging and a governance hazard. Use redaction, short retention on payloads, and access controls. Log metadata forever and payloads briefly.

**Follow-ups they will ask.**

*"How do you decide GPU capacity?"*
From a demand model, not a guess. I'd measure tokens per second per GPU per model under realistic batch conditions. That number varies enormously with sequence length and batch composition, so I'd benchmark with replayed production traffic rather than synthetic uniform prompts. Then I'd size interactive capacity for p99 demand with headroom, and let batch fill the rest. The key insight is that batch workloads make the utilization math work. Sizing for the interactive peak alone means paying for idle GPUs most of the day. I'd review capacity monthly against queue-wait metrics, and treat sustained interactive queueing as the trigger to buy.

*"A team says the platform is slower than calling the API directly. What do you do?"*
Measure, and be honest. The gateway adds real overhead for auth, policy, logging, and routing, which should be low single-digit milliseconds. If it is more than that, it is a bug and I'd profile it. Usually, though, the complaint is about queueing on self-hosted capacity under load, which is a capacity or priority-class problem. The fix might be moving that team's workload to a vendor endpoint by routing policy, and the alias system makes that a config change. What I would not do is let them bypass the gateway, because then governance and cost control evaporate one exception at a time.

*"How do you handle model deprecation?"*
Aliases make deprecation tractable, but not free. The process is: announce with a timeline, use the usage logs to identify every affected team and endpoint, and produce a diff report by running each team's own eval set, or a canary set, against the old and the new model. Migrate automatically the teams whose canary shows no regression, and work individually with the rest. Keep the old model available at a premium price for stragglers, because that creates the right incentive. The platform should publish a dashboard of who is still on deprecated aliases, because visibility drives migration far better than emails.

*"What's the cost model for chargeback?"*
Vendor calls are pass-through at list price, plus a small platform margin to fund the team. Self-hosted is amortized: the total GPU cost, meaning hardware or cloud plus a share of platform engineering, divided by tokens served, computed monthly and published. I'd deliberately price self-hosted below the vendor equivalent, to steer high-volume workloads there. Pricing is the steering mechanism for a platform, and it works better than policy documents. Batch-priority traffic gets a discount, which is how you get teams to move work off-peak voluntarily.

*"How do you support fine-tuned models?"*
For self-hosted models, serve LoRA adapters from a shared base model. This is the whole reason to prefer LoRA over full fine-tunes in a platform context. Dozens of adapters can share one base model's weights and batch together, whereas dozens of full fine-tunes need dozens of deployments. I'd give teams an adapter registry, a training pipeline they can invoke, and adapter-level aliases. I'd also require an eval before an adapter goes to production, because the most common outcome of an enthusiastic first fine-tune is a model that is worse than the base model with a good prompt.

*"How do you handle streaming and long-running agent requests?"*
Stream end-to-end over SSE. That means the gateway cannot buffer full responses, which constrains the middleware design: token counting and logging happen incrementally, and guardrail checks must be streaming-compatible or applied to buffered windows. For agent loops that run for minutes across many calls, I'd offer an async job API: submit, get a job ID, then poll or receive a webhook. That decouples the client from connection timeouts and lets those jobs run at batch priority. I'd also enforce per-job budget caps in tokens and dollars, because an agent in a loop is the most reliable way to generate a surprise five-figure bill.

---

## 9. Document extraction pipeline

### Q: "Design a document extraction pipeline — invoices and contracts in, structured data out."

**Clarifying questions to ask first**

- **How many distinct document templates, and do they repeat?** If 80% of volume comes from 50 recurring vendors, template-specific handling beats a general model. If every document is novel, it is a general extraction problem.
- **What's the cost of an extraction error?** A wrong invoice amount posted to the ledger is a financial error that needs reconciliation. A wrong contract date in a search index is a minor annoyance. This answer sets the confidence threshold and decides whether humans review everything.
- **Are documents born-digital PDFs, scans, or photos from phones?** Born-digital documents give you exact text and coordinates. Phone photos add skew, glare, and crop problems, and then OCR quality becomes the dominant error source.
- **Is the target schema fixed, or does it vary by customer?** Per-customer schemas mean the extraction prompt and model must be configurable at runtime, and evaluation must be per schema.
- **What's the throughput and latency requirement — real-time on upload, or nightly batch?** Batch lets me use expensive multi-pass extraction. Real-time constrains it.
- **Does a human review step exist today, and can I keep it?** Human-in-the-loop is usually the right answer here, and knowing the available review capacity shapes the confidence thresholds.

**Assume:** 200k documents/month, mixed invoices (70%) and contracts (30%). ~60% born-digital PDF, 40% scans. 2,000 recurring vendors covering 80% of invoice volume. Errors on invoice amounts, dates, and vendor identity are financially material. Per-customer schema variation exists. Near-real-time (under 2 minutes) expected. A 6-person review team exists.

**The design.**

Here is the framing. **This is a pipeline with a confidence-routed human review stage, not a model.** Any design that ends at "the LLM returns JSON" fails, because the interesting question is what happens to the 8% it gets wrong.

*Ingest.* Documents arrive by email, API upload, or SFTP drop. Each one gets a content hash for deduplication, because the same invoice routinely arrives three times. Each one also gets a document ID and immutable storage of the original bytes. Never lose the source, because every extraction dispute is settled by looking at it. State lives in a workflow engine, so a document's journey is resumable and observable.

*Preprocessing.* Classify the document type, detect page count and orientation, deskew and denoise scans, and split multi-document PDFs. A 40-page file that is actually 12 invoices is common and easy to miss. Born-digital PDFs get their text extracted directly with coordinates. Scans go to OCR. I'd keep OCR and layout analysis as a distinct stage that produces **text with bounding boxes**, because those coordinates are what make everything downstream verifiable. You can point at where in the document each value came from.

*Extraction* runs as a cascade:

1. **Template match.** Fingerprint the document by vendor identity from the logo and text, plus a layout hash. If it matches a known vendor template with learned field positions, extract by position plus local pattern matching. This is fast, nearly free, and highly accurate for the 80% of volume that comes from recurring vendors. Templates are *learned*, not hand-authored. After N successful extractions from a vendor, the system infers the stable field locations and promotes a template.
2. **Model extraction.** For unmatched documents, a vision-language model gets the page images, the OCR text, and the target JSON schema. It returns structured output with a per-field confidence and a source bounding box for each value. Requiring the model to cite a bounding box is the key trick, because it makes hallucinated values detectable. A value that does not appear anywhere in the OCR text at the claimed location gets rejected automatically.
3. **Validation**, which is pure business logic and does a lot of the real work. Line items must sum to the subtotal. Subtotal plus tax must equal the total. Dates must be plausible and ordered. The vendor must exist in the vendor master, fuzzy-matched. The currency must be valid. The PO number must exist and have a remaining balance. The invoice number must not be a duplicate for that vendor. Arithmetic validation alone catches a large share of extraction errors for free, because a misread digit almost always breaks the sum.

*Confidence routing.* Combine the model confidence, the validation results, and the field criticality into one decision. **Auto-approve** when all validations pass, confidence is high, the vendor is known, and the amount is below the threshold. **Review** means the item is queued to a human, with the document rendered and the extracted fields overlaid on their bounding boxes, so review is a two-second glance rather than a re-keying job. **Reject** means the document is unreadable or of the wrong type.

The review UI is where the leverage is. Reviewers should confirm or correct, never re-type. Every correction is written back as a labeled example. That correction store is the pipeline's most valuable asset, because it drives template learning, prompt few-shot selection, and eventual fine-tuning.

```
email/API/SFTP ──> dedupe(hash) ──> raw store (immutable) ──> workflow engine
                                                                   │
              classify + split + deskew ──> OCR/text + bounding boxes
                                                                   │
                        ┌──────────── template match? ─────────────┤
                     yes│                                        no│
              positional extract                      VLM extract (schema + bbox)
                        └──────────────┬───────────────────────────┘
                                       v
                        VALIDATION (arithmetic, vendor master, PO, dupes)
                                       │
                     ┌─────────────────┼──────────────────┐
                auto-approve       REVIEW QUEUE        reject
                     │              (overlay UI)          │
                     └──────> ERP / downstream <──────────┘
                                       │
                             corrections ──> template learning + eval set
```

*Output* goes to the ERP or the downstream system through an idempotent write keyed by document ID. A reconciliation job then verifies that what landed matches what was extracted.

On throughput, 200k documents a month is about 5 documents a minute on average, which is trivially handled. However, arrivals are bursty at month-end. So I'd size the queue for 20x the average, let latency degrade for batch drops, and keep the interactive upload path prioritized.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is the OCR, the vision-language extraction for novel layouts, and the fuzzy vendor matching. The VLM handles roughly 20% of volume, which is the non-template documents. That is a good illustration of the 20/80 point: the model is the smallest piece of a system dominated by ingestion, validation, review tooling, and ERP integration.

I would **not** use a model for arithmetic. Compute the sums in code, because asking a model to verify that line items add up is slower, costlier, and less reliable than `sum()`. I would not use a model for duplicate detection, which is hashing plus a database constraint, for the ERP write, for currency conversion, which is a rates table, or for deciding whether an invoice should be paid, which is an approval workflow with authority limits. And I would not let the model output values that do not appear in the source document. Every extracted value must be traceable to a span.

**The hard tradeoff** — *general model on everything vs. template-first with model fallback.*

A single VLM on every document is far simpler to build and maintain. There is no template infrastructure and no fingerprinting, there is one code path, and it handles new vendors on day one. However, it costs roughly 10–50x more per document, it is slower, and it is *nondeterministic*. The same invoice can extract differently on two runs, which is a nightmare for a finance team doing reconciliation.

Template-first is more machinery. However, it gives deterministic, auditable, near-free extraction on the bulk of the volume, and the model is reserved for the tail. For a finance system where reproducibility is a real requirement, I'd take template-first. What changes my mind is the volume concentration. If the vendor distribution were flat instead of 80/20, templates would never pay for themselves, so I'd go model-only, possibly with caching keyed on layout hash to recover some determinism.

**How I'd evaluate it.**

Offline, I'd build a labeled set of a few thousand documents, stratified by type, vendor, and scan quality, with ground-truth field values. Per field I'd measure **exact-match accuracy** for structured fields such as dates, amounts, and IDs, normalizing before comparison, because "01/02/2026" and "2026-01-02" are the same value. I'd use fuzzy match for names and addresses, and **F1 for line-item extraction**, because that is a set-matching problem rather than a single value. Report per field, never aggregate. A 95% average accuracy can hide 70% on the total amount field, which is the only field that matters financially.

The metric I'd actually optimize is the **straight-through processing rate at a fixed error budget**: what fraction of documents can be auto-approved while post-approval errors stay below, say, 0.1%. That single number captures both quality and cost, and the business cares about it because it maps directly to review headcount.

Online, I'd track the STP rate, review queue depth and time-to-clear, per-reviewer throughput, and correction rate by field. Which fields humans fix most is your roadmap. I'd also track the **downstream error rate**, meaning errors caught after approval by the ERP, by reconciliation, or by a vendor complaint. That last number is the true false-negative measure, and it is the only honest check on whether the confidence thresholds are right.

**Failure modes I name before the interviewer does.**

- **Silent OCR degradation** on a new scanner or fax pipeline — accuracy drops and nothing alerts, because the model still returns confident JSON. Mitigation: monitor OCR confidence distributions and per-source STP rates.
- **Hallucinated values that pass validation** — the model invents a plausible total that is arithmetically consistent, because it also invented the line items. Mitigation: bounding-box grounding, so every value is traceable to source text.
- **Multi-document PDFs mis-split** — two invoices merged into one record. This needs explicit splitting with validation on invoice-number changes.
- **Template drift** — a vendor changes their invoice layout and positional extraction silently pulls the wrong field. Mitigation: validation catches most of it, plus monitoring per-vendor rejection rates for step changes.
- **Duplicate payments** — the same invoice is ingested twice through different channels with slightly different bytes, so the hash differs. This needs semantic dedup on (vendor, invoice number, amount) as a hard constraint in the ERP write.
- **Reviewer rubber-stamping** — under queue pressure humans approve without looking, and the review stage becomes theater. Mitigation: inject known-bad items to measure the reviewer catch rate, and monitor per-reviewer time-per-document.
- **Schema evolution** — a customer adds a required field and historical documents lack it. This needs versioned schemas and a backfill strategy.

**Follow-ups they will ask.**

*"How do you handle a 90-page contract where you need 15 specific clauses?"*
That is a different problem from invoices. It is retrieval plus extraction, not layout parsing. I'd chunk by clause structure, because contracts have reliable heading hierarchies. Then embed and index the chunks. For each target field, run a retrieval to find the candidate clauses and extract from just those. That keeps the prompt small, and it lets me cite the exact clause and page for each extracted value, which lawyers require. For fields that may be absent, the model must be able to return null with confidence. The most common contract-extraction error is inventing a termination-notice period that the contract simply does not specify.

*"What confidence threshold do you use?"*
A derived one, not a chosen one. I'd plot the precision-versus-coverage curve on the labeled set per field, then pick the threshold that hits the error budget with the highest coverage. Thresholds are **per field and per amount band**. A \$50 invoice and a \$500,000 invoice should not share a threshold, so I'd route all high-value documents to review regardless of confidence. I'd also recalibrate quarterly, because model confidence is not stable across model versions and a vendor upgrade silently shifts the operating point.

*"How do you bootstrap when you have no labeled data?"*
Route everything to human review for the first few weeks. That is not a failure state, it is the data collection plan. The review team is doing the work today anyway, and the UI just captures their output as labels. After a few thousand documents I have a real eval set, per-vendor templates for the top vendors, and measured confidence calibration. Then I raise the thresholds gradually while monitoring downstream errors. Launching with auto-approval before you have calibration data is how you get a finance incident in week one.

*"What if the customer's schema changes?"*
Schemas are versioned config, not code, so a new field is a config change plus an eval. The mechanics are: add the field with a nullable default, run extraction on a sample to measure accuracy on the new field specifically, route it to review-always until its accuracy clears the bar, then let it participate in auto-approval. Historical documents get backfilled by re-running extraction from the immutable raw store, which is exactly why keeping the original bytes forever matters. Never migrate by mutating extracted records.

*"How do you handle non-English documents?"*
Do OCR language detection first, because OCR accuracy is language-dependent and the wrong language model produces garbage. The VLM handles most major languages for extraction, but accuracy varies substantially, so I'd measure it per language rather than assume. Validation logic needs localization too: date formats, decimal separators, tax structures, and address formats. Decimal separators are a genuine source of 1000x amount errors, because 1.234,56 and 1,234.56 look alike. I'd treat each language and region as a separate segment with its own eval set and its own thresholds, because a single global threshold will be too loose somewhere.

*"How does this integrate with the ERP?"*
Idempotently and defensively. Writes are keyed by document ID with an upsert, so retries are safe. I'd write to a staging table first, run reconciliation to check that the total we posted matches the total we extracted, and then commit. The ERP is usually old, it sometimes has surprising validation rules, and it often rejects records for reasons unrelated to extraction quality. So I need a rejection-handling path that routes ERP failures back to a human queue with the error message. In my experience this integration is a third of the project's effort and none of its glamour, and I'd budget for it explicitly.

---

## 10. Fraud detection

### Q: "Design a fraud detection system for our payments product."

**Clarifying questions to ask first**

- **What fraud types — stolen cards, account takeover, first-party/friendly fraud, merchant collusion, or bust-out?** These have almost nothing in common. Stolen-card fraud is caught by device and velocity signals. First-party fraud looks like a normal customer until it does not. Bust-out plays out over months.
- **What's the current fraud rate and the current false-positive rate?** A 0.1% fraud rate means extreme class imbalance, and it means false positives dominate the cost. Blocking good customers is usually more expensive than the fraud.
- **Do we bear the loss, or does the issuer?** Liability determines the objective function entirely. If we eat chargebacks, we optimize loss dollars. If we do not, we optimize customer experience and network compliance.
- **What's the decision latency budget?** An inline authorization decision has maybe 100ms end-to-end. A post-transaction review has hours.
- **What's the feedback delay on labels?** Chargebacks arrive 30–90 days later. So my labels are always stale, and I cannot evaluate a model deployed last week using confirmed labels.
- **What manual review capacity exists?** This sets how much of the uncertain band can be routed to humans instead of auto-decided.

**Assume:** card-not-present marketplace payments, 3M transactions/day, fraud rate ~0.15% of transactions and ~0.4% of dollars, we bear chargeback liability. Inline decision budget 100ms p99. Chargeback labels arrive with a 45-day median lag. A 20-person review team.

**The design — and the first thing I'd say out loud: this is not an LLM problem.**

I'd say that explicitly in the interview, because the temptation to reach for a generative model here is exactly the failure mode this round tests for. Fraud detection is a **tabular, imbalanced, adversarial, low-latency classification problem with delayed labels**. The right tools are gradient-boosted trees, graph features, and rules. An LLM is too slow for a 100ms budget and too expensive at 3M/day. It is not more accurate on tabular data. And decisively, it is not explainable in a way that satisfies regulators who require adverse-action reasons. A language model *does* help at the edges: summarizing a case for a human reviewer, reading free-text merchant descriptions or dispute narratives, and helping analysts write rules. Those uses are real but peripheral.

*Architecture.* Three layers sit on the inline path, in order of cost.

**1. Rules engine (sub-millisecond).** This handles hard blocks and allows: sanctions and OFAC screening, cards on the known-fraud list, velocity limits such as N transactions per card per hour and M cards per device per day, impossible-geography checks, and allowlists for trusted merchants. Rules are essential and underrated. They are instantly deployable when an attack starts, because a model needs data and a retrain while a rule needs five minutes. They are fully explainable and deterministic. Every mature fraud system is rules plus a model, never a model alone.

**2. Feature computation and model (~30–50ms).** This is the engineering core. Features come in three families.
- *Transaction features*: amount, amount relative to the account's history, merchant category, currency, time of day, and card BIN characteristics.
- *Velocity and aggregate features*: counts and sums over 1h, 24h, 7d, and 30d windows, keyed by card, account, device, IP, email, and shipping address. These are the workhorses. Computing them in under 50ms at 3M/day requires a streaming aggregation layer, such as Flink writing to a low-latency store. On-demand queries will not make the budget.
- *Graph features*: the highest-value family and the hardest. Entities such as accounts, devices, cards, emails, addresses, and IPs form a graph, and fraud is intensely clustered inside it. Features such as "number of distinct cards on this device in 30 days", connected-component size, share of neighbors with prior chargebacks, and shortest-path distance to a known fraudster catch organized fraud that per-transaction features miss entirely. I'd maintain the graph incrementally with precomputed neighborhood aggregates, because a live traversal will not fit the latency budget.

The model is a **gradient-boosted tree ensemble** such as XGBoost or LightGBM. It is the right choice for four reasons. It is excellent on tabular data. It is fast to score, at sub-millisecond for hundreds of trees. It is robust to missing features. And it is interpretable through SHAP values, which matters for both reviewer tooling and regulatory explanation. I'd handle class imbalance with scale-weighting rather than resampling, and I'd train against a **dollar-weighted** loss, because a \$4,000 fraud and a \$12 fraud are not equally important.

**3. Decision layer.** The model outputs a calibrated probability. The decision policy converts that probability into an action using expected value:

$$\mathbb{E}[\text{loss}_{\text{approve}}] = p_{\text{fraud}} \cdot (\text{amount} + \text{chargeback fee}) \quad \text{vs.} \quad \mathbb{E}[\text{loss}_{\text{decline}}] = (1 - p_{\text{fraud}}) \cdot \text{customer LTV impact}$$

In words: approving costs you the transaction amount plus the chargeback fee, but only when the transaction is actually fraud. Declining costs you the damage to a good customer, and that cost applies whenever the transaction is not fraud. You take whichever action has the smaller expected loss. Actions are graded, not binary: approve, approve with step-up authentication such as a 3DS challenge, route to manual review, or decline. Step-up is the most valuable middle option, because it converts a decline into mild friction. It is where a well-designed system recovers most of the revenue that a binary system throws away.

```
TXN ──> RULES (µs: sanctions, blocklist, velocity caps) ──block──> DECLINE
          │ pass
          v
   FEATURE SERVICE ──┬── streaming aggregates (Flink -> low-latency store)
      (~20ms)        ├── entity graph neighborhood features (precomputed)
                     └── account/card profile store
          │
          v
    GBDT SCORE (<5ms) ──> calibrated p(fraud) ──> DECISION POLICY (expected value)
          │                                            │
          │                              approve | 3DS step-up | review | decline
          v
   log txn + FEATURE SNAPSHOT (critical) ──> training store
                                                  │
   chargebacks (45d lag) + review outcomes (hours) ──> labels ──> weekly retrain
```

*The label problem is the defining constraint.* Chargebacks arrive 45 days late. So a model trained today uses features from transactions that are at least 45 days stale, and I cannot evaluate this week's model on confirmed outcomes. There are three mitigations. First, use fast proxy labels for early signal: manual review decisions available in hours, customer fraud reports, issuer decline codes, and account-takeover confirmations. Second, run a **maturity-aware evaluation**, where a dataset window counts as complete only after the chargeback horizon has passed. Third, monitor score-distribution drift as a leading indicator, because a model degrading against a new attack shows up as a distribution shift long before it shows up in chargebacks.

*Logging feature snapshots at decision time is non-negotiable.* Features are time-windowed aggregates, so you cannot reconstruct "what was this card's 24h velocity at 3:14pm on the 8th" from historical tables. If you try, you leak future information and train a model that looks brilliant offline and fails in production. Snapshotting the exact scored feature vector is the most important implementation detail in the whole system.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

ML is the GBDT scorer, plus optionally graph embeddings or an anomaly-detection model for unsupervised novelty detection. That is all of it on the inline path.

I would **not** use an LLM for the decision. It is too slow, too expensive, no more accurate on tabular features, and unexplainable. I would not use a model for sanctions screening, because that is a regulatory list match and must be deterministic and auditable. I would not use one for hard velocity limits, or for the final action policy, which is expected-value arithmetic in code with thresholds owned by risk management. I'd use an LLM only off the critical path: generating case summaries for reviewers, parsing free-text dispute narratives and merchant descriptors, and helping analysts translate an observed pattern into a candidate rule. Those are genuinely useful and worth building second.

**The hard tradeoff** — *strict thresholds (block more fraud, decline more good customers) vs. loose thresholds with step-up authentication.*

Strict blocking minimizes fraud loss and is easy to defend internally. However, false declines are enormously expensive: you lose the transaction, you often lose the customer, because a good customer wrongly declined frequently never returns, and you pay support cost. The industry's open secret is that false-decline losses typically exceed actual fraud losses by a wide margin.

I'd run loose thresholds with heavy use of step-up authentication and manual review in the middle band. And I'd insist on measuring the false-decline cost rather than assuming it. The way to measure it is a small randomized approval holdout: approve a sample of transactions the model would have declined, eat the fraud loss, and learn the true precision of your declines. It costs real money, and it is the only way to know whether your decline population is 80% fraud or 20% fraud. That measurement is what changes my mind. If declines are overwhelmingly fraud, I tighten. If they are mostly good customers, I loosen and lean harder on step-up.

**How I'd evaluate it.**

Offline, I'd use a strict **temporal split**: train on weeks 1–8 and test on weeks 9–10. Never use random splits, because fraud is temporally correlated and random splitting leaks. I'd measure **precision-recall AUC**, not ROC-AUC, because ROC-AUC is misleadingly flattering at a 0.15% positive rate. I'd measure **dollar-weighted recall at a fixed decline rate**, which is the business metric: what share of fraud dollars do we stop while declining only X% of transactions. And I'd measure calibration, because the expected-value policy requires probabilities that mean what they say.

Online, I'd track fraud loss in basis points of volume, decline rate, step-up rate and step-up pass rate, manual review volume and precision, chargeback rate by cohort, and approval rate for good customers. I'd segment that last one, because an approval-rate drop concentrated in one country or card type is a targeted regression. My guardrails are the false-decline estimate from the randomized holdout, and disparate-impact monitoring across protected-attribute proxies, which is a regulatory requirement in many jurisdictions.

**Failure modes I name before the interviewer does.**

- **Adversarial adaptation** — fraudsters probe with small transactions to find the threshold, then scale. This needs rapid retraining, rules for fast response, and randomized thresholds so the boundary is not cleanly discoverable.
- **Feature leakage from delayed data** — a feature computed from data that was not available at decision time. Prevented by feature snapshotting and point-in-time-correct training joins.
- **Label bias and selection bias** — you only observe outcomes for approved transactions, so the model never learns about the declined population. This is the deepest problem in fraud ML, and the randomized approval holdout is the only real fix.
- **Concept drift after a product launch** — a new payment method or geography shifts the distribution and the model's calibration breaks. This needs drift monitoring per segment; see problem 14.
- **Graph feature staleness** — the entity graph lags, so a device that just linked to 40 cards looks clean. This needs streaming graph updates with monitored lag.
- **Cold start on new accounts** — there is no history, so all velocity features are null. This needs a separate model or rule set for thin-file accounts, rather than feeding nulls to the main model.
- **Over-blocking a legitimate merchant or corridor** — one bad feature interaction wipes out a country's approval rate. This needs per-segment approval-rate alerting, which catches the problem in hours rather than at the monthly business review.

**Follow-ups they will ask.**

*"When would an LLM actually help here?"*
In three places, all off the critical path. First, reviewer productivity. Generate a case summary that pulls the account history, the graph neighborhood, and the top SHAP contributors into one paragraph, which cuts review time meaningfully. Second, unstructured signals. Merchant descriptors, product listings, dispute narratives, and support transcripts contain fraud signal that tabular pipelines ignore, and an LLM can turn those into features computed asynchronously and cached. Third, analyst tooling. An LLM can translate "cards with a Baltic BIN buying gift cards under \$50 within 10 minutes of signup" into a candidate rule with a backtest. I'd build the reviewer summary first, because it has the clearest ROI.

*"How do you handle a brand-new attack pattern?"*
Detection comes before classification. Unsupervised anomaly detection on the transaction stream and on the graph structure, watching for a sudden dense component forming, flags novelty without needing labels. Then a human analyst investigates, and the immediate response is a **rule**, deployed within the hour. That is why the rules engine exists. In parallel, confirmed cases get labeled and fed into an emergency retrain, which lands in days. The rule stays until the model demonstrably covers the pattern, and then it is retired. I'd track the share of fraud caught by rules versus by the model as a health metric, because a rising rule share means the model is falling behind.

*"How often do you retrain, and how do you deploy safely?"*
Weekly retrains as a baseline, with the ability to trigger off-cycle when drift alerts fire. Deployment is shadow, then canary. The new model scores live traffic without acting for a week, so I can compare score distributions and decision disagreements against the incumbent. Then it canaries on a small traffic share with close monitoring of approval and decline rates. Because true labels lag 45 days, the canary decision has to be made on proxy metrics and distributional comparison, not on confirmed fraud rates. I'd also keep the previous model warm at all times for instant rollback.

*"How do you explain a decline to a customer or a regulator?"*
SHAP values from the GBDT give per-decision feature attributions. Those map to human-readable reason codes through a maintained mapping, such as "unusual transaction amount for this account" or "new device." Regulations in several jurisdictions require adverse-action reasons, and that is a large part of why a tree model beats a neural network or an LLM here: the explanation is derived from the actual decision, not generated after the fact. I'd store the reason codes with the decision record. I'd also be careful that the explanations given externally do not reveal enough for an attacker to reverse-engineer the thresholds.

*"How do you decide the review queue priority?"*
By expected value of review, not by score. A transaction with p=0.4 and \$5,000 at stake is worth far more reviewer time than p=0.9 on \$30, and the second one should just be auto-declined. So priority is roughly $p_{\text{fraud}} \times \text{amount} \times$ (uncertainty), with a boost for cases where review yields transferable knowledge, such as a novel pattern or a large graph cluster. I'd also cluster related transactions, so one review decision applies to a whole ring. Otherwise reviewers adjudicate 200 transactions from one fraud cluster individually. That clustering is often the biggest single win in review efficiency.

*"What's your feature store story?"*
It is critical, and it is problem 12 in this document. The requirements here are point-in-time-correct historical joins for training, sub-20ms online reads at 3M/day, and, most importantly, identical aggregation logic in both paths. I'd define each feature once in a declarative spec that generates both the streaming job and the batch backfill. I'd also run a continuous consistency check that compares online-served values against recomputed offline values on a sample. Fraud is the domain where training-serving skew hurts most, because the features are time-windowed aggregates and the ways to get them subtly wrong are numerous.

---

## 11. Multi-agent research assistant with tool use

### Q: "Design a multi-agent research assistant that can use tools — web search, internal docs, code execution — to answer complex research questions."

**Clarifying questions to ask first**

- **What's the acceptable latency and cost per question — 30 seconds and \$0.10, or 20 minutes and \$5?** This one answer decides whether I can afford parallel sub-agents at all. Deep research at \$5 a question is a different product from an interactive assistant.
- **Are the tools read-only, or can the agent write/execute in systems that matter?** Read-only makes this a quality problem. Write access makes it a safety and permissions problem, and then I'd want a policy layer like the one in problem 2.
- **Who's the user, and what's the cost of a plausible-but-wrong answer?** For an analyst who verifies everything, some error is tolerable. For a decision input, I need citations, calibrated uncertainty, and possibly a refusal path.
- **Is the question space open-ended, or a known set of research patterns?** If 80% of questions are "competitive analysis of X" or "summarize what we know about customer Y", those become structured workflows and I use far fewer agents.
- **How much does the internal corpus matter vs. the public web?** Internal-heavy means permissions and RAG quality dominate. Web-heavy means source quality and freshness dominate.
- **Do users need to see and steer the process, or just get an answer?** Steerable means a streaming UI that shows the plan and the intermediate findings. That is substantially more product engineering, and it is much better for trust.

**Assume:** internal analysts, open-ended questions, tolerance of 3–10 minutes and roughly \$0.50–\$2 per question. Tools: web search/fetch, internal document RAG (permissioned), a SQL warehouse (read-only), and a sandboxed Python executor. Answers must be cited. Users see progress.

**The design.**

Here is the honest starting position, and I'd state it out loud. **Most "multi-agent" systems should start as a single agent with good tools.** Multi-agent adds value in exactly one situation: when subtasks are genuinely independent and parallelizable, so you are buying wall-clock time and context isolation. It costs coordination complexity, token multiplication, and a much harder debugging story. So I'd design a **lead agent with parallel sub-agents on independent branches**, not a committee of specialists chatting.

*Topology.* A **lead orchestrator** agent decomposes the question into independent sub-questions, spawns one sub-agent per branch, and synthesizes the result. Sub-agents are identical workers that differ only in their assigned sub-question and their tool allowlist. They do not talk to each other, because agent-to-agent chat is where token budgets and coherence go to die. Each sub-agent returns a structured findings object: claims with source citations, a confidence value, and notes on what it could not determine.

Decomposition quality is the highest-leverage part. A bad decomposition such as "research the market" produces three sub-agents doing the same thing. So I'd have the lead produce an explicit plan whose sub-questions are *disjoint and concrete*, show that plan to the user, and, in the first version, let the user edit it. The plan is also a natural checkpoint for cost control.

*Tools.* Every tool is a typed, permissioned, rate-limited interface with a hard timeout.
- `web_search(query)` and `fetch_url(url)` — with a domain reputation filter and a content-size cap.
- `search_internal(query)` — the RAG system from problem 1, running under the *user's* permissions, never the agent's. Identity propagates end to end, so the agent never has ambient authority.
- `query_warehouse(sql)` — a read-only role, row limits, a statement timeout, and a cost guard so a full-table scan does not cost \$400.
- `run_python(code)` — sandboxed with gVisor or Firecracker, no network, an ephemeral filesystem, CPU and memory caps, and a hard wall-clock limit.

Tool results go to a **shared artifact store**, not into the conversation. Sub-agents write findings and fetched documents as artifacts with IDs, and they pass the IDs around. The lead reads only what it needs. This is the key context-management move. Without it, a research run's context grows quadratically, and you hit both the window limit and a large token bill.

*Control loop.* Each agent runs a bounded ReAct-style loop with explicit budgets: max iterations of about 15, max tool calls, max tokens, and max wall-clock time. The harness enforces those budgets. They are not requested in the prompt. When a budget is exhausted, the agent must produce a partial answer that states what it did not finish. That is a graceful degradation path, not a crash.

*Synthesis.* The lead reads the findings and cross-checks the claims. Where sub-agents' sources disagree, it flags the disagreement, and that flag is genuinely valuable output. Then it writes a cited answer. A separate verification pass checks that every citation resolves to a real fetched artifact containing supporting text, and unsupported claims get dropped or marked. That verifier is cheap, and it is the difference between a research tool and a plausible-text generator.

```
QUESTION ──> LEAD AGENT: decompose ──> PLAN (shown to user, editable)
                  │
      ┌───────────┼───────────┬───────────┐   (parallel, isolated context)
   SUB-AGENT 1  SUB-AGENT 2  SUB-AGENT 3  ...
      │ tools: web | internal RAG (user's ACLs) | SQL (ro) | python (sandbox)
      └───────────┴───────────┴───────────┘
                  │  structured findings + artifact IDs
                  v
          ARTIFACT STORE (fetched docs, tables, charts)
                  │
        LEAD: cross-check, synthesize ──> VERIFIER (citations resolve?) ──> ANSWER
                  │
        full trace: every prompt, tool call, result, cost ──> observability
```

*Observability is the product for the engineers.* Every run produces a trace tree with agent spans, tool calls including arguments and results, token counts, and cost per node. Debugging a multi-agent failure without this trace is impossible. You cannot tell whether the answer was wrong because search returned junk, because a sub-agent misread a document, or because synthesis dropped a caveat.

*Cost control.* Set a per-run budget in dollars, enforced by the harness and visible to the user. Cap the sub-agent count, and I'd start at 4. Use a cheaper model for the sub-agents and the expensive model for decomposition and synthesis. That split matters, because sub-agent work is high-volume and mechanical, while planning and synthesis are where reasoning quality pays. Cache tool results by `hash(tool, args)` within and across runs with a short TTL, because research questions on the same topic repeat.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is the decomposition, the per-branch reasoning and tool selection, and the synthesis. That is the genuinely new capability.

I would **not** use a model for permissions, because identity propagation is enforced at each tool. Not for sandboxing decisions. Not for budget enforcement, which is harness-level counters. Not for retry logic. And not for deciding whether a citation is valid, because that is string and span matching against the artifact, which is deterministic and reliable. I'd also not use an agent where a workflow suffices. If a question type is common and its structure is known, hard-code the pipeline. Agents are for the open-ended tail. A system that routes 60% of questions to deterministic workflows and 40% to agents is better and cheaper than one that agents everything.

**The hard tradeoff** — *parallel sub-agents vs. a single sequential agent.*

Parallelism cuts wall-clock time roughly by the branch factor, and it gives each branch a clean context window, which measurably improves quality on broad questions. However, token usage multiplies. A multi-agent run can easily use 10–15x the tokens of a single-agent run on the same question. Sub-agents also duplicate work, miss cross-branch connections, and produce findings that contradict each other, which the lead must then reconcile.

I'd use parallelism only when the lead's plan contains genuinely independent branches, and I'd route narrow questions to a single agent. The plan structure makes that decision, not a global setting. What changes my mind is measured quality per dollar. If evaluation shows the single agent within a few points of the multi-agent system at a fraction of the cost, then the multi-agent path should be reserved for an explicitly requested "deep research" mode. I'd want that comparison run before building the parallel infrastructure, not after.

**How I'd evaluate it.**

This is hard, because there is no single right answer. So I'd evaluate on three axes.

Offline, I'd build a set of about 150 research questions with expert-written **rubrics** rather than reference answers. A rubric lists the facts a good answer must contain, the sources it should find, and the errors it must not make. I'd score with an LLM judge against the rubric, calibrated against human scoring. I'd also keep a **verifiable subset** of questions whose answers are checkable, such as a number in the warehouse or a fact with a canonical source. That subset gives an objective accuracy number, which anchors the fuzzy rubric scores.

Component metrics matter for debugging. I'd track decomposition quality, meaning do the sub-questions cover the question, judged by a rubric. I'd track tool-call success rate, retrieval recall on the internal branch, and the **citation validity rate**, meaning the share of citations that actually support their claim. I'd want that above 95%, and I'd measure it by spot-checking with humans, because the automated verifier only checks that a citation resolves, not that it supports the claim.

Online, I'd track task completion rate, user edits to the plan, because a high edit rate means bad decomposition, time and cost per question, thumbs, and whether the user copied or exported the answer. That last one is the best signal. My guardrails are p95 cost per run, the share of runs hitting budget limits, and the hallucinated-citation rate.

**Failure modes I name before the interviewer does.**

- **Runaway cost** — an agent loops on a failing tool 40 times. Use hard budget enforcement in the harness, plus circuit breakers on repeated identical tool calls.
- **Context overflow mid-run** — the agent's history exceeds the window and important early findings get truncated. Mitigated by the artifact store and periodic compaction into a structured state object.
- **Confident synthesis over thin evidence** — one blog post becomes "industry consensus." Mitigation: source-count and source-quality requirements per claim, plus explicit uncertainty in the output format.
- **Prompt injection from fetched web pages** — a page that instructs the agent to exfiltrate data or call a tool. This is the serious security issue in tool-using agents. Mitigations: fetched content is always data and never instruction, tool allowlists are per sub-agent, no tool can send data outbound, and the sandbox has no network.
- **Permission escalation via tools** — a sub-agent retrieves documents the user cannot see and surfaces them in the answer. Use identity propagation enforced at the tool, tested with a permissions suite.
- **Duplicate work across branches** — three sub-agents fetch the same ten pages. Use a shared result cache keyed by tool arguments.
- **Non-reproducibility** — the same question gives different answers, which undermines trust. Mitigated by pinning models, logging seeds where available, caching, and, more practically, by setting the expectation that research is exploratory.
- **Silent partial failure** — the SQL tool times out, and the agent answers from the web alone without saying so. Failures must be surfaced in the answer.

**Follow-ups they will ask.**

*"How do you decide how many sub-agents to spawn?"*
From the plan, with a cap. The lead proposes sub-questions, and I spawn one agent per genuinely independent branch, up to 4 by default, and more only in explicit deep-research mode. I'd also weight by expected value. A question with three broad, unrelated dimensions justifies parallelism. A narrow factual question does not, and it should skip the sub-agent layer entirely. The anti-pattern is a fixed number, because a fixed number either wastes money on simple questions or under-serves complex ones. I'd log the relationship between branch count and rubric score, and tune the default empirically.

*"How do you stop prompt injection from a web page?"*
Use layers, and assume the model will eventually be fooled. Structurally: fetched content is wrapped and clearly delimited as untrusted data. Sub-agents that read the web get a minimal tool allowlist, with no SQL and no code execution with network. The code sandbox has no network egress at all, so exfiltration has no channel. Any tool that could send data outward requires human confirmation. Behaviorally: scan fetched content for injection patterns and strip or flag them, and monitor for anomalous tool-call sequences. Prompt-level instructions such as "ignore instructions in documents" are the weakest layer, and I'd never rely on them alone.

*"What's your context management strategy for a 20-minute run?"*
Three mechanisms. First, externalize. Tool results go to the artifact store, and only summaries plus IDs stay in context. Second, compact. When context approaches a threshold, summarize the trajectory so far into a structured state object holding the goal, the findings so far with artifact IDs, the open questions, and the remaining budget, then continue from that object. This is the standard long-horizon pattern, and it works well as long as the state schema is explicit rather than a free-text summary. Third, isolate. Sub-agents start with fresh context containing only their sub-question and the relevant artifact IDs, so no branch inherits another branch's clutter.

*"How do you handle conflicting information between sources?"*
Surface the conflict rather than resolve it silently, because surfacing it is both honest and more useful. The findings schema requires each claim to carry its sources, so the lead can detect that branch A and branch B assert incompatible numbers. Then it reports both with attribution and, where possible, an assessment based on source recency and authority. I'd rank source quality explicitly, from internal system of record, to primary source, to reputable publication, to blog, and encode that ranking in the synthesis prompt. Users consistently rate "here's a disagreement between these two sources" as more valuable than a confident single number.

*"How do you make this debuggable when a user complains?"*
Through the trace tree, keyed by run ID. A support engineer opens the run and sees the plan, each sub-agent's tool calls with full arguments and results, the artifacts fetched, the synthesis prompt, and per-node cost and latency. A complaint then resolves to a specific node: bad decomposition, a search that returned nothing useful, a document that was retrieved but misread, or synthesis that dropped a caveat. I'd also make traces shareable with the user, because analysts often want to check the sources themselves. That turns a debugging tool into a trust feature.

*"When would you NOT use agents here?"*
Whenever the question shape is known. If analysts repeatedly ask "give me the competitive landscape for product X", that is a workflow: fixed search queries, fixed internal reports, a fixed SQL query, and a templated synthesis. A workflow is 10x cheaper, 10x faster, reproducible, and testable. I'd instrument the question distribution and continuously promote the head of it into workflows, leaving agents to handle the tail. A team that never does this ends up paying agent prices for template work, which is the most common way these systems fail their cost justification.

*"How do you handle a question the system can't answer?"*
It must say so, and that has to be designed in. The findings schema has an explicit "could not determine" field, and the lead is required to propagate it, so the answer format includes a "what we couldn't establish" section. I'd also detect the pattern where all branches return thin findings, and route that to an explicit "insufficient information found, here's what we searched" response rather than synthesizing something from scraps. Measuring this is worth doing. On a deliberately unanswerable eval subset, the metric is the refusal rate. A system that confidently answers unanswerable questions is worse than useless for research.

---

## 12. Feature store / training-serving consistency

### Q: "Design an ML feature store — the core problem being training-serving consistency."

**Clarifying questions to ask first**

- **How many models and teams will share features, and is there actual reuse?** A feature store for one team and two models is over-engineering. The value comes from sharing. If there is no reuse, the honest answer is "you don't need this yet."
- **What's the online read latency and QPS requirement?** 10ms p99 at 50k QPS dictates an in-memory store and precomputation. 100ms at 500 QPS allows much simpler infrastructure.
- **Do you need streaming (real-time aggregate) features, or is batch enough?** Streaming features are where most of the complexity and most of the skew live. If daily batch features suffice, this problem is a quarter as hard.
- **Is point-in-time correctness a known requirement, and does the team understand label leakage?** If they have been doing random splits and naive joins, the first deliverable is fixing the training data, not building a store.
- **Who owns feature definitions — a central platform team or the model teams?** This is an org question, and it decides whether the store is a registry with governance or a shared library.
- **Buy or build?** Managed options exist and are usually right, so I'd want to know what forces a build.

**Assume:** 8 ML teams, ~40 models in production, ~600 features with meaningful reuse. Online reads 30k QPS at p99 15ms. Mix of batch, streaming, and on-demand features. Point-in-time correctness is required and currently done wrong.

**The design.**

State the core insight early. **Training-serving skew is not caused by having two stores. It is caused by having two implementations.** A feature store's job is to make one definition produce both paths.

*Feature definition.* Each feature is declared once, in a versioned spec in a repo, with its entity key, source, transformation, aggregation window, freshness SLA, owner, and type. From that single spec the system generates the batch job, the streaming job, and the online serving read. Teams write the transformation once. Where the transformation cannot be expressed declaratively, meaning SQL over the source or a constrained DSL, the escape hatch is a Python function used by *both* paths from the same code artifact. What you must never have is a batch SQL query plus a separately written Flink job that "do the same thing." That divergence is the number one cause of models that look great offline and disappoint online.

*Three feature types, three paths:*
- **Batch** (daily or hourly): computed in the warehouse, then materialized to both the offline store and the online store. An example is "user's 90-day purchase count."
- **Streaming** (seconds): computed by Flink or Spark Streaming from an event topic into the online store, with the same aggregation logic mirrored into the offline store for training. An example is "transactions on this card in the last hour."
- **On-demand / request-time**: computed at serving from the request context, for example the distance between the request IP and the account's home location. These cannot be precomputed, so the definition must be a shared function called by the serving path and by the training pipeline's backfill.

*Storage.* The offline store is the warehouse or a lakehouse such as Iceberg or Delta. It holds the full history with event timestamps, and that history is what makes point-in-time joins possible. The online store is a low-latency KV store such as Redis, DynamoDB, or ScyllaDB. It holds only the latest value per entity key, optimized for reads. Materialization jobs push from offline to online, and monitoring their **lag** is a first-class SLO, because a stale online store is silent model degradation.

*Point-in-time correctness* is the heart of the system. Training data is built from an entity dataframe of `(entity_id, event_timestamp, label)`, joined against the feature history under one constraint: take the latest feature value **as of** `event_timestamp`, respecting each feature's availability delay. People miss that last part. A feature computed by a daily batch job at 2am is not actually available for a decision at 1am, even though its logical timestamp says the previous day. Encoding each feature's *availability* time separately from its *event* time is what prevents subtle leakage. A naive `LEFT JOIN ... ON entity_id` is the classic bug, and it produces a model with suspiciously good offline metrics and no online lift.

```
   feature_spec.yaml (single definition, versioned, owned)
         │
    ┌────┴─────────────────┬──────────────────────┐
    v                      v                      v
 BATCH job             STREAM job           ON-DEMAND fn
 (warehouse)           (Flink)              (shared library)
    │                      │                      │
    ├──> OFFLINE STORE (full history + event_ts + available_ts)
    │            │
    │            └──> point-in-time join ──> TRAINING SET ──> model
    │
    └──> ONLINE STORE (latest value, KV) ──> SERVING ──> model
                                                  │
                     logged feature vector <──────┘
                                │
                  CONSISTENCY CHECK: logged online values vs. recomputed offline
```

*The consistency check is the mechanism that actually enforces the promise.* A daily job samples logged serving requests, recomputes each feature offline for that entity and that timestamp, and compares the two values. Any feature whose mismatch rate exceeds the threshold raises an alert against its owner. Without this check you have a feature store that *claims* consistency. With it, you have one that proves it. I'd treat mismatch rate as an SLO per feature.

*Serving path.* A model declares a feature view, which is the ordered list of features it needs. At inference, one batched multi-get from the online store fetches all of them by entity key. Ordering and defaults come from the registry, so a feature added to the view does not silently shift column positions, which is another classic bug. The latency budget is a multi-get at p99 under 10ms, which means keeping the feature count per request bounded and avoiding cross-region reads.

*Logging.* Every serving request logs the exact feature vector used, with feature and model versions. This does double duty. It is the input to the consistency check, and it is the source of truth for training the next model version, which sidesteps point-in-time issues entirely for online-logged features.

*Governance.* Keep a registry UI showing each feature's definition, owner, freshness, consumers, and cost. The consumer list is what makes deprecation possible, because you cannot safely delete a feature when you cannot see that four models use it.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

There is essentially no AI in a feature store, and that is worth saying explicitly. This is data infrastructure: stream processing, storage, joins, orchestration, and monitoring. It exists to serve models, and it contains none.

The only place a model might appear is anomaly detection on feature distributions. Even there I'd start with simple statistical tests such as PSI and KS before anything learned. I would **not** use a model to auto-generate features. I would not use one to decide which features a model should use, because that is feature selection during training and it has to be reproducible. And I would not use one to impute missing values at serving time unless the identical imputation runs in training.

**The hard tradeoff** — *precompute everything vs. compute on demand.*

Precomputation gives predictable low serving latency and simple serving code. However, it costs storage for every entity whether or not it is ever read, and 30M users × 600 features is a lot of rows updated on a schedule. It also introduces staleness bounded by the materialization interval. On-demand computation is always fresh and only pays for what is used. However, it puts computation in the request path, and for aggregate features that means querying source data at serving time, which is usually impossible within a 15ms budget.

I'd precompute batch and streaming aggregates, and compute only genuinely request-dependent features on demand. The interesting middle case is a high-cardinality feature that is read rarely. For those I'd use lazy materialization with a cache and accept a cold-read penalty. What changes my mind is the read/write ratio. If a feature is written for 30M entities and read for 50k, on-demand or lazy computation is dramatically cheaper. I'd want that ratio visible in the registry, so the choice is data-driven rather than a default.

**How I'd evaluate it.**

The primary metric is **training-serving consistency**, meaning the mismatch rate from the daily check, per feature. I'd target below about 0.1% for exact-valued features, and a tight tolerance for floats. This metric is the whole reason the system exists.

For operational metrics I'd track online read p50 and p99 latency, availability, **materialization lag** per feature against its freshness SLA, and the null and default rate at serving, because a spike there means an upstream pipeline broke. On the training side I'd verify point-in-time join correctness with a test suite of deliberately tricky timestamp cases, and I'd track training set build time.

Adoption metrics measure whether the platform is worth its cost. Track the number of features reused across two or more models, the time from feature idea to production availability, which should drop dramatically and is the main business case, and the share of production models reading through the store versus computing features ad hoc.

**Failure modes I name before the interviewer does.**

- **Silent materialization lag** — the batch job fails, the online store serves yesterday's values, and the model degrades subtly. This needs freshness monitoring with alerting per feature, not per job.
- **Definition drift via the escape hatch** — a team "temporarily" writes a custom serving computation and it never comes back. Prevented by the consistency check catching the divergence, and by making the sanctioned path easier than the workaround.
- **Point-in-time leakage from availability delay** — the subtlest bug in ML infrastructure. It inflates offline metrics and produces no online lift. Prevented by modeling available-time explicitly and by a test suite.
- **Hot-key overload** — a feature keyed by something skewed, such as a merchant with 30% of traffic, melts one shard. This needs key-level caching and a shard-aware design.
- **Backfill cost explosion** — adding a feature requires recomputing two years of history for 30M entities. This needs incremental backfill and cost estimation before the job runs.
- **Schema evolution breaking consumers** — a feature's type or semantics change and four models silently degrade. Features are versioned, so a semantic change creates a new version and never mutates the old one.
- **Cost with no owner** — the store accumulates hundreds of features nobody reads. This needs registry-driven usage tracking and a deprecation process with real teeth.

**Follow-ups they will ask.**

*"How do you do the point-in-time join efficiently?"*
As an AS-OF join. Sort both the entity dataframe and the feature history by timestamp within each entity, then merge. In Spark that is a window function with a range condition. In a modern warehouse there is often a native ASOF JOIN. The performance trick is partitioning the feature history by entity key and time, so the join is local, and bounding the lookback window, because you rarely need feature values from two years before the label event. For very large training sets I'd materialize the joined dataset once and version it. That way repeated experiments do not re-run an expensive join, and two experiments are actually comparable.

*"What if a team needs a feature that doesn't fit the framework?"*
Give them a first-class escape hatch rather than forcing a bad fit. That means an arbitrary Python transformation registered as a feature and packaged as a versioned artifact used by both paths. The requirement is not that the transformation be simple. The requirement is that it be *the same code* in both paths and covered by the consistency check. What I'd resist is letting a team compute features entirely outside the store and log them, because then nothing is verified. If they genuinely cannot fit, that is useful information about the framework's gaps, and I'd track those requests as a roadmap input.

*"Buy or build?"*
Buy, or adopt open source, almost always. Feast, Tecton, Databricks Feature Store, and cloud-native options solve the common cases well. Building this from scratch is a multi-quarter project that produces no differentiated value. I'd build only for a hard constraint the managed options cannot meet: an unusual latency requirement, a regulated data-residency need, or an existing streaming stack that does not integrate. Even then I'd build the thinnest layer that gives me the single-definition property and the consistency check, and use existing storage underneath.

*"How do you handle features that depend on other features?"*
Use a feature DAG with dependency-aware materialization. Derived features are computed after their inputs, and the orchestrator respects that ordering. The subtlety is freshness. A derived feature is only as fresh as its stalest input, so the registry should compute and display effective freshness transitively. Otherwise a team believes a feature is real-time when its upstream is daily. I'd cap dependency depth at two or three levels, because deep chains make lineage debugging painful and they amplify any single upstream failure across many models.

*"How does this interact with model deployment?"*
The model artifact declares the feature view and the feature versions it was trained against. Deployment then validates that all of them exist online with acceptable freshness. Deploying a model whose features are not materialized should fail loudly at deploy time, not produce nulls at serving time. On the other side, deprecating a feature checks the registry's consumer list and blocks if any live model depends on it. This coupling is what makes the store a platform rather than a library, and it requires organizational buy-in more than engineering.

*"What's the migration path from an existing mess?"*
Incremental, starting with the highest-pain model rather than a big-bang migration. Pick one model, define its features in the store, and run the store's values in shadow alongside the existing pipeline. Then use the consistency check to find where the two disagree. That exercise alone usually uncovers real bugs, and it builds the case for the platform. Then cut that model over and repeat. I'd explicitly avoid migrating everything before proving value, and I'd expect the first migration to take much longer than the estimate, because it surfaces every undocumented assumption in the old pipeline.

---

## 13. Text-to-SQL over a company database

### Q: "Design a text-to-SQL system so non-technical people can query our company database in English."

**Clarifying questions to ask first**

- **How many tables, and is there a curated semantic layer or is it raw production schema?** 40 curated analytics tables is a tractable problem. 4,000 raw tables with cryptic names and no documentation is a data-modeling project first, and I'd say so.
- **Read-only analytics, or could this touch production data?** Read-only against a warehouse replica is the only design I'd propose. Anything else is a wrong answer.
- **What happens if the answer is subtly wrong — does someone make a decision on it?** A silently wrong number is worse than an error, because nobody catches it. This drives how much verification and how much "show your work" the product needs.
- **Are there existing certified queries, dashboards, or dbt models I can learn from?** These are gold. They encode the real business definitions, and they give me few-shot examples and a semantic layer for free.
- **Who are the users — analysts who can read SQL, or executives who can't?** Analysts can verify the SQL, which changes the risk profile enormously. Executives can only verify the number, and they cannot.
- **What are the permission requirements — row-level security, column masking?** These must be enforced by the database, not by the generated SQL.

**Assume:** cloud warehouse (Snowflake/BigQuery-class), ~4,000 raw tables but ~120 curated dbt models that cover 90% of questions. Mixed audience, majority non-technical. Read-only. Row-level security exists in the warehouse. ~2,000 questions/week expected. Existing library of ~500 dashboard queries.

**The design.**

Here is the honest opening. **On the BIRD benchmark, which uses realistic messy databases, the top published systems reach about 82% execution accuracy on the test set, against a human baseline of roughly 93%.** So roughly one in five hard queries is wrong, even for state-of-the-art systems. That number should drive the entire design. The product cannot be "ask a question, get a number." It has to be built so that wrong answers are visible and correctable.

*Layer 1: the semantic layer, which is most of the work and most of the value.* I would not point a model at 4,000 raw tables. Instead I'd expose a curated set of about 120 models with clear names, column descriptions, documented business definitions such as "active_user means logged in within 28 days", declared join paths and grains, and certified metrics such as "revenue = sum of net_amount excluding internal accounts." Where dbt already has this, it is a matter of harvesting `schema.yml` and the query history. Where it does not, building it is the project. A published analysis of BIRD-style benchmarks makes the point that the data model *is* the semantic layer. Most text-to-SQL errors come from the model not knowing your business definitions, not from the model being bad at SQL syntax.

*Layer 2: retrieval.* With 120 models and hundreds of columns I cannot put the whole schema in the prompt, and I should not, because irrelevant schema actively degrades accuracy. So schema selection is a retrieval problem. Embed the table and column descriptions plus sample values, retrieve the top 10 tables for the question, and include their full DDL, descriptions, join paths, and 3 sample rows each. Sample values matter more than people expect. Knowing that `status` contains `'ACTIVE'`, `'CHURNED'`, and `'TRIAL'` stops the model from inventing `'active'`.

I'd also retrieve **similar past queries**: the 500 dashboard queries plus accumulated successful queries, embedded by their natural-language description. A near-match past query is the strongest possible few-shot example. For the repetitive head of question traffic it often just needs a parameter change.

*Layer 3: generation with verification.* The model generates SQL along with a plain-English restatement of what it is computing. Then a verification cascade runs before anything executes:
1. **Parse and validate** with a SQL parser against the real schema. Every table and column must exist, and types must be compatible. This catches hallucinated columns for free and deterministically.
2. **Static policy checks** — read-only is enforced by the connection's role, not by inspecting the SQL. Also check that required row-level predicates are present, that there are no cross-joins without conditions, and that a `LIMIT` is applied.
3. **Dry run or EXPLAIN** for cost estimation. If the plan scans 40TB, block it and suggest a narrower question. This is the difference between a helpful tool and a \$50,000 warehouse bill.
4. **Execute** with a statement timeout and a row cap, under the *user's* warehouse role, so RLS and column masking apply.

*Layer 4: presentation, which is where accuracy problems get managed.* The user sees the result, the English restatement, the SQL in a collapsible panel, and the assumptions the system made, such as "counted distinct users, excluded internal accounts, used the order_date not ship_date." Ambiguity is surfaced, not silently resolved. If "revenue" could mean gross or net, the system either asks or states which one it chose. I'd also show a sanity panel with the row count, the date range covered, and a comparison to the same metric from a certified dashboard when one exists.

```
QUESTION ──> intent/ambiguity check ──ambiguous──> clarify with user
                │
        SCHEMA RETRIEVAL (embed table/col descriptions) ──> top ~10 tables
        SIMILAR QUERY RETRIEVAL (500 certified + past successes)
                │
        LLM ──> SQL + English restatement + stated assumptions
                │
        VERIFY: parse vs schema -> policy checks -> EXPLAIN cost estimate
                │        │ fail -> repair loop (max 2, with error message)
                v
        EXECUTE as the USER (RLS, masking, timeout, row cap)
                │
        RESULT + SQL + assumptions + sanity panel ──> feedback (👍/👎/edit)
                                                          │
                                          certified query library <── analyst promotes
```

*The repair loop* is worth designing explicitly. On a validation or execution error, feed the error message back with the SQL and let the model fix it, capped at 2 attempts. This recovers a meaningful share of failures at low cost. However, it cannot fix a query that runs successfully and returns the wrong number, and that is the failure mode that matters.

*The feedback flywheel.* Every query gets thumbs and an edit affordance. Analyst-corrected queries go into a review queue, and approved ones join the certified library, which improves few-shot retrieval for everyone. Over time the head of the question distribution gets covered by certified queries, so accuracy on common questions approaches 100% while the tail stays at model-level accuracy. That asymmetry is the product strategy: make the common case certifiably right, rather than making everything slightly better.

For caching, identical questions hit a result cache with a short TTL. The query text cache lives longer, because the SQL for a given question rarely changes.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI does two things. It picks the relevant tables, which is a retrieval problem, and it writes SQL from intent plus schema. Everything else is ordinary engineering: validation, policy, permissions, cost control, execution, and presentation. And the semantic layer that makes the AI work is data modeling, not ML.

I would **not** use a model for permissions. Execute as the user and let the database enforce RLS, because a model that "remembers to add the tenant filter" is a data breach waiting to happen. I would not use a model for read-only enforcement, which is a database role, for cost limits, which are EXPLAIN plus quotas, or for validating that columns exist, which is a parser. And I would not use a model to interpret results into a business recommendation without the user seeing the numbers. I would also not let generated SQL run against production OLTP.

**The hard tradeoff** — *open-ended generation over the full schema vs. a constrained semantic layer with certified metrics.*

Open generation answers any question, including ones nobody anticipated, and it requires no upfront modeling investment. However, accuracy on messy real schemas caps out around the BIRD numbers, and the errors are silent, because a plausible number computed from the wrong join looks fine. Constrained generation over certified metrics and dimensions, which essentially means generating a semantic-layer query that compiles to SQL, gets much higher accuracy on the questions it covers and guarantees consistent business definitions. However, it returns "I can't answer that" for anything outside the model.

I'd go constrained-first with an open fallback that is clearly labeled as unverified and shows its SQL prominently. For a mostly non-technical audience making decisions on the numbers, consistent-and-limited beats flexible-and-sometimes-wrong. What changes my mind is the audience. For a room full of analysts who read the SQL anyway, open generation is a genuine productivity win, and the verification burden sits with a competent user.

**How I'd evaluate it.**

Offline, I'd build a benchmark of a few hundred questions **on your own schema**, because public benchmarks tell you about the model, not about your database. I'd build it from real questions in the analytics Slack channel, with analyst-written gold SQL. The metric is **execution accuracy**: does the generated query return the same result set as the gold query, compared as sets, order-insensitive, with a numeric tolerance. Exact SQL match is the wrong metric, because many correct queries are textually different.

I'd break the results down by difficulty: single-table, joins, aggregations with filters, window functions, and nested logic. Aggregate accuracy hides the fact that complex analytical questions are much worse. I'd also track the **valid-execution rate**, meaning does the query run at all, separately from correctness, because those two have different fixes.

Online, I'd track thumbs rate, query edit rate, which is a strong implicit correctness signal because it counts how often a user modifies the SQL, re-ask rate, the share of questions answered from the certified library versus generated, warehouse cost per question, and p95 latency. The metric I'd watch hardest is the **silent error rate**. I'd estimate it by having an analyst audit a random sample of about 50 answered questions per week and judge correctness. Everything else can look healthy while that number is bad.

**Failure modes I name before the interviewer does.**

- **Silently wrong joins** producing plausible numbers — the defining failure. Fan-out on a one-to-many join inflates sums and nothing errors. Mitigated by declared join paths and grains in the semantic layer, and by sanity comparison against certified metrics.
- **Business-definition mismatch** — the system's "active user" is not finance's. Mitigated by certified metrics and by stating the assumptions in every answer.
- **Runaway query cost** — a cross join on two billion-row tables. Mitigated by EXPLAIN gating, row caps, timeouts, and per-user warehouse quotas.
- **Ambiguity resolved silently** — "last quarter" meaning fiscal or calendar. This must be surfaced, and where the system guesses, it must say so.
- **Stale schema in the retrieval index** — a column is renamed and the model generates against the old name. This needs schema sync on every dbt deploy, with monitored lag.
- **Permission bypass** — the classic disaster, where the service account has full access and the generated SQL is trusted to filter. Prevented by executing under the user's role, and tested with a permissions suite.
- **Over-trust** — users treat outputs as authoritative and stop checking. Mitigated by explicit confidence and verification status in the UI, and by making the certified-versus-generated distinction visually obvious.

**Follow-ups they will ask.**

*"How do you handle 'show me revenue last quarter' when revenue is defined three ways?"*
I do not let the model choose silently. If the question maps to an ambiguous metric, the system either asks, for example "gross bookings, net revenue, or recognized revenue?", or it picks the certified default and states that choice prominently in the answer. The deeper fix is organizational. The certified metric library forces the company to actually decide, and the text-to-SQL project frequently becomes the forcing function for defining metrics that have been ambiguous for years. I'd flag that in the interview, because it is the kind of non-technical work that determines whether the project succeeds.

*"What if the question needs a table that isn't in the curated set?"*
Fall back to retrieval over the full schema, with a clear "this used uncurated tables, please verify" label, and log it as a coverage gap. Those logs are the prioritized backlog for what to curate next. If 40 people a week ask about a table that is not modeled, that is an obvious data-modeling task. I'd rather have a measured coverage gap than a system that silently produces lower-quality answers on uncurated data with no visibility.

*"How do you keep warehouse costs under control?"*
Use several layers, because a single one will be bypassed. EXPLAIN-based estimation blocks queries above a byte-scan threshold before execution. Statement timeouts and row limits cap the damage from anything that gets through. Per-user and per-team credit quotas in the warehouse itself are the hard backstop. Result caching handles repeats, which are a large fraction of traffic. And I'd default to querying pre-aggregated tables where they exist rather than raw fact tables. That is usually a 100x cost difference for the same answer, and it is a semantic-layer routing decision, not something the model should reason about.

*"How do you handle multi-turn — 'now break that down by region'?"*
Keep the conversation state as the previous SQL plus its result schema, and treat the follow-up as a transformation of that query rather than a new generation from scratch. That is both more accurate and cheaper. I'd represent the query structurally, as a parsed AST or a semantic-layer query object, so "break down by region" is a mechanical group-by addition rather than a regeneration that might change the filters too. Where the follow-up cannot be expressed as a transformation, fall back to full regeneration with the prior turn as context, and re-state the assumptions, because they may have changed.

*"Which model would you use, and would you fine-tune?"*
Start with the strongest available general model, because SQL generation quality tracks general reasoning closely and the cost per query is small relative to warehouse cost. I'd invest in the semantic layer and retrieval before touching the model, because those move accuracy far more. Fine-tuning becomes attractive once you have a few thousand certified query pairs on your schema and you want to cut cost or latency. A smaller fine-tuned model on your own schema can beat a larger general one, because most of the difficulty is schema-specific knowledge rather than SQL skill. However, I'd treat that as a phase-two optimization with a measured baseline.

*"How would you support 'why did revenue drop' style questions?"*
That is not text-to-SQL. It is root-cause analysis, and conflating the two is a common product mistake. It requires generating and running many queries, meaning dimensional decompositions across region, segment, product, and cohort, and then ranking the contributions to the change. I'd implement it as a structured workflow, not as free-form generation: enumerate the dimensions from the semantic layer, run the decomposition, compute each dimension's contribution to the variance, and present the top drivers with charts. The LLM's role is picking which dimensions are plausible and writing the narrative. The analysis itself is deterministic arithmetic, and that is the right split.

---

## 14. Model and data drift detection

### Q: "Design a system to detect and handle model and data drift in production."

**Clarifying questions to ask first**

- **How many models, and are they the same type?** Twenty GBDTs sharing a feature store lets me build one monitoring system. A mix of GBDTs, embeddings, and LLM endpoints needs different detectors for each.
- **How delayed are the labels?** Immediate labels such as ad clicks let me monitor real performance directly, so drift detection is a secondary concern. 90-day labels, as in credit, churn, and fraud, mean proxy signals are the primary defense.
- **What's the cost of acting on a false alarm vs. missing real drift?** Auto-retraining on every alert is expensive and risky. Ignoring alerts makes the system decorative. This answer sets the alerting threshold.
- **Is retraining automated and safe today?** Detection is useless without a response path. If retraining takes a human three weeks, the design should focus on making retraining cheap, not on fancier detectors.
- **Are there regulatory requirements around model monitoring and revalidation?** In finance and healthcare this is mandated with specific documentation, which changes what I build.
- **How segmented is the population?** Aggregate stability with per-segment collapse is common, and it is only visible if you slice.

**Assume:** 40 models across 8 teams, mostly tabular GBDTs plus a few embedding-based rankers and two LLM-backed features. Label delay ranges from minutes to 60 days. Automated retraining exists for 15 of the 40. Some models are regulated and need documented monitoring.

**The design.**

Drift is not one thing, and naming the four kinds precisely is high signal.

1. **Data/covariate drift** — $P(X)$ changes, meaning the input distributions move. It is detectable immediately, with no labels needed.
2. **Prediction drift** — $P(\hat{Y})$ changes, meaning the model's output distribution moves. It is also immediate, and it is often the most sensitive early warning.
3. **Concept drift** — $P(Y \mid X)$ changes, meaning the relationship itself changed. It is only detectable with labels, and it is the kind that actually destroys model value.
4. **Upstream data quality breaks** — a schema change, a broken join, a unit change from meters to feet. This is not really drift, and it is by far the most common cause of production model failures. Any monitoring system that only does statistical drift, and misses "this column became all nulls at 3am", is solving the wrong problem first.

*Architecture.* Serving logs, meaning inputs, predictions, feature vectors, and model version, stream to a monitoring store. A scheduled job then computes the following per model, per segment, and per window, using hourly and daily windows.

- **Data quality checks first**, because they have the highest yield: null rate per feature, cardinality, type conformance, range violations, and freshness. These are assertions with thresholds, and they catch the majority of real incidents. Use Great Expectations-style checks, run as a gate.
- **Distribution drift** per feature against a fixed reference window, which is the training distribution. For numeric features use **PSI** and the Kolmogorov–Smirnov statistic. For categorical features use PSI and chi-square. The conventional PSI reading is **< 0.1 no significant shift, 0.1–0.25 moderate, > 0.25 significant**. Those bands are widely used, but they are a rule of thumb rather than a statistical guarantee. So I'd calibrate them per feature by measuring historical PSI over periods when the model was known to be healthy.
- **Prediction drift** — PSI on the score distribution, plus the mean and the quantiles of predictions over time. It is cheap, label-free, and it moves early.
- **Performance monitoring** where labels exist, with a maturity-aware window. A metric is only computed on data old enough for its labels to have arrived. For delayed-label models I'd compute metrics on a rolling matured window and report the label maturity explicitly, so nobody misreads a partially labeled recent window as a performance drop.
- **Proxy performance** to cover the label gap: agreement with a human-reviewed sample, calibration drift, meaning do predicted probabilities still match observed rates on matured data, and score-distribution stability.

Segmentation is not optional. Every metric is computed per key segment, meaning geography, customer tier, device, channel, and product line. A model can be perfectly stable in aggregate while one segment collapses, and the segment that collapses is usually the one with a business owner who will notice loudly.

Statistical care matters here. With 600 features × 40 models × 10 segments, naive per-feature p-value alerting produces thousands of daily false alarms. So I'd use effect-size thresholds such as PSI rather than p-values, because p-values flag trivially small shifts at high sample size. I'd apply multiple-testing correction, and I'd require persistence, meaning the drift must hold for N consecutive windows. Then I'd rank alerts by **impact-weighted drift**, which is a feature's PSI multiplied by its importance in the model. A high-drift feature with 0.1% importance is noise. A moderate drift in the top feature is an emergency.

```
SERVING ──> logs (features, prediction, model_ver, ts) ──> monitoring store
                                                                │
   scheduled per model × segment × window:
      1. DATA QUALITY assertions (nulls, types, ranges, freshness)  <- highest yield
      2. FEATURE DRIFT (PSI, KS) vs. training reference
      3. PREDICTION DRIFT (PSI on scores)
      4. PERFORMANCE on matured labels + calibration
                                                                │
        rank by PSI × feature_importance, require persistence    │
                                                                v
                                       ALERT ──> triage runbook ──> RESPONSE
                                                                     │
    ┌────────────────┬───────────────────┬────────────────┬──────────┴────────┐
 no action      fix upstream        retrain            rollback         fall back
 (expected)     (data bug)      (shadow->canary)    (prev version)     (rules/baseline)
```

*The response path is the part people forget, and it is where I'd spend my design time.* An alert must route to a runbook, not to a Slack channel nobody reads. Triage asks four questions. Is this a data quality bug, which is the most common case and means fixing upstream? Is it an expected seasonal shift, which means annotate and suppress? Is it a real distribution change with performance impact, which means retrain? Or is it a catastrophic break, which means roll back or fall back to a baseline?

Automated retraining is appropriate for models with fast labels and a proven pipeline, and it must be gated. The retrained model has to beat the incumbent on a held-out set *and* pass a shadow comparison before promotion. Auto-retraining without a gate is how a data quality bug becomes a permanently baked-in model regression. For regulated models, retraining requires documented revalidation and cannot be automatic.

I'd also maintain a **fallback** per model: the previous version kept warm, and a simple rules or heuristic baseline underneath that. Being able to say "we degraded to the rules baseline while we investigated" is a much better incident outcome than "the model kept serving garbage for six hours."

*For LLM-backed features*, drift looks different. There is input distribution drift, meaning users ask new things. There are vendor model updates, which is why problem 5 has the frozen canary. And there is output drift, meaning response length, refusal rate, tool-call rate, and judge scores. I'd monitor those with the same infrastructure but different detectors.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

Almost none. This is statistics, scheduling, storage, and alerting. Drift detection is one of the clearest cases where the sophisticated-sounding option, meaning train a model to detect drift, is worse than PSI plus a null-rate check.

There is one legitimate ML use: a **domain classifier** trained to distinguish reference-window data from current-window data. If it can separate them with high AUC, then multivariate drift exists even when no single feature moved. That catches correlation-structure changes that univariate tests miss. I'd add it as a secondary detector, not a primary one, because it is harder to interpret. I would **not** use a model to decide whether to retrain, because that is a policy with thresholds. I would not use one to set alert thresholds, because those are calibrated from historical data. And I would not use one to explain drift to on-call, although an LLM writing the incident summary from the computed statistics is a genuinely nice quality-of-life feature and a fair place to use one.

**The hard tradeoff** — *automatic retraining on drift vs. human-gated retraining.*

Automatic retraining keeps models fresh without headcount, and it responds within hours, which is worth a lot in fast-moving domains such as fraud and recommendations. However, it can bake in a data bug, it makes the production model a moving target that is hard to reason about during an incident, and in regulated contexts it is not permitted. Human-gated retraining is safer and auditable, but it is slow, so a model can stay degraded for weeks waiting in someone's queue.

I'd split by model criticality and label speed. Use automatic retraining with hard promotion gates for high-velocity, low-stakes models where labels arrive fast. Use human-gated retraining for high-stakes and regulated models. Either way, the retrained model always goes through shadow and canary rather than direct promotion. What changes my mind is the observed track record. If six months of automated retrains have never produced a regression that the gates missed, I'd widen the automatic set. A single bad promotion sends me back to gating.

**How I'd evaluate it.**

Evaluate the detector, which people rarely do. First, **detection recall**: replay historical incidents such as known outages, schema changes, and performance drops that were eventually noticed, and measure what fraction the system would have caught and how many hours earlier than the actual discovery. That is the ROI number. Second, **false alarm rate**: alerts per model per week that triage marks as no-action. Above roughly two or three a week per team, alerts get ignored and the system is worthless whatever its recall. Third, **time to detection** and **time to resolution** per incident class.

I'd also run synthetic injection. Deliberately corrupt a feature in a shadow stream by nulling it out, shifting its mean by 2σ, or swapping its units, then verify the system fires within the expected window. That is a continuous test of the monitoring itself, and it catches the embarrassing case where monitoring silently stopped working, which happens more often than anyone admits.

**Failure modes I name before the interviewer does.**

- **Alert fatigue** — the dominant failure. Hundreds of statistically significant but meaningless alerts arrive, and everyone mutes the channel. Mitigated by effect sizes, importance weighting, persistence requirements, and aggressive tuning of what pages versus what only appears on a dashboard.
- **Reference window rot** — comparing against a training distribution from two years ago flags every normal evolution. This needs a policy: a fixed reference for "have we left the training distribution", plus a rolling reference for "did something change suddenly."
- **Seasonality misread as drift** — Black Friday fires every detector. This needs seasonal baselines, or year-over-year comparison for known-seasonal features.
- **Aggregate stability hiding segment collapse** — the most common way real degradation is missed.
- **Drift with no performance impact** — a feature moves but the model does not care. This wastes triage time, and it is mitigated by importance weighting and by always asking for the performance evidence before acting.
- **Performance drop with no drift** — the labels changed meaning, or the business process changed. Detectors will not see it. Only label monitoring and human reports will.
- **Monitoring the model but not the pipeline** — the feature store's materialization job failed, so the model serves stale features. That looks like *less* drift, not more. Freshness monitoring is separate and essential.
- **The monitoring system itself failing silently** — the job stops, no alerts fire, and everyone assumes health. This needs a heartbeat and the synthetic injection test.

**Follow-ups they will ask.**

*"What's your reference window?"*
Two of them, deliberately. A **fixed** reference, which is the training distribution, answers "are we operating outside what the model learned." That is the question that matters for validity. A **rolling** reference, typically the previous 7 or 28 days, answers "did something change suddenly." That is the question that matters for incidents. They fire on different things and both are useful. Gradual drift shows only against the fixed reference, while a schema break shows immediately against the rolling one. Reporting both prevents the common confusion where a team suppresses gradual-drift alerts and then cannot see that they have drifted a long way from training.

*"How do you handle drift when labels take 60 days?"*
Lean on the label-free signals and on proxies. Feature and prediction drift give same-day warning. Calibration on matured data gives a delayed but trustworthy read. In between, I'd invest in fast proxy labels. A human-reviewed sample of even 200 items a week gives useful signal. Early indicators that correlate with the eventual label help too, because a first-payment default predicts eventual charge-off. So do business metrics that move faster than labels. I'd also report label maturity explicitly alongside every performance metric, because the most common analytical mistake in delayed-label settings is reading a partially matured recent window as a performance cliff.

*"How do you avoid alert fatigue?"*
Tier the responses. Very few conditions should page: a data quality break on a critical feature, a performance drop beyond a wide threshold on matured labels, or a materialization failure. Most drift belongs on a weekly review dashboard, not on anyone's phone. Then use effect size over significance, weight by feature importance, require persistence across windows, and group correlated alerts into a single incident, because twenty features drifting at once is one upstream problem, not twenty alerts. Finally, measure the false-alarm rate, and treat a rise in it as a bug in the monitoring system with the same urgency as a production bug.

*"When do you retrain versus rebuild?"*
Retraining means the same features and the same architecture on fresh data. It handles ordinary distribution shift, and it should be routine and cheap. Rebuilding is warranted when retraining stops recovering performance, because that signals the relationship changed in a way the current feature set cannot capture. It is also warranted when the world genuinely changed, such as a new product line, a new fraud modality, or a new regulation. The diagnostic is straightforward: retrain on recent data and evaluate. If the retrained model recovers, it was distribution shift. If it does not, you need new features or a new formulation. I'd track "performance recovered by retraining" as a metric per model, because a declining trend is the signal that a rebuild is due.

*"How do you monitor an embedding-based system?"*
Different detectors, same framework. Monitor the embedding distribution, using mean cosine distance to the training centroid or PSI over projected dimensions. Monitor the retrieval score distribution, because a drop in top-1 similarity across the board means queries are moving away from the corpus. Monitor the share of queries with no good match above the threshold, and the click or engagement rate on retrieved results. Monitor corpus drift separately too, because the index changing under a static query distribution is a distinct failure. And track the embedding model version everywhere, because the most dramatic drift event in these systems is a silent model upgrade.

*"How does this fit with the eval system from question 5?"*
They are the same system with different detectors, and I'd build them on shared infrastructure: the same logging, the same metric store, the same segmented time-series comparison, and the same alerting and triage. What differs is what gets computed. Tabular models need statistical distances and performance metrics. LLM products need assertion rates and judge scores. Building them separately, which is the default in most organizations, means two dashboards, two on-call runbooks, and no shared view when an incident spans both. I'd unify them explicitly and treat model quality observability as one platform.

---

## Closing note on what this round is actually testing

Read back over the fourteen answers and notice what the *design* sections spend their words on. Connectors, queues, idempotency keys, ACL propagation, bounding boxes, point-in-time joins, priority classes, audit logs, review queues, and runbooks. The model choice is usually one or two sentences, such as "a cross-encoder reranker", "a GBDT", or "a vision-language model", and it is rarely the interesting decision.

That ratio is the message. In every one of these systems the AI is a component inside a much larger piece of ordinary software. The failure modes that kill the product in production are almost never "we picked the wrong model." They are stale permissions, double-executed refunds, unlogged feature vectors, alert fatigue, an OCR pipeline quietly degrading, and a vendor silently updating a model.

Four habits separate a strong performance from an average one.

- **Say what you would not use a model for.** It is the fastest way to demonstrate judgment, and most candidates never do it.
- **Name the human in the loop.** Reviewers, approvers, on-call, analysts. Systems that assume full automation on day one do not survive contact with production.
- **Design the failure path as carefully as the happy path.** Say what serves when the model is down, what happens on a timeout, and what the rollback looks like.
- **Quantify.** Give QPS, p95, dollars per thousand requests, and error budgets. Even rough numbers, clearly labeled as assumptions, show you have built something before.

And when you genuinely do not know a number, say "I'd assume roughly X and measure it." Interviewers trust that far more than a confident fabrication.

---

## Sources for cited figures

Numbers in this document that are not marked as assumptions come from:

- [BIRD-SQL benchmark leaderboard](https://bird-bench.github.io/) — top execution accuracy ~82% on test, human baseline 92.96% (used in problem 13).
- [Your Data Model Is the Semantic Layer — MotherDuck](https://motherduck.com/blog/bird-bench-and-data-models/) — argument that text-to-SQL errors trace to business/data modeling rather than SQL generation (problem 13).
- [Hybrid Search: BM25, Vector & Reranking Reference 2026 — Digital Applied](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026) — WANDS nDCG figures: BM25 0.6983, dense KNN 0.6953, RRF hybrid 0.7068, hybrid with field boosting 0.7497 (problems 1 and 4).
- [How Continuous Batching Enables 23x Throughput in LLM Inference — Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference) — continuous batching throughput gains over naive batching (problem 8).
- [Measuring Data Drift with the Population Stability Index — Fiddler AI](https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index) and [Population Stability Index (PSI) Metrics — Arthur](https://docs.arthur.ai/docs/population-stability-index-psi-metrics) — the conventional PSI bands of 0.1 and 0.25 (problem 14).

All latency budgets, cost figures, corpus sizes, QPS numbers, and traffic assumptions in the "Assume" blocks are stated scenario assumptions, not measurements. Any figure introduced with "I'd assume" or "(assumption)" should be treated as a planning estimate to be verified against your own system.
