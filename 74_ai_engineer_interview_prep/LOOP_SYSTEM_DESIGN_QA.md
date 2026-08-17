# ML / AI System Design Round — Question & Answer Bank

Fourteen design prompts with complete worked answers. These are the prompts that actually get asked in the ML/AI system design loop, written so you can study a full answer and reproduce its shape under pressure.

The single most common failure in this round is spending forty-five minutes on model selection. In every one of these systems, roughly **20% of the work is AI and 80% is ordinary engineering** — ingestion, chunking, permissions, idempotency, retries, queues, caches, schema migrations, integrations, monitoring, on-call. Interviewers are hiring someone who can ship the whole thing. A candidate who says "we'll use a good embedding model" and then spends thirty minutes on the ingestion pipeline, ACL propagation, and eval harness scores far higher than one who compares six embedding models and never mentions how a document gets into the index.

Numbers below are either sourced (linked at the end) or explicitly marked as assumptions. In an interview, saying "I'd assume ~200ms p50 for the reranker, I'd measure it" is strictly better than stating a fake benchmark.

---

## The framework (read once, then spend your time on the problems)

Six moves, in order. Total time in a 45-minute round is in brackets.

**1. Clarify, then commit [5 min].** Ask 4–6 *specific* questions — not "what's the scale" but "are permissions per-document or per-folder, and do they change hourly or monthly?" Then state your assumptions out loud and proceed. Never ask more than six; interviewers read endless clarification as stalling.

**2. Sketch the data flow before the model [5 min].** Where does data enter, where does it rest, what shape is it in when the model sees it, where does the output go, who consumes it. Draw the boxes. The model is one box.

**3. Name the AI surface precisely, and its boundaries [5 min].** Say exactly which decision the model makes, and say out loud what you would *not* use a model for. "Permissions are enforced in the query filter, never by the LLM" is the single highest-signal sentence available to you in this round.

**4. Build the offline path and the online path [10 min].** Ingestion/training is a batch system with backfills, versioning, and idempotency. Serving is a low-latency system with caching, timeouts, and fallbacks. They must share feature/embedding logic or you get skew.

**5. Evaluation before scale [8 min].** A golden set, an offline metric, an online metric, and a guardrail metric. If you can't say how you'd know it got worse, you haven't designed it.

**6. Failure modes and the one hard tradeoff [7 min].** Name failures before the interviewer does. Then pick the genuine fork in the road, argue both branches, and say what evidence would flip you.

Two habits that carry every answer: **quantify** (QPS, doc count, p95, dollars per 1k requests) and **degrade gracefully** (what serves when the model is down — usually BM25, a cached result, or a rules baseline).

---

## 1. RAG for enterprise document search

### Q: "Design a RAG system for enterprise document search. It's multi-tenant, and permissions matter."

**Clarifying questions to ask first**

- **Are permissions per-document, per-folder, or per-field, and how often do they change?** Per-folder ACLs that change monthly let me denormalize labels onto chunks and refresh nightly. Per-document ACLs changing hourly force a live authorization check at query time against the source system, which adds a network hop into my p95.
- **Is tenant isolation a compliance requirement (separate indexes) or a logical one (filter on tenant_id)?** Compliance means one index per tenant, which changes my cost model from "one big index" to "10,000 small indexes" and makes small tenants expensive.
- **What's the document mix — clean HTML/Confluence, or scanned PDFs and spreadsheets?** Scanned PDFs mean an OCR stage and a whole class of extraction failures; spreadsheets mean chunking by row groups, not by tokens.
- **Do users expect answers with citations, or a ranked list of documents?** Generated answers need hallucination guardrails and a much heavier eval harness. A ranked list is a search problem and I might not need an LLM in the response path at all.
- **What's the freshness SLA — is a document searchable within seconds of upload, or is nightly fine?** Seconds means a streaming ingestion path with a write-ahead queue; nightly means a much simpler batch job.
- **Corpus size and QPS?** 1M chunks and 5 QPS is a single Postgres box with pgvector. 5B chunks and 5k QPS is a sharded distributed index with a very different cost story.

**Assume:** 5,000 tenants; largest has 20M documents, median has 3,000. ~500M chunks total. Per-document ACLs sourced from the customer's IdP, changing on the order of hours. Mixed content, 30% scanned. Users want cited answers. Freshness SLA of 5 minutes. Peak 300 QPS. Target p95 of 3 seconds to first token.

**The design.**

Two systems that share nothing but a schema: an **ingestion plane** and a **query plane**.

Ingestion starts at connectors — SharePoint, Google Drive, Confluence, S3. Each connector runs a cursor-based incremental sync and publishes a change event per document to Kafka, partitioned by tenant so one huge tenant can't starve others. This is the part candidates skip and it's most of the work: connectors need OAuth token refresh, rate-limit backoff, deleted-document tombstones, and resumable cursors so a crash doesn't force a full re-crawl.

A parsing worker consumes the event, fetches bytes, and routes by type. Native PDFs go through a text extractor; scanned ones go to OCR. I'd emit a `parse_confidence` score and quarantine anything below threshold to a dead-letter queue with a human review UI — silently indexing garbage OCR is how you poison a corpus.

Then chunking. I'd chunk on structural boundaries (headings, sections) with a target of ~500 tokens and ~15% overlap, and — importantly — prepend the document title and heading path to each chunk before embedding. That header injection is cheap and reliably beats raw chunk text on retrieval quality, because a chunk that says "the limit is 30 days" is meaningless without "Refund Policy > Enterprise Tier" attached.

Each chunk gets embedded and written to the index with a metadata payload: `tenant_id`, `doc_id`, `acl_hash`, `source`, `updated_at`, `chunk_ordinal`. I'd use a hybrid index — dense vectors plus BM25 — because on real enterprise corpora full of product codes, error strings, and acronyms, lexical matching catches things dense embeddings miss entirely. Published hybrid comparisons show fusing BM25 and dense with reciprocal rank fusion beating either alone by roughly 1–7% nDCG depending on the domain, with the biggest gains on jargon-heavy corpora.

Critically, **embedding is content-addressed**: the key is `hash(chunk_text + model_version)`. Re-syncing an unchanged document costs zero embedding dollars. At 500M chunks this is the difference between a \$50k re-index and a \$0 one.

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

Query plane. Request arrives with a user identity. First I resolve that user's authorization set — group memberships and accessible ACL hashes — from a cache keyed by user with a short TTL (say 5 minutes), refreshed asynchronously from the IdP. **This filter is applied inside the vector search as a pre-filter, not as a post-filter, and never by the LLM.** Post-filtering is both a correctness bug (you ask for 100 and get 3 after filtering) and a latency bug. Pre-filtering with a selective predicate degrades HNSW recall, so for tenants with very restrictive ACLs I'd over-fetch — request $k = 500$ to return 100 — and monitor a filtered-recall metric.

Retrieval: hybrid search returns ~100 candidates, fused with RRF. A cross-encoder reranker cuts that to the top 8. Reranking is the highest-leverage quality knob in RAG and I'd budget ~150–250ms for it (assumption; measure on your shortlist size). Then synthesis: an LLM gets the 8 chunks with IDs, and is instructed to answer only from context and cite chunk IDs. A post-processor validates that every cited ID actually appeared in context and drops uncited claims where possible.

Caching sits at three layers: ACL sets, embedding of the query string, and a full response cache keyed by `(normalized_query, acl_hash)` — the ACL hash in the key is what makes response caching safe in a multi-tenant permissioned system.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is in exactly three places: the embedding model, the reranker, and the answer synthesizer. That's it. I would **not** use a model for permissions (a filter predicate, always), not for routing queries to tenants (a lookup), not for deciding document freshness (a timestamp), and not for query understanding in v1 — no LLM query rewriting until I have data showing it helps, because it adds a serial LLM call to every request and often hurts precision on short keyword queries. I'd also not use an LLM to decide *whether* to retrieve; a cheap classifier or just always-retrieve is fine.

**The hard tradeoff** — *Denormalized ACL labels on chunks vs. live authorization at query time.*

Denormalizing means every chunk carries an `acl_hash`, retrieval is one fast filtered search, and p95 stays low — but a permission revocation isn't effective until the reindex lands, which is a real data-leak window. Live authorization means calling the source system to check access on retrieved results, which is always correct but adds 50–300ms and a hard dependency on a system you don't control, and breaks pre-filtering entirely.

I'd take denormalized labels with a fast revocation path: revocations are a high-priority Kafka topic processed within seconds, separate from the normal sync lane. What changes my mind is the regulatory posture — if this serves healthcare or defense customers where a five-minute leak window is unacceptable, I flip to live checks and eat the latency, probably with a hybrid where denormalized filtering narrows the set and a live check validates the final 8.

**How I'd evaluate it.**

Offline: a golden set of 300–500 real queries per major tenant with labeled relevant chunks. Measure **recall@100** for the retrieval stage (this is the ceiling — nothing downstream recovers a missed chunk) and **nDCG@8** after reranking. Separately, a permissions test suite: for each of N user/document pairs with known access, assert the document never appears for unauthorized users. That suite runs on every deploy and a single failure blocks release.

End-to-end answer quality: an LLM-as-judge rubric scoring groundedness (is every claim supported by a cited chunk), completeness, and citation validity — calibrated against ~200 human-labeled examples so I know the judge's agreement rate before trusting it.

Online: click-through on citations, session-level "did the user reformulate" rate (reformulation is a strong negative signal), explicit thumbs, deflection rate if this feeds support, and p50/p95 time-to-first-token. Guardrail metrics: rate of answers with zero citations, and rate of "I don't know" — a spike in either means retrieval broke.

**Failure modes I name before the interviewer does.**

- **Stale ACLs after a revocation** — the leak window above; mitigated by the priority lane and a nightly full reconciliation.
- **Pre-filter recall collapse** — a user with access to 0.01% of a tenant's corpus gets HNSW returning almost nothing relevant, because graph traversal wanders into inaccessible regions. Mitigation: route highly selective queries to an exact/IVF path or a per-user partition.
- **OCR garbage silently indexed** — quarantine on low parse confidence, and monitor the share of chunks with unusual character distributions.
- **Noisy-neighbor tenant** — one customer syncing 20M documents starves everyone. Per-tenant partitions and rate quotas on the ingestion lane.
- **Embedding model upgrade** — you cannot mix embedding versions in one index. Needs dual-write, shadow index, and a cutover, which for 500M chunks is a multi-day, five-figure operation. Design for it on day one by putting `model_version` in the index name.
- **Chunk-boundary answer loss** — the answer straddles two chunks and neither scores. Mitigated by overlap plus a "fetch neighbors of top hits" expansion step.
- **Prompt injection from documents** — a document that says "ignore previous instructions." Treat retrieved text as untrusted data, never instructions; keep tool access out of this path entirely.

**Follow-ups they will ask.**

*"How do you handle a document that's 400 pages?"*
Chunk it structurally and index every chunk, but also generate a document-level summary embedding so that "what is the master services agreement about" retrieves the document rather than a random clause. At query time, if several top hits come from one document, I collapse them and expand context within that document rather than showing eight near-duplicate chunks. For very long documents I'd add a hierarchical layer: section summaries indexed alongside leaf chunks, so retrieval can land on a section and then drill down.

*"A user says the answer is wrong. How do you debug it?"*
The trace is the product. Every request logs query, resolved ACL hash, retrieval candidates with scores, rerank scores, final chunks, prompt, and completion, under a request ID the user can quote. Debugging is then a decision tree: was the correct chunk in the index at all (ingestion bug), was it retrieved in the top 100 (retrieval bug), did it survive reranking (rerank bug), or did the LLM ignore it (synthesis bug)? Each branch has a different owner and fix. Without this trace you're guessing, and most "the LLM hallucinated" reports turn out to be ingestion failures.

*"How do you keep costs down at 300 QPS?"*
Embedding cache by content hash kills re-embedding costs. Response cache on `(query, acl_hash)` typically absorbs 20–40% of enterprise search traffic because people ask the same policy questions (assumption — measure it). Use a small model for synthesis and reserve the large one for queries the reranker flags as low-confidence. Cap retrieved context at 8 chunks; the marginal quality of chunks 9–20 is near zero while the token cost is linear. Track dollars per resolved query as a first-class dashboard metric, not a quarterly finance exercise.

*"Why not just put everything in a 1M-token context window?"*
Cost and latency scale with context, and long-context recall degrades in the middle of the window. But the real reason is permissions: you cannot stuff a tenant's whole corpus into a prompt when different users see different subsets — you'd need a per-user assembly step, which is retrieval with extra steps. Long context is a great *complement*: retrieve at the document level, then put a handful of whole documents in the window instead of fragments.

*"How do you support 'find me the latest version of X'?"*
That's a metadata query wearing a semantic query's clothes. I'd detect version/recency intent and apply a sort or a recency boost on `updated_at`, plus a document-lineage graph so superseded versions are demoted. Pure vector search has no notion of "latest" — embeddings of v3 and v7 of a policy are nearly identical, so the model literally cannot help here. This is a good example of the 80%: the fix is metadata plumbing, not a better model.

*"How would you add a new connector?"*
Behind a stable interface: `list_changes(cursor)`, `fetch(doc_id)`, `get_acl(doc_id)`, `resolve_user_groups(user)`. Everything downstream is shared. The work per connector is auth, pagination, rate limits, and ACL model translation — that last one is the hard part, since every system models permissions differently (SharePoint's inheritance vs. Drive's sharing links vs. Confluence spaces). I'd normalize to a principal-set model and write conformance tests per connector.

*"What if a tenant has 20M documents and 3 users?"*
The index is oversized for the traffic. I'd tier storage: hot tenants on memory-resident HNSW, cold tenants on disk-based indexes (DiskANN-style) or even lazily built on first query with a warmup penalty. Cost per tenant should be roughly proportional to corpus size for storage and to QPS for compute; conflating them is how RAG products lose money on enterprise accounts.

---

## 2. Customer support agent that takes actions

### Q: "Design a customer support agent that can actually take actions — issue refunds, change account settings, cancel subscriptions."

**Clarifying questions to ask first**

- **What's the maximum blast radius of a single action — a \$20 refund or a \$20,000 wire?** Small and reversible means I can let the agent act autonomously with post-hoc audit. Large or irreversible means every action goes through human approval or a hard policy engine, and the agent becomes a drafting tool.
- **Do we have an existing rules-based refund policy, or is it tribal knowledge in agents' heads?** If a documented policy exists, it becomes a deterministic policy engine and the LLM only extracts arguments. If it's tribal, my first project is codifying it — and I'd say that out loud, because that's the actual work.
- **Is the agent user-facing or agent-facing (a copilot for human reps)?** Copilot-first is dramatically lower risk and gets you the training data and eval set to justify going autonomous later.
- **What does the downstream billing system look like — does it have idempotency keys?** If not, my retry story is broken from the start and I need an idempotency layer before I write a line of agent code.
- **What's the regulatory surface — chargebacks, GDPR deletion, financial disclosure requirements?** Regulated actions need mandatory disclosure text and immutable audit logs with retention.
- **What fraction of contacts do we want to deflect, and what's the cost of a wrong action vs. an escalation?** This is the objective function. If a bad refund costs 50x an unnecessary escalation, I tune the confidence threshold hard toward escalation.

**Assume:** consumer subscription business, 40k contacts/week. Actions: refund (up to \$200), cancel subscription, change plan, update email/address, resend receipt. A documented refund policy exists. User-facing chat, with human escalation available. Billing is Stripe-like with idempotency key support. Goal: deflect 50% of contacts with a wrong-action rate under 0.1%.

**The design.**

The core architectural claim: **the LLM decides *what* to do and extracts *arguments*; a deterministic policy engine decides whether it's *allowed*; a durable workflow engine actually *executes* it.** Three separate components with three separate failure modes. Collapsing them into "the LLM calls the Stripe API" is the design that gets you a headline.

Flow. A message arrives over chat. First, before any model runs, we load context deterministically: user ID, subscription state, payment history, prior refunds in the last 90 days, open tickets, and account flags. This is a plain service call — the agent should never have to "figure out" who the user is.

Then the orchestrator. I'd use a tool-calling loop with a strong model, given a small set of tools split into two classes. **Read tools** (`get_order`, `get_subscription`, `search_help_center`) are free to call, unlimited, no approval. **Write tools** (`issue_refund`, `cancel_subscription`, `change_email`) are proposals, not executions — the model calls `propose_refund(order_id, amount, reason_code)` and what comes back is a decision from the policy engine, not a completed refund.

The policy engine is ordinary code: refund amount ≤ \$200, order within 30 days, fewer than 3 refunds in 90 days, subscription not already cancelled, user not flagged for abuse. It returns `ALLOW`, `DENY(reason)`, or `NEEDS_APPROVAL`. This must be code and not a prompt, for three reasons: it's auditable, it's testable with unit tests, and it doesn't change behavior when someone tweaks a system prompt. When policy returns `DENY`, that reason is fed back to the model so it can explain to the user — the model handles the conversation, the engine handles the decision.

Execution goes through a durable workflow (Temporal or equivalent), not an inline HTTP call. Every action gets an idempotency key derived from `hash(conversation_id, action_type, args)`. This matters more than anything model-related: without it, a timeout plus a retry equals a double refund, and double refunds are how these projects get cancelled. The workflow handles retries with backoff, compensating actions on partial failure, and a terminal state written to an immutable audit log.

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

Escalation is a first-class path, not an error path. Triggers: policy `DENY` on something the user clearly wants, sentiment/frustration signals, three turns without progress, any mention of legal/chargeback/regulator, or model confidence below threshold. Escalation hands the human a *summary plus the full trace*, so the rep doesn't ask the user to repeat themselves — the single biggest driver of CSAT damage from bad bot handoffs.

Grounding for the conversational part: retrieval over the help center and policy docs, so answers cite real policy rather than the model's memory of it. Same RAG hygiene as problem 1, but a much smaller corpus.

Rollout is staged and I'd volunteer this unprompted: (1) shadow mode — agent proposes, humans act, we measure agreement; (2) copilot — agent drafts, rep clicks approve; (3) autonomous for the narrowest, most reversible action (resend receipt) only; (4) widen action-by-action as the wrong-action rate holds. Each stage gates on measured numbers, not on vibes.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI does three things: understand the user's intent and messy phrasing, extract structured arguments from conversation, and write the reply. That's genuinely hard and genuinely valuable.

I would **not** use a model to decide policy eligibility, to compute refund amounts (arithmetic on order lines — a function), to authenticate the user, to decide escalation thresholds (a rules layer over signals), or to write to any system directly. And no free-form SQL or shell tools — the tool surface is a fixed, small, typed set. Every one of these is a place where a model's 99% accuracy is a 1% incident rate on money.

**The hard tradeoff** — *autonomous action vs. human-in-the-loop approval.*

Autonomous gets you real deflection — a refund in 30 seconds at 2am — and the ROI that funds the project. But at 40k contacts/week and even a 0.5% wrong-action rate, that's 200 bad refunds a week, plus the trust damage. Human approval makes the error rate near zero but caps throughput at rep capacity, which means you've built an expensive autocomplete.

I'd go autonomous only for actions that are **cheap and reversible**, and gate on a measured per-action error rate from shadow mode. Refunds under \$50 with a clean account history: autonomous. Everything else: approval. What changes my mind is the shadow-mode data — if agreement with human reps is above ~98% on a class of action and the residual errors are recoverable, I widen. If errors cluster in a way I can't predict from features, I don't, no matter how good the average looks.

**How I'd evaluate it.**

Offline: a regression suite of a few hundred real conversations, replayed. Metrics: **action accuracy** (did it propose the same action a senior rep would), **argument accuracy** (right order ID, right amount), **policy adherence** (did it ever propose something the engine rejected — high rates mean a prompt problem), and **escalation precision/recall**. I'd also keep an adversarial set: users trying to talk the agent into refunds, prompt injection attempts, ambiguous multi-order cases.

Online: **deflection rate** (resolved without human), **wrong-action rate** (the guardrail — measured by sampling audited actions plus tracking reversals/chargebacks), **CSAT** split by bot-resolved vs. escalated, handle time for escalated conversations, and repeat-contact rate within 7 days (a resolution that generates a second contact isn't a resolution). Dollar metrics: refund dollars per contact, watched against a pre-launch baseline — a model that resolves everything by refunding everything looks great on CSAT and destroys margin.

**Failure modes I name before the interviewer does.**

- **Double execution** on retry or duplicate message — solved by idempotency keys, and this must be tested with deliberate fault injection.
- **Social engineering** — "I'm the account owner's spouse, just change the email." Fix: identity verification is a hard precondition on write tools, enforced outside the model.
- **Prompt injection via user-supplied content** — an order note containing instructions. Treat all retrieved and user text as data; never let it expand the tool set.
- **Policy drift between the engine and the prompt** — the prompt says 30 days, the engine says 14, users get told yes then denied. Fix: generate the policy summary in the prompt *from* the engine's config, single source of truth.
- **Refund-maximizing behavior** — the model learns (from being rewarded on CSAT) that yes is always easier. Watch refund dollars per contact as a guardrail metric.
- **Silent partial failure** — subscription cancelled but refund failed. Compensating transactions in the workflow, plus alerting on any workflow stuck in a non-terminal state.
- **Escalation black hole** — handoffs at 3am with no rep online. Needs explicit expectation-setting and a callback commitment.

**Follow-ups they will ask.**

*"How do you stop it from being talked into a refund it shouldn't give?"*
Two layers. The policy engine makes it structurally impossible to exceed limits regardless of what the model believes — persuasion can't change a rule evaluated in code with the account's real history. Second, I detect manipulation patterns as features (repeated reformulation after denial, escalating emotional language, claims contradicting account data) and route those to humans. Prompt-level defenses ("do not be persuaded") are the weakest layer and I wouldn't rely on them at all; I mention them last on purpose.

*"How do you handle a user with three orders who says 'refund the broken one'?"*
Disambiguation is a conversation design problem. The agent should present the candidates with distinguishing details (date, item, amount) and ask, rather than guessing. I'd hard-block the write path when argument confidence is low: the `propose_refund` tool requires an explicit `order_id`, and a resolver that returns multiple matches forces a clarifying turn. Guessing wrong here is the most common source of wrong-action incidents and it's cheap to prevent.

*"What's your latency budget?"*
For chat, first token within ~1.5s and full reply under 5s keeps the conversation feeling live (assumption — tune to your CSAT data). The tool loop is the risk: each read tool call adds a round trip, so I'd parallelize independent reads, prefetch the obvious context before the model's first turn, and cap the loop at ~5 iterations before forcing either a reply or an escalation. Actions execute asynchronously — the user gets "processing your refund now" immediately and a confirmation when the workflow lands, rather than staring at a spinner through a Stripe retry.

*"How do you version and roll out prompt changes?"*
Prompts are code: in the repo, code-reviewed, versioned, and pinned per deployment. Every change runs the offline regression suite, and the diff report is part of the PR. Rollout is canary — 5% of traffic, watch wrong-action and escalation rates for 24 hours, then widen. Every conversation logs the prompt version and model version so an incident can be traced to a specific change. Untracked prompt edits in a web console are the single most common cause of "it worked last week."

*"What if the LLM provider has an outage?"*
Tiered degradation. First, fail over to a secondary provider or a smaller self-hosted model, accepting reduced quality — the abstraction layer for this is worth building early. Second, if all models are down, fall back to a deterministic intent classifier plus templated flows for the top intents, which covers maybe 30% of volume. Third, queue and route to humans with an honest wait-time message. What I would *not* do is retry a hanging provider until the queue backs up; circuit-break at the client with a hard timeout.

*"How would you measure whether it's actually saving money?"*
Not by deflection rate alone — that's the metric that gets gamed. I'd run a holdout: a randomized 5% of contacts routed to humans only, permanently. Then compare fully loaded cost per resolved contact (model tokens + infra + the human time on escalations + the cost of wrong actions and reversals) against the control, along with 7-day repeat-contact rate and CSAT. A holdout costs a little revenue and is the only way to get a causal number; I'd fight to keep it.

*"Where does the training data for improvements come from?"*
The escalation queue is the gold mine — every escalation is a labeled example of "the agent couldn't do this," and the rep's resolution is the target. I'd instrument the rep tool to capture the action taken and a short reason code, then mine disagreements between agent proposals and rep actions in shadow/copilot mode. That dataset drives prompt improvements, few-shot example selection, and eventually fine-tuning of a smaller cheaper model on the high-volume intents.

---

## 3. LLM-powered code review assistant

### Q: "Design an LLM-powered code review assistant for our monorepo."

**Clarifying questions to ask first**

- **Is this replacing human review or augmenting it?** Augmenting means I optimize for precision — a few high-value comments. Replacing means I need blocking gates and a much higher bar, which I'd argue against for v1.
- **Monorepo size and PR volume?** 500 PRs/day at 2k lines each is a very different cost problem than 30 PRs/day, and it determines whether I can afford a large model on every diff.
- **What do reviewers actually complain about — style, bugs, missing tests, or architectural drift?** Style is a linter's job, not a model's. If the complaint is "we miss null-deref bugs," that's a static analysis + LLM hybrid. If it's "people don't follow our internal patterns," that's retrieval over the codebase.
- **Can the assistant see the whole repo, or just the diff?** Diff-only reviews produce confidently wrong comments about functions it can't see. Repo access means building a code index and a symbol resolver — most of the work.
- **What's the tolerance for false positives?** This is the make-or-break number. Developers abandon a bot after roughly two or three bad comments; I'd target ≥70% of comments being actioned.
- **Are there existing CI signals (tests, coverage, linters, SAST) I can condition on?** A failing test in the diff's blast radius is a far better trigger for a comment than a model's suspicion.

**Assume:** 4,000-engineer monorepo, ~800 PRs/day, median diff 180 lines, p95 2,000 lines. Go/TypeScript/Python. Existing CI with linters, type checkers, and a SAST tool. Goal: augment, never block. Target ≥70% of posted comments marked useful or acted on.

**The design.**

The framing that wins this question: **a code review assistant is a precision problem, not a capability problem.** The model can find plenty of issues; the product dies if 40% of its comments are noise. So the architecture is mostly about *suppression*.

Trigger: a webhook on PR open and on each push. First, a cheap deterministic gate before any model runs — skip generated files, lockfiles, vendored code, pure-formatting diffs, and PRs over a size threshold (those get "this PR is too large for useful automated review" instead of a hallucinated summary). This gate alone removes a large chunk of volume and cost.

Context assembly is the real engineering, and it's where I'd spend most of the design time. For each changed hunk I want: the full changed function (not just the diff lines), the definitions of symbols the hunk calls, the callers of the changed function, the file's tests, and any repo conventions relevant to the touched area. That requires a code index — I'd build one with tree-sitter parsing plus a language server or a `scip`/LSIF-style symbol index, incrementally updated on merge to main. Retrieval is symbol-graph traversal first and embeddings second: for code, "find the definition of `chargeCard`" is an exact lookup, and using a vector search for it is strictly worse. Embeddings are for the fuzzy query — "how do we usually handle idempotency here" — retrieved from a corpus of internal design docs, ADRs, and exemplar code.

Then review generation. I'd fan out per-hunk rather than sending the whole diff, because a single 2,000-line prompt produces vague comments while per-hunk prompts produce specific ones — and per-hunk parallelizes, keeping wall-clock under a couple of minutes. Each hunk prompt gets the assembled context and a rubric restricted to categories where models are actually good: logic errors and edge cases, missing error handling, concurrency and resource-leak issues, security-relevant patterns, missing test coverage for new branches, and violations of retrieved repo conventions. Explicitly excluded from the rubric: formatting, naming preferences, and anything a linter already checks.

The model must emit structured output: `{file, line, category, severity, claim, suggested_fix, confidence, evidence_symbols}`. Free-form prose comments are unfilterable.

Then the suppression pipeline, which is what makes this shippable:
1. **Deduplicate** against comments already posted on this PR and against unresolved comments on prior pushes.
2. **Cross-check with deterministic tools** — if the model claims a null deref and the type checker says the value is non-nullable, drop it. Static analysis is a much better arbiter than a second LLM call.
3. **Self-verification pass** — a second model call sees the claim plus the cited evidence and answers "is this definitely true given only this evidence?" Cheap, and it kills a large fraction of confident nonsense.
4. **Confidence + severity threshold**, tuned per repo from the feedback loop.
5. **Volume cap** — at most 5 comments per PR, ranked by severity. A bot that leaves 30 comments is ignored regardless of quality.

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

Every comment carries a reaction affordance, and resolved/unresolved plus "did the author change the code in response" get logged. That feedback store is the asset — it's how thresholds get tuned per team and how you build the eval set.

Cost control: at 800 PRs/day with, say, 6 hunks each, that's ~5k model calls/day. I'd route by risk — a cheap model does a first pass and only hunks it flags, or hunks touching sensitive paths (auth, payments, migrations), get escalated to the expensive model. Cache by `hash(hunk + context)` so re-pushes don't re-review unchanged hunks.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI reads code and reasons about intent versus implementation. That's the one thing that was previously impossible.

I would **not** use a model for formatting (gofmt/prettier), lint rules (existing linters, deterministically), type errors (the type checker), known CVEs in dependencies (a scanner), test execution, or coverage measurement. I'd also not have the model *block* a merge — a nondeterministic gate on the critical path of 4,000 engineers is an availability and morale disaster. And I would not have it auto-apply fixes without an author click; suggested diffs, yes, auto-commit, no.

**The hard tradeoff** — *review every PR shallowly vs. review a subset deeply.*

Every PR with a cheap model and tight context gives broad coverage and low cost, but comments are generic and precision suffers, which is exactly what kills adoption. Deep review — full symbol context, expensive model, self-verification — produces genuinely valuable catches but costs maybe 10x per PR and can't cover everything.

I'd go deep on a risk-scored subset: PRs touching payments/auth/data-migration paths, PRs from engineers with under 90 days' tenure, PRs with no tests added despite new branches, and PRs where CI is already unhappy. Everything else gets the cheap pass. What flips me is measured precision — if the cheap pass hits 60%+ actioned-comment rate, universal coverage is worth more than depth, because the value is in the volume of small catches.

**How I'd evaluate it.**

Offline: a benchmark built from your own history — take merged PRs that were later reverted or followed by a hotfix within 7 days, and check whether the assistant flags the offending hunk. That's a real "did it catch the bug" metric grounded in your codebase, and it's far more convincing than a public benchmark. Complement with a labeled set of ~200 hunks with known injected bugs for regression testing, plus a **false-positive set**: clean, idiomatic code where the correct output is zero comments. Precision on that set is the number I'd watch.

Online: **actioned-comment rate** (author edited the code near the comment, or resolved it as useful) — target ≥70%; 👎 rate; comments per PR; time-to-first-human-review (does the bot's pass make humans faster or slower); and the counterfactual — escaped-defect rate measured as reverts/hotfixes per 100 PRs, compared against a holdout set of repos where the bot is off. Keep that holdout for at least a quarter.

**Failure modes I name before the interviewer does.**

- **Confidently wrong comments about unseen code** — "this can be null" when the caller guarantees otherwise. Mitigation: symbol context and dropping any claim whose evidence symbols weren't in the prompt.
- **Nitpick flood** — the model comments on style because it always has something to say. Mitigation: rubric restriction and the volume cap.
- **Duplicate comments across pushes** — infuriating. Requires stable comment identity across rebases, which is genuinely fiddly (anchor on hunk content hash, not line number).
- **Leaking secrets or proprietary code to a third-party API** — needs a self-hosted model or a vendor with contractual guarantees, plus secret scanning before the prompt is built.
- **Gaming** — engineers split PRs to slip under the size gate, or ignore the bot entirely. Watch adoption per team; a team with 0% action rate is a signal to investigate, not to nag.
- **Index staleness** — the symbol index lags main, so context is wrong after a big refactor. Monitor index lag and degrade to diff-only review with a visible caveat when stale.
- **Cost blowup on a mega-refactor PR** — a 50k-line rename generating thousands of calls. Hard per-PR spend cap.

**Follow-ups they will ask.**

*"How do you handle a 3,000-line PR?"*
I don't try to review it hunk-by-hunk exhaustively — that's expensive and produces noise. I'd triage: classify hunks as mechanical (renames, generated, import churn) versus substantive using cheap heuristics plus a small model, review only the substantive ones, and post a top-level comment saying which files got real review and which were skipped. Being honest about coverage preserves trust. For genuinely large refactors the more useful output is a structural summary — "this changes the retry semantics in 12 call sites, 3 of which lack tests" — rather than line comments.

*"How do you teach it our internal conventions?"*
Retrieval, not fine-tuning, at least initially. I'd index ADRs, style guides, and — most valuably — past human review comments, which are the actual encoded conventions. When reviewing a hunk, retrieve the most similar past review comments on similar code and include them as few-shot examples. This adapts per-team automatically and updates the moment a new convention doc lands. Fine-tuning becomes worthwhile only when you have tens of thousands of accepted comments and want to shrink the model for cost.

*"Won't developers just ignore it?"*
Yes, if precision is bad — that's why precision is the primary metric rather than recall. Beyond quality, adoption levers: post comments before human reviewers are assigned so the author fixes issues privately (much less socially costly than a public nit), let teams opt into categories, and never block merges. I'd also publish per-team dashboards of the bot's action rate so teams can see whether it's helping them, and let a team turn off any category that isn't earning its keep.

*"How does this interact with security review?"*
It complements SAST but doesn't replace it — a model shouldn't be your control for OWASP categories that a deterministic tool covers with better recall. Where the model adds value is context-dependent security: "this endpoint reads `org_id` from the request body rather than the session, so it's an IDOR," which requires understanding the auth pattern used elsewhere in the repo. I'd route sensitive-path diffs to a dedicated security rubric with a lower confidence threshold (higher recall, accepting more false positives) and route those comments to the security team's queue rather than to the author, since the false-positive tolerance is different.

*"What about test generation?"*
Adjacent and higher value than review comments, in my experience. Given a hunk with new branches and no test coverage delta, generate a test, run it in CI sandbox, and only post it if it passes and actually increases coverage. That execution loop is the whole trick — an unexecuted generated test is a liability. It also gives you a clean, objective reward signal, which makes it a much better candidate for automated improvement than review comments, where "correct" is subjective.

*"How do you keep the code index fresh in a monorepo with 500 merges a day?"*
Incremental indexing on merge, keyed by file, with a symbol graph updated only for affected files and their reverse dependencies. Full rebuilds nightly as a correctness backstop. I'd track index lag as a p95 metric and alert above a few minutes. For a monorepo this size the index is a real distributed system — sharded by directory, with a serving layer that answers "definition of symbol X" in single-digit milliseconds. This is again the 80%: the reviewer's quality is bounded by index quality, and index quality is plain infrastructure work.

---

## 4. Semantic search / recommendations for a marketplace

### Q: "Design semantic search and recommendations for a marketplace — think eBay or Etsy scale."

**Clarifying questions to ask first**

- **Is inventory unique (one-of-a-kind listings) or replenishable (many identical SKUs)?** Unique inventory means brutal cold-start on every item and a churning index; replenishable means stable item embeddings and rich per-item interaction history.
- **What are we optimizing — GMV, conversion rate, or buyer retention?** Optimizing clicks gets you clickbait listings; optimizing GMV biases toward expensive items; the honest objective is usually a weighted blend with a return-rate penalty.
- **Two-sided constraints — do we need seller fairness or new-seller exposure guarantees?** If yes, that's an explicit constraint in the ranker or a re-ranking step, not something you hope emerges.
- **How much of search traffic is navigational ("nike air max 270 size 10") vs. exploratory ("gift for someone who likes camping")?** Navigational is a lexical/faceted problem where semantic search can actively hurt; exploratory is where embeddings earn their keep.
- **Latency budget and QPS?** Search at p95 200ms with 10k QPS constrains the whole architecture — it means a two-stage retrieve-and-rank with a strict per-stage budget.
- **Is there a paid-placement/ads business?** Ads change the ranker into a joint auction-and-relevance problem, which is a materially different design.

**Assume:** 200M active listings, 60% unique/one-of-a-kind. 8k search QPS peak, 40M DAU sessions/week. Objective: GMV per session, with a relevance guardrail. New-seller exposure is a stated business goal. p95 budget 250ms. Ads exist but are a separate slot allocation.

**The design.**

Standard, and standard is correct here: **candidate generation → filtering → ranking → policy re-ranking.** The mistake is trying to do it in one stage with one clever model.

*Indexing.* Every listing produces a document with lexical fields (title, description, structured attributes, category path) and a multimodal embedding. For a marketplace, image matters enormously — half of the "does this look like what I want" signal is visual — so I'd embed title+attributes+primary image into a shared space with a CLIP-style dual encoder, ideally fine-tuned on your own click/purchase pairs. That fine-tuning is the highest-ROI model work in this whole system: an off-the-shelf embedder doesn't know that in your marketplace "vintage" means pre-1990 and "boho" is a real category.

Ingestion has to handle 200M listings with high churn: new listings must be searchable within a minute (sellers watch this obsessively), sold-out listings must vanish immediately. So the index is a streaming write path with a fast delete lane, plus nightly reconciliation against the source of truth to catch drift.

*Candidate generation* runs several retrievers in parallel, each returning ~500:
- **BM25/lexical** over title and attributes — non-negotiable for navigational queries, brand names, and model numbers.
- **Dense ANN** over the multimodal embedding — HNSW for the hot set. This is what handles "gift for someone who likes camping."
- **Collaborative/behavioral** — item-to-item co-purchase and co-view neighbors, plus a two-tower user-embedding retriever for recommendations. This surfaces things text and images can't.
- **Business retrievers** — trending in category, new-seller exposure pool.

Results are fused (RRF or a learned blend) into ~1,000 candidates. On real hybrid comparisons, fusing lexical and dense beats either alone by a few points of nDCG — one published e-commerce comparison shows RRF at 0.7068 nDCG vs 0.6983 for BM25 and 0.6953 for dense alone, rising to 0.7497 with field boosting. The lesson isn't the exact number, it's that domain-specific field boosting beat the fancy part.

*Filtering* is hard business logic and happens after retrieval: shipping availability to the buyer's country, price and size facets, prohibited items, blocked sellers, out-of-stock. This is deterministic, and getting it wrong is worse than any ranking error — showing an item that can't ship is a guaranteed bad session.

*Ranking* is a gradient-boosted tree or a small DNN over ~200 features, scoring the ~1,000 candidates in under ~40ms. Features: query-item relevance scores from each retriever, item quality (seller rating, photo quality, return rate, dispute rate), price relative to category median, recency, personalization (user's category affinities, price band, past sellers), and context (device, session position). I'd use a multi-objective model predicting $P(\text{click})$, $P(\text{purchase} \mid \text{click})$, and expected value, combined as roughly

$$\text{score} = P(\text{click}) \cdot P(\text{purchase} \mid \text{click}) \cdot f(\text{price}) - \lambda \cdot P(\text{return})$$

with $\lambda$ tuned so we don't optimize into a returns problem.

*Policy re-ranking* is the last stage and it's pure code: diversity (no more than 3 listings per seller in the top 20), new-seller exposure quota, ads slot interleaving, and dedup of near-identical listings. Business constraints belong here, explicitly and tunably, not buried in a loss function.

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

Logging is the product's future: every impression with position, every click, every purchase, with the exact feature vector used at scoring time. Without logged features you cannot train an unbiased next-generation ranker, and reconstructing them later is impossible.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is in the embedding model, the behavioral retrievers, and the ranker. Notably, **there is no LLM in the request path.** An LLM at 8k QPS and 250ms p95 is neither affordable nor fast enough, and it doesn't beat a well-tuned GBDT on ranking.

I'd use LLMs offline, where they're excellent: generating structured attributes from messy seller descriptions (this materially improves faceting and lexical matching), producing synthetic query-item training pairs for the embedder, normalizing category taxonomies, and writing listing quality assessments. I would **not** use a model for stock, shipping eligibility, prohibited-item enforcement (a model can assist but the block must be rule-plus-review), or price. And I would not replace BM25 with pure vector search — that's the classic marketplace regression where someone searches an exact part number and gets vibes.

**The hard tradeoff** — *personalization depth vs. result consistency and cold start.*

Heavy personalization lifts engagement for known users but wrecks the experience for the ~40% of sessions that are logged-out or new, makes results unreproducible (support can't debug "I saw it yesterday"), and creates filter bubbles that suppress inventory discovery — which in a two-sided marketplace hurts seller supply.

I'd personalize the *ranker* moderately (session context and coarse affinities) but keep candidate generation mostly query-driven, so the same query returns a broadly similar set for everyone with reordering on top. What changes my mind: if A/B shows deep personalization lifting GMV per session by more than a couple of percent *without* degrading new-seller exposure or long-tail impression share, I'd push further. I'd want that seller-side metric in the readout, not just the buyer-side one.

**How I'd evaluate it.**

Offline: from logged interactions, **recall@1000** for candidate generation (the ceiling) and **nDCG@10 / MRR** for the ranker, using purchases as the strongest relevance label and clicks as weak labels. Correct for position bias with inverse propensity weighting — naive offline evaluation on logged clicks reliably overstates gains because the old ranker chose what got seen. Plus a human-rated relevance set of a few thousand query-item pairs as a guardrail, since a ranker can improve GMV while getting less relevant.

Online: interleaving experiments for fast, sensitive relevance comparisons (they need far less traffic than A/B), then A/B on the real objective: **GMV per session**, conversion rate, and search-abandonment rate. Guardrails: relevance rating, return rate, new-seller impression share, seller Gini coefficient, and p95 latency. Run for at least two weeks to catch novelty effects.

**Failure modes I name before the interviewer does.**

- **Popularity feedback loop** — the ranker promotes what's popular, which makes it more popular, and the long tail dies. Mitigation: exploration slots, propensity-weighted training, and monitoring impression-share distribution.
- **Cold-start listings** — a unique item with no interaction history has no behavioral signal. Mitigation: content-based embeddings carry it initially, plus an explicit exploration budget for new listings.
- **Query drift on seasonal terms** — "boots" means something different in December. Embeddings are static; the fix is recency features and periodic retraining, not a better encoder.
- **Stale index showing sold items** — the worst buyer experience. Fast delete lane plus a final availability check before render.
- **Ads cannibalizing relevance** — organic quality quietly degrades as ad load rises. Needs a fixed relevance floor for ad slots and a monitored organic-relevance metric.
- **Training-serving skew in features** — the offline pipeline computes "seller rating" over a different window than the online one. This is the single most common source of "great offline, flat online." Solved by a shared feature definition (see problem 12).
- **Multi-lingual and misspelled queries** — a huge share of real marketplace traffic. Needs spell correction and query translation before retrieval; embeddings alone don't fix "addidas."

**Follow-ups they will ask.**

*"How do you handle 'gift for my dad who likes fishing'?"*
That's an exploratory query where lexical retrieval fails completely and dense retrieval does most of the work. I'd add an offline-built query understanding layer: for the head and torso of exploratory queries (which are surprisingly repetitive), precompute an expansion — LLM-generated category and attribute hints, cached — so at serve time it's a lookup, not a generation. For the true tail, dense retrieval alone. I'd also detect exploratory intent with a cheap classifier and shift the fusion weights toward the dense retriever and toward diversity, because for a gift query the user wants breadth, not the 20 closest matches.

*"How do you evaluate the embedding model in isolation?"*
Build query-item pairs from purchase logs — a query that led to a purchase is a positive, sampled non-purchased impressions are hard negatives. Then measure recall@k of the embedder alone on held-out purchases from a later time window (temporal split, never random, or you leak). I'd also keep a small human-labeled set for semantic categories the logs won't cover. And I'd always compare against BM25 as a baseline — an embedder that doesn't beat BM25 on your domain isn't worth its serving cost, and that's a more common outcome than people admit.

*"What's your reindexing story when you change the embedding model?"*
Dual index. Build the new index in a shadow cluster while the old serves, backfilling 200M embeddings at whatever throughput the budget allows — probably a couple of days. Then shadow-evaluate: replay live traffic against both, compare recall and ranker features offline. Cut over per-region behind a flag, with instant rollback. Crucially, the ranker consumes retriever scores as features, so a new embedder shifts the feature distribution — the ranker needs retraining on data from the new retriever, or you'll see an unexplained ranking regression that has nothing to do with the ranker itself.

*"How do you get recommendations for a brand-new user?"*
Context is all you have and it's more than people think: entry point (which category page or referral), geography, device, time of day, and the first click in the session. I'd serve a popularity-by-context baseline, personalized within the session by the first one or two interactions using a lightweight session-based model (a GRU or transformer over the click sequence). Session-based recommenders are the right tool here precisely because they need no user history. I'd also not be too clever — for a logged-out first-time visitor, "best-selling in this category, high seller rating" is a strong baseline that many personalized models fail to beat.

*"How do you prevent the ranker from just learning 'cheap items get clicked'?"*
That's the multi-objective problem. A click-only objective absolutely finds price as the dominant feature. I'd train against purchase and post-purchase outcomes (delivered, not returned, not disputed), weight by margin or GMV depending on the business objective, and add the return-rate penalty. Then I'd monitor the price distribution of served results against the inventory distribution — if the served median drifts well below the inventory median, the ranker has found the cheap-item shortcut. I'd also examine feature importances and, if needed, apply monotonicity constraints in the GBDT so price can't dominate arbitrarily.

*"Do you need a GPU to serve this?"*
For the ranker, no — a GBDT over 1,000 candidates runs on CPU in tens of milliseconds and is cheaper and more debuggable. GPUs are needed for the embedding model, but only for indexing (batch, offline) and for encoding the query at serve time. Query encoding is a single short forward pass; I'd either run it on a small GPU pool with batching or use a distilled query encoder on CPU, and cache embeddings for the head queries — which in marketplace traffic covers a large share of volume. Reserving GPUs for the ranker would be spending the expensive resource on the part that needs it least.

---

## 5. Evaluation system for an LLM product

### Q: "Design an evaluation system for an LLM product. How do you know when it got worse?"

**Clarifying questions to ask first**

- **What's the unit of success — a single response, or a whole task/session?** Per-response scoring misses agents that take ten correct steps and then fail; session-level evaluation needs trajectory scoring and is much harder to build.
- **Is there a ground-truth signal in the product (a resolved ticket, an accepted suggestion, a completed purchase), or is quality purely subjective?** A real outcome signal changes everything — it lets you evaluate continuously on production traffic instead of on a frozen test set.
- **What changes underneath us — our prompts, our retrieval corpus, or the vendor's model?** Vendor model updates are the sneaky one: nothing in your repo changed and behavior moved. That requires a continuous canary, not just pre-deploy gates.
- **What's the cost tolerance per eval run?** If a full run costs \$500 and 3 hours, it can't gate every PR; I'd need a fast tier and a slow tier.
- **Who adjudicates disagreement — is there a domain expert available, and how many hours per week?** Human labeling capacity is the real constraint on eval quality, and I'd rather design around 4 hours/week of an expert than pretend I have unlimited labeling.
- **Do we need per-segment guarantees (by language, customer tier, topic)?** Aggregate metrics hide segment regressions, and a big customer's segment breaking is what actually causes escalations.

**Assume:** an LLM support/answering product, ~50k requests/day. Weak outcome signal available (ticket resolved without escalation, thumbs). We control prompts and retrieval; the model is a vendor API that updates. One domain expert, ~5 hours/week. Need per-language and per-top-customer segment views.

**The design.**

Four layers, cheapest and fastest first. The organizing idea: **most regressions should be caught by assertions, not by judges.** Judges are for the fuzzy residue.

**Layer 1 — Assertions (deterministic, milliseconds, free).** For every response, mechanical checks: valid JSON when structured output is expected, every citation ID exists in the retrieved context, no PII in the output, required disclaimers present when the topic triggers them, length within bounds, no refusal on benign input, no leaked prompt text. These run both in CI and *on every production response* as an online guardrail. In my experience this catches a large majority of real regressions, because most regressions are structural — a prompt edit breaks the output format, a retrieval change drops citations. Assertions are also the only layer you can afford to run on 100% of traffic.

**Layer 2 — Golden set with reference answers (minutes, cheap).** 200–500 curated examples with expert-written reference answers, stratified across intents, languages, difficulty, and known past failures. Every production incident becomes a permanent golden case — that's the ratchet that stops regressions from recurring. Scoring uses exact/fuzzy match where possible and an LLM judge with a reference where not. This tier gates every PR.

**Layer 3 — LLM-as-judge on production samples (hourly, moderate cost).** Sample a few hundred production interactions per hour, stratified (not uniformly — uniform sampling drowns you in easy cases), and score with a rubric judge: groundedness, helpfulness, tone, task completion. The judge must be **calibrated**: I hold out ~200 human-labeled examples and measure the judge's agreement with the expert (Cohen's kappa). If kappa is below about 0.6 the judge isn't trustworthy and I fix the rubric before I trust its trend. I'd also use pairwise comparison rather than absolute scoring where possible — judges are much more reliable at "is A better than B" than at "rate this 1–5," and absolute scores drift over time and across judge model versions.

Critically, **the judge model and its prompt are pinned and versioned**. If the judge changes, all historical scores are incomparable. When I upgrade the judge, I re-score a historical window with both to establish a conversion.

**Layer 4 — Human review (weekly, expensive).** The expert's 5 hours go to the highest-information cases: judge–assertion disagreements, low-confidence outputs, user-reported failures, and a small random sample for unbiased calibration. Their labels feed back into judge calibration and the golden set.

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

**Detecting "it got worse" is a change-detection problem, not a threshold problem.** Each metric is a time series segmented by version, language, intent, and customer tier. I'd apply CUSUM or a simple sequential test per segment, with the comparison being *this version vs. the previous version on the same traffic mix* — because traffic mix shifts alone will move aggregate metrics. Alerts fire on statistically significant degradation, not on crossing an arbitrary line.

Two things I'd build that people forget. First, a **frozen canary set replayed against the vendor's model daily** — same 100 prompts, same temperature-0 settings, diffed against yesterday's outputs. When the vendor silently updates, this is the only thing that tells you. Second, **shadow evaluation**: new prompt versions run against a mirror of live traffic without serving, so you get a full distribution comparison before a single user is exposed.

Release process: PR gates on L1+L2. Merge deploys behind a flag. Canary at 5% for 24 hours with automated comparison of L1 rates, L3 scores, and product metrics. Auto-rollback on assertion-rate regression; human decision on judge-score regression, since judge noise is real.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is the judge in layer 3, and nothing else. Layers 1, 2's matching logic, sampling, statistics, alerting, and rollback are ordinary software.

I would **not** use an LLM to decide whether to roll back (a statistical test on metrics), to generate the golden set unsupervised (LLM-generated test cases are systematically easier than reality and give false confidence — I'd use them only as drafts for expert editing), to score anything an assertion can check deterministically, or to summarize the eval results into a "quality score" single number that hides segment regressions. And I'd never let the same model family judge its own output as the sole signal without human calibration.

**The hard tradeoff** — *fast, cheap gates on every PR vs. thorough evaluation that's too slow to gate.*

A 5-minute gate keeps velocity but only catches structural breaks; it will pass a prompt change that makes answers subtly less helpful. A 3-hour, \$500 full evaluation catches quality drift but can't run per-PR, so regressions land and are found later.

I'd do both, deliberately: cheap gates block merges, and the thorough run happens nightly plus on canary. The regression window is then up to 24 hours — acceptable if the canary is only 5% of traffic and rollback is one click. What changes my mind is severity: for a product where a bad answer is a compliance event, I'd move the thorough evaluation pre-merge and eat the velocity cost, probably by shrinking the thorough set with stratified sampling until it fits in 20 minutes.

**How I'd evaluate it.** (Yes — the eval system needs evaluation.)

The meta-metrics: **judge–human agreement** (kappa, tracked over time, per segment); **regression detection recall** — inject known-bad prompt variants and measure what fraction the system catches before canary ends; **false alarm rate** — how often does it flag a regression that a human then confirms is noise (above ~20% and engineers start ignoring alerts); **time-to-detection** for the vendor-update canary; and **golden set coverage** — what share of production incidents in the last quarter were represented by an existing golden case before they happened. That last number is the honest measure of whether your eval set reflects reality.

**Failure modes I name before the interviewer does.**

- **Golden set overfitting** — everyone tunes prompts against the same 300 cases until scores are meaningless. Mitigation: a held-out set nobody sees, rotated quarterly, plus continuous addition from production.
- **Judge drift** — the vendor updates the judge model and every historical score shifts. Pin versions; re-score overlap windows.
- **Aggregate metrics hiding segment collapse** — overall score flat while Japanese-language quality falls off a cliff. Mandatory per-segment breakdowns with minimum sample sizes.
- **Survivorship bias in feedback** — thumbs-down comes disproportionately from users who cared enough to click. Never treat thumbs as the primary metric; use the stratified sample for unbiased estimates.
- **Eval cost exceeding inference cost** — easy to hit if you judge everything with a large model. Sample, and use a smaller judge for high-volume tiers.
- **Slow-boiling drift** — quality declines 0.5% a week and no single comparison is significant. Mitigation: compare against a fixed baseline version from N months ago, not only against the previous release.
- **Nobody looks at it** — the most common failure. The eval dashboard needs to be in the release process and on-call runbook, or it rots.

**Follow-ups they will ask.**

*"How big should the golden set be?"*
Big enough that a change you care about is detectable. If the metric is a proportion around 0.8 and you want to detect a 5-point drop with reasonable power, you need on the order of a few hundred examples per segment — the binomial math sets the floor. So 300 total is fine for one aggregate number and completely inadequate for ten segments. I'd rather have 200 well-curated, hard, expert-labeled cases per key segment than 5,000 auto-generated easy ones. Composition matters more than size: stratify by intent and difficulty, and deliberately over-sample the hard tail.

*"How do you handle the case where there's no single right answer?"*
Pairwise preference instead of absolute scoring. Generate responses from the candidate and the incumbent, present them blind to a judge (or human) in randomized order, and compute a win rate. This sidesteps rubric calibration entirely and is far more stable across judge versions. I'd also decompose subjective quality into checkable sub-properties — factually grounded, addresses the question asked, right length, right tone — because those are individually much less ambiguous than "good."

*"The vendor pushed a model update and things feel off, but metrics look flat. What now?"*
Trust the report, then find the segment. Flat aggregates plus real complaints almost always means a segment or behavior-class regression. I'd diff the canary set outputs token-by-token to characterize *how* behavior changed (longer? more hedging? different refusal boundary?), then slice production metrics by every dimension I have. I'd also check metrics the eval doesn't cover — response length, latency, refusal rate, tool-call frequency — since a behavior shift often shows in those before it shows in quality scores. If the complaint is real and unmeasured, that becomes a new assertion and new golden cases.

*"How do you evaluate multi-turn conversations?"*
Session-level, with trajectory metrics: task completion (did the user's goal get met, judged from the full transcript plus the outcome signal), turns-to-resolution, and recovery rate after an error. I'd also build the eval as a replay harness with a simulated user — an LLM playing the user role from a persona and goal — which lets me test multi-turn paths deterministically. Simulated users are noticeably more cooperative than real ones, so I'd calibrate against real transcripts and treat simulated results as a relative signal between versions, not an absolute quality estimate.

*"How much should you spend on evals?"*
As a rule of thumb I'd budget 10–20% of inference spend on evaluation (assumption — a planning heuristic, not a measured law), rising during major changes. The framing I'd use with a skeptical exec: the cost of one undetected regression running for a week across 350k requests dwarfs the eval bill. But I'd also keep eval spend visible and tiered, so the expensive layers can be dialed down in steady state and up during a migration.

*"What do you do on day one when you have no data?"*
Write 50 test cases by hand from the product spec, before writing the prompt. That forces specification of what "correct" means and instantly catches the worst failures. Add assertions immediately — they need no labels. Then instrument production heavily from launch, review every failure manually for the first weeks, and let the golden set grow from real traffic. The trap to avoid is generating a synthetic eval set with an LLM and declaring victory at 95%; that number is meaningless and buys false confidence at exactly the moment you most need honest signal.

---

## 6. Real-time content moderation

### Q: "Design a real-time content moderation system for a large user-generated content platform."

**Clarifying questions to ask first**

- **Which modalities — text, images, video, audio, or all?** Video is a fundamentally different cost and latency problem (sampling frames, transcoding) and dominates the budget if present.
- **Is moderation pre-publish (blocking) or post-publish (takedown)?** Pre-publish puts the model on the critical path with a hard latency budget; post-publish lets me use slow, expensive, accurate models but accepts exposure time.
- **What's the harm taxonomy and its legal structure?** CSAM and terrorist content have mandatory reporting and zero-tolerance handling; spam is a nuisance; harassment is contextual. These need entirely different pipelines, not one classifier with many labels.
- **What are the regulatory obligations — DSA, appeals, transparency reporting?** If yes, I need per-decision audit records, an appeals workflow, and statistics reporting as core features, not add-ons.
- **What's the relative cost of a false positive vs. a false negative, per category?** Wrongly removing a benign post is a user-trust cost; missing CSAM is unbounded. Thresholds must be per-category and derived from this.
- **Volume and peak burstiness?** Moderation load spikes hard around events, and a coordinated brigading attack is a 100x spike on one topic.

**Assume:** text and images, 50M posts/day (~600/s average, 3k/s peak). Post-publish for most categories with a pre-publish block for the highest-severity ones. Categories: CSAM/terror (zero tolerance), violence/gore, hate, harassment, adult, spam/scams, self-harm. DSA-style appeal obligations. Human review capacity ~2,000 items/hour.

**The design.**

A **cascade**, ordered by cost, with human review as the scarce resource everything else is designed to protect.

**Stage 0 — Hash and rule matching (sub-millisecond).** Every image is hashed with PhotoDNA/PDQ-style perceptual hashing and checked against known-bad databases (NCMEC, GIFCT). Text runs through exact and fuzzy matching against known scam templates and banned strings. This is not machine learning and it catches a large share of the worst content instantly and deterministically. Any candidate who jumps straight to an LLM classifier has missed the most important stage in the system. Hash matches on the zero-tolerance categories go straight to block-plus-report, no model involved.

**Stage 1 — Cheap classifiers (~5–20ms, on everything).** Small fine-tuned transformer classifiers for text (a distilled BERT-class model) and an image classifier, both multi-label over the taxonomy, running on every item. They output calibrated probabilities per category. Most content is clearly benign and exits here. I'd also compute non-content signals in parallel: account age, prior violations, posting velocity, device/IP reputation, network features (who shares this content). These behavioral features are often more predictive of spam and coordinated abuse than the content itself — a brand-new account posting 200 identical links is spam regardless of what the text says.

**Stage 2 — Expensive model on the uncertain band (~200ms–2s, ~2–5% of items).** Items where the cheap classifier is in the ambiguous range, or where behavioral risk is elevated, go to a large multimodal model with the full context: the post, the image, the parent thread, the community's norms, and the relevant policy text. This is where an LLM genuinely earns its cost, because the hard cases are contextual — reclaimed slurs, satire, news reporting of violence, medical discussion of self-harm. A classifier can't read context; a model with the policy in its prompt can, and it can produce a rationale citing the specific policy clause, which is exactly what a human reviewer and an appeals process need.

**Stage 3 — Human review (the scarce resource).** Queued by expected harm × uncertainty × reach. Reach matters enormously: a borderline post with 2 views and one with 200k views are not the same priority. Reviewers see the item, the model's rationale, the policy clause, and account history. Their decisions are the training data for everything upstream.

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

The **enforcement service** is deliberately separate from detection. Detection says "this is 0.93 hate speech"; enforcement decides the action — remove, demote, age-gate, label, warn, suspend — based on category, confidence, account history, and jurisdiction. Same content can warrant different actions for a first-time user versus a repeat offender, and geography matters (content legal in one country isn't in another). This is a rules engine, versioned, with every decision logged immutably.

**Appeals are a first-class pipeline**, not an afterthought. An appeal re-runs the item through the expensive model with the user's stated reason, then routes to a human who did not make the original decision. Appeal outcomes are gold-standard labels: a high overturn rate on a category means your threshold is wrong, and that's the fastest quality signal in the system.

For the pre-publish block on zero-tolerance categories, the budget is tight — hash lookup plus cheap classifier, under 50ms, with a fail-open-to-review policy on timeout for most categories and fail-closed for the worst.

Throughput engineering matters as much as the models: Kafka ingest, autoscaled classifier workers with GPU batching, and backpressure that degrades gracefully — under a 3k/s spike, I'd keep stage 0 and 1 at full coverage and sample stage 2, rather than letting the queue grow unbounded.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is the cheap classifiers and the contextual multimodal model. That's real value — nothing else scales to 50M items/day.

I would **not** use a model for known-bad content (hashing is exact, instant, and legally defensible), not for the final enforcement decision on high-severity categories (human confirmation required — the legal and ethical cost of an automated error is too high), not for jurisdiction rules (a lookup table), and not for the appeals decision. I'd also resist a single LLM doing everything: at 50M items/day even at \$0.10 per thousand items that's meaningful money, and it would be strictly worse than a distilled classifier on the easy 95%.

**The hard tradeoff** — *aggressive automated enforcement vs. human-confirmed enforcement.*

Aggressive automation removes harmful content in seconds and scales without headcount, but every false positive is a user wrongly punished, and at 50M items/day a 0.1% false positive rate is 50,000 wrongful removals daily. Human confirmation makes errors rare but caps enforcement at ~2,000/hour and leaves harmful content live for hours.

I'd set thresholds per category from the actual harm asymmetry: auto-enforce aggressively on spam and hash-matched CSAM (cheap errors, or no errors), require human confirmation on harassment and hate (contextual, high false-positive cost, high user-trust stakes), and use reversible actions — demotion and age-gating rather than removal — in the uncertain middle. Reversible enforcement is the underrated move: it reduces harm exposure without the trust cost of a wrongful takedown. What changes my mind is the measured appeal overturn rate; if overturns on a category exceed a few percent, I raise the threshold or move that category to human-confirmed.

**How I'd evaluate it.**

Offline: per-category **precision/recall at the operating threshold**, on a stratified labeled set with an adversarial slice (deliberately evasive content — leetspeak, image text, cropped memes). Precision-recall curves per category, not one aggregate F1, because the operating points differ wildly. Also **calibration** — if the model says 0.9, is it right 90% of the time? Calibration matters more than raw accuracy here, since thresholds are the control surface.

The recall problem is the hard one: you can't measure what you didn't catch. Solution is a **random audit sample** — a few thousand items/day pulled uniformly from published content and reviewed by humans regardless of model score. That gives an unbiased prevalence estimate: what fraction of live content violates policy. Prevalence, not classifier recall, is the metric leadership should care about, and it's the one used in public transparency reports.

Online: prevalence by category, time-to-action (p50/p95 from post to enforcement), views-before-removal (the actual harm exposure metric), appeal rate and overturn rate, reviewer throughput and agreement rate, and user reports per 1k posts. Guardrails: false-positive rate estimated from appeals, and enforcement rate by demographic segment to detect bias.

**Failure modes I name before the interviewer does.**

- **Adversarial evasion** — users adapt within hours of a new rule (character substitution, text in images, coded language). Requires continuous retraining and a fast-response path for new patterns; a static classifier decays measurably within weeks.
- **Context collapse** — removing a news organization's report of violence or a survivor's account of harassment. The single biggest source of PR damage. Mitigated by context in stage 2 and by trusted-publisher allowlists.
- **Coordinated brigading spike** — 100x volume on one topic overwhelming both queues. Needs burst detection, correlated-content clustering (review one representative, apply to the cluster), and preemptive rate limits.
- **Reviewer trauma and turnover** — a real operational failure mode with quality consequences. Design for it: rotation, blurring, wellness limits on exposure to the worst queues.
- **Feedback loop bias** — the model is trained on reviewed items, which are the items the model flagged, so it never learns about what it systematically misses. Fixed only by the random audit sample entering the training set.
- **Language and cultural coverage gaps** — performance is far worse outside English, and this is where the worst real-world harms have historically occurred. Requires per-language metrics with minimum sample sizes and explicit investment, not a multilingual model and hope.
- **Over-blocking a legitimate community** — a term that's a slur in one context and an in-group identifier in another. Per-community threshold tuning and appeal monitoring by community.

**Follow-ups they will ask.**

*"How do you handle a brand-new harm type that appears overnight?"*
A fast path that doesn't require retraining: rules and hash-based blocking deployable within minutes by the policy team, plus a nearest-neighbor lane where a handful of confirmed examples are embedded and anything similar in embedding space is queued for review. That gets meaningful coverage within hours from ~50 examples. Meanwhile human reviewers label aggressively, and a classifier retrain lands in days. The key architectural property is that policy changes are configuration, not deploys — if adding a new rule requires a model retrain, you'll always be a week behind.

*"How do you decide the threshold for each category?"*
From the cost asymmetry, made explicit. I'd write down the cost of a false positive and false negative per category in comparable units (user-trust damage, harm exposure, legal risk), then pick the threshold minimizing expected cost on the calibrated probability. In practice that's a conversation with policy and legal, and my job is to give them the precision-recall curve and say "at this threshold you get 92% precision and 60% recall, at that one 75% and 85%; which errors do you want?" Then revisit quarterly using appeal overturn rates as ground truth.

*"What about video?"*
Sample frames (scene-change detection plus a fixed rate), run image classification on frames, transcribe audio and run text classification on the transcript, and check the video's perceptual hash against known-bad. The expensive part is transcode and frame extraction, not inference. For live video the design is different: sample more aggressively at the start (where most policy-violating streams declare themselves), escalate viewer-reported streams immediately, and weight by concurrent viewers since a stream with 50k viewers needs sub-minute action.

*"How do you keep the human reviewers' labels consistent?"*
Treat labeling as a measured process: a shared rubric with worked examples, regular calibration tests where every reviewer labels the same gold items, and inter-annotator agreement tracked per reviewer and per category. Disagreement above threshold means the policy is ambiguous, not that reviewers are bad — that's a signal to rewrite the policy. I'd also route a fraction of items to multiple reviewers for ongoing agreement measurement, and never train on single-reviewer labels for categories with low agreement.

*"What's the cost structure?"*
Dominated by stage 1 running on 100% of traffic and by human review, not by the fancy model. At 600/s average, cheap classifier inference on batched GPU is a modest fleet. Stage 2 at 3% of 50M/day is 1.5M expensive calls/day — that's the line item to watch, and it's controlled entirely by how wide the uncertainty band is. Human review at 2,000/hour with round-the-clock coverage is likely the largest single cost in the system. So the highest-leverage optimization isn't a better model, it's narrowing the uncertainty band and clustering duplicate items so reviewers see each distinct piece of content once.

*"How do you handle a regulator asking why a specific post was removed?"*
Immutable per-decision records: content hash, timestamp, model versions and scores, the policy clause cited, the enforcement rule applied, the reviewer ID if human, and the appeal history. Retention per jurisdiction. This is a straightforward but non-trivial data engineering requirement — 50M decisions/day of structured audit records with query-by-content-ID access, kept for years — and it's the kind of requirement that quietly dominates storage design. I'd build it from day one because retrofitting audit logging onto an existing pipeline is painful and you'll never fully recover the historical records.

---

## 7. Personalized news / feed ranking

### Q: "Design a personalized news feed ranking system."

**Clarifying questions to ask first**

- **Is the content pool from followed sources only, or open (anything on the platform)?** Follow-graph feeds have a small candidate set and the problem is ordering; open feeds need real retrieval over millions of items and cold-start dominates.
- **What's the objective — time spent, or something like "meaningful interactions" / long-term retention?** Time spent is easy to optimize and well documented to produce outcomes nobody wants. If the objective is retention, I need long-horizon evaluation and surrogate metrics.
- **Content lifetime?** News decays in hours; evergreen content in weeks. This sets the retraining cadence and how heavily recency features weigh.
- **Are there editorial or integrity constraints — misinformation demotion, source diversity, political balance?** These are hard constraints in a re-ranking layer, and they need to be designed in, not bolted on.
- **How often does a user visit, and how much do they consume per session?** Someone opening the app 20 times a day needs dedup and freshness guarantees across sessions; a daily visitor needs a digest.
- **Latency and scale?** Feed loads are the highest-QPS surface on most platforms, and the p95 budget shapes everything.

**Assume:** open feed, 50M DAU, ~10M new items/day, average 8 sessions/user/day, 25 items consumed per session. Objective is 28-day retention, with weekly active engagement as the surrogate. Integrity constraints: demote low-quality/misinfo sources, cap source concentration. p95 budget 300ms for a feed load.

**The design.**

Same skeleton as the marketplace — retrieve, rank, re-rank — but the interesting differences are **recency, repetition, and long-horizon objectives**.

*Candidate generation.* From a pool of ~10M fresh items I need ~1,000 candidates per request in tens of milliseconds. Sources in parallel:
- **Follow-graph / subscribed sources** — recent items from what the user follows.
- **Two-tower embedding retrieval** — a user tower over recent interaction history and a content tower over item text/media, ANN-searched. Item embeddings are computed once at publish; user embeddings are recomputed on interaction (or approximated by pooling recent item embeddings, which is much cheaper and nearly as good).
- **Collaborative signals** — items engaged with by users similar to this one, and items trending within the user's topic clusters.
- **Fresh/exploration pool** — items too new to have signal, deliberately injected.

Pre-computation is essential at 50M DAU: for most users I'd precompute a candidate pool asynchronously (say every 15 minutes, and on significant interaction), so the request path is "fetch precomputed pool, merge in the freshest items, rank." Trying to do full retrieval synchronously at feed-load QPS is the wrong cost curve.

*Filtering* is deterministic and heavy here: already-seen items (a per-user Bloom filter or seen-store — this is critical and unglamorous), blocked sources, muted topics, region-restricted content, and integrity blocklists.

*Ranking.* A multi-task neural ranker predicting several outcomes per item: $P(\text{click})$, $P(\text{dwell} > 30s)$, $P(\text{share})$, $P(\text{hide/report})$, $P(\text{follow source})$. Combined into a single score with tuned weights:

$$\text{score} = \sum_i w_i \cdot P(\text{action}_i) \; - \; \sum_j v_j \cdot P(\text{negative}_j)$$

The negative terms carry real weight — hides and reports are the strongest available proxy for "this feed is getting worse," and a ranker without them optimizes straight into rage-bait. Features: user-item affinity from the towers, user's topic/source affinities over multiple time windows (1 day, 7 day, 90 day), item quality and source reputation, item age with an explicit decay, engagement velocity (how fast this item is accruing engagement relative to its age — the key freshness signal), and session context (position in session, time of day, device).

Recency needs care: a raw "engagement count" feature guarantees a rich-get-richer loop and buries new items. I'd use velocity normalized by age and impressions, plus an explicit decay term $e^{-\lambda t}$ with $\lambda$ tuned per content type — hours for news, days for evergreen.

*Re-ranking* is where the constraints live, and it operates on the whole slate, not per item: source diversity (no more than 2 items from one source in the top 10), topic diversity, integrity demotion multipliers for low-quality sources, ad interleaving, and an exploration quota (say 10% of slots to items the model is uncertain about). Slate-level optimization matters — the marginal value of the fifth article on the same story is near zero even if each scores well individually. I'd also cluster near-duplicate stories (the same news event covered by 40 outlets) and show one representative with a "more coverage" affordance.

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

Logging: every impression with position and dwell, every interaction, and the feature vector used. Position-bias correction is mandatory — the top slot gets clicked regardless of quality, so training on raw clicks teaches the model to predict position.

Retraining cadence is fast: user interests and news cycles move daily, so I'd retrain the ranker daily and update user embeddings near-real-time. Item embeddings are static per item.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is the two-tower retrievers, the item content embeddings, the multi-task ranker, and offline content understanding (topic classification, quality scoring, near-duplicate clustering).

I would **not** put an LLM in the request path — at this QPS it's economically impossible and adds nothing a ranker doesn't do better. LLMs go offline: generating item topic labels and summaries, clustering stories, and assessing content quality signals. I would **not** use a model for the seen-store, source diversity caps, integrity blocklists, or region restrictions — all deterministic. And I'd resist letting the model implicitly set the objective: the weights $w_i$ are a product decision made by humans and written down, not something discovered by the optimizer.

**The hard tradeoff** — *optimizing engagement vs. optimizing long-term retention.*

Engagement metrics are measurable in hours and move reliably; they're also well documented to drift toward outrage, clickbait, and content users regret consuming. Retention is the objective you actually want but takes 28 days to measure, has terrible statistical power, and can't gate weekly experiments.

I'd use engagement as the training signal but constrain it heavily: strong negative weights on hide/report, a survey-based "was this worth your time" signal collected from a small panel and used to train a quality model that enters the ranker, and holdout experiments measuring long-horizon retention on a slower cadence. What changes my mind: if the long-horizon holdouts show engagement-optimized variants winning on retention too, I'd loosen the constraints. In practice they usually diverge, and when they do I'd trust the 28-day number and take the short-term hit — but that's a leadership decision I'd want made explicitly, not by a loss function.

**How I'd evaluate it.**

Offline: replay logged sessions with IPS or doubly-robust off-policy estimation to correct for the fact that the old ranker chose what was shown. Metrics: nDCG on engagement labels, plus AUC per task head. Offline gains here are notoriously weakly correlated with online gains, so I'd treat offline as a *filter* for obviously-bad candidates, not as a decision.

Online A/B, minimum two weeks to clear novelty effects. Primary: sessions per user per day and 7-day retention. Secondary: dwell time, meaningful interactions. Guardrails, which matter more than the primary here: hide/report rate, source diversity (Gini over sources shown), share of impressions to low-quality sources, share of feed from the top-10 sources, and the fraction of users whose topic diversity narrowed. Plus a long-horizon holdout — a persistent small cell on the old ranker — to measure cumulative effects that week-long A/Bs structurally cannot see.

**Failure modes I name before the interviewer does.**

- **Filter bubble narrowing** — the ranker exploits known interests and the user's topic diversity collapses over months. Invisible in a two-week A/B; visible in a long-horizon holdout. Mitigation: exploration quota and diversity constraints.
- **Clickbait and rage-bait ascendance** — direct consequence of a click-weighted objective. Mitigation: negative signals, dwell quality, quality model.
- **Stale feed for heavy users** — 8 sessions a day exhausts the candidate pool and the user sees repeats. Mitigation: robust seen-store, freshness quotas, and pool refresh triggered by consumption depth.
- **Cold-start for new users** — no history, and the two-tower user embedding is garbage. Mitigation: onboarding topic selection, geo/context popularity baseline, fast session-based adaptation.
- **Breaking news latency** — a major event happens and the feed is 20 minutes stale because candidate pools are precomputed. Needs an event-detection bypass that injects high-velocity items into feeds immediately.
- **Coordinated manipulation** — actors gaming engagement velocity to get amplified. Needs an integrity layer scoring the *audience* of engagement, not just its volume.
- **Position-bias feedback loop** — training on uncorrected clicks makes the model progressively more confident about whatever it already ranked highly.

**Follow-ups they will ask.**

*"How do you handle breaking news?"*
A separate real-time lane. An event detector watches for abnormal velocity in story clusters (posting rate, engagement rate, geographic spread) and, on firing, injects the representative item into the merge step of every relevant user's feed request, bypassing the precomputed pool. Relevance is by geography and topic affinity. I'd keep this deliberately conservative and rate-limited, because it's an obvious manipulation target — the entry criteria should include source credibility, and I'd want human editorial override available for the highest-reach injections.

*"How do you prevent the same story from 40 outlets filling the feed?"*
Story clustering, offline and continuous: embed items and cluster by content similarity with a time window, producing a story ID. Then re-ranking treats the story as the unit — pick the best single item per story for the user (best source for their preferences, or the highest quality) and suppress the rest, with a "more coverage" expansion. This is a good example of where an offline LLM/embedding pipeline creates a lot of value invisibly: the user just experiences a feed that isn't repetitive.

*"How do you serve 50M DAU with 300ms p95?"*
Precomputation and caching do the heavy lifting. Candidate pools are built asynchronously and cached; user features are precomputed in an online feature store with single-digit-millisecond reads; item features are cached in memory on the ranking service since 10M items/day is small enough to keep hot. The synchronous path is then: fetch pool (~10ms), fetch user features (~5ms), score 1,000 items with a small model (~30ms on CPU with batching), re-rank (~5ms), assemble. The ranker must be small — this is a case where a 10x bigger model that's 2% better is not worth 10x the fleet.

*"What if a user says 'my feed got worse'?"*
Individual complaints are usually a specific mechanism, so I'd instrument for it: per-user feed composition over time (topic mix, source mix, share of followed vs. recommended), and a diff view showing what changed. Common real causes are a single viral interaction pulling the user's embedding toward a topic they don't actually want, seen-store failure causing repeats, or a source they liked being demoted by integrity. I'd also give users direct controls — "show less of this," topic mutes, "why am I seeing this" — both because they're good product and because they generate high-quality negative labels that are otherwise very hard to collect.

*"How do you weight the multi-task heads?"*
Not by intuition and not by the optimizer. I'd calibrate the weights against the long-horizon objective: run experiments with different weight vectors and measure 28-day retention, then fit the relationship. In practice you get a handful of experiments' worth of signal, so it's coarse. A useful discipline is to express weights in interpretable units — "one share is worth as much as eight clicks" — so product leadership can argue about them directly. Weights should be reviewed on a schedule, since the right tradeoff shifts as the platform and content mix change.

*"How do you deal with the exploration cost?"*
Exploration slots have measurably worse immediate engagement, so there's real short-term cost, and there's constant pressure to cut them. I'd defend the budget by measuring what exploration buys: the rate at which explored items become long-term engaged interests, and the effect on catalog coverage. I'd also make exploration smarter than random — Thompson sampling or upper-confidence-bound over the ranker's uncertainty concentrates the budget where the information value is highest, which typically cuts the cost of exploration substantially versus uniform random injection.

---

## 8. Internal LLM serving platform

### Q: "Design an LLM serving platform for internal teams — multiple models, multiple teams, and we need cost control."

**Clarifying questions to ask first**

- **Self-hosted open models, vendor APIs, or both?** Both means the platform's main job is abstraction and routing. Self-hosted only means it's a GPU capacity and scheduling problem, which is a very different system.
- **What's the workload mix — interactive chat, batch processing, or agentic loops?** Batch can be scheduled off-peak on cheap capacity; interactive needs headroom and low latency; agentic loops are bursty and can blow budgets fast because one user request becomes fifty model calls.
- **Is cost control advisory (show teams their spend) or enforcing (hard quotas that cut them off)?** Hard quotas need a real-time accounting path and a well-designed failure mode, since cutting off a production service to save money is usually the wrong call.
- **What are the data-governance requirements — can any team send any data to a vendor?** If some data can't leave, I need routing policies enforced at the gateway with data classification, not team discretion.
- **How many teams, and how sophisticated are they?** Twenty ML teams want raw access and control; two hundred product teams want a simple endpoint and sane defaults.
- **Is there an SLA, and who's on call?** A platform without a defined SLA becomes everyone's scapegoat during incidents.

**Assume:** 60 internal teams, both vendor APIs and 4 self-hosted open models on a shared 64-GPU cluster. Mix of interactive and batch. Cost control is enforcing with per-team quotas and a break-glass override. Some data classifications cannot leave the network. ~5M requests/day.

**The design.**

The platform is a **gateway plus a scheduler plus an accounting system**. The models are the easy part.

*Gateway.* One OpenAI-compatible API surface for everything — vendor models and self-hosted alike. Compatibility matters practically: teams can use existing SDKs, and switching a model is a config change. The gateway handles authentication (per-team service tokens), request validation, data-classification tagging, routing, rate limiting, retries, and logging.

Routing is policy-driven, evaluated per request against `(team, model_alias, data_class, priority)`. Teams request **aliases**, not model versions — `fast-chat`, `deep-reason`, `cheap-classify` — which is the single most important design decision in the whole system. Aliases let the platform migrate everyone off a deprecated model, or shift traffic to a cheaper equivalent, without 60 teams changing code. Teams that need a pinned version can ask for one, and pay for the privilege of being on the migration hook.

Data classification is enforced here: a request tagged `confidential` cannot route to a vendor endpoint, full stop. That's a gateway policy check, not a guideline in a wiki.

*Self-hosted serving.* vLLM (or equivalent) behind the gateway, with continuous batching and paged KV-cache management — the throughput difference between continuous batching and naive request-level batching is large, with published measurements up to 23x on realistic mixed-length workloads. Each model gets a deployment sized to demand; small models are co-located, large ones get dedicated tensor-parallel groups. I'd keep the number of distinct self-hosted models deliberately small — every additional model fragments GPU memory and hurts batch efficiency, and four well-utilized models beat twelve half-idle ones.

Capacity is the hard scheduling problem. I'd run three priority classes: **interactive** (latency-sensitive, guaranteed headroom), **standard**, and **batch** (preemptible, runs on whatever's free). Batch jobs go to a separate queue with a completion SLA in hours rather than a latency SLA, and get preempted when interactive load spikes. This is what makes a 64-GPU cluster serve a workload that would otherwise need 100 — you're filling the troughs.

*Caching.* An exact-match cache on `(model, prompt, params)` with a TTL catches a surprising amount of internal traffic — teams re-running the same evaluation, retry storms, duplicated pipeline stages. Prefix caching in the serving layer is more valuable still: internal workloads have enormous shared prefixes (the same system prompt across a million classification calls), and reusing that KV cache cuts both latency and cost materially. I'd expose an explicit "this is my stable prefix" hint in the API.

*Accounting.* Every request logs team, alias, resolved model, input/output tokens, cached-token counts, latency, and computed cost — vendor cost from the price sheet, self-hosted cost from an amortized GPU-second rate. This flows to a real-time counter per team (Redis) for quota enforcement and to a warehouse for reporting. Quota enforcement is soft-then-hard: at 80% the team gets alerted, at 100% non-production traffic is rejected while production traffic continues with escalating alerts, and break-glass requires a manager approval that's logged. Cutting off production to enforce a budget is almost always worse than the overspend.

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

*Reliability.* Multi-provider failover behind each alias, circuit breakers on failing endpoints, per-team rate limits so one team's runaway loop doesn't degrade everyone, and hard request timeouts. A shared platform's dominant failure mode is noisy neighbors, and per-team isolation is the answer.

*Developer experience* is most of the adoption battle: a playground UI, prompt/version management, one-line SDK setup, sane defaults, and a dashboard where a team sees its own spend broken down by endpoint. If the platform is harder to use than calling the vendor directly, teams route around it and you lose both cost control and governance.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

The models are the payload; the platform contains almost no AI of its own. That's the point, and saying it plainly is good signal.

The one place I'd consider a model is **semantic caching** (returning a cached response for a semantically similar prompt) and **routing by difficulty** (a small classifier deciding whether a query needs the expensive model). I'd treat both as opt-in and prove them per workload, because semantic cache hits on subtly different prompts produce wrong answers, and difficulty routing that mispredicts silently degrades quality.

I would **not** use a model for quota decisions, data-classification enforcement, scheduling, retries, or cost attribution. And I'd resist the temptation to build "automatic prompt optimization" into the platform — that belongs to teams who can evaluate it, not to shared infrastructure that can't.

**The hard tradeoff** — *self-hosting vs. vendor APIs.*

Self-hosting is cheaper at high sustained utilization, keeps data in-network, and gives you fixed capacity and no rate limits. But it means owning GPU capacity planning, model upgrades, inference-stack bugs, and 3am pages, and it's badly economic at low utilization — a mostly-idle 8-GPU deployment costs more than the API calls it replaces. Vendor APIs have no ops burden and frontier quality, but variable cost that scales linearly forever, rate limits you don't control, and data leaving the boundary.

I'd run both, with a clear rule: **self-host the high-volume, stable, small-model workloads** (classification, extraction, embeddings — where utilization is high and the quality bar is met by open models), and **use vendor APIs for low-volume frontier-quality work** (complex reasoning, agents). The crossover is a utilization calculation I'd do explicitly per workload. What changes my mind is sustained utilization: below roughly 40% on a dedicated deployment, self-hosting is losing money and I'd move that workload to the API.

**How I'd evaluate it.**

Platform metrics, not model metrics: **GPU utilization** (the number that justifies the cluster), tokens/second/GPU by model, p50/p95/p99 latency and time-to-first-token per alias and priority class, queue depth and wait time by class, cache hit rate, error rate by cause (rate limit, timeout, provider error), and **cost per million tokens by alias** compared against the vendor equivalent — that comparison is the platform's ROI statement.

Adoption metrics matter as much: share of company LLM spend flowing through the gateway (traffic routing around it is a failure), number of teams onboarded, and time-to-first-successful-call for a new team. Also: number of teams pinned to a deprecated model, which is the platform's technical-debt gauge.

I'd also run a continuous quality canary per alias — the frozen-prompt daily replay from problem 5 — so that when the platform silently reroutes an alias to a cheaper model, quality change is visible.

**Failure modes I name before the interviewer does.**

- **Noisy neighbor** — one team's agentic loop consuming the whole interactive pool. Per-team rate limits and priority isolation.
- **Silent quality change from alias rerouting** — the platform swaps the model behind `fast-chat` and three teams regress. Requires notice, canary evals per alias, and opt-out pinning.
- **Retry storms** — a provider slows down, every client retries, load triples, everything collapses. Needs client-side jittered backoff, circuit breakers, and load shedding at the gateway.
- **Cost attribution gaps** — a shared service calls the platform on behalf of many teams and everything lands on one cost center. Needs propagated request attribution through internal service calls.
- **Vendor rate limits during a spike** — you don't control them. Multi-provider failover and a queue for non-interactive traffic.
- **GPU fragmentation** — too many model variants leaves memory stranded and batch sizes small. Enforce a small supported model set.
- **Prompt/PII leakage into logs** — logging full prompts is invaluable for debugging and a governance hazard. Redaction, short retention on payloads, and access controls; log metadata forever and payloads briefly.

**Follow-ups they will ask.**

*"How do you decide GPU capacity?"*
From a demand model, not a guess. I'd measure tokens/second/GPU per model under realistic batch conditions (this varies enormously with sequence length and batch composition, so I'd benchmark with replayed production traffic rather than synthetic uniform prompts), then size interactive capacity for p99 demand with headroom, and let batch fill the rest. The key insight is that batch workloads make the utilization math work — sizing for interactive peak alone means paying for idle GPUs most of the day. I'd review capacity monthly against queue-wait metrics, and treat sustained interactive queueing as the trigger to buy.

*"A team says the platform is slower than calling the API directly. What do you do?"*
Measure and be honest. The gateway adds real overhead — auth, policy, logging, routing — which should be low single-digit milliseconds; if it's more, that's a bug, and I'd profile it. But usually the complaint is about queueing on self-hosted capacity under load, which is a capacity or priority-class problem. The fix might be moving that team's workload to a vendor endpoint via routing policy, which the alias system makes a config change. What I wouldn't do is let them bypass the gateway, because then governance and cost control evaporate one exception at a time.

*"How do you handle model deprecation?"*
Aliases make it tractable but not free. Process: announce with a timeline, use usage logs to identify every affected team and endpoint, provide a diff report from running each team's own eval set (if they have one) or a canary set against old and new, migrate teams whose canary shows no regression automatically, and work individually with the rest. Keep the old model available at a premium price for stragglers to create the right incentive. The platform should publish a dashboard of who's still on deprecated aliases, because visibility drives migration far better than emails.

*"What's the cost model for chargeback?"*
Vendor calls are pass-through at list price plus a small platform margin to fund the team. Self-hosted is amortized: total GPU cost (hardware or cloud, plus a share of platform engineering) divided by tokens served, computed monthly and published. I'd deliberately price self-hosted below vendor equivalents to steer high-volume workloads there — pricing is the steering mechanism for a platform, and it works better than policy documents. Batch-priority traffic gets a discount, which is how you get teams to voluntarily move work off-peak.

*"How do you support fine-tuned models?"*
For self-hosted, LoRA adapters served from a shared base model — this is the whole reason to prefer LoRA over full fine-tunes in a platform context, since dozens of adapters can share one base model's weights and batch together, whereas dozens of full fine-tunes need dozens of deployments. I'd give teams an adapter registry, a training pipeline they can invoke, and adapter-level aliases. I'd also require an eval before an adapter goes to production, because the most common outcome of an enthusiastic first fine-tune is a model that's worse than the base with a good prompt.

*"How do you handle streaming and long-running agent requests?"*
Streaming end-to-end via SSE, which means the gateway can't buffer full responses — that constrains the middleware design (token counting and logging happen incrementally, guardrail checks have to be streaming-compatible or applied to buffered windows). For agent loops that run minutes with many calls, I'd offer an async job API: submit, get a job ID, poll or receive a webhook. That decouples the client from connection timeouts and lets those jobs run at batch priority. I'd also enforce per-job budget caps in tokens and dollars, because an agent in a loop is the single most reliable way to generate a surprise five-figure bill.

---

## 9. Document extraction pipeline

### Q: "Design a document extraction pipeline — invoices and contracts in, structured data out."

**Clarifying questions to ask first**

- **How many distinct document templates, and do they repeat?** If 80% of volume comes from 50 recurring vendors, template-specific handling beats a general model. If every document is novel, it's a general extraction problem.
- **What's the cost of an extraction error?** A wrong invoice amount posted to the ledger is a financial error requiring reconciliation; a wrong contract date in a search index is a minor annoyance. This sets the confidence threshold and whether humans review everything.
- **Are documents born-digital PDFs, scans, or photos from phones?** Born-digital gives you exact text and coordinates. Phone photos add skew, glare, and crop problems, and OCR quality becomes the dominant error source.
- **Is the target schema fixed, or does it vary by customer?** Per-customer schemas mean the extraction prompt/model must be configurable at runtime and evaluation must be per-schema.
- **What's the throughput and latency requirement — real-time on upload, or nightly batch?** Batch lets me use expensive multi-pass extraction; real-time constrains it.
- **Does a human review step exist today, and can I keep it?** Human-in-the-loop is usually the right answer here and knowing the available review capacity shapes the confidence thresholds.

**Assume:** 200k documents/month, mixed invoices (70%) and contracts (30%). ~60% born-digital PDF, 40% scans. 2,000 recurring vendors covering 80% of invoice volume. Errors on invoice amounts, dates, and vendor identity are financially material. Per-customer schema variation exists. Near-real-time (under 2 minutes) expected. A 6-person review team exists.

**The design.**

The framing: **this is a pipeline with a confidence-routed human review stage, not a model.** Any design that ends at "the LLM returns JSON" fails, because the interesting question is what happens to the 8% it gets wrong.

*Ingest.* Documents arrive by email, API upload, or SFTP drop. Each gets a content hash for deduplication (the same invoice arrives three times, routinely), a document ID, and immutable storage of the original bytes — never lose the source, because every extraction dispute is settled by looking at it. State lives in a workflow engine so a document's journey is resumable and observable.

*Preprocessing.* Classify document type, detect page count and orientation, deskew and denoise scans, split multi-document PDFs (a 40-page file that's actually 12 invoices — common and easy to miss). Born-digital PDFs get text extracted directly with coordinates; scans go to OCR. I'd keep OCR and layout analysis as a distinct stage producing **text with bounding boxes**, because those coordinates are what make everything downstream verifiable — you can point at where in the document each value came from.

*Extraction*, in a cascade:

1. **Template match.** Fingerprint the document (vendor identity from logo/text, layout hash). If it matches a known vendor template with learned field positions, extract by position plus local pattern matching. This is fast, nearly free, and highly accurate for the 80% of volume from recurring vendors. Templates are *learned*, not hand-authored: after N successful extractions from a vendor, the system infers stable field locations and promotes a template.
2. **Model extraction.** For unmatched documents, a vision-language model gets the page images plus the OCR text and the target JSON schema, and returns structured output with per-field confidence and a source bounding box for each value. Requiring the model to cite a bounding box is the key trick — it makes hallucinated values detectable, because a value that doesn't appear anywhere in the OCR text at the claimed location gets rejected automatically.
3. **Validation**, which is pure business logic and does a lot of the real work: line items must sum to the subtotal, subtotal plus tax must equal total, dates must be plausible and ordered, the vendor must exist in the vendor master (fuzzy-matched), the currency must be valid, the PO number must exist and have remaining balance, the invoice number must not be a duplicate for that vendor. Arithmetic validation alone catches a large share of extraction errors for free, because a misread digit almost always breaks the sum.

*Confidence routing.* Combine model confidence, validation results, and field criticality into a decision: **auto-approve** (all validations pass, high confidence, vendor known, amount below threshold), **review** (queued to a human with the document rendered and extracted fields overlaid on their bounding boxes, so review is a two-second glance rather than a re-keying job), or **reject** (unreadable, wrong document type).

The review UI is where the leverage is. Reviewers should confirm or correct, never re-type. Every correction is written back as a labeled example — the correction store is the pipeline's most valuable asset, driving template learning, prompt few-shot selection, and eventual fine-tuning.

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

*Output* goes to the ERP or downstream system through an idempotent write keyed by document ID, with a reconciliation job that verifies what landed matches what was extracted.

Throughput: 200k/month is ~5 documents/minute average, trivially handled, but arrivals are bursty (month-end) so I'd size the queue for 20x average and let latency degrade for batch drops while keeping the interactive upload path prioritized.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is OCR, the vision-language extraction for novel layouts, and fuzzy vendor matching. The VLM handles roughly 20% of volume — the non-template documents — which is a nice illustration of the 20/80 point: the model is the smallest piece of a system dominated by ingestion, validation, review tooling, and ERP integration.

I would **not** use a model for arithmetic (compute the sums in code — asking a model to verify that line items add up is slower, costlier, and less reliable than `sum()`), for duplicate detection (hashing plus a database constraint), for the ERP write, for currency conversion (a rates table), or for deciding whether an invoice should be paid (that's an approval workflow with authority limits). And I would not let the model output values that don't appear in the source document — every extracted value must be traceable to a span.

**The hard tradeoff** — *general model on everything vs. template-first with model fallback.*

A single VLM on every document is far simpler to build and maintain: no template infrastructure, no fingerprinting, one code path, and it handles new vendors on day one. But it's roughly 10–50x the per-document cost, slower, and — importantly — *nondeterministic*: the same invoice can extract differently on two runs, which is a nightmare for a finance team doing reconciliation.

Template-first is more machinery but gives deterministic, auditable, near-free extraction on the bulk of volume, with the model reserved for the tail. For a finance system where reproducibility is a real requirement, I'd take template-first. What changes my mind is the volume concentration: if the vendor distribution were flat instead of 80/20, templates would never pay for themselves and I'd go model-only, possibly with caching keyed on layout hash to recover some determinism.

**How I'd evaluate it.**

Offline: a labeled set of a few thousand documents stratified by type, vendor, and scan quality, with ground-truth field values. Metrics per field: **exact-match accuracy** for structured fields (dates, amounts, IDs — normalized before comparison, since "01/02/2026" and "2026-01-02" are the same value), fuzzy match for names and addresses, and **F1 for line-item extraction** (a set-matching problem, not a single value). Report per-field, never aggregate — 95% average accuracy can hide 70% on the total amount field, which is the only one that matters financially.

The metric I'd actually optimize is **straight-through processing rate at a fixed error budget**: what fraction of documents can be auto-approved while keeping post-approval errors below, say, 0.1%. That single number captures both quality and cost, and it's the one the business cares about because it maps directly to review headcount.

Online: STP rate, review queue depth and time-to-clear, per-reviewer throughput, correction rate by field (which fields humans fix most is your roadmap), and **downstream error rate** — errors caught after approval, by the ERP, by reconciliation, or by a vendor complaint. That last one is the true false-negative measure and it's the only honest check on whether the confidence thresholds are right.

**Failure modes I name before the interviewer does.**

- **Silent OCR degradation** on a new scanner or fax pipeline — accuracy drops and nothing alerts because the model still returns confident JSON. Mitigation: monitor OCR confidence distributions and per-source STP rates.
- **Hallucinated values that pass validation** — the model invents a plausible total that happens to be arithmetically consistent because it also invented the line items. Mitigation: bounding-box grounding, every value traceable to source text.
- **Multi-document PDFs mis-split** — two invoices merged into one record. Needs explicit splitting with validation on invoice-number changes.
- **Template drift** — a vendor changes their invoice layout and positional extraction silently pulls the wrong field. Mitigation: validation catches most of it, plus monitoring per-vendor rejection rates for step changes.
- **Duplicate payments** — the same invoice ingested twice through different channels with slightly different bytes so the hash differs. Needs semantic dedup on (vendor, invoice number, amount) as a hard constraint in the ERP write.
- **Reviewer rubber-stamping** — under queue pressure, humans approve without looking, and the review stage becomes theater. Mitigation: inject known-bad items to measure reviewer catch rate, and monitor per-reviewer time-per-document.
- **Schema evolution** — a customer adds a required field and historical documents lack it. Needs versioned schemas and a backfill strategy.

**Follow-ups they will ask.**

*"How do you handle a 90-page contract where you need 15 specific clauses?"*
Different problem from invoices — it's retrieval plus extraction, not layout parsing. I'd chunk by clause structure (contracts have reliable heading hierarchies), embed and index the chunks, then for each target field run a retrieval to find candidate clauses and extract from just those. That keeps the prompt small and lets me cite the exact clause and page for each extracted value, which lawyers require. For fields that may be absent, the model must be able to return null with confidence — the most common contract-extraction error is inventing a termination-notice period that the contract simply doesn't specify.

*"What confidence threshold do you use?"*
Derived, not chosen. I'd plot the precision-versus-coverage curve on the labeled set per field, then pick the threshold that hits the error budget with the highest coverage. Critically, thresholds are **per field and per amount band** — a \$50 invoice and a \$500,000 invoice shouldn't share a threshold, so I'd route all high-value documents to review regardless of confidence. I'd also recalibrate quarterly, since model confidence is not stable across model versions and a vendor upgrade silently shifts the operating point.

*"How do you bootstrap when you have no labeled data?"*
Route everything to human review for the first few weeks. That's not a failure state, it's the data collection plan — the review team is doing the work today anyway, and the UI just captures their output as labels. After a few thousand documents I have a real eval set, per-vendor templates for the top vendors, and measured confidence calibration. Then I raise thresholds gradually, monitoring downstream errors. Trying to launch with auto-approval before you have calibration data is how you get a finance incident in week one.

*"What if the customer's schema changes?"*
Schemas are versioned config, not code, so a new field is a config change plus an eval. The mechanics: add the field with a nullable default, run extraction on a sample to measure accuracy on the new field specifically, route it to review-always until its accuracy clears the bar, then let it participate in auto-approval. Historical documents get backfilled by re-running extraction from the immutable raw store — which is exactly why keeping original bytes forever matters. Never migrate by mutating extracted records.

*"How do you handle non-English documents?"*
OCR language detection first, since OCR accuracy is language-dependent and a wrong language model produces garbage. The VLM handles most major languages for extraction, but accuracy varies substantially and I'd measure it per language rather than assume. Validation logic needs localization — date formats, decimal separators (1.234,56 vs 1,234.56 is a genuine source of 1000x amount errors), tax structures, and address formats. I'd treat each language/region as a separate segment with its own eval set and its own thresholds, because a single global threshold will be too loose somewhere.

*"How does this integrate with the ERP?"*
Idempotently and defensively. Writes are keyed by document ID with an upsert, so retries are safe. I'd write to a staging table first, run reconciliation (does the total of what we posted match the total of what we extracted), then commit. The ERP is usually old, sometimes has surprising validation rules, and often rejects records for reasons unrelated to extraction quality — so I need a rejection-handling path that routes ERP failures back to a human queue with the error message. In my experience this integration is a third of the project's effort and none of its glamour, and I'd budget for it explicitly.

---

## 10. Fraud detection

### Q: "Design a fraud detection system for our payments product."

**Clarifying questions to ask first**

- **What fraud types — stolen cards, account takeover, first-party/friendly fraud, merchant collusion, or bust-out?** These have almost nothing in common. Stolen-card fraud is caught by device and velocity signals; first-party fraud looks like a normal customer until it doesn't; bust-out plays out over months.
- **What's the current fraud rate and the current false-positive rate?** A 0.1% fraud rate means extreme class imbalance and a system where false positives dominate the cost. And blocking good customers is usually more expensive than the fraud.
- **Do we bear the loss, or does the issuer?** Liability determines the objective function entirely. If we eat chargebacks, we optimize loss dollars; if not, we optimize customer experience and network compliance.
- **What's the decision latency budget?** An inline authorization decision has maybe 100ms end-to-end; a post-transaction review has hours.
- **What's the feedback delay on labels?** Chargebacks arrive 30–90 days later, which means my labels are always stale and I can't evaluate a model deployed last week using confirmed labels.
- **What manual review capacity exists?** This sets how much of the uncertain band can be routed to humans instead of auto-decided.

**Assume:** card-not-present marketplace payments, 3M transactions/day, fraud rate ~0.15% of transactions and ~0.4% of dollars, we bear chargeback liability. Inline decision budget 100ms p99. Chargeback labels arrive with a 45-day median lag. A 20-person review team.

**The design — and the first thing I'd say out loud: this is not an LLM problem.**

I'd say that explicitly in the interview, because the temptation to reach for a generative model here is exactly the failure mode this round tests for. Fraud detection is a **tabular, imbalanced, adversarial, low-latency classification problem with delayed labels**, and the right tools are gradient-boosted trees, graph features, and rules. An LLM is too slow for a 100ms budget, too expensive at 3M/day, not more accurate on tabular data, and — decisively — not explainable in a way that satisfies regulators who require adverse-action reasons. Where a language model *does* help is at the edges: summarizing a case for a human reviewer, reading free-text merchant descriptions or dispute narratives, and helping analysts write rules. Those are real but peripheral.

*Architecture.* Three layers on the inline path, in order of cost:

**1. Rules engine (sub-millisecond).** Hard blocks and allows: sanctions/OFAC screening, cards on the known-fraud list, velocity limits (N transactions per card per hour, M cards per device per day), impossible-geography checks, and allowlists for trusted merchants. Rules are essential and underrated — they're instantly deployable when an attack starts (the model needs data and a retrain; a rule needs five minutes), fully explainable, and deterministic. Every mature fraud system is rules plus a model, never a model alone.

**2. Feature computation + model (~30–50ms).** This is the engineering core. Features come in three families:
- *Transaction features*: amount, amount relative to the account's history, merchant category, currency, time of day, card BIN characteristics.
- *Velocity/aggregate features*: counts and sums over windows (1h, 24h, 7d, 30d) keyed by card, account, device, IP, email, and shipping address. These are the workhorses. Computing them in under 50ms at 3M/day requires a streaming aggregation layer — Flink or equivalent writing to a low-latency store — not on-demand queries.
- *Graph features*: the highest-value and hardest. Entities (accounts, devices, cards, emails, addresses, IPs) form a graph; fraud is intensely clustered in it. Features like "number of distinct cards on this device in 30 days," "connected-component size," "share of neighbors with prior chargebacks," and shortest-path distance to a known fraudster catch organized fraud that per-transaction features miss entirely. I'd maintain the graph incrementally with precomputed neighborhood aggregates, since a live traversal won't fit the latency budget.

The model is a **gradient-boosted tree ensemble** (XGBoost/LightGBM). It's the right choice: excellent on tabular data, fast to score (sub-millisecond for hundreds of trees), robust to missing features, and interpretable via SHAP values, which matters for both reviewer tooling and regulatory explanation. I'd handle class imbalance with scale-weighting rather than resampling, and I'd train against **dollar-weighted** loss, since a \$4,000 fraud and a \$12 fraud are not equally important.

**3. Decision layer.** The model outputs a calibrated probability; the decision policy converts it to an action using expected value:

$$\mathbb{E}[\text{loss}_{\text{approve}}] = p_{\text{fraud}} \cdot (\text{amount} + \text{chargeback fee}) \quad \text{vs.} \quad \mathbb{E}[\text{loss}_{\text{decline}}] = (1 - p_{\text{fraud}}) \cdot \text{customer LTV impact}$$

Actions are graded, not binary: approve, approve with step-up authentication (3DS challenge — the most valuable middle option, since it converts a decline into a mild friction), route to manual review, or decline. Step-up is where a well-designed system recovers most of the revenue that a binary system throws away.

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

*The label problem is the defining constraint.* Chargebacks arrive 45 days late, so a model trained today uses features from transactions that are at least 45 days stale, and I cannot evaluate this week's model on confirmed outcomes. Mitigations: use fast proxy labels (manual review decisions available in hours, customer fraud reports, issuer decline codes, account-takeover confirmations) for early signal; maintain a **maturity-aware evaluation** where any dataset window is only considered complete after the chargeback horizon; and monitor score-distribution drift as a leading indicator, since a model degrading against a new attack shows up as a distribution shift long before it shows up in chargebacks.

*Logging feature snapshots at decision time is non-negotiable.* Because features are time-windowed aggregates, you cannot reconstruct "what was this card's 24h velocity at 3:14pm on the 8th" from historical tables — you'd leak future information and train a model that looks brilliant offline and fails in production. Snapshotting the exact scored feature vector is the single most important implementation detail in the whole system.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

ML is the GBDT scorer and, optionally, graph embeddings or an anomaly-detection model for unsupervised novelty detection. That's it on the inline path.

I would **not** use an LLM for the decision — too slow, too expensive, not more accurate on tabular features, and unexplainable. I would not use a model for sanctions screening (a regulatory list match, deterministic and auditable), for hard velocity limits, or for the final action policy (expected-value arithmetic in code, with thresholds owned by risk management). I'd use an LLM only off the critical path: generating case summaries for reviewers, parsing free-text dispute narratives and merchant descriptors, and helping analysts translate an observed pattern into a candidate rule. Those are genuinely useful and worth building second.

**The hard tradeoff** — *strict thresholds (block more fraud, decline more good customers) vs. loose thresholds with step-up authentication.*

Strict blocking minimizes fraud loss and is easy to defend internally, but false declines are enormously expensive: the lost transaction, the lost customer (a good customer wrongly declined often never returns), and support cost. The industry's open secret is that false-decline losses typically exceed actual fraud losses by a wide margin.

I'd run loose thresholds with heavy use of step-up authentication and manual review in the middle band, and I'd insist on measuring the false-decline cost rather than assuming it. The way to measure it: a small randomized approval holdout — approve a sample of transactions the model would have declined, eat the fraud loss, and learn the true precision of your declines. It costs real money and it's the only way to know whether your decline population is 80% fraud or 20% fraud. What changes my mind is that measurement: if declines are overwhelmingly fraud, tighten; if they're mostly good customers, loosen and lean harder on step-up.

**How I'd evaluate it.**

Offline, with a strict **temporal split** — train on weeks 1–8, test on weeks 9–10, never random splits, because fraud is temporally correlated and random splitting leaks. Metrics: **precision-recall AUC** (not ROC-AUC, which is misleadingly flattering at 0.15% positive rate), **dollar-weighted recall at fixed decline rate** (the business metric: what share of fraud dollars do we stop while declining only X% of transactions), and calibration — the expected-value policy requires probabilities that mean what they say.

Online: fraud loss in basis points of volume, decline rate, step-up rate and step-up pass rate, manual review volume and precision, chargeback rate by cohort, and approval rate for good customers (segmented — an approval-rate drop concentrated in one country or card type is a targeted regression). Guardrails: false-decline estimate from the randomized holdout, and disparate-impact monitoring across protected-attribute proxies, which is a regulatory requirement in many jurisdictions.

**Failure modes I name before the interviewer does.**

- **Adversarial adaptation** — fraudsters probe with small transactions to find the threshold, then scale. Requires rapid retraining, rules for fast response, and randomized thresholds so the boundary isn't cleanly discoverable.
- **Feature leakage from delayed data** — a feature computed from data that wasn't available at decision time. Prevented by feature snapshotting and point-in-time-correct training joins.
- **Label bias / selection bias** — you only observe outcomes for approved transactions, so the model never learns about the declined population. This is the deepest problem in fraud ML; the randomized approval holdout is the only real fix.
- **Concept drift after a product launch** — a new payment method or geography shifts the distribution and the model's calibration breaks. Needs drift monitoring per segment (see problem 14).
- **Graph feature staleness** — the entity graph lags, so a device that just linked to 40 cards looks clean. Needs streaming graph updates with monitored lag.
- **Cold start on new accounts** — no history, all velocity features null. Needs a separate model or rule set for thin-file accounts rather than feeding nulls to the main model.
- **Over-blocking a legitimate merchant or corridor** — one bad feature interaction wipes out a country's approval rate. Needs per-segment approval-rate alerting, which catches this in hours rather than at the monthly business review.

**Follow-ups they will ask.**

*"When would an LLM actually help here?"*
Three places, all off the critical path. First, reviewer productivity: generating a case summary that pulls together the account history, the graph neighborhood, and the top SHAP contributors into a paragraph, which cuts review time meaningfully. Second, unstructured signals: merchant descriptors, product listings, dispute narratives, and support transcripts contain fraud signal that tabular pipelines ignore — an LLM can turn those into features computed asynchronously and cached. Third, analyst tooling: translating "cards with a Baltic BIN buying gift cards under \$50 within 10 minutes of signup" into a candidate rule with a backtest. I'd build the reviewer summary first because it has the clearest ROI.

*"How do you handle a brand-new attack pattern?"*
Detection before classification. Unsupervised anomaly detection on the transaction stream and on graph structure (a sudden dense component forming) flags novelty without needing labels. Then a human analyst investigates, and the immediate response is a **rule**, deployed within the hour — that's why the rules engine exists. In parallel, confirmed cases get labeled and fed into an emergency retrain, which lands in days. The rule stays until the model demonstrably covers the pattern, then it's retired. I'd track "share of fraud caught by rules vs. model" as a health metric: rising rule share means the model is falling behind.

*"How often do you retrain, and how do you deploy safely?"*
Weekly retrains as a baseline, with the ability to trigger off-cycle when drift alerts fire. Deployment is shadow-then-canary: the new model scores live traffic without acting for a week so I can compare score distributions and decision disagreements against the incumbent, then canary on a small traffic share with close monitoring of approval and decline rates. Because true labels lag 45 days, the canary decision has to be made on proxy metrics and distributional comparison, not confirmed fraud rates. I'd also always keep the previous model warm for instant rollback.

*"How do you explain a decline to a customer or a regulator?"*
SHAP values from the GBDT give per-decision feature attributions, which map to human-readable reason codes through a maintained mapping ("unusual transaction amount for this account," "new device"). Regulations in several jurisdictions require adverse-action reasons, and this is a large part of why a tree model beats a neural network or LLM here — the explanation is derived from the actual decision, not generated post hoc. I'd store the reason codes with the decision record, and be careful that the explanations given externally don't reveal enough for an attacker to reverse-engineer thresholds.

*"How do you decide the review queue priority?"*
By expected value of review, not by score. A transaction with p=0.4 and \$5,000 at stake is worth far more reviewer time than p=0.9 on \$30 — the latter should just be auto-declined. So priority is roughly $p_{\text{fraud}} \times \text{amount} \times$ (uncertainty), with a boost for cases where review yields transferable knowledge (a novel pattern, a large graph cluster). I'd also cluster related transactions so one review decision applies to a whole ring rather than making reviewers adjudicate 200 transactions from one fraud cluster individually — that clustering is often the biggest single win in review efficiency.

*"What's your feature store story?"*
Critical, and it's problem 12 in this document. The requirements here are point-in-time-correct historical joins for training, sub-20ms online reads at 3M/day, and — most importantly — the identical aggregation logic in both paths. I'd define each feature once, in a declarative spec that generates both the streaming job and the batch backfill, and I'd run a continuous consistency check comparing online-served values to recomputed offline values on a sample. Fraud is the domain where training-serving skew hurts most, because the features are time-windowed aggregates and the ways to get them subtly wrong are numerous.

---

## 11. Multi-agent research assistant with tool use

### Q: "Design a multi-agent research assistant that can use tools — web search, internal docs, code execution — to answer complex research questions."

**Clarifying questions to ask first**

- **What's the acceptable latency and cost per question — 30 seconds and \$0.10, or 20 minutes and \$5?** This single answer determines whether I can afford parallel sub-agents at all. Deep research at \$5/question is a different product from an interactive assistant.
- **Are the tools read-only, or can the agent write/execute in systems that matter?** Read-only is a quality problem. Write access makes it a safety and permissions problem, and I'd want a policy layer like problem 2's.
- **Who's the user, and what's the cost of a plausible-but-wrong answer?** For an analyst who verifies everything, some error is tolerable. For a decision input, I need citations and calibrated uncertainty, and possibly a refusal path.
- **Is the question space open-ended, or a known set of research patterns?** If 80% of questions are "competitive analysis of X" or "summarize what we know about customer Y," those become structured workflows and I use far fewer agents.
- **How much does the internal corpus matter vs. the public web?** Internal-heavy means permissions and RAG quality dominate; web-heavy means source quality and freshness dominate.
- **Do users need to see and steer the process, or just get an answer?** Steerable means a streaming UI showing the plan and intermediate findings, which is substantially more product engineering but dramatically better for trust.

**Assume:** internal analysts, open-ended questions, tolerance of 3–10 minutes and roughly \$0.50–\$2 per question. Tools: web search/fetch, internal document RAG (permissioned), a SQL warehouse (read-only), and a sandboxed Python executor. Answers must be cited. Users see progress.

**The design.**

The honest starting position, which I'd state: **most "multi-agent" systems should start as a single agent with good tools.** Multi-agent adds value in exactly one situation — when subtasks are genuinely independent and parallelizable, so you're buying wall-clock time and context isolation. It costs coordination complexity, token multiplication, and a much harder debugging story. So I'd design for a **lead agent with parallel sub-agents on independent branches**, not a committee of specialists chatting.

*Topology.* A **lead/orchestrator** agent decomposes the question into independent sub-questions, spawns a sub-agent per branch, and synthesizes. Sub-agents are identical workers differing only in their assigned sub-question and tool allowlist; they do not talk to each other (agent-to-agent chat is where token budgets and coherence go to die). Each returns a structured findings object: claims with source citations, confidence, and notes on what it couldn't determine.

Decomposition quality is the highest-leverage part. A bad decomposition ("research the market") produces three sub-agents doing the same thing. I'd have the lead produce an explicit plan with sub-questions that are *disjoint and concrete*, show it to the user, and — for the first version — let the user edit it. That plan is also a natural checkpoint for cost control.

*Tools.* Every tool is a typed, permissioned, rate-limited interface with a hard timeout:
- `web_search(query)` / `fetch_url(url)` — with a domain reputation filter and a content-size cap.
- `search_internal(query)` — the RAG system from problem 1, running under the *user's* permissions, never the agent's. Identity propagates end to end; the agent never has ambient authority.
- `query_warehouse(sql)` — read-only role, row limits, statement timeout, and a cost guard so a full-table scan doesn't cost \$400.
- `run_python(code)` — sandboxed (gVisor/Firecracker), no network, ephemeral filesystem, CPU and memory caps, hard wall-clock limit.

Tool results go to a **shared artifact store**, not into the conversation. Sub-agents write findings and fetched documents as artifacts with IDs and pass IDs around; the lead reads what it needs. This is the key context-management move — without it, a research run's context grows quadratically and you hit both the window limit and a large token bill.

*Control loop.* Each agent runs a bounded ReAct-style loop with explicit budgets: max iterations (say 15), max tool calls, max tokens, max wall-clock. Budgets are enforced by the harness, not requested in the prompt. On budget exhaustion the agent must produce a partial answer stating what it didn't finish — a graceful degradation path, not a crash.

*Synthesis.* The lead reads the findings, cross-checks claims (flagging where sub-agents' sources disagree — this is genuinely valuable output), and writes a cited answer. A separate verification pass checks that every citation resolves to a real fetched artifact containing supporting text; unsupported claims get dropped or marked. That verifier is cheap and it's the difference between a research tool and a plausible-text generator.

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

*Observability is the product for the engineers.* Every run produces a trace tree: agent spans, tool calls with arguments and results, token counts, and cost per node. Debugging a multi-agent failure without this is impossible — you cannot tell whether the answer was wrong because search returned junk, because a sub-agent misread a document, or because synthesis dropped a caveat.

*Cost control.* A per-run budget in dollars, enforced by the harness and visible to the user. Sub-agent count capped (I'd start at 4). Use a cheaper model for sub-agents and the expensive one for decomposition and synthesis — that split matters because sub-agent work is high-volume and mechanical while planning and synthesis are where reasoning quality pays. Cache tool results by `hash(tool, args)` within and across runs with a short TTL; research questions on the same topic repeat.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI is the decomposition, the per-branch reasoning and tool selection, and the synthesis. That's the genuinely new capability.

I would **not** use a model for permissions (identity propagation, enforced at each tool), for sandboxing decisions, for budget enforcement (harness-level counters), for retry logic, or for deciding whether a citation is valid (string/span matching against the artifact — deterministic and reliable). I'd also not use an agent where a workflow suffices: if a question type is common and its structure is known, hard-code the pipeline. Agents are for the open-ended tail, and a system that routes 60% of questions to deterministic workflows and 40% to agents is better and cheaper than one that agents everything.

**The hard tradeoff** — *parallel sub-agents vs. a single sequential agent.*

Parallel cuts wall-clock time roughly by the branch factor and gives each branch a clean context window, which measurably improves quality on broad questions. But token usage multiplies — a multi-agent run can easily use 10–15x the tokens of a single-agent run on the same question — and sub-agents duplicate work, miss cross-branch connections, and produce findings that contradict each other in ways the lead must reconcile.

I'd use parallelism only when the lead's plan contains genuinely independent branches, and route narrow questions to a single agent. The decision should be made by the plan structure, not a global setting. What changes my mind is measured quality per dollar: if evaluation shows the single agent within a few points of the multi-agent system at a fraction of the cost, the multi-agent path should be reserved for explicitly requested "deep research" mode. I'd want that comparison run before building the parallel infrastructure, not after.

**How I'd evaluate it.**

This is hard because there's no single right answer, so I'd evaluate on three axes.

Offline: a set of ~150 research questions with expert-written **rubrics** rather than reference answers — a list of facts a good answer must contain, sources it should find, and errors it must not make. Score with an LLM judge against the rubric, calibrated against human scoring. Plus a **verifiable subset**: questions whose answers are checkable (a number in the warehouse, a fact with a canonical source), which gives an objective accuracy number that anchors the fuzzy rubric scores.

Component metrics matter for debugging: decomposition quality (do sub-questions cover the question, judged), tool-call success rate, retrieval recall on the internal branch, and **citation validity rate** (share of citations that actually support their claim — I'd want this above 95% and I'd measure it by spot-checking with humans, since the automated verifier only checks resolution, not support).

Online: task completion rate, user edits to the plan (high edit rates mean bad decomposition), time and cost per question, thumbs and — the best signal — whether the user copied or exported the answer. Guardrails: p95 cost per run, share of runs hitting budget limits, and hallucinated-citation rate.

**Failure modes I name before the interviewer does.**

- **Runaway cost** — an agent loops on a failing tool 40 times. Hard budget enforcement in the harness, plus circuit breakers on repeated identical tool calls.
- **Context overflow mid-run** — the agent's history exceeds the window and important early findings get truncated. Mitigated by the artifact store and periodic compaction into a structured state object.
- **Confident synthesis over thin evidence** — one blog post becomes "industry consensus." Mitigation: source-count and source-quality requirements per claim, and explicit uncertainty in the output format.
- **Prompt injection from fetched web pages** — a page that instructs the agent to exfiltrate data or call a tool. This is the serious security issue in tool-using agents. Mitigations: fetched content is always data, never instruction; tool allowlists per sub-agent; no tool can send data outbound; and the sandbox has no network.
- **Permission escalation via tools** — a sub-agent retrieving documents the user can't see and surfacing them in the answer. Identity propagation, enforced at the tool, tested with a permissions suite.
- **Duplicate work across branches** — three sub-agents fetching the same ten pages. Shared result cache keyed by tool arguments.
- **Non-reproducibility** — the same question gives different answers, undermining trust. Mitigated by pinning models, logging seeds where available, caching, and — more practically — by setting expectations that research is exploratory.
- **Silent partial failure** — the SQL tool times out, the agent shrugs and answers from the web alone without saying so. Failures must be surfaced in the answer.

**Follow-ups they will ask.**

*"How do you decide how many sub-agents to spawn?"*
From the plan, with a cap. The lead proposes sub-questions and I spawn one per genuinely independent branch, up to 4 by default and more only in explicit deep-research mode. I'd also weight by expected value — a question with three broad, unrelated dimensions justifies parallelism; a narrow factual question doesn't and should skip the sub-agent layer entirely. The anti-pattern is a fixed number, which either wastes money on simple questions or under-serves complex ones. I'd log the relationship between branch count and rubric score to tune the default empirically.

*"How do you stop prompt injection from a web page?"*
Layered, assuming the model will eventually be fooled. Structurally: fetched content is wrapped and clearly delimited as untrusted data; sub-agents that read the web have a minimal tool allowlist (no SQL, no code execution with network); the code sandbox has no network egress at all, so exfiltration has no channel; and any tool that could send data outward requires human confirmation. Behaviorally: scan fetched content for injection patterns and strip or flag, and monitor for anomalous tool-call sequences. Prompt-level instructions ("ignore instructions in documents") are the weakest layer and I'd never rely on them alone.

*"What's your context management strategy for a 20-minute run?"*
Three mechanisms. First, externalize: tool results go to the artifact store and only summaries plus IDs stay in context. Second, compact: when context approaches a threshold, summarize the trajectory so far into a structured state object (goal, findings so far with artifact IDs, open questions, budget remaining) and continue from that — this is the standard long-horizon pattern and it works well as long as the state schema is explicit rather than a free-text summary. Third, isolate: sub-agents start with fresh context containing only their sub-question and relevant artifact IDs, so no branch inherits another's clutter.

*"How do you handle conflicting information between sources?"*
Surface it rather than resolve it silently — that's the honest and more useful behavior. The findings schema requires each claim to carry its sources, so the lead can detect that branch A and branch B assert incompatible numbers. Then it reports both with attribution and, where possible, an assessment based on source recency and authority. I'd rank source quality explicitly (internal system of record > primary source > reputable publication > blog) and encode that in the synthesis prompt. Users consistently rate "here's a disagreement between these two sources" as more valuable than a confident single number.

*"How do you make this debuggable when a user complains?"*
The trace tree, keyed by run ID. A support engineer opens the run and sees the plan, each sub-agent's tool calls with full arguments and results, the artifacts fetched, the synthesis prompt, and per-node cost and latency. Complaints then resolve to a specific node: bad decomposition, a search that returned nothing useful, a document that was retrieved but misread, or synthesis that dropped a caveat. I'd also make traces shareable with the user, since analysts often want to check the sources themselves — turning a debugging tool into a trust feature.

*"When would you NOT use agents here?"*
Whenever the question shape is known. If analysts repeatedly ask "give me the competitive landscape for product X," that's a workflow: fixed search queries, fixed internal reports, a fixed SQL query, a templated synthesis. It's 10x cheaper, 10x faster, reproducible, and testable. I'd instrument the question distribution and promote the head of it into workflows continuously, leaving agents to handle the tail. A team that never does this ends up paying agent prices for template work, which is the most common way these systems fail their cost justification.

*"How do you handle a question the system can't answer?"*
It must say so, and this needs designing in. The findings schema has an explicit "could not determine" field, and the lead is required to propagate it — the answer format includes a "what we couldn't establish" section. I'd also detect the pattern where all branches return thin findings and route to an explicit "insufficient information found, here's what we searched" response rather than synthesizing something from scraps. Measuring this is worth doing: on a deliberately unanswerable eval subset, the metric is the refusal rate, and a system that confidently answers unanswerable questions is worse than useless for research.

---

## 12. Feature store / training-serving consistency

### Q: "Design an ML feature store — the core problem being training-serving consistency."

**Clarifying questions to ask first**

- **How many models and teams will share features, and is there actual reuse?** A feature store for one team and two models is over-engineering; the value comes from sharing, and if there's no reuse the honest answer is "you don't need this yet."
- **What's the online read latency and QPS requirement?** 10ms p99 at 50k QPS dictates an in-memory store and precomputation; 100ms at 500 QPS allows much simpler infrastructure.
- **Do you need streaming (real-time aggregate) features, or is batch enough?** Streaming features are where most of the complexity and most of the skew live. If batch daily features suffice, this problem is a quarter as hard.
- **Is point-in-time correctness a known requirement, and does the team understand label leakage?** If they've been doing random splits and naive joins, the first deliverable is fixing training data, not building a store.
- **Who owns feature definitions — a central platform team or the model teams?** This is an org question that determines whether the store is a registry with governance or a shared library.
- **Buy or build?** Managed options exist and are usually right; I'd want to know what forces a build.

**Assume:** 8 ML teams, ~40 models in production, ~600 features with meaningful reuse. Online reads 30k QPS at p99 15ms. Mix of batch, streaming, and on-demand features. Point-in-time correctness is required and currently done wrong.

**The design.**

The core insight to state early: **training-serving skew is not caused by having two stores. It's caused by having two implementations.** A feature store's job is to make one definition produce both paths.

*Feature definition.* Each feature is declared once in a versioned spec in a repo — entity key, source, transformation, aggregation window, freshness SLA, owner, and type. From that single spec, the system generates the batch job, the streaming job, and the online serving read. Teams write the transformation once. Where the transformation can't be expressed declaratively (SQL over the source, or a constrained DSL), the escape hatch is a Python function used by *both* paths from the same code artifact — never a batch SQL query and a separately-written Flink job that "do the same thing." That divergence is the number one cause of models that look great offline and disappoint online.

*Three feature types, three paths:*
- **Batch** (daily/hourly): computed in the warehouse, materialized to both the offline store and the online store. e.g. "user's 90-day purchase count."
- **Streaming** (seconds): computed by Flink/Spark Streaming from an event topic into the online store, with the same aggregation logic mirrored into the offline store for training. e.g. "transactions on this card in the last hour."
- **On-demand / request-time**: computed at serving from request context (e.g. distance between the request IP and the account's home location). These can't be precomputed, so the definition must be a shared function called by the serving path and by the training pipeline's backfill.

*Storage.* Offline store is the warehouse or a lakehouse (Iceberg/Delta), holding the full history with event timestamps — this is what makes point-in-time joins possible. Online store is a low-latency KV store (Redis, DynamoDB, or ScyllaDB) holding only the latest value per entity key, optimized for reads. Materialization jobs push from offline to online, and monitoring their **lag** is a first-class SLO — a stale online store is silent model degradation.

*Point-in-time correctness* is the heart of it. Training data is built from an entity dataframe of `(entity_id, event_timestamp, label)`, joined against feature history with the constraint: take the latest feature value **as of** `event_timestamp`, respecting each feature's availability delay. That last part is what people miss — a feature computed by a daily batch job at 2am isn't actually available for a decision at 1am, even though its logical timestamp says the previous day. Encoding each feature's *availability* time separately from its *event* time is what prevents subtle leakage. A naive `LEFT JOIN ... ON entity_id` is the classic bug that produces a model with suspiciously good offline metrics and no online lift.

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

*The consistency check is the mechanism that actually enforces the promise.* A daily job samples logged serving requests, recomputes each feature offline for that entity and timestamp, and compares. Any feature whose mismatch rate exceeds a threshold raises an alert against its owner. Without this check, you have a feature store that *claims* consistency; with it, you have one that proves it. I'd treat mismatch rate as an SLO per feature.

*Serving path.* A model declares a feature view — the ordered list of features it needs. At inference, one batched multi-get from the online store fetches all of them by entity key. Ordering and defaults come from the registry, so a feature added to the view doesn't silently shift column positions (another classic bug). Latency budget: multi-get p99 under 10ms, which means keeping the feature count per request bounded and avoiding cross-region reads.

*Logging.* Every serving request logs the exact feature vector used, with feature and model versions. This does double duty: it's the input to the consistency check, and it's the source of truth for training the next model version, which sidesteps point-in-time issues entirely for online-logged features.

*Governance.* A registry UI showing each feature's definition, owner, freshness, consumers, and cost. The consumer list is what makes deprecation possible — you cannot safely delete a feature you can't see is used by four models.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

There is essentially no AI in a feature store, and that's worth saying explicitly. This is data infrastructure: stream processing, storage, joins, orchestration, and monitoring. It exists to serve models; it doesn't contain any.

The only place a model might appear is anomaly detection on feature distributions, and even there I'd start with simple statistical tests (PSI, KS) before anything learned. I would **not** use a model to auto-generate features, to decide which features a model should use (that's feature selection during training, and it needs to be reproducible), or to impute missing values at serving time without the identical imputation in training.

**The hard tradeoff** — *precompute everything vs. compute on demand.*

Precomputation gives predictable low serving latency and simple serving code, but it costs storage for every entity whether or not it's ever read (30M users × 600 features is a lot of rows updated on a schedule), and it introduces staleness bounded by the materialization interval. On-demand computation is always fresh and only pays for what's used, but puts computation in the request path, which for aggregate features means querying source data at serving time — usually impossible within a 15ms budget.

I'd precompute batch and streaming aggregates, and compute only genuinely request-dependent features on demand. The interesting middle case is high-cardinality features read rarely: for those I'd use lazy materialization with a cache, accepting a cold-read penalty. What changes my mind is the read/write ratio — if a feature is written for 30M entities and read for 50k, on-demand or lazy is dramatically cheaper, and I'd want that ratio visible in the registry so the choice is data-driven rather than default.

**How I'd evaluate it.**

The primary metric is **training-serving consistency**: the mismatch rate from the daily check, per feature, targeted below something like 0.1% for exact-valued features and within a tight tolerance for floats. This is the whole reason the system exists.

Operational metrics: online read p50/p99 latency, availability, **materialization lag** per feature against its freshness SLA, and null/default rate at serving (a spike means an upstream pipeline broke). Training-side: point-in-time join correctness verified by a test suite with deliberately tricky timestamp cases, and training set build time.

Adoption metrics, which measure whether the platform is worth its cost: number of features reused across two or more models, time from feature idea to production availability (this should drop dramatically — it's the main business case), and share of production models reading through the store versus computing features ad hoc.

**Failure modes I name before the interviewer does.**

- **Silent materialization lag** — the batch job fails, the online store serves yesterday's values, the model degrades subtly. Needs freshness monitoring with alerting per feature, not per job.
- **Definition drift via the escape hatch** — a team "temporarily" writes a custom serving computation and it never comes back. Prevented by the consistency check catching the divergence and by making the sanctioned path easier than the workaround.
- **Point-in-time leakage from availability delay** — the subtlest bug in ML infrastructure, and it inflates offline metrics with no online lift. Prevented by modeling available-time explicitly and by a test suite.
- **Hot-key overload** — a feature keyed by something with skew (a merchant with 30% of traffic) melts one shard. Needs key-level caching and shard-aware design.
- **Backfill cost explosion** — adding a feature requires recomputing two years of history for 30M entities. Needs incremental backfill and cost estimation before the job runs.
- **Schema evolution breaking consumers** — a feature's type or semantics change and four models silently degrade. Features are versioned; semantic changes create a new version, never mutate the old one.
- **Cost with no owner** — the store accumulates hundreds of features nobody reads. Registry-driven usage tracking and a deprecation process with real teeth.

**Follow-ups they will ask.**

*"How do you do the point-in-time join efficiently?"*
As an AS-OF join: sort both the entity dataframe and the feature history by timestamp within entity, then merge. In Spark that's a window function with a range condition; in a modern warehouse there's often a native ASOF JOIN. The performance trick is partitioning feature history by entity key and time so the join is local, and bounding the lookback window (you rarely need feature values from two years before the label event). For very large training sets I'd materialize the joined dataset once and version it, so repeated experiments don't re-run an expensive join — and so two experiments are actually comparable.

*"What if a team needs a feature that doesn't fit the framework?"*
Give them a first-class escape hatch rather than forcing a bad fit — an arbitrary Python transformation registered as a feature, packaged as a versioned artifact used by both paths. The requirement isn't that the transformation be simple; it's that it be *the same code* in both paths and covered by the consistency check. What I'd resist is letting a team compute features entirely outside the store and log them, because then nothing is verified. If they genuinely can't fit, that's useful information about the framework's gaps, and I'd track those requests as a roadmap input.

*"Buy or build?"*
Buy or adopt open source, almost always. Feast, Tecton, Databricks Feature Store, and cloud-native options solve the common cases well, and building this from scratch is a multi-quarter project that produces no differentiated value. I'd build only for a hard constraint the managed options can't meet — an unusual latency requirement, a regulated data-residency need, or an existing streaming stack that doesn't integrate. Even then I'd build the thinnest layer that gives me the single-definition property and the consistency check, and use existing storage underneath.

*"How do you handle features that depend on other features?"*
A feature DAG with dependency-aware materialization: derived features are computed after their inputs, and the orchestrator respects the ordering. The subtlety is freshness — a derived feature is only as fresh as its stalest input, so the registry should compute and display effective freshness transitively rather than letting a team believe a feature is real-time when its upstream is daily. I'd cap dependency depth (two or three levels) because deep chains make lineage debugging painful and amplify any single upstream failure across many models.

*"How does this interact with model deployment?"*
The model artifact declares the feature view and the feature versions it was trained against, and deployment validates that all of them exist online with acceptable freshness — deploying a model whose features aren't materialized should fail loudly at deploy time, not produce nulls at serving time. On the other side, deprecating a feature checks the registry's consumer list and blocks if any live model depends on it. This coupling is what makes the store a platform rather than a library, and it's the part that requires organizational buy-in more than engineering.

*"What's the migration path from an existing mess?"*
Incrementally, and starting with the highest-pain model rather than a big-bang migration. Pick one model, define its features in the store, run the store's values in shadow alongside the existing pipeline, and use the consistency check to find where they disagree — that exercise alone usually uncovers real bugs and builds the case for the platform. Then cut that model over, and repeat. I'd explicitly avoid migrating everything before proving value, and I'd expect the first migration to take much longer than the estimate because it surfaces every undocumented assumption in the old pipeline.

---

## 13. Text-to-SQL over a company database

### Q: "Design a text-to-SQL system so non-technical people can query our company database in English."

**Clarifying questions to ask first**

- **How many tables, and is there a curated semantic layer or is it raw production schema?** 40 curated analytics tables is a tractable problem; 4,000 raw tables with cryptic names and no documentation is a data-modeling project first, and I'd say so.
- **Read-only analytics, or could this touch production data?** Read-only against a warehouse replica is the only design I'd propose. Anything else is a wrong answer.
- **What happens if the answer is subtly wrong — does someone make a decision on it?** A silently wrong number is worse than an error, because nobody catches it. This drives how much verification and how much "show your work" the product needs.
- **Are there existing certified queries, dashboards, or dbt models I can learn from?** These are gold: they encode the real business definitions and give me few-shot examples and a semantic layer for free.
- **Who are the users — analysts who can read SQL, or executives who can't?** Analysts can verify the SQL, which changes the risk profile enormously. Executives can only verify the number, which they can't.
- **What are the permission requirements — row-level security, column masking?** These must be enforced by the database, not by the generated SQL.

**Assume:** cloud warehouse (Snowflake/BigQuery-class), ~4,000 raw tables but ~120 curated dbt models that cover 90% of questions. Mixed audience, majority non-technical. Read-only. Row-level security exists in the warehouse. ~2,000 questions/week expected. Existing library of ~500 dashboard queries.

**The design.**

The honest opening: **on the BIRD benchmark, which uses realistic messy databases, the top published systems reach about 82% execution accuracy on the test set against a human baseline of roughly 93%.** So roughly one in five hard queries is wrong even for state-of-the-art systems. That number should drive the entire design — the product cannot be "ask a question, get a number." It has to be built so that wrong answers are visible and correctable.

*Layer 1: the semantic layer, which is most of the work and most of the value.* I would not point a model at 4,000 raw tables. Instead I'd expose a curated set of ~120 models with: clear names, column descriptions, documented business definitions ("active_user means logged in within 28 days"), declared join paths and grains, and certified metrics ("revenue = sum of net_amount excluding internal accounts"). Where dbt already has this, it's a matter of harvesting `schema.yml` and query history; where it doesn't, building it is the project. A published analysis of BIRD-style benchmarks makes the point that the data model *is* the semantic layer — most text-to-SQL errors are the model not knowing your business definitions, not the model being bad at SQL syntax.

*Layer 2: retrieval.* With 120 models and hundreds of columns, I can't put the whole schema in the prompt (and shouldn't — irrelevant schema actively degrades accuracy). So schema selection is a retrieval problem: embed table and column descriptions plus sample values, retrieve the top ~10 tables for the question, and include their full DDL, descriptions, join paths, and 3 sample rows each. Sample values matter more than people expect — knowing that `status` contains `'ACTIVE'`, `'CHURNED'`, `'TRIAL'` prevents the model from inventing `'active'`.

Also retrieve **similar past queries**: the 500 dashboard queries plus accumulated successful queries, embedded by their natural-language description. A near-match past query is the strongest possible few-shot example, and for the repetitive head of question traffic it often just needs a parameter change.

*Layer 3: generation with verification.* The model generates SQL along with a plain-English restatement of what it's computing. Then a verification cascade before anything runs:
1. **Parse and validate** with a SQL parser against the real schema — every table and column must exist, types must be compatible. This catches hallucinated columns for free, deterministically.
2. **Static policy checks** — read-only enforced (no DML/DDL, enforced by the connection's role, not by inspecting the SQL), required row-level predicates present, no cross-joins without conditions, a `LIMIT` applied.
3. **Dry run / EXPLAIN** for cost estimation. If the plan scans 40TB, block it and suggest a narrower question — this is the difference between a helpful tool and a \$50,000 warehouse bill.
4. **Execute** with a statement timeout and row cap, under the *user's* warehouse role so RLS and column masking apply.

*Layer 4: presentation, which is where accuracy problems get managed.* The user sees the result, the English restatement, the SQL (collapsible), and — critically — the assumptions the system made ("counted distinct users, excluded internal accounts, used the order_date not ship_date"). Ambiguity is surfaced, not silently resolved: if "revenue" could mean gross or net, the system asks or states which it chose. I'd also show a sanity panel: row count, date range covered, and comparison to the same metric from a certified dashboard when one exists.

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

*The repair loop* is worth designing explicitly: on a validation or execution error, feed the error message back with the SQL and let the model fix it, capped at 2 attempts. This recovers a meaningful share of failures at low cost. What it cannot fix is a query that runs successfully and returns the wrong number, which is the failure mode that matters.

*The feedback flywheel.* Every query gets thumbs and an edit affordance. Analyst-corrected queries go into a review queue, and approved ones join the certified library — which improves few-shot retrieval for everyone. Over time, the head of the question distribution gets covered by certified queries and accuracy on common questions approaches 100% while the tail stays at model-level accuracy. That asymmetry is the product strategy: make the common case certifiably right rather than making everything slightly better.

Caching: identical questions hit a result cache with a short TTL; the query text cache is longer-lived since the SQL for a given question rarely changes.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

AI does two things: pick the relevant tables (a retrieval problem) and write SQL from intent plus schema. Everything else — validation, policy, permissions, cost control, execution, presentation — is ordinary engineering, and the semantic layer that makes the AI work is data modeling, not ML.

I would **not** use a model for permissions (execute as the user; the database enforces RLS — a model that "remembers to add the tenant filter" is a data breach waiting to happen), for read-only enforcement (a database role), for cost limits (EXPLAIN plus quotas), for validating that columns exist (a parser), or for interpreting the results into a business recommendation without the user seeing the numbers. And I would not let generated SQL run against production OLTP.

**The hard tradeoff** — *open-ended generation over the full schema vs. a constrained semantic layer with certified metrics.*

Open generation answers any question including ones nobody anticipated, and requires no upfront modeling investment. But accuracy on messy real schemas caps out around the BIRD numbers, and the errors are silent — a plausible number computed from the wrong join. Constrained generation over certified metrics and dimensions (essentially generating a semantic-layer query that compiles to SQL) gets much higher accuracy on the questions it covers, guarantees consistent business definitions, but returns "I can't answer that" for anything outside the model.

I'd go constrained-first with an open fallback that is clearly labeled as unverified and shows its SQL prominently. For a mostly non-technical audience making decisions on the numbers, consistent-and-limited beats flexible-and-sometimes-wrong. What changes my mind is the audience: for a room full of analysts who read the SQL anyway, open generation is a genuine productivity win and the verification burden sits with a competent user.

**How I'd evaluate it.**

Offline: a benchmark of a few hundred questions **on your own schema** — public benchmarks tell you about the model, not about your database. Build it from real questions in the analytics Slack channel, with analyst-written gold SQL. The metric is **execution accuracy**: does the generated query return the same result set as the gold query (compared as sets, order-insensitive, with numeric tolerance). Exact SQL match is the wrong metric — many correct queries are textually different.

Break it down by difficulty (single-table, joins, aggregations with filters, window functions, nested logic) because aggregate accuracy hides that complex analytical questions are much worse. I'd also track **valid-execution rate** (does it run at all) separately from correctness — these have different fixes.

Online: thumbs rate, query edit rate (how often does a user modify the SQL — a strong implicit correctness signal), re-ask rate, share of questions answered from the certified library vs. generated, warehouse cost per question, and p95 latency. The metric I'd watch hardest is **silent error rate**, estimated by having an analyst audit a random sample of ~50 answered questions per week and judge correctness. Everything else can look healthy while this is bad.

**Failure modes I name before the interviewer does.**

- **Silently wrong joins** producing plausible numbers — the defining failure. Fan-out on a one-to-many join inflates sums and nothing errors. Mitigated by declared join paths and grains in the semantic layer, and by sanity comparison against certified metrics.
- **Business-definition mismatch** — the system's "active user" isn't finance's. Mitigated by certified metrics and by stating assumptions in every answer.
- **Runaway query cost** — a cross join on two billion-row tables. Mitigated by EXPLAIN gating, row caps, timeouts, and per-user warehouse quotas.
- **Ambiguity resolved silently** — "last quarter" meaning fiscal or calendar. Must be surfaced, and where the system guesses, it must say so.
- **Stale schema in the retrieval index** — a column is renamed and the model generates against the old name. Needs schema sync on every dbt deploy with monitored lag.
- **Permission bypass** — the classic disaster where the service account has full access and the generated SQL is trusted to filter. Prevented by execution under the user's role, and tested with a permissions suite.
- **Over-trust** — users treat outputs as authoritative and stop checking. Mitigated by explicit confidence/verification status in the UI and by making the certified-vs-generated distinction visually obvious.

**Follow-ups they will ask.**

*"How do you handle 'show me revenue last quarter' when revenue is defined three ways?"*
I don't let the model choose silently. If the question maps to an ambiguous metric, the system either asks ("gross bookings, net revenue, or recognized revenue?") or picks the certified default and states it prominently in the answer. The deeper fix is organizational: the certified metric library forces the company to actually decide, and the text-to-SQL project frequently becomes the forcing function for defining metrics that have been ambiguous for years. I'd flag that in the interview because it's the kind of non-technical work that determines whether the project succeeds.

*"What if the question needs a table that isn't in the curated set?"*
Fall back to retrieval over the full schema with a clear "this used uncurated tables, please verify" label, and log it as a coverage gap. Those logs are the prioritized backlog for what to curate next — if 40 people a week ask about a table that isn't modeled, that's an obvious data-modeling task. I'd rather have a measured coverage gap than a system that silently produces lower-quality answers on uncurated data with no visibility.

*"How do you keep warehouse costs under control?"*
Multiple layers, because a single one will be bypassed. EXPLAIN-based estimation blocks queries above a byte-scan threshold before execution. Statement timeouts and row limits cap the damage from anything that gets through. Per-user and per-team credit quotas in the warehouse itself provide the hard backstop. Result caching handles repeats, which are a large fraction of traffic. And I'd default to querying pre-aggregated tables where they exist rather than raw fact tables — usually a 100x cost difference for the same answer, and it's a semantic-layer routing decision, not something the model should reason about.

*"How do you handle multi-turn — 'now break that down by region'?"*
Keep conversation state as the previous SQL plus its result schema, and treat the follow-up as a transformation of that query rather than a new generation from scratch. That's both more accurate and cheaper. I'd represent the query structurally (a parsed AST or semantic-layer query object) so "break down by region" is a mechanical group-by addition rather than a regeneration that might change the filters too. Where the follow-up can't be expressed as a transformation, fall back to full regeneration with the prior turn as context, and re-state the assumptions since they may have changed.

*"Which model would you use, and would you fine-tune?"*
Start with the strongest available general model, because SQL generation quality tracks general reasoning closely and the cost per query is small relative to warehouse cost. I'd invest in the semantic layer and retrieval before touching the model — those move accuracy far more. Fine-tuning becomes attractive once you have a few thousand certified query pairs on your schema and want to shrink cost or latency; a smaller fine-tuned model on your own schema can beat a larger general one, because most of the difficulty is schema-specific knowledge rather than SQL skill. But I'd treat that as a phase-two optimization with a measured baseline.

*"How would you support 'why did revenue drop' style questions?"*
That's not text-to-SQL, it's root-cause analysis, and conflating them is a common product mistake. It requires generating and running many queries — dimensional decompositions across region, segment, product, cohort — then ranking the contributions to the change. I'd implement it as a structured workflow, not free-form generation: enumerate the dimensions from the semantic layer, run the decomposition, compute contribution to variance, and present the top drivers with charts. The LLM's role is picking which dimensions are plausible and writing the narrative; the analysis itself is deterministic arithmetic, which is the right split.

---

## 14. Model and data drift detection

### Q: "Design a system to detect and handle model and data drift in production."

**Clarifying questions to ask first**

- **How many models, and are they the same type?** Twenty GBDTs sharing a feature store lets me build one monitoring system. A mix of GBDTs, embeddings, and LLM endpoints needs different detectors for each.
- **How delayed are the labels?** Immediate labels (ad clicks) let me monitor real performance directly and drift detection is a secondary concern. 90-day labels (credit, churn, fraud) mean proxy signals are the primary defense.
- **What's the cost of acting on a false alarm vs. missing real drift?** Auto-retraining on every alert is expensive and risky; ignoring alerts makes the system decorative. This sets the alerting threshold.
- **Is retraining automated and safe today?** Detection is useless without a response path. If retraining takes a human three weeks, the design should focus on making retraining cheap, not on fancier detectors.
- **Are there regulatory requirements around model monitoring and revalidation?** In finance and healthcare this is mandated with specific documentation, which changes what I build.
- **How segmented is the population?** Aggregate stability with per-segment collapse is common and only visible if you're slicing.

**Assume:** 40 models across 8 teams, mostly tabular GBDTs plus a few embedding-based rankers and two LLM-backed features. Label delay ranges from minutes to 60 days. Automated retraining exists for 15 of the 40. Some models are regulated and need documented monitoring.

**The design.**

Drift is not one thing, and naming the four kinds precisely is high signal:

1. **Data/covariate drift** — $P(X)$ changes. Input distributions move. Detectable immediately, no labels needed.
2. **Prediction drift** — $P(\hat{Y})$ changes. The model's output distribution moves. Also immediate, and often the most sensitive early warning.
3. **Concept drift** — $P(Y \mid X)$ changes. The relationship itself changed. Only detectable with labels, and it's the one that actually destroys model value.
4. **Upstream data quality breaks** — a schema change, a broken join, a unit change from meters to feet. Not really "drift," and by far the most common cause of production model failures. Any monitoring system that only does statistical drift and misses "this column became all nulls at 3am" is solving the wrong problem first.

*Architecture.* Serving logs — inputs, predictions, feature vectors, model version — stream to a monitoring store. A scheduled job computes, per model and per segment, per window (hourly and daily):

- **Data quality checks first**, because they're the highest-yield: null rate per feature, cardinality, type conformance, range violations, and freshness. These are assertions with thresholds and they catch the majority of real incidents. Great Expectations-style checks, run as a gate.
- **Distribution drift** per feature against a fixed reference window (the training distribution). For numeric features, **PSI** and the Kolmogorov–Smirnov statistic; for categorical, PSI and chi-square. The conventional PSI reading is **< 0.1 no significant shift, 0.1–0.25 moderate, > 0.25 significant** — widely used, though the thresholds are a rule of thumb rather than a statistical guarantee, and I'd calibrate them per feature by measuring historical PSI on periods where the model was known to be healthy.
- **Prediction drift** — PSI on the score distribution, plus the mean and quantiles of predictions over time. Cheap, label-free, and moves early.
- **Performance monitoring** where labels exist, with a maturity-aware window: a metric is only computed on data old enough for its labels to have arrived. For delayed-label models I'd compute metrics on a rolling matured window and report the label maturity explicitly, so nobody misreads a partially-labeled recent window as a performance drop.
- **Proxy performance** for the label gap: agreement with a human-reviewed sample, calibration drift (are predicted probabilities still matching observed rates on matured data), and score-distribution stability.

Segmentation is not optional. Every metric is computed per key segment — geography, customer tier, device, channel, product line — because a model can be perfectly stable in aggregate while one segment collapses, and the segment that collapses is usually the one with a business owner who will notice loudly.

Statistical care matters: with 600 features × 40 models × 10 segments, naive per-feature p-value alerting produces thousands of daily false alarms. So I'd use effect-size thresholds (PSI) rather than p-values (which flag trivially small shifts at high sample size), apply multiple-testing correction, require persistence (drift must hold for N consecutive windows), and rank alerts by **impact-weighted drift** — a feature's PSI multiplied by its importance in the model. A high-drift feature with 0.1% importance is noise; a moderate drift in the top feature is an emergency.

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

*The response path is the part people forget, and it's where I'd spend my design time.* An alert must route to a runbook, not a Slack channel nobody reads. Triage: is this a data quality bug (fix upstream — most common), an expected seasonal shift (annotate and suppress), a real distribution change with performance impact (retrain), or a catastrophic break (roll back or fall back to a baseline)?

Automated retraining is appropriate for models with fast labels and a proven pipeline, and it must be gated: the retrained model has to beat the incumbent on a held-out set *and* pass a shadow comparison before promotion. Auto-retraining without a gate is how a data quality bug becomes a permanently baked-in model regression. For regulated models, retraining requires documented revalidation and cannot be automatic.

I'd also maintain a **fallback** per model — the previous version kept warm, and a simple rules or heuristic baseline underneath that. Being able to say "we degraded to the rules baseline while we investigated" is a much better incident outcome than "the model kept serving garbage for six hours."

*For LLM-backed features*, drift looks different: input distribution drift (users asking new things), vendor model updates (see problem 5's frozen canary), and output drift (response length, refusal rate, tool-call rate, judge scores). I'd monitor those with the same infrastructure but different detectors.

**Where the AI actually is — and what I would deliberately NOT use a model for.**

Almost none. This is statistics, scheduling, storage, and alerting. Drift detection is one of the clearest cases where the sophisticated-sounding option (train a model to detect drift) is worse than PSI and a null-rate check.

There's one legitimate ML use: a **domain classifier** trained to distinguish reference-window data from current-window data. If it can separate them with high AUC, multivariate drift exists even when no single feature moved — this catches correlation-structure changes that univariate tests miss. I'd add it as a secondary detector, not a primary one, because it's harder to interpret. I would **not** use a model to decide whether to retrain (a policy with thresholds), to set alert thresholds (calibrated from historical data), or to explain drift to on-call — though an LLM writing the incident summary from the computed statistics is a genuinely nice quality-of-life feature and a fair place to use one.

**The hard tradeoff** — *automatic retraining on drift vs. human-gated retraining.*

Automatic keeps models fresh without headcount and responds within hours, which for fast-moving domains (fraud, recommendations) is worth a lot. But it can bake in a data bug, it makes the production model a moving target that's hard to reason about during an incident, and in regulated contexts it's not permitted. Human-gated is safer and auditable but slow — a model can be degraded for weeks waiting on someone's queue.

I'd split by model criticality and label speed: automatic retraining with hard promotion gates for high-velocity, low-stakes models where labels arrive fast; human-gated for high-stakes and regulated ones. And regardless of which, the retrained model always goes through shadow and canary rather than direct promotion. What changes my mind is the observed track record — if six months of automated retrains have never produced a regression that the gates missed, I'd widen the automatic set; a single bad promotion sends me back to gating.

**How I'd evaluate it.**

Evaluate the detector, which people rarely do. **Detection recall**: replay historical incidents (known outages, schema changes, performance drops that were eventually noticed) and measure what fraction the system would have caught, and how many hours earlier than the actual discovery. That's the ROI number. **False alarm rate**: alerts per model per week that triage marks as no-action — above roughly two or three a week per team, alerts get ignored and the system is worthless regardless of its recall. **Time to detection** and **time to resolution** per incident class.

I'd also run synthetic injection: deliberately corrupt a feature in a shadow stream (null it out, shift its mean by 2σ, swap units) and verify the system fires within the expected window. That's a continuous test of the monitoring itself, and it catches the embarrassing case where monitoring silently stopped working — which happens more often than anyone admits.

**Failure modes I name before the interviewer does.**

- **Alert fatigue** — the dominant failure. Hundreds of statistically significant but meaningless alerts and everyone mutes the channel. Mitigated by effect sizes, importance weighting, persistence requirements, and aggressive tuning of what pages versus what appears on a dashboard.
- **Reference window rot** — comparing against a training distribution from two years ago flags every normal evolution. Needs a policy: fixed reference for "have we left the training distribution" plus a rolling reference for "did something change suddenly."
- **Seasonality misread as drift** — Black Friday fires every detector. Needs seasonal baselines or year-over-year comparison for known-seasonal features.
- **Aggregate stability hiding segment collapse** — the most common way real degradation is missed.
- **Drift with no performance impact** — a feature moves but the model doesn't care. Wastes triage time; mitigated by importance weighting and by always asking for the performance evidence before acting.
- **Performance drop with no drift** — the labels changed meaning, or the business process changed. Detectors won't see it; only label monitoring and human reports will.
- **Monitoring the model but not the pipeline** — the feature store's materialization job failed and the model is serving stale features, which looks like *less* drift, not more. Freshness monitoring is separate and essential.
- **The monitoring system itself failing silently** — the job stops, no alerts fire, everyone assumes health. Needs a heartbeat and the synthetic injection test.

**Follow-ups they will ask.**

*"What's your reference window?"*
Two of them, deliberately. A **fixed** reference — the training distribution — answers "are we operating outside what the model learned," which is the question that matters for validity. A **rolling** reference, typically the previous 7 or 28 days, answers "did something change suddenly," which is the question that matters for incidents. They fire on different things and both are useful: gradual drift shows only against the fixed reference, while a schema break shows immediately against the rolling one. Reporting both prevents the common confusion where a team suppresses gradual-drift alerts and then can't see that they've drifted a long way from training.

*"How do you handle drift when labels take 60 days?"*
Lean on the label-free signals and on proxies. Feature and prediction drift give same-day warning. Calibration on matured data gives a delayed but trustworthy read. In between, I'd invest in fast proxy labels — a human-reviewed sample (even 200 items/week gives useful signal), early indicators that correlate with the eventual label (a first-payment default predicts eventual charge-off), and business metrics that move faster than labels. I'd also explicitly report label maturity alongside every performance metric, because the most common analytical mistake in delayed-label settings is reading a partially-matured recent window as a performance cliff.

*"How do you avoid alert fatigue?"*
Tier the responses. Very few conditions should page: a data quality break on a critical feature, a performance drop beyond a wide threshold on matured labels, or a materialization failure. Most drift belongs on a weekly review dashboard, not in anyone's phone. Then use effect size over significance, weight by feature importance, require persistence across windows, and group correlated alerts into a single incident (twenty features drifting simultaneously is one upstream problem, not twenty alerts). Finally, measure the false-alarm rate and treat a rise in it as a bug in the monitoring system, with the same urgency as a production bug.

*"When do you retrain versus rebuild?"*
Retrain — same features, same architecture, fresh data — handles ordinary distribution shift and should be routine and cheap. Rebuild is warranted when retraining stops recovering performance, which signals that the relationship changed in a way the current feature set can't capture, or when the world genuinely changed (a new product line, a new fraud modality, a new regulation). The diagnostic is straightforward: retrain on recent data and evaluate; if the retrained model recovers, it was distribution shift; if it doesn't, you need new features or a new formulation. I'd track "performance recovered by retraining" as a metric per model, since a declining trend is the signal that a rebuild is due.

*"How do you monitor an embedding-based system?"*
Different detectors, same framework. Monitor the embedding distribution (mean cosine distance to the training centroid, or PSI over projected dimensions), the retrieval score distribution (a drop in top-1 similarity across the board means queries are moving away from the corpus), the share of queries with no good match above threshold, and the click/engagement rate on retrieved results. Also monitor corpus drift separately — the index changing under a static query distribution is a distinct failure. And crucially, track the embedding model version everywhere, since the most dramatic "drift" event in these systems is a silent model upgrade.

*"How does this fit with the eval system from question 5?"*
They're the same system with different detectors, and I'd build them on shared infrastructure — the same logging, the same metric store, the same segmented time-series comparison, the same alerting and triage. What differs is what gets computed: statistical distances and performance metrics for tabular models, assertion rates and judge scores for LLM products. Building them separately, which is the default in most organizations, means two dashboards, two on-call runbooks, and no shared view when an incident spans both. I'd unify them explicitly and treat "model quality observability" as one platform.

---

## Closing note on what this round is actually testing

Read back over the fourteen answers and notice what the *design* sections spend their words on. Connectors, queues, idempotency keys, ACL propagation, bounding boxes, point-in-time joins, priority classes, audit logs, review queues, runbooks. The model choice is usually one or two sentences — "a cross-encoder reranker," "a GBDT," "a vision-language model" — and it's rarely the interesting decision.

That ratio is the message. In every one of these systems, the AI is a component inside a much larger piece of ordinary software, and the failure modes that kill the product in production are almost never "we picked the wrong model." They're stale permissions, double-executed refunds, unlogged feature vectors, alert fatigue, an OCR pipeline quietly degrading, a vendor silently updating a model.

Four habits that separate a strong performance from an average one:

- **Say what you would not use a model for.** It is the fastest way to demonstrate judgment, and most candidates never do it.
- **Name the human in the loop.** Reviewers, approvers, on-call, analysts. Systems that assume full automation on day one don't survive contact with production.
- **Design the failure path as carefully as the happy path.** What serves when the model is down, what happens on a timeout, what the rollback looks like.
- **Quantify.** QPS, p95, dollars per thousand requests, error budgets. Even rough numbers, clearly labeled as assumptions, show you've built something before.

And when you genuinely don't know a number, say "I'd assume roughly X and measure it." Interviewers trust that far more than a confident fabrication.

---

## Sources for cited figures

Numbers in this document that are not marked as assumptions come from:

- [BIRD-SQL benchmark leaderboard](https://bird-bench.github.io/) — top execution accuracy ~82% on test, human baseline 92.96% (used in problem 13).
- [Your Data Model Is the Semantic Layer — MotherDuck](https://motherduck.com/blog/bird-bench-and-data-models/) — argument that text-to-SQL errors trace to business/data modeling rather than SQL generation (problem 13).
- [Hybrid Search: BM25, Vector & Reranking Reference 2026 — Digital Applied](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026) — WANDS nDCG figures: BM25 0.6983, dense KNN 0.6953, RRF hybrid 0.7068, hybrid with field boosting 0.7497 (problems 1 and 4).
- [How Continuous Batching Enables 23x Throughput in LLM Inference — Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference) — continuous batching throughput gains over naive batching (problem 8).
- [Measuring Data Drift with the Population Stability Index — Fiddler AI](https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index) and [Population Stability Index (PSI) Metrics — Arthur](https://docs.arthur.ai/docs/population-stability-index-psi-metrics) — the conventional PSI bands of 0.1 and 0.25 (problem 14).

All latency budgets, cost figures, corpus sizes, QPS numbers, and traffic assumptions in the "Assume" blocks are stated scenario assumptions, not measurements. Any figure introduced with "I'd assume" or "(assumption)" should be treated as a planning estimate to be verified against your own system.
