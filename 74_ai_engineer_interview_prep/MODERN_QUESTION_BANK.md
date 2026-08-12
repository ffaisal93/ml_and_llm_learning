# The modern AI engineer question bank

Scope: the applied GenAI layer — the job of *building with* models rather than training them. RAG, agents, prompting and context engineering, evaluation, LLMOps, cost and latency, deployment, safety.

This chapter deliberately does **not** cover classical ML theory, transformer internals, backprop, or optimizer math. Those live in `13_interview_qa` (classical ML Q&A), `64_integrated_ai_ml_interview_synthesis` (cross-cutting synthesis), and `73_night_before_review` (the compressed pass). Retrieval mechanics in depth — index structures, ANN algorithms, embedding geometry — are in `39_rag_retrieval_augmented_generation`. Read those first if you are shaky on fundamentals; an AI engineer loop will still drop a bias-variance or attention question on you, it just is not the discriminating part.

**A note on how to use this.** The questions here are aggregated from public sources plus what the shape of the role actually demands (sourcing note in §11). The questions are the cheap part — they are on twenty websites. The answers are the point. Every answer below is written as what a strong candidate would actually say out loud: specific, honest about the tradeoff, usually naming the failure mode. Where an answer depends on facts that move — model capabilities, pricing, context limits, tooling — it is marked **[verify before quoting]**.

---

## 1. The opener

### "Walk me through a GenAI system you built end to end."

This is the first question in most AI engineer loops and it silently determines the rest of the interview. The interviewer is building a model of you in the first four minutes and then spends forty minutes testing that model. Get this wrong and you spend the loop climbing out.

**The four things the interviewer is actually listening for:**

1. **Did you own a real constraint?** Latency budget, cost ceiling, accuracy bar, compliance requirement. Candidates who built demos talk about capabilities. Candidates who shipped talk about constraints. "We had 3 seconds p95 and a \$0.02/query ceiling" is worth more than any architecture diagram.
2. **Do you have numbers?** Not vanity numbers — decision numbers. "Retrieval recall@10 went 71% → 89% after we added BM25, which moved end-to-end answer accuracy from 62% to 74%." If you cannot say what you measured, the interviewer concludes you did not measure.
3. **Did you make a tradeoff and can you defend the road not taken?** Every real system has three or four forks. Name one, name what you gave up, name what would have changed your mind.
4. **What broke in production?** This is the single strongest signal. Anyone can describe a happy path. Only people who operated a system can tell you the failure that surprised them. Have one ready that is genuinely embarrassing and genuinely instructive.

**Structure to use — roughly 4 minutes, then stop and let them dig:**

- *Problem and constraint* (30s): who the user was, what they were doing before, the hard number you had to hit.
- *Architecture in one pass* (60s): the flow, no detours. "Query → rewrite → hybrid retrieve over 400k chunks → rerank top 50 to top 8 → generate with citations → groundedness check before display."
- *The one hard decision* (60s): the fork, both branches, why you picked.
- *Evaluation* (45s): how you knew it worked, offline and online.
- *What went wrong and what you did* (45s): the production failure.
- *Where it is now / what you would change* (20s).

Then stop talking. Do not narrate every component. The interviewer will pick the thread they care about, and that is where you want the depth to go.

**Worked example answer:**

> "We built an internal support assistant for a B2B SaaS company — support agents were spending most of a ticket reading through a 12,000-document knowledge base plus historical tickets. The constraint that shaped everything was that this sat inside the agent's workflow, so anything over about 4 seconds and they would just go back to Ctrl-F. And we had a hard requirement that every claim be traceable to a source document, because wrong billing guidance had actual money attached.
>
> Architecture: query goes through a lightweight rewrite step that resolves pronouns against conversation history and expands product acronyms. Hybrid retrieval — BM25 plus dense over roughly 400k chunks — fused with reciprocal rank fusion, top 100 candidates, cross-encoder rerank down to 8. Generation with mandatory inline citations, then a cheap groundedness check on the output before it renders.
>
> The hard decision was whether to rerank at all. It cost us about 300ms and a real per-query fee. We ran it both ways on a 200-question eval set and reranking moved answer accuracy 9 points, mostly because our chunks were noisy and the embedding model was pulling in topically-adjacent-but-wrong content. Nine points was worth 300ms. If our corpus had been cleaner I think we would have skipped it — reranking is partly a patch for bad retrieval.
>
> Evaluation was in two layers, deliberately. Retrieval on its own against a labeled set of question-to-document pairs, because when the system was wrong I needed to know whether the right document was even in the context. Then end-to-end, with an LLM judge on faithfulness and helpfulness that we calibrated against about 300 human-labeled examples — the judge agreed with humans around 85% of the time on faithfulness, which was good enough to use it as a regression gate but not good enough to report as ground truth.
>
> The thing that broke: we were incrementally re-indexing on document update, and deletes were not propagating. So we had a live index serving deprecated pricing pages for about three weeks. The eval set did not catch it because the eval set was frozen and the corpus was not. That taught me that index freshness is a monitored property, not a pipeline that you write once — we added a nightly reconciliation job comparing source-of-truth doc IDs against index IDs, and an alert on drift."

Note what that answer does: names a constraint, gives numbers, defends a fork, admits the reranker is partly a patch, and the failure is a real one that generalizes. It also seeds four follow-up threads the candidate is happy to go deep on.

**Common failure modes on this question:**

- Narrating the tech stack instead of the system ("we used LangChain, Pinecone, and GPT-4o"). Nobody asked what you imported.
- No numbers at all.
- Describing a system that was never used by anyone. If it was a prototype, say so up front and pivot to what you learned — pretending otherwise falls apart in the second follow-up.
- Talking for eleven minutes. Four minutes, then stop.

---

## 2. RAG

The questions here are component questions — what a thing is, when you would use it, what it costs. The *symptom* questions that build on them (something suddenly broke; retrieval looks fine and answers do not; prove that change helped; five documents; a million documents) are worked end to end in [`RAG_FAILURE_DIAGNOSIS.md`](RAG_FAILURE_DIAGNOSIS.md), because they are a different skill and a harder one.

The most heavily-asked cluster in the entire loop, at essentially every company. See `39_rag_retrieval_augmented_generation` for the mechanics of indexing, ANN search, and embedding models; this section is the interview layer.

### "Why RAG instead of fine-tuning?"

They are not alternatives to each other; they solve different problems and the strong answer says so immediately. RAG injects *knowledge* the model does not have, at query time, from a source you control. Fine-tuning changes *behavior* — format, tone, task structure, domain vocabulary, following an unusual instruction pattern. If your problem is "the model does not know our Q3 pricing," fine-tuning is the wrong tool: you would be encoding facts into weights that go stale, you cannot cite them, and you cannot delete them when a customer invokes their right to erasure.

Pick RAG when the knowledge changes, when you need attribution, when access control is per-user, or when the corpus is large relative to what you could reasonably train on. Pick fine-tuning when the *task* is unusual and prompting is not getting you there — structured extraction into a weird schema, a house style, a classification boundary that is hard to write down. Often you do both: fine-tune a small model for the task shape, RAG for the facts.

The honest addendum: in 2026 the bar for "just try a better prompt with a stronger model first" is high. A lot of fine-tuning projects are a solution to a prompt-engineering problem that was never seriously attempted.

### "How do you chunk documents, and why that size?"

The chunk is the unit of retrieval, so chunk size is a bet on how much context the answer needs. Too small and each chunk is semantically incomplete — you retrieve a paragraph that references "this policy" without saying which. Too large and you dilute the embedding: one vector is now the average of five topics, so it matches everything weakly and nothing strongly, and you burn context on irrelevant text.

The honest answer is that there is no universal number and anyone who gives you one without asking about the corpus is guessing. What I actually do: **start structural, not fixed-size.** Split on document structure first — headings, sections, function definitions, contract clauses — because that structure encodes where the semantic boundaries already are. Then only split further if a section exceeds the budget. In practice this lands most prose corpora somewhere in the 300–800 token range with 10–15% overlap, but the number is an outcome, not an input.

Two techniques worth naming because they signal you have done this:

- **Contextual prepending** — before embedding, prepend a short generated summary of where this chunk sits (document title, section, a sentence of surrounding context). This fixes the "this policy" problem cheaply and measurably.
- **Decoupling the retrieval unit from the generation unit.** Embed and search over small precise units, but pass the *parent* section to the model. You get the precision of small chunks and the completeness of large ones. This is often the single highest-leverage retrieval change.

And the thing that actually decides it: run three chunking configs against a labeled retrieval eval set and look at recall@k. It takes an afternoon and it beats any heuristic.

### "How do you pick an embedding model?"

Constraints first, leaderboard second. Dimensionality drives index memory and cost. Max sequence length has to exceed your chunk size or you are silently truncating. Multilingual matters if your corpus is. Whether you can self-host matters if the data cannot leave your network. Cost matters at re-index time, not query time — re-embedding 10M chunks after a model swap is the expensive event.

Then: **do not trust MTEB rankings as a proxy for your corpus.** Public benchmarks are heavily overfit at this point and the top of the leaderboard is a cluster of models within noise of each other. Build a small domain eval — 100–200 real queries with labeled relevant documents, which you can bootstrap by having a strong model generate questions from your own chunks and then human-filtering — and compare candidates on recall@10 on *your* data. Models frequently reorder between MTEB and a specific domain, especially technical/jargon-heavy corpora. **[verify before quoting: specific model names and rankings move fast; name the method, not the model, unless you checked this week.]**

The migration cost is the thing people underestimate: switching embedding models means re-embedding everything and you cannot mix vector spaces, so plan for a dual-index cutover.

### "Which vector store, and why?"

Wrong framing, and saying so politely scores points. The real question is whether you need a dedicated vector database at all. Under a few hundred thousand vectors, `pgvector` inside the Postgres you already run is usually correct — you get transactions, joins to your metadata, one backup story, one access-control story, one on-call rotation. A lot of teams add a separate vector database and inherit a second consistency problem for a query pattern that Postgres handled fine.

Reach for a dedicated store when you have real scale (tens of millions of vectors and up), need high-QPS low-latency ANN, want native hybrid search and filtering that does not fall over, or need features like multi-tenancy isolation built in. At that point the differentiators are: does filtered search stay accurate (pre-filter vs post-filter matters enormously — post-filtering a top-k can return almost nothing after a restrictive filter), what is the update/delete story, and what happens operationally during reindex.

The answer that lands: "We used pgvector until we hit 4M chunks and filtered-search latency became the problem, then moved. I would not have started with a dedicated store."

### "What is hybrid search and why do you need it?"

Dense embedding retrieval matches meaning; it is bad at matching *strings*. Rare tokens get averaged into the pooled vector and disappear. So error codes, SKUs, function names, ticket IDs, version numbers, and rare proper nouns are exactly where dense retrieval quietly fails — `ERR_SSL_VERSION_OR_CIPHER_MISMATCH` and `RTX-4090` versus `RTX-4070` are the canonical examples. BM25 nails those and is bad at paraphrase. Running both and fusing is not a nicety; on any corpus with identifiers in it, it is the difference between working and not.

Fusion, two options:

**Reciprocal rank fusion** — for each document, sum $\frac{1}{k + \text{rank}}$ across retrievers:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

with $k = 60$ by convention. It is score-scale agnostic, which matters because BM25 scores and cosine similarities are not comparable, and it needs zero labeled data. The cost is that it throws away score *magnitude* — a document that barely won rank 1 counts the same as one that dominated.

**Convex combination** — $\alpha \cdot s_{\text{dense}} + (1-\alpha) \cdot s_{\text{sparse}}$ after normalizing each. Better if you have labeled data to tune $\alpha$; roughly a few dozen labeled pairs is enough to beat RRF. Reported practice puts technical documentation nearer $\alpha \approx 0.3$ (favor lexical) and conversational or policy content nearer $\alpha \approx 0.7$. **[verify: these are directional, tune on your own data.]**

Start with RRF because it works with no tuning, move to weighted fusion once you have an eval set.

### "When do you add a reranker, and what does it cost?"

A reranker is a cross-encoder: it sees the query and the document *together* and scores relevance, rather than comparing two independently-computed vectors. That joint attention is why it is far more accurate — and why it cannot be precomputed, and therefore why it cannot run over your whole corpus.

Standard shape: retrieve broad and cheap (top 50–100 from hybrid), rerank precise and expensive down to the 5–10 you actually put in context. Keeping the rerank candidate set under about 50 is the usual latency constraint; it typically adds 100–400ms depending on model and batch. **[verify: reranker latency and pricing move; measure yours.]**

The honest framing: reranking usually buys a large accuracy jump, and part of that jump is compensating for retrieval that is not very good. If a reranker takes you from bad to good, the interesting question is why first-stage retrieval was bad — bad chunking, wrong embedding model, missing hybrid. Fix that too. But ship the reranker; it is one of the highest ROI components in RAG.

### "Retrieval returns irrelevant documents. What do you do?"

Diagnose before you fix, and say the diagnosis out loud because that is what is being tested.

First question: **is the right document even in the index?** If not, this is not a retrieval problem, it is an ingestion problem — parsing failure (PDFs with tables and multi-column layouts are the usual culprit), a doc that never got crawled, a chunk that got truncated. Check by grepping the raw store, not the vector index.

If it is in the index but not retrieved: is it a *lexical* miss (query has an identifier, you have no BM25) or a *semantic* miss (query phrasing is far from document phrasing)? Lexical → add hybrid. Semantic → query rewriting/expansion, or the embedding model is wrong for your domain, or the chunk is too big and got diluted.

If it is retrieved but ranked below your cutoff → reranker, or raise $k$ and let the reranker sort it out.

If it is in the context and the model still answers wrong → this is a generation problem, not retrieval, and you have been debugging the wrong half.

And the part most candidates miss: **the system should be allowed to retrieve nothing.** Set a relevance floor. If nothing clears it, say "I don't have information on that" rather than stuffing the top-5 nearest garbage into the prompt. Vector search always returns *something* — that is its most dangerous property. Forced-answer-from-irrelevant-context is a large fraction of RAG hallucinations.

### "How do you handle multi-hop questions?"

"Which of our enterprise customers who churned last quarter had an open P1 ticket?" cannot be answered by one similarity search — no single chunk contains it. Single-shot RAG structurally cannot do this and pretending otherwise is the failure.

Options, cheapest first:

- **Query decomposition**: model splits the question into sub-questions, retrieve for each, synthesize. Works well, roughly multiplies retrieval cost by the number of hops.
- **Iterative / agentic retrieval**: the model retrieves, reads, decides what it still needs, retrieves again, with a hard iteration cap. More powerful, much higher latency variance, and it can wander — you need the budget cap.
- **Graph or structured augmentation**: if your entities and relations are actually structured (customers, tickets, subscriptions), the honest answer is that this is a *query* problem, not a retrieval problem. Text-to-SQL over the real database beats any amount of chunk retrieval. A lot of "hard multi-hop RAG" is a database query wearing a costume.

The senior answer names the third option. Interviewers notice when you decline to use RAG for something RAG is bad at.

### "How do you evaluate retrieval separately from generation?"

You must, or you cannot debug. They are different failure modes with different fixes and an end-to-end score conflates them.

**Retrieval** is classical IR and you should use classical IR metrics on a labeled set of (query, relevant doc IDs): recall@k — did the right document make it into the context at all, which is the ceiling on everything downstream — plus MRR or nDCG for ranking quality, and precision@k because junk in context actively hurts generation. Recall@k is the one to lead with.

**Generation, given retrieved context**, is measured separately: faithfulness/groundedness (is every claim supported by the provided context), answer relevance (does it address the question), and completeness. Critically, you evaluate generation with *fixed* context so retrieval changes do not contaminate the measurement.

Building the labeled set is the actual work. Bootstrap it: have a strong model generate questions from each chunk, which gives you a free (question, gold chunk) pair; human-filter aggressively because maybe 60% are usable; supplement with real user queries from logs, which are messier and more valuable. A few hundred examples is enough to make decisions.

### "How do you keep the index fresh?"

Three regimes, and knowing which you are in is the answer:

- **Batch rebuild** — fine below a few hundred thousand documents and hourly-or-slower freshness needs. Simplest, most robust: rebuild into a new index, swap atomically, keep the old one for rollback.
- **Incremental upsert** — you need near-real-time. This is where the bugs live. Every document needs a stable ID, chunks need to be deterministically derived from it so you can delete-then-insert, and **deletes must actually propagate**. Soft deletes that never get filtered, or tombstones that the ANN index ignores until compaction, are how you end up serving retracted content.
- **CDC-driven** — change data capture off the source system into an indexing queue. Correct for high-churn sources, more infrastructure.

The operational point that separates people who have run this: **freshness is a monitored metric, not a pipeline.** Emit index lag (time since source change to index visibility), run a periodic reconciliation between source-of-truth IDs and index IDs, and alert on drift. Silent index staleness is invisible to every eval you have because your eval corpus is frozen.

### "How do you do per-user access control in RAG?"

The failure mode is data exfiltration across tenants and it is the question most likely to end an interview badly if you handwave it.

**Filter at retrieval, never at generation.** Do not retrieve broadly and ask the model to ignore documents the user should not see — the model is not an authorization boundary and prompt injection makes that trivially bypassable. Every chunk carries the ACL metadata of its source document, and the query is filtered by the caller's identity *inside* the search, as a pre-filter.

Pre-filter vs post-filter is the technical trap: post-filtering means you retrieve top-100 globally and then drop unauthorized ones, which is both a latency waste and a correctness bug (a user with narrow access gets 3 results instead of 10, or zero). Your vector store needs real pre-filtered ANN.

Three more things to name:

- **Permissions change; embeddings do not.** ACLs must be resolved at query time against current state, not baked in at index time. Group membership changes are the common leak.
- **Multi-tenant isolation**: for hard boundaries prefer separate namespaces/indexes per tenant over a metadata filter, because a filter is one bug away from a cross-tenant leak and a namespace is not.
- **Caches and logs inherit the problem.** A semantic cache keyed only on query text will serve tenant A's answer to tenant B. Key caches by (tenant, user-scope, query). And trace payloads containing retrieved chunks are now a data-residency artifact.

---

## 3. Agents and tool use

### "When does an agent actually beat a workflow?"

The strong answer starts by pushing back on the premise. Most things called agents should be workflows: a fixed, predetermined sequence of LLM calls with code in between. Workflows are debuggable, testable, cost-predictable, and latency-predictable. Agents — where the model decides its own next step in a loop — buy you flexibility and pay for it in every one of those properties.

Use an agent when the number of steps genuinely cannot be known ahead of time and the branching is too wide to enumerate: open-ended research, debugging, anything where the next action depends on what the last one returned in a way you cannot chart. Use a workflow whenever you *can* chart it. The prevailing guidance from people who build these — Anthropic's "building effective agents" being the most-cited version — is to find the simplest thing that works and only add agentic autonomy when the task demands it, because you pay for autonomy in reliability.

Practical test: sketch the flow chart. If you can draw it, build the flow chart. If drawing it requires "...and then it depends," you have an agent.

### "How do you design a tool schema?"

Tool definitions are prompt engineering, not API design, and treating them like an OpenAPI dump is the most common mistake. The model reads the name, the description, and the parameter descriptions and has to decide from those alone. So:

- **Name and describe from the model's decision point.** The description should answer "when should I use this instead of the others," not "what does this endpoint do." `search_orders` and `get_order_by_id` need descriptions that make the choice between them obvious.
- **Fewer, coarser tools beat many fine-grained ones.** Twenty tools with overlapping purposes produces selection errors. Wrapping three API calls into one purposeful tool is usually better than exposing all three.
- **Constrain the parameter space.** Enums over free strings. Required over optional. If the model can pass an arbitrary string, it will eventually pass a wrong one; if it must pass one of four enum values, it cannot.
- **Return errors the model can act on.** `{"error": "invalid_date_format", "message": "Use YYYY-MM-DD", "example": "2026-03-14"}` gets self-corrected. `500 Internal Server Error` gets retried identically forever.
- **Return the minimum useful payload.** Tool results land in context and stay there for the rest of the loop. A tool that returns a 40KB JSON blob poisons everything downstream and costs you on every subsequent turn.

The tell that someone has built agents: they mention that they iterated on tool *descriptions* the way you iterate on prompts, with evals.

### "How does the model actually choose a tool?"

Tools are serialized into the model's context as part of the prompt, the model emits a structured call, your code executes it, and you feed the result back as another message. There is no magic dispatcher — it is next-token prediction over a structured format, usually with constrained decoding so the arguments are schema-valid.

Two implications worth saying because they explain most bugs: (1) tool selection quality is a function of how the tools are *described*, so ambiguous descriptions cause selection errors that look like model stupidity; (2) schema validity does not imply semantic correctness — the model will happily emit a perfectly-formed call with a hallucinated ID. Validate arguments against reality, not just against the schema.

### "A tool call fails. What happens?"

Classify the failure, because the right response differs:

- **Transient** (timeout, 503, rate limit) → retry with exponential backoff *in your code*, not by returning the error to the model. The model has no idea about backoff and will just call again immediately.
- **Malformed call** (bad arguments, schema violation) → return a specific, actionable error to the model and let it retry. Cap this at 2–3 attempts, because a model that has misunderstood the tool will loop.
- **Legitimate empty/negative result** ("no orders found") → this is not an error, it is data. Return it plainly. Systems that treat empty results as errors cause the model to retry endlessly instead of telling the user there are no orders.
- **Permanent** (auth failure, tool is down) → do not let the model keep trying. Remove the tool from the loop, degrade gracefully, and tell the user what the system cannot currently do.

The overall principle: the agent loop needs a supervisor that is *code*, not the model. Retry policy, circuit breaking, and giving up are control-flow concerns and the model is a bad controller.

### "How do you terminate the loop and budget it?"

Multiple independent limits, because each catches a different pathology:

- **Max iterations** (typically 10–25 depending on task). Catches infinite loops.
- **Token/cost budget per session.** Catches the loop that is making progress but expensively. This is the one that saves you money — an agent that costs \$0.15 per run in eval and \$14 on one adversarial production input is a real thing that happens.
- **Wall-clock deadline.** Users leave.
- **No-progress detection.** The subtle one: an agent calling the same tool with the same arguments three times is stuck, even though it is under every other limit. Hash (tool, args) and break on repeats.
- **Explicit completion.** The model signals done, or a validator confirms the goal state.

And crucially: **what happens when you hit a limit is a product decision, not an error.** Returning a stack trace is wrong. Return partial results with an honest statement of what was not completed, or hand off to a human. "I found 3 of the records you asked about and could not locate the other 2" is a good outcome; a timeout is not.

### "When do you require human approval?"

Gate on **reversibility and blast radius**, not on model confidence. Model confidence is not calibrated and using it as an authorization signal is a mistake worth naming out loud.

Reads: never gate. Reversible writes (draft, tag, comment): usually not. Irreversible or externally-visible actions (send the email, issue the refund, delete the record, execute the trade, merge the PR): gate — or make them reversible so you do not have to. Turning "send email" into "create draft" removes the approval requirement entirely and is almost always the better engineering answer.

Additional dimension worth raising: the OWASP-adjacent "lethal trifecta" framing — access to private data, exposure to untrusted content, and the ability to communicate externally. An agent with all three can be turned into an exfiltration channel by a single injected instruction. Meta's "Agents Rule of Two" formulation is that an agent acting without human approval should have at most two of the three; all three requires a human in the loop. That is a genuinely useful design heuristic and citing it signals you follow the security side.

Design the approval UX for the reviewer, not the log: show the diff, show the reasoning, make approve/reject one click. Approval fatigue means people rubber-stamp, and a rubber-stamped gate is worse than no gate because it manufactures false accountability.

### "Multi-agent or single agent?"

Default to single agent with good tools. Multi-agent systems are frequently a way of making an architecture diagram look sophisticated while multiplying the failure surface: every handoff is a lossy context transfer, latency compounds, cost multiplies, and debugging goes from reading one trace to reconstructing a conversation between four things that each partially misunderstood the last.

Multi-agent earns its cost in three cases: (1) genuine parallelism over independent subtasks — a research task fanning out across ten sources is real speedup; (2) hard context isolation, where one sub-agent should not see another's data for security reasons; (3) genuinely distinct tool sets or permission scopes where one agent with forty tools would fail at selection.

If you do build it, the orchestrator-worker pattern is the one that works: one planner decomposes and delegates, workers are stateless and return structured results, the planner synthesizes. Free-form agent-to-agent chat is where systems go to die.

### "What is MCP and why does it matter?"

The Model Context Protocol standardizes how applications expose tools, resources, and prompts to models — an open client-server protocol so that an integration written once works across any MCP-supporting client instead of being reimplemented per framework. It solved a real M×N problem: before it, every tool integration was bespoke per agent framework.

State as of 2026: broad adoption across major clients and IDEs, a registry with on the order of two thousand servers, two transports (stdio for local subprocesses, streamable HTTP for remote), and an auth story that has converged on OAuth 2.1 with resource indicators. **[verify: numbers and auth spec details move quarterly.]**

The parts to be honest about, because this is where the follow-up goes: enterprise gaps remain in standardized audit trails, multi-tenancy patterns, rate limiting and cost attribution, and portable configuration across clients. And the security surface is real — an MCP server is code you are running that can inject text directly into the model's context, so tool descriptions from a third-party server are an injection vector (a "rug pull," where a server changes its tool description after approval, is the named attack). Treat third-party MCP servers like npm dependencies with prompt-level privileges: pin them, review them, and do not connect an untrusted one to an agent that has the other two legs of the trifecta.

---

## 4. Prompting and context engineering

### "How do you design a system prompt?"

Structure over prose. A system prompt that reads like an essay performs worse than one that reads like a spec sheet — role, then capabilities, then hard constraints, then output format, then examples of edge cases. Put the most important constraints where the model attends best, which is early, and repeat genuinely critical ones at the end.

Things that actually move the needle:

- **Positive instructions beat negative ones.** "Answer only from the provided context; if it is not there, say you don't know" works. "Don't hallucinate" does nothing.
- **Specify what to do at the boundaries.** Most production prompt failures are unhandled edge cases: ambiguous question, empty retrieval, out-of-scope request, user asking the system to break its own rules. Each of those needs an explicit branch.
- **Keep it stable at the front.** Prompt caching keys on prefixes, so anything dynamic (user name, date, retrieved docs) goes at the *end* or you invalidate the cache on every request. This is a cost decision hiding in a formatting decision.
- **Do not put policy in the prompt that belongs in code.** "Never reveal prices above \$X" is not enforced by a prompt; it is enforced by a filter.

### "Few-shot or zero-shot?"

Zero-shot first, because instruction-following in current frontier models is strong enough that examples are often redundant cost. Reach for few-shot when the task has an output *shape* or *judgment boundary* that is hard to describe but easy to demonstrate — tone, an idiosyncratic label taxonomy, a formatting convention with fiddly rules, edge cases where you want a specific call.

The failure mode nobody mentions: **examples leak.** The model will copy surface features of your examples — entity names, structure, length, even the domain — into unrelated outputs. If all three examples answer in two sentences, you have accidentally capped answer length. Vary your examples deliberately along the dimensions you do *not* want copied.

Also: with strong models, a small number of well-chosen examples (2–5) usually captures nearly all the benefit; going to 20 mostly buys cost and context pollution. And dynamic few-shot — retrieving the $k$ most similar labeled examples for each query — outperforms a fixed set on classification-type tasks and is cheap to build if you already have a vector store.

### "How do you get reliable structured output?"

Layered, because each layer catches what the previous one misses:

1. **Constrained decoding / native structured outputs.** The provider masks the token distribution so only schema-valid continuations are possible. This gives you schema validity as a guarantee, not a hope. Contrary to the common worry, the benchmark evidence indicates constrained decoding does not degrade task quality and can slightly improve it — up to a few points, even on reasoning tasks like GSM8K. Use it.
2. **Parse and validate against a real type** (Pydantic, Zod). Schema-valid is not the same as semantically valid.
3. **Business-rule validation.** The most important layer and the one people skip. The model will return a perfectly-shaped object with a hallucinated `order_id`, a date in the future, a `total` that does not equal the sum of `line_items`. JSON mode is not a contract about *truth*. Check referential integrity against your actual database.
4. **Retry with the validation error in the message.** Bounded, 2–3 attempts, and the error must be specific.
5. **A defined failure path.** What renders when all retries fail. Not an exception in the user's face.

The compressed version to say out loud: "Constrained decoding gets me valid JSON. It does not get me *correct* JSON — the fields can be confidently wrong — so the validation that matters is against my own data, not against the schema."

### "How do you manage the context window?"

Start by rejecting the premise that a big context window solves this. The Chroma "context rot" work is the citation to have: performance degrades measurably as input length grows, across retrieval, long-conversation QA, and even trivial replication tasks — with gaps exceeding tens of points between focused (~300 token) and full (~113k token) prompts on the same underlying question. Degradation is not uniform: it is worse when the needle is semantically dissimilar to the query, and worse when distractors are present. **[verify: model-specific results change with each release; the qualitative finding has held.]**

So: **context is a scarce resource you curate, not a bucket you fill.** Practically —

- Put the most important material early. Attention to opening content is reliably better than to the middle.
- Aggressively remove distractors. Topically-adjacent-but-wrong content does more damage than unrelated content, because it is what the model latches onto.
- For conversations: sliding window over recent turns plus a running summary of older ones, plus pinned facts extracted into a structured scratchpad. Summarization is lossy in ways that bite — entity names and numbers get dropped — so extract those into structured state rather than trusting the summary.
- For agents: compact tool results before they accumulate. Truncate, summarize, or store-and-reference.
- Measure it. Track token count per request as a distribution, not an average; the p99 is where you blow the window and where the quality collapse lives.

### "How does prompt caching work and when does it pay?"

The provider caches the KV state for a prefix of your prompt so subsequent requests sharing that prefix skip recomputation. You get a large discount on cached input tokens and a meaningful latency reduction on time-to-first-token.

As of mid-2026, all three major providers offer roughly 90% off cached input tokens. Minimum cacheable prefix lengths sit around 1,024 tokens for OpenAI, 512–4,096 for Anthropic depending on model, and 2,048–4,096 for Gemini. Anthropic charges a cache-write premium (1.25× input for the 5-minute TTL, 2× for the 1-hour); OpenAI and Gemini implicit caching do not charge for writes. TTLs are on the order of 5 minutes to an hour by default with longer options. **[verify before quoting — this is the fastest-moving table in the chapter; check provider docs the week of your interview.]**

The engineering consequence is the part that matters in an interview: **caching is a prompt-ordering constraint.** Static content — system prompt, tool definitions, few-shot examples, long shared documents — goes at the front, in a stable order. Anything varying per request goes at the end. A single dynamic token near the front (a timestamp, a user ID) invalidates the entire cache and you will see it as an unexplained cost increase, not as an error. The other big win is agent loops, where the entire growing conversation is a stable prefix across turns.

### "How do you version and test prompts?"

Prompts are code that happens to be in a string, and the strong answer treats them that way: in version control, not in a database that someone edits in a web UI without review. Every prompt has an ID and a version, and every logged request records which version produced it — otherwise you cannot attribute a quality regression to a change.

Testing means an eval set that runs in CI on prompt change, with the pass criteria defined per-prompt (exact match where possible, structural checks where not, a judge where nothing else works). The realistic bar is not "the eval must improve"; it is "no regression on the cases we already fixed." Maintain a growing regression suite of every bug you have ever shipped, as a test case.

Rollout: prompts get the same treatment as code — canary a percentage of traffic, watch online metrics, roll back on regression. And the thing that surprises people: **prompt changes interact.** Fixing behavior A by adding a line to the system prompt frequently breaks behavior B, which is exactly why you need the regression suite rather than testing the case you just fixed.

### "Why is prompt injection not solved?"

Because it is not a bug, it is the architecture. The model receives system instructions, user input, and retrieved or fetched content as one undifferentiated token stream. There is no privileged channel — nothing in the representation says "this part is trusted policy and this part is data." Every proposed fix is a heuristic on top of a system that fundamentally cannot distinguish the two, and heuristics lose to an adversary who gets unlimited attempts.

This is not a hot take; it is the consensus position. OWASP's 2026 material maps prompt injection into a majority of its Top 10 for agentic applications and the reporting has shifted from theoretical to cataloguing actual CVEs and breaches. **Indirect** injection is the serious version: the attacker does not talk to your system at all, they put instructions in a document, a web page, an email, a code comment, or a tool description that your agent will later read.

What you actually do, since you cannot solve it:

- **Architectural containment over detection.** Assume injection succeeds and limit what a successful one can do. Least privilege on tools, no ambient credentials, human gates on irreversible actions, and the trifecta rule from §3.
- **Treat all retrieved content as untrusted**, including content from your own corpus — anyone who can write a support ticket can write into your index.
- **Enforce policy in code, outside the model.** The model asking for a refund is not authorization to issue one; the refund service checks its own rules.
- **Detection layers** (classifiers, spotlighting/delimiting, output filters) as defense in depth, understood as raising cost rather than closing the hole.

The candidate who says "we use a prompt injection detection model, so we're covered" fails this question. The one who says "we assume it gets through and we scoped the blast radius" passes.

---

## 5. Evaluation

The cluster most candidates fumble, and increasingly the one that separates senior from mid. Say early that eval is the actual engineering work in GenAI — the model is a dependency, the eval is your product's definition of correct.

### "How do you evaluate a system with no single right answer?"

Stop trying to score the answer and start decomposing it into properties that *are* checkable. "Is this summary good?" is unanswerable. "Does it contain all five required facts, is every claim supported by the source, is it under 200 words, does it avoid naming individuals" is four assertions you can measure, three of them without an LLM.

The hierarchy I would give, cheapest and most reliable first:

1. **Deterministic checks.** Valid JSON, required fields present, citation IDs resolve to real documents, no PII pattern in output, length bounds, latency. Free, fast, zero ambiguity. Surprising amounts of quality live here.
2. **Reference-based** where you can get references — exact/fuzzy match on extraction tasks, retrieval metrics against labeled documents.
3. **LLM-as-judge** for the genuinely subjective residue, calibrated against human labels (see below).
4. **Human review** on a sample, always, forever. It is the only thing that catches what your rubric does not encode.
5. **Online behavioral signals** — the ground truth you actually care about.

Also useful to say: pairwise comparison is more reliable than absolute scoring. Judges are bad at "rate this 1–10" and much better at "which of these two is better," and you can convert wins into a ranking.

### "Talk to me about LLM-as-a-judge and its biases."

Judges are useful and unreliable in specific, documented, correctable ways. Naming them concretely is what scores here.

- **Position bias.** In pairwise comparison, judges favor a position rather than a response. The systematic study of this reports position consistency ranging roughly 0.57–0.82 on MTBench pairwise settings — meaning that on the order of 18–43% of judgments flip when you swap the order — and worse for list-wise comparisons. It is systematic, not random: repetition stability is high (≥0.93 for capable models), so re-running does not average it out. It is worst exactly when it matters most, when the two answers are close in quality. Mitigation is to run both orders and count only consistent verdicts, treating flips as ties.
- **Verbosity bias.** Longer answers score higher independent of quality. Control by normalizing length or scoring against an explicit rubric with length as its own axis.
- **Self-preference.** Judges rate their own family's outputs higher. Use a different model family for judging than for generation where you can.
- **Agreeableness / sycophancy.** Judges drift toward approving, especially with leading prompts. Force a specific rubric and require the judge to cite the failing span.

The non-negotiable practice: **calibrate the judge against human labels before you trust it.** Label a few hundred examples yourself, measure agreement (Cohen's kappa, not raw accuracy — raw accuracy is inflated when classes are imbalanced), and only use the judge in the regime where it agrees. A judge at 85% agreement on faithfulness is a fine CI gate and is not a number you report to leadership as your accuracy.

### "How do you build an eval set?"

Sources, in increasing order of value: synthetic generation from your corpus (fast, gets you to day one, systematically misses how real users phrase things), real production logs (the good stuff — sample stratified, not top-of-head), and captured failures (every bug becomes a permanent test case).

Composition matters more than size. A useful set is deliberately stratified: the common happy path, the known-hard cases, the edge cases, adversarial inputs, and — the one people forget — **cases where the correct answer is refusal or "I don't know."** If your eval set has no unanswerable questions, you are actively selecting for a system that always answers, which is how you ship a confident hallucinator.

Size: a few hundred well-chosen examples beats ten thousand scraped ones. You need enough that a change is distinguishable from noise, which for a binary metric around 80% means roughly a couple hundred examples to detect a 5-point move. Bigger sets mostly buy slower CI.

Hygiene: hold out a slice you never look at, or you will overfit your prompts to the eval set — this happens fast and silently. Version the eval set alongside the code. And when the set stops finding bugs, that is not success, that is the set going stale relative to how usage has shifted.

### "Offline versus online eval."

Offline is a proxy you control: same inputs every time, runs in CI, gates deploys, tells you about regressions before users see them. It cannot tell you whether users are better off, and it drifts from reality as usage changes.

Online is the truth and it is slow, noisy, and partially unactionable. Implicit signals are the valuable ones: did the user accept the suggestion, copy the output, rephrase the question immediately (a strong negative), escalate to a human, come back tomorrow. Explicit thumbs-up/down has terrible coverage — low single-digit percent response rates — and is heavily biased toward angry users, so treat it as an anomaly detector, not a metric.

You need both and you need to know their relationship. The most useful artifact is the correlation between your offline metric and the online outcome you care about, computed once: if offline faithfulness moves and downstream escalation rate does not, your offline metric is measuring something nobody cares about.

Also: A/B tests on generative systems need more traffic and longer runs than people expect. Output variance is high, effect sizes are often small, and there is a novelty effect. Powering the test properly is part of the answer.

### "How do you regression-test prompts?"

Eval set in CI, triggered on any change to a prompt, model version, retrieval config, or tool definition. Fail the build on regression against the pinned suite. Non-determinism is handled by pinning temperature to 0 where the task allows, running $n$ samples and using a threshold where it does not, and — importantly — accepting that your gate is statistical: define a tolerance band so normal variance does not produce a flaky red build that everyone learns to ignore.

The organizational half of the answer: the regression suite has to be cheap enough to run on every PR or it will not be run. If a full eval costs \$40 and 20 minutes, split it — a fast smoke set on every commit, the full set nightly and before release.

### "How do you catch quality drift?"

Nothing about your system needs to change for its quality to change. Provider model updates, corpus drift, and user behavior drift all move quality under a frozen codebase.

What I monitor:

- **Input distribution drift.** Embedding-space monitoring of incoming queries against a reference window; a new cluster means new usage you have never evaluated.
- **Output property drift.** Refusal rate, answer length distribution, citation counts, tool-call frequency, JSON parse failure rate. These are cheap, high-frequency, and they move *before* anyone complains.
- **Behavioral drift.** Escalation rate, retry rate, immediate-rephrase rate, session abandonment.
- **A canary eval on a schedule**, not just on deploy — a small fixed set run hourly against production config. This is what catches a silent provider-side model change.
- **Sampled human review** on a fixed cadence, weighted toward outliers.

The failure this prevents: quality drops 8%, no alert fires because nothing errored, and you find out from a customer escalation six weeks later.

### "How do you measure groundedness and hallucination?"

Groundedness (or faithfulness) is a claim-level question, and the answer should say that: decompose the output into atomic claims, and for each, ask whether the retrieved context entails it. Score is the fraction supported. Doing it at the whole-answer level is much less useful because one unsupported sentence in a good answer gets averaged away — and that one sentence is the incident.

Implementation options, honestly ranked: an NLI model per (claim, evidence) pair is cheap and fast and works well; an LLM judge with the requirement that it quote the supporting span is more flexible and more expensive; citation-verification is the cheapest useful proxy — require inline citations and mechanically check that each cited chunk actually exists and contains overlapping content.

Two things to be honest about, which distinguish a real answer:

- **Groundedness is not correctness.** An answer perfectly grounded in a wrong document is faithfully wrong. You need source quality separately.
- **The most dangerous hallucinations are omissions and subtle qualifier changes**, not fabricated facts. Dropping "not," changing "may" to "will," or omitting the exception clause in a policy — these pass most groundedness checks because the words are all in the source. Sampled human review is the only reliable catch.

And the design lever that beats measurement: make the system *able* to abstain. A relevance floor on retrieval and an explicit "insufficient information" path removes a whole class of hallucination rather than measuring it.

---

## 6. LLMOps, cost, and latency

### "What do you monitor in an LLM application?"

Four layers, and saying them as layers is the answer:

- **Infrastructure**: latency (p50/p95/p99, and separately time-to-first-token vs total, because streaming makes TTFT the number users feel), error rate by type, throughput, provider rate-limit rejections.
- **Model/usage**: input and output tokens per request as distributions, cost per request and per user, cache hit rate, model version actually served, retry counts.
- **Quality proxies**: refusal rate, JSON parse failure rate, groundedness on a sample, tool-call success rate, empty-retrieval rate.
- **Business/behavioral**: task completion, escalation, retention, whatever the product's actual outcome is.

The rule that ties it together: **every alertable metric needs a defined response.** A dashboard nobody acts on is decoration. Cost per user has a budget; p95 has an SLO; refusal rate has a normal band.

### "How do you trace a request?"

One trace ID from the user request through every span — retrieval, rerank, each model call, each tool call, each retry — with the full payloads attached (prompt, retrieved chunk IDs, raw completion, token counts, cost, latency, model version, prompt version). Without payloads a trace tells you *that* something was slow, not *why* it was wrong, and wrongness is the common failure in this domain.

The ecosystem has converged on OpenTelemetry's GenAI semantic conventions as the wire format, which matters because it means you are not locked into one vendor's SDK — you can emit standard spans and point them at whatever backend. **[verify: the GenAI conventions were still stabilizing; check the current status of the spec.]**

Two operational caveats worth raising unprompted: payloads are expensive to store at volume, so sample (all errors, all slow requests, a percentage of the rest); and payloads contain user data, so your trace store is now in scope for your privacy and residency requirements.

### "How do you track and control token cost?"

Attribute first, optimize second. Log token counts per request tagged with user, tenant, feature, and prompt version, and compute cost from a version-pinned price table rather than from a provider dashboard, so you can do the arithmetic yourself and catch a pricing change. Cost per *successful task* is a better metric than cost per call, because retries and agent loops mean these diverge a lot.

Where the money actually is, in my experience order:

1. **Retries and agent loops.** Cost distribution is long-tailed. The p99 request may cost 50× the median, and it is almost never the median that blows the budget. Cap loops.
2. **Context bloat.** Stuffing 20 chunks when 5 would do, or never compacting conversation history. Input tokens dominate most RAG workloads.
3. **Model overkill.** Frontier model on a routing/classification step that a small model does at parity.
4. **Cache misses from prompt instability** — see §4; a single moving token near the front of the prompt.

Controls: budget caps per tenant enforced in code, alerting on cost-per-task moving, and a kill switch. And the organizational one: put cost in the same dashboard as quality, or you will optimize one into the ground.

### "How do you reduce latency?"

Measure the breakdown first — retrieval, rerank, TTFT, generation, post-processing — because the intuition is usually wrong. In RAG systems the surprise is often that reranking or a slow tool dominates, not the LLM.

Then, roughly in order of leverage:

- **Stream.** Does not reduce total latency at all; reduces *perceived* latency enormously because TTFT becomes the number users experience. Almost always the first thing to do.
- **Cut output tokens.** Generation is sequential and output tokens are the dominant term in end-to-end time. Ask for less. A prompt that produces a 150-token answer instead of 600 is nearly a 4× win on the generation phase.
- **Route by difficulty.** Small/fast model for the easy majority, escalate to the big model on a confidence or complexity signal. Real systems get large wins here because the query distribution is heavily skewed toward easy.
- **Cache.** Exact-match cache for repeated queries; semantic cache (embed the query, serve on high similarity) for near-duplicates — with the warning that semantic caching serves *wrong* answers when the threshold is loose, and must be keyed by tenant and user scope (§2).
- **Parallelize.** Independent retrievals, independent tool calls, speculative prefetch of the likely next step.
- **Prompt caching** for TTFT on long stable prefixes.
- **Move work off the critical path.** Precompute, do it async, do it at index time instead of query time.

### "How do you handle rate limits and quotas?"

Assume you will hit them; they are a normal operating condition, not an exception. Client-side token-bucket limiting so you shape your own traffic rather than discovering the limit as 429s. Exponential backoff with jitter — without jitter your retries synchronize and you self-DDoS on recovery. Respect `Retry-After` when provided.

Beyond the basics: a request queue with priority, so interactive user traffic preempts batch jobs; per-tenant quotas so one customer's bulk import does not consume the shared limit; and provider-level headroom — spread across multiple keys, deployments, or regions if the volume justifies it. Batch APIs for anything not user-facing, since they are substantially cheaper and sit outside the interactive rate limit. **[verify: batch discounts and limits by provider.]**

### "How do you handle model versions and upgrades?"

**Pin explicitly.** Never point production at a floating alias. A floating alias means the provider can change your product's behavior without a deploy, and you will find out from an eval failure at best or a customer at worst.

Upgrade as a controlled process: run the new version against the full eval suite offline, expect some regressions even when the model is better overall (newer models often have different refusal behavior, different verbosity, different formatting instincts), fix prompts against the new version, then canary a small traffic percentage with online metrics watched, then ramp. Keep the old version routable for rollback until you are confident.

The thing to say that shows experience: **prompts are coupled to model versions.** A prompt tuned against one model is not guaranteed to transfer, and "we upgraded the model and quality dropped" is usually "our prompt was compensating for the old model's quirks." Budget prompt work into every model migration; it is not a config change.

Also track deprecation dates. Providers retire models on a schedule and being forced into an unplanned migration under a deadline is a bad place to be.

### "The provider is down. What happens?"

Have a documented degradation ladder rather than a single answer, because the right response depends on how degraded:

1. **Retry with backoff** — most provider blips are seconds.
2. **Fail over to a second provider** on the same task, with a prompt already validated for that model. This is the reason to keep prompts model-portable and to run your eval suite against your fallback periodically — an untested fallback is not a fallback.
3. **Serve from cache** where a slightly stale answer is acceptable.
4. **Degrade the feature**: return retrieval results without generation ("here are the 5 most relevant documents"), which is often genuinely useful.
5. **Queue for later** if the task is async.
6. **Fail honestly.** A clear "this feature is temporarily unavailable" beats a hung spinner or a garbage answer.

Circuit breakers so you stop hammering a dead provider, and health checks that test an actual completion rather than a status page. And say this: **multi-provider redundancy has a real ongoing cost** — two sets of prompts, two eval runs, two integrations to maintain — so it is justified by the criticality of the feature, not adopted by default.

---

## 7. Deployment and architecture

### "How would you serve this?"

Most GenAI applications are ordinary stateless web services whose heavy lifting happens over the network at a provider, and saying that plainly is correct. The interesting parts are the ones that differ from normal web services:

- **Requests are long and streaming**, so connection handling, timeouts, and load balancer configuration matter in ways they do not for a 50ms JSON API. Long-lived SSE connections break naive autoscaling.
- **Concurrency is I/O-bound**, so async is the right model and worker-per-request thread pools waste enormous capacity.
- **State lives somewhere.** Conversation history, agent scratchpads. Keep the app tier stateless and put session state in Redis or a database, or you cannot scale horizontally or survive a deploy mid-conversation.
- **Idempotency.** LLM calls cost money; a retried request that regenerates is a double charge and possibly a double side effect. Idempotency keys on anything with a side effect.
- **Backpressure**, because your capacity is bounded by provider rate limits, not by your own CPU.

### "Self-hosted or API?"

API by default, and be specific about what flips it. APIs give you frontier capability with zero ops, elastic scale, and no GPU capacity risk. You pay per token, you accept the provider's data terms, you accept their availability and their deprecation schedule, and you cannot deeply customize.

Self-hosting wins on four specific triggers: (1) **data cannot leave your environment** — regulated or air-gapped, and this is the most common genuine reason; (2) **volume economics** — at sustained high throughput on a task a small open model handles, per-token API pricing eventually loses to amortized GPU, but the crossover is much higher than people assume once you include engineering and idle capacity; (3) **customization** — you need weight-level control, unusual fine-tuning, or logits access; (4) **latency floors** you cannot hit over a network hop.

The costs people leave out of the comparison and that an interviewer is waiting for: GPU idle time (you pay for the off-peak hours), the engineer(s) who now own an inference stack, the capability gap versus frontier models, and the fact that you are now responsible for your own safety layer that the API was providing for free.

### "How do you size GPUs for inference?"

Two components, and knowing that there are two is most of the answer.

**Weights**: parameters × bytes per parameter, times an overhead factor of roughly 1.15 for activations, framework buffers, and CUDA context. Bytes per parameter: 4 (FP32), 2 (FP16/BF16), ~1 (FP8/INT8), ~0.5 (4-bit). So a 70B model at BF16 is ~140GB before overhead and does not fit on one 80GB card; at 4-bit it is ~35GB and does.

**KV cache**, which is the part people forget and which is what actually limits your concurrency:

$$\text{bytes per token} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes per element}$$

The leading 2 is keys and values. Multiply by context length for one sequence, then by concurrent sequences. Grouped-query attention is why modern models are tractable here — $n_{\text{kv heads}}$ is much smaller than $n_{\text{heads}}$.

The framing that lands: **once the weights fit, the remaining VRAM is your context and concurrency budget.** A worked version — a 4-bit ~35B model on a 96GB card is ~20GB of weights with overhead, leaving ~76GB; at roughly 0.26MB per token for a GQA model of that size that is on the order of 290K tokens of aggregate cache, or double that with FP8 KV quantization. **[verify: per-token figures are architecture-specific; compute from the actual config.]**

Then mention the serving stack does the hard part: continuous batching, paged attention for cache fragmentation, prefix caching. And that throughput and latency trade against each other via batch size — bigger batches raise tokens/sec/GPU and raise per-request latency.

### "What about data residency and privacy?"

Map the data flow before answering, and say you would: what data goes into the prompt, where does the provider process it, where do logs and traces land, what is retained and for how long, and who can read it.

Concrete levers: regional endpoints and provider commitments on processing location; zero-retention or no-training agreements (usually available on enterprise tiers, and worth checking rather than assuming); PII redaction *before* the prompt is constructed, not after; and the frequently-missed one, **your own observability stack is a copy of all of it** — traces with full payloads are the largest uncontrolled store of customer data in most GenAI systems.

Then: retention and deletion. If a user invokes deletion rights you need to remove their data from the source store, the vector index, the caches, and the trace store. Vector indexes with lazy deletes and trace backends with immutable retention both make this harder than expected, and that is worth designing for up front.

### "When and how would you fine-tune?"

When prompting has genuinely plateaued and the gap is *behavioral* rather than *informational* — output format, domain style, a task the model consistently misunderstands, or you want a small cheap model to match a big one on one narrow task (distillation, which is the most economically compelling case in practice).

Method by budget: prompt optimization first, then few-shot, then LoRA/PEFT (which is what almost everyone should actually do — small adapters, cheap, fast to iterate, easy to roll back), then full fine-tuning only with strong justification, then preference tuning (DPO and relatives) when the objective is subjective quality that you can express as pairwise preferences but not as a rubric. The underlying optimization math is in `13_interview_qa` and `64_integrated_ai_ml_interview_synthesis`; here the question is about the decision, not the gradients.

The unglamorous truth to say out loud: **data quality dominates method.** A few hundred to a few thousand carefully curated, consistent examples beat a hundred thousand scraped ones, and the most common failure is inconsistent labeling — if two of your examples handle the same situation differently, you are training in ambiguity. Also: hold out an eval set before you start, check for catastrophic forgetting on general capability, and have the base model routable for rollback. And plan for the fact that the base model will be deprecated and you will do this again.

### "What guardrails do you put in?"

Both directions, and named as such. **Input**: injection detection, off-topic/out-of-scope classification, PII detection, abuse and rate limiting. **Output**: PII leakage checks, toxicity, policy compliance, groundedness gate, schema validation.

Design principles that matter more than the specific tools: run them in parallel with generation where possible so they do not stack latency; make failure modes explicit and logged rather than silent; and decide fail-open versus fail-closed per guardrail deliberately — a PII filter fails closed, a topical classifier probably fails open, and getting that backwards produces either leaks or an unusable product.

The honest caveat: guardrails are classifiers with false positives and false negatives, and the false positives are a product problem. An over-aggressive filter that refuses legitimate requests destroys trust faster than an occasional bad output. Measure both error rates and tune the threshold as a product decision, not a safety absolute.

---

## 8. Safety and failure

### "How do you defend against prompt injection?"

Covered in §4 — the short interview version: you do not solve it, you contain it. Assume injection succeeds; scope what a successful one can reach. Least-privilege tools, no ambient credentials, policy enforced in code outside the model, human approval on irreversible actions, all retrieved content treated as untrusted, and detection layers as defense-in-depth rather than as the answer.

### "How do you handle jailbreaks?"

Distinguish them from injection first, because conflating the two is a common tell: a jailbreak is the *user* trying to get the model to violate its own policy; injection is a *third party* getting instructions into the context. Different threat models, different defenses, different people to be angry at.

Practically: system-prompt hardening is the weakest layer and cannot be your only one, since it is a natural-language instruction competing with an adversarial natural-language instruction. Output filtering catches what generation lets through and is independent of how the model was persuaded. Enforcing capability limits in code means a successful jailbreak yields text rather than action, which is the whole ballgame. Then rate limiting and account-level detection for the users doing it systematically, and a red-team suite of known jailbreak patterns in your regression evals so a model or prompt change does not silently reopen one.

Proportionality is part of a good answer: a consumer-facing assistant with a broad audience needs far more here than an internal tool with authenticated employees, and spending equally on both is a misallocation.

### "How do you handle PII?"

Minimize, detect, control, delete — in that order, because the first one is the only one that fully works.

Minimization means not putting PII in the prompt when the task does not require it. Most tasks do not require the customer's actual name and address; they require the account's state. Redaction and tokenization before prompt construction, with reversible mapping if you need to reinsert real values into the final output.

Detection is a layered problem: regex for structured identifiers (card numbers, SSNs, emails) and NER for names, addresses, and free-text mentions. Run it on both input and output — the output path matters because the model can surface PII from retrieved context that the user was not authorized to see.

Then the parts people forget, which is where the interview points are: **logs, traces, caches, and eval sets all accumulate PII.** Your eval set built from production logs is a permanent PII store in your repo. Trace payloads are the same. Have retention policies on each, and have an actual tested procedure for subject deletion that covers the vector index and the trace backend, not just the primary database.

### "The model said something harmful in production. Walk me through what you do."

This is an incident-response question wearing a safety costume, and the structure is what is being graded. Do not lead with root cause; lead with containment.

**Contain first (minutes).** Stop the bleeding: kill switch on the feature, or roll back to the last known-good prompt/model, or force a conservative fallback. You do not need to understand it to stop it. If you have no kill switch, that is a finding for later.

**Assess blast radius.** How many users, over what window, which segments, is it still happening. Query logs by the pattern. This number drives everything else, including legal and comms.

**Preserve evidence.** Full traces of affected requests before any retention window expires or any rollback overwrites config.

**Notify.** Whoever your org says: support, legal, comms, the affected customer. Do this in parallel with the technical work, not after it. Under-communicating internally is the most common way an incident becomes a crisis.

**Then diagnose.** Was it the model (a version change, a distribution shift), the prompt (a recent edit, an unhandled edge case), retrieval (poisoned or wrong content — remember anyone who can write into your corpus can write into your context), the guardrail (a gap or a config change), or an adversarial user?

**Fix at the right layer, and add the test.** The fix is not "add a line to the prompt saying don't do that." The fix is at whichever layer failed, plus — non-negotiably — the exact case added to the regression eval so it can never silently recur.

**Blameless postmortem** with the systemic question: what made this possible, why did it take N hours to detect, what monitoring would have caught it in minutes.

The two things that distinguish a strong answer: **containment before diagnosis**, and treating detection latency as a separate finding from the bug itself. If a customer told you before your monitoring did, that is its own incident.

---

## 9. Judgement questions

These have no clean answer and that is the point. They are asked to senior candidates to see whether you can reason about a system you do not control, hold two conflicting considerations at once, and disagree with the premise when the premise is wrong. Answering these with confident certainty is worse than answering with structured uncertainty.

### "How would you know this project is not worth doing?"

The strongest version of this answer starts before the project: define the kill criteria up front, while everyone is still optimistic and nobody is invested. "If we cannot exceed 80% task accuracy on the eval set within six weeks, we stop." Written down, agreed, dated. Kill criteria set after you have sunk three months are negotiated, not applied.

The signals I would actually look for:

- **The ceiling is below the bar.** Not "the model is at 70% and needs to be at 90%," but: even with perfect retrieval and a human writing the prompt by hand, the task tops out below what the use case tolerates. Test this directly — hand-construct the ideal context for 20 cases and see what the model does. If it fails with perfect input, no engineering fixes it.
- **The economics do not close.** Cost per task exceeds the value of the task, with no plausible path via smaller models or caching. Do this arithmetic in week one.
- **The error cost is asymmetric and unbounded.** If a wrong answer costs far more than a right answer saves, and you cannot bound the error rate, the expected value is negative regardless of accuracy.
- **A deterministic solution exists.** A large fraction of "AI projects" are a search index, a rules engine, or a SQL query. Noticing this is valuable, not defeatist.
- **Nobody owns the outcome.** If you cannot find the person whose metric improves, there is no project.

And the honest closer: the harder skill is killing something that *works* but does not matter. Plenty of GenAI features hit their technical bar and get no usage.

### "The model is good enough but the product is failing. What now?"

Accept the framing and go look, because the diagnosis is not in the model layer and continuing to tune it is the trap the question is testing for.

Most likely causes, roughly in the order I would check them:

- **Trust.** Users cannot tell when to believe it, so they verify everything, so it saves them nothing. The fix is calibration and transparency — citations, confidence signaling, visible reasoning, and above all the ability to say "I don't know." A system that is right 85% of the time and never signals which 15% is often worse than useless, because verification cost exceeds generation savings.
- **Workflow fit.** It is a separate tool requiring a context switch. Being 90% as good inside the tool people already use beats being 100% as good somewhere else.
- **Latency.** It is technically fine and too slow for the moment of use.
- **Wrong unit of work.** It answers questions when the user wanted the task done, or vice versa.
- **Onboarding.** People do not know what to ask. Blank-box products fail this constantly; a few good starting prompts often move usage more than a model upgrade.
- **The problem was not a real problem.** The bar was "acceptable," not "valuable."

The method matters more than the list: watch five users, do not survey fifty. And segment the funnel — where exactly do people drop, first use or second? Never returning after a good first session is a different disease from never starting.

### "How do you decide between shipping and improving accuracy?"

Reframe it away from a single accuracy number, because that framing is the trap. The questions I would actually ask:

**What does the error cost, and who absorbs it?** An error that a user notices and corrects in two seconds is cheap. An error that silently propagates into a decision is not. This is the dominant term and it is not captured by an accuracy percentage.

**Can I ship in a lower-stakes posture?** This usually dissolves the dilemma. Ship to an internal team, ship as a draft the human edits, ship to 5% of traffic, ship with a confidence threshold where the system only answers when it is sure and defers otherwise. A 70% system with a good abstention policy can be genuinely valuable; a 90% system that always answers can be net negative.

**What is the marginal return on more work?** If the last three weeks bought two points, the next three will buy less. Improvement curves flatten, and recognizing the flattening is the skill.

**What do I learn only by shipping?** Real usage distribution, real failure modes, real value. You cannot get these from an eval set, and the information often outweighs the incremental accuracy. This is the argument for shipping earlier than feels comfortable — but it is only valid if you have monitoring good enough to actually learn, and a rollback fast enough to survive being wrong.

The line I would give: "I would rather ship a narrower system I trust than a broader one I do not." Cut scope to where accuracy is already sufficient rather than delaying the whole thing.

### "Your eval says it improved but users complain. What is going on?"

Start by believing the users. The eval is a proxy; users are the thing. Then work through why the proxy broke — the hypotheses, roughly in order of how often they are the answer:

- **The eval set does not represent real traffic.** Built from synthetic data or from usage as it was six months ago. Check: sample recent production queries and see how many resemble your eval set. Usually the answer is "not many."
- **You improved the average and hurt the tail.** Aggregate metrics hide segment regressions. A change that helps 80% by 5 points and destroys 5% is a net win on the dashboard and a disaster in the inbox, because the 5% are the ones who write in. Always break metrics down by segment and query type.
- **You optimized a proxy, not the outcome.** Judge-scored helpfulness went up because answers got longer and more hedged. Verbosity bias in the judge (§5) is a real and common cause of exactly this pattern.
- **The complaint is about something the eval does not measure at all** — latency, tone, formatting, verbosity, refusal rate. Very common: a change that improves faithfulness by making the model more cautious also makes it more annoying, and no faithfulness metric captures annoying.
- **Overfitting to the eval set.** You have been iterating against it for months. Check against a held-out slice you have never looked at.
- **Change aversion.** Real, and usually decays. Distinguishable by whether complaints subside over two weeks and whether behavioral metrics move or only sentiment does.
- **The complaints are not representative.** A small vocal segment. Check whether behavioral metrics agree before you act — but do not use this as the default explanation, because it is the comfortable one and therefore the one you will reach for wrongly.

The output of this is not just a fix: every one of these should end with the eval set getting better, because the eval failed to catch a real problem and that is a bug in the eval.

---

## 10. Rapid fire

Short answers, one to three sentences. These get asked as warm-ups, as filler between deeper questions, and as a check that you know the vocabulary without needing to look it up.

**Temperature vs top-p?** Temperature rescales the logit distribution before sampling; top-p truncates to the smallest set of tokens whose cumulative probability exceeds $p$, then samples. Tune one, not both. Temperature 0 for extraction, classification, and anything you want reproducible.

**Does temperature 0 give you determinism?** Not fully. Batching, GPU non-associativity in floating point, and MoE routing all introduce variation. Plan for near-determinism, not determinism.

**What is a token, practically?** A subword unit. English averages roughly 4 characters per token; code, non-Latin scripts, and rare identifiers are far denser, which is why non-English users cost more and hit context limits sooner.

**Why is output more expensive than input?** Input is processed in parallel in one forward pass (prefill); output is generated sequentially, one forward pass per token (decode). Different compute profiles, different prices.

**TTFT vs TPOT?** Time to first token, dominated by prefill and therefore by prompt length; time per output token, dominated by decode and memory bandwidth. Streaming makes TTFT the number the user feels.

**What is the KV cache?** Cached key and value tensors for previous tokens so each new token does not recompute attention over the whole prefix. It is the main memory consumer in long-context serving. Mechanics in `13_interview_qa`.

**Continuous batching?** The scheduler swaps finished sequences out of a batch and new ones in per-step rather than waiting for the whole batch to finish. Large throughput win because generation lengths vary.

**Speculative decoding?** A small draft model proposes several tokens, the large model verifies them in one pass; accepted tokens are free. Latency win, no quality loss when done correctly.

**Quantization, one line?** Store weights (and optionally KV cache) at lower precision to fit more model and more context in the same VRAM. 8-bit is nearly lossless in practice; 4-bit is usually acceptable; below that depends heavily on the model and task.

**Distillation?** Train a small model on a large model's outputs to get most of the capability at a fraction of the cost. Often the most economically compelling fine-tuning use case.

**LoRA?** Train small low-rank adapter matrices instead of full weights. Cheap, fast, swappable, easy to roll back. What most teams should use.

**RLHF vs DPO?** RLHF trains a reward model then optimizes against it with RL; DPO optimizes preference pairs directly with a classification-style loss, skipping the reward model. DPO is simpler and cheaper; RLHF is more flexible. Math in `64_integrated_ai_ml_interview_synthesis`.

**What is a reasoning model?** One trained to spend inference-time compute on extended internal reasoning before answering. Better on hard multi-step problems, worse on latency and cost, and often no better on simple extraction. Route to it selectively.

**Cosine similarity vs dot product for embeddings?** Identical if vectors are L2-normalized, which most embedding models produce. If not normalized, dot product is length-sensitive and will favor longer texts.

**What is ANN and why?** Approximate nearest neighbor. Exact search is $O(n)$ per query; HNSW and IVF trade a small recall loss for orders-of-magnitude speedup. Details in `39_rag_retrieval_augmented_generation`.

**HNSW vs IVF, one line?** HNSW: better recall/latency, higher memory, expensive to build. IVF: lower memory, faster build, needs tuning of probe count. HNSW is the common default.

**Recall@k vs precision@k in RAG?** Recall@k is your ceiling — if the right document is not in the top $k$, nothing downstream can save you. Precision@k matters because irrelevant context actively degrades generation. Lead with recall.

**What is MRR?** Mean reciprocal rank — average of $1/\text{rank}$ of the first relevant result. Good when there is one right answer; use nDCG when there are several with different relevance grades.

**Why chunk overlap?** So a sentence spanning a boundary is not orphaned from its context in both chunks. 10–15% is typical; more is wasted storage and duplicated retrieval hits.

**What is contextual retrieval?** Prepending a short generated description of a chunk's surrounding context before embedding it, so chunks that reference "this policy" or "the above" are still retrievable. Cheap, measurable gain.

**Parent-document retrieval?** Search over small precise chunks, pass the larger parent section to the model. Precision of small units, completeness of large ones. Frequently the highest-leverage single change.

**What is HyDE?** Hypothetical document embeddings: have the model write a fake answer to the query, embed that, and search with it — because answers are embedding-closer to answers than questions are. Helps when queries and documents are stylistically mismatched; costs a model call.

**Query rewriting, why?** Real queries are underspecified and full of pronouns and follow-ups. Rewriting into a standalone, expanded query before retrieval is one of the cheapest large wins in conversational RAG.

**What is RRF, in one line?** Sum $1/(k+\text{rank})$ across retrievers with $k=60$; scale-agnostic fusion that needs no labeled data.

**Why not just use a 1M-token context window instead of RAG?** Cost, latency, and quality — context rot means performance degrades with length, so stuffing everything in is worse *and* more expensive than retrieving well. Also no attribution and no access control. Long context complements RAG; it does not replace it.

**When does long context genuinely beat RAG?** Small bounded corpora, tasks needing global understanding of one document, and prototyping where retrieval infrastructure is not worth building yet.

**What is semantic caching and its risk?** Serve a cached answer when a new query is embedding-similar to an old one. Risk: a loose threshold serves confidently wrong answers to different questions, and an unscoped cache key leaks across tenants.

**What is ReAct?** Interleaved reasoning and acting: the model alternates thought, action, observation in a loop. The base pattern most agent loops are a variant of.

**Function calling vs tool use — different?** Same thing, different naming across providers. The model emits a structured call against a schema you supplied and your code executes it.

**Why cap agent iterations?** Because loops fail silently and expensively. Combine an iteration cap with a token budget, a wall-clock deadline, and repeated-call detection.

**What is an agent's biggest practical failure mode?** Not reasoning failure — context accumulation. Tool results pile up, the context degrades, and the agent gets worse the longer it runs. Compact aggressively.

**Orchestrator-worker pattern?** A planner decomposes and delegates to stateless workers that return structured results, then synthesizes. The multi-agent pattern that actually works.

**What is MCP in one line?** An open protocol for exposing tools, resources, and prompts to models so integrations are written once instead of once per framework.

**Biggest MCP risk?** A third-party server's tool descriptions go straight into your model's context, so an untrusted server is an injection vector with tool privileges. Pin and review them like dependencies.

**Direct vs indirect prompt injection?** Direct: the user types the attack. Indirect: the attacker plants it in a document, page, email, or tool description your system will later read. Indirect is the serious one.

**The "lethal trifecta"?** Private data access, exposure to untrusted content, and external communication. An agent with all three can be turned into an exfiltration channel; keep it to two without a human in the loop.

**Fail-open or fail-closed guardrails?** Depends on the guardrail. PII filter fails closed; topical classifier probably fails open. Decide deliberately per guardrail — getting it backwards causes either leaks or an unusable product.

**Best way to reduce hallucination in RAG?** Let the system say "I don't know." A relevance floor plus an explicit insufficient-information path removes a class of hallucination rather than measuring it.

**Groundedness vs correctness?** Groundedness is whether claims are supported by the provided context. An answer perfectly grounded in a wrong document is faithfully wrong.

**Which hallucinations are most dangerous?** Omissions and altered qualifiers — dropping a "not," changing "may" to "will," skipping the exception clause. They pass groundedness checks because the words are all in the source.

**Biggest LLM-judge bias?** Position bias in pairwise comparison — a large fraction of verdicts flip on swapping order, worst when the answers are close in quality. Run both orders and treat flips as ties.

**How do you validate a judge?** Human-label a few hundred examples and measure agreement with Cohen's kappa, not raw accuracy. Use the judge only in the regime where it agrees.

**Pairwise or pointwise judging?** Pairwise. Models are poor at absolute 1–10 scoring and much better at "which is better."

**How big should an eval set be?** A few hundred well-chosen, stratified examples. Composition beats size; ten thousand scraped examples is worse than three hundred deliberate ones.

**What is the most-forgotten eval case?** Questions where the correct answer is "I don't know." Without them you are selecting for a system that always answers.

**Golden set vs regression set?** The golden set defines quality; the regression set is every bug you have shipped, kept forever as test cases. Different purposes, both required.

**How do you handle eval non-determinism?** Temperature 0 where possible; otherwise $n$ samples with a threshold and a tolerance band so normal variance does not create flaky red builds.

**What online signal is most useful?** Implicit ones — immediate rephrase, copy, accept, escalate. Thumbs-up/down has single-digit coverage and is biased toward angry users.

**Why pin the model version?** So the provider cannot change your product's behavior without a deploy. Floating aliases mean silent regressions.

**Why do prompts break on model upgrades?** They were partly compensating for the old model's quirks. Budget prompt work into every migration.

**Biggest hidden cost driver?** Retries and agent loops — the cost distribution is long-tailed and p99 can be tens of times the median. Cap loops before optimizing prompts.

**Cheapest latency win?** Streaming. It does not change total latency at all and transforms perceived latency.

**Second cheapest?** Fewer output tokens. Generation is sequential, so output length is close to linear in generation time.

**What is model routing?** Send the easy majority to a small fast model and escalate on a difficulty or confidence signal. Big win because real query distributions are heavily skewed toward easy.

**How does prompt caching change prompt design?** Static content first in a stable order, dynamic content last. One moving token near the front invalidates everything and shows up as a cost increase, not an error.

**What is a batch API for?** Non-interactive work at a substantial discount outside the interactive rate limit. Use it for backfills, evals, and offline enrichment.

**Retry strategy for a 429?** Exponential backoff with jitter, respecting `Retry-After`. Without jitter your retries synchronize and you self-DDoS on recovery.

**What is idempotency doing for you here?** Preventing a retried request from double-charging you and double-executing a side effect.

**KV cache formula?** $2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes}$ per token. The 2 is keys and values; GQA shrinks $n_{\text{kv heads}}$, which is why long context is affordable at all.

**How much VRAM for a 70B model?** ~140GB at BF16, ~70GB at 8-bit, ~35GB at 4-bit, each before ~15% overhead — then whatever is left is your KV cache and therefore your concurrency budget.

**When is self-hosting actually right?** Data cannot leave your environment; or sustained high volume on a task a small open model handles well. Most other reasons do not survive the total-cost arithmetic.

**Pre-filter or post-filter for access control?** Pre-filter, inside the search. Post-filtering is both a latency waste and a correctness bug, and asking the model to ignore unauthorized documents is not an authorization boundary.

**Where does PII accumulate that people forget?** Traces, logs, semantic caches, and eval sets built from production data. Each needs a retention policy and a tested deletion path.

**First thing you do in a bad-output incident?** Contain — kill switch or rollback — before diagnosing. Then assess blast radius, preserve traces, notify, and only then find root cause.

**What is the tell of someone who has only built demos?** They talk about capabilities and frameworks instead of constraints, numbers, and the failure that surprised them.

---

## 11. Where these came from

An honest note, because the sourcing on this topic is worse than it looks.

**Sources actually consulted and what each was worth:**

- [Exponent — AI engineer interview questions](https://www.tryexponent.com/blog/ai-engineer-interview-questions). The best of the listicles, and the only one with credible company attribution — specific questions tied to Anthropic (inference batching under synchronous load), Scale AI (insurance-claims RAG agent with token-cost control), Sierra (customer-service agent build), OpenAI (GPU credit system, fine-tuning strategy). Its framing that AI engineer loops are "a software or ML engineer loop with retrieval, serving, and agent questions added" matches what the rest of the evidence shows.
- [UPenn Career Services — 45 AI engineer interview questions](https://careerservices.upenn.edu/blog/2026/06/25/45-ai-engineer-interview-questions-answers-2026-guide/). Fetched, and it returns substantially the same top-10 list as Exponent, in the same order, with the same company attributions. Useful mainly as **direct evidence of the recycling problem**: a university career-services page reproducing a vendor blog's list is how these questions propagate. Its "what interviewers look for" notes are decent — separating retrieval eval from decision accuracy, validating LLM-as-judge against human labels.
- [KodeKloud — AI interview questions](https://kodekloud.com/blog/ai-interview-questions/). 30 questions, roughly the first 24 of which are definitional ("what is the difference between AI, ML, and deep learning?", "what is overfitting?"). Its scenario section (Q25–30: debugging a RAG chatbot that returns wrong answers, hallucination in production, hosted vs self-hosted, prod/test gap, cost and latency control, prompt injection) is genuinely discriminating and directly informed §2, §6, and §8 here.
- [InterviewCoder — AI engineer interview questions](https://www.interviewcoder.co/blog/ai-engineer-interview-questions). 50 questions across five clusters. The most structurally useful source for topic coverage, with the sharpest single framing I found: that evaluation is "the new system design." Its RAG and agent clusters (chunking, BM25 vs dense, RRF, reranker latency cost, scaling 1K→10M docs, ReAct, planner-executor, iteration caps) map closely to what the technical sources independently confirm matters.
- [Adil Shamim — "Every AI engineer interview question... from 100+ real interviews"](https://adilshamim8.medium.com/every-ai-engineer-interview-question-you-need-to-know-in-2026-from-100-real-interviews-b5b7ae4b961a). Broad taxonomy plus useful take-home examples. Caveat worth stating: its own sources section cites Reddit, YouTube, and Glassdoor, so it is aggregated crowd data, not insider access, and it blurs "asked at AI companies generally" with "asked of me." The OpenAI line it quotes — *"Is there an actual eval framework here, or is it vibes-based?"* — is the single most representative question in this whole chapter.
- [HimankSehgal/AI-interview-prep](https://github.com/HimankSehgal/AI-interview-prep). Checked, and it is currently a stub: a README promising 20+ companies, one completed file (Microsoft Applied Scientist 2), Amazon and Navi marked "coming soon", three commits total. Nothing usable was drawn from it. Listing it because the brief named it and verifying that a source is empty is part of the job.

**Technical sources used to ground the answers** (these, not the listicles, are where the numbers came from):

- [Chroma — Context Rot](https://www.trychroma.com/research/context-rot), for long-context degradation: performance falls with input length across retrieval, long-conversation QA, and even trivial replication; gaps exceeding tens of points between ~300-token focused and ~113k-token full prompts; degradation worse for low needle-question similarity and in the presence of distractors.
- [Judging the Judges: position bias in LLM-as-a-judge](https://arxiv.org/html/2406.07791v9), for the judge-bias numbers: position consistency 0.57–0.82 on MTBench pairwise, lower for list-wise; repetition stability ≥0.93 confirming the bias is systematic rather than stochastic; bias worst when answer quality gap is small.
- [Help Net Security on OWASP's 2026 agentic AI findings](https://www.helpnetsecurity.com/2026/06/11/owasp-prompt-injection-ai-security-failures/), for prompt injection mapping into six of ten Top 10 categories, the "lethal trifecta" framing, and Meta's "Agents Rule of Two."
- [Generating Structured Outputs from Language Models: Benchmark and Studies](https://arxiv.org/html/2501.10868v1), for the finding that constrained decoding does not degrade downstream task quality and improves it by up to ~4%, including on GSM8K.
- [LeanLM — prompt caching comparison](https://leanlm.ai/blog/prompt-caching), for the ~90% cached-input discount across providers, minimum cacheable prefix lengths, Anthropic's 1.25×/2× cache-write premiums, and TTL behavior. **Most volatile facts in this chapter.**
- [TianPan — hybrid search in production](https://tianpan.co/blog/2026-04-12-hybrid-search-production-bm25-dense-embeddings), for where dense retrieval fails (error codes, SKUs, function names, rare entities), RRF's $k=60$ convention versus tuned convex combination, and the top-100 retrieve / top-30–50 rerank latency shape.
- [WorkOS — MCP in 2026](https://workos.com/blog/everything-your-team-needs-to-know-about-mcp-in-2026), for MCP adoption scale, transports, the OAuth 2.1 auth convergence, and the enterprise gaps (audit trails, multi-tenancy, rate limiting, cost attribution).
- [DigitalApplied — VRAM, quantization, KV cache](https://www.digitalapplied.com/blog/how-much-vram-run-llm-quantization-kv-cache-context-2026), for the sizing formulas and the "once the weights fit, what is left is your context budget" framing.
- Anthropic's "building effective agents" guidance, via [secondary discussion](https://www.aihero.dev/building-effective-agents), for the workflow-before-agent default.

**The caveat, stated plainly.** Most public "AI engineer interview questions" content recycles the same lists. The UPenn/Exponent overlap above is a clean demonstration: the same ten questions, the same order, the same attributions, on two unrelated sites. Treat any listicle's claim of "most commonly asked" as unverified unless it names a company and a role.

**The answers here are original.** No answer text was taken from any source. The sources supplied the question distribution and the checkable facts; the answers are written from practice, and where a claim depends on a number I went and got the number rather than recalling it.

**Deliberately excluded as filler**, with reasons:

- *"What is AI / ML / deep learning?"*, *"what is overfitting?"*, *"bias-variance tradeoff"*, *"why split train/val/test?"*, *"what is feature engineering?"*, *"classification vs regression"* — roughly a third of the KodeKloud list. Real questions, wrong chapter: covered in `13_interview_qa` and `73_night_before_review`.
- *"What is a transformer / attention / what are embeddings?"* — covered in `64_integrated_ai_ml_interview_synthesis`. Kept only the KV-cache and inference-cost angles, which are applied-layer.
- *"What are the main risks and ethical concerns with AI?"* — asked, but as a values screen, not a technical discriminator, and any answer is unfalsifiable. §8's incident-response question tests the same territory with actual signal.
- *"What does parameter count tell you about a model?"* — increasingly meaningless post-MoE and post-distillation, and interviewers who ask it are usually reading from a list.
- *"Explain how RAG works"* — appears in every source. Kept implicitly (§2 assumes it) but not as its own entry, because if you cannot answer it the rest of §2 is unreachable anyway.
- Leetcode-style algorithm questions and generic behavioral questions ("tell me about a conflict"). Real parts of the loop, not specific to this role, and covered better elsewhere.
- Company-specific infrastructure puzzles like "implement a GPU credit management system" — a distributed-systems question wearing GenAI clothing. Named in §11 for realism; the answer is rate limiting and quota accounting, not AI engineering.
