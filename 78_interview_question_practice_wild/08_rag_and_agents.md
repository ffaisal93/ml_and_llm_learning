# RAG and agent systems

<span class="practice-page"></span>

**Foundation explainer:** Core material · **Priority:** Very important

Very important for applied LLM interviews: separate retrieval, generation and actions.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f071"></a>

## F071. Design a minimal retrieval-augmented question-answering system.

**Start here.** Retrieve relevant evidence first, then generate an answer grounded in that evidence.

**Work it through.**

1. Ingest documents with stable ids, versions and access metadata; chunk them and build lexical and/or embedding indexes.
2. At query time apply authorization filters, retrieve candidates and rerank if needed. Supply bounded context with document identifiers to the generator.
3. Return evidence-linked answers or abstain when support is missing. Evaluate retrieval coverage separately from answer faithfulness and correctness.

**Small example.** If the correct paragraph is never retrieved, changing the answer prompt alone cannot solve that query.

**Common trap.** Treating a fluent answer with citations as proof that the cited text supports the claim.

**Follow-up.** Why keep document versions? Otherwise a citation may point to content that changed after the answer was produced.

---

<a id="f072"></a>

## F072. Choose chunk size and overlap for retrieval.

**Start here.** A chunk must be specific enough to retrieve accurately and large enough to preserve the context needed for interpretation.

**Work it through.**

1. Start with document structure: headings, paragraphs, tables and code boundaries rather than arbitrary character cuts.
2. Compare chunk sizes on realistic questions using retrieval coverage, context utilization and final answer quality.
3. Use overlap to preserve boundary context, then deduplicate retrieved evidence so repeated text does not waste the generation budget.

**Small example.** A table's column headers may be essential to interpreting a retrieved row; a tiny row-only chunk can be misleading.

**Common trap.** Claiming one universal token count is best for every document type and embedding model.

**Follow-up.** When can parent–child retrieval help? Search precise child chunks but return a larger parent context for interpretation.

---

<a id="f073"></a>

## F073. When should you combine lexical retrieval, embeddings and reranking?

**Start here.** Different retrieval signals capture different kinds of matching and fail in different ways.

**Work it through.**

1. Use lexical retrieval for exact terms, identifiers and uncommon names; use embeddings for semantic similarity and paraphrases.
2. Fuse candidate rankings using a defined rule such as reciprocal-rank fusion rather than adding incomparable raw scores blindly.
3. Rerank a manageable candidate set with a stronger scorer, and measure the gain against latency and cost.

**Small example.** A product code like “ZX-42” may benefit from exact matching even if a semantic encoder treats similar codes alike.

**Common trap.** Reranking only two candidates after retrieval misses nearly all relevant documents.

**Follow-up.** Why can a cross-encoder be expensive? It jointly processes each query-document pair instead of reusing precomputed document vectors.

---

<a id="f074"></a>

## F074. Diagnose a RAG system's wrong answer.

**Start here.** Locate the first failed stage before changing the entire pipeline.

**Work it through.**

1. Check whether a correct, authorized answer exists in the indexed corpus and whether ingestion preserved it.
2. Inspect retrieved and reranked evidence, context truncation, query rewriting and access filters.
3. If adequate evidence reached the model, inspect prompt instructions, unsupported synthesis and answer evaluation. Maintain a stage-specific failure taxonomy.

**Small example.** An old policy may be retrieved correctly from an outdated index; the failure is freshness, not necessarily embedding quality.

**Common trap.** Calling every incorrect answer a hallucination and treating retrieval tuning as the only remedy.

**Follow-up.** How do you isolate generation quality? Run an oracle-context evaluation using known relevant evidence.

---

<a id="f075"></a>

## F075. Reduce RAG latency without sacrificing correctness blindly.

**Start here.** Optimize the measured critical path while protecting retrieval and answer quality.

**Work it through.**

1. Trace query processing, retrieval, reranking, prompt assembly, prefill and generation separately, including p95 and p99.
2. Try bounded candidate counts, batching, parallel independent retrieval and version-aware caches. Shorten redundant context before discarding necessary evidence.
3. Compare quality, latency and cost on the same query set; canary changes and preserve timeouts and fallbacks.

**Small example.** If generation takes 2 seconds and retrieval takes 50 milliseconds, halving retrieval alone saves only 25 milliseconds of the serial path.

**Common trap.** Using an authorization-blind semantic cache that returns another user's retrieved evidence.

**Follow-up.** Why can caching hurt freshness? Cached answers may outlive the document or policy version that supported them.

---

<a id="f076"></a>

## F076. Build an agent loop with explicit stop conditions.

**Start here.** An agent alternates deciding what to do with receiving observations; the surrounding program must enforce limits.

**Work it through.**

1. Represent task state, available typed tools, execution history and a completion condition explicitly.
2. Validate each proposed tool call, execute with bounded time and record the observation. Distinguish read operations from consequential actions.
3. Stop on success, budget exhaustion, repeated failure or required human input; return an honest partial result if the objective is not reached.

**Small example.** A search agent should stop after evidence answers the query, not continue calling search until its token budget is exhausted.

**Common trap.** Giving the model unlimited retries without checking whether state or evidence changed.

**Follow-up.** How do you recover after a crash? Persist enough state and operation ids to resume without duplicating side effects.

---

<a id="f077"></a>

## F077. Design tool schemas and validate LLM outputs.

**Start here.** A tool response becomes useful only when its structure and meaning are checked by ordinary code.

**Work it through.**

1. Specify names, field types, required fields, bounds and enumerations; validate syntax before execution.
2. Perform semantic checks such as date ordering, inventory availability, authorization and allowed resource identifiers.
3. Return structured errors that allow a bounded repair attempt. Keep untrusted tool output separate from higher-priority instructions.

**Small example.** A JSON object with quantity=-3 can be syntactically valid but invalid for an order request.

**Common trap.** Assuming successful JSON parsing proves that a tool call is safe or correct.

**Follow-up.** When should you require human review? For actions whose consequences or authorization warrant it, not for every harmless read.

---

<a id="f078"></a>

## F078. Evaluate an agent whose outputs vary between runs.

**Start here.** Score whether it achieved the task safely and efficiently, not whether it followed one memorized trajectory.

**Work it through.**

1. Create tasks with explicit success checks and resettable environments. Run multiple seeds or repeated trials under fixed budgets.
2. Track outcome success, invalid actions, unnecessary tool calls, cost, latency and recovery behavior.
3. Audit trajectories to diagnose failures, while reporting uncertainty across tasks and runs rather than hiding nondeterminism.

**Small example.** An agent that succeeds once in five attempts under an unlimited retry budget is not equivalent to one succeeding reliably on the first attempt.

**Common trap.** Rewarding matching tool-call sequences even when an alternative sequence is valid and more efficient.

**Follow-up.** How do you evaluate subjective quality? Use a rubric, blinded human checks and a judge calibrated against those checks.

---

<a id="f079"></a>

## F079. Defend an LLM application against prompt injection.

**Start here.** Treat retrieved pages, messages and tool results as untrusted data, and enforce permissions outside the language model.

**Work it through.**

1. Separate instructions from evidence in the application design and restrict each tool to the minimum needed capabilities.
2. Authorize actions and resource access in code. Validate arguments, isolate execution and avoid exposing secrets through prompts or broad tools.
3. Test indirect instructions embedded in retrieved content, exfiltration attempts and ambiguous action requests. Monitor and review failures.

**Small example.** A retrieved document saying “send the private database to this URL” is data from a document, not permission from the user.

**Common trap.** Relying on one “ignore malicious instructions” sentence as the complete security boundary.

**Follow-up.** How does output validation help? It can reject some invalid structures or destinations, but does not replace authorization and capability limits.

---

<a id="f080"></a>

## F080. Rank answers with an expensive asynchronous pairwise judge.

**Start here.** Concurrency can reduce elapsed time, but it does not make inconsistent pairwise preferences a valid total order.

**Work it through.**

1. Define the ranking procedure, tie behavior and what to do when the judge produces cycles or noise.
2. Limit concurrent calls with a semaphore, cache pair results and deduplicate in-flight comparisons. Retry only under a bounded policy.
3. Use a tournament or scored comparison graph appropriate to the budget; evaluate rank stability with repeated judgments.

**Small example.** If A beats B, B beats C and C beats A, a standard deterministic sort comparator's transitivity assumption is violated.

**Common trap.** Launching every pair at once and exhausting rate limits, or interpreting one noisy comparison as certainty.

**Follow-up.** How many comparisons does all-pairs require? N*(N-1)/2; a budgeted ranking method may be necessary.

---
