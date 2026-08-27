# AI system design scenarios

These are scenario questions, not knowledge questions. Nobody is checking whether you can define a vector index. They give you a vague ask, a broken requirement, or a hard constraint, and they watch what you do with it. Almost every question below has the same hidden test: can you turn a wish into a specification that contains numbers? A specification has a metric, a dataset, a threshold, a load level, and a budget. A wish has adjectives.

The answers here give the reasoning, not a template to recite. If you memorise the words you will fail the follow-up, because the interviewer changes one constraint and the recital collapses. For the full design method, see [`ML_SYSTEM_DESIGN_DEEP_DIVE.md`](ML_SYSTEM_DESIGN_DEEP_DIVE.md).

---

### Q1. You have been asked to add an AI feature to an existing product, but the PM only says "make it smarter." Walk through every clarifying question you would ask before writing a single line of design.

"Make it smarter" is a product wish, not a requirement. I cannot design against it, because it names no user, no task, and no threshold. So my whole job before design is to convert it into a problem statement with measurable outcomes. I would ask in three groups, and I would not start architecture until all three are answered.

First, the problem. What user behaviour or complaint drove this request? What does "smarter" mean here — faster, more accurate, more personalised, or fewer steps? Those are four different systems. How do users solve this today, and what is wrong with that path? If there is no current path, the feature is new behaviour and adoption is the risk, not accuracy.

Second, the users and the usage. Who uses this, and how technical are they? How often — a few times a month, or every minute? Is it real-time, batch, or triggered by an event? That single answer sets the latency budget, and the latency budget rules out most model choices before I draw anything. And what does a good output actually look like? I want three examples of a good answer and three of a bad one, written by the PM.

Third, the data and the quality bar. What data exists, is it clean, and is any of it labelled? What is the baseline I am trying to beat, because a model with no baseline cannot be judged. What is the error tolerance? If the model is wrong five percent of the time, is that an annoyance the user corrects, or a compliance incident? Those two answers produce opposite designs: the first can ship a single model call, and the second needs confidence scores, abstention, and human review.

These questions take an hour. Skipping them costs weeks, because you discover the real requirement after you have built the wrong thing.

> **Say it.** "Make it smarter" is a wish, so my first move is to turn it into a spec. I ask three groups of questions. What is the actual friction, and does smarter mean faster, more accurate, or more personal. Who uses it, how often, and is it real-time or batch, because that sets my latency budget. And what data exists, what is the baseline, and what error rate is acceptable. That last one matters most: five percent wrong is fine in a draft assistant and unacceptable in a billing flow, and those are different architectures.

---

### Q2. You are given the goal "improve user experience using AI." How do you break this down into a concrete, scoped system with measurable outcomes?

You cannot improve user experience in general, so the first step is to find one friction point in one journey. I walk the funnel with analytics and support tickets and look for the step where users drop out, retry, or contact support. That step becomes the target. Everything else is out of scope for version one.

The second step is to translate that friction into an AI task type, because the task type determines the architecture. "Users cannot find the content they need" is a retrieval problem, so it becomes search plus reranking. "Everyone gets the same generic response" is a personalisation problem, so it becomes ranking over user features. "Users retype the same data" is extraction or automation, so it becomes a structured-output model with schema validation. Naming the task type early stops the team from reaching for a chat box by reflex.

The third step is metrics, and I want two kinds. A leading metric measures system output quality and moves within days, for example retrieval recall at five or extraction F1. A lagging metric measures real user impact and moves in weeks, for example support ticket volume. Leading metrics tell you the system works. Lagging metrics tell you it mattered. You need both, because a system can be accurate and useless.

Then define done as a number. Not "faster answers" but "reduce median time to answer from 3 minutes to under 45 seconds for the top 20 support intents." That sentence is testable, and it tells you when to stop.

Finally, scope the MVP to one user type, one workflow, and one data source. One data source matters most, because each extra source multiplies the ingestion, permission, and freshness work without changing what you learn. Ship that, measure, then widen.

> **Say it.** I would refuse the goal as stated and find one friction point in the journey, using drop-off data and support tickets. Then I name the AI task type, because cannot-find is retrieval, generic-response is personalisation, and retyping is extraction, and those are three different systems. I set one leading metric like retrieval recall and one lagging metric like ticket volume. I define done as a number, such as median time to answer under 45 seconds, down from 3 minutes. Then I scope version one to a single user type, workflow, and data source.

---

### Q3. A traditional backend engineer joins your AI team. What are the three biggest mindset shifts they need to make?

The first shift is from deterministic to probabilistic thinking. In backend work, the same input gives the same output, and a single test proves correctness. With a model, the same input can give different outputs, so quality is a property of a distribution, not of one call. You measure it statistically over an evaluation set and report a rate, for example 94 percent exact-match on 500 held-out cases. A single passing example proves nothing, and a single failing example is not necessarily a bug. This is the hardest shift, because it removes the feeling of certainty that backend testing gives.

The second shift is from code correctness to data quality. When an AI system misbehaves, the code is usually fine. The cause is normally the data: a stale index that does not contain the document, chunks split so badly that the answer spans two of them, or an evaluation set that does not represent real traffic. So debugging starts with looking at what went into the model, not at the call stack. A useful habit is to log the exact retrieved context with every response, because most "the model is dumb" reports turn out to be "the model never saw the answer."

The third shift is from binary pass or fail to graceful degradation. Backend services either return 200 or they fail loudly. Models fail quietly and partially. They will sometimes be wrong, sometimes slow, and sometimes unavailable. Therefore you design the fallback path first: a smaller model when the primary is rate-limited, cached or template output when both fail, and an explicit "I do not know" when confidence is low. Degrading to a worse answer is usually better than an error page, but only if the user can tell the difference, so surface it.

The measurement side of shift one is covered in [`../03_evaluation_metrics/README.md`](../03_evaluation_metrics/README.md).

> **Say it.** Three shifts. First, probabilistic thinking: the same input can give different outputs, so quality is a rate over an eval set, not an assertion on one call. Second, data quality over code correctness: when the output is wrong, the code is usually fine and the index is stale or the chunk is split badly, so I log retrieved context with every response. Third, graceful degradation instead of pass or fail: the model will be wrong or slow sometimes, so I design the fallback model and the I-do-not-know path before I ship.

---

### Q4. Your AI system produces slightly different outputs for the same input. How does this non-determinism affect downstream components?

It breaks four things that the rest of the stack assumes. Caching layers assume that the same input maps to one correct output, so caching a model response is fine for cost but it freezes whatever you happened to get, including a bad answer, until the entry expires. Deterministic tests that assert exact strings fail constantly, so they get muted and then the suite protects nothing. Downstream integrations that parse structured output break on formatting drift, because a stray markdown fence or a renamed key is enough. And you cannot reproduce a bug by re-running the input, which removes the normal debugging loop.

The responses are practical. Use temperature 0 on any path where the output feeds a machine rather than a person, which means classification, extraction, and routing. Log the prompt version, the model ID, the sampling parameters, and any seed alongside every response, because without those four you cannot even attempt a reproduction. Replace exact-match tests with semantic assertions or a grader model, and validate the schema at every stage boundary rather than only at the end, so a malformed output fails where it was produced.

One correction that matters: temperature 0 is not true determinism. It selects the highest-probability token, but the probabilities themselves can shift between runs. Server-side batching changes the shapes of the matrix operations, floating-point addition is not associative on GPU so a different reduction order gives a slightly different sum, and mixture-of-experts routing can send the same token to different experts depending on what else is in the batch. When two candidate tokens are nearly tied, a tiny numerical difference flips the choice, and one flipped token changes the rest of the generation. So treat temperature 0 as low variance, not zero variance, and keep the schema validation regardless.

> **Say it.** Non-determinism breaks caching, exact-match tests, structured-output parsers, and bug reproduction. So I set temperature 0 wherever the output feeds a machine, log the prompt version, model ID and sampling parameters with every response, and validate schema at each stage boundary instead of only at the end. Tests become semantic or grader-based. The trap is assuming temperature 0 gives determinism. It does not, because batching, floating-point non-associativity on GPU, and expert routing all shift the logits slightly, and a near-tie flips a token.

---

### Q5. After a model upgrade, accuracy improves by 15 percent but p99 latency doubles. How do you decide what to prioritise?

This is a product trade-off, not a technical one, so the answer depends entirely on who is waiting. I would not answer it globally. I would split traffic by use case and decide per path.

On a real-time chat surface the user is watching a cursor blink. If p99 goes from 1.5 seconds to 3 seconds, a slice of sessions crosses the point where people abandon or retry, and a retry costs a second full request. The 15 percent accuracy gain never reaches those users, because they left before the answer arrived. So I block the upgrade on real-time paths until latency comes down. The levers are streaming, so time to first token stays low even if total time grows, speculative decoding, and prompt shortening, since prefill cost scales with input length. Those figures are illustrative; the real ones come from your own load test.

On an overnight batch pipeline the calculus inverts. A 2-hour job becoming a 4-hour job has zero user impact as long as it finishes before the business day. Nobody is waiting, so I take the accuracy and ship it the same week.

The principle is that latency and accuracy are never evaluated in isolation. You map both against user tolerance for each specific path, and then you route. The practical outcome is model tiering: the fast model serves interactive traffic, the accurate model serves batch, asynchronous, and high-stakes traffic, and a router picks between them on request type rather than on a global config flag. That also gives you a safe rollout, because you can move one path at a time and watch abandonment rather than flipping everything at once.

One thing to check before any of this: is the 15 percent a real gain? Confirm it on a held-out set that matches production traffic, not on the benchmark the vendor quoted, because an upgrade that improves a public benchmark can be flat or worse on your distribution.

Serving-side latency levers are covered in [`../06_llm_inference/README.md`](../06_llm_inference/README.md).

> **Say it.** I would not answer this globally, because it depends on who is waiting. On a real-time chat path, p99 moving from 1.5 to 3 seconds pushes users into abandoning, and the accuracy gain never reaches anyone who left, so I block the upgrade until streaming or speculative decoding brings latency back. On an overnight batch job, 2 hours becoming 4 hours costs nothing, so I ship it immediately. The end state is model tiering with a router, fast model for interactive traffic and accurate model for batch. And I verify the 15 percent on my own held-out set first.

---

### Q6. You are launching an AI feature in 4 weeks. How do you define success metrics before launch?

Metrics defined after launch are storytelling. The reason is mechanical, not philosophical: you cannot measure what you did not log, and adding telemetry after the fact gives you no baseline to compare against. So the metric definition is part of the build, and for each metric I write down three things — the exact event to log, the current baseline value, and the target at day 30, 60 and 90.

I use three tiers. System quality covers output accuracy on a fixed evaluation set, hallucination or ungrounded-claim rate, p95 and p99 latency, and cost per query. User behaviour covers adoption, task completion rate, and the retry or edit rate, which is the most honest signal in the whole list, because a user who rewrites the model's output is telling you it was wrong without filing a ticket. Business impact covers support ticket volume, retention, and end-to-end workflow time.

The tiers do not all matter at once. In week one I watch system metrics, because that is where fires start and where a bad deploy shows first. In month one I work on user behaviour metrics, since adoption and edit rate tell me whether the output is actually usable. In month three I judge business impact against the cost of running the thing, which is the only tier that decides whether the feature survives.

Two practical notes. Freeze the evaluation set before launch and version it, otherwise your accuracy number moves for reasons unrelated to the model. And log cost per query from day one, because cost is the metric that quietly kills features, and it is the easiest one to add late and regret.

> **Say it.** I define them before launch, because you cannot measure what you did not instrument and you have no baseline afterwards. For each metric I write the event to log, the baseline, and the target at day 30, 60 and 90. Three tiers: system quality is accuracy, hallucination rate, p95 and p99 latency, and cost per query. User behaviour is adoption, completion, and edit or retry rate, which is the most honest signal because it means the output was wrong. Business impact is tickets, retention, and workflow time. Week one I watch system, month three I judge business impact.

---

### Q7. A junior engineer defines requirements as "the model should be accurate and fast." What is wrong with this?

It is an ambition, not a specification. You cannot design against it, you cannot test against it, and you cannot tell when you are finished, so the team will keep tuning forever or stop arbitrarily.

"Accurate" is undefined in three ways. On what dataset — a public benchmark, or a curated set that looks like your traffic? Measured how — precision, recall, F1, or exact match, which can differ by twenty points on the same model? And with what error distribution, because false positives and false negatives carry different costs. A fraud filter that misses a fraud loses money; one that blocks a good customer loses the customer. Until someone names which error is worse, the model has no target to optimise.

"Fast" is equally vague. Fast for the median user or at the 99th percentile? Under what concurrency, since latency is meaningless without load? And measured as time to first token or as total response time, which differ by seconds on a streaming interface where the user is already reading.

The rewrite makes all of that explicit: "the extraction pipeline must achieve at least 92 percent F1 on our golden evaluation set of 500 edge-case documents, and p95 end-to-end latency must be under 800 milliseconds at 100 concurrent requests." Now there is a dataset, a metric, a threshold, a percentile, and a load level. Every one of those five is a decision someone had to make, and writing them down is what turns a wish into engineering. I would also add the error-cost line — for example, a false extraction is worse than a missing one, so bias the threshold toward abstaining — because that is what tells the engineer which way to move when the two metrics conflict.

> **Say it.** It is an ambition, not a spec, so nobody can test it or know when it is done. Accurate on which dataset, measured by precision, recall or F1, and is a false positive worse than a false negative. Fast at the median or the 99th percentile, under what concurrency, and time to first token or total time. I would rewrite it as: at least 92 percent F1 on a golden set of 500 edge-case documents, and p95 latency under 800 milliseconds at 100 concurrent requests. That has a dataset, a metric, a threshold, a percentile, and a load level.

---

### Q8. Your system must respond in under 150 ms end to end. Walk through how this constraint shapes design.

A 150 millisecond budget removes the standard LLM architecture from consideration, so the first thing I do is write the budget down and subtract. Network and TLS overhead takes about 30 ms. Pre-processing and post-processing take about 20 ms. Retrieval takes about 30 ms. That leaves roughly 70 ms for inference. These splits are illustrative and you should measure your own, but the shape holds: inference gets less than half the budget.

That number rules out hosted frontier models immediately. Their time to first token alone is typically several hundred milliseconds and can reach a few seconds under load, before a single output token is generated. So the model must be small and self-hosted on dedicated accelerators with the weights already resident: an encoder in the BERT family fine-tuned for the task, or a generative model in the roughly 1B to 7B range with a short output. If the task is classification, ranking, or extraction, an encoder is the right answer and it is often a 10 to 20 ms forward pass.

Retrieval gets the same treatment. Vector search over millions of documents typically costs 50 to 100 ms against a networked vector database, which does not fit in 30 ms once you add a network hop. So the index must be in memory in the serving process, for example HNSW held in RAM, or the results must be precomputed and cached in a low-latency store such as Redis. Anything that crosses a network boundary twice is already over budget.

The architectural rule that follows is: no sequential model calls. A reranker after a retriever, or a second LLM pass to check the first, each add their full latency to the critical path. Everything is parallel, cached, or precomputed. Practically that means firing retrieval and any classification concurrently, warming the cache from traffic patterns, and pushing anything that can be stale out of the request path entirely.

If 150 ms is genuinely non-negotiable and the task genuinely needs a large model, the correct answer is to challenge the requirement or make the response asynchronous, not to pretend.

> **Say it.** I write the budget down first. About 30 milliseconds network, 20 for pre and post processing, 30 for retrieval, leaving roughly 70 for inference. That rules out hosted frontier models, because their time to first token alone is hundreds of milliseconds. So it is a small self-hosted model, a fine-tuned encoder or something in the 1B to 7B range on a dedicated GPU. Retrieval has to be an in-memory HNSW index or a precomputed cache, because a networked vector database costs 50 to 100 milliseconds. And no sequential model calls at all.

---

### Q9. Under load, one pipeline component will bottleneck first. How do you identify which one before production?

I profile rather than guess, because intuition about bottlenecks is usually wrong and the cost of guessing is scaling the wrong component.

The first step is to isolate each stage and benchmark it alone. I measure embedding generation, vector lookup, reranking, and model inference separately, and for each I record throughput in requests per second and latency at rising concurrency levels — say 1, 10, 50, 100, 200. What I am looking for is the point where latency starts climbing while throughput stops climbing. That knee is the component's capacity ceiling, and the concurrency at which it appears is the number I care about.

The second step is to find which resource saturates first at that knee: GPU utilisation, CPU, memory bandwidth, or connection pool limits. In most retrieval-augmented pipelines the ceiling is model inference or the reranker, because both are compute-bound and both process every candidate. A cross-encoder reranker over 50 candidates is 50 forward passes for one user request, so it saturates a GPU far earlier than people expect.

The third step is end-to-end load testing with realistic traffic, including spikes rather than only a steady ramp, because a spiky shape exposes queueing behaviour that a smooth ramp hides. Then I watch queue depth between stages. The stage whose input queue grows without draining is the bottleneck, and queue depth finds it faster than latency graphs do, because latency rises everywhere once one stage backs up.

Then I design mitigations for that specific component rather than scaling everything: dynamic batching and autoscaling for inference, a smaller candidate set or a cheaper reranker, and fallback routing to a lighter model when the queue crosses a threshold. Shedding load deliberately at a known point beats collapsing at an unknown one.

Pipeline structure and reranking cost are covered in [`../39_rag_retrieval_augmented_generation/README.md`](../39_rag_retrieval_augmented_generation/README.md).

> **Say it.** I profile each stage in isolation first, measuring throughput and latency at rising concurrency until latency climbs while throughput flattens. That knee is the capacity ceiling. Then I check which resource saturates there, and in most RAG pipelines it is inference or the reranker, because a cross-encoder over 50 candidates is 50 forward passes per request. Then end-to-end load tests with spiky traffic, watching queue depth between stages, because the stage with a growing input queue is the bottleneck. Then I add batching, autoscaling, or fallback routing to that stage only.

---

### Q10. How does explicitly mapping data flow at each stage change the quality of your design?

It converts a hand-wavy AI idea into system engineering, because you cannot draw a data flow without answering questions you were previously able to skip.

It forces schemas. Once you draw the arrow between retrieval and generation, you must say what travels along it: which JSON fields, which types, and what token limit. That last one is where most designs break, because the retrieval stage happily returns twenty chunks and the generation stage has room for six. Writing the contract exposes the mismatch on a whiteboard instead of in production.

It forces error handling at every boundary. Each arrow has a failure mode, and drawing it makes you name them: retrieval returns zero results, the model returns malformed JSON, the assembled prompt exceeds the context window and gets truncated silently. Silent truncation is the dangerous one, because the system keeps returning confident answers built from half the evidence. A boundary with a defined error path turns that into a caught, logged, and handled event.

It identifies telemetry points. Every boundary is a place to log input, output, latency, and cost. Deciding this at design time gives you a trace that shows exactly which stage failed, instead of one log line saying the answer was bad.

And it decouples the components. Once retrieval, orchestration, and generation have written contracts, three people can work on them in parallel, and you can swap the model or the vector database without breaking anything downstream, provided the contract holds.

The result is a system that is testable at each boundary, observable in production, and modular enough to change one part at a time. That is the actual difference between a demo and a product.

> **Say it.** Drawing the data flow forces three things you can otherwise skip. First, schemas on every arrow, including token limits, which is where retrieval returning twenty chunks meets a generation stage with room for six. Second, an error path at every boundary, and the one that matters is silent context truncation, because the system keeps answering confidently from half the evidence. Third, telemetry points, so a trace tells you which stage failed instead of just that the answer was bad. It also decouples the teams, so you can swap a model without breaking downstream.

---

### Q11. You have been asked to design a system where the core logic is "call GPT-4 and return the result." Why is this not a system design?

Because an API call is a prototype. The system design is everything around that call that makes it reliable, safe, scalable, and affordable. The call itself is maybe 20 lines. The system is five layers, and each one exists because of a specific failure that happens without it.

Guardrails and safety come first, because the input is user-controlled text going into a component that follows instructions. That means prompt-injection defence, input validation and length limits, output scrubbing for personal data that the model may have copied out of retrieved context, and moderation on both sides. Context management is next: multi-turn history has to be pruned or summarised, the token budget has to be enforced before the call rather than discovered by an error, and the prompt has to be assembled from versioned parts so you know which version produced which output.

Resilience is the third layer. The provider will be down, rate-limited, or slow, so you need fallback routing to a second model, exponential backoff with jitter on 429 and 5xx responses, timeouts shorter than the user's patience, and a circuit breaker so a failing dependency does not hold every worker thread. Performance and cost is the fourth: semantic caching for repeated questions, deduplication of identical in-flight requests, and streaming so time to first token, not total time, is what the user feels.

Observability is the fifth and the one people skip. You need traces that capture the prompt inputs, the retrieved context, token counts, cost per request, and the evaluation result, because without them you cannot debug a bad answer or explain a bill.

Without these layers you have a fragile wrapper around somebody else's API, and its uptime, cost, and safety posture are all decided by a vendor. The injection and data-exfiltration side is covered in [`../65_llm_security/README.md`](../65_llm_security/README.md).

> **Say it.** An API call is a prototype. The design is the five layers around it. Guardrails, because the input is user text going into something that follows instructions, so prompt injection defence and output PII scrubbing. Context management, so history pruning and token budgets are enforced before the call. Resilience, so fallback models, backoff, and circuit breakers. Performance and cost, so semantic caching and streaming. And observability, so traces of prompt, tokens, and cost. Without those you have a fragile wrapper whose uptime and bill a vendor controls.

---

### Q12. You need to design a system for legal document review. What are the boundaries and contracts?

High-stakes domains need an explicit system boundary and zero-trust verification contracts, so I would write the boundary before the architecture.

The boundary: the system never gives legal advice and never makes a final decision. It is limited to three jobs — flagging clauses that match known risk patterns, comparing clauses against a standard template, and mapping citations to their sources. An attorney makes every judgement. Stating this narrows the design enormously, because it converts an open-ended generation problem into a bounded extraction and comparison problem, which is far easier to evaluate.

Then the contracts, one per boundary. The ingestion contract requires the parser to preserve exact page numbers, line numbers, and bounding-box coordinates for every extracted span, because without coordinates you cannot ground an answer back to the page and the whole verification story fails. The extraction contract requires a confidence score and a verbatim snippet pointer for every finding, with no ungrounded summary permitted; if the model cannot point at text, the finding does not exist. The evaluation contract sets the hallucination rate on factual extraction to zero as a hard requirement, which means the system must abstain rather than guess, and any low-confidence output is emitted as a "needs human review" flag rather than a claim.

Human-in-the-loop is the last piece and it is a design requirement, not a UI nicety. Every output needs a one-click path from the finding to the highlighted source text in the original PDF. If verifying a flag takes longer than reading the clause, the attorney stops using the system, and then the accuracy of the model is irrelevant.

The cost model follows from this: a false negative, a missed risky clause, is far worse than a false positive, so tune recall high and let the attorney discard noise.

> **Say it.** I would write the boundary first: the system never gives legal advice or a final decision, it only flags risk clauses, compares against a template, and maps citations. Then contracts at each boundary. Ingestion must preserve page numbers, line numbers, and bounding boxes, because without coordinates you cannot ground anything. Extraction must return a confidence score and a verbatim snippet pointer, with no ungrounded summaries. Evaluation requires zero hallucination on factual extraction, so uncertainty becomes a needs-human-review flag. And every finding gets one-click verification against the highlighted PDF.

---

### Q13. Your system has retrieval, orchestration and generation. Due to a budget cut, one must be simplified. What is your trade-off strategy?

I choose the simplification that costs the least end-user utility, and that is not the one that saves the most money. So I rank the three by what breaks.

Simplifying generation means swapping a frontier model for a small or open one. The prose gets less polished and complex reasoning suffers, but factual accuracy holds up well if retrieval is still good, because the model is mostly summarising evidence you handed it. The saving is large — small models often cost one to two orders of magnitude less per token than frontier models, though exact prices change constantly and you should check current rates rather than trust any figure in a document. This is my first cut.

Simplifying orchestration means replacing multi-step agent loops with single-shot retrieval-augmented generation. You lose multi-turn tool use and complex workflows, but straightforward question answering stays fast and actually gets more reliable, because each removed step was also a failure point. It removes several model calls per request, so it cuts both cost and latency. This is my second cut.

Simplifying retrieval means dropping hybrid dense plus keyword search back to plain BM25, and this is the one I refuse. Retrieval quality sets the ceiling on everything downstream. When the right passage is not in the context, the model does not fail loudly — it produces a fluent, confident, wrong answer, because generating text is exactly what it will do with whatever it was given. Bad context turns a good model into a hallucination engine, and no amount of generation quality recovers it.

So: cut generation first, orchestration second, never retrieval. If the budget cut is severe enough that retrieval must be touched, the correct move is to reduce scope — cover fewer document types or fewer users — rather than degrade retrieval for everyone.

> **Say it.** I cut generation first. Swapping a frontier model for a small one loses some polish, but factual accuracy holds if retrieval is solid, and the token cost drops by one to two orders of magnitude. Orchestration second: replacing agent loops with single-shot RAG loses complex workflows but removes failure points and API calls. I never degrade retrieval. If the right passage is not in context, the model does not fail loudly, it produces a fluent wrong answer, so bad retrieval turns a good model into a hallucination engine. If retrieval must be touched, I cut scope instead.

---

### Q14. You have a 500 dollar per month infrastructure budget to serve 10,000 users. Architect this cheaply.

I start with unit economics, because the per-query budget decides the architecture before any other choice. Ten thousand users at about 3 queries a day is 900,000 queries a month. Five hundred dollars divided by 900,000 gives about 0.00055 dollars per query, or roughly one twentieth of a cent. A frontier model call at around 0.03 dollars per query is therefore 54 times over budget. That is not a tuning problem, so the frontier model is out on the first line of arithmetic.

Model choice follows. I allocate about half the budget, 250 dollars, to inference, and use either the cheapest small hosted models or a fine-tuned small open model on serverless GPUs that scale to zero, so idle hours cost nothing. Small and fine-tuned beats large and general here, because a 3B model tuned on your task can match a much larger general model on a narrow job at a fraction of the cost.

Caching is the lever that closes the remaining gap. A semantic cache that reaches a 70 percent hit rate cuts real model calls from 900,000 to 270,000, which roughly triples the affordable cost per real call. A hit costs nothing in inference and returns in tens of milliseconds, so it improves latency and cost together. For the rest, use an embedded vector database on the same server rather than a managed service, since managed vector search alone can consume the whole budget, and host the backend on a serverless platform so you pay for requests rather than idle capacity.

The caveat on caching matters. A semantic cache trades correctness for cost, because a near-miss returns the answer to a slightly different question. Set the similarity threshold deliberately, and measure the wrong-answer rate the cache introduces on a labelled sample. Do not report the hit rate without it, because a 90 percent hit rate with a loose threshold is a bug, not an achievement.

> **Say it.** I start with unit economics. Ten thousand users at 3 queries a day is 900,000 queries a month, so 500 dollars gives about 0.00055 dollars per query. A frontier model at about 0.03 a query is 54 times over budget, so it is out immediately. I use a small fine-tuned model on serverless GPUs that scale to zero, an embedded vector database instead of a managed one, and a semantic cache targeting a 70 percent hit rate, which cuts real calls to 270,000. But I measure the wrong-answer rate the cache introduces, because a loose threshold answers a different question.

---

### Q15. Your prototype works flawlessly in a notebook. List the assumptions it makes which production will invalidate.

A notebook assumes an ideal world in five ways.

Data cleanliness. The notebook ran on well-formed samples that someone chose because they worked. Production sends scanned images with no text layer, wrong encodings, missing headers, and files large enough to exceed the context window. So the parser needs an OCR fallback and a rejection route, not a stack trace.

Concurrency. The notebook runs one request at a time with the whole machine to itself. Production sends hundreds of concurrent requests with spiky arrival, so you need pooling, batching, queueing, and a rule for a full queue.

API availability. The notebook assumes every call succeeds. Production sees rate limits returning 429, timeouts returning 504, and provider-side degradation where the call succeeds but the answer is worse than yesterday. Retries with backoff, timeouts, and fallback models are all missing.

Security. The notebook assumes a friendly user, because the only user was you. Production users probe for prompt injection, try to read other tenants' data, and abuse the endpoint for free generation. Input validation, tenancy isolation, and per-user rate limits are absent.

Cost. The notebook cost a few dollars over a week. Production has no token caps, no caching, and a bill that scales with traffic and with the length of whatever users paste in. Without a per-request token limit and per-tenant quotas, one user with a large document dominates the spend.

The notebook proved the task is possible, not the system.

> **Say it.** The notebook proved feasibility and nothing else. It assumes clean data, but production sends scanned images, bad encodings and oversized files. It assumes one sequential request, but production sends hundreds concurrently with spikes, so batching and queueing are missing. It assumes the API always answers, but production gets 429s, 504s and degraded models, so retries and fallbacks are missing. It assumes a friendly user, but production gets prompt injection and abuse. And it assumes trivial cost, because there are no token caps and no caching.

---

## The pattern across these

Every question here is the same question in different clothes: turn something vague into something with numbers in it. "Make it smarter" becomes a metric, a dataset, and a threshold. "Accurate and fast" becomes 92 percent F1 at p95 under 800 milliseconds at 100 concurrent requests. A budget becomes a cost per query, and that one number decides the model.

The second pattern is that constraints choose the architecture, not the other way round. A 150 millisecond budget rules out hosted frontier models before you draw a box. A tight cost per query rules them out again. A high-stakes domain rules out ungrounded generation. Find the binding constraint first and most of the design is already decided.

The third pattern is that failure is normal, so it is designed rather than handled. Fallback models, abstention, and queue shedding belong in the drawing.

For interview-format practice on these, see [`../74_ai_engineer_interview_prep/README.md`](../74_ai_engineer_interview_prep/README.md).
