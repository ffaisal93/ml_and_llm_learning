# Distributed systems depth

**Foundation explainer:** Further depth · **Priority:** Important — second pass

Important for infrastructure-heavy interviews; study after the ML core.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f161"></a>

## F161. Explain CAP during a real network partition.

**Start here.** Under the CAP model, a partitioned service cannot guarantee both linearizable behavior and a successful available response to every request at every non-failing node.

**Work it through.**

1. State what counts as consistency and availability in this model.
2. Trace conflicting reads and writes on isolated replicas: a consistency-first design may reject or delay some operations; an availability-first design may accept divergence.
3. Explain reconciliation after communication returns and normal-operation latency tradeoffs separately.

**Small example.** A ledger and a reaction counter can reasonably choose different policies because the consequences of conflicts differ.

**Common trap.** Reducing CAP to “choose any two” without discussing an actual partition or the meaning of guarantees.

**Follow-up.** Why mention PACELC? It highlights latency-versus-consistency choices even when no partition exists.

---

<a id="f162"></a>

## F162. Design a durable work queue with retries and leases.

**Start here.** A lease gives a worker temporary ownership, while durable state lets another worker recover abandoned tasks.

**Work it through.**

1. Store task ids, states, attempt counts and lease expirations atomically with claiming.
2. Require heartbeats or bounded lease renewal, and return expired work to the ready queue under a retry policy.
3. Make task effects idempotent and route permanent failures to a dead-letter path with observable reasons.

**Small example.** If a worker completes a side effect and crashes before acknowledgment, the task can be delivered again.

**Common trap.** Claiming exactly-once effects from an at-least-once queue without an idempotency or transactional mechanism.

**Follow-up.** How do fencing tokens help? They allow downstream systems to reject writes from an expired owner.

---

<a id="f163"></a>

## F163. Handle dependency-aware job scheduling.

**Start here.** A task becomes runnable only when all required predecessors satisfy their completion policy.

**Work it through.**

1. Represent jobs as a directed graph, validate unknown nodes and detect cycles.
2. Track unresolved predecessor counts; enqueue zero-count tasks, then decrement dependents on durable completion.
3. Define failure propagation, retries and cancellation separately from graph traversal.

**Small example.** If C depends on A and B, completion of A alone must not release C.

**Common trap.** Decrementing a dependency twice after duplicate completion messages.

**Follow-up.** What complexity should a static topological pass have? O(V+E) for nodes and dependency edges.

---

<a id="f164"></a>

## F164. Design a read-once secret-sharing service.

**Start here.** The key correctness property is atomic consumption under concurrent requests, not just hiding a URL.

**Work it through.**

1. Create a high-entropy token and store encrypted payload, expiry and access metadata under a minimal retention policy.
2. Atomically transition unused to consumed while retrieving or deleting the payload; define failures between consumption and delivery.
3. Avoid logging secrets, use secure transport and explicit expiration, and explain the tradeoff between strict one-attempt access and retryability.

**Small example.** Two concurrent readers must not both observe an unused record and both receive the secret.

**Common trap.** Promising perfect one-successful-read semantics across arbitrary client/network failures without defining what counts as consumption.

**Follow-up.** How does a GET prefetcher affect design? Link previews can accidentally consume content, so the interaction contract must account for them.

---

<a id="f165"></a>

## F165. Design idempotency for an order or booking API.

**Start here.** Clients can retry a request without accidentally creating duplicate business operations.

**Work it through.**

1. Require an idempotency key scoped to the caller and operation; bind it to a normalized request fingerprint.
2. Atomically store processing state and the final result with the business transaction where feasible.
3. On retry, return the same result or current status; reject a key reused with conflicting parameters and define retention.

**Small example.** A timeout after successful booking should not create a second booking when the client retries the same key.

**Common trap.** Checking the key and inserting it in separate unprotected operations.

**Follow-up.** What if a downstream service is not transactional with your database? Use a durable state machine and reconciliation, not an unsupported exactly-once claim.

---

<a id="f166"></a>

## F166. Design a token-bucket rate limiter.

**Start here.** A token bucket permits controlled bursts while enforcing an average refill rate.

**Work it through.**

1. Track capacity, refill rate, token count and the last update time for each key.
2. On a request, add elapsed_time*rate up to capacity, then atomically consume the required tokens if available.
3. Choose where state lives and define clock, concurrency and degraded-mode behavior across replicas.

**Small example.** A capacity of 10 with refill 2/s permits an initial burst of 10 one-token requests, then about 2/s sustainably.

**Common trap.** Updating shared counters without atomicity or using one global bucket when per-tenant fairness is required.

**Follow-up.** Why might you need both request and token budgets for LLMs? Request sizes can differ dramatically.

---

<a id="f167"></a>

## F167. Design preemptible long-running generation jobs.

**Start here.** Treat a generation request as durable state with progress and versioned intermediate artifacts.

**Work it through.**

1. Record job inputs, priority, resource needs, model/configuration versions and checkpoint capability.
2. Schedule compatible resources, checkpoint before preemption when supported and resume from verified state; otherwise state the recomputation cost.
3. Make final publication idempotent and track partial, failed, canceled and completed outcomes separately.

**Small example.** Preempting a 30-minute job after 29 minutes without a usable checkpoint can waste nearly all of its compute.

**Common trap.** Assuming every model's internal state can be checkpointed and resumed cheaply.

**Follow-up.** How does fairness interact with preemption? High-priority traffic must not cause permanent starvation of long jobs.

---

<a id="f168"></a>

## F168. Design a state-wide temperature ingestion and query system.

**Start here.** Different query patterns often need different materialized views of the same event stream.

**Work it through.**

1. Ingest readings with sensor id, event time, position and value; validate duplicates and late events.
2. Maintain a latest-reading store, a top-K view and spatially aggregated heat-map tiles rather than forcing one layout to serve everything.
3. Define freshness, retention and late-correction rules, with replayable raw events for rebuilding views.

**Small example.** A heat-map tile cache avoids resending every raw sensor point whenever a browser refreshes the map.

**Common trap.** Letting an old out-of-order event overwrite a newer latest-reading record.

**Follow-up.** How do you define hottest sensors? Latest readings, a time-window maximum and a time-window average are different queries.

---

<a id="f169"></a>

## F169. Explain backpressure in a multi-stage data pipeline.

**Start here.** When downstream work is slower, unbounded buffering turns overload into memory exhaustion and extreme latency.

**Work it through.**

1. Measure service rates and queue age at each stage.
2. Use bounded queues and propagate flow control, admission limits or a deliberate drop/degrade policy upstream.
3. Preserve durable work where required and scale the actual bottleneck; retries need budgets so they do not amplify overload.

**Small example.** A producer at 1000 items/s feeding a sustained 800 items/s consumer adds 200 queued items every second.

**Common trap.** Adding retry threads when a downstream service is already saturated.

**Follow-up.** What should an overload response include? A clear retry policy or status, rather than silent loss or indefinite waiting.

---

<a id="f170"></a>

## F170. Design a cache without serving incorrect data.

**Start here.** A cache key and invalidation policy are part of correctness, especially for personalized ML results.

**Work it through.**

1. Include the inputs and versions that affect the result: model, features, prompt, documents and authorization scope as needed.
2. Choose expiry, invalidation and negative-caching policies; protect against stampedes with controlled request coalescing.
3. Measure hit rate, freshness, error behavior and tail latency, keeping a correct uncached path.

**Small example.** Two identical question strings from users with different document permissions must not automatically share a retrieved-answer cache entry.

**Common trap.** Using a short text key that omits the model version or user access boundary.

**Follow-up.** Why can invalidation be preferable to TTL alone? Critical updates may need to take effect sooner than the ordinary expiry window.

---
