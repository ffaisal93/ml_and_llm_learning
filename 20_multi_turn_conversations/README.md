# Topic 20: Multi-Turn Conversation and Long Context

Every LLM problem, compounded by *time*. A single-turn system either works or it doesn't; a
conversation degrades, and it degrades in ways that no single-turn evaluation will ever show you.

> **Start here:** [`MULTI_TURN_DEEP_DIVE.md`](MULTI_TURN_DEEP_DIVE.md), read straight through once. [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md) is
> active-recall questions for afterwards, and [`conversation_py.md`](conversation_py.md) is a minimal working loop.

## What the deep dive covers

**Part 1 — long context.** What actually degrades as context grows, separating three claims that
routinely get conflated: models attending unevenly by position, performance dropping from sheer length
even when the answer is trivially findable, and failure to integrate across distant pieces. The gap
between advertised and *effective* context — the single most useful fact in the area, and one that is
measured in orders of magnitude rather than percentages. Why needle-in-a-haystack is a bad benchmark and
what replaced it. RoPE and why long context is mechanically hard. Attention sinks, KV eviction, and
shipped sparse attention. Then the economics: how the KV cache grows over a conversation, how prefix
caching works at each provider, and why appending is nearly free while editing history is not.

**Part 2 — the system.** Memory architectures and what the current crop of memory products can and
cannot be shown to do. Context engineering — compaction, sub-agent isolation, just-in-time retrieval.
Evaluating a conversation rather than a response. Multi-turn safety, which is a genuinely different
problem from single-turn safety. State, sessions, concurrency, and the production shape. Ends with a
reference architecture walked end to end and the ten questions interviewers actually ask.

## The three findings worth knowing cold

**Multi-turn degradation is a variance collapse, not a capability loss.** Across 15 models and 200k+
simulated conversations, splitting a task across turns costs about 39% — but the decomposition is that
aptitude falls ~16% while *unreliability rises ~112%*. The model still knows how; it just stops doing it
consistently. The mechanism is premature commitment: it guesses an interpretation early, builds on it,
and does not recover. Temperature 0 does not fix it.

**Compaction destroys constraints, not facts.** Summarizing a long session retains only around 17% of
standing session constraints — the "always use British spelling," "never touch prod" instructions that
look like boilerplate to a summarizer and are load-bearing to the user. Recency truncation retains none.

**The memory-layer benchmarks are not trustworthy.** An independent audit of the standard long-conversation
benchmark found 6.4% of the answer key wrong and the standard judge accepting 62.81% of deliberately
wrong answers. The strongest published memory-layer paper shows plain full-context *beating* it on
accuracy. The honest case for a memory layer is latency and cost, not quality — and being able to say
that is worth more in an interview than quoting a vendor number.

## How to use it

Every term is defined at first use. Seventeen **"Saying it out loud"** blocks give you the words in
natural speech — cover the block, say your own version, compare. Evidence is marked throughout, and
where the field is genuinely contested or a number is vendor-published, the text says so. A benchmark
figure quoted as current when it is a year stale is an easy thing to get caught on.

## Next

- `06_llm_inference` — the serving mechanics behind the KV-cache and prefix-caching arithmetic here.
- `39_rag_retrieval_augmented_generation` — the retrieval half of the long-context-versus-RAG question.
- `65_llm_security` — single-turn safety, which this chapter's Part 2 contrasts against.
- `74_ai_engineer_interview_prep` — the interview layer, including agent and context-engineering questions.
