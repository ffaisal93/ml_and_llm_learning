# Topic 76: Links to Read

A parking lot. Things worth reading that have not been read yet, plus the ones that have been read and
turned into chapters, kept so the trail back to the source stays visible.

Add a line, keep it one line, and move it up to **Read** when it is done. The point of a list like this is
that it stays low-friction — a reading list that needs curating stops getting used.

---

## Queue

Nothing outstanding right now. New links land here.

<!--
Format:
- [Title](url) — one line on what it is and why it might be worth the time. `added: YYYY-MM-DD`
-->

---

## Read, and what came of it

- [The Governed Scientific AI Workflow](https://studio.aneeshsathe.com/posts/governed-scientific-ai-workflow/) —
  Aneesh Sathe, Aug 2026. Fourteen responsibilities for wrapping probabilistic AI in deterministic
  governance: typed tools, deterministic validators, provenance, approval gates, budgets as a correctness
  control. → became `75_governed_ai_workflows`.
- [How to Optimize RAG for Latency in Production](https://www.linkedin.com/pulse/how-optimize-rag-latency-production-ayush-singh-vrezc/) —
  Ayush Singh, Aug 2026. Fifty sections on RAG latency; the five-level optimization hierarchy and the
  worked 5s→2s example are the durable parts. → [`74_ai_engineer_interview_prep/RAG_LATENCY_IN_PRODUCTION.md`](../74_ai_engineer_interview_prep/RAG_LATENCY_IN_PRODUCTION.md),
  with measured figures substituted for the illustrative ones.
- [RAG in 2025: From RAG to Context Engineering](https://ragflow.io/blog/rag-review-2025-from-rag-to-context) —
  RAGFlow. The frontier survey — RAPTOR, GraphRAG, PageIndex, ColPali, late interaction. Vendor-published,
  so read with that in mind. → [`39_rag_retrieval_augmented_generation/RAG_TO_CONTEXT_ENGINE.md`](../39_rag_retrieval_augmented_generation/RAG_TO_CONTEXT_ENGINE.md), which
  flags ten unverifiable claims.
- [HimankSehgal/AI-interview-prep](https://github.com/HimankSehgal/AI-interview-prep) — mostly a stub, but
  `microsoft.md` documents a complete five-stage Applied Scientist 2 loop in useful detail. →
  [`74_ai_engineer_interview_prep/THE_LOOPS.md`](../74_ai_engineer_interview_prep/THE_LOOPS.md).

---

## Standing references

Not "to read" so much as "to reread when the topic comes up."

- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) — check the current version
  before citing it; the numbering has changed once already.
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) —
  when an agent is the right call and when a workflow is.
- [Model Context Protocol](https://modelcontextprotocol.io/) — the spec moves; do not answer MCP questions
  from memory.
- [Weaviate ANN benchmarks](https://docs.weaviate.io/weaviate/benchmarks/ann) — one of the few vector
  search benchmarks that publishes hardware, parameters, recall, and percentiles together. The reference
  point for "is retrieval actually my bottleneck."
- [The Tail at Scale](https://blog.acolyer.org/2015/01/15/the-tail-at-scale/) — Dean and Barroso, via the
  morning paper. The fan-out tail math, and still the clearest statement of it.
- [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) —
  use these span and metric names rather than inventing your own.

---

## A note on sources

Two-thirds of the content published on applied AI topics is search-optimized filler with invented
numbers. While researching the latency chapter, one cited "benchmark" turned out to be a 404 on a
staffing site, and another surfaced with a headline figure that appeared nowhere on the actual page.

So the habit worth keeping when adding to this list: note whether the source publishes a measurement
setup. Vendor documentation with stated conditions is usable. An engineering blog from a company that
operates the system is usable. A page with a confident millisecond table and no hardware, region, or
trial count is not a source, however professional it looks.
