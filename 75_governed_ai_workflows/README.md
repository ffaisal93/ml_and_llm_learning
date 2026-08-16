# Topic 75: Governed AI Workflows

What you build around a model when being confidently wrong is expensive.

Most of this repository is about making models work — better retrieval, better evaluation, better
serving. This folder is about the architecture that surrounds one in a domain where the output feeds a
decision somebody is accountable for: clinical, regulatory, financial, legal, scientific. The organizing
idea is a single sentence worth memorizing:

> **Agents can own workflows. They should not own truth.**

## Files in this folder

| File | Purpose |
|---|---|
| `GOVERNED_WORKFLOW_ARCHITECTURE.md` | The fourteen responsibilities — orchestration, typed tools, the four retrieval modes, persistent state, provenance, sandboxing, deterministic validators, permissions, approval gates, budgets, observability, evaluation, versioning — plus the deployment sequence and why any of it belongs in an ML repository. |

## Why it is worth reading even if you never work in a regulated domain

Three of its ideas are just good agent engineering and are underused everywhere:

**Validators run before execution, not after.** A deterministic check that returns the same verdict for
the same input, sitting between the model's proposed tool call and the tool, is a different and stronger
thing than asking a second model whether the first one looked right. Both have a place; only one is
validation.

**Budgets are a correctness control, not only a cost control.** Unbounded search followed by selective
reporting is the researcher-degrees-of-freedom problem that inflates false positives. An agent that
generates a thousand candidates and surfaces the best twelve has p-hacked, whatever you call it. That
reframing — a step cap as an epistemic safeguard rather than a billing safeguard — is not in most agent
documentation.

**Agents sharing a model are not independent verifiers.** Three agents agreeing, when they are the same
model with the same instructions over the same evidence, is one opinion stated three times. Multi-agent
architectures routinely sell correlated votes as consensus.

## The one that changes how you answer interview questions

"Retrieval" is four different things, and collapsing them into "RAG" loses capability: structured
database queries for exact identifiers and curated values; lexical search for accession numbers, error
codes, and rare names; vector search for semantic similarity across wording; and knowledge-graph
traversal for canonical entities and multi-hop typed relationships. Supplying retrieved passages to a
generative model is useful for documents and does not replace an exact database query or an
ontology-constrained graph walk.

Being able to say that distinguishes you immediately in any system-design round where the domain has real
structured data behind it — which is most of them.

## How to use it

Read it once for the architecture, then a second time asking a narrower question: for the system you are
actually working on, which of the fourteen responsibilities does *nothing* currently own? That list is
usually short and usually alarming. State and provenance are the two most commonly missing, and they are
also the two the source insists cannot be retrofitted.

---

## Cross-references

- `65_llm_security` — least privilege, prompt injection, and the trust boundary.
- `74_ai_engineer_interview_prep` — the interview layer, including OWASP for RAG.
- `39_rag_retrieval_augmented_generation` — the mechanics behind the retrieval component.
- `03_evaluation_metrics`, `47_statistical_inference` — calibration and proper scoring rules.
- `69_ai_infrastructure_engineering` — sandboxing and the operational half.
- `76_links_to_read` — where this article was parked before it became a chapter.
