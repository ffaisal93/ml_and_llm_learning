# Governed AI workflows: putting deterministic boundaries around probabilistic components

Most of this repository is about making models work. This chapter is about the architecture you build
*around* a model when being wrong has consequences — when the output feeds a decision someone is
accountable for, rather than a chat window someone reads.

The organizing sentence, from Aneesh Sathe's essay on governed scientific AI workflows, is worth
memorizing because it compresses the whole design stance:

> **Agents can own workflows. They should not own truth.**

That distinction does real work. It says the probabilistic component is allowed to plan, draft, search,
propose, and route — all the things it is good at — and is not allowed to be the thing that decides what
is true, what gets recorded, or what happens next. Those get deterministic machinery and, at the
consequential points, a human.

The setting in the source is scientific research, but almost none of the architecture is
science-specific. Swap "hypothesis" for "case decision" and "assay" for "transaction" and the same
fourteen responsibilities show up in medical, legal, financial, and compliance systems. Read it as the
general answer to "how do you build an agentic system in a domain where being confidently wrong is
expensive."

---

## 1. The mental model shift

The failure mode this architecture exists to prevent is not hallucination in the narrow sense. It is a
system that produces a fluent, plausible, well-cited-looking output that nobody can trace, reproduce, or
disagree with productively — and that a busy expert approves because it looks right and there are forty
more in the queue.

So the shift is from thinking of the system as **a chatbot, or an oracle that combines data** to thinking
of it as **a workflow engine with AI inside it**. Probabilistic code wrapped in a deterministic
framework. Every design question then becomes: which of these two layers should own this responsibility?

That framing is also, conveniently, the thing that separates a strong system-design answer from a weak
one. Asked to design an AI system for a regulated or high-stakes domain, the weak answer describes a
better prompt and a better model. The strong answer describes what the model is *not allowed to do*.

### Three concepts worth being able to define cleanly

**Ontology.** A machine-readable specification of the domain's entities and the relationships they are
permitted to have — `Compound`, `Target`, `Assay`, `Measurement`, and which edges between them are legal.
Standards: OWL 2, and OBO Foundry for coordinating biomedical ontologies. The important nuance is that an
ontology is a *maintained model* requiring human review, not an immutable prescription handed down from
somewhere.

**Knowledge graph.** Nodes are entities, typed edges are relationships — `inhibits`, `measured_in`,
`derived_from`, `contradicts`. Every assertion carries provenance: who asserted it, from what source, by
what method, when. Worth stating clearly in an interview because people conflate them: **a knowledge
graph represents knowledge; a graph neural network learns from graph-structured data.** Different
objects, different purposes.

**Neurosymbolic.** The composition of the two: LLMs extract candidate relations, ontologies constrain
which entity and relation types are permitted, the graph stores assertions with evidence, and
deterministic validators check proposed updates before they land. The neural layer answers *"what might
this mean?"*; the symbolic layer answers *"does this conform to what the system permits?"*

The honest caveat, which the source makes and which is worth repeating: neurosymbolic systems are not
automatically correct. The architectural promise is narrower than the marketing — whether your
constraints actually catch the scientifically important errors is an empirical question you have to
evaluate in the setting you deploy in.

---

## 2. The fourteen responsibilities

These are logical responsibilities, not products. Several can live in one platform, and reading them as a
procurement checklist is a misreading. I have grouped them by what they are for.

### Structuring the work

**1. Orchestrator and task graph.** Turn the objective into explicit interdependent steps — a DAG, where
nodes are tasks and edges are prerequisite outputs. This buys inspectability (declared inputs, outputs,
failure states, ownership) and parallelism (literature search and database query run concurrently, then
synchronize at a synthesis step). Established engines exist and are worth reusing: Nextflow, Snakemake,
CWL.

The guidance that matters: **draw the DAG before choosing an agent framework**, define success, failure,
and escalation for *every* node, and — the important one — **do not let the planner create unbounded
subgraphs.** A planner that can spawn arbitrary sub-planners has no worst case.

**2. Specialized capabilities.** Narrow roles with distinct tools, prompts, data access, and evaluation
metrics, because literature review, causal analysis, cheminformatics, and experimental design have
genuinely different evidence standards and failure modes. Google's Co-Scientist separates generation,
reflection, ranking, evolution, proximity, and meta-review; FutureHouse's Robin combines literature and
data-analysis agents iteratively.

The trap, and it is a good one to be able to name: **outputs from agents that share a model,
instructions, or evidence are not independent verification.** Three agents agreeing when they are the
same model with the same context is one opinion stated three times. Multi-agent architectures routinely
sell correlated votes as consensus.

Also: evaluate the *handoffs* between capabilities as carefully as the capabilities themselves. That is
where information gets silently dropped or reshaped.

### Constraining the interface

**3. Typed tool interfaces.** Formal schemas on inputs and outputs, so malformed calls are rejected
before they reach a database, instrument, or LIMS. A dose-calculation tool requires a positive numerical
amount, an allowed mass unit, a body weight with a unit, and a species identifier — and returns a
structured object rather than interpretable prose.

Two specifics worth carrying. Handle units with an explicit unit library (Pint, QUDT) rather than
encoding them in field names or hoping the prose says "mg". And **validate at both ingress and egress**.
The division of labor is the memorable part: **the LLM selects a tool and proposes arguments; ordinary
software determines whether those arguments are valid.**

**4. Retrieval layer — not "a RAG."** This is the paragraph most worth internalizing if you have been
thinking of retrieval as one thing. Four complementary modes, and collapsing them loses capability:
structured database queries for exact identifiers and curated values; lexical retrieval for exact terms,
accession numbers, and rare names; vector retrieval for semantic similarity across different wording; and
knowledge-graph queries for canonical entities, typed relationships, and multi-hop structure.

> "Retrieval-augmented generation usually means supplying retrieved passages to a generative model. That
> is useful for documents, but it does not replace exact database query or ontology-constrained graph
> traversal."

Operationally: hybrid lexical + vector for text with reranking over small candidate sets, passage-level
citations preserved, entities resolved to stable identifiers *before* joining across sources, and a
record of the query, retrieval date, index version, filters, and document IDs. That last list is what
makes a result reconstructable six months later, and it is exactly what a chat transcript does not
capture.

### Holding state and truth

**5. Persistent scientific state.** An explicit record of the program's current beliefs — hypotheses,
their status, supporting and contradicting evidence, open questions, decisions, planned experiments.
"Hypothesis H17 is `proposed`, supported by two observational studies, contradicted by one perturbation
experiment, awaiting replication," with defined transitions like `proposed → under_test → supported`.

The line to remember: **appending corrective prose to a chat transcript is not state tracking.** If the
only place the system's beliefs live is a conversation, then the system has no beliefs, it has a
transcript. Requirements are typed entities, status transitions, stable identifiers, evidence *linked*
rather than copied, human decision records, and supersession tracking.

**6. Evidence and provenance graph.** Searchable ancestry for every claim, separating *"why should I
believe this?"* from *"how was this made?"* The W3C PROV model gives you the vocabulary — entities,
activities, agents, related by `used`, `wasGeneratedBy`, `wasAttributedTo`. For AI-assisted work, a claim
should link to retrieved passages, tool outputs and transformations, model and prompt versions, validator
verdicts, and human approvals.

Two instructions that are easy to state and hard to follow. **Create provenance as the workflow runs,
not during manuscript cleanup** — retrofitted provenance is reconstruction, which is to say fiction. And
**preserve contradictions; do not average away disagreement between studies with different designs.** A
system that silently reconciles conflicting sources has destroyed the most informative thing it found.

Also: treat citations produced from model memory as unverified until the identifier resolves *and* the
cited source actually supports the claim. Both halves. A resolvable DOI that says something else is the
more embarrassing failure.

### Bounding what can happen

**7. Code sandbox.** Isolated execution with real constraints — limits on filesystem, network,
credentials, memory, CPU, and wall clock; read-only input mounts; a designated artifact directory;
network denied by default; short-lived credentials injected only for approved tasks. What does *not*
qualify: a notebook kernel with unrestricted network access and the user's inherited credentials. Capture
stdout, errors, package locks, and environment metadata, because reproducibility is the chain of code,
data, dependencies, and execution order — not the final notebook.

**8. Deterministic validators.** Same inputs and rules produce the same verdict, every time, inside the
loop, *before* malformed output propagates. They check schema conformance, dimensional consistency,
identifier format, permissions, allowed state transitions, ontology constraints, and physical or
mathematical invariants — probabilities in [0, 1], non-negative mass, required control groups, values
inside instrument-validated ranges. SHACL does this for graph data.

The sentence to steal: **"asking another LLM whether the first LLM's answer looks correct is critique,
not deterministic validation."** LLM-as-judge has a real role, and it is not this one. Validation must
happen before tool execution, before a claim is promoted, and before anything enters shared state — and
it should distinguish hard rejection from a warning that needs human judgment, returning machine-readable
failures either way.

**9. Identity and permissions.** Least privilege, applied to agents. The literature agent reads approved
indexes and cannot write to the LIMS. The analysis agent runs code over de-identified data and cannot
export row-level records. A workflow can draft a purchase request without authority to submit it.
Mechanically: separate service identities, short-lived credentials, explicit allow-lists, action-level
authorization, tool calls bound to the initiating user and workflow, and denied actions logged alongside
successful ones.

The security principle underneath, which is the same one that governs prompt injection everywhere:
**instructions inside a paper, web page, or email must not grant new capabilities.** Retrieved text and
uploads are untrusted data, always.

**10. Human approval gates.** Not "a human in the loop" as a slogan, but a specification of where the
system stops, what the reviewer sees, and what approval authorizes.

Place gates by a two-dimensional map: **consequence if wrong** against **difficulty of reversal.**
Retrieving papers, running validated queries, and drafting hypotheses are low-consequence and reversible
— let them proceed with logging and later review. Ordering experiments, promoting a finding to confirmed,
and publishing an external claim need synchronous approval. A gate should present the proposed action,
the supporting evidence, the uncertainty, the validator results, the alternatives, the resources
committed, and what happens after approval.

And the failure mode that makes most approval gates decorative: **if reviewers get more cases than they
can inspect, the gate is a rubber stamp.** This is documented human-automation research (Parasuraman and
Riley, 1997), not a hunch — nominal oversight does not produce effective control. Require explicit
approve / reject / modify decisions, record who and when, start conservative, and expand autonomy only
when measured performance supports it.

**11. Retry, timeout, and budget controls.** Bound retries per step with backoff and an explicit terminal
state; bound wall clock per tool and per workflow; bound tokens, accelerator time, API spend, and
laboratory resources; bound recursion depth and spawned task count; bound how many retrieved documents
and candidates a human is asked to review.

The scientific argument for budgets is sharper than the engineering one and it is the part worth quoting:
**unbounded search followed by selective reporting is the researcher-degrees-of-freedom problem** that
inflates false positives (Simmons, Nelson and Simonsohn, 2011). An agent that generates a thousand
hypotheses and surfaces the twelve that look best has p-hacked, whatever else you call it. On exhaustion,
stop visibly, preserve partial results, and escalate for reauthorization rather than silently raising the
limit.

### Knowing whether it works

**12. Observability.** Per run: the task graph, model calls, retrieved items, tool arguments and results,
validator verdicts, state transitions, latency and compute, errors. Distinct from provenance —
**operational traces diagnose behavior across many runs; provenance establishes the lineage of one
artifact.** You need both and they answer different questions.

A specific and unusually well-grounded warning: **do not assume storing hidden chain-of-thought is
necessary, available, or a faithful account of what actually caused the answer** (Lanham et al., 2023).
Record actionable summaries, tool traces, and evidence links — externally checkable audit objects rather
than the model's narration of itself.

Build views for the domain expert, not only for the SRE. *"Which sources changed this ranking?"* matters
as much as *"which request timed out?"* And protect the traces, because they contain everything.

**13. Evaluation harnesses.** Four levels, and most teams stop at the first:

1. **Component** — schema adherence, extraction accuracy, retrieval recall, tool-call success.
2. **Scientific task** — representative in-domain problems, with temporal and external validation.
3. **Human-system** — whether experts *with* the system make better decisions than without it.
4. **Prospective** — whether recommendations survive new experiments and later evidence.

Level 3 is the one that answers the question anyone funding this actually asked, and almost nobody runs
it.

Three sampling traps worth knowing by name. **Random train/test splits are often inadequate** because
records from the same patient or time period leak across splits (Kapoor and Narayanan, 2023). **In
ligand-based benchmarks, structural similarity rewards memorization** rather than generalization (Wallach
and Heifets, 2018). And **accuracy or ranking performance does not imply calibrated probabilities** (Guo
et al., 2017) — if the system reports confidence, measure calibration and discrimination separately, with
proper scoring rules (log score, Brier) that reward honest forecasts (Gneiting and Raftery, 2007).

Version test sets and rubrics, include negative controls and adversarial cases and known invariants,
compare against simple baselines, stratify by subgroup, and rerun whenever the model, prompts, tools,
retrieval corpus, or policies change. **A larger LLM is not automatically an improved scientific system.**

**14. Versioning and reproducibility.** Every result should carry an immutable manifest: source data and
retrieval snapshot, ontology and graph releases, code and workflow commit, container digest, model
provider and version, system and task prompts, tool and validator versions, parameters and seeds,
hardware metadata, approvals and manual edits.

Hosted models make bit-for-bit repetition impossible — you control neither the model artifact nor the
inference environment. So the practical targets are **reconstructability** (knowing exactly what was run)
and **robust reproducibility** (conclusions stable across permissible reruns). For nondeterministic
components, run repeated trials and define semantic or statistical equivalence rather than identical
prose. Model cards and dataset datasheets cover intended use, limitations, collection, and maintenance.

---

## 3. The loop

Compressed, the whole thing runs as: scientist states the question and constraints → planner decomposes
→ ontology defines legal terms and relationships → graph, documents, and databases supply evidence →
models and domain software compute → deterministic validators reject constraint violations → the
provenance graph binds claims to sources → the system presents a recommendation *with uncertainty*, not
a conclusion → **the human decides** → the result becomes new evidence that updates the record.

Note where the human sits: not reading the output at the end, but as the decision point the architecture
is built around. The source enumerates six roles the expert keeps — principal (sets purpose and
acceptable risk), domain authority (governs definitions and evidence standards), experimentalist (decides
which uncertainty is worth resolving), critic (spots the formally valid answer that is scientifically
nonsense), accountable decision-maker, and learner (updates both the science and the workflow).

The fourth one is the one software cannot do at all, and it is the reason the whole architecture keeps a
person in it.

---

## 4. How you would actually deploy this

The failure mode here is real and worth naming before the sequence: **building the complete platform
before testing one real task is a reliable way to create infrastructure that people route around.**

1. Write the **context of use** — user, decision, inputs, outputs, foreseeable error, consequence.
2. Define the **human/software contract** — what the expert decides versus what software recommends,
   drafts, or executes.
3. Pick **one bounded, reversible, read-only task.**
4. Build the task graph and typed tools; reuse validated domain software rather than reimplementing it.
5. Ground it — authoritative databases, document retrieval, stable identifiers, the minimum useful
   ontology.
6. **Add state and provenance at the start.** Retrofitting these does not work.
7. Place deterministic boundaries and human gates derived from failure consequences.
8. Build evaluation *before* expanding autonomy, and compare human-only against human-plus-AI.
9. Observe, version, and bound every run; treat model, prompt, data, ontology, and policy changes as
   system changes.
10. Close the loop — feed confirmed, contradicted, and inconclusive outcomes back into state and eval.

And redefine success. Not hypotheses generated, not minutes saved drafting, but **decision yield**:
better-supported decisions and more informative experiments per unit of scarce expert time and
laboratory resource. That reframing is the most quietly useful thing in the essay, because it is the
metric under which most impressive-looking agentic systems score badly.

Rigor should be **proportional to context of use** — what the system does, who relies on it, what happens
if it is wrong — which is the same stance as the NIST AI Risk Management Framework and the FDA's draft
risk-based credibility framework for AI supporting drug-development decisions.

---

## 5. Why this is in an ML repository

Three reasons.

**It is the answer to a whole class of system-design question.** "Design an AI system for clinical
decision support / regulatory review / financial compliance / legal discovery" is a common interview
prompt and a common real project, and the expected answer is not a better RAG pipeline. It is this: what
does the model own, what does deterministic software own, where does a human decide, and how would you
prove afterward what happened. Candidates who answer with prompt engineering lose to candidates who
answer with boundaries.

**Several of these ideas generalize past the regulated setting.** Typed tool interfaces, validators
before execution rather than after, budgets as a correctness control rather than a cost control,
provenance written during the run, and the observation that agents sharing a model are not independent
verifiers — all of these are just good agent engineering, and all of them are underused in ordinary
products.

**It names a failure mode the rest of the field is quiet about.** Unbounded generation plus selective
reporting is p-hacking with extra steps, and it is what a large fraction of "the agent generated 500
ideas" demos are actually doing. Having the vocabulary for that — researcher degrees of freedom,
automation bias, rubber-stamp gates — is worth more than another benchmark number.

---

## 6. Where to go deeper in this repository

- `65_llm_security` — least privilege, prompt injection, and the trust boundary, which components 9 and
  10 depend on.
- `74_ai_engineer_interview_prep` — the interview layer, including OWASP for RAG and the diagnostic
  questions.
- `39_rag_retrieval_augmented_generation` — the retrieval mechanics behind component 4, and the reason
  "just use RAG" is an incomplete answer.
- `03_evaluation_metrics` and `47_statistical_inference` — calibration, proper scoring rules, and the
  sampling traps in component 13.
- `69_ai_infrastructure_engineering` — sandboxing, budgets, and the operational half.

---

## Source

Aneesh Sathe, ["The Governed Scientific AI Workflow"](https://studio.aneeshsathe.com/posts/governed-scientific-ai-workflow/),
12 August 2026, CC BY 4.0. The fourteen components, the "agents can own workflows, they should not own
truth" framing, the consequence-versus-reversibility gate map, the decision-yield metric, and the
literature citations are his. The commentary on which parts generalize beyond science, and the
interview framing in section 5, are mine.

Standards and papers referenced throughout: OWL 2, OBO Foundry, SHACL, W3C PROV, RO-Crate, FAIR, CWL,
NIST AI RMF, FDA draft guidance on AI in regulatory decision-making; Simmons/Nelson/Simonsohn (2011) on
researcher degrees of freedom, Parasuraman and Riley (1997) on automation bias, Lanham et al. (2023) on
chain-of-thought faithfulness, Guo et al. (2017) on calibration, Kapoor and Narayanan (2023) on leakage,
Wallach and Heifets (2018) on ligand-benchmark memorization, Gneiting and Raftery (2007) on proper
scoring rules.
