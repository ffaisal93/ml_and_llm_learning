# The interview loop: what actually happens

Every other chapter in this guide is about *content* — what attention does, why your retriever underperforms, how to think about variance. This one is about *process*: the shape of the hiring pipeline you have to walk through before anyone cares what you know.

This matters more than people expect. The most common way a strong candidate fails an AI/ML loop is not "did not know enough." It is one of these:

- They prepared for a different role's loop than the one they were actually in.
- They spent six weeks on LeetCode and got knocked out by a statistics round they did not know existed.
- They gave a technically correct answer in a register the interviewer was not grading.
- They treated the hiring manager round as a formality.

None of those are knowledge failures. They are process failures, and process is learnable.

### A note on evidence

Hiring processes are badly documented on purpose. Companies publish vague guidance; candidates publish specific anecdotes; content farms publish confident fabrications built from both. Throughout this chapter I mark evidence level explicitly:

| Marker | Meaning |
|---|---|
| **[official]** | The company publishes this on its own careers site |
| **[practitioner]** | A working interviewer or hiring manager describing what they do |
| **[one report]** | A single named candidate's documented experience — a data point, not a process |
| **[aggregated]** | Crowd-sourced across many reports; directionally useful, individually unreliable |
| **[my read]** | My inference from patterns, offered as opinion |

Where I have nothing, I say so. **Do not extrapolate a company's process from a table you found online, including this one.** Ask your recruiter. They will usually tell you the round structure if you ask directly, and asking makes you look prepared rather than presumptuous.

---

## 1. The roles are not the same job

"AI/ML role" is not a category with a shared interview. It is at least five distinct jobs that share some vocabulary. Candidates conflate them constantly. Interviewers never do — they are grading against a specific rubric for a specific ladder.

The single highest-leverage thing you can do before preparing is figure out **which of these you are interviewing for**, because the same answer scores differently in each.

### The taxonomy

**AI Engineer / LLM Engineer / GenAI Engineer.** Builds products on top of models someone else trained. RAG pipelines, agents, tool use, prompt and context engineering, evaluation harnesses, latency and cost management, integration with an actual product. The distinguishing feature: you almost never train anything. You compose, evaluate, and operate.

This is the newest of the five and the least standardized. A community field guide cataloguing AI engineering take-homes found that RAG systems account for over 40% of assignments and agentic tool-calling systems for around 30% **[practitioner]** — which tells you where the bar sits. ([alexeygrigorev/ai-engineering-field-guide](https://github.com/alexeygrigorev/ai-engineering-field-guide/blob/main/interview/questions/06-home-assignments.md))

**ML Engineer.** Trains, serves, and operates models. Feature pipelines, training infrastructure, distributed training, serving latency, monitoring, retraining, drift. Closer to a backend/infra engineer who understands modeling than to a scientist who can code. The loop is heavier on systems and lighter on statistics than people expect.

Note that the title is unstable across companies. One widely used ML system design curriculum opens by conceding that "ML Engineer" roles lack standardization and that the interview splits into at least four distinct types — applied ML system design, ML infrastructure design, AI/ML research, and AI/ML research engineering **[practitioner]**. ([Hello Interview](https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/introduction))

**Applied Scientist.** Research-adjacent but shipping. Expected to read papers fluently, sometimes publish, and translate literature into production systems. This role carries the widest loop of the five: you can face DSA, statistics, ML depth, *and* a research discussion in the same week. Amazon publishes an Applied Scientist prep page describing a loop that assesses technical topics, problem-solving, coding, breadth *and* depth of knowledge, and technical presentation skills, plus behavioral **[official]**. ([amazon.jobs](https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep))

**Research Scientist / Research Engineer.** Produces novel methods. Publications are the primary currency for RS; implementation ability at scale is the primary currency for RE. The line between them has blurred substantially — historically RS (usually PhDs) formulated architectures and RE implemented them, but at frontier labs the roles now overlap heavily **[aggregated]**. ([Sundeep Teki](https://www.sundeepteki.org/advice/the-ultimate-ai-research-engineer-interview-guide-cracking-openai-anthropic-google-deepmind-top-ai-labs))

One documented European research job hunt across Google Brain, FAIR, DeepMind, MSR, Amazon, and Apple describes a consistent shape: one or two hour-long phone screens, then a full-day onsite built around a PhD research presentation plus roughly one-hour individual interviews with team members and researchers from adjacent groups **[one report]**. ([generalizederror](https://generalizederror.github.io/My-Machine-Learning-Research-Jobhunt/))

**Data Scientist.** Analytics, experimentation, causal inference, metric design. At product companies this is often *less* about modeling than candidates assume and much more about product sense and experimental rigor. Crowd-sourced reports of Meta's DS loop consistently describe SQL, product metrics and experimentation cases, and analytical execution, with "product-thinking focus rather than advanced statistical theory" **[aggregated]** — and the source is explicit that the round names come from candidate reports, not Meta's official terminology. ([Interview Query](https://www.interviewquery.com/interview-guides/facebook-data-scientist))

### What each loop actually tests

| Role | Day job | Loop usually includes | Rarely includes | Background that gets you in |
|---|---|---|---|---|
| AI / LLM Engineer | Ship features on top of foundation models | Practical coding, RAG/agent design, **evaluation**, product sense, behavioral | Statistics theory, DSA hards, training infra | Strong SWE + shipped LLM products |
| ML Engineer | Train, serve, operate models | DSA, ML breadth, **ML system design**, coding, behavioral | Paper discussion, causal inference | SWE + production ML ownership |
| Applied Scientist | Research → production | DSA, **statistics**, ML depth, paper/project deep dive, SQL, behavioral | Pure infra design | MS/PhD + publications + shipping |
| Research Scientist / Engineer | Novel methods | **Research talk**, paper defense, ML theory/math, implementation coding | SQL, product metrics | PhD + top-venue publications |
| Data Scientist | Analytics, experiments, causality | **SQL**, product metrics, experiment design, stats, case analysis | ML system design, DSA hards | Stats/econ/quant background + product fluency |

### The same question, graded five ways

This is the part worth internalizing. Take one question: **"How would you evaluate a RAG system?"**

- **AI Engineer.** They want an eval harness. Golden dataset, retrieval metrics (recall@k, MRR), generation metrics (faithfulness, answer relevance, context relevance), offline regression suite plus online metrics, and a plan for what to do when they disagree. The best answer names the failure modes you have personally seen in production. Score is highest for concreteness and operational awareness. One practitioner write-up of GenAI loops calls the evaluation round the round that "trips most candidates," precisely because traditional ML prep does not cover it **[aggregated]**. ([techinterview.org](https://www.techinterview.org/post/3233476396/what-genai-engineer-interviews-test/))
- **ML Engineer.** They want the same thing plus the serving story: how the eval runs in CI, how you detect drift after deploy, what the retraining/reindexing trigger is, cost per query.
- **Applied Scientist.** They want the experimental design. What is your hypothesis, what is the control, how many annotations do you need for the difference you care about to be detectable, how do you handle annotator disagreement.
- **Research Scientist.** They want to know whether the benchmark is *valid*. Is recall@k measuring what you think? What is the confound? Has anyone shown this metric correlates with downstream utility? Expect to be pushed toward "the standard evaluation is wrong, what would you build."
- **Data Scientist.** They want the online experiment. What is the primary metric, what are the guardrails, unit of randomization, expected effect size, sample size, how long you run it, what you do if the primary is flat but a secondary moves.

Five correct answers. Give the wrong one and you sound like you do not understand the role.

**[my read]** If you are unsure which loop you are in, the fastest tell is what the recruiter schedules. A statistics round means Applied Scientist or DS. A research talk means RS/RE. An evaluation-focused design round means AI Engineer. A distributed-training design round means MLE.

---

## 2. The standard pipeline

Almost every AI/ML process is some subset of the same seven stages. Companies differ in which they include, not in what they are.

### Stage by stage

**1. Recruiter screen (20–30 min).** Filtering for: does your background plausibly match the level and role, are you actually available, are your compensation expectations in range, do you have competing processes. Almost nobody fails on technical grounds here — people fail by being vague about what they have built or by anchoring compensation badly.

*Do here:* ask what the loop consists of. Ask what level. Ask about team match. Recruiters answer these questions; candidates rarely ask them.

**2. Technical screen (45–60 min).** One or two rounds. Usually coding, sometimes coding plus ML fundamentals. Amazon states one to two 60-minute technical phone screens with senior team leaders, each covering science competencies, technical knowledge, *and* Leadership Principles **[official]**. Meta's official guidance describes a 45-minute screen: 5 minutes intro, 35 minutes of coding covering two problems, 5 minutes for your questions **[official]**. ([Meta Careers](https://www.metacareers.com/life/preparing-for-your-software-engineering-interview-at-meta))

*Where people fail:* not finishing, or finishing silently. The screen is short and the signal is noisy, so interviewers lean on how legible your thinking was.

**3. Take-home or online assessment.** Not universal. OpenAI describes a "skills-based assessment" stage that "varies by team and may include pair coding interviews, take-home projects, technical tests" **[official]** ([OpenAI interview guide](https://openai.com/interview-guide/)). Anthropic describes live coding in Colab and CodeSignal where you may reference materials, with the expectation that you show "comfort with basic syntax and standard libraries" **[official]** ([Anthropic careers](https://www.anthropic.com/careers)).

Section 6 covers take-homes in detail.

**4. The onsite / virtual loop.** The core. Typically 4–6 rounds over one or two days. OpenAI publishes "4–6 hours of final interviews with 4–6 people over 1–2 days," virtual or at their SF office **[official]**. Amazon publishes a four-interview loop of 55 minutes each **[official]**. Meta publishes three 45-minute rounds — coding, design, behavioral **[official]**. Microsoft's general careers guidance says "typically 2–4 conversations lasting up to an hour each" **[official]** ([Microsoft Careers](https://careers.microsoft.com/v2/global/en/hiring-tips.html)) — notably vaguer than the others, and notably at odds with the five-stage Applied Scientist loop documented in Section 4, which is a good reminder that official pages describe the median, not your loop.

**5. Hiring manager round.** Sometimes inside the loop, sometimes separate. Assessing: do I want this person on my team, can they operate at the level, will they be a maintenance burden. See Section 7.

**6. Team match.** The stage nobody prepares for. At companies that hire centrally (Google historically, Meta at some periods), passing the loop gets you approved, not hired — you then have to match with a team that wants you. Reports describe candidates passing the full loop and still ending with no offer because no match happened **[aggregated]**. Amazon, by contrast, hires against a specific opening, so passing the loop generally means the offer is yours **[aggregated]**. ([fyjump](https://www.fyjump.com/post/the-swe-interview-process-in-2026-top-11-big-tech-companies-broken-down))

*Ask your recruiter in the first call whether the role is team-specific or central.* It changes both your timeline expectations and your leverage.

**7. Offer.** Amazon publishes a "Candidate Promise" of decisions within 2 business days after phone screens and 5 business days after final interviews **[official]** ([About Amazon](https://www.aboutamazon.com/news/workplace/amazon-interview-process-phone-screens-loops)). OpenAI says roughly one week after final interviews, with references possibly requested at that stage **[official]**. Amazon's overall process is described as typically 3–6 weeks **[official]**.

### Timelines at a glance

| Stage | Typical duration | Typical wait after |
|---|---|---|
| Recruiter screen | 20–30 min | days |
| Technical screen | 45–60 min | 2 days to 1 week |
| Take-home | stated 2–4h of work, 2–7 day window | 1–2 weeks |
| Onsite loop | 3–6 hours, 1–2 days | 5 days to 2 weeks |
| Team match | n/a | days to months |
| **Total** | | **3–8 weeks typical** |

### Elimination vs non-elimination

Most pipelines are **elimination** models: fail a round, the process stops, remaining rounds are cancelled. This is why the technical screen matters disproportionately — it is the highest-volume filter and the one with the least context about you.

Some are **non-elimination**: all rounds run regardless of how any individual round went, and the decision is made holistically afterward by a committee or the hiring manager reading all the feedback. The Microsoft Applied Scientist 2 experience in Section 4 documents exactly this — "4 Loop Rounds (all 4 happen, it's not elimination based)" **[one report]**. Google's hiring committee model is structurally similar: interviewers write feedback, a committee that never met you reads the packet **[aggregated]**.

**Why this changes your strategy.** Under elimination, every round is a gate and a bad round is fatal — so play conservatively, secure the base case, do not gamble. Under non-elimination, a single weak round is survivable if the others are strong, and the committee is reading for a *pattern*. Two consequences:

1. **A bad round is not over.** Candidates who bomb round 2 and then mentally check out for rounds 3 and 4 convert a recoverable loop into a rejection. This is the most avoidable failure mode in this chapter.
2. **Consistency is scored.** Under a committee, "strong in three, catastrophic in one" reads worse than "solid in all four," because the committee is looking for risk. Under elimination with a single decision-maker who watched you recover, the spiky profile fares better.

**[my read]** Ask which model you are in. "Is this a holistic review across all rounds, or is each round a gate?" is a completely normal recruiter question, and the answer tells you how hard to push on the ambitious version of an answer versus the safe one.

---

## 3. Round types, one by one

### 3.1 Coding / DSA

**What it looks like.** 45–60 minutes, one to three problems, shared editor. Amazon and Meta both publish that coding rounds are part of the loop for scientist and engineer roles **[official]**. In ML-adjacent roles the difficulty band skews easier than pure SWE loops but the ceiling is real — the Microsoft AS2 report documents an Easy plus two LeetCode Hards in a single round **[one report]**.

**What it is really testing.** Not whether you have seen the problem. Whether you can take an underspecified problem, state assumptions, produce a working baseline, improve it deliberately, and verify it. The Microsoft report is explicit: the focus was "not just on getting the correct answer — they wanted to see how you build your approach from brute force to optimal and how you dry run on test cases" **[one report]**.

**How to prepare.** Patterns over volume: two pointers, sliding window, hashing, binary search, BFS/DFS, heaps, DP on strings and intervals. For ML roles, add the from-scratch implementations that show up constantly — k-means, k-NN, a decision tree split, softmax, attention, a training loop. The research jobhunt account lists exactly these as a recurring round type **[one report]**.

**Failure modes.**

- Silence. The interviewer cannot grade thinking they cannot hear.
- Jumping to the optimal solution you memorized, then being unable to justify it.
- Never running the code mentally on an example. Dry-running is graded.
- Writing code before agreeing on the problem. Two minutes of clarification saves twenty.
- Panicking at a Hard. State the brute force, get it working, *then* optimize. A working $O(n^2)$ with a clear articulation of why $O(n \log n)$ is possible beats a broken optimal attempt.

### 3.2 ML depth

**What it looks like.** Pick one thing you claim to know and go down until you hit bedrock. Usually anchored on your résumé. "You used a transformer — write the attention formula. Why the scaling factor? What breaks without it?"

**What it is really testing.** Whether your knowledge is derived or memorized. Memorized knowledge has a floor; derived knowledge does not. The Microsoft report's summary of Microsoft's philosophy — "intuition over exact knowledge" — is the explicit version of what most depth rounds are implicitly doing **[one report]**.

**How to prepare.** For every architecture and technique on your résumé, be able to (a) write the core math, (b) explain *why* each piece exists, (c) name what fails without it, (d) name a credible alternative and the tradeoff. Depth questions bottom out fast on things like: why divide by $\sqrt{d_k}$, batch norm vs layer norm and why transformers use the latter, what causes vanishing and exploding gradients and what actually fixes them, why multiple attention heads.

**Failure modes.**

- Claiming a technique on your résumé you cannot derive. Interviewers target résumé claims preferentially. Remove anything you cannot defend for ten minutes.
- Answering the "what" when asked the "why."
- Bluffing. Depth rounds are *designed* to reach your edge; reaching it is expected. Pretending you have not reached it is the failure.

### 3.3 ML breadth

**What it looks like.** Rapid coverage across areas — classical ML, deep learning, evaluation, data, sometimes NLP/CV specifics. Amazon's published Applied Scientist criteria explicitly list both "depth" and "breadth" of knowledge as assessed competencies **[official]**.

**What it is really testing.** Whether you have a map of the field, and whether you know where the edges of your map are. Also: can you pick the right tool rather than the fanciest one.

**How to prepare.** Be able to answer "when would you *not* use a neural network," "how do you handle class imbalance and what are the three approaches' failure modes," "precision vs recall for this business context," "what does regularization actually do." Breadth rounds reward the person who answers with a decision procedure rather than a fact.

**Failure modes.** Reaching for deep learning on every problem. Reciting definitions without context. Not saying "I have not worked with that, here is how I would reason about it" — which is a perfectly good breadth answer.

### 3.4 ML system design

**What it looks like.** 45–60 minutes on an open problem: "design the ranking system for X," "design fraud detection," "design a support assistant." No code. Whiteboard or shared doc. Increasingly common at mid-level and effectively standard for senior **[practitioner]**.

**What it is really testing.** Five dimensions, per one widely used curriculum **[practitioner]**: problem navigation (turning a vague business goal into a measurable ML problem, including deciding whether ML is even appropriate), data and features (including label design and leakage), model design and tradeoffs, integration and evaluation in production, and communication. ([Hello Interview](https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/introduction))

That first dimension is where most of the score lives and where most candidates spend the least time.

**How to prepare.** Have a framework and use it out loud:

1. **Clarify and scope.** Who is the user, what is the business metric, what is the volume, what is the latency budget, what data exists.
2. **Frame as an ML problem.** What is the prediction, what is the label, how is the label obtained, what is the training/serving unit. Say whether ML is the right tool.
3. **Metrics.** Offline proxy metric, online business metric, guardrails. State the relationship between them and where it breaks.
4. **Data and features.** Sources, volume, leakage risks, cold start, feedback loops.
5. **Model.** Start with a baseline you would actually ship. Then the upgrade path and the cost of each step.
6. **Serving.** Latency, throughput, batch vs real-time, caching, cost per request.
7. **Evaluation and monitoring.** Online experiment design, drift, retraining trigger, rollback.
8. **Failure modes.** What goes wrong, how you detect it.

For GenAI-flavored versions of this round, the same practitioner write-up notes strong candidates start pragmatic — pgvector before Pinecone — and state cost-versus-recall tradeoffs explicitly rather than over-engineering **[aggregated]**.

**Failure modes.**

- Going straight to model architecture. The architecture is maybe 15% of the score.
- No metric, or a metric with no connection to the business outcome.
- Ignoring data collection and labeling entirely.
- No monitoring story. "There is a gulf between a good idea in a notebook and what works in production" is the thing this round exists to test **[practitioner]**.
- Over-engineering. Proposing a distributed training cluster for a problem with 50k rows is a negative signal, not a positive one.

### 3.5 Research / paper discussion

This round decides Applied Scientist and Research loops, and it is the worst-covered round in the prep literature. So: specifics.

**What it looks like.** Two variants, sometimes both.

*Variant A — your own work.* The interviewer has your paper (or your CV and picks one). They ask you to walk through it and then push. The Microsoft AS2 screening round is a clean documented example: the interviewer took a paper on distilling Graph Transformers into MLPs and went into "the math, the motivation, and the results," asking what a GNN is, what a Graph Transformer is, what the current problems are (the $O(N^2)$ complexity), what the motivation for distillation was, the approach and results, and why an MLP specifically — inference speed and deployment simplicity **[one report]**.

*Variant B — someone else's work.* You are given a paper in advance, or asked about a well-known one, and asked to critique it. Less common but used at research-heavy labs.

*Variant C — the research talk.* At labs hiring PhDs, a 30–45 minute presentation of your thesis work to the team, followed by questions. The documented European lab jobhunt describes exactly this as the anchor of the onsite day, with questioning ranging from shallow to detailed technical proofs **[one report]**.

**What it is really testing.** Three things, in order of weight:

1. **Do you understand your own work at the level of the decisions, not the results?** Every design choice in your paper is a question. Why that baseline, why that dataset, why that ablation, what you would do with 10x compute, what the reviewer criticism was and whether you agreed.
2. **Can you situate it?** What problem in the field does this address, what came before, what came after, what did it fail to solve.
3. **Are you honest about limitations?** This is the highest-signal part and the one candidates handle worst. A researcher who cannot name the weakest part of their own paper is either not thinking or not honest, and both are disqualifying.

**How to prepare — your own work.**

- Write out a **90-second version**, a **5-minute version**, and a **20-minute version**. You will be interrupted; know where you are.
- Rebuild the **motivation chain**: problem → why existing approaches fail → your insight → why it should work → evidence that it does. Interviewers probe the middle two links.
- List every **design decision** and have a one-sentence justification plus one alternative you considered and why you rejected it. This is the highest-density question source.
- Know the **math cold**. Not the intuition — the actual derivation of the core loss or mechanism.
- Prepare your **limitations paragraph**: three things the paper does not establish, one thing you would redo, one result you are least confident in.
- Know **what happened since**. If the field has moved past your method, say so before they do, and say why.
- If it is **team work**, be precise about your contribution. Overclaiming is detectable by anyone who reads the paper; underclaiming reads as low ownership. State the boundary explicitly: "I owned the distillation objective and the ablations; my co-author built the graph sampling."

**How to prepare — someone else's work.** Have a repeatable critique structure:

1. What is the claim, in one sentence.
2. What evidence is offered, and is the experimental design sufficient to support the claim?
3. What is the baseline, and is it fair? (Undertuned baselines are the most common real flaw.)
4. What is confounded? What ablation is missing?
5. Would this hold at a different scale / distribution / task?
6. What would you run next, and what would change your mind?

Then say what you *liked*. A critique with no positive read sounds like posturing.

**Failure modes.**

- Presenting the abstract instead of the reasoning. If you narrate results, they will ask why, and you will have burned your best framing.
- Being unable to answer "what is the weakness." Prepare this one sentence and you clear a bar most candidates do not.
- Getting defensive. The push is the interview. Treat a hard objection as a research conversation: "That's the right objection. We looked at it — here's what we found, and here's what's still open."
- Not knowing the fundamentals underneath your own work. The Microsoft report shows the interviewer descending from the paper to "what is a GNN" to attention math to batch-vs-layer norm to vanishing gradients **[one report]**. Paper rounds bottom out in fundamentals. Prepare the floor, not just the ceiling.
- Choosing your most impressive paper over your best-understood one, when you get to choose. Choose the one you can defend for 45 minutes.

### 3.6 Statistics and experimentation

**What it looks like.** Fundamentals, rapid-fire, often with follow-ups that go one layer past the definition. This round is standard in Data Scientist loops and shows up in Applied Scientist loops more often than candidates expect.

The Microsoft AS2 report has an entire dedicated round, and the candidate's own commentary is the lesson: "This round catches a lot of people off guard. Most AI/ML folks skip statistics prep assuming it won't come up. It does." **[one report]** Documented questions included t-tests, p-values, how to carry out hypothesis testing, the Central Limit Theorem, the five assumptions of linear regression and what breaks when each is violated, deriving the linear regression loss, and $R^2$ — including what it means if $R^2 < 0$ and if $R^2 > 1$.

**What it is really testing.** Whether you can reason about uncertainty rather than recite definitions. The follow-up is always the real question. "What is a p-value" is a warm-up; "your p-value is 0.04 and the effect size is 0.1% — do you ship?" is the interview.

**How to prepare.** Cover: hypothesis testing end to end (null, test statistic, decision, error types), power and sample size, confidence intervals and what they do *not* mean, CLT, linear regression assumptions and violations, $R^2$ and its pathologies, multiple testing correction, A/B test design, novelty and primacy effects, network interference, and what to do when the primary metric is flat but a secondary moves.

Two traps worth memorizing: **$R^2 < 0$** happens when your model is worse than predicting the mean — possible on out-of-sample data or a model without an intercept. **$R^2 > 1$** is not possible for ordinary least squares on the fitted data; if you see it, you have a bug, a wrong formula, or computed it against the wrong baseline. Saying "that indicates an error, here is what I would check" is the correct answer, not an attempt to rationalize it.

**Failure modes.** Definitional recall with no operational follow-through. Misstating what a confidence interval means. Ignoring practical significance. Not asking about the unit of randomization.

### 3.7 SQL and data manipulation

**What it looks like.** Anything from "basic filtering and joins" to multi-CTE window function gymnastics. In the Microsoft AS2 report SQL appeared in the *hiring manager* round and was basic: database filtering and matching, joins, aggregations **[one report]**. In DS loops it is a full round with real difficulty **[aggregated]**.

**What it is really testing.** For engineers and scientists: can you get your own data without help. For DS: can you express a business question as a query, and do you state your assumptions about the data.

**How to prepare.** Joins including anti-joins, GROUP BY with HAVING, window functions (ROW_NUMBER, RANK, LAG/LEAD, running aggregates), CTEs, date handling, and — the underrated one — narrating your assumptions about the schema out loud before writing.

**Failure modes.** Writing before asking whether the table has duplicates or nulls. Silence. Not sanity-checking the result. Treating a "basic" SQL question as beneath you — it is a floor check, and failing a floor check is worse than struggling with a hard one.

### 3.8 Take-home projects

Covered in full in Section 6.

### 3.9 Live debugging

**What it looks like.** You are handed a broken notebook or script — a training loop that does not converge, a pipeline producing wrong numbers — and asked to find and fix it. Reported as a round type at frontier labs, described as "ML debugging using broken Jupyter notebooks" **[aggregated]**.

**What it is really testing.** Whether you debug systematically or by vibes. This is the round that correlates best with actual day-to-day competence, which is why it is spreading.

**How to prepare.** Practice a method and state it aloud: reproduce, isolate, form a hypothesis, design the cheapest test that discriminates it, check. Know the standard ML bug catalogue: shuffling applied to features but not labels, train/test leakage, a normalization applied at train but not inference, wrong axis on a reduction, loss not being zeroed, learning rate off by orders of magnitude, an off-by-one in sequence shifting, class imbalance masked by accuracy.

**Failure modes.** Randomly changing things. Not reading the error message carefully. Not checking the data before checking the model — the data is usually where the bug is. Fixing the symptom without explaining the cause.

### 3.10 Hiring manager / behavioral

Covered in Section 7.

---

## 4. A worked example: the Microsoft Applied Scientist 2 loop

**Source and caveat.** Everything in this section comes from [HimankSehgal/AI-interview-prep](https://github.com/HimankSehgal/AI-interview-prep), specifically [`microsoft.md`](https://raw.githubusercontent.com/HimankSehgal/AI-interview-prep/main/microsoft.md), documenting the author's own Applied Scientist 2 interview. Full credit to the author for publishing it — first-hand, specific, round-by-round accounts of AI/ML loops are rare and this one is unusually detailed.

**This is one candidate's documented experience, not an official published Microsoft process.** Microsoft's own careers page describes something much vaguer — "typically 2–4 conversations" — and loops vary by org, level, and role. Treat this as a concrete instance of a *shape* that recurs, not as a specification.

**The structure.** Screening round, then four loop rounds, with all four happening regardless — the author states explicitly that it is "not elimination based." The stated philosophy: "they test your **intuition**, not just your knowledge."

| Round | Focus |
|---|---|
| Screening | Research paper deep dive + coding |
| Round 1 — DSA | 3 coding questions, brute force to optimal |
| Round 2 — Statistics | Stats fundamentals, linear regression, hypothesis testing |
| Round 3 — ML Depth | Transformers, VLMs, project deep dive |
| Round 4 — Hiring Manager | Behavioural, SQL, role discussion |

### Screening: paper deep dive + coding

**Asked.** The interviewer went deep on the candidate's published work on distilling Graph Transformers into simple MLPs — motivation, math, results. What is a GNN. What is a Graph Transformer. What are the current problems in Graph Transformers (the $O(N^2)$ complexity). Why distill into an MLP (inference speed, deployment simplicity). Then attention: Q/K/V matrices, the self-attention formula, why we divide by $\sqrt{d}$, softmax. Then fundamentals: batch norm vs layer norm and when to use each, vanishing and exploding gradients and how to handle both. Then a coding question: first non-recurring character in a string.

**What it tested.** Whether the paper on the résumé is *yours* in the sense of understood, and whether the fundamentals underneath it are solid. Note the descent: paper → mechanism → math → training dynamics. That descent is the design of the round.

**Strong answer.** Motivation first, in one breath: Graph Transformers are expensive because attention over $N$ nodes is quadratic, MLPs are cheap at inference, the question is whether the inductive structure can be transferred without the cost. Then approach, then results, then — before being asked — the limitation. On $\sqrt{d}$: not "for stability," but the actual argument. The dot product of two $d$-dimensional vectors with unit-variance components has variance growing with $d$; without scaling, the softmax saturates as $d$ grows and gradients vanish.

**Weak answer.** Narrating the abstract. Answering "why $\sqrt{d}$" with "it normalizes it." Being unable to distinguish batch norm from layer norm beyond "one is over the batch."

### Round 1: DSA

**Asked.** Valid Anagram (LC 242, Easy), Minimum Number of Taps to Open to Water a Garden (LC 1326, Hard), Minimum Insertion Steps to Make a String Palindrome (LC 1312, Hard).

**What it tested.** Per the author: not just correctness, but "how you build your approach from brute force to optimal and how you dry run on test cases."

**Strong answer.** On the anagram: state the counting approach, note the sorting alternative and the complexity difference, write it, run it on an example. On the Hards: state the brute force first — even if it is exponential — establish correctness, *then* find the structure. LC 1312 is longest-common-subsequence with its reverse, or interval DP; saying "this looks like a palindrome DP, let me define $dp[i][j]$ as the minimum insertions for substring $i..j$" is most of the round.

**Weak answer.** Freezing on the Hard because it was not in your last 50 problems. Writing the memorized greedy for 1326 without being able to argue why the greedy is correct. Never dry-running.

### Round 2: Statistics

**Asked.** t-test. p-value. How you carry out hypothesis testing (null → test → conclusion). Central Limit Theorem. The five assumptions of linear regression and what happens when each breaks. Deriving the linear regression loss. $R^2$ — including what $R^2 < 0$ means and what $R^2 > 1$ means.

**What it tested.** Whether an ML person has the statistical foundation, or has only ever called `.fit()`. The author's warning is the takeaway: most AI/ML candidates skip stats prep and it comes up.

**Strong answer.** On the assumptions: linearity, independence of errors, homoscedasticity, normality of errors, no perfect multicollinearity — and for each, what specifically breaks. Violate independence and your standard errors are wrong so your p-values lie, but the coefficients stay unbiased. Violate homoscedasticity and the same thing happens; the fix is robust standard errors, not a different model. Violate linearity and the coefficients themselves are biased. Notice the pattern in a strong answer: *which* thing breaks, not just "the model is bad."

On $R^2 > 1$: say it cannot happen for OLS on the fitted data and that observing it means a bug. Confidently explaining an impossible number is a bad signal.

**Weak answer.** Definitions with no follow-through. "p-value is the probability the null is true" (it is not). Listing four assumptions and stalling.

### Round 3: ML depth

**Asked.** The attention matrix, walked through mathematically. What is BERT, what is GPT, how do they differ. How do you handle image and text together in a transformer. How do you tokenize images (patch embeddings, ViT). Why multiple attention heads. Then the project deep dive — a Blinkit project extracting attributes from product images — covering the problem statement, why Gemini models (better at image reasoning), which Gemini model and why, and how you decide which LLM to use: define the task type, evaluate cost, evaluate latency (offline vs real-time). Then open discussion: where is the ML field heading, how do you define tools for AI agents.

**What it tested.** Depth on architectures plus judgment on engineering decisions. The LLM-selection question is a judgment question wearing a knowledge question's clothes.

**Strong answer.** On BERT vs GPT: bidirectional encoder trained with masked LM versus causal decoder trained with next-token prediction, and therefore what each is good for and why you would not use BERT for generation. On multi-head: each head is a separate subspace projection, letting the model attend to different relations simultaneously; a single head of the same total width cannot represent the same set of relations because the softmax forces one distribution. On model choice: give the decision procedure the author gives — task type, cost, latency — and then add the missing fourth, evaluation. "I picked the model after measuring both on 200 labeled examples" is the sentence that separates candidates.

**Weak answer.** "We used Gemini because it was better." Better at what, measured how, against what, at what cost per call. The project deep dive is graded on whether your decisions were reasoned or inherited.

### Round 4: Hiring manager (director level)

**Asked.** Basic SQL — database filtering and matching, joins, aggregations. Behavioral: how do you handle disagreements with your manager; what feedback did your previous manager give you and how did you implement it. Then role and culture: day-to-day responsibilities, how agents are being introduced at Dynamics 365, expectations from the role.

The author describes this as the best round — "very senior, very humble," more conversation than interview — and notes the tip that this round is as much you evaluating them.

**What it tested.** Level, self-awareness, coachability, and whether you actually want *this* job. The feedback question is a self-awareness probe: an answer with no real feedback in it fails.

**Strong answer.** On disagreement: a specific instance, what your reasoning was, how you surfaced it, what data you brought, what the outcome was, and — critically — what you did after the decision went against you. On feedback: name a real weakness, name the specific change you made, name the evidence it worked.

**Weak answer.** "My manager said I work too hard." "I've never really disagreed with a manager." Having no questions at the end.

### What generalizes from this loop

**[my read]** Three things, and they hold well beyond Microsoft:

1. **The paper/project round descends into fundamentals.** Prepare the floor under your work.
2. **Statistics is under-prepared relative to how often it appears** in scientist loops.
3. **Non-elimination means a bad round is survivable — if you do not give up.** The author sat four rounds regardless of outcome in any single one. Under that model, the worst thing you can do after a rough round is disengage.

---

## 5. Other loops, and how they differ

Company-specific detail is where interview content goes wrong, so here is what is actually published, followed by a comparison of *kinds* of employer, which is more useful anyway.

### What companies publish about themselves

| Company | Published detail | Level |
|---|---|---|
| Amazon | 1–2 technical phone screens (60 min each), then 4 interviews of 55 min; decision in 5 business days; 3–6 weeks total; 2–7 interviewers including a Bar Raiser; all rounds assess Leadership Principles; STAR method recommended with metrics | **[official]** |
| OpenAI | 5 phases: application (~1 week review), intro calls, skills-based assessment (pair coding / take-home / technical test, varies by team), final interviews of 4–6 hours with 4–6 people over 1–2 days, decision within a week; engineering assessed on design quality, code quality, performance, test coverage | **[official]** |
| Anthropic | Google Meet; live coding in Colab / CodeSignal with materials allowed; non-technical roles conversational; explicitly unhurried timeline; published candidate AI-use policy; no interview feedback given; 12-month reapplication window | **[official]** |
| Meta | 45-min technical screen (5 intro / 35 coding, two problems / 5 questions), then three 45-min rounds: coding, design (systems or product), behavioral | **[official]** |
| Microsoft | "Typically 2–4 conversations lasting up to an hour each"; evaluates respect, integrity, accountability, growth mindset | **[official]** |

That is genuinely all that is well-published. Note how thin it is: none of these describe an ML-role loop specifically except Amazon's Applied Scientist page, and even that one lists competencies rather than rounds.

### What is reported but not published

- **Anthropic**: a reported 3-stage shape — 30-min recruiter, 60–90 min coding challenge (sometimes a 90-minute CodeSignal take-home, skippable for referrals), then a 4–5 hour onsite of roughly five sessions: hiring manager, coding, system design, a role-specific second coding round, and a values round; ~3–4 weeks **[aggregated, based on conversations with engineers]**. ([interviewing.io](https://interviewing.io/anthropic-interview-questions))
- **Google**: hiring-committee model, and the well-known consequence that passing the loop does not guarantee an offer without a team match **[aggregated]**.
- **DeepMind**: reported to include a fundamentals quiz round and rounds resembling "a PhD defense mixed with a rigorous engineering exam" **[aggregated]** — I would treat the specifics as unverified.
- **Frontier labs generally**: reported to weight implementation coding more heavily than research discussion for engineer-titled roles **[aggregated]**.

I could not verify company-specific ML loops for Apple, Netflix, NVIDIA, or the major AI startups from primary sources. If you see a table listing those loops round-by-round, assume it is reconstructed from Glassdoor.

### The more useful comparison: kinds of employer

**[my read]**, informed by the sources above. Weights are relative emphasis, not absolutes.

| Employer type | Research output | Shipping speed | Systems / infra | Product sense | Signature round | What kills you |
|---|---|---|---|---|---|---|
| **Frontier AI lab** (research org) | Very high | Medium | High | Low | Research talk / paper defense | No novel contribution; cannot implement what you propose |
| **Frontier AI lab** (product/applied org) | Low | Very high | High | High | Practical coding under production constraints | Treating it as a research interview |
| **Big tech applied science** | High | Medium | Medium | Medium | Paper/project deep dive + statistics | Skipping stats; cannot defend own work |
| **Big tech ML engineering** | Low | Medium | Very high | Medium | ML system design | No production/monitoring story; over-engineering |
| **Big tech data science** | Low | Medium | Low | Very high | Product metrics + experimentation case | No product framing; textbook stats with no decision |
| **Growth-stage startup (AI product)** | Very low | Very high | Medium | Very high | Take-home + defense | Over-engineering; no evaluation; slow |
| **Early startup (seed/A)** | Very low | Extreme | Medium | High | Paid trial or scoped project | Needing structure; can't work under ambiguity |
| **Non-tech enterprise / consultancy** | Low | Low | Medium | High | Case + stakeholder communication | Being unable to explain a model to a non-technical exec |

Two practical consequences:

**The same company runs different loops for different orgs.** "How does Company X interview" is usually the wrong question. "How does this team interview for this role at this level" is the right one, and your recruiter can answer it.

**Match your register to the employer type.** The candidate who talks about ablations at a growth-stage startup, and the one who talks about shipping velocity at a research lab, are both giving true answers to the wrong audience.

---

## 6. The take-home

Take-homes are the highest-variance stage. Done well, they are the best chance you have to demonstrate judgment. Done badly they waste 20 hours and produce a rejection.

### What good and bad take-homes look like

**Good take-home (from the company's side):** a scoped problem with a stated time budget, a clear deliverable, real-ish data, and no hidden requirements. It resembles the actual job. It is followed by a discussion where you defend your choices.

**Bad take-home:** unbounded scope, no time limit, a task that is really free consulting, or a task requiring 30 hours dressed as "a few hours." You are allowed to decline these, and asking "what is the expected time investment?" is both reasonable and informative — the answer tells you a lot about the company.

Reported norms for AI engineering assignments: **2–7 day deadlines with 2–4 hours of actual expected work** **[practitioner]**, with the honest caveat that candidates should "double your time estimate."

### How long to actually spend

Spend the stated time, plus a fixed overhead for documentation. If the brief says four hours, spend four hours on the work and one on the README. Then stop, and *write down what you would have done with more time.*

This is not moral advice. It is scoring advice. Reviewers compare submissions against the stated budget. A submission that obviously took 25 hours against a 4-hour brief signals poor judgment about scope — which is a job skill they are testing — and it distorts the comparison in a way experienced reviewers notice and dislike.

### What reviewers actually grade

From a hiring engineer's account of reviewing take-homes **[practitioner]** ([BigPanda Engineering](https://medium.com/bigpanda-engineering/secrets-from-the-interview-room-what-reviewers-look-for-in-a-take-home-coding-assignment-1aaec70dabe0)), the criteria are:

1. **Does it run?** First try, on a clean machine. Missing dependencies are the single most common failure — things installed globally on your machine that are not in your requirements file.
2. **Is it structured?** Consistent style, sensible module boundaries, no leftover `TODO`s or debug prints. Use a linter so you spend your judgment on architecture instead.
3. **Is it documented?** A README written assuming the reviewer knows nothing: install, run, usage example, assumptions you made, tradeoffs you took.
4. **Is it tested?** Tests show you thought about edge cases.
5. **Is it presented professionally?** Hosted on a git remote with sensible commits, not emailed as a zip. The commit history shows how you work.

For ML and AI engineering specifically, add the one that dominates: **did you evaluate it?** The AI engineering field guide is blunt — build an eval harness *before* the main logic, and missing evaluation is treated as a red flag **[practitioner]**.

Note what is *not* on the list: whether your model is the best possible. Reviewers are not comparing your F1 against a leaderboard. A well-evaluated logistic regression with a documented reason for choosing it beats an unevaluated fine-tune almost every time.

### A structure that works

```
README.md          <- how to run, what you did, what you'd do next, tradeoffs
requirements.txt   <- pinned, tested from clean
src/               <- the actual code, importable, not one notebook
  data.py
  model.py
  evaluate.py
tests/             <- a few real tests, not 40 trivial ones
notebooks/         <- exploration, clearly marked as exploration
results/           <- metrics, plots, the eval output
```

The README is the highest-leverage file. Structure it as: what the problem is as you understood it, how to run it, what you built and why, **how you evaluated it and what the numbers are**, what the limitations are, and what you would do with more time. That last section is where you get credit for everything you consciously chose not to build.

### The traps

- **Over-engineering.** Kubernetes manifests for a take-home. A custom training loop where sklearn would do. This reads as poor prioritization.
- **No README, or a README that is just install instructions.** You are being graded on communication and you skipped the communication artifact.
- **No evaluation.** The most common ML-specific failure. A model with no baseline and no metric is not a result.
- **No baseline.** Always include the dumbest reasonable approach and its score. It makes every other number interpretable.
- **Ignoring the stated time limit** in either direction.
- **A single giant notebook.** Fine for exploration, not for the deliverable.
- **Not preparing for the defense.** Most take-homes are followed by a discussion round where you justify every choice. The submission is the *setup*; the defense is where the score lands. Reread your own code before that call.
- **Silent assumptions.** If the brief is ambiguous, ask. If you cannot ask, document the assumption in the README. Ambiguity is often deliberate, and how you resolve it is being graded.

---

## 7. Behavioral, for technical people

Engineers underprepare this round systematically, for a bad reason: they assume it cannot be prepared, or that it is a formality. Neither is true. It is the round with the most predictable question set in the entire loop, and at some companies it is weighted equally with the technical rounds — Amazon states that all Leadership Principles apply regardless of role, and that behavioral assessment runs through the phone screens as well as the loop **[official]**.

### The questions that actually get asked

The set is small and stable:

- Walk me through a project you owned end to end.
- Tell me about a time you disagreed with someone. (Manager, peer, PM — the variants matter little.)
- Tell me about a failure. What did you learn?
- Tell me about a time you had to make a decision without enough data.
- Tell me about a time you influenced someone without authority.
- What feedback have you received, and what did you do with it?
- Tell me about a time you had to change direction.
- What is the hardest technical problem you have solved?
- Why this company, why this team, why now?

The Microsoft AS2 hiring manager round asked two of these directly: how you handle disagreements with your manager, and what feedback your previous manager gave you and how you implemented it **[one report]**.

### Why engineers underprepare

**[my read]** Three reasons. First, technical people treat these as "soft" and therefore unscoreable — but interviewers score them against a written rubric, often mapped to explicit company values. Second, the questions look answerable on the fly, and they are, badly. Third, the failure is invisible: you leave thinking the conversation went fine, and the feedback says "did not demonstrate ownership," which you would never have guessed.

### STAR, applied to ML work

STAR — Situation, Task, Action, Result — is the standard structure and Amazon explicitly recommends it, including metrics in the answer **[official]**. But generic STAR advice produces generic answers. For ML work specifically:

- **Situation.** The *business* context, not the technical one. "Search relevance was hurting conversion on long-tail queries, which were 30% of traffic." Not "we had a BERT model."
- **Task.** What *you* owned, stated as a boundary. "I owned the ranking model and the offline eval; another engineer owned serving."
- **Action.** This is where ML answers usually go wrong. Narrate **decisions and their alternatives**, not steps. "I started with a gradient-boosted baseline on existing features rather than fine-tuning, because we had no labeled data and I wanted a number in three days" is an Action. "I trained a model" is not. Include at least one tradeoff you consciously took and one thing you deliberately did not do.
- **Result.** Numbers. Offline *and* online if you have both, and be honest when they disagreed — "offline NDCG improved 8% but the A/B was flat, and here is what we learned about why" is a stronger answer than a clean win, because it demonstrates you actually shipped and measured.

Two ML-specific additions worth building into your stories:

- **Say what the baseline was.** ML people report improvements without baselines constantly. It makes the result uninterpretable and reads as inexperience.
- **Say what happened after.** Did it stay deployed? Did it degrade? Did anyone retrain it? Ownership through the maintenance tail is exactly what separates senior from mid-level in these rubrics.

### The three stories everyone needs

Have these written out. Not memorized word-for-word — written out, so the structure is solid and the numbers are correct.

**1. A project, end to end.** Your flagship. From ambiguous problem statement through to a measured outcome in production. Should include: how the problem was scoped, one hard technical decision with its alternative, one thing that went wrong mid-project, the metric movement, and what happened to it after launch. This story gets used for "walk me through a project," "hardest problem," "time you owned something," and "time you dealt with ambiguity." Prepare a 2-minute and a 10-minute version.

**2. Something you got wrong.** Must be a *real* failure with a *real* cost. The anti-pattern is the disguised strength ("I cared too much about quality"). A good version: you shipped a model that looked good offline and hurt a metric online, you found out why, you changed how you validate. What is being scored is not the failure — it is whether you have an accurate model of your own mistakes and whether you changed your process. End on the process change, not on the apology.

**3. A disagreement you handled.** Must include: a specific technical or prioritization disagreement, how you made your case (data helps), *and the outcome when it did not go your way.* Interviewers are specifically checking whether you disagree productively and then commit. A story where you were right and everyone eventually agreed is a weaker answer than one where you lost, committed fully, and the thing worked out — or where you lost, committed, it went badly, and you handled that well too.

### What the hiring manager is really assessing

Behind the questions, four things:

1. **Level.** Does the scope of your stories match the ladder level? A senior candidate telling stories about tasks they were assigned rather than problems they defined will be down-leveled.
2. **Ownership.** Do you say "I" appropriately, and do you own outcomes including bad ones? Excessive "we" reads as either humility or hiding; interviewers probe to find out which.
3. **Coachability.** Can you name a real weakness and a real change? This is what the feedback question is for.
4. **Do I want to work with this person for three years?** Unscoreable but real. It is mostly conveyed by whether you are curious about their problems.

**[my read]** The single most common hiring-manager-round failure is having no questions. It reads as not caring which job you get. See Section 9.

---

## 8. Reading the room

### Signals a round is going well

- The interviewer starts asking follow-ups *past* the question — pushing on your answer rather than moving on. Depth of pursuit is engagement.
- They start talking about their own team's version of the problem.
- They go over time.
- They start selling — describing the roadmap, the team, why it is interesting.
- They ask "what would you do with more time on this?"

### Signals it is going badly

- Rapid movement through questions without follow-ups.
- The round ends 15 minutes early with "any questions for me?"
- They start giving you hints early and often. (Hints are fine; *escalating* hints mean they are trying to salvage a signal.)
- They stop taking notes.
- They repeat a question you thought you answered. This is a gift — they are telling you that you missed the point. Ask what part they want you to expand.

Two cautions. First, these signals are noisy: some interviewers are warm and reject you, and some are brusque and advocate for you. Second, **your read on the round is not reliable.** Candidates systematically misjudge, and the practical consequence is: never let a perceived bad round affect the next one, especially under a non-elimination model.

### What to do when you do not know

This is the highest-leverage skill in the whole loop, because *everyone* hits their edge — depth rounds are designed to find it. What varies is what happens next.

The bad version: bluff, or freeze and say "I don't know" flatly and stop.

The good version has four parts:

1. **Name the boundary honestly.** "I have not worked with that directly."
2. **Say what you do know that is adjacent.** "I have worked with the related case of X."
3. **Reason forward out loud.** "So I would expect it to behave like — here is my reasoning."
4. **Say what would settle it.** "I would check whether Y holds; if it does, my answer is A, if not, B."

That sequence converts a knowledge gap into a demonstration of reasoning, which is what a depth round is grading anyway. The Microsoft report's framing — intuition over exact knowledge, "nobody knows everything" — is exactly this **[one report]**.

Three notes. **Do not fake it.** Interviewers ask about things they know deeply; a confident wrong answer is far worse than an honest gap, because it destroys the credibility of everything else you said. **Do not over-apologize** — one sentence acknowledging the gap, then reason. And **accept hints**. A candidate who takes a hint and runs with it scores better than one who stubbornly continues down a dead end; collaboration is being observed.

### Clarifying questions without stalling

Clarifying is scored positively, but there is a limit past which it reads as avoidance.

A good pattern is **bounded clarification**: two or three specific questions, then state your assumptions and start. "I have two questions: is the input sorted, and can it contain duplicates? … Okay — I am going to assume ASCII input and that we optimize for time over space; tell me if that's wrong. Starting with the brute force."

Good questions are the ones whose answers change your approach: constraints and scale, what the input can contain, what "best" means (latency? accuracy? cost?), whether an edge case matters. Bad questions are ones you could resolve by assumption, or a long list front-loaded to delay starting.

**Time-box it.** Two minutes in a coding round; five in a system design round, where scoping *is* the round.

### Recovering a round you started badly

It happens. Some things that actually work:

- **Name it and reset, once.** "I went down the wrong path there — let me restart with a cleaner approach." Interviewers respect this; it demonstrates self-correction, which is a job skill. Do it once, not three times.
- **Secure the base case.** If you are 25 minutes in with nothing working, abandon the elegant solution and get the brute force running. A working suboptimal solution scores; an unfinished optimal one often does not.
- **Ask for the hint.** "I am stuck between two approaches — do you have a preference for where I spend the remaining time?" This is not weakness; it mirrors how you would work with a colleague.
- **Fix the wrong thing you said.** If you realize five minutes later that you misstated something, say so. "Earlier I said X — that's not right, it's actually Y." This is a strong signal and candidates almost never do it.
- **Do not carry it forward.** The next interviewer usually has not read the previous feedback. A fresh round is genuinely fresh — act like it.

---

## 9. Questions to ask them

Two purposes: signal, and diligence. You are choosing too, and the questions that surface problems are different from the ones that impress.

Ask questions appropriate to who you are talking to. Asking a recruiter about architecture wastes a turn; asking an engineer about compensation bands wastes goodwill.

### Recruiter

| Question | What the answer tells you |
|---|---|
| What does the full loop consist of, round by round? | Lets you prepare the right rounds. They almost always tell you. |
| Is this a holistic review or is each round a gate? | Elimination vs non-elimination — see Section 2. |
| What level is this role, and what is the band? | Prevents a six-week process ending in a mismatch. |
| Is this for a specific team, or is there a team-match stage? | Whether passing the loop means an offer. |
| How long has this role been open? | Long-open roles mean an unrealistic bar, a hard-to-fill team, or repeated backfill. |
| Is this a backfill or a new headcount? | Backfill invites the follow-up: why did the last person leave? |
| What is the typical timeline from here? | Manageable expectations, and useful when running parallel processes. |

**Red flag:** a recruiter who will not tell you the level, band, or loop structure. That opacity usually continues after you join.

### Engineer / peer interviewer

| Question | What the answer tells you |
|---|---|
| What did you ship last quarter? | The single best question. Concrete answer = healthy team. Vague answer = the work is diffuse or blocked. |
| What does a typical week look like? How much is meetings? | Real time allocation. |
| How does a model get from a notebook to production here? | Whether ML infrastructure exists or you will be building it. This determines what your job actually is. |
| How do you evaluate models before shipping? | If there is no answer, evaluation is not a practice, and you will be the one introducing it. |
| What is on call like? | Load, and whether the systems are stable. |
| What is the most frustrating thing about working here? | Everyone has one. An answer of "nothing" means they will not be candid with you. |
| How long have people on the team been here? | Tenure distribution is the cheapest attrition signal available. |

**Red flags:** cannot name what they shipped; no path to production; "we don't really have time for evaluation"; visible hesitation before answering the frustration question.

### Hiring manager

| Question | What the answer tells you |
|---|---|
| What does success look like for this person at 3, 6, and 12 months? | If they cannot answer, the role is not defined and you will be inventing it. |
| What is the biggest problem you are hoping this hire solves? | The actual job, as distinct from the job description. |
| How is work prioritized? Who decides what the team works on? | Whether the team has agency or is a service desk. |
| How do you measure the team's impact? | Whether ML work is tied to business metrics, or is a science project with an expiry date. |
| What has the team tried that did not work? | Institutional honesty, and whether failure is survivable here. |
| How do you handle disagreement about technical direction? | The mirror of the question they asked you. |
| How stable is this team's headcount and mandate? | Especially relevant for AI teams, which get reorganized frequently. |

**Red flags:** no definition of success; "we'll figure out the roadmap once you join" for a non-founding role; an answer to the impact question with no metric in it; obvious discomfort about team stability.

### Skip-level / senior leader

| Question | What the answer tells you |
|---|---|
| Where does this team fit in the org's strategy over the next two years? | Whether the team is core or discretionary — i.e. what happens in a cost-cutting cycle. |
| How is AI/ML investment here funded — is it a product line or a cost center? | The most predictive question about long-term stability. |
| What would make you shut this team down? | Uncomfortable, honest, and enormously informative. Ask it if the rapport supports it. |
| How do you decide between building and buying model capabilities? | Whether there is a coherent technical strategy. |
| What is the promotion path for scientists/engineers here? | Whether the IC ladder is real above senior. |

**Red flags:** cannot articulate why the team exists; the strategy is a restatement of a trend; the IC ladder tops out well below the management ladder.

### General principles

- **Ask what you actually want to know.** Performed questions are visible.
- **Ask the same question of several interviewers.** Divergent answers to "how does work get prioritized" is itself the finding.
- **Save compensation for the recruiter.** Never the engineers, and generally not the hiring manager mid-loop.
- **Always have at least two.** Ending with "no, I think you covered everything" is the most common own-goal in the last five minutes of a loop.
- **Take the invitation seriously when it is offered.** The Microsoft report describes the director round as genuinely conversational, with the explicit note that it is as much about you evaluating them **[one report]**. Some interviewers mean it.

---

## Sources

Primary anchor:

- [HimankSehgal/AI-interview-prep](https://github.com/HimankSehgal/AI-interview-prep) — and its [`microsoft.md`](https://raw.githubusercontent.com/HimankSehgal/AI-interview-prep/main/microsoft.md), the documented Microsoft Applied Scientist 2 loop used throughout Section 4. One candidate's first-hand account, published openly. Credit to the author.

Official company sources:

- [Amazon — Applied Scientist Interview Prep](https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep)
- [About Amazon — the interview process, phone screens to loops](https://www.aboutamazon.com/news/workplace/amazon-interview-process-phone-screens-loops)
- [OpenAI — Interview Guide](https://openai.com/interview-guide/)
- [Anthropic — Careers](https://www.anthropic.com/careers)
- [Meta — Preparing for your software engineering interview](https://www.metacareers.com/life/preparing-for-your-software-engineering-interview-at-meta)
- [Microsoft — How we hire](https://careers.microsoft.com/v2/global/en/hiring-tips.html)

Practitioner and community sources:

- [Hello Interview — ML System Design](https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/introduction)
- [alexeygrigorev/ai-engineering-field-guide — home assignments](https://github.com/alexeygrigorev/ai-engineering-field-guide/blob/main/interview/questions/06-home-assignments.md)
- [BigPanda Engineering — what reviewers look for in a take-home](https://medium.com/bigpanda-engineering/secrets-from-the-interview-room-what-reviewers-look-for-in-a-take-home-coding-assignment-1aaec70dabe0)
- [generalizederror — My Machine Learning Research Jobhunt](https://generalizederror.github.io/My-Machine-Learning-Research-Jobhunt/)
- [interviewing.io — Anthropic interview questions](https://interviewing.io/anthropic-interview-questions)
- [techinterview.org — what GenAI engineer interviews test](https://www.techinterview.org/post/3233476396/what-genai-engineer-interviews-test/)
- [Sundeep Teki — AI research engineer interview guide](https://www.sundeepteki.org/advice/the-ultimate-ai-research-engineer-interview-guide-cracking-openai-anthropic-google-deepmind-top-ai-labs)
- [Interview Query — Meta data scientist guide](https://www.interviewquery.com/interview-guides/facebook-data-scientist)
- [fyjump — SWE interview processes in 2026](https://www.fyjump.com/post/the-swe-interview-process-in-2026-top-11-big-tech-companies-broken-down)

---

## The short version

1. Find out which of the five roles you are actually interviewing for. Prepare that loop.
2. Ask the recruiter for the round structure and whether it is elimination or holistic. They will tell you.
3. Everything on your résumé is a question. Remove what you cannot defend for ten minutes.
4. Prepare statistics even if you think you are an engineer.
5. In take-homes, evaluation and the README beat model quality.
6. Write out three behavioral stories with real numbers.
7. When you hit your edge — and you will — name it, reason forward, say what would settle it.
8. Have questions. Ask what they shipped last quarter.
