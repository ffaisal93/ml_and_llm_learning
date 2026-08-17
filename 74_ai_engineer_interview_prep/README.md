# Topic 74: The AI Engineer Interview

Everything numbered below 73 in this repository teaches you material. Topic 73 compresses it for the
night before. This folder is about the interview itself — the loop you are walking into, and the
questions being asked in it *right now* for the applied GenAI roles that did not exist as a job title
four years ago.

It exists because of a specific failure mode I kept seeing. Someone prepares seriously — works through
transformers, derives backprop, grinds LeetCode — walks into an "AI Engineer" loop, and gets asked how
they would keep a vector index fresh, what they would do when an LLM judge disagrees with their humans,
and how they would bound the cost of an agent that will not stop calling tools. All fair questions for
the job. None of them anywhere in the preparation.

## Files in this folder

| File | Purpose |
|---|---|
| `LOOP_RESEARCH_QA.md` | 51 questions on your own paper or project — the walkthrough, motivation, method depth, results and rigor, limitations, extension — plus the variant where they hand you a paper to critique. Model answers are built on one consistent running example so you can see the shape end to end. |
| `LOOP_ML_DEPTH_QA.md` | 57 questions with full answers. Attention and transformers, training dynamics, generalization, LLM-specific, and the classical ML that still gets asked. The derivations are done, not gestured at. |
| `LOOP_STATISTICS_QA.md` | 60 questions. The round most ML candidates skip and get eliminated by. Inference, regression as a statistics object, A/B testing and causal inference, probability, and the applied traps. Numbers are computed. |
| `LOOP_CODING_QA.md` | 60 problems with runnable solutions. ML implementations from scratch, the DSA that actually appears in these loops, data manipulation, and debugging. Every solution was executed and checked against an independent oracle. |
| `LOOP_SYSTEM_DESIGN_QA.md` | 14 design prompts worked end to end — clarifying questions, the design, where the AI actually is and what you would not use a model for, the hard tradeoff, evaluation, failure modes, and the follow-ups. |
| `LOOP_BEHAVIORAL_QA.md` | 52 questions with model answers, plus 28 questions to ask them grouped by who you are asking. Built on one consistent persona, and the answers admit real failures rather than humble-bragging. |
| `THE_LOOPS.md` | The process. Five distinct roles that share vocabulary and nothing else, the standard pipeline stage by stage, what each round type is actually grading, a fully worked real loop, take-homes, behavioral, and reading the room. |
| `MODERN_QUESTION_BANK.md` | The content. Eleven sections of applied GenAI questions — RAG, agents, prompting and context, evaluation, LLMOps, deployment, safety, judgement — plus 65 rapid-fire, each with the answer written the way you would say it out loud. |
| `RAG_FAILURE_DIAGNOSIS.md` | The diagnostic drill. Five symptom-shaped RAG questions worked end to end — sudden regression, the retrieval-generation gap, proving an embedding change helped, five-document synthesis, and 10k to 1M documents — each with the ordered hypotheses, the experiment that discriminates between them, and the follow-ups. |
| `RAG_LATENCY_IN_PRODUCTION.md` | The latency chapter. Where the milliseconds actually go (with measured numbers, and a warning about the invented ones), how to set a budget and read percentiles, the five-level optimization hierarchy, caching including semantic caching in full, and the worked answer to "how would you optimize this for latency." |


## Start with the round you are actually in

The six `LOOP_*_QA.md` files are the substance of this folder: **294 questions with the answers written
out in full.** They are organized by round, because that is how a loop is organized and because the same
topic is graded differently in different rooms.

Work them by covering the answer, saying yours out loud, then uncovering and comparing. The gap between
what you said and what is written is almost always the same two things — the tradeoff you did not name,
and the number you did not have.

`THE_LOOPS.md` is the map, not the material: which rounds exist, what each one grades, and how the
pipeline is shaped. Read it once so you know which files to work, then spend your time in the banks.


---

## Why the process chapter comes first

The most common way a strong candidate fails is not "did not know enough." It is preparing for a
different role's loop than the one they are in. "AI/ML role" is at least five separate jobs — AI
engineer, research scientist, applied scientist, ML engineer, MLOps — and the same answer scores
differently in each. An applied scientist loop will ask you to defend a modeling choice statistically.
An AI engineer loop will ask what happens when the tool call times out. Both are reasonable; guessing
wrong costs you the loop before you say anything technical.

So `THE_LOOPS.md` opens with the taxonomy, and the single highest-leverage thing in this folder is the
advice to just ask your recruiter which one you are in. They will tell you. Asking reads as prepared,
not presumptuous.

## A note on evidence

Hiring processes are badly documented, and the gap gets filled by content farms writing confident
fabrications. Both chapters mark evidence level inline — `[official]` for something a company publishes
itself, `[practitioner]` for a working interviewer describing what they do, `[one report]` for a single
named candidate's experience, `[aggregated]` for crowd-sourced, `[my read]` for my own inference offered
as opinion.

Where I could not verify a company's process, it is absent rather than guessed. Several well-known
labs are missing from the comparison table for exactly that reason, and there is an employer-*type*
comparison in their place, which is the part that generalizes anyway.

The question bank carries a related marker: `[verify before quoting]` on anything that depends on facts
that move — model capabilities, context limits, pricing, tooling. Those answers were right when written
and will rot on a schedule.

## On question recycling

While assembling the bank I checked several of the widely circulated question lists against each other
and found substantially the same questions in the same order across independent sites. That is worth
knowing in both directions. It means the public question surface is small and genuinely learnable. It
also means an interviewer who has seen those lists is deliberately asking around them, which is why
§9 (judgement questions) and the follow-up chains matter more than the rapid-fire section does.

Section 11 documents where the questions came from and, more usefully, what was deliberately left out
and why.

---

## Why there is a whole chapter on RAG failures

`RAG_FAILURE_DIAGNOSIS.md` exists because of a distinct question type that the other two chapters do not
cover well. The question bank asks *what is hybrid search and why* — component knowledge, answerable by
anyone who has built one. The diagnostic chapter asks *your system suddenly started giving wrong answers,
what do you investigate and how would you prove it* — which is answerable only by someone who has
operated one.

The gap between those two is where most candidates lose applied GenAI loops. Building teaches you the
happy path and gives you a vocabulary for components. Operating teaches you the failure surface, which
is much larger and shaped completely differently. Asked a symptom question, a builder lists components,
and the interviewer hears someone who has never had one of these break at 2am.

The chapter is organized so the transferable part comes first: the five moves every diagnostic answer
makes, of which the one nearly everybody skips is saying how you would *prove* the cause. That is the
part being graded. Memorizing the five answers does not survive an interviewer changing one detail;
knowing the shape does.

---

## How this folder relates to the others

- `13_interview_qa` and `56_spoken_interview_question_bank` are the classical ML and general question
  banks. This folder does not repeat them — it assumes an AI engineer loop will still drop a
  bias-variance question on you, it just is not the discriminating part.
- `64_integrated_ai_ml_interview_synthesis` connects topics across the repository. This folder is
  narrower and more current.
- `73_night_before_review` is the compressed retrieval pass. Read this folder weeks out; read that one
  the evening before.
- `50_ml_coding_interview_patterns` and `68_leetcode_patterns` cover the coding round, which neither
  chapter here duplicates.
- Depth on any answer here lives in the numbered topics — retrieval mechanics in
  `39_rag_retrieval_augmented_generation`, serving and latency in `63_paged_attention_and_llm_serving`,
  alignment and safety in `66_frontier_alignment_rl`.

## How to use it

Read `THE_LOOPS.md` once, early — ideally before you start preparing at all, because it tells you what
to prepare. Then work `MODERN_QUESTION_BANK.md` the way you would work any question bank: answer out
loud before reading the answer, and pay attention to the gap between what you said and what is written.
That gap is almost always the same thing — the tradeoff you did not name, or the number you did not
have.
