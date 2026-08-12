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
| `THE_LOOPS.md` | The process. Five distinct roles that share vocabulary and nothing else, the standard pipeline stage by stage, what each round type is actually grading, a fully worked real loop, take-homes, behavioral, and reading the room. |
| `MODERN_QUESTION_BANK.md` | The content. Eleven sections of applied GenAI questions — RAG, agents, prompting and context, evaluation, LLMOps, deployment, safety, judgement — plus 65 rapid-fire, each with the answer written the way you would say it out loud. |

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
