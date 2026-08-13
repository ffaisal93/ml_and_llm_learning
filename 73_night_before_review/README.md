# Topic 73: The Night Before

Everything else in this repository is for learning. This folder is for the evening before an interview,
when learning is over and the only useful activity is refreshing what you already know.

## Files in this folder

| File | Purpose |
|---|---|
| `CODE_FROM_MEMORY.md` | The implementations you must be able to write cold — softmax, linear and logistic regression, attention, multi-head attention, a transformer block, the training loop — in NumPy and PyTorch. Every snippet verified by execution. |
| `FORMULA_SHEET.md` | One page of losses, gradients, activations, norms, optimizers, metrics, and scaling rules. For staring at, not reading. |
| `DEPTH_AND_BREADTH_QA.md` | Ten depth ladders that follow a topic down five levels, ~100 rapid-fire breadth questions, the ones people fumble, and questions to ask them. |
| `AI_ENGINEER_ONE_PAGER.md` | The applied GenAI compression — RAG failure modes, the latency budget and optimization hierarchy, what breaks at scale, OWASP for RAG, and the things people get backwards. For an AI/GenAI engineer loop rather than a classical ML one. |

---

## How to use this tonight

The single most common mistake the night before an interview is trying to learn something new. It does
not stick, and it costs you the sleep that would have made everything you *do* know accessible. Tonight
is for retrieval, not acquisition.

A ninety-minute pass that works:

**First twenty minutes — the formula sheet.** Read it once, slowly. Do not take notes. You are refreshing
recognition, not building anything.

**Next forty minutes — the code.** Read `CODE_FROM_MEMORY.md` through once. Then close it and retype two
or three of the implementations from memory in a blank editor. Pick the ones you feel least sure about,
which for most people means scaled dot-product attention and the multi-head reshape. This step is the
highest-value thirty minutes available to you, because recognition and recall are different skills and
only recall gets tested.

**Next twenty minutes — the depth ladders.** Read them out loud. Not silently. The gap between "I
understand this" and "I can say this fluently" is exactly what an interview measures, and speaking is the
only way to find that gap before someone else does.

**Last ten minutes — the fumble list.** Skim the questions that sound easy and are not. These are cheap
points and they are lost by people who know the material perfectly well.

**If tomorrow is an AI/GenAI engineer loop rather than a classical ML one**, swap the middle forty
minutes for `AI_ENGINEER_ONE_PAGER.md` and read it twice. That loop asks about retrieval quality,
evaluation, latency, and failure modes rather than derivations, and the code round is usually ordinary
software engineering rather than implementing attention. The formula sheet is still worth twenty minutes
— a bias-variance or attention question will still land, it just is not the discriminating part.

Then stop. Sleep matters more than the next hour of review, and that is not a motivational sentiment —
recall is measurably worse when tired, and interviews test recall.

---

## The morning of

Skim the formula sheet once over coffee. Retype scaled dot-product attention one final time. Do not open
anything you have not already seen; a half-remembered new fact is worse than no fact, because it feels
like knowledge while you are saying it.

Have ready, in a sentence each: one project you can describe end to end, one thing you got wrong and what
you changed, and one thing you would do differently with more time. Interviewers ask some version of all
three, and having thought about them beforehand is the difference between a considered answer and a
rambling one.

---

## What this folder assumes

That you have already worked through the material. The compression here only helps if there is something
underneath it to decompress — a formula sheet read by someone who never derived the formulas is just a
list of symbols.

If you are more than a day out, close this and go read the topic folders properly. Come back the night
before.

---

## Cross-references

- Full derivations for anything on the formula sheet live in the numbered topic folders — logistic
  regression in `01_classical_ml`, optimization in `02_gradient_descent`, attention in
  `05_attention_mechanisms`, scaling in `70_scaling_laws`.
- `13_interview_qa` and `56_spoken_interview_question_bank` are the longer question banks; this folder is
  the compressed pass over them.
- `74_ai_engineer_interview_prep` is what `AI_ENGINEER_ONE_PAGER.md` compresses — the interview loop
  itself, the modern question bank, RAG failure diagnosis, and latency engineering. Read that folder
  weeks out; read the one-pager tonight.
- `50_ml_coding_interview_patterns` and `68_leetcode_patterns` cover the coding round, which this folder
  deliberately does not.
