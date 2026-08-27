# Topic 29: ML System Design

The open-ended round — "design YouTube's recommender" — and the one candidates most often
under-prepare for, because it is not really a modelling question.

> **Start here:** [`ML_SYSTEM_DESIGN_DEEP_DIVE.md`](ML_SYSTEM_DESIGN_DEEP_DIVE.md). It is a single long document meant to be read
> straight through once and then dipped into. [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md) is 55 active-recall questions for
> after you have read it.

## What is in the deep dive

The document is in three parts, and they build.

**Part 1 — foundations and vocabulary.** How the round works and where the forty-five minutes go, the
six-step framework, and then the section that most people actually need: every piece of production
infrastructure explained from zero. What Redis is and what problem an in-memory key-value store solves.
What Kafka is and why you would use a distributed log instead of calling a service. What a feature store
is for, and what point-in-time correctness means — carefully, since it is the most misunderstood idea in
the area. What Prometheus scrapes and what Grafana does and does not do. What an SLO and an error budget
are, with the arithmetic worked. And an extended treatment of automated evaluation: eval suites,
regression gates in CI, how golden sets rot, LLM-as-judge and its failure modes, and why offline and
online results disagree.

Each entry says what the thing is in one sentence, what the world looked like before it existed, how it
works mechanically, where it shows up in an ML system, and when it is the wrong choice.

**Parts 2 and 3 — seven designs, worked end to end.** A video recommender, web search ranking, ads
ranking, fraud detection, content moderation, an LLM serving platform, and semantic image search. Each
one is: the scenario in the interviewer's voice, what the question is really testing, then six steps —
clarify, frame as an ML problem, data, architecture, evaluation, production — followed by the one hard
tradeoff, the follow-ups they will ask, and a spoken summary you can deliver in ninety seconds.

The seven are ordered so they build. The first four share the retrieve-and-rank shape and vary the hard
part: corpus scale, then biased labels, then calibration, then adversarial drift with a hard deadline.
The last three deliberately break that shape, because an interviewer who has heard retrieve-and-rank
three times will reach for a prompt where it does not apply.

## How to use it

Every term is defined where it is first used, so you can read start to finish without a second tab open.
Roughly thirty **"Saying it out loud"** blocks give you the words in natural speech — those are the
deliverable, and the way to use them is to cover the block, say your own version, then compare.

Then drill: twenty-five minutes on one prompt, out loud, standing up, timer visible. The parts you skip
under time pressure are always the same three — the clarifying questions, the failure modes, and how you
would evaluate it — and those are exactly the parts that separate a senior answer.

## Scope

This is **platform-scale** system design. For product and business case studies — "design churn
prediction," where the work is framing a business problem rather than serving a billion requests — see
`28_business_use_cases/`.

## Next

- [`30_ab_testing`](../30_ab_testing/README.md) — how you decide whether the thing you designed actually shipped an improvement.
- [`39_rag_retrieval_augmented_generation`](../39_rag_retrieval_augmented_generation/README.md) — the same exercise for a retrieval-augmented LLM.
- [`06_llm_inference`](../06_llm_inference/README.md) and [`63_paged_attention_and_llm_serving`](../63_paged_attention_and_llm_serving/README.md) — the serving internals behind Design 6.
- [`74_ai_engineer_interview_prep/LOOP_SYSTEM_DESIGN_QA.md`](../74_ai_engineer_interview_prep/LOOP_SYSTEM_DESIGN_QA.md) — fourteen more design prompts, worked in the
  same shape.
