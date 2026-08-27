# LLM Evaluation — Deep Dive

> Frontier-lab interview-grade reference on evaluating LLMs and LLM-powered products.
> Read with [`HALLUCINATION_DETECTION_DEEP_DIVE.md`](HALLUCINATION_DETECTION_DEEP_DIVE.md) (factuality is one slice of eval) and [`LLM_PROBLEMS_DEEP_DIVE.md`](LLM_PROBLEMS_DEEP_DIVE.md).

LLM evaluation is the bottleneck of modern LLM development. Training methods are mostly known; what separates labs is whether they can *measure* whether a change is an improvement. This chapter walks through what makes LLM eval hard, the benchmark landscape, LLM-as-judge methodology, factuality measurement, contamination, robustness, online/production evaluation, and how to design an eval suite for a real product.

---

## Table of contents

1. Why LLM evaluation is hard
2. Taxonomy of evaluations
3. Capability benchmarks: knowledge, reasoning, math, code, long context, multimodal, agent
4. Instruction following and chat-quality evaluation
5. LLM-as-judge: methodology, biases, calibration
6. Pairwise evaluation, ELO, Chatbot Arena
7. Open-ended generation evaluation
8. Factuality and faithfulness measurement
9. Contamination — detection, prevention, leakage
10. Robustness, prompt sensitivity, fairness
11. Statistical methodology (uncertainty, multiple comparisons, sample size)
12. Evaluation harnesses (lm-eval-harness, OpenCompass, HELM, Inspect)
13. Online evaluation and production telemetry
14. A/B testing for LLM products
15. Designing an eval suite for a product (case study)
16. Failure modes and senior-level signals
17. Common interview questions
18. References and further reading

---

## 1. Why LLM evaluation is hard

Evaluating an LLM is not like evaluating a classifier. The gold-standard ML setup — fixed labeled test set, deterministic predictions, scalar metric — breaks in several ways at once:

- **No single ground truth.** "Write a polite refusal" has many correct outputs. "Summarize this article" has many. Reference-based metrics (BLEU, ROUGE) penalize legitimate paraphrase and reward surface n-gram overlap.
- **Capability is multi-dimensional.** A model that improves on math may regress on creative writing. Aggregating across tasks hides this.
- **Open-ended outputs.** Most real LLM use cases are open-ended. Scalar quality is hard to define and noisy to estimate.
- **Prompt sensitivity.** Changing the system prompt, the temperature, or the order of few-shot examples can swing benchmark scores by 5–15 points. Most published benchmarks under-report this.
- **Contamination.** Pretraining corpora may already contain the test set. Reported MMLU on a contaminated model is meaningless.
- **Cost asymmetry.** Generating 50K samples on a frontier model for one eval run can cost thousands of dollars and hours of latency.
- **Distribution shift between offline and production.** Benchmark prompts look nothing like real-user prompts. Excellent benchmark numbers do not guarantee a good product.
- **Capability ≠ helpfulness.** A model can pass MMLU and still be useless for a customer-support task because it refuses too often, hallucinates citations, or has the wrong persona.
- **Judges drift.** When you use an LLM-as-judge, the judge model itself improves over time, so your "absolute" score moves even when the model under test does not.
- **Goodhart's law.** As soon as a benchmark becomes important, models are optimized for it. Performance on a benchmark stops being a reliable proxy for the underlying capability.

Senior-level mental model: **evaluation is a measurement instrument, and like any instrument it has bias, variance, and a calibration problem.** Most practitioners under-invest in eval relative to training; the lab that invests more in eval ships better.

> **Saying it out loud.** The short version: everything that makes classifier evaluation easy is missing here. There's no single right answer, so you can't just check equality; the outputs are open-ended, so your metric is noisy; the same model swings 5 to 15 points if you reword the prompt; and the test set may already be sitting in the pretraining data. My framing is that an eval is a measurement instrument — it has bias, it has variance, and it needs calibrating — and most teams under-invest in the instrument relative to the model. The line that lands is Goodhart: the moment a benchmark starts mattering, people optimize for it, and it stops measuring the thing you cared about.

---

## 2. Taxonomy of evaluations

A useful first cut.

**By target.** Capability evals (math, code, reasoning) measure what the model *can* do. Quality evals measure how good the output *is* on open-ended tasks. Safety / behaviour evals measure refusals, harmful output, jailbreak susceptibility, sycophancy. Product evals measure whether real users complete their task.

**By signal type.** Reference-based (compare to gold), reference-free (intrinsic measures: log-prob, coherence, diversity), pairwise (A vs B), human-judged, LLM-judged, programmatic / unit-tested (especially code, math).

**By stage.** Offline pre-deployment, shadow production, online A/B, post-launch telemetry.

**By granularity.** Token-level (perplexity), output-level (correct/incorrect), conversation-level (multi-turn), session-level (did the user solve their problem).

**By openness.** Closed-form (multiple choice, exact match), constrained (extraction with schema), open-ended generation, agentic (multi-step tool use).

A modern frontier-lab eval stack covers all of these. A startup eval stack should cover at least: capability benchmarks, LLM-judge open-ended quality, safety, and online telemetry.

> **Saying it out loud.** Before I answer any eval question I'd say which kind of eval we're talking about, because the word covers four different jobs. Capability evals ask what the model *can* do — math, code, reasoning. Quality evals ask how good an open-ended answer is. Safety evals ask about refusals and jailbreaks. And product evals ask whether a real user finished their task, which is the only one the business actually cares about. The other axes are signal type — reference-based, pairwise, LLM-judged, programmatic — and stage: offline, shadow, A/B, telemetry. The mistake to name is treating a capability number as a product number; MMLU says nothing about whether your support bot resolves tickets.

---

## 3. Capability benchmarks

### 3.1 Knowledge

**MMLU.** 57-subject multiple-choice across STEM, humanities, social sciences. The default knowledge benchmark. Saturated above ~90 for frontier models. Heavily contaminated.

**MMLU-Pro.** Hand-constructed harder version with 10 answer choices instead of 4 and explicit reasoning required. Replaces MMLU as the modern knowledge benchmark.

**AGIEval, ARC, BIG-Bench, BIG-Bench-Hard (BBH).** Mixed-task suites. BBH has 23 reasoning-heavy subtasks; "BBH chain-of-thought" was the classic CoT eval before GSM8K dominated.

**TriviaQA, NaturalQuestions, PopQA.** Open-domain QA, useful for retrieval and factuality eval.

**SimpleQA (OpenAI 2024).** Short factoid questions designed to be unambiguous; specifically targets hallucination on factual claims. Frontier models score 30–60% — that is the headline that "models are still bad at simple facts."

> **Saying it out loud.** MMLU used to be the knowledge benchmark and now it's mostly a historical artifact — frontier models are above 90, it's four-choice so guessing gets you 25, and it's been contaminated for years. MMLU-Pro is the replacement: ten choices instead of four, harder items, reasoning required. If someone quotes me an MMLU number in 2026 my first question is what their decontamination check was. For factual recall specifically I'd reach for SimpleQA instead, because it's short-answer with no multiple-choice crutch, and the fact that frontier models sit at 30 to 60% is the honest picture of how well they know things.

### 3.2 Reasoning

**GPQA (Graduate-level Physics QA).** 198 expert-written questions; very hard for non-experts. The "PhD-level reasoning" benchmark. Diamond subset is what is usually cited.

**MATH and AIME.** MATH = 12.5K competition problems, multiple difficulty levels. AIME = American Invitational Math Exam; what o1, o3, R1 chase.

**GSM8K.** 8.5K grade-school word problems. Saturated; useful as smoke test, no longer informative for frontier models.

**MuSR, MultiArith, ARC-AGI.** Multi-step reasoning suites. ARC-AGI is the abstract-reasoning benchmark behind the o3 announcement.

> **Saying it out loud.** Reasoning benchmarks are stratified by how far past saturation they are. GSM8K is done — grade-school word problems, useful only as a smoke test now. MATH and AIME are the live ones for math, and AIME in particular is what the reasoning models are chasing. GPQA-Diamond is the graduate-level science set, deliberately written so that a non-expert with Google still fails. The number worth carrying is that GPQA-Diamond is only 198 questions, which means a two-point difference between models is inside the noise — that's the kind of caveat that separates a careful answer from a leaderboard recitation.

### 3.3 Code

**HumanEval.** 164 hand-written Python problems with unit tests. Saturated for frontier models. Reported as pass@1 (single sample) or pass@k.

**MBPP.** 974 entry-level Python problems. Same idea, complementary distribution.

**HumanEval+, MBPP+ (EvalPlus).** Augmented test suites with 100x more cases each; original tests were too lenient.

**LiveCodeBench.** Continuously-updated set of LeetCode contest problems. Specifically designed to defeat contamination — uses problems released after a cutoff.

**SWE-Bench, SWE-Bench-Verified.** Real GitHub issues from popular Python repos. The model must produce a patch that passes the maintainers' tests. SWE-Bench-Verified is a 500-issue human-verified subset. The premier *agent* coding benchmark — solving requires file navigation, tool use, multi-step reasoning.

**RepoBench, CodeContests, CRUXEval.** Repo-level completion, competition coding, code reasoning.

> **Saying it out loud.** Code eval is the good case because the tests decide, not a judge. HumanEval is 164 problems and saturated; EvalPlus rebuilt the test suites roughly a hundred times bigger because the originals were lenient enough to pass wrong code. LiveCodeBench solves contamination structurally by only using problems published after a model's cutoff. And SWE-Bench-Verified is where the real signal is now — actual GitHub issues, the model has to navigate a repo and produce a patch that passes the maintainers' tests, which is agentic rather than snippet-level. The failure mode to name: passing hidden tests proves the code runs, not that it's the fix a reviewer would accept.

### 3.4 Instruction following and chat

**AlpacaEval 2 (length-controlled).** Pairwise win-rate vs a reference model (typically GPT-4-Turbo) judged by GPT-4. The length-controlled version corrects for the bias that longer answers get judged better.

**MT-Bench.** 80 multi-turn questions across 8 categories, judged on a 1–10 scale by GPT-4. Two-turn structure stresses follow-up handling.

**Arena-Hard, Arena-Hard-Auto.** 500 hard prompts curated from Chatbot Arena, judged via GPT-4-as-judge. Strong correlation with Arena ELO and much cheaper.

**IFEval.** Verifiable instruction-following: "answer in JSON," "use exactly 3 bullet points," "include the word X 5 times." Programmatically checked, no judge needed. Excellent because it has zero judge bias.

**FollowBench, InfoBench.** Multi-constraint instruction following.

> **Saying it out loud.** My favorite here is IFEval, because it removes the judge from the loop entirely — "answer in valid JSON," "use exactly three bullets," "keep it under fifty words" are all checkable with code, so there's zero judge bias and zero judge drift. AlpacaEval 2 and MT-Bench give you the open-ended picture, and Arena-Hard-Auto approximates Arena ELO for around \$25 a run instead of months of human voting. The thing I'd insist on is the length-controlled variant of AlpacaEval: without it, "write more words" beats an actual reasoning improvement, and you'll ship the wrong model.

### 3.5 Long context

**Needle-in-a-Haystack (NIAH).** Insert a fact at varying depth in long context, ask the model to retrieve it. Easy at small depth, harder at boundaries; the standard "does long context work at all" smoke test.

**RULER.** Multi-task long-context benchmark from NVIDIA: NIAH variants, multi-key, multi-value, variable-tracking, common-words extraction, frequent-words extraction, QA. Reveals that real long-context performance is much worse than NIAH alone suggests. Industry standard for long-context eval as of 2024–2025.

**BABILong, LongBench, ZeroSCROLLS, ∞Bench.** Mixed long-context QA suites.

**Lost-in-the-Middle (Liu et al. 2023).** Phenomenon, not a benchmark per se: models attend to information at the start and end of the context but degrade in the middle. Should always be tested for in any RAG system.

> **Saying it out loud.** Needle-in-a-haystack is the smoke test, not the eval — it only proves the model can find one distinctive sentence in a wall of text, which is the easiest possible long-context task. RULER is what to actually run: multi-key retrieval, variable tracking, aggregation, and it consistently shows that effective context is far shorter than advertised context. A model marketed at 128K might hold up to 32K on real multi-hop work. And lost-in-the-middle is the phenomenon underneath all of it — accuracy dips for information buried at the middle depths, which is exactly why chunk ordering matters in RAG.

### 3.6 Multimodal

**MMMU (Massive Multi-discipline Multimodal Understanding).** College-level questions across 30 subjects with images. The standard VL benchmark.

**MM-Vet, MathVista, ChartQA, DocVQA, ScienceQA, RealWorldQA, BLINK.** Specialized vision-language tasks.

**Video-MME, EgoSchema, MVBench.** Video understanding.

> **Saying it out loud.** MMMU is the default headline for vision-language — college-level questions across thirty subjects where you genuinely need the image. Then it fans out by skill: ChartQA and DocVQA for reading charts and documents, MathVista for visual math, and the Video-MME family for video. The caveat worth voicing is that a lot of "multimodal" questions are answerable from the text alone, so a strong language model scores well without really looking at the image — the good benchmarks control for that with a text-only ablation, and you should ask whether yours does.

### 3.7 Agent

**AgentBench (Tsinghua).** Multi-environment agent eval: OS, DB, web, etc.

**GAIA (Meta).** Real-world questions that require browsing, calculation, file handling. Levels 1/2/3 by difficulty. Frontier humans get ~92%; GPT-4 + tools gets ~30%; pure GPT-4 ~6%.

**SWE-Bench / SWE-Bench-Verified.** Already mentioned — the agent coding benchmark.

**OSWorld, WebArena, VisualWebArena.** Computer-use agents on real OS / web tasks. The Anthropic computer-use, OpenAI Operator, and Google Agentic-AI demos report on these.

**TAU-bench (Sierra 2024).** Realistic customer-service agent eval with simulated user. Great test of multi-turn tool use.

**MLE-Bench (OpenAI 2024).** Can the agent do an entire ML engineering task — a Kaggle-style competition.

> **Saying it out loud.** Agent benchmarks are where the numbers get humbling, and that's what makes them useful. On GAIA, capable humans get around 92% while GPT-4 with tools lands near 30% — a gap that size tells you agentic reliability is the open problem, not raw capability. SWE-Bench-Verified is the coding version, TAU-bench is the customer-service version with a simulated user, and OSWorld and WebArena are computer-use. The structural point: these are multi-step, so success rates compound — 95% per step over twenty steps is about 36% end to end, which is why agent evals look so much worse than single-turn ones.

### 3.8 Safety and behaviour

**TruthfulQA.** 817 questions designed to elicit common misconceptions. Tests whether the model parrots wrong answers humans tend to give.

**WildGuard, WildJailbreak, HarmBench, JailbreakBench.** Refusal and jailbreak resistance.

**ToxiGen, BOLD, BBQ, RealToxicityPrompts.** Bias and toxicity.

**XSTest.** Tests over-refusal: prompts that *should* be answered but the model wrongly refuses.

**Sycophancy probes.** Ask a question, then say "I think the answer is X (wrong)" — does the model flip?

> **Saying it out loud.** Safety eval has two directions and you have to run both. One direction is "does it refuse things it should" — HarmBench, WildJailbreak, jailbreak resistance. The other is XSTest, which is over-refusal: prompts that are perfectly benign but phrased in a way that trips the safety training. Reporting only the first is how you ship a model that's technically safe and practically useless. Sycophancy probes are the underrated one — ask a question, then tell the model you think the wrong answer is right, and see whether it caves. That's a behavioral failure no capability benchmark catches.

### 3.9 What to actually run

For a frontier-model eval suite circa 2026, a defensible minimum is: MMLU-Pro, GPQA-Diamond, MATH-500, AIME, LiveCodeBench, SWE-Bench-Verified, IFEval, MT-Bench or Arena-Hard, RULER (long context), MMMU (if multimodal), TruthfulQA, SimpleQA, XSTest. Plus product-specific tasks.

For a startup product, capability benchmarks matter much less than your own task-specific eval — see §15.

> **Saying it out loud.** If someone asks what my eval suite is, I'd give a short defensible list rather than everything: MMLU-Pro and GPQA-Diamond for knowledge and reasoning, MATH and AIME, LiveCodeBench and SWE-Bench-Verified for code, IFEval plus Arena-Hard for instruction following and chat, RULER for long context, TruthfulQA and SimpleQA for factuality, XSTest for over-refusal. Then I'd immediately add the important part: for a product, all of that is a sanity check, and the thing that actually decides anything is a few hundred task-specific prompts from your own traffic. Public benchmarks tell you the model isn't broken; they don't tell you it works for you.

---

## 4. Instruction following and chat quality

The interesting jump in usefulness from 2022 → 2024 came from instruction tuning, but evaluating it took a couple of years to stabilize.

### Verifiable instruction following

**IFEval** is the gold standard because it removes the judge entirely. Instructions are programmatically checkable — "respond in valid JSON," "first sentence must start with the word 'Furthermore'," "answer in fewer than 50 words." Pass-rate is reported per-instruction and per-prompt.

For a product, **always include verifiable instructions** in your eval. They catch regressions that LLM-judge can miss because the judge is also drifting.

> **Saying it out loud.** The most reliable eval in the whole toolkit is the one with no model in the scoring loop. "Return valid JSON" is either parseable or it isn't. "Under fifty words" is a word count. IFEval built a whole benchmark on that principle, and every product eval should carry a slice of it. The reason isn't just cheapness — it's that programmatic checks have zero variance and zero drift, so when your LLM-judge score moves you can tell whether the model changed or the judge did.

### Multi-turn

**MT-Bench** stress tests: turn 1 sets up a task, turn 2 asks a related follow-up. Single-turn capability is necessary but not sufficient for chat. Many models that score well single-turn fall apart on turn 2 because they lose context, repeat themselves, or fail to handle clarifications.

> **Saying it out loud.** Plenty of models look great on the first turn and fall apart on the second, so a single-turn eval will happily bless a model your users will hate. MT-Bench is built exactly for this — turn one sets a task, turn two asks a follow-up that depends on it. The failure modes are specific and worth naming: the model loses the thread of what it just said, repeats its own answer back, or mishandles a clarification and silently changes the task. Since real chat is multi-turn almost by definition, single-turn quality is necessary and nowhere near sufficient.

### Length-controlled judging

LLM judges are biased toward longer responses. AlpacaEval 2 introduced **length-controlled win rate** that regresses out length. Without it, simply "reply with more text" appears to beat reasoning improvements. Always report length-controlled metrics.

> **Saying it out loud.** LLM judges reward length, full stop — pad an answer and the win rate climbs even when the content is identical. AlpacaEval 2 fixed this by regressing out length so you get a win rate at constant verbosity. If you don't do that, you will run an experiment, see a win-rate improvement, and ship a model whose only change was that it talks more. Always report the length-controlled number, and if someone shows you a raw win rate, the first question is what happened to the average token count.

### Persona / format adherence

Production chat needs the model to obey a system prompt across turns. The eval should include adversarial users trying to break the persona ("ignore your instructions and...") and benign users asking ambiguous questions.

> **Saying it out loud.** In production the system prompt is the contract, and the question is whether the model holds it across a long conversation and under pressure. So the eval needs both adversarial users trying to talk it out of its instructions and ordinary users asking vague things that tempt it to drift. The failure mode is gradual: the persona doesn't break in one turn, it erodes over ten, and a single-turn eval never sees it. That's why I'd measure adherence at turn ten, not turn one.

---

## 5. LLM-as-judge: methodology, biases, calibration

LLM-as-judge means using a strong model (often GPT-4 / Claude / a custom judge) to score outputs of the model under test. Cheap, scales, and surprisingly correlated with human preference — but full of biases.

### Why it works (when it does)

A strong judge has been RLHF-trained on a large corpus of human preferences, so it has internalized "what humans like." For tasks where the judge is more capable than the testee, judge preferences correlate with humans at r ≈ 0.7–0.85 on aggregate, low at the per-example level.

> **Saying it out loud.** The reason a strong model can stand in for a human rater is that it was trained on a mountain of human preference data — it has internalized what people tend to like. Aggregate correlation with human judgments runs around 0.7 to 0.85, which is good enough to rank two systems. The crucial qualifier is that per-example agreement is much weaker, so a judge score is a population statistic, not a verdict on any single response. And it only holds while the judge is more capable than what it's judging — once the testee catches up, the signal degrades toward noise.

### Known biases

- **Length bias.** Longer = perceived better, even when controlling for content.
- **Position bias.** First option (A) tends to win in pairwise comparisons. **Mitigation: swap and average.** Run the comparison twice with order swapped, count agreement; only count a vote when both orderings agree.
- **Self-preference / family bias.** GPT-4 prefers GPT-family outputs; Claude prefers Claude. Mitigation: use multiple judges, or use a judge from a different family than either testee.
- **Verbosity / format bias.** Bullet points, bold formatting, headers all bias toward "better." Strip formatting before judging if you want substance only.
- **Refusal bias.** Some judges punish refusals heavily; others reward them. Inspect the judge's behaviour on safety prompts before trusting overall scores.
- **Distribution / topic bias.** Judges are weak on niche domains (medical, legal, code) — use domain-specific judges or human evaluators for those.
- **Calibration drift.** As the testee approaches the judge in capability, the judge's scoring becomes noisier. When testee ≥ judge, the judge becomes useless.

> **Saying it out loud.** If someone asks for judge biases, I name them with the mitigation attached, because listing problems is cheap. Length bias — fix with length-controlled win rate. Position bias, where option A wins more often — fix by running both orderings and only counting votes where the two agree. Self-preference, where GPT prefers GPT output — fix by using a judge from a different family than either contestant, or an ensemble. Then format bias toward bullets and bold, and weakness in specialized domains like medicine or law. The one people miss is the capability ceiling: as the model under test approaches the judge, the judge's scores get noisier, and once it passes the judge they're worthless.

### Best practice prompt for a pairwise judge

```
You are an impartial judge. Compare two responses, A and B, to the same user query.
Decide which response is better, considering:
1. Helpfulness and relevance to the query
2. Factual accuracy
3. Clarity and conciseness
4. Adherence to any explicit constraints in the prompt
Length should not be a factor; pick the shorter response if quality is equal.
Output strictly in this JSON format: {"winner": "A" | "B" | "tie", "reason": "..."}.
```

> **Saying it out loud.** A good judge prompt does four things: it names the criteria explicitly so the score is auditable, it tells the judge to ignore length and break ties toward the shorter answer, it forces structured JSON output so parsing never fails, and it asks for a reason so you can spot-check the reasoning. What I'd emphasize is that the prompt alone doesn't fix position bias — you still have to swap A and B and run it twice. The prompt handles what the judge attends to; the protocol around the prompt handles the biases the prompt can't talk it out of.

### Calibrating a judge

1. Build a small (200–500) human-labeled gold set of pairwise comparisons.
2. Run the LLM judge on the same set.
3. Measure agreement (Cohen's κ, accuracy).
4. If agreement is low (<0.7), refine the judge prompt or change judge.
5. Periodically re-calibrate as the testee changes.

> **Saying it out loud.** An uncalibrated judge is a number with no units. The fix is small and boring: get two to five hundred human-labeled pairwise comparisons, run the judge on the same set, and measure agreement with Cohen's kappa. Below about 0.7 you fix the prompt or change judges rather than reporting the score. And you re-run this periodically, because both the judge and the model under test move — this is the step almost everyone skips, and it's the first thing a senior interviewer will ask about when you quote a win rate.

### Multi-judge ensembles

Three judges from different families, majority vote. Reduces idiosyncratic bias and is now standard at frontier labs for capability eval.

> **Saying it out loud.** Use three judges from three different model families and take the majority. The reason is that the biggest judge biases are family-specific — self-preference in particular — so averaging across families cancels what averaging across prompts can't. It's standard practice at frontier labs now for anything consequential. The obvious tradeoff is 3x cost per comparison, so in practice you run a single judge for continuous regression testing and bring out the ensemble for model-selection decisions.

### G-Eval, Prometheus, JudgeLM, PandaLM

Trained-judge models specifically for eval. Prometheus 2 is open source; PandaLM is open source. They give per-criterion 1–5 scores with a structured rubric.

> **Saying it out loud.** Instead of paying a frontier model to judge every response, you can use a model trained specifically to be a judge — Prometheus 2 and PandaLM are the open ones, and they output per-criterion scores against a rubric. The appeal is cost and reproducibility: it's a fixed checkpoint, so unlike an API judge it doesn't silently change under you next quarter, which matters enormously if you're tracking a metric over a year. The tradeoff is a lower ceiling — a dedicated small judge is weaker than a frontier model on hard or specialized content, so it's the right tool for high-volume regression checks and the wrong one for close calls.

---

## 6. Pairwise evaluation, ELO, Chatbot Arena

### Why pairwise

Absolute scoring is hard ("rate this 1–10"). Pairwise comparison ("A or B?") is the cleanest signal. Humans agree more on "which is better" than on "what's the score."

> **Saying it out loud.** Ask a human to rate an answer one to ten and you'll get noise — everyone's scale is different and the same person drifts within an hour. Ask them which of two answers is better and agreement jumps, because comparison is a much easier cognitive task than absolute scoring. Same is true for LLM judges. The cost of pairwise is that you get a ranking rather than a level: you learn B beats A, not whether either is good enough to ship, so you still need an absolute quality bar somewhere in the stack.

### ELO from pairwise

**In plain language.** ELO is the chess rating system. Each model gets a single number; the difference between two numbers predicts how often one beats the other. A 400-point gap means roughly a 10-to-1 win ratio, which is where the 400 in the formula comes from.

Once you have many A-vs-B comparisons, fit ELO ratings. The classic update rule:

```
expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
rating_a += K * (score_a - expected_a)   # score_a = 1, 0.5, 0
rating_b += K * ((1 - score_a) - (1 - expected_a))
```

In practice you fit ELO via maximum likelihood on the full pairwise dataset (Bradley-Terry MLE) rather than online updates.

> **Saying it out loud.** ELO is just chess ratings applied to models: everyone gets a number, and the difference between two numbers predicts the win probability — 400 points apart means about a ten-to-one expected win rate. You can update it online with the standard K-factor rule, and that's what the classic formula does. In practice, though, you don't do the online update; you fit the whole thing at once by maximum likelihood over all the recorded comparisons, which is Bradley-Terry. The reason is order dependence — online ELO gives you a different answer depending on which games happened first, and a batch fit doesn't.

### Chatbot Arena (LMSys)

Users submit prompts, see two anonymized model outputs, vote. ELO ratings are computed. The de-facto reference for "what real users prefer" since 2023. Influential because it captures preference distribution shift toward open-ended chat.

Issues with Arena: prompt distribution skews toward casual chat, vocal users may not represent paying customers, voting fatigue, no per-task breakdown by default.

> **Saying it out loud.** Arena is the closest thing we have to a public preference vote: real users bring their own prompts, see two anonymized answers, and pick one, and those votes become ELO ratings. That's its strength — the prompt distribution comes from people rather than from benchmark authors. Its weaknesses are worth stating because interviewers want to hear them: the prompts skew casual, the voters are self-selected and are not your paying customers, and there's no per-task breakdown by default. So I treat Arena as evidence about general chat appeal and not as evidence about anything domain-specific.

### Arena-Hard-Auto

500 prompts curated from Arena to be hard, judged by GPT-4 with position-bias correction. Scores correlate with Arena ELO at r ≈ 0.95 but cost ~\$25/run instead of months of voter time.

> **Saying it out loud.** Arena-Hard-Auto is the cheap approximation of Arena: five hundred hard prompts pulled from real Arena traffic, judged by a strong model with the position-bias swap built in. It correlates with Arena ELO at about 0.95 and costs roughly \$25 a run rather than months of accumulating human votes. That tradeoff is the whole pitch — you can run it on every candidate checkpoint instead of once a quarter. The caveat is that it inherits judge bias, so it's a fast proxy for a human signal, not a replacement for one.

### Bradley-Terry MLE

**In plain language.** Bradley-Terry is the statistical model underneath ELO. Each model gets a hidden strength score, and the chance that one beats another is the logistic function of the difference between their scores. Fitting it means finding the scores that make the observed win-loss record most likely.

```python
def fit_bt(comparisons, n_models, lr=0.01, steps=2000):
    # comparisons: list of (winner_idx, loser_idx)
    import numpy as np
    s = np.zeros(n_models)
    for _ in range(steps):
        grad = np.zeros(n_models)
        for w, l in comparisons:
            p = 1 / (1 + np.exp(s[l] - s[w]))   # P(w beats l)
            grad[w] += (1 - p)
            grad[l] -= (1 - p)
        s += lr * grad
        s -= s.mean()      # identifiability
    return s
```

> **Saying it out loud.** Bradley-Terry says each model has a latent strength and the probability one beats another is a logistic function of the difference — which is exactly ELO, just written as a likelihood you can maximize instead of an online update. Fitting it on the whole comparison set at once gets rid of the order dependence and lets you put real confidence intervals on the ratings by bootstrapping. Note the mean-centering line in the code: the scores are only identified up to a constant shift, so you have to pin the scale down somewhere. And the standard caveat — with sparse pairings, two models that never played each other have ratings connected only through third parties, so those intervals get wide fast.

---

## 7. Open-ended generation evaluation

The hardest case. The model writes an essay, an answer, code with no unit tests, a creative response. There is no gold reference.

### Reference-based metrics (the wrong default)

BLEU, ROUGE, METEOR, chrF, BERTScore. They reward n-gram or embedding overlap with a reference. Reasonable for translation; **bad for instruction-following or creative tasks.** Two correct answers with no shared vocabulary score zero. Use them only when you have a clearly canonical reference (translation, grammar correction).

> **Saying it out loud.** BLEU and ROUGE are the wrong default for anything open-ended, and I'd say that plainly. They score n-gram overlap with a reference, so two perfectly good answers that share no vocabulary both score near zero. That's fine for translation or grammar correction where there genuinely is a canonical target, and it's actively misleading for instruction following or creative writing. The one thing they're still good for is regression detection — if ROUGE against a fixed reference falls off a cliff, something broke, even if the absolute number means nothing.

### Embedding-based metrics

BERTScore, BLEURT, COMET. Slightly better than BLEU but inherit similar issues. COMET is best for translation; for everything else, prefer LLM-judge or human eval.

> **Saying it out loud.** BERTScore and friends fix the vocabulary problem by comparing embeddings instead of exact tokens, so a paraphrase no longer scores zero. But they inherit the deeper issue: semantic similarity to a reference is not the same as being correct, and they're notoriously insensitive to negation — flip "is" to "isn't" and the embedding barely moves while the meaning inverts. COMET is genuinely good for translation because it's trained on human quality judgments in that domain. For everything else I'd skip straight to an LLM judge or a programmatic check.

### LLM-judge with rubric

The default approach. Define 3–6 criteria (relevance, accuracy, clarity, completeness, harmlessness), score each 1–5, weight, sum. The rubric makes the judge auditable.

> **Saying it out loud.** For open-ended quality the default is a rubric-based judge: pick three to six criteria — relevance, accuracy, clarity, completeness — score each one to five, and weight them. The rubric is doing the real work, because it makes the score auditable: when quality drops you can see it was accuracy and not style. Without a rubric the judge collapses everything into a vibe and you can't act on the number. The cost is that you now have a judge with all the biases we listed, so this only counts if it's been calibrated against human labels.

### Programmatic checks

For every open-ended task, ask: **what's the smallest piece of structure I can verify automatically?**
- "Output JSON" → parse it
- "Cite a source" → does the URL exist
- "Be ≤ 100 words" → count
- "Mention X" → string match

These cheap checks catch a lot of regressions and have zero judge variance.

> **Saying it out loud.** For any open-ended task, the question I'd ask first is: what's the smallest piece of this I can check with code? Does the JSON parse, does the URL resolve, is it under a hundred words, does it mention the required disclaimer? None of that captures quality, and it catches a surprising share of real regressions — with zero cost and zero variance. The line I'd use: judge scores tell you if it got worse, programmatic checks tell you if it got broken, and broken is the failure that actually pages someone.

### Pairwise instead of absolute

For open-ended quality, pairwise A/B against a reference model is more reliable than 1–10 scoring. Report length-controlled win rate.

> **Saying it out loud.** For open-ended quality I'd run A-versus-B against a fixed reference model rather than asking anyone to score one to ten, because comparison is more reliable than absolute rating for humans and judges alike. Then report the win rate length-controlled, or you're partly measuring verbosity. The tradeoff worth naming: pairwise gives you relative movement but no absolute bar, so you need to keep the reference model frozen across releases — the moment you upgrade the reference, your historical win rates stop being comparable.

### Diversity evaluation

For creative tasks, you also want diversity, not just quality. Self-BLEU within K samples (low = diverse), distinct-n, embedding spread.

> **Saying it out loud.** For creative work, quality alone will mislead you, because the easiest way to make a judge happy is to write the same safe answer every time. So you also measure spread: self-BLEU across samples, where low means diverse, plus distinct-n and embedding spread. This is the classic quality-diversity tradeoff and it's a real one — heavy RLHF reliably collapses output variety, which is why aligned models feel same-y even as their per-answer scores go up. If you're evaluating a creative product and only reporting quality, you're measuring half of it.

---

## 8. Factuality and faithfulness measurement

(See [`HALLUCINATION_DETECTION_DEEP_DIVE.md`](HALLUCINATION_DETECTION_DEEP_DIVE.md) for the full chapter; this is the eval-specific summary.)

### Factuality benchmarks

**TruthfulQA** — does the model parrot human misconceptions? Adversarial: questions are written to elicit them. Often used as a "rejection of misinformation" eval rather than pure factuality.

**SimpleQA (OpenAI 2024)** — short factoid questions. Score = correct / (correct + incorrect + abstained). Models are explicitly allowed to abstain. Frontier ~50%.

**LongFact (DeepMind 2024)** — long-form factuality. Each response is decomposed into atomic facts via an LLM, each fact graded by web search via SAFE.

**FactScore (Min et al. 2023)** — atomic-fact precision against Wikipedia. Decompose response → check each fact → compute fraction supported.

**FACTS Grounding (Google 2024)** — RAG faithfulness benchmark. Given context, does the response stay grounded?

**HaluEval, RAGTruth** — hallucination detection benchmarks with annotated outputs.

> **Saying it out loud.** I'd pick the factuality benchmark by output length, because short-form and long-form need different machinery. Short-form is SimpleQA, and notice its scoring lets the model abstain — correct over correct-plus-incorrect-plus-abstained — which is the right design because refusing to answer should not be scored the same as guessing wrong. Long-form is LongFact and FactScore, which decompose a response into atomic facts and grade each one, giving you a precision-style number instead of a single verdict. TruthfulQA is a slightly different animal — it's really about whether the model repeats popular misconceptions, not whether it knows things.

### SAFE (Search-Augmented Factuality Evaluator)

Pipeline: extract atomic claims → search Google → judge each claim as supported / not-supported / irrelevant. Released by DeepMind alongside LongFact. Approximates expensive human annotation at ~1% of the cost.

> **Saying it out loud.** SAFE automates what used to be human fact-checking: break the answer into atomic claims, run a search for each, and grade it supported, unsupported, or irrelevant. DeepMind released it with LongFact and reported roughly a hundredfold cost reduction over human annotation while tracking human judgments closely. Where it breaks is where search breaks — claims that need a paywalled source, very recent events, or anything where the top results are themselves wrong. So it inherits the failure modes of the retrieval layer, which is worth saying before the interviewer says it.

### Citation-grounded factuality

For RAG / citation systems, evaluate two separate things:
1. **Citation existence** — does the cited source exist and is it accessible?
2. **Citation faithfulness** — does the cited source actually support the claim?

These can be tested with NLI: claim entailed by source span?

> **Saying it out loud.** Two separate checks and people conflate them. Does the cited source exist and resolve — that's a HTTP request. Does it actually support the specific sentence it's attached to — that's an entailment problem, and that's the one that fails. Both are cheap to automate once you separate them, and reporting them separately is what lets you tell "the model invented a URL" apart from "the model cited a real page that says something else." The second failure is far more common and far more damaging, because a real link looks trustworthy.

### RAGAS (RAG eval framework)

- **Faithfulness** = fraction of generated claims supported by retrieved context.
- **Answer relevancy** = how well the answer addresses the question.
- **Context precision** = fraction of retrieved chunks that were actually relevant.
- **Context recall** = fraction of needed information that was retrieved.

Now standard for RAG eval.

> **Saying it out loud.** RAGAS gives four numbers and the way to hold them is two-and-two: faithfulness and answer relevancy grade the generator, context precision and recall grade the retriever. Read together they localize the bug — poor faithfulness with good context means the model is confabulating, poor faithfulness with bad context means fix retrieval first. Faithfulness is the one you can compute continuously in production from your own logs. Context recall is the awkward one, because it needs a gold answer to know what "all the needed information" was, so it stays an offline metric.

### Calibration as factuality proxy

A well-calibrated model should be unsure when it is wrong. Measure ECE on a multi-choice eval. Models trained with RLHF are typically *over-confident* — calibration regresses post-RLHF (Tian et al. 2023, Kadavath et al. 2022).

> **Saying it out loud.** A useful shortcut: instead of checking whether the model is right, check whether it *knows* when it's right. If confidence and accuracy line up, you can build a refusal policy on top and contain the damage even at fixed accuracy. Measure it with ECE on a multiple-choice set and draw a reliability diagram. The finding to cite is that RLHF makes calibration worse — Kadavath showed base models are surprisingly well calibrated, and Tian and others showed the alignment step pushes everything toward overconfidence. That's why post-RLHF log-probs are a weaker uncertainty signal than people expect.

---

## 9. Contamination

The headline hazard. If a benchmark is in pretraining data, scores are meaningless.

### How it happens

- The benchmark was published on the web; the crawler picked it up.
- A user pasted it into a public forum.
- It leaked into a training corpus through a derivative dataset (instruction-tuning data, synthetic data based on benchmark questions).
- The benchmark was deliberately included to optimize for it (rare at honest labs, common in published-but-suspect models).

> **Saying it out loud.** Contamination is usually accidental and it has four routes. The benchmark was published to the web and the crawler ate it. Someone pasted questions into a forum or a GitHub issue. It came in secondhand through an instruction-tuning set or synthetic data generated from the benchmark — that's the sneaky one, because you can decontaminate against the original strings and still have paraphrases in there. And occasionally it's deliberate. The reason to enumerate the routes is that exact-string decontamination only defends against the first two, which is why teams keep finding contamination after they "cleaned" it.

### Detection methods

**In plain language.** These are ways to ask "did this model already see this test question during training?" without having access to the training data. They work by looking for signs of memorization: does the model complete the item verbatim, does it assign suspiciously high probability to it, does its score collapse when you reword it.

**Memorization probes.** Show the model the first half of a test example, ask it to complete. If it reproduces verbatim, contamination is likely.

**Membership inference.** Min-K%-prob (Shi et al. 2024): for a candidate test example, compute the average log-prob of the K% lowest-probability tokens. Members of training data have higher (less negative) min-K% prob than non-members.

**Time-shifted benchmarks.** Use benchmarks that were created **after** the model's training cutoff. LiveCodeBench, LiveBench, GAIA partially do this.

**Perturbation tests.** Slightly rephrase benchmark questions. If accuracy drops sharply, the model was relying on memorized text rather than understanding.

**Canary strings.** Insert known canary strings into your held-out test data; later inspect public model outputs for them.

> **Saying it out loud.** Four practical checks. Feed the first half of a test item and see if the model completes it verbatim — that's the strongest evidence and it needs no infrastructure. Membership inference like Min-K%-prob looks at the least-likely tokens in an example, since a memorized example has no genuinely surprising tokens. Time-shifted benchmarks like LiveCodeBench sidestep the problem entirely by only using items released after the training cutoff. And the one I'd run first because it's the cheapest: perturbation — rephrase the questions and see if accuracy falls off a cliff. A model that drops fifteen points on a paraphrase was pattern-matching text, not reasoning.

### Prevention

- Hold out at least one private test set per benchmark. Never publish it.
- For competition models, use closed/private holdouts (e.g., the SWE-Bench-Verified hidden tests).
- Use continuously-updating benchmarks for ongoing evaluation.

> **Saying it out loud.** Prevention is mostly discipline: keep a private held-out split you never publish, use hidden tests for anything competitive, and rotate in continuously-updating benchmarks so there's always a slice the model provably couldn't have seen. Canary strings help too — plant a unique token in your held-out data and later grep model outputs for it. The tradeoff nobody likes: the moment you publish a benchmark it starts dying, so the useful ones are either private or perpetually refreshed, and neither is free to maintain.

### Reporting

If you publish results, be explicit about which benchmarks you have decontaminated against, and how (n-gram overlap removal, exact-match removal, fuzzy-match removal).

> **Saying it out loud.** If you're publishing numbers, say what you decontaminated against and how — n-gram overlap, exact match, fuzzy match — because "we decontaminated" without a method is not a claim anyone can check. This sounds like paperwork and it's actually the credibility signal: a results table with a decontamination note reads completely differently from one without. And be explicit about which benchmarks you did *not* check, since silence gets read as either sloppiness or something worse.

---

## 10. Robustness, prompt sensitivity, fairness

A model that scores 90% on MMLU with one prompt template and 75% with another is fragile. Robustness eval measures this directly.

### Prompt sensitivity

- **Template variations.** Rewrite the same question 5 ways; compute std of accuracy.
- **Few-shot ordering.** Permute examples; measure variance.
- **System prompt variations.** "You are a helpful assistant" vs "Answer concisely" vs nothing.
- **Whitespace, punctuation, capitalization.** Yes — these matter, especially for smaller models.

A robustness number is std-of-accuracy across template variants. Report it alongside the headline.

> **Saying it out loud.** A model that scores 90 with one prompt template and 75 with another doesn't have a score, it has a range, and reporting only the high end is how leaderboards get gamed. So I'd write the same question five ways, permute the few-shot ordering, vary the system prompt, and report the standard deviation next to the headline number. Whitespace and capitalization matter too, especially on smaller models — which sounds absurd until you've watched it happen. The practical rule: if the gap between two models is smaller than the spread across your own templates, you haven't measured a difference.

### Adversarial robustness

- **PromptBench** — adversarial perturbations to prompts.
- **CheckList** — behavioral testing for NLP (e.g., negation handling).
- **AdvGLUE** — adversarial GLUE.

> **Saying it out loud.** Beyond benign rewording, you want deliberate perturbation — PromptBench for adversarial prompt edits, CheckList for behavioral probes like negation handling, AdvGLUE for adversarial classification. The point isn't the specific suites, it's that average-case accuracy hides worst-case behavior, and users who are trying to break your product are the worst case by definition. The number I'd want is the gap between clean and adversarial accuracy, because that gap is the size of your exposure.

### Fairness and bias

- **BBQ** — bias benchmark for QA (gender, race, religion, etc.).
- **CrowS-Pairs, StereoSet** — stereotype probes.
- **Winogender, Winobias** — coreference + gender.

For products serving diverse populations, slice your eval by demographic where possible. Aggregate metrics hide subgroup regressions.

> **Saying it out loud.** BBQ, CrowS-Pairs, StereoSet, Winogender — those are the standard probes for stereotype and coreference bias. But the point I'd actually make for a product is simpler: slice your existing eval by demographic wherever the data allows, because an aggregate metric can improve while a subgroup regresses, and the aggregate is what you'll be looking at. That's the failure mode — you ship an average improvement that made things worse for the group least represented in your eval set, and telemetry surfaces it as a complaint rather than a number.

### OOD robustness

Hold out an explicitly out-of-distribution slice. For a customer-support model, that might be a topic the training data did not cover. Measure performance and calibration drop.

> **Saying it out loud.** Deliberately carve out a slice the model hasn't seen the shape of — for a support bot, that's a product area your training data didn't cover — and measure both accuracy and calibration on it. Calibration is the part people leave out and it's the more important half: an out-of-distribution model that knows it's out of its depth can escalate, and one that's confidently wrong cannot. So the metric I care about isn't just how much accuracy drops, it's whether confidence drops with it.

---

## 11. Statistical methodology

Most LLM papers under-report uncertainty and over-claim. Senior interview signal: ask back about confidence intervals.

### Sample size

**In plain language.** This is the standard binomial error bar. If your benchmark is "each question is right or wrong," the uncertainty in your score depends only on the score and the number of questions — and for the sizes benchmarks actually use, that uncertainty is bigger than most reported differences.

For a benchmark with binary correctness:
- 95% CI half-width ≈ 1.96 × √(p(1-p)/n).
- For p ≈ 0.5, n = 100 gives ±10pp; n = 400 gives ±5pp; n = 1000 gives ±3pp.
- Most benchmarks (HumanEval = 164, GPQA-Diamond = 198) are too small to distinguish 1–2 point differences.

> **Saying it out loud.** The arithmetic here should be reflexive. For a pass/fail benchmark near 50%, a hundred items gives you roughly plus or minus ten points, four hundred gives you five, and a thousand gives you three. Now look at the benchmarks people quote: HumanEval is 164 items, GPQA-Diamond is 198. That means a one- or two-point gap on either of those is statistically nothing, and half the leaderboard movement you read about is noise. Being able to say that number out loud is one of the fastest credibility wins in an eval interview.

### Confidence intervals

Report 95% CIs (Wilson or bootstrap). A 2-point gap with overlapping CIs is not a real difference.

> **Saying it out loud.** Report a 95% interval with every headline number — Wilson for a proportion, bootstrap for anything more complicated. The reason isn't statistical piety: if two models' intervals overlap, you have not shown a difference, and shipping on that basis means you'll sometimes ship a regression. The habit to demonstrate in an interview is asking for n whenever someone quotes you a benchmark delta. A two-point gap on a two-hundred-item set is indistinguishable from a coin flip.

### Multiple comparisons

When you eval on 20 benchmarks, with α=0.05 you expect ≈1 false positive by chance. Apply Bonferroni or Benjamini-Hochberg if you are hypothesis-testing.

> **Saying it out loud.** Run twenty benchmarks at the usual 5% threshold and you should expect about one significant-looking result from pure chance. So if you're evaluating on a big suite and reporting the wins, you're partly reporting luck. Bonferroni or Benjamini-Hochberg fixes it when you're genuinely hypothesis-testing. The version that bites in practice is slice analysis — you cut the data fifty ways looking for where the new model helps, you find a slice, and it doesn't replicate next month.

### Pass@k for code

**In plain language.** Pass@k asks: if the model gets k attempts, how often does at least one of them work? The combinatorial formula is just the unbiased way to estimate that from a larger pool of n samples, instead of literally running k attempts over and over.

Sample n responses, compute fraction that pass tests. Pass@1 with k=1 is the standard metric. Pass@k (Chen et al. 2021) reports the probability that at least one of k samples passes:

```
pass@k = 1 - C(n - c, k) / C(n, k)
```

where n is the number of samples and c is the number that pass. Use n ≥ 20 for stable estimates.

> **Saying it out loud.** Pass@k is the probability that at least one of k samples passes the tests. The formula looks fussy but it's just an unbiased estimator — you generate a bigger pool of n samples, count how many pass, and compute the chance a random draw of k contains at least one, which is far cheaper than repeatedly running k-sample trials. You want n of twenty or more for it to be stable. The framing point: pass@1 is the number that matters for a product, because the user gets one answer; high pass@10 with low pass@1 means the model *can* solve it and can't reliably pick the solution, which is a reranking problem, not a capability problem.

### LLM-judge variance

Two runs of the same judge on the same outputs do not agree perfectly. Report inter-run agreement; with greedy / temperature=0 judging, agreement should be ~95%; if it's lower, investigate.

> **Saying it out loud.** Run the same judge twice over the same outputs and it won't fully agree with itself, which surprises people who assume the judge is a fixed function. At temperature zero you should see about 95% self-agreement, and if it's meaningfully lower something is wrong — usually an ambiguous rubric or a prompt that lets the judge waffle. Report that inter-run agreement alongside the score. It's the cheapest diagnostic you have, it costs one extra run, and it tells you the noise floor beneath every judged comparison you make.

### Sample size for LLM-judge

Pairwise win rate p with n comparisons has SE = √(p(1-p)/n). To distinguish 50% from 55% at 95% confidence, n ≈ 1500.

> **Saying it out loud.** Win rates have error bars too, and they're wider than people expect. To distinguish 50% from 55% at 95% confidence you need on the order of 1,500 comparisons — not the 100 that a quick eval run gives you. So when someone reports beating a baseline 54 to 46 on 200 examples, that's inside the noise. Carrying that number lets you push back in the room, and it's also a budgeting fact: 1,500 judged comparisons is a real API bill, which is why teams reach for cheaper trained judges on the high-volume path.

### Correlated samples

If you sample 5 responses per prompt and aggregate, treat the prompt as the unit, not the sample. Otherwise CIs are anti-conservatively narrow.

> **Saying it out loud.** If you generate five responses per prompt, you do not have five independent data points — you have one prompt sampled five times, and treating them as independent makes your confidence intervals too narrow. So cluster at the prompt level: average within prompt first, then compute the interval across prompts. It's the same clustered-standard-error issue as any repeated-measures experiment. The consequence of getting it wrong is the worst kind: you'll confidently declare significance that isn't there, and it'll look rigorous because you did report an interval.

---

## 12. Evaluation harnesses

### lm-eval-harness (EleutherAI)

Open-source, supports hundreds of tasks. Handles multiple-choice (logprob comparison) and generation. The reference implementation; HuggingFace Open LLM Leaderboard runs on it. **Default for academic eval and reproducibility.**

### OpenCompass (Shanghai AI Lab)

Larger task catalog including Chinese benchmarks. Strong support for distributed eval. Becoming standard alongside lm-eval-harness.

### HELM (Stanford)

Holistic eval framework — runs many benchmarks and reports across multiple dimensions (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency). Slower but the most thorough.

### Inspect (UK AISI)

Eval framework focused on safety / dangerous-capability evaluations. Used by AISI, Anthropic, DeepMind alignment teams.

### EleutherAI BIG-Bench, OpenAI Evals

Older but still used in some labs.

### What to use

- **Academic / reproducibility:** lm-eval-harness.
- **Frontier-model eval suite:** combination of lm-eval-harness + custom internal infra; most labs have rolled their own.
- **Safety eval:** Inspect.
- **RAG eval:** RAGAS, TruLens.
- **Open-ended chat eval:** Arena-Hard-Auto, MT-Bench infra, plus internal LLM-judge.

> **Saying it out loud.** My default stack: lm-eval-harness for anything that needs to be reproducible or comparable with published numbers, Inspect for safety and dangerous-capability work, RAGAS or TruLens for RAG, and Arena-Hard plus an internal judge for open-ended chat. HELM if you want the thorough multi-dimensional picture and can afford the runtime. The honest observation is that every frontier lab ends up writing their own harness anyway, because the public ones are built for comparability and internal eval is built for iteration speed — and those two goals pull in opposite directions.

---

## 13. Online evaluation and production telemetry

Offline eval gets you to launch. Online eval keeps you honest after launch.

### Telemetry every LLM product needs

- Per-request latency (p50/p95/p99) at each stage (retrieval, prefill, decode, post-processing).
- Tokens in, tokens out (cost).
- Model used, prompt version, retrieval pipeline version.
- User-visible signals: thumbs up/down, copied response, conversation regenerated, conversation abandoned.
- Tool-call success/failure (for agents).
- Refusal rate.
- Empty-response rate.
- Length distribution of outputs.

> **Saying it out loud.** The instrumentation list is short and nobody has all of it: latency broken out per stage so you can tell retrieval from decode, tokens in and out for cost, the exact model and prompt version on every request, and the user-behavior signals — thumbs, copies, regenerates, abandons. Then refusal rate, empty-response rate, output length distribution, and tool-call success if you're agentic. The one I'd insist on is prompt version, because without it you can't attribute a quality change to anything, and prompts change more often than models do.

### Surrogate quality metrics

- **Regenerate rate.** Proxy for dissatisfaction.
- **Edit rate** (for code/copilot products).
- **Conversation length / depth.** Longer = engagement, but also = friction. Monitor distribution.
- **Time-to-first-token.** Latency proxy.
- **Tool-call success rate.** Agent reliability proxy.
- **Citation click rate** (for RAG/search products).

> **Saying it out loud.** You almost never get a direct quality label in production, so you lean on proxies. Regenerate rate is the best one — a user hitting regenerate is telling you the answer was bad without filling in a survey. Edit rate is the equivalent for code products, and citation click-through for search. The trap is that these proxies are ambiguous: longer conversations can mean engagement or can mean the user is stuck going in circles, and you cannot tell which from the metric alone. So you pair every surrogate with a sampled audit that tells you which direction it's pointing.

### Sampling for online eval

You cannot run an LLM judge on every production request. Sample 1–5% and run the full eval pipeline (judge + factuality checks + safety filters) on the sample. Stratify by route or persona to detect subgroup regressions.

> **Saying it out loud.** Running a judge and a factuality pipeline on 100% of traffic is unaffordable, so you sample one to five percent and run the full stack on that. The part to get right is stratification — sample by route, by persona, by customer tier — because uniform sampling gives you a great estimate of your average and near-zero coverage of the rare high-stakes path. A 1% uniform sample of a low-volume enterprise route is a handful of requests a week, which detects nothing. Stratify, then reweight.

### Drift detection

Track per-week aggregates of: refusal rate, response length, latency, hallucination flags from the detector. Alert when any moves >2σ. Distribution shift in user prompts is the typical cause; model drift is rare but possible (model hot-swap, RAG corpus update).

> **Saying it out loud.** Watch weekly aggregates of refusal rate, response length, latency, and hallucination flags, and alert when anything moves more than two sigma. What you're usually catching isn't the model changing — it's the *input* changing, because user prompts drift with seasons, marketing pushes, and new features. The other common culprit is a RAG corpus update quietly changing what gets retrieved. Naming that ordering matters: nine times out of ten it's data drift, not model drift, and teams waste days looking at the model first.

### Logging and replay

Store inputs + outputs (with retention/PII policy). Build a replay harness so you can re-run last week's traffic against a new model. Replay is the cheapest way to catch regressions before A/B.

> **Saying it out loud.** Log inputs and outputs under a real retention and PII policy, then build a replay harness so you can run last week's actual traffic through a candidate model before anyone sees it. Replay is the highest-value thing on this whole page, because it catches regressions on your real distribution rather than on a benchmark, and it costs one batch inference run instead of an A/B test. The limit worth stating: replay tells you the output changed, not that the user's outcome changed — for anything interactive or agentic, the conversation would have branched, so replay is a screen, not a verdict.

---

## 14. A/B testing for LLM products

The mechanics resemble standard A/B but with LLM-specific gotchas.

### Setup

- Bucket users into Control (A) and Treatment (B). Assignment must be sticky per user / session.
- Define primary success metric in advance (one). Examples: task completion rate, retention, conversion, regenerate rate (lower is better).
- Define guardrail metrics: latency p95, cost per query, safety flag rate.
- Pre-register sample size and duration. Stop only at the planned time, not on early peeks.

> **Saying it out loud.** Standard A/B discipline applies and it's where most LLM teams get sloppy. Assignment sticky per user, one pre-registered primary metric — not five — and guardrails on latency p95, cost per query, and safety flags. Then the hard part: stop at the planned time rather than the moment the numbers look good, because peeking inflates your false-positive rate enormously. Picking the primary metric in advance is the discipline that matters most, since with enough LLM metrics on the dashboard something is always up.

### Sample size

**In plain language.** This formula answers "how many users do I need per arm to detect a lift this small?" The key thing to notice is the square in the denominator: halving the effect you want to detect quadruples the users you need.

For binary metric p with absolute lift δ:

```
n_per_arm ≈ 16 * p(1-p) / δ²  (for 80% power, α=0.05)
```

For p=0.5, δ=0.02: n ≈ 20K per arm.

> **Saying it out loud.** The rule of thumb is sixteen times p times one-minus-p over delta squared per arm, and the important bit is that delta is squared — halving the effect you want to detect costs you four times the traffic. Concretely: a base rate of 50% and a two-point lift needs about twenty thousand users per arm. That number is what kills most LLM A/B tests before they start, because small products simply don't have the traffic to detect the size of improvement they're actually shipping. Saying that up front reframes the conversation toward offline eval and replay rather than a test that could never have concluded.

### Variance reduction

CUPED (Microsoft / Netflix) — regress out pre-period covariates from the outcome. Routinely reduces required sample size by 30–50%.

> **Saying it out loud.** If you can't get more traffic, get more precision out of the traffic you have. CUPED regresses out each user's pre-experiment behavior — a heavy user is a heavy user in both arms, so subtracting their baseline removes variance that has nothing to do with your treatment. In practice it cuts the required sample size by 30 to 50%, which can be the difference between a two-week test and a two-month one. It only works when you have pre-period data on the same users, so it's free for logged-in products and unavailable for anonymous traffic.

### LLM-specific gotchas

- **Latency leak.** A model that's even 100ms slower can suppress engagement enough to look "worse" on quality metrics. Always control for latency.
- **Prompt/version coupling.** Treatment may have a new prompt **and** a new model. Disentangle with multi-cell experiments.
- **Output length.** Longer responses can drive engagement up while user satisfaction goes down. Monitor both.
- **Memorization of system prompt.** Users adapt to the assistant; switching personas mid-experiment can hurt.
- **Regenerate cascade.** Bad outputs cause regenerates which add cost and noise. Track per-conversation cost, not just per-request.
- **Selection bias from refusals.** If treatment refuses more, the conversations that "complete" are a biased sample. Always include refused requests in the denominator.

> **Saying it out loud.** The LLM-specific traps are mostly confounds. Latency leak is the classic — a model that's a hundred milliseconds slower suppresses engagement, so it looks worse on quality metrics when it's actually just slower, and you have to control for it. Prompt and model changing together in one cell means you learn nothing about which caused what. Longer outputs can lift engagement while satisfaction falls. And the subtle one: if treatment refuses more, then only the easier conversations complete, so your completion-rate comparison is measuring a biased sample — always keep refusals in the denominator.

### Shadow / canary deploy first

Before A/B with real users on a large fraction:
1. **Shadow.** Mirror traffic to the new model, compare offline.
2. **Canary.** 1–5% of traffic, monitor for hours/days, no business decision yet.
3. **A/B.** 50/50 with proper sample size.
4. **Rollout.** Gradual ramp with rollback gates.

> **Saying it out loud.** Never go from offline straight to a 50/50 split. Shadow first — mirror the traffic to the new model, serve nothing from it, compare offline. Then canary to one to five percent and watch for hours or days without making a business decision. Then the real A/B at a proper sample size, then a gradual ramp with automatic rollback gates. Each stage catches a different class of failure at a different cost: shadow catches broken output, canary catches operational problems like latency and rate limits, and only the A/B can tell you about user behavior.

### Sequential testing

For online eval, mSPRT or always-valid CIs let you stop earlier without inflated false positives. Used at Microsoft, Netflix, Linkedin. Less needed than for traditional A/B because LLM cost per query is higher and sample sizes are lower; still good practice.

> **Saying it out loud.** If you want to peek at your experiment as it runs — and everyone does — use a method that permits it: mSPRT or always-valid confidence intervals let you stop early without inflating false positives, which is what naive peeking does. It's standard at Microsoft, Netflix, and LinkedIn. For LLM products the payoff is smaller than usual, because your bottleneck is typically total traffic and cost per query rather than test duration, so you rarely have enough data to stop early anyway. Still worth doing — it makes the daily dashboard-checking legitimate instead of self-deception.

---

## 15. Designing an eval suite for a product (case study)

**Product: customer-support agent for a SaaS company.** It answers user questions, can call tools (lookup user account, file ticket, escalate), and operates from a knowledge base.

A good eval suite has four layers.

### Layer 1: Capability sanity (does the model work at all)

- MMLU-Pro or domain-equivalent (~5K questions, run quarterly).
- IFEval to catch instruction-following regressions.
- HumanEval+ if the agent generates any code.
- Safety: XSTest, WildGuard for refusal calibration.

These are run on every model swap.

> **Saying it out loud.** Layer one answers exactly one question: is this model fundamentally broken or fundamentally different? It's a few thousand items of general capability plus IFEval for instruction following and XSTest for refusal calibration, and you run it on every model swap. It won't tell you anything about your product, and it's not supposed to. Its job is to catch the case where a vendor's minor version update quietly changed refusal behavior or reasoning quality before you find out from your task-specific eval.

### Layer 2: Task-specific offline eval

Hand-build a 500-prompt **golden set** that covers:
- Common questions (top 50% of intent volume).
- Hard but answerable questions (questions where the KB has the answer but it's hidden or paraphrased).
- Out-of-scope (should escalate).
- Adversarial (jailbreak attempts, social engineering).
- Multi-turn (clarification, follow-up).
- Multi-language if relevant.

For each, define:
- Reference answer or rubric.
- Required tool calls.
- Required citations.
- Acceptable refusal.

Run the agent end-to-end. Score with:
- Exact match for tool calls.
- LLM-judge with rubric for free-form text, calibrated to 200 human-labeled examples.
- RAGAS faithfulness (no claim outside retrieved context).
- Citation existence + faithfulness (NLI against source).
- Hallucination flag from the detector cascade (see Hallucination Detection Deep Dive).
- Refusal calibration (refused when should have, answered when should have).

> **Saying it out loud.** This is the layer that actually decides anything: five hundred hand-built prompts covering your real intent distribution, plus hard-but-answerable cases, out-of-scope cases that should escalate, adversarial attempts, and multi-turn flows. For each one you write down what a good answer looks like, which tools should get called, and whether refusing is correct. Then you score it with a mix — exact match on tool calls, a calibrated judge on the prose, RAGAS faithfulness, citation checks. The single most important design choice is including out-of-scope prompts, because a golden set made only of answerable questions will bless a model that never knows when to escalate.

### Layer 3: Online telemetry

- Per-conversation: completed (no escalation, no regenerate), partial, escalated, abandoned.
- Per-turn: thumbs, regenerate, time-to-response.
- Sample 2% to run hallucination detector and judge offline.
- Track: refusal rate, hallucination rate, citation faithfulness, latency p95, cost.

> **Saying it out loud.** Layer three measures the thing the business cares about: did the conversation end resolved, escalated, or abandoned. Per turn you have thumbs, regenerates, and response time; on a two percent sample you run the full judge and hallucination pipeline offline. The reason you need this layer even with a great golden set is that offline eval measures the distribution you thought you had, and telemetry measures the one you actually have — and the gap between them is where products fail. The headline pair to watch is hallucination rate against refusal rate, because they move in opposite directions and either one alone can be gamed.

### Layer 4: Continuous improvement

- Sample 100 escalations and 100 thumbs-down per week. Categorize failures. Add tough examples to golden set.
- Periodically refresh the golden set from real traffic (with PII redaction).
- Re-calibrate the LLM-judge every quarter with fresh human labels.

> **Saying it out loud.** The flywheel: every week pull a hundred escalations and a hundred thumbs-down, categorize the failures, and promote the interesting ones into the golden set. Refresh the golden set from real traffic periodically with PII stripped, and re-calibrate the judge quarterly against fresh human labels. What makes this work is that your eval set drifts toward the real distribution instead of away from it. The tension to name is overfitting your own eval — if you only ever add cases you failed, the set becomes an adversarial collection that no longer represents typical traffic, so you mix in random samples too.

### Failure modes this eval catches

- New model regresses on a specific intent (Layer 2 slice).
- Retrieval pipeline change drops faithfulness (Layer 2 RAGAS).
- Refusal rate creeps up after an alignment update (Layer 1 XSTest + Layer 3 telemetry).
- Latency degrades after deployment (Layer 3 p95).
- Hallucination detector's false-positive rate inflates after RAG corpus update (Layer 3 sampled detector).

> **Saying it out loud.** The reason to build all four layers is that each catches something the others can't see. A model that regresses on one intent shows up only in a Layer 2 slice — the aggregate barely moves. A retrieval change shows up as a faithfulness drop with everything else flat. Refusal creep after a vendor alignment update shows up in XSTest and in telemetry, never in accuracy. And latency regressions only appear in production. That mapping — failure to layer — is the strongest thing you can say in a design interview, because it shows the architecture is derived from failure modes rather than copied from a blog post.

---

## 16. Failure modes and senior-level signals

### Common eval failure modes

- **Single-number reporting.** Reporting MMLU 87% with no CI, no slice, no robustness. A senior eng asks: "what's the std across prompt templates? what's the per-subject variance? what's your contamination check?"
- **LLM-judge with no calibration.** Reporting "GPT-4 win rate 72%" with no human-validation set. The judge could be ranking by length.
- **No contamination check.** A new model "beats" SOTA on a benchmark released years ago. Probably contamination.
- **Optimizing the eval, not the capability.** Once a benchmark is part of training (deliberately or by leak), it stops measuring what you wanted.
- **Ignoring latency in quality eval.** "Better" model is 3x slower, makes the product worse.
- **No safety eval.** Helpfulness improves, refusals collapse, jailbreaks succeed; ship anyway.
- **Single judge, single template.** No pairwise, no ensemble, no template variants.
- **No production telemetry.** Offline numbers great, users hate it.

> **Saying it out loud.** The recurring sins are worth naming crisply. Single-number reporting with no interval, no slices, no contamination check. An LLM judge quoted with no human-validated calibration set, so the win rate might just be measuring length. Beating an old benchmark and not asking whether it leaked. Optimizing the eval instead of the capability. And ignoring latency, so the "better" model is three times slower and the product is worse. If I had to pick the one that does the most damage in practice, it's the uncalibrated judge, because it produces a confident, quotable number that nobody can check.

### What senior interviewers want to hear

- You separate capability eval from product eval.
- You know that LLM-judge has biases and you list them precisely (length, position, family, format).
- You always design a calibration set and report agreement with humans.
- You report uncertainty (CIs, n).
- You actively look for contamination.
- You think of eval as a **measurement instrument** with bias-variance tradeoffs.
- You design the offline → shadow → canary → A/B pipeline.
- You ground production decisions in primary + guardrail metrics, pre-registered.
- You appreciate the offline/online gap and have telemetry to bridge it.
- You can sketch an eval suite for any product in <5 minutes using the four-layer pattern.

> **Saying it out loud.** What they're listening for is whether you treat evaluation as a measurement instrument with its own bias and variance, rather than a scoreboard. Concretely: you separate capability eval from product eval, you can list judge biases with their mitigations, you always build a human-labeled calibration set, you report n and confidence intervals, you actively hunt for contamination, and you can draw the offline-to-shadow-to-canary-to-A/B pipeline. The tell that closes it is being able to sketch a four-layer eval suite for any product in under five minutes — that shows it's a pattern you've internalized rather than a case study you memorized.

---

## 17. Common interview questions

A subset; full grill in [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

1. Why is LLM evaluation harder than evaluating a classifier?
2. List 5 LLM-judge biases and how you mitigate each.
3. What's the difference between BLEU and BERTScore? When would you use either?
4. How do you set up a pairwise win-rate eval to be position-bias-free?
5. What is contamination in LLM eval, and how do you detect it?
6. Walk through how you'd design an eval suite for a customer-support chatbot.
7. What's pass@k? When does it matter?
8. Difference between reference-based and reference-free metrics? Examples of each?
9. How do you size an A/B test for a chatbot product?
10. Why is length-controlled win rate important?
11. What is RAGAS and what does it measure?
12. Why is calibration a factuality proxy? How would you measure it?
13. How would you detect that a model regressed on a niche slice without a golden labeled set?
14. What is ELO and how is it computed from pairwise comparisons?
15. Compare HELM, lm-eval-harness, OpenCompass.
16. What's lost-in-the-middle and how would you test for it?
17. Why isn't MMLU enough?
18. What's IFEval and why is it useful?
19. How does SAFE work? Where does it fail?
20. What metrics do you log for an LLM product in production?
21. What's the difference between offline eval and shadow / canary?
22. What's CUPED and why does it matter for LLM A/B tests?
23. Describe a multi-judge ensemble. Why use it?
24. How do you decontaminate a benchmark?
25. How do you evaluate an agent?

> **Saying it out loud.** Practicing these out loud is different from reading them, and the difference shows up in the first thirty seconds of an answer. My rule for each one: lead with a one-sentence punchline, then give the mechanism, then close on a tradeoff or a number. "Why is LLM eval hard" becomes "because there's no single right answer, so every metric is a proxy — and the proxies have their own biases, like judges preferring longer responses." Time yourself at sixty seconds; if you can't finish, you're front-loading detail instead of the answer.

---

## 18. References and further reading

### Benchmarks (canonical)

- **MMLU** — Hendrycks et al., 2021.
- **MMLU-Pro** — Wang et al., 2024.
- **GPQA** — Rein et al., 2023.
- **GSM8K** — Cobbe et al., 2021. **MATH** — Hendrycks et al., 2021.
- **HumanEval** — Chen et al., 2021. **MBPP** — Austin et al., 2021. **EvalPlus** — Liu et al., 2023.
- **SWE-Bench** — Jimenez et al., 2024. **SWE-Bench-Verified** — OpenAI 2024.
- **LiveCodeBench** — Jain et al., 2024.
- **TruthfulQA** — Lin et al., 2022.
- **SimpleQA** — Wei et al., OpenAI 2024.
- **GAIA** — Mialon et al., 2023 (Meta).
- **MMMU** — Yue et al., 2023.
- **TAU-bench** — Sierra 2024.
- **MLE-Bench** — OpenAI 2024.

### Methodology

- **Lost in the Middle** — Liu et al., 2023.
- **Length-controlled AlpacaEval** — Dubois et al., 2024.
- **MT-Bench** — Zheng et al., 2023.
- **Arena-Hard-Auto** — Li et al., LMSys 2024.
- **IFEval** — Zhou et al., 2023.
- **RULER** — Hsieh et al., NVIDIA 2024.
- **HELM** — Liang et al., 2023.
- **G-Eval** — Liu et al., 2023.
- **Prometheus / Prometheus 2** — Kim et al., 2024.

### Factuality

- **FactScore** — Min et al., 2023.
- **LongFact + SAFE** — Wei et al., DeepMind 2024.
- **RAGAS** — Es et al., 2024.
- **FACTS Grounding** — Google DeepMind, 2024.

### Contamination

- **Min-K%-prob** — Shi et al., 2024.
- **Carlini et al.,** Quantifying Memorization, 2022.
- **LiveBench, LiveCodeBench** — for time-shifted evaluation.

### Calibration

- **Kadavath et al.** Language Models (Mostly) Know What They Know, 2022.
- **Tian et al.** Just Ask for Calibration, 2023.

### Frameworks

- lm-eval-harness (EleutherAI) — github.com/EleutherAI/lm-evaluation-harness
- OpenCompass — github.com/open-compass/opencompass
- Inspect (UK AISI) — github.com/UKGovernmentBEIS/inspect_ai
- DeepEval, TruLens, RAGAS — RAG eval ecosystems.

### Statistics for ML

- **CUPED** — Deng et al., 2013.
- Bradley-Terry / Plackett-Luce — classical pairwise modelling.

### Surveys

- **A Survey on Evaluation of Large Language Models** — Chang et al., 2023.
- **Beyond the Imitation Game (BIG-Bench paper)** — Srivastava et al., 2022.

---

## How to use this chapter

1. Read straight through once for the lay of the land.
2. Memorize §2 (taxonomy), §5 (judge biases), §11 (statistical methodology), §15 (case study).
3. Drill yourself with §17 questions — be able to give a 60-second answer to each.
4. Pick one product you've used (or invent one) and design its eval suite using the §15 four-layer pattern. Whiteboard it.
5. Pair with [`HALLUCINATION_DETECTION_DEEP_DIVE.md`](HALLUCINATION_DETECTION_DEEP_DIVE.md) for the factuality slice.

If you can fluently distinguish capability vs product eval, identify all the LLM-judge biases, design contamination detection, size A/B tests, and articulate the offline → shadow → canary → A/B pipeline — you'll handle frontier-lab and big-tech eval interviews well.
