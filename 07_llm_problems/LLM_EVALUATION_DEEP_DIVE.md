# LLM Evaluation — Deep Dive

> Frontier-lab interview-grade reference on evaluating LLMs and LLM-powered products.
> Read with `HALLUCINATION_DETECTION_DEEP_DIVE.md` (factuality is one slice of eval) and `LLM_PROBLEMS_DEEP_DIVE.md`.

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

---

## 2. Taxonomy of evaluations

A useful first cut.

**By target.** Capability evals (math, code, reasoning) measure what the model *can* do. Quality evals measure how good the output *is* on open-ended tasks. Safety / behaviour evals measure refusals, harmful output, jailbreak susceptibility, sycophancy. Product evals measure whether real users complete their task.

**By signal type.** Reference-based (compare to gold), reference-free (intrinsic measures: log-prob, coherence, diversity), pairwise (A vs B), human-judged, LLM-judged, programmatic / unit-tested (especially code, math).

**By stage.** Offline pre-deployment, shadow production, online A/B, post-launch telemetry.

**By granularity.** Token-level (perplexity), output-level (correct/incorrect), conversation-level (multi-turn), session-level (did the user solve their problem).

**By openness.** Closed-form (multiple choice, exact match), constrained (extraction with schema), open-ended generation, agentic (multi-step tool use).

A modern frontier-lab eval stack covers all of these. A startup eval stack should cover at least: capability benchmarks, LLM-judge open-ended quality, safety, and online telemetry.

---

## 3. Capability benchmarks

### 3.1 Knowledge

**MMLU.** 57-subject multiple-choice across STEM, humanities, social sciences. The default knowledge benchmark. Saturated above ~90 for frontier models. Heavily contaminated.

**MMLU-Pro.** Hand-constructed harder version with 10 answer choices instead of 4 and explicit reasoning required. Replaces MMLU as the modern knowledge benchmark.

**AGIEval, ARC, BIG-Bench, BIG-Bench-Hard (BBH).** Mixed-task suites. BBH has 23 reasoning-heavy subtasks; "BBH chain-of-thought" was the classic CoT eval before GSM8K dominated.

**TriviaQA, NaturalQuestions, PopQA.** Open-domain QA, useful for retrieval and factuality eval.

**SimpleQA (OpenAI 2024).** Short factoid questions designed to be unambiguous; specifically targets hallucination on factual claims. Frontier models score 30–60% — that is the headline that "models are still bad at simple facts."

### 3.2 Reasoning

**GPQA (Graduate-level Physics QA).** 198 expert-written questions; very hard for non-experts. The "PhD-level reasoning" benchmark. Diamond subset is what is usually cited.

**MATH and AIME.** MATH = 12.5K competition problems, multiple difficulty levels. AIME = American Invitational Math Exam; what o1, o3, R1 chase.

**GSM8K.** 8.5K grade-school word problems. Saturated; useful as smoke test, no longer informative for frontier models.

**MuSR, MultiArith, ARC-AGI.** Multi-step reasoning suites. ARC-AGI is the abstract-reasoning benchmark behind the o3 announcement.

### 3.3 Code

**HumanEval.** 164 hand-written Python problems with unit tests. Saturated for frontier models. Reported as pass@1 (single sample) or pass@k.

**MBPP.** 974 entry-level Python problems. Same idea, complementary distribution.

**HumanEval+, MBPP+ (EvalPlus).** Augmented test suites with 100x more cases each; original tests were too lenient.

**LiveCodeBench.** Continuously-updated set of LeetCode contest problems. Specifically designed to defeat contamination — uses problems released after a cutoff.

**SWE-Bench, SWE-Bench-Verified.** Real GitHub issues from popular Python repos. The model must produce a patch that passes the maintainers' tests. SWE-Bench-Verified is a 500-issue human-verified subset. The premier *agent* coding benchmark — solving requires file navigation, tool use, multi-step reasoning.

**RepoBench, CodeContests, CRUXEval.** Repo-level completion, competition coding, code reasoning.

### 3.4 Instruction following and chat

**AlpacaEval 2 (length-controlled).** Pairwise win-rate vs a reference model (typically GPT-4-Turbo) judged by GPT-4. The length-controlled version corrects for the bias that longer answers get judged better.

**MT-Bench.** 80 multi-turn questions across 8 categories, judged on a 1–10 scale by GPT-4. Two-turn structure stresses follow-up handling.

**Arena-Hard, Arena-Hard-Auto.** 500 hard prompts curated from Chatbot Arena, judged via GPT-4-as-judge. Strong correlation with Arena ELO and much cheaper.

**IFEval.** Verifiable instruction-following: "answer in JSON," "use exactly 3 bullet points," "include the word X 5 times." Programmatically checked, no judge needed. Excellent because it has zero judge bias.

**FollowBench, InfoBench.** Multi-constraint instruction following.

### 3.5 Long context

**Needle-in-a-Haystack (NIAH).** Insert a fact at varying depth in long context, ask the model to retrieve it. Easy at small depth, harder at boundaries; the standard "does long context work at all" smoke test.

**RULER.** Multi-task long-context benchmark from NVIDIA: NIAH variants, multi-key, multi-value, variable-tracking, common-words extraction, frequent-words extraction, QA. Reveals that real long-context performance is much worse than NIAH alone suggests. Industry standard for long-context eval as of 2024–2025.

**BABILong, LongBench, ZeroSCROLLS, ∞Bench.** Mixed long-context QA suites.

**Lost-in-the-Middle (Liu et al. 2023).** Phenomenon, not a benchmark per se: models attend to information at the start and end of the context but degrade in the middle. Should always be tested for in any RAG system.

### 3.6 Multimodal

**MMMU (Massive Multi-discipline Multimodal Understanding).** College-level questions across 30 subjects with images. The standard VL benchmark.

**MM-Vet, MathVista, ChartQA, DocVQA, ScienceQA, RealWorldQA, BLINK.** Specialized vision-language tasks.

**Video-MME, EgoSchema, MVBench.** Video understanding.

### 3.7 Agent

**AgentBench (Tsinghua).** Multi-environment agent eval: OS, DB, web, etc.

**GAIA (Meta).** Real-world questions that require browsing, calculation, file handling. Levels 1/2/3 by difficulty. Frontier humans get ~92%; GPT-4 + tools gets ~30%; pure GPT-4 ~6%.

**SWE-Bench / SWE-Bench-Verified.** Already mentioned — the agent coding benchmark.

**OSWorld, WebArena, VisualWebArena.** Computer-use agents on real OS / web tasks. The Anthropic computer-use, OpenAI Operator, and Google Agentic-AI demos report on these.

**TAU-bench (Sierra 2024).** Realistic customer-service agent eval with simulated user. Great test of multi-turn tool use.

**MLE-Bench (OpenAI 2024).** Can the agent do an entire ML engineering task — a Kaggle-style competition.

### 3.8 Safety and behaviour

**TruthfulQA.** 817 questions designed to elicit common misconceptions. Tests whether the model parrots wrong answers humans tend to give.

**WildGuard, WildJailbreak, HarmBench, JailbreakBench.** Refusal and jailbreak resistance.

**ToxiGen, BOLD, BBQ, RealToxicityPrompts.** Bias and toxicity.

**XSTest.** Tests over-refusal: prompts that *should* be answered but the model wrongly refuses.

**Sycophancy probes.** Ask a question, then say "I think the answer is X (wrong)" — does the model flip?

### 3.9 What to actually run

For a frontier-model eval suite circa 2026, a defensible minimum is: MMLU-Pro, GPQA-Diamond, MATH-500, AIME, LiveCodeBench, SWE-Bench-Verified, IFEval, MT-Bench or Arena-Hard, RULER (long context), MMMU (if multimodal), TruthfulQA, SimpleQA, XSTest. Plus product-specific tasks.

For a startup product, capability benchmarks matter much less than your own task-specific eval — see §15.

---

## 4. Instruction following and chat quality

The interesting jump in usefulness from 2022 → 2024 came from instruction tuning, but evaluating it took a couple of years to stabilize.

### Verifiable instruction following

**IFEval** is the gold standard because it removes the judge entirely. Instructions are programmatically checkable — "respond in valid JSON," "first sentence must start with the word 'Furthermore'," "answer in fewer than 50 words." Pass-rate is reported per-instruction and per-prompt.

For a product, **always include verifiable instructions** in your eval. They catch regressions that LLM-judge can miss because the judge is also drifting.

### Multi-turn

**MT-Bench** stress tests: turn 1 sets up a task, turn 2 asks a related follow-up. Single-turn capability is necessary but not sufficient for chat. Many models that score well single-turn fall apart on turn 2 because they lose context, repeat themselves, or fail to handle clarifications.

### Length-controlled judging

LLM judges are biased toward longer responses. AlpacaEval 2 introduced **length-controlled win rate** that regresses out length. Without it, simply "reply with more text" appears to beat reasoning improvements. Always report length-controlled metrics.

### Persona / format adherence

Production chat needs the model to obey a system prompt across turns. The eval should include adversarial users trying to break the persona ("ignore your instructions and...") and benign users asking ambiguous questions.

---

## 5. LLM-as-judge: methodology, biases, calibration

LLM-as-judge means using a strong model (often GPT-4 / Claude / a custom judge) to score outputs of the model under test. Cheap, scales, and surprisingly correlated with human preference — but full of biases.

### Why it works (when it does)

A strong judge has been RLHF-trained on a large corpus of human preferences, so it has internalized "what humans like." For tasks where the judge is more capable than the testee, judge preferences correlate with humans at r ≈ 0.7–0.85 on aggregate, low at the per-example level.

### Known biases

- **Length bias.** Longer = perceived better, even when controlling for content.
- **Position bias.** First option (A) tends to win in pairwise comparisons. **Mitigation: swap and average.** Run the comparison twice with order swapped, count agreement; only count a vote when both orderings agree.
- **Self-preference / family bias.** GPT-4 prefers GPT-family outputs; Claude prefers Claude. Mitigation: use multiple judges, or use a judge from a different family than either testee.
- **Verbosity / format bias.** Bullet points, bold formatting, headers all bias toward "better." Strip formatting before judging if you want substance only.
- **Refusal bias.** Some judges punish refusals heavily; others reward them. Inspect the judge's behaviour on safety prompts before trusting overall scores.
- **Distribution / topic bias.** Judges are weak on niche domains (medical, legal, code) — use domain-specific judges or human evaluators for those.
- **Calibration drift.** As the testee approaches the judge in capability, the judge's scoring becomes noisier. When testee ≥ judge, the judge becomes useless.

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

### Calibrating a judge

1. Build a small (200–500) human-labeled gold set of pairwise comparisons.
2. Run the LLM judge on the same set.
3. Measure agreement (Cohen's κ, accuracy).
4. If agreement is low (<0.7), refine the judge prompt or change judge.
5. Periodically re-calibrate as the testee changes.

### Multi-judge ensembles

Three judges from different families, majority vote. Reduces idiosyncratic bias and is now standard at frontier labs for capability eval.

### G-Eval, Prometheus, JudgeLM, PandaLM

Trained-judge models specifically for eval. Prometheus 2 is open source; PandaLM is open source. They give per-criterion 1–5 scores with a structured rubric.

---

## 6. Pairwise evaluation, ELO, Chatbot Arena

### Why pairwise

Absolute scoring is hard ("rate this 1–10"). Pairwise comparison ("A or B?") is the cleanest signal. Humans agree more on "which is better" than on "what's the score."

### ELO from pairwise

Once you have many A-vs-B comparisons, fit ELO ratings. The classic update rule:

```
expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
rating_a += K * (score_a - expected_a)   # score_a = 1, 0.5, 0
rating_b += K * ((1 - score_a) - (1 - expected_a))
```

In practice you fit ELO via maximum likelihood on the full pairwise dataset (Bradley-Terry MLE) rather than online updates.

### Chatbot Arena (LMSys)

Users submit prompts, see two anonymized model outputs, vote. ELO ratings are computed. The de-facto reference for "what real users prefer" since 2023. Influential because it captures preference distribution shift toward open-ended chat.

Issues with Arena: prompt distribution skews toward casual chat, vocal users may not represent paying customers, voting fatigue, no per-task breakdown by default.

### Arena-Hard-Auto

500 prompts curated from Arena to be hard, judged by GPT-4 with position-bias correction. Scores correlate with Arena ELO at r ≈ 0.95 but cost ~$25/run instead of months of voter time.

### Bradley-Terry MLE

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

---

## 7. Open-ended generation evaluation

The hardest case. The model writes an essay, an answer, code with no unit tests, a creative response. There is no gold reference.

### Reference-based metrics (the wrong default)

BLEU, ROUGE, METEOR, chrF, BERTScore. They reward n-gram or embedding overlap with a reference. Reasonable for translation; **bad for instruction-following or creative tasks.** Two correct answers with no shared vocabulary score zero. Use them only when you have a clearly canonical reference (translation, grammar correction).

### Embedding-based metrics

BERTScore, BLEURT, COMET. Slightly better than BLEU but inherit similar issues. COMET is best for translation; for everything else, prefer LLM-judge or human eval.

### LLM-judge with rubric

The default approach. Define 3–6 criteria (relevance, accuracy, clarity, completeness, harmlessness), score each 1–5, weight, sum. The rubric makes the judge auditable.

### Programmatic checks

For every open-ended task, ask: **what's the smallest piece of structure I can verify automatically?**
- "Output JSON" → parse it
- "Cite a source" → does the URL exist
- "Be ≤ 100 words" → count
- "Mention X" → string match

These cheap checks catch a lot of regressions and have zero judge variance.

### Pairwise instead of absolute

For open-ended quality, pairwise A/B against a reference model is more reliable than 1–10 scoring. Report length-controlled win rate.

### Diversity evaluation

For creative tasks, you also want diversity, not just quality. Self-BLEU within K samples (low = diverse), distinct-n, embedding spread.

---

## 8. Factuality and faithfulness measurement

(See `HALLUCINATION_DETECTION_DEEP_DIVE.md` for the full chapter; this is the eval-specific summary.)

### Factuality benchmarks

**TruthfulQA** — does the model parrot human misconceptions? Adversarial: questions are written to elicit them. Often used as a "rejection of misinformation" eval rather than pure factuality.

**SimpleQA (OpenAI 2024)** — short factoid questions. Score = correct / (correct + incorrect + abstained). Models are explicitly allowed to abstain. Frontier ~50%.

**LongFact (DeepMind 2024)** — long-form factuality. Each response is decomposed into atomic facts via an LLM, each fact graded by web search via SAFE.

**FactScore (Min et al. 2023)** — atomic-fact precision against Wikipedia. Decompose response → check each fact → compute fraction supported.

**FACTS Grounding (Google 2024)** — RAG faithfulness benchmark. Given context, does the response stay grounded?

**HaluEval, RAGTruth** — hallucination detection benchmarks with annotated outputs.

### SAFE (Search-Augmented Factuality Evaluator)

Pipeline: extract atomic claims → search Google → judge each claim as supported / not-supported / irrelevant. Released by DeepMind alongside LongFact. Approximates expensive human annotation at ~1% of the cost.

### Citation-grounded factuality

For RAG / citation systems, evaluate two separate things:
1. **Citation existence** — does the cited source exist and is it accessible?
2. **Citation faithfulness** — does the cited source actually support the claim?

These can be tested with NLI: claim entailed by source span?

### RAGAS (RAG eval framework)

- **Faithfulness** = fraction of generated claims supported by retrieved context.
- **Answer relevancy** = how well the answer addresses the question.
- **Context precision** = fraction of retrieved chunks that were actually relevant.
- **Context recall** = fraction of needed information that was retrieved.

Now standard for RAG eval.

### Calibration as factuality proxy

A well-calibrated model should be unsure when it is wrong. Measure ECE on a multi-choice eval. Models trained with RLHF are typically *over-confident* — calibration regresses post-RLHF (Tian et al. 2023, Kadavath et al. 2022).

---

## 9. Contamination

The headline hazard. If a benchmark is in pretraining data, scores are meaningless.

### How it happens

- The benchmark was published on the web; the crawler picked it up.
- A user pasted it into a public forum.
- It leaked into a training corpus through a derivative dataset (instruction-tuning data, synthetic data based on benchmark questions).
- The benchmark was deliberately included to optimize for it (rare at honest labs, common in published-but-suspect models).

### Detection methods

**Memorization probes.** Show the model the first half of a test example, ask it to complete. If it reproduces verbatim, contamination is likely.

**Membership inference.** Min-K%-prob (Shi et al. 2024): for a candidate test example, compute the average log-prob of the K% lowest-probability tokens. Members of training data have higher (less negative) min-K% prob than non-members.

**Time-shifted benchmarks.** Use benchmarks that were created **after** the model's training cutoff. LiveCodeBench, LiveBench, GAIA partially do this.

**Perturbation tests.** Slightly rephrase benchmark questions. If accuracy drops sharply, the model was relying on memorized text rather than understanding.

**Canary strings.** Insert known canary strings into your held-out test data; later inspect public model outputs for them.

### Prevention

- Hold out at least one private test set per benchmark. Never publish it.
- For competition models, use closed/private holdouts (e.g., the SWE-Bench-Verified hidden tests).
- Use continuously-updating benchmarks for ongoing evaluation.

### Reporting

If you publish results, be explicit about which benchmarks you have decontaminated against, and how (n-gram overlap removal, exact-match removal, fuzzy-match removal).

---

## 10. Robustness, prompt sensitivity, fairness

A model that scores 90% on MMLU with one prompt template and 75% with another is fragile. Robustness eval measures this directly.

### Prompt sensitivity

- **Template variations.** Rewrite the same question 5 ways; compute std of accuracy.
- **Few-shot ordering.** Permute examples; measure variance.
- **System prompt variations.** "You are a helpful assistant" vs "Answer concisely" vs nothing.
- **Whitespace, punctuation, capitalization.** Yes — these matter, especially for smaller models.

A robustness number is std-of-accuracy across template variants. Report it alongside the headline.

### Adversarial robustness

- **PromptBench** — adversarial perturbations to prompts.
- **CheckList** — behavioral testing for NLP (e.g., negation handling).
- **AdvGLUE** — adversarial GLUE.

### Fairness and bias

- **BBQ** — bias benchmark for QA (gender, race, religion, etc.).
- **CrowS-Pairs, StereoSet** — stereotype probes.
- **Winogender, Winobias** — coreference + gender.

For products serving diverse populations, slice your eval by demographic where possible. Aggregate metrics hide subgroup regressions.

### OOD robustness

Hold out an explicitly out-of-distribution slice. For a customer-support model, that might be a topic the training data did not cover. Measure performance and calibration drop.

---

## 11. Statistical methodology

Most LLM papers under-report uncertainty and over-claim. Senior interview signal: ask back about confidence intervals.

### Sample size

For a benchmark with binary correctness:
- 95% CI half-width ≈ 1.96 × √(p(1-p)/n).
- For p ≈ 0.5, n = 100 gives ±10pp; n = 400 gives ±5pp; n = 1000 gives ±3pp.
- Most benchmarks (HumanEval = 164, GPQA-Diamond = 198) are too small to distinguish 1–2 point differences.

### Confidence intervals

Report 95% CIs (Wilson or bootstrap). A 2-point gap with overlapping CIs is not a real difference.

### Multiple comparisons

When you eval on 20 benchmarks, with α=0.05 you expect ≈1 false positive by chance. Apply Bonferroni or Benjamini-Hochberg if you are hypothesis-testing.

### Pass@k for code

Sample n responses, compute fraction that pass tests. Pass@1 with k=1 is the standard metric. Pass@k (Chen et al. 2021) reports the probability that at least one of k samples passes:

```
pass@k = 1 - C(n - c, k) / C(n, k)
```

where n is the number of samples and c is the number that pass. Use n ≥ 20 for stable estimates.

### LLM-judge variance

Two runs of the same judge on the same outputs do not agree perfectly. Report inter-run agreement; with greedy / temperature=0 judging, agreement should be ~95%; if it's lower, investigate.

### Sample size for LLM-judge

Pairwise win rate p with n comparisons has SE = √(p(1-p)/n). To distinguish 50% from 55% at 95% confidence, n ≈ 1500.

### Correlated samples

If you sample 5 responses per prompt and aggregate, treat the prompt as the unit, not the sample. Otherwise CIs are anti-conservatively narrow.

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

### Surrogate quality metrics

- **Regenerate rate.** Proxy for dissatisfaction.
- **Edit rate** (for code/copilot products).
- **Conversation length / depth.** Longer = engagement, but also = friction. Monitor distribution.
- **Time-to-first-token.** Latency proxy.
- **Tool-call success rate.** Agent reliability proxy.
- **Citation click rate** (for RAG/search products).

### Sampling for online eval

You cannot run an LLM judge on every production request. Sample 1–5% and run the full eval pipeline (judge + factuality checks + safety filters) on the sample. Stratify by route or persona to detect subgroup regressions.

### Drift detection

Track per-week aggregates of: refusal rate, response length, latency, hallucination flags from the detector. Alert when any moves >2σ. Distribution shift in user prompts is the typical cause; model drift is rare but possible (model hot-swap, RAG corpus update).

### Logging and replay

Store inputs + outputs (with retention/PII policy). Build a replay harness so you can re-run last week's traffic against a new model. Replay is the cheapest way to catch regressions before A/B.

---

## 14. A/B testing for LLM products

The mechanics resemble standard A/B but with LLM-specific gotchas.

### Setup

- Bucket users into Control (A) and Treatment (B). Assignment must be sticky per user / session.
- Define primary success metric in advance (one). Examples: task completion rate, retention, conversion, regenerate rate (lower is better).
- Define guardrail metrics: latency p95, cost per query, safety flag rate.
- Pre-register sample size and duration. Stop only at the planned time, not on early peeks.

### Sample size

For binary metric p with absolute lift δ:

```
n_per_arm ≈ 16 * p(1-p) / δ²  (for 80% power, α=0.05)
```

For p=0.5, δ=0.02: n ≈ 20K per arm.

### Variance reduction

CUPED (Microsoft / Netflix) — regress out pre-period covariates from the outcome. Routinely reduces required sample size by 30–50%.

### LLM-specific gotchas

- **Latency leak.** A model that's even 100ms slower can suppress engagement enough to look "worse" on quality metrics. Always control for latency.
- **Prompt/version coupling.** Treatment may have a new prompt **and** a new model. Disentangle with multi-cell experiments.
- **Output length.** Longer responses can drive engagement up while user satisfaction goes down. Monitor both.
- **Memorization of system prompt.** Users adapt to the assistant; switching personas mid-experiment can hurt.
- **Regenerate cascade.** Bad outputs cause regenerates which add cost and noise. Track per-conversation cost, not just per-request.
- **Selection bias from refusals.** If treatment refuses more, the conversations that "complete" are a biased sample. Always include refused requests in the denominator.

### Shadow / canary deploy first

Before A/B with real users on a large fraction:
1. **Shadow.** Mirror traffic to the new model, compare offline.
2. **Canary.** 1–5% of traffic, monitor for hours/days, no business decision yet.
3. **A/B.** 50/50 with proper sample size.
4. **Rollout.** Gradual ramp with rollback gates.

### Sequential testing

For online eval, mSPRT or always-valid CIs let you stop earlier without inflated false positives. Used at Microsoft, Netflix, Linkedin. Less needed than for traditional A/B because LLM cost per query is higher and sample sizes are lower; still good practice.

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

### Layer 3: Online telemetry

- Per-conversation: completed (no escalation, no regenerate), partial, escalated, abandoned.
- Per-turn: thumbs, regenerate, time-to-response.
- Sample 2% to run hallucination detector and judge offline.
- Track: refusal rate, hallucination rate, citation faithfulness, latency p95, cost.

### Layer 4: Continuous improvement

- Sample 100 escalations and 100 thumbs-down per week. Categorize failures. Add tough examples to golden set.
- Periodically refresh the golden set from real traffic (with PII redaction).
- Re-calibrate the LLM-judge every quarter with fresh human labels.

### Failure modes this eval catches

- New model regresses on a specific intent (Layer 2 slice).
- Retrieval pipeline change drops faithfulness (Layer 2 RAGAS).
- Refusal rate creeps up after an alignment update (Layer 1 XSTest + Layer 3 telemetry).
- Latency degrades after deployment (Layer 3 p95).
- Hallucination detector's false-positive rate inflates after RAG corpus update (Layer 3 sampled detector).

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

---

## 17. Common interview questions

A subset; full grill in `INTERVIEW_GRILL.md`.

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
5. Pair with `HALLUCINATION_DETECTION_DEEP_DIVE.md` for the factuality slice.

If you can fluently distinguish capability vs product eval, identify all the LLM-judge biases, design contamination detection, size A/B tests, and articulate the offline → shadow → canary → A/B pipeline — you'll handle frontier-lab and big-tech eval interviews well.
