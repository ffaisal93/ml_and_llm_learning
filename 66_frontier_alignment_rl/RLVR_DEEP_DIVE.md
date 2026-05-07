# RLVR — Reinforcement Learning with Verifiable Rewards — Deep Dive

> Frontier-lab research-scientist depth on RLVR: the algorithm zoo (DAPO, Dr. GRPO, RLOO, REINFORCE++, GSPO, VinePPO, PRIME), verifier design, reward shaping, failure modes, open-source infra, and a substantial section on **low-resource multilingual reasoning** — the most under-explored applied RLVR direction in 2025-2026, with concrete research-project blueprints.

Pair with `REASONING_MODELS_DEEP_DIVE.md` (the broader reasoning-RL story), `FRONTIER_REWARD_MODELING.md` (RM landscape), and `OPEN_SOURCE_POSTTRAIN_PLAYBOOKS.md` (R1, Tülu 3 recipes).

---

## Table of contents

1. RLVR — what it is, why it works, why it's special
2. The 2024-2025 RLVR algorithm zoo
3. Verifier design — the often-neglected half
4. Reward shaping — correctness, format, length, language, step
5. KL regularization choices
6. Curriculum, exploration, and warm-start
7. Common failure modes and debugging
8. Open-source RLVR infrastructure
9. Multi-modal RLVR — math, code, tool use, vision
10. **Low-resource multilingual reasoning with RLVR**
11. Concrete research project blueprints
12. Datasets and benchmarks
13. Open frontier questions
14. References

---

## 1. RLVR — what it is, why it works

**RLVR** = train a policy via RL where the reward signal comes from a *deterministic verifier* on the model's output, not from a human-feedback reward model.

Examples of verifiable rewards:
- Math final answer matches sympy-canonical ground truth.
- Generated code passes unit tests.
- Tool-use trace ends in the correct final state.
- Multiple-choice extraction matches the labeled letter.
- Lean / Coq verifier accepts a generated formal proof.
- A regex extracts the right structured answer.

**Why it's special.**

1. **Zero labeler cost.** Verifiers are programs; they scale to millions of problems for free.
2. **Zero labeler bias.** No length bias, sycophancy, format bias, or persona bias from the reward signal.
3. **Hard to reward-hack on outcome.** The model can't write more text, be more polite, or use bullet points to get a higher score; only correctness matters.
4. **Strong gradients on hard problems.** When the policy gets a problem right, it knows; when it doesn't, it knows. Crisp signal.
5. **Composes with non-verifiable rewards.** Use RLVR for the verifiable subset, RLHF/RLAIF for the rest.

**Why it works.** Pretraining gives the model priors over reasoning steps. RL turns those priors into a search policy by rewarding correct outcomes. The policy learns to allocate inference compute (long CoT) on hard problems and short answers on easy ones — discovered, not engineered.

**The one-line takeaway:** RLVR is the cleanest signal in modern post-training, and reasoning RL is currently the highest-impact application.

---

## 2. The 2024-2025 RLVR algorithm zoo

The base policy gradient is REINFORCE; everything else is variance reduction, sample efficiency, or robustness. Memorize this list.

### 2.1 PPO with value head

The classical RLHF setup. Policy + critic (value head). Advantage = reward + GAE bootstrap − baseline.

**Pros.** Lower variance via critic.
**Cons.** Critic has to be trained, doubles memory. Critic is often poorly calibrated for sparse / sequence-level rewards.

### 2.2 GRPO (Group Relative Policy Optimization, DeepSeek)

For a prompt $x$, sample $G$ rollouts. Use the group's mean reward as the baseline; standardize by group std. No critic.

$$
\hat A_{i,t} = \frac{r_i - \mathrm{mean}(\{r_1, \ldots, r_G\})}{\mathrm{std}(\{r_1, \ldots, r_G\})}.
$$

**Pros.** No critic; sample-efficient; particularly good for verifiable-reward settings (the reward is sequence-level, so a token-level critic is wasteful).
**Cons.** Group size adds compute; the std normalization is contentious (Dr. GRPO removes it).

### 2.3 Dr. GRPO

GRPO without the std normalization in the advantage. Argued to be more stable when reward variance correlates with problem difficulty (you don't want easy-problem easy-wins to dominate via small std).

$$
\hat A_{i,t} = r_i - \mathrm{mean}(\{r_1, \ldots, r_G\}).
$$

### 2.4 DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization, ByteDance 2024)

Four contributions stacked on GRPO:
- **Clip-Higher.** Asymmetric clip — `clip(ρ, 1−ε_low, 1+ε_high)` with `ε_high > ε_low`. Lets the policy take bigger steps on positive-advantage tokens (encouraging exploration) without amplifying negative-advantage updates.
- **Dynamic Sampling.** Filter rollout groups where all rewards are equal (no learning signal); resample.
- **Token-Level Policy Gradient Loss.** Sum over tokens not over sequence length-normalized. Long correct answers should not be down-weighted just because they're long.
- **Overlong Reward Shaping.** Penalize CoTs that hit the max-length cap (often correlated with non-termination / failure).

DAPO results on AIME match or exceed GRPO with significantly less compute. **The current canonical algorithm for reasoning RL outside DeepSeek's stack.**

### 2.5 RLOO (REINFORCE Leave-One-Out, Kool 2019; revived for RLHF by Ahmadian 2024)

Like GRPO but the baseline for sample $i$ is the *mean of the other samples* in the group:

$$
\hat A_i = r_i - \frac{1}{G - 1} \sum_{j \neq i} r_j.
$$

Unbiased; no critic; simple. Ahmadian et al. ("Back to Basics") showed RLOO matches or beats PPO on RLHF benchmarks.

### 2.6 REINFORCE++

REINFORCE with: token-level KL penalty, advantage normalization, clipping. Closer to PPO but without value head.

### 2.7 VinePPO

Variance-aware credit assignment via *Monte-Carlo rollout from each step*. From step $t$, sample $K$ continuations to estimate the value at $t$. Per-step advantage = (reward) − (estimated value). Closes the gap between sequence-level rewards and per-token credit.

**Pros.** Better credit assignment than GRPO/RLOO.
**Cons.** $K\times$ rollout cost.

### 2.8 PRIME (Process Reward Implicit through Implicit rewards, Cui 2025)

Joint training: a *generative reward model* (the "implicit PRM") and the policy. The implicit PRM gives token-level rewards derived from the difference of log-probs between policy and reference. Plus outcome reward from the verifier. No need to train a separate explicit PRM. Demonstrates strong gains on math.

### 2.9 GSPO (Group Sequence Policy Optimization, Qwen 2025)

Sequence-level (not token-level) importance ratio. Argued to stabilize MoE / large-model RL where token-level off-policy ratios can blow up.

### 2.10 Step-RL (PRM-supervised RL)

Use a Process Reward Model to give a per-step reward in the RL update. Conceptually appealing; in practice (per R1) hasn't beaten ORM-only with a strong base.

### 2.11 Iterative DPO / Online DPO

Not strictly RL, but adjacent. Repeat: sample policy, build preference pairs from verifier-graded outputs, DPO. Cheap, no value head, no rollout heavy lifting. Used by Llama 3 and Tülu 3 in some stages.

### 2.12 The "what to actually use" heuristic

- **Verifiable reward, big reasoning model:** GRPO or DAPO (ByteDance Stack: DAPO).
- **Verifiable reward, smaller / academic compute:** RLOO is hard to beat for simplicity.
- **Mixed verifiable + judge reward:** PPO with value head (because reward is more dense / smooth) or DAPO with composite reward.
- **Speed of iteration matters more than peak quality:** Iterative DPO.
- **You want a per-step signal without training a PRM:** PRIME (implicit PRM).

---

## 3. Verifier design — the often-neglected half

Most RLVR papers focus on the algorithm; the verifier is the actual moat. A bad verifier silently caps your model.

### 3.1 What a good verifier does

- **High recall on equivalent answers.** "0.5", "1/2", "\frac{1}{2}", "0.50" should all match.
- **Low false-positive rate.** Doesn't accept wrong answers that look like the right one.
- **Fast.** Will be run millions of times during training.
- **Robust to format.** Final-answer extraction handles `\boxed{}`, `<answer>...</answer>`, "Final answer: 42", etc.
- **Hard to game.** Can't be hacked by exotic equivalent-but-not-quite formats.

### 3.2 Math verifier patterns

- **Sympy canonicalization.** `sympy.simplify(expr1 - expr2) == 0`. Handles `1/2` vs `0.5`. Handles algebraic equivalents. Has corner cases (transcendental equivalence is undecidable).
- **Numeric tolerance for decimals.** `abs(pred - gold) < 1e-6 * max(1, abs(gold))`.
- **Set / list / multiset matching.** For "list all roots."
- **Formal proof verifier (Lean / Coq).** For IMO-level problems. Cleanest but limited domain.
- **Multi-choice extraction.** Match `(A|B|C|D)` from response.

### 3.3 Code verifier patterns

- **Hidden test suite.** Run code against held-out tests. Watch for: time limits, memory limits, sandboxing.
- **Functional vs syntactic.** Best practice: only count *passed-functional-tests*, not "compiles without error."
- **Fuzz testing.** Random inputs match a reference implementation.
- **Differential testing.** Compare outputs to a known-correct implementation.

### 3.4 Tool-use / agentic verifier patterns

- **Final-state checking.** Did the calendar slot get booked? Did the email send to the right person? (Sandboxed simulation.)
- **Sub-goal completion.** Multi-step task; verify each sub-goal.
- **Trace replay.** Compare against a reference trace.

### 3.5 Generative LLM-as-verifier

When deterministic verification is impossible (proof correctness in informal English, summary fidelity), use an LLM judge. Caveats:
- Slower (one judge call per rollout).
- Hackable (prompt injection in the rollout).
- Self-preference bias if judge family matches policy family.
- Needs adversarial training and held-out human gold to keep honest.

### 3.6 Composite verifiers

The 2025 frontier pattern: combine verifiable + LLM-judge.

```
total_reward = correctness * w_correct
             + format * w_format
             + language_consistency * w_lang
             + judge_score * w_judge   (only on non-verifiable subtasks)
```

Hierarchical is more stable than additive: only count `judge_score` if `correctness == True`, etc.

### 3.7 Verifier robustness — adversarial training

The policy WILL find verifier exploits. Defense:
- Build adversarial test cases into the verifier.
- Run negative-example fuzz tests on the verifier itself.
- Ensemble verifiers (sympy + numeric + regex agreement).
- Periodic manual sample inspection.

---

## 4. Reward shaping in RLVR

A practical zoo of components, all of which appear in published recipes.

### 4.1 Correctness reward

Binary or graded:
- **Binary.** $r \in \{0, 1\}$ (correct / wrong). Simple, sparse.
- **Partial credit.** Math: $r = 0.3$ for "correct setup, arithmetic error"; $r = 1$ for fully correct.
- **Per-test-case for code.** $r = (\text{tests passed}) / (\text{total tests})$. Smoother.

### 4.2 Format reward

Penalize / reward output structure: did it use `<think>...</think><answer>...</answer>`? Did it produce a parseable boxed answer?

Magnitude should be much smaller than correctness — otherwise the model learns to emit format with garbage answers.

### 4.3 Length reward

- **Soft penalty above a cap.** Clip CoT to N tokens; penalize `(actual_len − N) / N` if exceeded.
- **Soft bonus for non-trivial CoT.** Prevents the model from collapsing to greedy short answers.
- **DAPO's "overlong reward shaping."** Specifically penalize CoTs that hit the max-length context — these are usually non-terminating failures.

### 4.4 Language consistency reward

Critical for multilingual settings (and for R1-Zero's mid-CoT English↔Chinese drift). Detect dominant language in CoT; penalize switches.

```python
def language_consistency(cot, target_lang):
    detected = langdetect.detect_langs(cot)
    target_prob = next((d.prob for d in detected if d.lang == target_lang), 0)
    return target_prob   # in [0, 1]
```

### 4.5 Step-level (PRM) reward

Per-step from a trained PRM. Empirical evidence: helps in some setups (Math-Shepherd, OmegaPRM) but R1's headline finding is that ORM alone is sufficient for strong reasoning.

### 4.6 Diversity / exploration bonus

Reward variance within a group reduces; if your group's rewards are all equal (all wrong or all right), you have no learning signal. DAPO's "dynamic sampling" simply discards such groups and resamples. Alternative: entropy bonus.

### 4.7 Repetition penalty

Penalize n-gram repetition in the CoT. Prevents the policy from learning a degenerate "loop forever" attractor.

### 4.8 Reward composition pattern

Hierarchical with hard gate on format:

```python
def compute_reward(rollout):
    if not has_valid_format(rollout): return 0.0
    correctness = verifier(rollout)
    if not correctness: return -0.1   # small negative for wrong-but-formatted
    bonus = (
        0.1 * language_consistency(rollout) +
        0.05 * (1.0 if not overlong(rollout) else 0.0)
    )
    return 1.0 + bonus
```

---

## 5. KL regularization choices

KL keeps the policy near the reference (SFT or DPO model) — without it, the policy drifts into incoherence.

### 5.1 Forward KL vs reverse KL

- **Forward KL** $\mathrm{KL}(p \,\|\, q)$: mean-seeking; spreads probability mass; encourages diverse outputs.
- **Reverse KL** $\mathrm{KL}(q \,\|\, p)$: mode-seeking; concentrates on high-prob regions; encourages confident outputs.

Standard PPO/GRPO uses *reverse KL* (mode-seeking) which is fine for reasoning where you want confident correct outputs.

### 5.2 Token-level vs sequence-level KL

- **Token-level.** KL added to per-token reward. Strong regularization.
- **Sequence-level.** Estimated from log-prob ratio of full sequences. Weaker but cheaper.

### 5.3 Adaptive $\beta$

R1 uses adaptive $\beta$ with KL targets. If running KL > target, increase $\beta$; if < target, decrease. Standard PPO trick.

### 5.4 No-KL ablations

Some recent work (Dr. GRPO, some RLVR variants) argues that with strong correctness rewards and small enough updates, the KL term can be dropped. Empirically: depends on base model strength.

---

## 6. Curriculum, exploration, warm-start

### 6.1 Cold-start vs warm-start

- **Cold-start (R1-Zero style):** start RL directly from base model. Surfaces emergent reasoning but produces illegible CoTs.
- **Warm-start (R1, Tülu 3, Qwen):** start RL from an SFT'd model that already has the format. More stable and legible, possibly less exploratory.

The empirical answer: warm-start dominates for production. Cold-start is a research curiosity.

### 6.2 Curriculum

If the base model gets <1% on the training problems, gradients are pure noise. Solutions:
- Filter problems by difficulty (start at problems the base solves >5% of the time).
- Stage-wise: easy → medium → hard.
- Adaptive: drop problems with all-correct or all-wrong groups; promote problems with intermediate success rates.

### 6.3 Exploration

- **Temperature.** R1 uses T=0.6. Too low → no exploration; too high → noise.
- **Top-p sampling.** 0.9 is default.
- **Group size.** Larger groups = more diverse rollouts per prompt; cost scales linearly.
- **Best-of-N during data construction (rejection sampling).** Even RLVR can be stuck; rejection-sampled SFT data refresh helps.

### 6.4 Replay buffers

Store high-reward trajectories. Replay them during training to keep good behavior alive. Useful when reward is sparse.

---

## 7. Common failure modes

The interview signals — be ready for these.

- **Length explosion.** CoT grows to fill context window with no quality gain. Fix: length penalty + DAPO overlong reward shaping.
- **Mode collapse.** Single reasoning template wins; diversity dies. Fix: entropy bonus, KL cap, diversity-aware sampling.
- **Language mixing.** Mid-CoT switch between languages. Fix: language-consistency reward.
- **Verifier hack.** Model finds answer formats the verifier wrongly accepts. Fix: harden verifier with adversarial cases.
- **Format reward gaming.** Model emits perfect `<think><answer>` with garbage inside. Fix: gate other rewards on correctness; small format weight.
- **Reward sparsity.** All rollouts wrong → no gradient. Fix: curriculum, easier base data, larger group size, longer rollouts.
- **Off-policy drift.** Old samples in the buffer no longer represent current policy. Fix: importance-sampling clipping (PPO ratio clip), refresh buffer.
- **Catastrophic forgetting of non-RL tasks.** Reasoning model regresses on chat. Fix: rejection-sampling SFT after RL stage (R1's stage 3).
- **Reward overoptimization.** True quality plateaus or drops as proxy reward keeps climbing. Fix: KL cap, ensemble verifiers, periodic human eval.
- **Instability with MoE / large models.** Token-level ratios blow up. Fix: GSPO (sequence-level), gradient clipping, smaller LR.

---

## 8. Open-source RLVR infrastructure

What you'd actually use to run RLVR experiments today.

### 8.1 Libraries

- **TRL (HuggingFace).** PPO, DPO, RLOO, GRPO. Most accessible. Single-machine and multi-GPU. Best for small-to-mid scale.
- **veRL (ByteDance, 2024).** Production-grade. Supports DAPO, GRPO, PPO. Distributed training with vLLM rollouts. Used to reproduce R1-style training.
- **OpenRLHF.** Another PPO/DPO/RLHF library, distributed. Good docs.
- **HuggingFace Open-R1.** Specifically reproducing R1's recipe. Curated data + GRPO. Good starting point for replication studies.
- **NeMo-Aligner (NVIDIA).** Production-grade for large models.
- **TRL-Lite / GRPO trainers.** Lightweight community implementations (Will Brown, etc.) for academic-scale experiments.

### 8.2 Rollout backends

- **vLLM.** De facto standard. Fast inference for RL rollouts. Supports continuous batching and chunked prefill.
- **TGI (Text Generation Inference).** Alternative.
- **SGLang.** Flexible, good for tool-use rollouts.

### 8.3 Sample architecture

```
[Trainer (Ray actors, DeepSpeed/FSDP)] ← gradient updates
        ↓ weights sync
[vLLM Rollout Worker(s)] → rollouts → [Verifier(s)] → rewards
        ↑                       ↓
        └─────── replay buffer / batch ──────────┘
```

### 8.4 Compute budget reality check

To reproduce R1-style reasoning RL for a 7B-base model:
- ~1k-10k math problems.
- ~50k rollouts per RL epoch.
- 8-64 H100s.
- 2-7 days.

For a 70B+ model: 100-500 H100s, weeks. Out of academic reach without partnerships.

For a 1.5B-3B model + R1-Distill-style approach: feasible on 8 GPUs.

---

## 9. Multi-modal RLVR

RLVR generalizes beyond text math.

- **Vision-language reasoning.** MathVista with image-rendered math problems. Reward = numeric match.
- **Code with side-effects.** Generated code modifies a sandbox; verifier checks final state.
- **Tool-augmented reasoning.** Model thinks → calls calculator/Python/search → integrates result → continues. Verifier grades final answer. Critical: gradient must flow correctly through tool boundary (usually only the LM tokens are differentiated; tool call results are treated as fixed context).
- **Long-horizon agents.** SWE-Gym, OSWorld, AgentDojo, TAU-bench. Verifier = task-completion check. Credit assignment over hundreds of steps is hard; prevailing approach is final-only reward + GRPO with patience.

---

## 10. Low-resource multilingual reasoning with RLVR

This is the user's specific interest and a genuinely under-explored frontier in 2025-2026. Worth thinking through carefully.

### 10.1 The problem

Almost all reasoning RL has been done in English (and to some extent Chinese, via DeepSeek and Qwen). Yet:
- ~6 billion people don't speak English natively.
- Most low-resource languages have minimal high-quality reasoning data in pretraining.
- Reasoning models trained on English math do *not* transfer cleanly to math problems posed in Bengali, Swahili, Yoruba, Vietnamese, etc.
- Even when the model can produce a correct numeric answer, it often produces it via an English CoT, which is opaque to monolingual users.

There are roughly four communities of users for low-resource multilingual reasoning:
1. **Education** — math tutors for students who don't read English fluently.
2. **STEM access** — scientific reasoning in non-English languages.
3. **Government/legal** — rule-based reasoning over local documents.
4. **Cultural-context reasoning** — problems involving culturally-specific concepts.

### 10.2 Why this is hard

- **Pretraining gap.** A 70B base trained on Common Crawl has 5% non-English tokens, often worse for low-resource languages. The reasoning prior is weak.
- **CoT gap.** Even when the model "understands" the question, it tends to think in English.
- **Verifier-language mismatch.** Math verifiers (sympy, regex) are language-agnostic for numeric answers — a small mercy. But word-problem extraction varies by language: "the product of x and y" has language-specific phrasings.
- **Benchmark gap.** GSM8K-multilingual (MGSM), Belebele, AfriBench, BHASA exist but are smaller and noisier than English benchmarks.
- **Cultural conventions.** Lakh / crore (South Asian numerals), date formats, currency, units, conventions in problem framing.

### 10.3 What's currently done (and the gaps)

**PB-RLSVR — Pivot-Based RL with Semantically Verifiable Rewards (Faisal et al., 2509.25543, Sep 2025).** The first framework to systematically extend RLVR from English to multilingual reasoning *without target-language human annotation*. Uses a high-performing English LLM as a **pivot** to generate reference responses on reasoning tasks; the multilingual policy is rewarded for **semantic equivalence** to the English reference, computed via either embedding-based similarity or machine-translation-based equivalence. Reports +16.4% on Llama-3.1-8B and +10.2% on Qwen3-32B average multilingual performance vs PPO baselines. **The cleanest "pivot teacher" formulation in the literature for transferring English reasoning capability to other languages via RL — and it's the canonical reference for any frontier interview discussion of this problem.**

**MGSM (Multilingual Grade School Math).** Translation of GSM8K into 11 languages. Models score much worse on low-resource languages than English. *Gap: translation-only doesn't capture native problem-framing conventions.*

**Aya / Aya 23 (Cohere).** Multilingual instruction-tuning at scale. *Not reasoning-RL trained; weak on math beyond English.*

**Qwen / DeepSeek multilingual variants.** Strong on Chinese-English; weak on truly low-resource languages.

**MGSM-via-translate-then-reason.** "Translate to English, reason, translate back." Works but loses linguistic faithfulness; brittle to mistranslation.

**Cross-lingual fine-tuning experiments (academic).** Show that fine-tuning a strong English reasoning model on translated data yields modest cross-lingual transfer. *PB-RLSVR (above) is the SOTA RLVR-based approach.*

### 10.4 The technical opportunity

Three things are simultaneously true:
1. Verifiers for math are language-agnostic (the answer is `42`, not "the answer is forty-two").
2. The reasoning capability transfers from English-trained models to multilingual via SFT/distillation.
3. RLVR on multilingual reasoning data has barely been done at scale.

That's an exploitable gap.

### 10.5 Approach landscape

#### A. Translation-augmented training data

- Translate English math problems (GSM8K, MATH, AIME) into target language(s).
- Translate the CoT solutions too.
- SFT on the translated data.
- Then RLVR on translated problems with the language-agnostic verifier.

**Pros.** Cheap, leverages existing high-quality English data.
**Cons.** Translation artifacts; cultural-context loss; doesn't capture native problem-framing.

#### B. Distillation from a strong English reasoning teacher

- Generate long CoTs from R1 / o1 (in English) on math problems.
- Translate the CoT + answer into target language.
- SFT a multilingual base.
- Optional: light RLVR on top.

**Pros.** Inherits R1-grade reasoning. Most plausible path to a strong low-resource reasoning model in 2025.
**Cons.** Translation quality of long CoTs is variable; English-style reasoning patterns may not match target-language conventions.

#### B'. Pivot-based RL with semantically verifiable rewards (PB-RLSVR)

The natural RL extension of approach (B). Instead of distilling once and stopping, *keep the English pivot model as an oracle during RL training*:
- For each training prompt, the pivot model produces a reference response (or reference reasoning trace).
- The multilingual policy generates its own response in target language.
- Reward = semantic equivalence between policy response and pivot response, computed via:
  - Multilingual embedding similarity (e.g., LaBSE, multilingual-E5 cosine), OR
  - Round-trip machine translation + token-level overlap, OR
  - LLM-as-judge for semantic equivalence.

**This is what Faisal et al. 2025 introduces and validates.** The key insight: *you don't need a verifier specific to the target language; you only need a way to compare semantic equivalence to a known-good English reference.* And cross-lingual semantic similarity is a much easier problem than producing target-language ground truth.

**Pros.** No target-language ground truth needed. Pivot stays anchored to English-quality. Composable with verifiable rewards (when answer is numeric, both pivot match AND verifier signal can be combined).

**Cons.** Embedding-based reward can be hacked (model emits text superficially similar to pivot but semantically off). MT-based reward inherits MT errors. Pivot's reasoning style may dominate target-language outputs (homogenization risk).

#### C. Cross-lingual code-switched CoT

- Allow the model to think in English (its strongest reasoning language) but answer in target language.
- Enforce via reward: `correctness * (1 + 0.5 * answer_in_target_language)`.

**Pros.** Leverages strongest reasoning circuit; easy to verify.
**Cons.** Users may want to *see* the reasoning in target language.

#### D. Language-conditioned RLVR

- Add reward component: `language_consistency(CoT, target_language) * w_lang`.
- Penalize mid-CoT switches to English.
- Pair with curriculum: easier-target-language problems first.

**Pros.** Forces the model to learn target-language reasoning, not just answer translation.
**Cons.** Needs strong target-language pretraining priors. Compute may be wasted on language alignment.

#### E. Synthetic data generation

- Use a strong English LLM + translator to generate diverse target-language math problems with verifiable answers.
- Filter for translation quality.
- RLVR on this synthetic set.

**Pros.** Scales easily.
**Cons.** Quality control is hard; risk of distribution shift from real native problems.

#### F. Tool-augmented for low-resource

- Allow the model to call a calculator / Python / translator mid-CoT.
- Reduces the need for the model to do multi-digit arithmetic in target language (a known weakness).
- RLVR on full tool-augmented traces.

**Pros.** Practical; leverages tools.
**Cons.** Needs sandboxing; tool-trace credit assignment is harder.

#### G. Composite / multi-stage

The realistic recipe combines several:
1. Cold-start SFT on R1-distilled CoTs translated into target language.
2. Stage-2 RLVR with **(language-consistency + correctness + format)** reward.
3. Stage-3 rejection-sampling SFT to broaden distribution.
4. Final RLHF for politeness / cultural appropriateness.

This is essentially a "low-resource R1" recipe.

### 10.6 Key technical questions

- How much **target-language pretraining** is needed for RLVR to work? (Hypothesis: a base with ≥5% target-language tokens in pretraining is the floor.)
- How much does **R1-distill from English transfer** to target languages? (Hypothesis: more than translation alone, less than native English performance.)
- Is **language-consistency reward strong enough** to prevent code-switching, or does it suppress reasoning quality?
- Does **multilingual training help English** (positive transfer) or hurt (interference)?
- Are there target-language **cultural reasoning conventions** (e.g., South Asian "lakh" / "crore" numerals; Arabic right-to-left math layout) that need explicit handling?
- What's the **minimum viable problem set size** per language for RLVR to converge?

### 10.7 Failure modes specific to this setting

- **Code-switching collapse.** Model learns to flip to English mid-CoT, gets the answer, switches back. Verifier accepts; user is confused.
- **Translation cascading errors.** If training data is English → translation, then RLVR can amplify translation artifacts.
- **Numeric format mismatches.** Lakh vs million, date formats, decimal commas vs dots.
- **Verifier limitations.** Math verifier is language-agnostic for numbers, but word-answer problems ("what is the capital of X?") need language-aware verification.
- **Loss of cultural context.** A "fair price" problem assumes Western market conventions; low-resource users may have different price intuitions.

---

## 11. Concrete research project blueprints

These are framed as 1-month to 6-month projects. Each is publishable at a major venue if executed well, and any of them is a strong "what would you work on" answer in a research-scientist interview.

### 11.1 Project A: "BengaliMath-RL — RLVR for low-resource mathematical reasoning"

**Goal.** Build a strong Bengali mathematical reasoning model and characterize how RLVR transfers from English.

**Setup.**
- Base: Qwen 2.5 7B (multilingual) or Llama 3.1 8B.
- Stage 1: SFT on R1-distilled English CoTs translated to Bengali (~100k pairs).
- Stage 2: RLVR with GRPO / DAPO on Bengali-translated GSM8K + MATH (~50k problems).
- Reward: `correctness + 0.1 * language_consistency(CoT, "bn") + 0.05 * format`.
- Eval: MGSM-bn, AIME-bn (translate manually verified), out-of-distribution Bengali word problems collected from textbooks.

**Hypotheses to test.**
- H1: R1-distill alone gives ~70% of frontier English-on-Bengali performance.
- H2: RLVR on top recovers another 15-20% gap.
- H3: Language-consistency reward reduces code-switching from 40% to <5%.

**Deliverables.** Model weights, eval harness, ablation paper.

**Why it's interesting.** First public study of low-resource RLVR in Indo-Aryan family. Generalizable recipe.

### 11.2 Project B: "Cross-Lingual Reasoning Transfer via RLVR — A Systematic Study"

**Goal.** Rigorously measure how reasoning capability transfers from English-trained RLVR to N target languages.

**Setup.**
- Take an open-source English RLVR-trained reasoning model (Open-R1, R1-Distill).
- For each of {Bengali, Swahili, Yoruba, Vietnamese, Tamil, Pashto, Quechua}: light SFT on translated data; light RLVR; measure performance vs English baseline.
- Vary: (a) amount of target-lang SFT data, (b) RLVR steps, (c) language-consistency reward weight.

**Hypotheses.**
- Resource level (size of target lang in pretraining) predicts transfer slope.
- Language family matters — Indo-European > Bantu > tonal-isolating.
- RLVR transfer plateau is reached at ~1k target-lang RL examples per language.

**Deliverables.** Publishable empirical paper (à la "Scaling Laws for ...").

### 11.3 Project C: "Code-Switched RLVR — Allowing English-Inner-Voice Reasoning"

**Goal.** Show that allowing the model to think in English (its strongest language) and answer in target produces better accuracy than forcing target-language CoT.

**Setup.**
- Two reward variants: (i) all-target-lang reward, (ii) target-lang-answer-only reward.
- Compare accuracy, user-reported usefulness.

**Hypotheses.**
- (ii) wins on accuracy.
- User study (key!) reveals trade-off: users prefer target-language CoT for educational use even if accuracy is lower.

**Deliverables.** ACL / NAACL paper. Has a clean human-eval angle.

### 11.4 Project D: "Synthetic Multilingual Reasoning Data with Verifier-Guided Filtering"

**Goal.** Generate high-quality target-language math problems with verifiable answers via LLM + filter.

**Pipeline.**
- Strong LLM generates problem + reference answer in English.
- Translates to target language.
- Verifier checks: (a) translation quality (round-trip check); (b) answer correctness (sympy on the original).
- Keep ~10% that pass both.

**Hypotheses.**
- 1M filtered synthetic problems → 100k high-quality.
- Training on these vs human-curated translation has comparable downstream RLVR performance.

**Deliverables.** Public dataset + recipe.

### 11.5 Project E: "Multilingual PRMs — Process Supervision Without Per-Step Translation"

**Goal.** A PRM trained on English process labels, applied to target-language reasoning steps, transfers via shared embedding space.

**Setup.**
- PRM trained on PRM800K (English).
- Target-language RLVR with this PRM as a step-reward.
- Compare to ORM-only baseline.

**Why interesting.** Shows that step-level reasoning structure is language-agnostic enough to transfer.

### 11.6 Project F: "Tool-Augmented Multilingual RLVR"

**Goal.** Allow target-language models to invoke a calculator/Python tool mid-CoT. RLVR with full trace verification.

**Why.** Reduces target-language arithmetic burden on the model. Shifts the RLVR signal to *when* and *how* to use tools rather than how to do arithmetic in target language.

### 11.7 The simplest first step

If you have one week and one A100:
1. Take Qwen 2.5 1.5B-Instruct.
2. Translate ~5k GSM8K problems into target language with Google Translate / NLLB.
3. SFT on translated CoTs from R1-Distill-Qwen-1.5B.
4. Eval on MGSM. Compare to: original Qwen, R1-Distill.
5. Publish the gap analysis.

This is a tiny project but immediately useful and demonstrates competence.

---

## 12. Datasets and benchmarks

### 12.1 English math (transfer-source)

- **GSM8K** — 8.5k grade-school problems.
- **MATH** — 12.5k competition problems.
- **AIME** — American Invitational Math Exam.
- **MMLU-STEM** subset.
- **GPQA** — graduate-level science.
- **PRM800K** — step-level labels.

### 12.2 Multilingual math benchmarks

- **MGSM** (Multilingual GSM8K) — 11 languages, ~250 problems each.
- **MATHQA-cn** (Chinese math).
- **GSM-Symbolic** — perturbed GSM8K, more robust eval.
- **Hindu Math** — emerging dataset for Hindi math reasoning.
- **Sinhala Math, Bengali Math** — emerging community-curated.

### 12.3 Multilingual reasoning broadly

- **Belebele** — reading comprehension in 122 languages.
- **MMLU-translated** — knowledge across many languages.
- **AfriBench** — for African languages.
- **BHASA** — Indic-language eval suite.
- **MEGA / MEGA-Verse** — broad multilingual NLP eval.
- **Aya Eval** — multi-domain multilingual.

### 12.4 Translation infrastructure

- **NLLB-200 (Meta)** — 200 languages; reasonable for low-resource.
- **MADLAD-400 (Google)** — 400 languages.
- **Aya 23** — 23 languages, instruction-following.

---

## 13. Open frontier questions

- **Can RLVR teach a base model new reasoning capabilities or only surface what's latent?**
- **What fraction of multilingual transfer is "thinking in English internally" vs genuine target-language reasoning?**
- **Is there a scaling law for cross-lingual reasoning transfer?**
- **What's the role of formal verifiers (Lean / Coq) for non-English mathematical traditions (Indian classical math, Chinese counting rod methods)?**
- **Can multi-agent debate in multiple languages improve target-language reasoning?**
- **Is there a "low-resource bonus" — does training on low-resource languages improve robustness on English (positive transfer)?**
- **How do you do RLVR for non-numeric reasoning in target languages?**
- **What's the impact of translating-reasoning-back vs target-language-from-the-start?**
- **Can speech-to-speech reasoning (no transcription) work in low-resource settings?**

---

## 14. References

### RLVR algorithms

- Shao et al. (DeepSeek), *DeepSeekMath / GRPO*, 2024.
- Yu et al. (ByteDance), *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2024.
- Liu et al., *Dr. GRPO: Understanding R1-Zero-Like Training*, 2025.
- Ahmadian et al. (Cohere), *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback*, 2024.
- Kool et al., *Buy 4 REINFORCE Samples, Get a Baseline for Free*, 2019.
- Cui et al., *PRIME: Process Reinforcement through IMplicit rEwards*, 2025.
- Zheng et al. (Qwen), *Group Sequence Policy Optimization*, 2025.
- Kazemnejad et al., *VinePPO*, 2024.

### Reasoning RL

- DeepSeek-AI, *DeepSeek-R1*, 2501.12948, 2025.
- Lambert et al., *Tülu 3*, 2024.
- Lightman et al. (OpenAI), *Let's Verify Step by Step* (PRM800K), 2023.
- Wang et al., *Math-Shepherd*, 2024.
- Luo et al., *OmegaPRM*, 2024.

### Reward modeling and hacking

- Gao et al., *Scaling Laws for Reward Model Overoptimization*, 2023.
- Singhal et al., *A Long Way to Go: Investigating Length Correlations in RLHF*, 2024.
- Park et al., *Disentangling Length from Quality in DPO*, 2024.
- Sharma et al., *Towards Understanding Sycophancy in LMs*, 2023.
- Mahan et al., *Generative Reward Models*, 2024.

### Multilingual / low-resource

- **Faisal, Song, Wang, Ma, Liu, Deng, Indurthi**, *Aligning Multilingual Reasoning with Verifiable Semantics from a High-Resource Expert Model* (PB-RLSVR), arXiv:2509.25543, Sep 2025. **The canonical RLVR-for-multilingual paper.**
- Aya Collective (Cohere), *Aya 23*, 2024.
- Üstün et al., *Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model*, 2024.
- Shi et al., *Language Models are Multilingual Chain-of-Thought Reasoners* (MGSM), 2023.
- NLLB Team (Meta), *No Language Left Behind*, 2022.
- Adelani et al., *AfriBench*, 2024.
- Kudugunta et al., *MADLAD-400*, 2024.
- Ahuja et al., *MEGA*, 2023.
- Bandarkar et al., *Belebele*, 2024.
- *Self-Improving Multilingual Long Reasoning via Translation-Reasoning Integrated Training* — companion translation-augmented approach.
- *Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models* — characterizes the degradation when multilingual models reason in non-English.
- Various IndicNLP / BHASA / SEACrowd consortium papers.

### Open-source infrastructure

- TRL (HuggingFace) — `huggingface/trl`.
- veRL (ByteDance) — `volcengine/verl`.
- Open-R1 (HuggingFace) — `huggingface/open-r1`.
- vLLM — `vllm-project/vllm`.
- OpenRLHF — `OpenRLHF/OpenRLHF`.

---

## How to use this chapter

1. Read §§1-7 once for the core RLVR understanding.
2. Memorize §2 (algorithm zoo) and §7 (failure modes) — interview gold.
3. Skim §8-9 to know the infra landscape.
4. **Spend serious time on §10 (low-resource multilingual)** — that's your differentiation.
5. Pick one of the project blueprints in §11 and write a 1-page proposal as practice.
6. Read the actual DeepSeek-R1 paper and the DAPO paper — non-negotiable.

The single sentence to remember for the multilingual angle: **RLVR's reward signal is language-agnostic for math; the bottleneck is target-language pretraining priors and CoT-language consistency, both addressable via a 4-stage recipe of distill → SFT → RLVR-with-language-consistency-reward → final-RLHF, and this has barely been done at scale outside English/Chinese — making it a wide-open research direction.**
