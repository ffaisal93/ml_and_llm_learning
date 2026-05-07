# Reasoning Models — Frontier Deep Dive

> Frontier-lab research-scientist interview-grade reference on reasoning-model training: o1 / R1 / R1-Zero, test-time compute scaling, RLVR, PRMs vs ORMs, search + RL combinations, and reasoning distillation.
> Pair with `08_training_techniques/ALIGNMENT_DEEP_DIVE.md` (DPO/PPO/GRPO foundations) and `FRONTIER_REWARD_MODELING.md` (this folder).

If you walk into an OpenAI / DeepMind / Anthropic research-scientist interview in 2025–2026, **reasoning RL** is the single highest-information-content topic. It's where the frontier moved most visibly between mid-2024 and 2025, with o1 (Sep 2024), o3 (Dec 2024), DeepSeek-R1 (Jan 2025), Qwen QwQ, Anthropic's Claude 3.7 Sonnet thinking modes, and Google's Gemini 2.5 thinking. This chapter walks the territory in interview-ready detail.

---

## Table of contents

1. The reasoning-model paradigm shift
2. Test-time compute scaling — what changed
3. RLVR (Reinforcement Learning with Verifiable Rewards) — the core engine
4. Process Reward Models (PRMs) vs Outcome Reward Models (ORMs)
5. Search + RL combinations — STaR, Quiet-STaR, V-STaR, ReST^EM, Expert Iteration, MCTS-based
6. R1-Zero — pure RL from base, the "aha moment"
7. R1 — full pipeline (cold-start SFT → reasoning RL → SFT → RLHF)
8. o1 / o3 — what we know and what we infer
9. Distillation of reasoning — long-CoT into smaller models
10. Inference-time strategies — best-of-N, self-consistency, MBR, sequential revision, lookback
11. Long-CoT failure modes — overthinking, hallucinated reasoning, language mixing
12. Generative reward models, LLM-as-judge as reward signal
13. Reward shaping for reasoning RL
14. Open questions and frontier directions
15. Senior-level interview signals
16. References

---

## 1. The reasoning-model paradigm shift

Through 2023, alignment was about *taste*: SFT to follow instructions, RLHF to make outputs preferred by humans. The model already had the capability; alignment was an overlay that surfaced it.

In late 2024, a second paradigm appeared. Models were *trained to think*. Specifically:

- The model emits a long internal chain-of-thought (often 1k–100k tokens) before answering.
- The CoT is *trained* by reinforcement learning, not just prompted.
- The reward signal is *verifiable* on tasks where it can be (math final answer, code unit tests).
- Compute scales at *inference time*, not just at training time — give the model more thinking tokens, get a better answer.

The empirical claim — "scale inference compute and quality keeps going up" — is the new scaling law (Snell et al. 2024, "Scaling LLM Test-Time Compute Optimally"). It's complementary to pretraining-compute scaling: a model with $X$ pretraining FLOPs can be matched by a smaller model with more inference FLOPs, on reasoning tasks.

Why it works (intuition):

- Reasoning is *search* in a soft solution space. Generating multiple steps, exploring variants, and selecting good ones is provably better than greedy decoding.
- Pretraining gives priors over reasoning steps; RL on verifiable tasks turns those priors into search policies.
- Long CoTs let the model build intermediate scaffolding that's hard to encode in a single forward pass — especially backtracking, self-correction, and verification.

The interview-relevant implication: **a frontier reasoning model is fundamentally an inference-time search policy distilled into a single autoregressive model**. RLVR teaches the policy. Test-time compute deploys it.

---

## 2. Test-time compute scaling — what changed

Snell et al. (DeepMind 2024) showed empirically that for a fixed model, three "test-time" strategies all yield smooth scaling curves:

- **Best-of-N + reward model.** Sample $N$ completions, pick the one with highest reward-model score. Quality $\propto \log N$ for moderate $N$, then plateaus.
- **Sequential revision.** The model reads its own first attempt, revises, repeats. Quality scales with number of revisions, with steeper slope than best-of-N for hard tasks.
- **Search via verifier.** Use a PRM at each reasoning step; expand only promising branches; back up. Beats both above on hard math.

They also showed the **compute-optimal frontier**: for a fixed total inference budget, *easier problems* benefit more from sequential revision (deeper but narrower); *harder problems* benefit more from search/best-of-N (broader exploration). And — most importantly for interviews — **a smaller model with $\sim 14\times$ more inference compute can match a 14× larger model on hard reasoning tasks**.

The OpenAI o1 system card and the DeepSeek-R1 paper both report cleanly log-linear scaling between (CoT-token-budget) and (benchmark accuracy) on AIME / MATH / GPQA / Codeforces.

Why this matters for interviews:

- **Scaling is now multi-dimensional.** Capability ∝ pretraining-compute + post-training-compute + inference-compute. You should reason about all three.
- **The product of all three is what matters.** Allocating compute optimally across them is an open research question (Sardana et al. on optimal allocation, etc.).
- **It changes the deployment economics.** A reasoning model is much slower and more expensive per query. Which raises product-side questions about *when to invoke reasoning*.

---

## 3. RLVR — Reinforcement Learning with Verifiable Rewards

The engine of reasoning models. Coined widely in 2024 (Lambert, Tülu 3 team, etc.).

### 3.1 The verifiable reward

A *verifiable* reward is one where, given a problem and a final answer, you can deterministically (or near-deterministically) decide correctness:

- **Math (formal).** Parse the boxed answer; compare to ground truth. Or use sympy to canonicalize.
- **Math (semi-formal, IMO-style proofs).** Lean / Coq verifier on a generated proof.
- **Code.** Run the generated code against unit tests; pass/fail.
- **Multiple-choice / extraction.** String match against ground truth.
- **Tool-use traces.** Did the trace produce the correct final state in a sandboxed env (Bash, browser, calculator)?
- **Logic / formal reasoning.** Symbolic checker.

What's *not* verifiable: writing quality, summary fidelity, chat persona. Those need preference signals (RLHF), generative reward models, or LLM-as-judge.

### 3.2 Why verifiable rewards are special

- **Zero ambiguity.** No labeler bias, no labeler drift, no preference-data noise.
- **Cheap.** A test runner or sympy is much cheaper than human labels.
- **Cannot be hacked the way preference RMs are.** No length bias, no sycophancy, no formatting bias. The model gets reward only if the answer is right.
- **Scales.** Generate millions of math problems with verified ground truth.

Tradeoff: only a slice of useful capabilities is verifiable. Most chat tasks are not.

### 3.3 The RLVR objective

In its simplest form (used by R1-Zero and the Tülu 3 RLVR phase):

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\big[\, r(x, y) \,\big] \;-\; \beta\, \mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big),
$$

where $r(x, y) \in \{0, 1\}$ is the verifier output (correct / incorrect) — or a small graded set (correct / partial / wrong / format-error). $\pi_{\text{ref}}$ is typically the SFT model.

The optimizer is usually GRPO, RLOO, or REINFORCE++ (no critic — see `ALIGNMENT_DEEP_DIVE.md`). PPO with a value head also works but adds memory.

### 3.4 Format rewards and reward shaping

Pure correctness reward is sparse. Most implementations add:

- **Format reward.** "Did the answer appear in `\boxed{}` / `<answer>` tags?" → small bonus / penalty for wrong format.
- **Reasoning length reward.** Sometimes a small bonus for having a non-empty `<think>` block; sometimes a penalty for excessive length.
- **Language consistency reward.** Penalize CoT that mixes English+Chinese (R1's specific issue).
- **Step-level PRM reward.** Use a PRM (§4) to reward step-correctness in addition to final correctness.

A key R1 finding: with *just* correctness + format rewards, the model spontaneously learns long CoT, backtracking, and verification — no PRM needed. We'll come back to this.

### 3.5 The KL penalty and the reference

$\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ stops the policy from drifting into incoherent reasoning. Two variants:

- **Token-level KL.** Per-token KL term added to the reward.
- **Sequence-level KL.** Estimated from log-prob ratio, used in the PPO-style importance ratio (this is more standard).

Tradeoff: too high $\beta$ = no exploration, no improvement; too low $\beta$ = mode collapse, language drift, hallucination spirals. R1 uses adaptive $\beta$ scheduled with KL targets.

### 3.6 Common failure modes

- **Reward hacking format.** Model learns to emit `\boxed{42}` regardless of the question. Mitigation: format-only reward must be small relative to correctness.
- **Length explosion.** CoT grows to fill the context window with no quality gain. Mitigation: length penalty above a threshold; truncate context.
- **Language mixing.** Mid-CoT switch from English to Chinese (R1-Zero's headline). Mitigation: language-consistency reward.
- **Mode collapse.** Single reasoning template wins; diversity dies. Mitigation: entropy bonus, KL cap, diversity-aware sampling.
- **Reward sparsity for hard problems.** If success rate is <1%, gradients are essentially noise. Mitigation: curriculum (easier problems first), expert iteration, or reduce problem difficulty.

---

## 4. Process Reward Models (PRMs) vs Outcome Reward Models (ORMs)

Reward models in reasoning RL come in two flavors. Knowing the distinction cold is interview-table-stakes.

### 4.1 ORM — Outcome Reward Model

Scores the *final answer*. Trained on (problem, answer, correct/incorrect) triples or pairwise preferences over final answers.

- Easy to train (lots of data via verification).
- Generic: works on anything verifiable.
- **Sparse signal.** All steps before the answer get the same reward.

### 4.2 PRM — Process Reward Model

Scores *each reasoning step*. Trained on (problem, step-1, ..., step-k, label) tuples where each step has a label like "correct so far", "step is wrong", or "leads to wrong answer".

- Dense signal: rewards per step.
- Lightman et al. 2023 ("Let's Verify Step by Step", OpenAI's PRM800K dataset) — the canonical paper. Showed PRMs significantly outperform ORMs on MATH at fixed inference budget when used for re-ranking.
- Hard to train at scale: per-step labels are expensive. Two solutions to scale:
  - **Math-Shepherd (Wang et al. 2024).** Auto-label step correctness by Monte-Carlo rollouts: from each step, sample $K$ continuations; the step is "good" if a high fraction reach a correct answer.
  - **OmegaPRM (Luo et al. 2024).** MCTS-based PRM data construction. State-of-the-art for automatic PRM data.

### 4.3 PRM as RL reward

Two ways to use a PRM in RL training:

- **As re-ranker (search-based).** Generate multiple paths; PRM scores each; train policy on PRM-best paths via SFT or DPO.
- **As dense reward.** PRM score at each step is added as a per-step reward in the RL update.

DeepSeek-R1 explicitly tried PRMs and reported they **didn't help over a strong ORM** because of (1) PRM data noise at scale, (2) reward hacking against the PRM, and (3) extra complexity. Other groups (Tülu 3, Qwen) report mixed. **Open question for 2025: is PRM training necessary for frontier reasoning, or does ORM with good data suffice?**

### 4.4 Generative reward models (genRMs)

A different way to get rewards: instead of a discriminative scalar head, prompt a strong LLM with the problem, the candidate solution, and ask "is this correct? explain your reasoning, then output yes/no/partial." The judge's verdict (parsed) is the reward.

- Mahan et al. 2024, Zhang et al. 2024, "Generative Reward Modeling" — genRM models match or beat scalar RMs on hard reasoning, especially out-of-distribution.
- Lets the RM itself reason via CoT before deciding. The same test-time-compute lever applies to the RM.
- Used as part of OpenAI's deliberative-alignment pipeline (2024).

genRMs are central to the 2025 frontier and any interview question of the form "how would you build a reward model that doesn't get hacked" should mention them.

---

## 5. Search + RL combinations

The classical insight: **expert iteration** (Anthony et al. 2017, AlphaZero-style) — alternate (a) generate from policy with search, (b) supervise-distill the search-improved trajectories back into policy. Many reasoning-LM training methods are variants.

### 5.1 STaR (Self-Taught Reasoner, Zelikman et al. 2022)

Bootstrap reasoning:
1. Prompt model to generate CoT for a problem.
2. If final answer is correct, keep the CoT.
3. If wrong, give model the answer and ask it to generate a *rationalization* CoT.
4. SFT on collected (problem, CoT, answer) data.
5. Iterate.

**Key insight:** even rationalization improves reasoning, because the rationalized CoT teaches the model how a correct chain looks for that class of problem.

### 5.2 Quiet-STaR (Zelikman et al. 2024)

Train the model to predict *internal* "thoughts" between every token. The thoughts are not output. The objective is to maximize next-token prediction conditioned on a generated thought. The thought head is trained jointly. Result: small but consistent improvements in zero-shot reasoning. This is one of the more elegant training-time-compute → inference-time-compute distillation ideas.

### 5.3 V-STaR (Hosseini et al. 2024)

STaR + verifier model trained on STaR's negative samples. The verifier acts as a re-ranker and as a PRM-like signal. Gets significant improvement over STaR on math.

### 5.4 ReST^EM (Singh et al. 2024)

Reinforced Self-Training Expectation-Maximization:
1. **E-step:** sample $K$ CoTs per problem from current policy.
2. Filter to correct ones via verifier.
3. **M-step:** SFT on the filtered correct CoTs (only).
4. Repeat.

Cleaner than STaR; equivalent to a form of expert iteration. DeepMind used it for Gemini 1.5 reasoning improvements.

### 5.5 MCTS-based methods

- **AlphaZero-for-LLMs.** Train a value head + policy head; expand a tree at inference; MCTS. Used by Tian et al. (Toward Self-Improvement of LLMs via Imagination), Trinh et al. (AlphaProof, AlphaGeometry), Luo et al. (OmegaPRM), and rumored to be part of o1 internals.
- **Tradeoff.** MCTS is expensive and tricky to combine with discrete-token state spaces. AlphaProof (DeepMind 2024) — silver-medal IMO performance — uses Lean as the verifier, which makes the value function tractable. AlphaGeometry uses symbolic engines.

For a research-scientist interview, you should be able to sketch how MCTS is adapted to the LM token space (state = prefix, action = next-token-or-step, value = expected verifier reward, expansion via policy logprob, backup as in standard MCTS).

### 5.6 Best-of-N + verifier training (rejection sampling fine-tuning)

The simplest expert-iteration-flavored method:

1. Sample $N$ outputs per problem.
2. Verifier filters correct ones.
3. SFT on the correct outputs.
4. Optionally: train a verifier on (correct, incorrect) pairs.

Used as the SFT-data source by Tülu 3, Qwen 2.5 Math, Llama 3 (rejection-sampled tool-use data). Essentially free signal.

---

## 6. R1-Zero — pure RL from base, the "aha moment"

DeepSeek's R1-Zero (Jan 2025) is the empirical headline of the era and **the question to be ready for in any 2025 interview.**

### 6.1 The setup

- **Starting point:** DeepSeek-V3 base (no SFT, no instruction tuning).
- **Training data:** math, code, logic problems with verifiable answers.
- **Reward:** correctness (verifier) + format reward only. No PRM, no preference data, no RM.
- **Algorithm:** GRPO (sampling-only group baseline; no value head).
- **Prompt template:** `<think> ... </think><answer> ... </answer>`.

### 6.2 The "aha moment"

Reading the DeepSeek-R1 paper (arXiv 2501.12948), §2.2.4 famously documents the model spontaneously generating phrases like "Let me re-check my work" and "Wait, I made an error" mid-CoT — *without being trained on any data containing such phrases*. This is the emergence of *self-correction* purely from outcome reward.

The reasoning length grows from ~100 tokens to >2000 over training. Accuracy on AIME climbs from ~16% to ~71% pass@1.

### 6.3 What R1-Zero proves

- **You can elicit deep reasoning from a base model with pure RL.** No SFT cold-start needed.
- **Outcome reward is enough** — no PRM required to discover step-level reasoning.
- **The capability was latent in the base model.** RL surfaced it; pretraining created it.
- **Long CoT is a learned policy, not a hand-crafted prompt.** It's a length the model converged to because longer CoTs work better on hard math.

### 6.4 R1-Zero's failure modes

The paper is honest:
- Reasoning is illegible — switches between English/Chinese, uses non-standard formatting.
- Doesn't transfer well to non-math/code tasks.
- Refuses some queries unpredictably; less helpful as a chatbot.

This motivates the full R1 pipeline (next).

### 6.5 Why this matters for interviews

R1-Zero is a clean experimental result that an interviewer can probe in many directions:
- "Why didn't this work in 2022?" → base model wasn't strong enough; reasoning RL needs base reasoning priors.
- "What's the role of GRPO vs PPO here?" → no value head needed for verifiable rewards; sample efficiency from group baseline.
- "Could you replace verifiable reward with preference data?" → much harder; outcome verification is the privileged signal.
- "What's the failure mode of training only on math?" → distribution shift; reasoning skill doesn't transfer to chat without further alignment.

---

## 7. R1 — the full pipeline

DeepSeek-R1 (Jan 2025) is the first publicly documented frontier-reasoning recipe. Memorize it.

### 7.1 The four stages

1. **Cold-start SFT.** Curate a small (few-thousand-example) high-quality long-CoT dataset (mixture of human-written, generated by R1-Zero with rejection sampling, refined). SFT V3-base on this.
2. **Reasoning-oriented RL.** Same RLVR setup as R1-Zero but starting from the cold-start SFT model. Adds language-consistency reward to fix the language-mixing failure mode. Trains until reasoning saturates on math/code/logic.
3. **Rejection sampling SFT.** Generate millions of CoTs from stage-2 model, filter by verifier (math, code) and by RM/judge (other tasks); resulting SFT data is ~600k math+reasoning + ~200k other (writing, factual QA, role-play). SFT V3-base on the merged dataset.
4. **Final RLHF.** Standard RLHF on stage-3 SFT for helpfulness, harmlessness, alignment with human preferences, while keeping the reasoning capability.

The result: R1-zero capabilities + chat-friendly + safe + multilingual.

### 7.2 Why four stages instead of one

- **Pure RLVR (R1-Zero) is illegible.** Stage 1 cold-start gives it format and legibility.
- **Pure cold-start SFT doesn't learn deep reasoning.** Stage 2 RLVR pushes the capability beyond what's in the SFT data.
- **Stage 2 model is good at math but bad at chat.** Stage 3 SFT broadens distribution.
- **Stage 3 SFT loses some safety calibration.** Stage 4 RLHF restores helpful/harmless behavior.

### 7.3 Tülu 3 (AllenAI 2024)

The full open-source reasoning-and-instruction recipe before R1. Worth memorizing as a comparison point:

1. **SFT** on a curated mix (Tülu 3 SFT dataset).
2. **DPO** on a curated preference set.
3. **RLVR** on math + multi-turn instruction-following with code-graded rewards.

Tülu 3 RLVR was lighter than R1's stage 2 (less reasoning-heavy) but fully open-source — the recipe and data are public. Read the paper if you want a code-able recipe.

### 7.4 Llama 3 post-training

Meta's Llama 3 paper (2024) details a multi-round SFT + DPO recipe with:
- **Per-capability mixes** (code, math, reasoning, multilingual).
- **Rejection-sampled SFT data** at large scale.
- **DPO with carefully constructed pairs** rather than PPO.
- **Tool-use RL** with Python/calculator/search/scoring.
- **Adversarial safety RL** with red-team-generated prompts.

No reasoning-RL stage *yet* in Llama 3.1 (8B/70B/405B). Llama 4 reasoning model expected to add it.

### 7.5 Qwen 2.5 / QwQ / Qwen3 reasoning

Qwen team's published recipe is similar to R1's: cold-start SFT → RLVR with format rewards → preference-aligned RLHF. QwQ-32B (Nov 2024) and Qwen3-thinking models follow this pattern. They report PRM helped *in some experiments* but the canonical recipe uses ORM for stability.

### 7.6 The "interview cookbook" version

If asked "give me a recipe for a reasoning model from scratch," answer:

1. **Pretrain** a base model with strong math/code priors (large fraction of code+math+logic in pretraining mix).
2. **Cold-start SFT** on a few thousand high-quality long-CoT examples (human or filtered generated).
3. **RLVR** with verifiable rewards (math final-answer, code unit tests). Use GRPO/RLOO; format and language-consistency rewards; KL to SFT.
4. **Rejection-sampled SFT** to broaden distribution to chat / writing / tool-use, filtering by verifier where possible and by judge LLM otherwise.
5. **Final RLHF** for helpfulness, harmlessness, persona.

That's the canonical 5-step recipe. Customizations are mostly empirical.

---

## 8. o1 / o3 — what we know and what we infer

OpenAI hasn't published the recipe. From the system card, blog posts, and open-source models' reverse-engineering, the consensus inference:

- **Pure(-ish) RL** with a reasoning-oriented reward model.
- **Long CoT** trained, hidden from the user (output is a summary).
- **Test-time compute scaling** — accuracy curves shown to scale log-linearly with thinking-token budget on AIME, GPQA, Codeforces, ARC-AGI.
- **Search / verifier integration** at training (and possibly inference). Speculation: PRM-guided rollouts for data, possibly MCTS at training data generation, possibly best-of-N at inference.
- **Multi-stage training** likely similar in spirit to R1's pipeline.
- **Process supervision data** explicitly mentioned by OpenAI ("Let's Verify Step by Step" was OpenAI work).

What we *don't* know:
- Whether MCTS is used at inference (likely no, reasoning is single-stream autoregressive).
- The exact mix of PRM vs ORM signals.
- Whether deliberative alignment (next subsection) is layered.

For interviews, a safe answer: "the public details are limited, but it's consistent with: (1) base + cold-start SFT, (2) heavy RLVR with PRMs and ORMs, (3) test-time compute optimization, (4) safety training via deliberative alignment."

### 8.1 Deliberative Alignment (OpenAI 2024)

OpenAI's approach to safety in reasoning models. Train the model to *deliberate over the spec* — i.e., at thinking time, the model reasons about whether the request violates the model spec, and bases its action on that reasoning. Synthetic data: many (prompt, deliberation, action) triples generated from the spec, used in SFT and RL.

Different from RLHF on safety: the deliberation is *trainable*, not just a refusal. Crucial reading: "Deliberative Alignment" (Guan et al., OpenAI 2024).

### 8.2 Anthropic's reasoning approach

Less public detail. Claude 3.7 Sonnet's "extended thinking" mode emerged in early 2025. Likely similar architectural idea (long CoT trained via RL on verifiable + judge-graded tasks) plus Constitutional / RLAIF-flavored preference data. Anthropic's published Constitutional AI paper (2022) and RLAIF papers are the relevant priors.

### 8.3 Gemini 2.5 thinking

Google DeepMind. Public details indicate test-time compute scaling, MCTS-flavored search at training data generation (consistent with their AlphaProof / AlphaGeometry lineage), and a similar multi-stage RL recipe.

---

## 9. Distillation of reasoning

Distillation is essential for productizing reasoning. Big reasoning model is too slow/expensive; you want a smaller model that's *almost as good*.

### 9.1 Long-CoT distillation

DeepSeek released **R1-Distill** models (Llama, Qwen bases of various sizes). Recipe:

1. Generate long CoTs from R1 on a curated problem set (~800k problems).
2. SFT a smaller base model on (problem, R1 CoT, answer).
3. Optionally a follow-up DPO or RL stage.

**Headline result.** R1-Distill-Qwen-32B beats GPT-4o-2024-05 on AIME and MATH despite being far smaller. **Distillation transfers reasoning.** This is the most important "free lunch" of the era.

### 9.2 Why distillation works so well

- The teacher's CoT is a *demonstration* of the search policy. The student doesn't need to discover it via RL; it just imitates.
- The student inherits long-CoT habits, including self-correction and verification, from imitation alone.
- Tradeoff: the student is bounded by the teacher; can't exceed it. RL on top can break the bound.

### 9.3 Open-source reasoning distillates

- DeepSeek R1-Distill family (Jan 2025).
- OpenThinker, Bespoke-Stratos, Sky-T1 — community efforts using R1 generations.
- Hugging Face Open-R1 reproduction (Jan 2025+).

### 9.4 Implications

- **Reasoning capability democratizes fast.** Once one frontier model demonstrates the capability, distillation sweeps it across model families and sizes.
- **The moat is the verifier + RL infra**, not the final model.
- **Privacy of reasoning becomes a question** — should reasoning be hidden from users to prevent distillation? OpenAI hides o1 CoTs from users for several reasons including this.

---

## 10. Inference-time strategies

A reasoning model is a policy; how you decode matters.

### 10.1 Greedy / sampling temperature

For reasoning, **temperature=0 (greedy)** can lock the model into a single mode and miss the right answer; **temperature 0.6–1.0** enables exploration and pairs with best-of-N or self-consistency. R1 default is T=0.6.

### 10.2 Self-consistency (Wang et al. 2022)

Sample $N$ CoTs at high temperature; majority-vote the answers. Strictly better than greedy on most reasoning benchmarks. The simplest "test-time compute" lever.

### 10.3 Best-of-N + reward model

Sample $N$ CoTs; an ORM/genRM scores each; pick max. Strictly better than self-consistency when the RM is good.

### 10.4 Sequential revision

Generate one answer; feed it back with "is this correct? if not, try again"; iterate. Effective on hard problems where exploration matters. Used by Snell et al. for the test-time scaling curves.

### 10.5 MBR (Minimum Bayes Risk) decoding

Score each candidate by its average similarity to the others, weighted by their reward. Picks the *consensus + reward* candidate. More robust than max-reward when the RM is noisy.

### 10.6 Lookback / verifier-guided beam search

Run beam search where each beam expansion is rated by a PRM; prune low-PRM beams; back up. The PRM-guided variant of MCTS.

### 10.7 Compute-optimal inference

Snell et al. show: for a fixed inference budget, allocate more samples to harder problems (judged by self-confidence or verifier disagreement). At difficulty $D$, optimal $N_{\text{samples}} \approx K \cdot D^\alpha$ for empirical exponent $\alpha$.

### 10.8 What the user actually pays

A reasoning model run with best-of-32 and self-revision can be 50-200x more expensive per query than a single greedy decode. This is the new product-economics challenge: **when is reasoning worth the cost?** Routing layers (cheap LM detects "is this a reasoning task" → invoke reasoning model only if yes) are a 2025 product pattern.

---

## 11. Long-CoT failure modes

What goes wrong with reasoning models that an interviewer might probe.

### 11.1 Overthinking

Easy problem ("what is 2+2") → 5000-token CoT. Wastes compute and sometimes degrades correctness via accumulated noise. Mitigation:
- Length penalty above threshold.
- Adaptive thinking budget based on self-estimated difficulty.
- A "fast/slow" two-mode router (Claude 3.7 Sonnet does this).

### 11.2 Hallucinated reasoning

Model produces a long, confident, internally-consistent CoT that's wrong from step 1. Verifier saves the day for math/code; for non-verifiable tasks (e.g. legal reasoning), this is a serious risk.

### 11.3 Confident-but-wrong calibration

Reasoning models are generally *better* calibrated on math (you can see uncertainty in the CoT — "I think this might be..."). They are *worse* calibrated on factual QA where they extrapolate confident-sounding chains from incorrect priors.

### 11.4 Language mixing

R1-Zero's mid-CoT English → Chinese drift. Caused by the base model's bilingual pretraining and RL not penalizing it. Fixed in R1 with a language-consistency reward.

### 11.5 Reward gaming

Length (without correctness) gets rewarded if reward shaping is misspecified. Format-only outputs (`\boxed{}` with garbage). Verifier exploits — finding answer formats the verifier accepts but is wrong (e.g. floating-point precision tricks).

### 11.6 Reasoning collapse on out-of-distribution problems

A model trained on AIME-style math can fail dramatically on physics word problems with similar math content. The CoT *style* doesn't transfer; the model produces formula-soup.

### 11.7 Costly errors

A reasoning model that gets math wrong after 30000 thinking tokens is a unique failure mode. Confidence-based early termination is now common.

---

## 12. Generative reward models, LLM-as-judge as reward signal

For non-verifiable tasks (writing, multi-turn chat, complex tool use), the reward signal must come from a model. Two approaches:

### 12.1 Scalar RM

Train a regression head to predict a Bradley-Terry score. Issues: sycophancy, length bias, noise on long outputs.

### 12.2 Generative RM (genRM)

Prompt a strong LLM with the input + output and a rubric. The LLM emits CoT and a verdict. Parse the verdict as the reward. Variants:

- **Yes/no genRM.** Binary — is the output correct?
- **Rubric-based.** Score each criterion 1-5; sum.
- **Pairwise genRM.** Two outputs; which is better?
- **Reasoning genRM.** The judge model itself runs CoT before the verdict — and can be RL-trained for accuracy.

genRMs are now standard in OpenAI / Anthropic / Google reward pipelines. Mahan et al. 2024, Zhang et al. 2024, Lambert (Tülu 3), and Anthropic's RLAIF lineage all use them.

### 12.3 Constitutional AI / RLAIF

Anthropic's recipe for preference data without humans:
- Define a **constitution** — a list of principles ("don't be harmful", "be helpful", etc.).
- Generate model outputs.
- Use the model itself (or a separate model) to revise outputs against each principle.
- Train the RM on (revised better, original worse) preferences.

RLAIF (Bai et al. 2022) is the variant where AI labels preferences directly. Zheng et al. 2023 ("RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback") showed RLAIF matches RLHF in many domains.

### 12.4 Critic-LM

A specialized LLM trained to *critique* outputs. Used as a reward signal or as a search heuristic. McAleese et al. 2024 "LLM Critics Help Catch LLM Bugs" — code-task PRM-flavored critic.

### 12.5 Open question

Are genRMs the future or a stopgap? The tension: genRMs are themselves models with their own biases, and can be hacked. But they're cheaper and richer than human labels. For frontier 2025-2026, the answer is "both": humans for ground truth on a small set, genRMs to scale the signal.

---

## 13. Reward shaping for reasoning RL

Practical zoo, often asked in interviews.

- **Correctness.** Verifier output. The dominant signal.
- **Format.** Tags `<think>...</think><answer>...</answer>` present and well-formed.
- **Length.** Soft penalty above/below thresholds.
- **Language consistency.** Penalize mid-CoT language switches.
- **Step-level (PRM).** Per-step correctness signal.
- **Diversity bonus.** Reward exploration in the rollout group (reduces mode collapse in GRPO).
- **Repetition penalty.** Reward for not repeating phrases.
- **Self-consistency reward.** Award the answer that matches the majority across $N$ rollouts.
- **Generative judge reward.** For non-verifiable tasks.

### 13.1 The reward composition question

How do you combine? Two patterns:

1. **Linear combination.** $r = w_1 r_{\text{correct}} + w_2 r_{\text{format}} + w_3 r_{\text{length}} + ...$ with hand-tuned weights.
2. **Hierarchical.** Format gates correctness; correctness dominates everything; soft shaping terms only after threshold met.

Hierarchical is more stable; linear is easier to tune.

### 13.2 Why reward shaping is hard

Goodhart's law applies brutally. Anything you reward, the model will hack. Anything you don't reward, the model may regress on. Mitigations:
- Keep shaping terms small relative to the dominant signal.
- Periodic eval on held-out sanity tests.
- Manual sample inspection at every stage.
- Reward model retraining if exploitation is detected.

---

## 14. Open questions and frontier directions

These are interview gold — show you're at the frontier.

### 14.1 Can pure RL produce capabilities the base model doesn't have?

R1-Zero suggests RL *elicits* latent capability, not creates it. Whether RL can transcend the base model's reasoning capacity is open. Some evidence (R1-Zero exceeding GPT-4o on AIME) suggests yes, at least on narrow tasks.

### 14.2 Is the inference-compute scaling law universal?

Snell-style curves are clean on math/code/reasoning. Less clean on open-ended generation. Whether arbitrary tasks benefit from test-time compute is open.

### 14.3 Do PRMs help once ORM is good?

R1 paper says no. Tülu 3 inconclusive. Open.

### 14.4 Multi-agent / debate-based reward

Models debating each other for the reward signal. Irving et al. (2018), Khan et al. 2024 ("Debate Helps Supervise Unreliable Experts"). Frontier scalability open.

### 14.5 Self-play for reasoning

SPIN, Self-Rewarding LMs (Yuan et al. 2024) — model labels its own preference data, iterates. Works in narrow regimes; generalization open.

### 14.6 Continual / online RL

Production reasoning models update from real user interactions. The RL infra to do this safely is a frontier engineering problem (drift, reward hacking in production, safety regression).

### 14.7 RL on long-horizon agentic tasks

Reasoning RL on math is finite-horizon. RL on agent trajectories (multi-step browsing, multi-step coding) is much longer. Credit assignment becomes hard. Recent: TauBench-RL, OS-World-RL, SWE-Gym, AgentDojo-style training.

### 14.8 Safety of reasoning models

Reasoning models can produce long, plausible CoTs that justify dangerous outputs. Safety training must operate over the CoT, not just the answer. Deliberative alignment is one approach; constitutional reasoning is another. Open.

### 14.9 Spec compliance in reasoning

The model spec is a natural-language constitution. Frontier labs are converging on "reason about the spec at thinking time" as the safety pattern. This is what deliberative alignment does. Generalizing it to all behavioral guarantees is open.

### 14.10 Reasoning + tools

The frontier 2025-2026 reasoning model is likely a *reasoning-and-tool-using agent* — model thinks, calls a calculator / code interpreter / web tool mid-CoT, integrates tool output, continues. Training this end-to-end via RL is an open systems problem.

---

## 15. Senior-level interview signals

What separates a "knows the buzzwords" answer from a research-scientist answer.

- **You know R1's four-stage pipeline cold and can sketch it.**
- **You can articulate why R1-Zero is interesting** — pure RL surfaced latent capability, no PRM needed, the "aha moment."
- **You distinguish PRM from ORM**, with the Lightman et al. PRM800K context.
- **You know GRPO/RLOO are critic-free for sample-efficiency** in verifiable-reward settings.
- **You can sketch test-time compute scaling laws** and the compute-optimal frontier (Snell et al.).
- **You know genRMs are now central** to non-verifiable reward.
- **You can list reasoning failure modes** (overthinking, hallucinated reasoning, language mixing, reward gaming).
- **You know R1-Distill exists** and what it implies about distillation transferring reasoning.
- **You can name OpenAI's deliberative alignment** as the safety-in-reasoning pattern.
- **You don't oversell**: you separate what's published (R1, Tülu 3, Llama 3) from what's inferred (o1 internals).
- **You have an opinion** on the open questions in §14.
- **You think about cost** — reasoning is expensive; product routing is now necessary.

---

## 16. References

### Reasoning RL canonical

- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, arXiv 2501.12948, Jan 2025. **The single most important paper for 2025 interviews.**
- OpenAI, *o1 System Card*, Sep 2024. The closed but widely-discussed reference.
- Snell et al., *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters*, DeepMind 2024.
- Lambert et al. (AllenAI), *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*, Nov 2024.
- Lightman et al. (OpenAI), *Let's Verify Step by Step* (PRM800K), 2023.
- Wang et al., *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*, 2024.
- Luo et al., *Improve Mathematical Reasoning in Language Models by Automated Process Supervision* (OmegaPRM), 2024.

### Search + RL

- Zelikman et al., *STaR: Self-Taught Reasoner*, 2022.
- Zelikman et al., *Quiet-STaR*, 2024.
- Hosseini et al., *V-STaR*, 2024.
- Singh et al., *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models* (ReST^EM), 2024.
- AlphaProof / AlphaGeometry — DeepMind 2024.
- Tian et al., *Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing*, 2024.

### Reward modeling

- Bai et al. (Anthropic), *Constitutional AI: Harmlessness from AI Feedback*, 2022.
- Lee et al., *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*, 2023.
- Mahan et al., *Generative Reward Models*, 2024.
- Zhang et al., *Generative Verifiers: Reward Modeling as Next-Token Prediction*, 2024.
- McAleese et al. (OpenAI), *LLM Critics Help Catch LLM Bugs*, 2024.

### Self-improvement / self-rewarding

- Yuan et al. (Meta), *Self-Rewarding Language Models*, 2024.
- Chen et al., *SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models*, 2024.

### Safety / spec compliance

- Guan et al. (OpenAI), *Deliberative Alignment*, 2024.
- Khan et al., *Debate Helps Supervise Unreliable Experts*, 2024.
- Irving et al., *AI safety via debate*, 2018.

### Algorithms (review)

- Schulman et al., *PPO*, 2017.
- Shao et al. (DeepSeek), *DeepSeekMath / GRPO*, 2024.
- Kool et al., *RLOO*, 2019.
- Ahmadian et al., *Back to basics: Revisiting REINFORCE Style Optimization for RLHF*, 2024.

### Surveys / blogs

- Lambert blog (Interconnects) — running commentary on RLVR, R1, Tülu 3.
- Ouyang & Schulman et al., *Training language models to follow instructions with human feedback* (the InstructGPT paper, 2022) — RLHF foundations.
- "*Reverse-engineering o1*" community posts, 2024-2025.

---

## How to use this chapter

1. Read straight through once — this is the densest single document on 2025 reasoning RL.
2. Memorize the R1 4-stage pipeline (§7.1) — you will be asked.
3. Be able to sketch test-time compute scaling (§2) on a whiteboard with axes labeled.
4. Drill the senior signals (§15) and the open questions (§14).
5. Pair with `FRONTIER_REWARD_MODELING.md`, `OPEN_SOURCE_POSTTRAIN_PLAYBOOKS.md`, `INTERVIEW_GRILL.md` in this folder.
6. Spend 3 hours reading the actual DeepSeek-R1 paper if you haven't — it's the key.

The single sentence to remember: **a 2025 frontier reasoning model is a long-CoT autoregressive policy trained by RL on verifiable + judge rewards, distilled from a larger search procedure, and deployed with inference-time compute as a third scaling axis alongside pretraining and post-training compute.**
