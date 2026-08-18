# Open-Source Post-Training Playbooks — Frontier Recipes

> Memorizable recipes from the actual public reports of DeepSeek R1, AllenAI Tülu 3, Meta Llama 3, Alibaba Qwen 2.5, and HuggingFace Open-R1. For frontier-lab interviews, you should be able to walk an interviewer through any of these end-to-end.

This chapter is recipe-centric: each section is a stage-by-stage walkthrough of one published frontier recipe. If asked "design a post-training pipeline for a 70B reasoning model," you sketch one of these and adapt.

---

## Table of contents

1. DeepSeek-R1 (Jan 2025) — full reasoning recipe
2. Tülu 3 (AllenAI, Nov 2024) — open SOTA general-purpose post-training
3. Llama 3 (Meta, Jul 2024) — large-scale post-training
4. Qwen 2.5 / Qwen3 — Alibaba's recipe
5. Open-R1 (HuggingFace community, Jan 2025+) — reproduction notes
6. The frontier "interview cookbook" — synthesized recipe

---

## 1. DeepSeek-R1

The single most-discussed post-training paper of 2025. Memorize this.

### 1.1 Starting point

- **DeepSeek-V3 base** — 671B MoE, 37B activated; trained on 14.8T tokens; strong math/code priors.
- No SFT, no instruction tuning at start.

### 1.2 Two parallel tracks

The R1 paper actually presents *two* models:

#### Track A: R1-Zero

**Pure RL from base.** Goal: see how far reasoning can go with no SFT.

- Algorithm: GRPO (group-relative; no value head).
- Reward: correctness (verifier on math/code/logic) + format reward only.
- Prompt template: `<think>...</think><answer>...</answer>`.
- KL: standard sequence-level penalty to base.
- Group size: 16 rollouts per prompt for advantage normalization.

Result: AIME 2024 pass@1 climbs from 15.6% to 71.0% across training; CoT length grows from <100 to >2000 tokens. The "aha moment" — model spontaneously self-corrects ("Let me re-check") without ever seeing such phrases in training data.

Issues with R1-Zero:
- Mid-CoT language mixing (English ↔ Chinese).
- Illegible / non-standard formatting.
- Doesn't generalize beyond reasoning tasks; bad as a chatbot.

#### Track B: R1 (production model)

**Multi-stage pipeline:**

##### Stage 1: Cold-start SFT

- Curate a small (~few thousand) high-quality long-CoT dataset.
  - Sources: human-written, R1-Zero rejection-sampled, manual cleanup.
- SFT V3-base on this data.
- Output: legible long-CoT model, weak in pure capability but with good format.

##### Stage 2: Reasoning-oriented RL

- Same RLVR setup as R1-Zero: GRPO, correctness + format rewards.
- **Plus** language-consistency reward: penalize CoT that switches between English/Chinese.
- Train until convergence on math/code/logic benchmarks.
- Output: strong reasoning capability + legible CoTs.

##### Stage 3: Rejection-sampling SFT (data construction)

- Generate 600k+ samples on math/code/reasoning from Stage-2 model.
  - Filter by verifier (correct only) and judge (helpful, well-formatted).
- Generate 200k+ samples on writing, factual QA, role-play.
  - Filter by judge (helpfulness, no-harm).
- Total: ~800k SFT examples.

##### Stage 4: SFT + Final RLHF

- SFT V3-base again on the full 800k mixed dataset (overwrite Stage-2 weights).
- Run RL for helpfulness, safety, persona — using preference data rather than verifiable rewards.

Output: R1 — reasoning capability of Stage-2 + general usefulness + safety.

### 1.3 R1-Distill family

- Take ~800k generations from R1.
- SFT on Llama / Qwen base models of various sizes (1.5B, 7B, 8B, 14B, 32B, 70B).
- No RL.
- Headline: R1-Distill-Qwen-32B beats GPT-4o-2024-05 on AIME and MATH despite ~5x smaller.

> **Saying it out loud.** The distill story is almost too simple: take about eight hundred thousand generations from R1, supervised-fine-tune Llama and Qwen bases from 1.5B up to 70B on them, and stop — no RL on the student at all. DeepSeek's headline is that the 32B distillate beats GPT-4o from May 2024 on AIME and MATH at roughly a fifth of the size, and that's a vendor-published benchmark claim, so I'd say it as theirs rather than as fact. The mechanism is that the teacher's chain of thought is a worked demonstration of the search policy, so the student picks up backtracking and self-checking by imitation instead of discovering them through RL. The hard limit to name: imitation can't exceed the teacher, so distillation democratises a capability but doesn't advance the frontier — you need RL on top for that.

### 1.4 Why this paper matters

- First public **end-to-end frontier recipe** with full numerical results.
- First demonstrates **outcome reward suffices** for reasoning RL — no PRM needed.
- First demonstrates **distillation transfers reasoning** at scale.
- Key data + code + weights released, kicking off the Open-R1 reproductions.

### 1.5 Interview-ready 90-second answer

> R1 is a four-stage post-training pipeline starting from DeepSeek-V3 base. Stage one is cold-start SFT on a small high-quality long-CoT dataset to give the model legible reasoning format. Stage two is reasoning-RL with GRPO and verifiable rewards on math, code, and logic — same recipe as R1-Zero but starting from the SFT'd model — with an added language-consistency reward to fix R1-Zero's English/Chinese mixing. Stage three regenerates ~800k SFT examples from the stage-two model, mixing 600k reasoning and 200k chat/writing/QA, and re-SFTs from V3-base. Stage four is final RLHF for helpfulness and safety. Distillation: SFT smaller bases on R1's generations, no RL needed, and you get strong reasoning models very cheaply.

> **Saying it out loud.** R1 matters because it's the first time anyone published a complete frontier reasoning recipe with the numbers attached. There are really two results in the paper. R1-Zero is the scientific one: pure RL straight from the base model, verifier reward plus a format reward, GRPO, no SFT and no process reward model — and reasoning emerges anyway, with AIME pass-at-one going from about sixteen percent to seventy-one. R1 is the engineering one: four stages, where cold-start SFT buys legibility, RL buys capability, rejection-sampled SFT buys breadth, and final RLHF buys safety. If you remember one thing, remember that each stage exists to repair a specific defect of the one before it, and that the paper is unusually honest about those defects — R1-Zero mixes languages mid-chain and makes a bad chatbot, and they say so.

---

## 2. Tülu 3

AllenAI's fully open recipe. Released November 2024. The most reproducible frontier recipe before R1 — and the cleanest one to reference for "open-source SOTA."

### 2.1 Starting point

- Llama 3.1 base (8B, 70B). Pretrained but no instruction tuning.

### 2.2 Stages

##### Stage 1: SFT

- Tülu 3 SFT mix: open and synthetic data with explicit per-skill curation (chat, math, code, reasoning, multilingual, persona).
- ~939k examples after deduplication / decontamination.
- Standard cross-entropy SFT.

##### Stage 2: DPO

- Tülu 3 DPO Mix: ~270k preference pairs.
  - Both human-labeled (UltraFeedback-style sources) and synthetic.
- Length-controlled DPO loss to combat length bias.
- KL anchored to stage-1 SFT model.

##### Stage 3: RLVR (the new contribution)

- Verifiable reward training on:
  - **Math.** GSM8K-style; final-answer extraction; sympy verification.
  - **Multi-turn instruction following.** IFEval-style verifiable constraints (length, JSON, etc.).
  - **Code-graded rewards.** When applicable.
- Algorithm: PPO with value head (Tülu 3 uses PPO; later open-RL libraries like trl-lite and verl support GRPO).
- KL anchored to DPO model.

### 2.3 Headline

- 70B Tülu 3 matches GPT-3.5 / Claude 2 on standard benchmarks.
- 8B Tülu 3 beats most open-source 8B models.
- The full data, code, evals, and intermediate checkpoints are public — best teaching artifact for open-RL.

### 2.4 Why this matters for interviews

- **Fully reproducible.** "How would you build an instruction model from scratch?" → walk through Tülu 3.
- Demonstrates RLVR helps **even without a reasoning-RL phase** — verifiable rewards work for instruction-following too.
- Shows DPO + RLVR can coexist (DPO for general preferences, RLVR for verifiable subsets).

### 2.5 Interview answer (60-second)

> Tülu 3 is AllenAI's open-source post-training recipe. Three stages from Llama 3.1 base: SFT on a curated mix of ~939k examples, DPO on ~270k preference pairs with length-controlled loss, and RLVR with PPO on math final-answer verification and IFEval-style verifiable instruction-following constraints. The key contribution beyond standard SFT+DPO is RLVR — using verifiable rewards on a subset of skills where you can write a deterministic checker, even outside math/code. Headline is that 70B Tülu 3 matches GPT-3.5 with a fully reproducible pipeline.

> **Saying it out loud.** Tülu 3 is the one to cite when someone asks how you'd build an instruction model from scratch, because it's fully reproducible — data, code, evals and intermediate checkpoints all public, which no frontier lab offers. Three stages from Llama 3.1 base: SFT on a curated mix of around nine hundred thousand examples, DPO on about two hundred seventy thousand preference pairs with a length-controlled loss to fight length bias, then RLVR. The genuinely new contribution is that last stage, and specifically that they applied verifiable rewards *outside* math — IFEval-style instruction-following constraints like "answer in JSON" or "under fifty words" are deterministically checkable, so you can use exact rewards there too. The lesson to take away is that "verifiable" is a broader category than people assume, and finding new verifiable slices is cheap capability.

---

## 3. Llama 3

Meta's recipe published in *The Llama 3 Herd of Models* (Jul 2024). The most thorough public industrial-scale post-training description as of mid-2024. No reasoning-RL stage (Llama 3.1 family, as of writing); strong on multi-round SFT + DPO.

### 3.1 Starting point

- Llama 3.1 base (8B, 70B, 405B). Pretrained on 15.6T tokens.

### 3.2 Stages

##### Stage 1: SFT data construction

- **Rejection sampling.** Sample many outputs from the previous (or initial) model; filter by RM; keep best.
- **Per-capability mixes.** Separate data pipelines for code, math, reasoning, multilingual, tool-use, dialogue.
- Iterate over 6+ rounds, each refining data quality.

##### Stage 2: SFT

- Standard cross-entropy on the rejection-sampled data.
- Run for multiple epochs.

##### Stage 3: DPO

- DPO on preference pairs constructed from the rejection-sampled data and from human labelers.
- Per-capability fine-tuning loops for safety, code, reasoning, etc.

##### Stage 4: Tool-use RL

- Specifically for code interpreter, calculator, search, scoring tools.
- The model learns to invoke tools, parse results, integrate.

##### Stage 5: Adversarial safety

- Red-team-generated adversarial prompts.
- DPO on safe-vs-unsafe pairs.

### 3.3 Why no PPO?

Meta explicitly chose DPO over PPO for cost / stability reasons at scale. With careful preference-pair construction (rejection-sampling with RM), DPO is competitive with PPO and far cheaper.

> **Saying it out loud.** Meta went with DPO rather than PPO explicitly for cost and stability at 405B scale, and the reasoning is worth being able to give. PPO needs a reward model and a value head resident alongside the policy, plus an online generation loop, which at that scale is a serious memory and orchestration burden with a lot of ways to go unstable. DPO is just a supervised loss on preference pairs — no critic, no sampling loop during training. The catch is that DPO is offline, so it only learns from pairs you already have, which is why the recipe is *iterative*: better model, better rejection-sampled data, next round. The tradeoff to name is that you trade PPO's on-policy exploration for stability and cost, and you compensate by doing the exploration in the data-construction step instead.

### 3.4 Iterative DPO

The recipe is iterated: improved model → better rejection-sampled SFT data → next-round SFT → next-round DPO. Multiple rounds. This is *iterative DPO* / *online DPO*, not PPO.

### 3.5 Interview answer (60-second)

> Llama 3 is Meta's iterative SFT + DPO recipe. They construct SFT data via rejection sampling — sample many outputs, score with reward model, keep best — and iterate this for 6+ rounds, each round improving the model's outputs which improve the next round's data. Per-capability SFT pipelines for code, math, reasoning, multilingual, tool-use, and dialogue. After SFT, DPO with carefully constructed pairs, then a separate tool-use RL phase and an adversarial safety RL phase. They explicitly chose DPO over PPO for stability and cost at the 405B scale. No reasoning-RL stage in 3.1 — that's expected in Llama 4.

> **Saying it out loud.** Llama 3's post-training is the most detailed public account of doing this at industrial scale, and its shape is different from R1's — no reasoning-RL stage at all in the 3.1 family, just many rounds of SFT and DPO. The engine is rejection sampling: generate lots of candidates, score with a reward model, keep the best, fine-tune, and then use the improved model to generate better candidates next round. Six-plus rounds of that, with separate data pipelines per capability — code, math, multilingual, tool use, dialogue — plus a dedicated tool-use RL phase and an adversarial safety phase on red-team prompts. The interesting choice is DPO over PPO, made explicitly for cost and stability at 405B. The takeaway is that data quality iteration, not algorithm choice, is where most of the gain came from — which is a good thing to say because it's true and it's not the answer people expect.

---

## 4. Qwen 2.5 / Qwen3

Alibaba's recipe. Most detailed in the QwQ-32B and Qwen 2.5 Math papers.

### 4.1 Qwen 2.5 base

- Trained on 18T tokens.
- Math/code-heavy mix.

### 4.2 Qwen 2.5 Math (specialized)

- SFT on curated math reasoning data (200k high-quality CoTs).
- DPO on math preference pairs.
- RLHF with PPO on math + general tasks.
- PRM evaluated; included in re-ranking but **not** in the RL reward (after experiments showed it didn't help training).
- Tool-integrated reasoning: model can invoke a calculator mid-CoT.

### 4.3 QwQ-32B-Preview (Nov 2024) / QwQ-32B (Mar 2025) and Qwen3-thinking models

- Cold-start SFT on long-CoT data (similar in spirit to R1).
- RLVR for reasoning.
- Final RLHF.

The Qwen team's published ablations report that PRMs helped *modestly* in some experiments but the canonical recipe used ORM only — agreeing with R1's experience.

### 4.4 Why interesting

- Strong open-weights reasoning models (QwQ, Qwen3-thinking).
- Tool-integrated reasoning (calculator + Python via execution).
- Per-domain specialization recipes (Math, Code, Coder).

> **Saying it out loud.** Qwen's recipe rhymes with R1 — cold-start SFT on long chains, RLVR for reasoning, then a final RLHF pass — which is useful corroboration, because two independent teams landing on the same shape is stronger evidence than one paper. Two details worth carrying. First, on process reward models they report the same thing DeepSeek did: PRMs helped modestly in some experiments, but the shipped recipe used outcome reward only, and they kept the PRM for reranking at inference rather than as a training signal. That's now two independent groups saying the same thing, which is about as close to settled as this question gets. Second, tool-integrated reasoning — the model calls a calculator or Python mid-chain — which sidesteps the fact that multi-digit arithmetic is a genuinely bad use of a transformer's forward pass.

---

## 5. Open-R1 and community reproductions

Following R1's release, HuggingFace launched **Open-R1** (Jan 2025+) — community reproduction of the R1 recipe.

### 5.1 What's open-source

- **R1's actual training code** is public (GitHub).
- R1's *exact training data* is **not** public.
- Open-R1 is reproducing the data + recipe with open data sources.

### 5.2 Variants

- **OpenThinker** (Stanford / Bespoke). Apply R1-distill methodology to open models with open data. Strong AIME numbers.
- **Bespoke-Stratos**. Distillation reproduction at scale.
- **Sky-T1**. Open reasoning model from Berkeley.
- **TÜlU3-RLVR variants.** Add longer CoT to Tülu's RLVR phase.

### 5.3 Lessons from reproductions

- R1's pipeline reproduces well at smaller scales.
- Distillation works robustly: SFT on long-CoT generations transfers reasoning.
- Pure-RL from a non-frontier base is hard — R1-Zero needed V3-base's quality. Open reproductions of pure-RL on Llama 3 base get smaller gains.

> **Saying it out loud.** Three things came out of the community reproductions and they're more informative than the original paper in one respect. One: distillation reproduces robustly — everyone who fine-tuned a small base on long chains got the reasoning transfer, first try. Two: the pipeline reproduces well at smaller scale, so the recipe isn't secretly dependent on 671B parameters. Three, and this is the important one: pure RL from a non-frontier base does *not* reproduce well — R1-Zero's result leaned on how strong V3-base already was, and people running the same thing on Llama 3 base got much smaller gains. That's direct evidence for the elicitation story: RL surfaces reasoning the base model already has priors for, so base quality sets the ceiling.

### 5.4 Implications

- The data + RL infrastructure is the moat, not weights.
- Open-source is ~3-6 months behind frontier, closing fast on reasoning.

> **Saying it out loud.** What's actually open here is worth being precise about, because interviewers probe it. DeepSeek released the weights and the training code, but not the training data — so Open-R1 and the community efforts are reconstructing the data with open sources, which is the hard part. The reproductions taught us that distillation transfers robustly and the pipeline scales down fine, but that pure RL from a weaker base does not reproduce, which points back at base-model quality as the real constraint. The strategic conclusion people draw is that the moat is the data and the RL infrastructure, not the weights — and the empirical observation is that open source is running something like three to six months behind on reasoning and closing.

---

## 6. The "interview cookbook" — synthesized recipe

If asked "design a post-training pipeline for a frontier reasoning model from scratch" — answer with this 6-stage recipe (synthesizing R1, Tülu 3, Llama 3 best practices):

### Stage 0: Pretraining priors

- Strong math / code / reasoning fraction in pretraining.
- High-quality long-CoT documents in pretraining (textbooks, math papers, code with comments, problem-solution pairs).
- This sets the *ceiling* of what RL can elicit.

### Stage 1: Cold-start SFT (legibility + format)

- Curate ~5-50k high-quality long-CoT examples.
  - Sources: human-written, generated by an existing reasoning model (rejection-sampled), manually cleaned.
- SFT base on this.
- Output: a model that knows the long-CoT format and can produce legible reasoning.

### Stage 2: Reasoning-RL (capability)

- Verifiable problem set: math (GSM8K, MATH, AIME-level), code (HumanEval, LiveCodeBench), logic (LogiQA-style).
- Algorithm: GRPO (or RLOO / REINFORCE++).
- Reward: correctness + format + language-consistency. No PRM (start simple; add later if helpful).
- KL to Stage-1 SFT.
- Train until reasoning saturates.
- Output: strong reasoning, possibly weak chat.

### Stage 3: Rejection-sampling SFT data

- Generate ~500k-1M outputs from Stage-2 model.
- Filter:
  - Verifiable subset (math, code): keep correct.
  - Non-verifiable subset (writing, factual QA, persona): filter by judge (genRM).
- Mix: ~70% reasoning, ~30% chat.

### Stage 4: SFT (broaden distribution)

- Re-SFT base on the Stage-3 mix.
- Output: reasoning + chat capable, but possibly weaker on safety / persona.

### Stage 5: Final RLHF (helpfulness + safety)

- Preference data: human + AI-labeled (RLAIF) + Constitutional revisions.
- DPO or PPO with KL to Stage-4 SFT.
- Targets: helpfulness, harmlessness, persona, refusal calibration.
- Iterate 1-2 rounds.

### Stage 6: Distillation (productization)

- Sample ~1M generations from Stage-5 model.
- SFT smaller bases.
- Optionally: light DPO on top.

### Stage 7: Production

- Deploy reasoning model with test-time-compute knobs.
- Routing layer: cheap LM detects when reasoning is needed.
- Online telemetry, RM calibration tracking, safety monitoring.

### What you can additionally mention to impress

- **PRMs / process supervision** as an option, with the caveat that R1 reported ORM was sufficient.
- **MCTS at training data generation** (AlphaProof-style) for math.
- **Generative reward models** for non-verifiable subsets.
- **Deliberative alignment** for reasoning-aware safety.
- **Iterative DPO** as a cheaper alternative to PPO.
- **Test-time compute scaling laws** (Snell et al.) — match pretraining compute with inference compute on hard tasks.
- **Multi-objective reward** (Pareto / MORLHF) for helpful-honest-harmless tradeoffs.

> **Saying it out loud.** If I get "design a post-training pipeline from scratch", I'd give seven steps and say why each exists rather than just listing them. Stage zero is pretraining mix — heavy on math, code and worked problem-solution text — because that sets the ceiling on what RL can elicit, and no amount of clever post-training fixes a base without reasoning priors. Then cold-start SFT on a few tens of thousands of clean long chains for format and legibility. Then reasoning RL with GRPO or RLOO on verifiable problems, correctness plus format plus language consistency, KL back to the SFT model — that's where capability comes from. Then rejection-sampled SFT to broaden back out to chat and writing, then final RLHF for helpfulness and safety, then distillation into small models for actual deployment. The framing that scores is that stages two and three trade against each other — RL narrows the model onto reasoning, the SFT stage widens it back — so the ordering is a deliberate sequence of repairs, not a pipeline of independent improvements.

---

## How to use this chapter

1. Memorize §1.5 (R1 90-second), §2.5 (Tülu 3 60-second), §3.5 (Llama 3 60-second). These are oral exam answers.
2. For an open-ended "design from scratch" question, give the §6 synthesized recipe.
3. Be ready for follow-ups on any single stage (data construction, KL choice, GRPO vs PPO, why ORM not PRM, etc.).
4. Read the actual papers (R1, Tülu 3, Llama 3) once for ground truth.

Single sentence to remember: **R1 is the canonical 4-stage reasoning recipe (cold-start SFT → reasoning-RL → rejection-sampling SFT → final RLHF); Tülu 3 is the canonical open SFT+DPO+RLVR recipe; Llama 3 is the canonical iterative-SFT+DPO recipe at 405B scale; combine the best of all three for any frontier interview "design a pipeline" question.**
