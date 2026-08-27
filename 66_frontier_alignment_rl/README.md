# Topic 66: Frontier Alignment + RL — Reasoning Models, Reward Modeling, Post-Train Playbooks

> Calibrated for **OpenAI / DeepMind / Anthropic research-scientist** interview rounds.

This folder is the depth pass on the most-asked-about-in-2025 frontier topic: how reasoning models like o1 / R1 / R1-Zero are actually trained, what's at the frontier in reward modeling, and the published recipes you should be able to walk through end-to-end.

> 🔥 **Read in this order:**
> - **[`REASONING_MODELS_DEEP_DIVE.md`](REASONING_MODELS_DEEP_DIVE.md)** — the highest-density chapter. Paradigm shift; test-time compute scaling (Snell et al.); RLVR; PRMs vs ORMs; search + RL (STaR, Quiet-STaR, V-STaR, ReST^EM, expert iteration, MCTS-based); R1-Zero with the "aha moment"; R1 four-stage pipeline; what we know about o1 / o3; reasoning distillation (R1-Distill); inference-time strategies (best-of-N, self-consistency, MBR, verifier-guided search); long-CoT failure modes (overthinking, hallucinated reasoning, language mixing); generative reward models; deliberative alignment; open frontier questions; senior signals.
> - **[`RLVR_DEEP_DIVE.md`](RLVR_DEEP_DIVE.md)** — the dedicated RLVR chapter. The full 2024-2025 algorithm zoo (PPO, GRPO, Dr.GRPO, DAPO with its 4 tricks, RLOO, REINFORCE++, VinePPO, PRIME, GSPO, Step-RL, Iterative DPO); verifier design (math sympy / code unit tests / tool-use / formal proofs / generative judges / composite); reward shaping (correctness, format, length, language consistency, step rewards) with composition patterns; KL regularization choices; curriculum and warm-start; common failure modes; open-source infrastructure (TRL, veRL, Open-RLHF, Open-R1, vLLM); multi-modal RLVR. **Substantial section on low-resource multilingual reasoning** with 7 approach families (translation-augmented, R1-distill, code-switched CoT, language-conditioned, synthetic data, tool-augmented, composite multi-stage), 6 concrete research project blueprints (BengaliMath-RL, Cross-Lingual Transfer Study, Code-Switched RLVR, Synthetic Multilingual Data, Multilingual PRMs, Tool-Augmented Multilingual), failure modes specific to low-resource settings, datasets and benchmarks landscape, open frontier questions.
> - **[`FRONTIER_REWARD_MODELING.md`](FRONTIER_REWARD_MODELING.md)** — modern RM landscape: scalar vs generative, Bradley-Terry vs regression, RLAIF, Constitutional AI, self-rewarding, full reward-hacking taxonomy with mitigations (length / sycophancy / format / refusal / verifier hack / prompt-injection), reward overoptimization (Gao et al.), RewardBench, online vs offline, full production playbook.
> - **[`OPEN_SOURCE_POSTTRAIN_PLAYBOOKS.md`](OPEN_SOURCE_POSTTRAIN_PLAYBOOKS.md)** — memorizable recipes: DeepSeek-R1 (full 4-stage with R1-Zero track), Tülu 3 (SFT+DPO+RLVR), Llama 3 (iterative SFT+DPO at 405B), Qwen 2.5 / QwQ, Open-R1 reproductions, plus a synthesized 6-stage "interview cookbook" recipe to deploy on any "design from scratch" question.
> - **[`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md)** — 150 active-recall questions across A–M with quick-fire and a 7-day drill plan.

## Why this folder exists

Existing content in [`08_training_techniques/ALIGNMENT_DEEP_DIVE.md`](../08_training_techniques/ALIGNMENT_DEEP_DIVE.md) covers RLHF / DPO / PPO / GRPO / DAPO foundations. **What that file is missing is the 2024–2025 frontier movement** — reasoning RL, R1, test-time compute scaling, modern reward modeling. This folder fills that gap at the depth a frontier-lab research-scientist round expects.

## Core insights

> A 2025 frontier reasoning model is a **long-CoT autoregressive policy trained by RL on verifiable + judge rewards, distilled from a larger search procedure, and deployed with inference-time compute as a third scaling axis** alongside pretraining and post-training compute.

> The modern reward stack is **verifiable-where-possible, generative-where-nuanced, scalar-where-cheap**. You measure overoptimization by holding out a different judge and you fight reward hacking via KL caps, ensembles, refresh, and explicit anti-bias data curation.

> R1's pipeline is the canonical 4-stage reasoning recipe: **cold-start SFT → reasoning-RL → rejection-sampling SFT → final RLHF**. Tülu 3 is the canonical open SFT+DPO+RLVR recipe. Llama 3 is the canonical iterative-SFT+DPO recipe at 405B scale. Combine the best of all three for any "design a pipeline" question.

## What sets this chapter apart from the existing alignment chapter

- Names the **R1 paper, R1-Zero paper, Snell test-time compute paper, Lightman PRM paper, Math-Shepherd, OmegaPRM, Tülu 3 paper, Llama 3 paper, Mahan genRM paper, Gao overoptimization paper, Bai Constitutional AI, Lee RLAIF, Yuan Self-Rewarding, AlphaProof / AlphaGeometry**.
- Walks through **R1's 4 stages** in interview-deliverable form (90-second oral answer + full detail).
- Walks through **Tülu 3's 3 stages** and **Llama 3's iterative SFT+DPO** end-to-end.
- **Reward hacking taxonomy** with diagnose + mitigation per pattern.
- **Reward overoptimization** — Gao et al. curve, KL caps, ensembles, refresh.
- **Frontier open questions** — what the next research-scientist hires might work on.
- **150 grill questions** with a 7-day drill plan.

## How to use this folder

1. Read [`REASONING_MODELS_DEEP_DIVE.md`](REASONING_MODELS_DEEP_DIVE.md) once cover-to-cover — it's the densest single document on 2025 reasoning RL.
2. Read [`FRONTIER_REWARD_MODELING.md`](FRONTIER_REWARD_MODELING.md) next.
3. Read [`OPEN_SOURCE_POSTTRAIN_PLAYBOOKS.md`](OPEN_SOURCE_POSTTRAIN_PLAYBOOKS.md) and **memorize the 60-90 second oral answers** for R1, Tülu 3, Llama 3.
4. Drill [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md). Target 130+/150 before a frontier-lab interview.
5. Read the actual DeepSeek-R1 paper (arXiv 2501.12948) — non-negotiable.

## Cross-references

- **[`08_training_techniques/ALIGNMENT_DEEP_DIVE.md`](../08_training_techniques/ALIGNMENT_DEEP_DIVE.md)** — DPO/PPO/GRPO foundations (this folder builds on top).
- **[`07_llm_problems/LLM_EVALUATION_DEEP_DIVE.md`](../07_llm_problems/LLM_EVALUATION_DEEP_DIVE.md)** — eval methodology, RewardBench.
- **[`07_llm_problems/HALLUCINATION_DETECTION_DEEP_DIVE.md`](../07_llm_problems/HALLUCINATION_DETECTION_DEEP_DIVE.md)** — factuality / hallucination overlap with reward hacking and reasoning failure modes.
- **[`65_llm_security/LLM_SECURITY_DEEP_DIVE.md`](../65_llm_security/LLM_SECURITY_DEEP_DIVE.md)** — security perspective on jailbreaks of reasoning models.
- **`67_frontier_intuitive_questions/`** — companion folder for Bayesian / probabilistic reasoning questions.
