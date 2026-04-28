# Post-Training & Alignment — Interview Grill

> 50 brutal questions on post-training, RLHF, DPO, and the alphabet soup. Drill until you can answer 40+ cold.

---

## A. Pipeline foundations

**1. Walk me through the standard post-training pipeline.**
Pretraining → SFT (cross-entropy on demonstrations) → preference optimization (RLHF or DPO or variants). Each stage adds a different signal: SFT adds format and basic instruction following; preference optimization adds nuanced quality and alignment. Sometimes followed by online RL (e.g. GRPO on verifiable tasks) or rejection sampling.

**2. Why is SFT alone not enough?**
SFT teaches imitation of one good response per prompt. It doesn't capture the space of acceptable responses or pairwise preferences. Models trained only on SFT often produce technically-correct-but-flawed responses because they can't learn from "this is better than that" signals.

**3. What's the Bradley-Terry preference model?**
$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$. Probability of preferring the winner equals sigmoid of the reward gap. Used in chess Elo; underpins almost all preference-based methods including the reward model in RLHF and the implicit objective of DPO.

**4. How is the reward model trained?**
Initialized from SFT model with a scalar head (replacing the LM head). Trained on preference pairs $(x, y_w, y_l)$ with the binary cross-entropy loss $-\log \sigma(r(x, y_w) - r(x, y_l))$. Typically tens of thousands of pairs, taking less than a day on the same hardware as SFT.

**5. What's the RLHF training objective?**
$\max_\pi \mathbb{E}[r(x, y) - \beta \cdot \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})]$. Maximize expected reward minus a KL penalty against a reference policy (the SFT model). The KL penalty is essential: it bounds how far the policy can drift, which limits reward hacking and preserves capabilities.

**6. Why is the KL penalty there?**
Three reasons. (a) Bound reward hacking — the RM is approximate and the KL anchor limits how much you can exploit its errors. (b) Capability preservation — without the anchor, the policy "forgets" pretrained knowledge that isn't being directly rewarded. (c) Distribution match — the RM is reliable only on data near its training distribution, which is near $\pi_{\text{ref}}$. Drifting too far makes the reward signal unreliable.

**7. What does $\beta$ control?**
The trade-off between matching the reward model (low $\beta$, aggressive) and staying close to $\pi_{\text{ref}}$ (high $\beta$, conservative). $\beta = 0.01$–$0.1$ is typical. Smaller $\beta$ increases reward but risks instability and reward hacking; larger $\beta$ is safer but limits how much improvement you get.

---

## B. PPO and RL specifics

**8. Why PPO and not policy gradient?**
PPO's clipped objective $\min(\rho_t \hat A_t, \mathrm{clip}(\rho_t, 1 \pm \varepsilon) \hat A_t)$ provides a soft trust region: it prevents updates that move the policy too far in one optimization step. Plain policy gradient is high-variance and unstable for the kinds of long horizons and large action spaces that LLMs have.

**9. What models are in memory during PPO-RLHF?**
Four: policy (training), reference policy (frozen, for KL), reward model (frozen, for rewards), value function (training, for advantages). For a 70B policy, this is ~1 TB of memory before optimizer state and KV cache.

**10. What's the value function for in PPO?**
Estimates the expected discounted reward $V(s)$ from each state. Used to compute advantages $\hat A_t = R_t - V(s_t)$, which reduce variance in the policy gradient. Trained jointly with the policy via MSE on observed returns.

**11. Why is the value function hard to train for LLMs?**
Variance: rewards are sparse (one reward per response). Distribution shift: as the policy improves, the value function lags. Compute: another full-size model. GRPO eliminates the value function by replacing it with a group-mean baseline.

**12. What's GRPO?**
Group Relative Policy Optimization (introduced in DeepSeekMath, Shao et al. 2024; popularized by DeepSeek-R1). Sample $K$ rollouts of the same prompt; advantage $= (\text{reward} - \text{group-mean}) / \text{group-std}$. Replaces the value function with a Monte Carlo group baseline. Cheaper, more stable for LLMs.

**13. When is GRPO especially well-suited?**
Verifiable-reward settings where you can sample many candidates and grade them deterministically (math, code). The group mean gives a clean per-prompt baseline. For preference-based rewards (helpfulness, safety) GRPO works but offers less obvious advantages over PPO.

---

## C. DPO

**14. Walk me through the DPO derivation.**
**One-line story** (verbal): "DPO turns the RLHF objective into a supervised loss because the optimal policy has a closed form, and the partition function cancels in the preference comparison."

**Whiteboard version** (5 steps):
1. Start: $\max_\pi \mathbb{E}[r] - \beta \mathrm{KL}(\pi \| \pi_{\mathrm{ref}})$.
2. Closed-form optimum: $\pi^*(y|x) = \tfrac{1}{Z(x)} \pi_{\mathrm{ref}}(y|x) \exp(r/\beta)$.
3. Invert for $r$: $r = \beta \log(\pi^*/\pi_{\mathrm{ref}}) + \beta \log Z(x)$.
4. Plug into Bradley-Terry $P(y_w \succ y_l) = \sigma(r_w - r_l)$ — $\log Z(x)$ depends only on prompt, so cancels in the difference.
5. DPO loss = NLL of this preference probability. No reward model, no rollouts.

**15. Why does the partition function $Z$ cancel?**
$Z(x)$ depends only on the prompt, not on the response. So $Z(x)$ appears with the same value for both $y_w$ and $y_l$ in the Bradley-Terry difference, and they cancel. This is the elegant trick that makes DPO possible.

**16. Is DPO equivalent to RLHF?**
Theoretically equivalent under the assumption that the optimal RLHF policy has the closed-form $(1/Z) \pi_{\text{ref}} \exp(r/\beta)$ and that the reward model perfectly fits Bradley-Terry preferences. In practice the equivalence is approximate because both assumptions are violated.

**17. DPO vs PPO trade-offs.**
DPO: simpler implementation, more stable training, off-policy. PPO: on-policy, can keep adapting as policy drifts, more expressive optimization.

**18. When does DPO fail?**

- When the policy needs to drift far from the preference data distribution (off-policy).
- When the preference data is biased in ways the implicit reward picks up (length bias is the classic example).
- When the preference data is near-deterministic (gradient blowup; IPO fixes this).

**19. What's length bias in DPO?**
The implicit reward $\beta \log(\pi / \pi_{\text{ref}})$ sums over tokens. Longer responses can have larger reward gaps purely because they have more terms. The policy learns to produce longer responses than necessary to win preferences. SimPO's length normalization fixes this.

**20. What's IPO and what does it fix?**
Identity Preference Optimization (Azar et al. 2023). Replaces DPO's sigmoid loss with a squared error: $\mathcal{L}_{\text{IPO}} = \mathbb{E}[(\beta \cdot \text{gap} - 1/2)^2]$. Bounded loss, doesn't blow up on near-deterministic preferences. More robust to label noise.

**21. What's KTO?**
Kahneman-Tversky Optimization (Ethayarajh et al. 2024). Works with **unpaired** binary feedback (thumbs up / thumbs down) instead of preference pairs. Asymmetric loss inspired by prospect theory: penalize bad more than reward good. Useful when you have unpaired labels at scale.

**22. What's ORPO?**
Odds-Ratio Preference Optimization (Hong et al. 2024). Combines SFT and preference optimization in a single stage. Adds an odds-ratio penalty on disliked responses to the SFT loss. No reference model needed. Faster than DPO, comparable quality.

**23. What's SimPO?**
Simple Preference Optimization (Meng et al. 2024). Length-normalizes the implicit reward and removes the reference policy: implicit reward becomes $\beta / |y| \cdot \log \pi_\theta(y \mid x) - \gamma$. Eliminates length bias and saves memory (no reference model in memory).

---

## D. Failure modes

**24. What's reward hacking?**
The policy finds outputs that score high under the RM but aren't actually good. Examples: longer-is-better (length bias in RM), authoritative-sounding-is-better (style mimicry), repetition of certain phrases. The fundamental cause: the RM is approximate, and aggressive optimization exploits its errors.

**25. How do you detect reward hacking?**
Compare RL-policy outputs to SFT outputs on held-out prompts using a *different* RM (held out from RL training). Have humans grade. If the trained RM scores the policy highly but the held-out RM or humans don't, you have reward hacking. The "Goodhart curve" — true reward (human) on y-axis, KL distance from $\pi_{\text{ref}}$ on x-axis — typically shows true reward rising then falling as KL increases.

**26. What's KL blowup?**
The KL divergence $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ grows uncontrollably. Symptoms: the policy diverges from the SFT model, language quality drops, capabilities are lost, outputs become idiosyncratic. Caused by $\beta$ too small or RM exploiting OOD regions. Fix: increase $\beta$, add gradient clipping, monitor KL during training.

**27. What's mode collapse?**
The policy collapses to a narrow distribution — same response or near-same response regardless of prompt. Signs: low entropy, similar outputs across diverse prompts. Common in late-stage RL when the policy has "found" what the RM likes and stops exploring.

**28. What's sycophancy and where does it come from?**
The policy agrees with whatever the user implies. Came from RLHF training: human preference labelers tend to prefer responses that agree with their phrasing. The RM picks this up; the policy amplifies it.

**29. What's the "alignment tax"?**
Capability loss from preference optimization. The RL/DPO process can degrade capabilities measured by capability benchmarks (MMLU, HumanEval, etc.) even as preference benchmarks improve. The KL anchor mitigates this; aggressive $\beta$ reduction makes it worse.

**30. What's overoptimization in the Goodhart sense?**
Continued RL training drives RM-reward up but eventually drives true (human) reward down. Caused by going off-distribution from RM's training data. The peak-then-decline pattern is the signature. Documented in Gao et al. 2023 ("Scaling Laws for Reward Model Overoptimization").

**31. How do you mitigate overoptimization?**
Early stopping on held-out human/RM evaluation. Iterated DPO with refreshed preference data. KL constraint with larger $\beta$. RM ensembles (penalize uncertainty as well as mean).

---

## E. Specific techniques and recipes

**32. Walk me through Constitutional AI.**
Anthropic's Constitutional AI uses a written set of principles ("constitution") to generate critique-and-revision pairs from the model itself. Steps: (1) generate a response to a potentially-harmful prompt; (2) the model critiques its own response against constitutional principles; (3) the model rewrites following the critique. The (original, rewritten) pairs become preference data. Reduces dependence on human labelers for harmlessness.

**33. What's RLAIF?**
RL from AI Feedback. Use an AI judge (often a stronger model, or the model itself) to label preferences instead of humans. Cheaper and faster than RLHF. Risk: judge may have biases or hallucinate quality. Requires evaluation against held-out human labels to validate.

**34. Process supervision vs outcome supervision?**
Outcome: reward correctness of final answer only. Easy to label but sparse signal. Process: reward correctness of each reasoning step. Denser signal, better empirically for math/logic, but expensive to label per-step. OpenAI's PRM800K showed process supervision substantially beats outcome supervision for math reasoning.

**35. Why do reasoning models (o1, R1) increasingly use outcome rewards?**
For verifiable tasks (math, code), the outcome reward is exact (does the program pass tests? Is the math right?). No RM needed; no reward hacking possible. The trade-off: only works on verifiable tasks. Combined with chain-of-thought generation, outcome rewards on long traces have produced the most capable reasoning models.

**36. What's iterated DPO (or rejection sampling DPO)?**
Apply DPO once. Sample new responses from the updated policy. Have a judge grade them to produce fresh preference pairs. Apply DPO again. Bridges the on-policy gap that DPO has versus PPO. Used in Tülu 2/3, Llama 3.

**37. What's a good ratio of SFT data to preference data?**
Domain-dependent. Frontier-lab recipes typically use 100K–1M SFT examples and 100K–1M preference pairs. SFT data quality dominates SFT quantity (LIMA showed 1K high-quality examples competitive with much larger sets). Preference data tends to need more volume to capture diversity.

---

## F. Reward modeling

**38. What biases do reward models have?**
Length (longer = "more thorough"), certainty (confident-sounding = "more accurate"), formatting (lists and structure = "well-written"), style (matches labeler's writing preferences). Inherits biases from the labelers. Common to use standardized labeler training and length-normalized scoring to combat.

**39. Single RM vs ensemble RM?**
Single: cheaper, simpler, works for moderate optimization. Ensemble: more robust to RM quirks, can use disagreement as uncertainty estimate. Ensembles help most when training to extremes — the policy can't find a single weakness if multiple RMs must agree.

**40. What's reward shaping?**
Augmenting the learned reward with hand-crafted terms: length penalty to combat length bias, repetition penalty for n-gram repetition, refusal scoring on harmful prompts. Done well, mitigates RM weaknesses. Done poorly, introduces new failure modes.

**41. RewardBench — what is it and what does it tell us?**
A benchmark for evaluating reward models on held-out preference data (Lambert et al.). Reveals: many open-source RMs are noticeably miscalibrated, especially for chat and reasoning. RM quality matters a lot for downstream RLHF/DPO — a mediocre RM will produce a mediocre aligned model.

---

## G. Evaluation

**42. How do you evaluate an aligned model?**
Capability benchmarks (MMLU, GSM8K, HumanEval) — should not regress from SFT. Preference benchmarks (AlpacaEval, MT-Bench, Arena-Hard) — should improve. Specific safety benchmarks. Online human evaluation for ground truth.

**43. What's AlpacaEval and what's wrong with it?**
LLM-judge-based win-rate evaluation against a baseline (often GPT-4-turbo). Cheap, scalable. Wrong/limited because: judge model has biases (length bias is famous), the prompts are limited, agreement with human judgment is imperfect.

**44. What's the alignment tax and how do you measure it?**
Capability degradation from RL training. Measure by benchmark scores (MMLU, HumanEval, MATH) before vs after preference training. Healthy run: <2-point regression on each. Concerning: 5+ point regression. Almost-zero alignment tax requires careful KL tuning and data curation.

**45. Why is calibration important after alignment?**
Aligned models tend to become overconfident. RL pushes the policy to sharpen its distribution toward "good" responses, often at the cost of well-calibrated uncertainty. Symptoms: model never says "I don't know"; refuses with low confidence; etc. Test by comparing predicted vs observed correctness on held-out factual prompts.

---

## H. Online RL on verifiable rewards

**46. Walk me through DeepSeek-R1's training.**
**3-beat story**: Pretrain → cold-start SFT on reasoning traces → GRPO with verifiable rewards. **The trick**: outcome-only rewards on math/code (right answer = 1, wrong = 0) make reward hacking impossible — there's nothing to game. Detail beats: $K = 16$ samples per prompt for group-mean advantage, long generation horizons (thousands of tokens for chain-of-thought), KL anchor to prevent capability loss.

**47. Why can o1/R1 learn reasoning without explicit reasoning supervision?**
Outcome rewards on math/code create a learning signal for any reasoning that leads to correct answers. The model spontaneously discovers chain-of-thought, self-verification, and even "aha moments" because they're correlated with correctness. With enough RL, these behaviors become reliable strategies.

**48. What are the limits of outcome-reward RL?**
Only works on verifiable tasks. Helpfulness, creativity, judgment — these don't have automatic verifiers. So the recipe is: outcome rewards on math/code/STEM; preference rewards (RLHF/DPO) on open-ended tasks. The two rewards live in tension; balancing them is hard.

---

## I. 2024-2025 frontier methods

**49. What does DAPO fix in GRPO?** (ByteDance 2025)
Four named tricks: **Clip-Higher** (asymmetric upper/lower clip ranges to prevent entropy collapse on rare tokens), **Dynamic Sampling** (drop prompts where all $K$ rollouts succeed-or-fail since variance is zero), **token-level loss** (average over tokens not samples → fixes vanilla GRPO's length bias), **overlong-reward shaping** (soft length penalty). State-of-the-art for RLVR as of 2025.

**50. What does Dr. GRPO fix in GRPO?**
Two biases. Drops the $\sigma_{\mathrm{group}}$ normalization (which over-emphasizes easy prompts where $\sigma$ is small) and switches token aggregation from mean to sum (which was making longer responses count less). $\hat A = r - \mu_{\mathrm{group}}$, no std.

**51. What's RLOO?**
REINFORCE Leave-One-Out. Sample $K$ rollouts; advantage = own reward − mean of the *other* $K-1$. No critic, no PPO clipping needed at small $K$. Surprisingly competitive with PPO/GRPO at smaller scale. Used by Tülu / Ai2 in some recipes.

**52. What's REINFORCE++?**
Plain REINFORCE with reward whitening, baseline subtraction, gradient clipping, KL anchor. Demonstrates that "vanilla" REINFORCE — once tuned — can match PPO/GRPO. Strong baseline; minimal complexity.

**53. What's RLVR?**
RL with **Verifiable Rewards**. The reward is a programmatic verifier (exact-match on math, unit tests on code, JSON schema check). Eliminates reward hacking — you can't game an exact-match. Combined with GRPO/DAPO, it's the dominant alignment paradigm for capability-pushing in 2025 (o1, R1, QwQ).

**54. What's TDPO (Token-level DPO)?**
DPO assigns one preference label to a whole response. TDPO breaks this down per-token via a sequence-level utility plus per-token KL regularization. Reduces DPO's "all-or-nothing" credit assignment problem and improves stability when responses differ in only a few tokens.

**55. What's Step-DPO?**
Process-level preferences for reasoning. Collect (good step, bad step) pairs at each reasoning step, apply DPO at step granularity. Combines DPO's simplicity with process-supervision's denser signal.

**56. What's Self-Rewarding Language Models?** (Yuan et al. 2024)
The model is both policy AND judge. Iteratively: (1) generate, (2) self-score with built-in LLM-judge ability, (3) form preference pairs, (4) DPO. Removes external reward models. Risk: judge biases compound.

**57. What's SPIN (Self-Play Fine-tuning)?**
Treat SFT data as "expert," current model generations as "learner." Train the model to prefer expert over its own generations via DPO-style loss. Iterates until the gap closes. Pure self-improvement from existing SFT data — no new labels.

**58. What's Iterative / Online DPO?**
DPO is off-policy. To approximate online RL: alternate (a) sample fresh responses from the current policy, (b) judge them, (c) form new preference pairs, (d) DPO again. Multiple rounds bridge the on-policy gap. Used in production Tülu and Llama 3 recipes.

**59. What's NLHF (Nash Learning from Human Feedback)?**
Game-theoretic alternative to Bradley-Terry. Drops the transitivity assumption. Seeks the policy that wins (in expectation) against any other policy in pairwise comparisons — a Nash equilibrium of the preference game. Algorithm: regret minimization / mirror descent. More principled when preferences are intransitive; less common in practice.

**60. Best-of-N at inference — when use it?**
Don't change the policy at all. Sample $N$ responses, score with reward model or verifier, return the highest. Trade compute for quality. Production use: math reasoning (sample 16, return the verified one). Often paired with RL — RL concentrates probability on good responses; Best-of-N polishes the tail.

**61. When use DAPO vs Dr. GRPO vs RLOO vs PPO?**
- DAPO: large-scale verifiable-reward RL with plenty of compute (frontier reasoning systems).
- Dr. GRPO: similar to DAPO but cleaner; equally good empirically.
- RLOO: small-budget RLHF; few rollouts per prompt; no critic needed.
- PPO: classic RLHF on preference rewards (helpfulness/harmlessness); requires reward model + critic.

**62. Why does the 2025 alignment stack look so different from 2023?**
2023: RLHF dominant, PPO + reward model + KL. 2025: RLVR replaced reward models on verifiable tasks; DAPO/Dr. GRPO replaced PPO; preference-based methods (DPO, ORPO, SimPO) handle open-ended tasks; Best-of-N at inference time; iterative pipelines (online DPO + RL stages) replace single-stage. The big shift: outcome verifiability beats reward modeling whenever it's available.

**63. The 2025 decision tree — when use which alignment method?**
- Math/code with verifier → DAPO + RLVR
- Tight compute on verifiable task → RLOO + verifier
- Single-shot preference pairs → DPO or SimPO
- Iterating allowed → online DPO
- Unpaired thumbs up/down → KTO
- One-stage SFT + alignment → ORPO
- Step-level reasoning signal → Step-DPO or process RL
- No new labels → SPIN
- No training, inference only → Best-of-N + verifier

---

## J. Quick-fire

**64.** *Default $\beta$ for RLHF?* $0.01$–$0.1$.
**65.** *RM training data scale?* 10K–1M preference pairs.
**66.** *Standard SFT batch size?* Hundreds of thousands of tokens per batch.
**67.** *DPO main advantage over PPO?* Simpler, stabler, no reward model.
**68.** *PPO main advantage over DPO?* On-policy, more expressive.
**69.** *KL blowup symptom?* Policy diverges from SFT, capabilities crash.
**70.** *Reward hacking symptom?* RM-reward up, human-reward flat or down.
**71.** *Mode collapse symptom?* Low output entropy, similar responses across prompts.
**72.** *GRPO group size typical?* 16–64 rollouts per prompt.
**73.** *Process vs outcome rewards?* Per-step vs final-answer.
**74.** *Constitutional AI key concept?* AI critique against principles.
**75.** *Alignment tax?* Capability degradation from RL.
**76.** *DAPO's four tricks?* Clip-Higher, Dynamic Sampling, token-level loss, overlong-reward shaping.
**77.** *Dr. GRPO drops?* $\sigma_{\mathrm{group}}$ normalization.
**78.** *RLOO advantage formula?* $r_i - \frac{1}{K-1}\sum_{j \neq i} r_j$.
**79.** *RLVR — what's the R?* **Verifiable** rewards (programmatic verifier replaces reward model).
**80.** *TDPO granularity?* Per-token (vs DPO's per-response).
**81.** *Self-Rewarding LM components?* Policy and judge are the same model.
**82.** *SPIN expert source?* SFT data treated as expert; current generations as learner.
**83.** *Online DPO key step?* Re-sample fresh responses, re-judge, re-DPO.
**84.** *Best-of-N at inference — what does it need?* A reward model or verifier.
**85.** *Frontier alignment recipe in one phrase?* RLVR + DAPO/Dr. GRPO for verifiable + DPO/iterative for everything else.

---

## Self-grading

If you can't answer 1–15, you don't know post-training. If you can't answer 16–35, you can't pass an alignment-focused MLE round. If you can't answer 36–50, you'll fall short in frontier-lab applied scientist screens. If you can't answer 49–63 (the 2024–2025 frontier methods), you'll be behind on what frontier labs actually use today.

Aim for 60+/85 cold before any alignment-focused interview.
