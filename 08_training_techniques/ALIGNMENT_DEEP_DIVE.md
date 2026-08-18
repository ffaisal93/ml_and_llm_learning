# Post-Training & Alignment: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Post-training (SFT → RLHF / DPO / GRPO / etc.) is the area where applied scientists at frontier labs spend most of their careers, and the area where interview questions probe deepest. This document derives every major method from first principles, explains why each one exists, and walks through the failure modes (reward hacking, length bias, KL collapse, mode collapse) that interviewers love to ask about.

---

## 1. The post-training stack, end to end

After pretraining, a model is a competent next-token predictor but not a useful assistant. Post-training turns it into one. The standard pipeline:

```
Pretrained base model
   ↓  Supervised Fine-Tuning (SFT) on (prompt, response) pairs
SFT model (knows the assistant format, can follow instructions roughly)
   ↓  Preference Optimization (RLHF / DPO / IPO / KTO / GRPO / ORPO)
Aligned model (preferred responses, refuses bad ones, stays consistent in style)
   ↓  Optional: rejection sampling, online RL, constitutional AI, etc.
Production model
```

Every method below is some answer to: **how do we shape the model's distribution over outputs to match human preferences without destroying the capabilities it learned in pretraining?**

> **Saying it out loud.** So the way I think about post-training is: pretraining gives you something that knows a lot but doesn't know it's supposed to be helping you. SFT teaches it the shape of a conversation, and preference optimisation teaches it which of many acceptable answers people actually like. Every method in this space — RLHF, DPO, GRPO — is answering the same question, which is how to move the model toward what humans want without knocking out the capabilities pretraining paid for. That tension has a name people use in interviews: the alignment tax.

---

## 2. Stage 1: SFT (supervised fine-tuning)

**What it is.** Standard cross-entropy training on (prompt, response) pairs. The response is teacher-forced; loss is summed over response tokens only (not prompt tokens).

**What it accomplishes.**

- Teaches the model the **format** of being an assistant (turn-taking, style, persona).
- Teaches **basic instruction following** — "answer the question" vs. continuing the prompt.
- Teaches **task-specific behaviors** that may be rare in pretraining (specific code style, refusal patterns, etc.).

**What it doesn't accomplish.**

- Doesn't teach the model what to do when there's no SFT example (out-of-distribution prompts).
- Doesn't capture pairwise preferences ("response A is better than B"). SFT only sees one good response per prompt.
- Doesn't optimize for a reward signal — just imitates demonstrations.

**Why SFT alone is not enough.** A skilled human writes one of many possible good responses. SFT teaches the model that this *specific* response is correct, not the *space* of acceptable responses. The model learns a narrow imitation rather than the underlying concept of "good." Preference optimization fixes this by teaching from comparisons.

**Common SFT subtleties (interview-relevant):**

- Loss masking on the prompt (don't compute loss on tokens the user typed).
- Packing multiple short examples into one sequence to reduce padding waste.
- Curriculum: easier instructions first, harder ones later, often improves results.
- Data quality dominates quantity. 10K high-quality examples beats 1M scraped ones (LIMA, Zhou et al. 2023).

> **Saying it out loud.** SFT is just next-token prediction again, only on curated (prompt, answer) pairs, and you mask the loss so the model isn't graded on text the user typed. What it buys you is format and instruction-following — the model learns it's an assistant, not an autocomplete. What it can't buy you is a sense of *better versus worse*, because it only ever sees one good answer and never a comparison. And the headline number here is LIMA: about 1,000 to 10,000 carefully written examples beat a million scraped ones, so quality dominates quantity.

---

## 3. Stage 2: The Bradley-Terry preference model

> **In plain language.** This section is about turning "a human clicked A over B" into a number. Bradley-Terry says each response has a hidden score, and how often people pick one over the other tells you the *gap* between those scores. Everything downstream — reward models, DPO, RLHF — is built on that one assumption.

Almost all preference-based methods start here. Given two responses $y_w$ (winner) and $y_l$ (loser) for prompt $x$, suppose there's a latent **reward function** $r(x, y)$ such that humans choose probabilistically:

$$
P(y_w \succ y_l \mid x) = \sigma\!\big(r(x, y_w) - r(x, y_l)\big)
$$

The probability of preferring $y_w$ is the sigmoid of the reward gap. This is the **Bradley-Terry model**, used in chess Elo ratings since the 1950s.

To learn $r$, fit by maximum likelihood on a preference dataset:

$$
\mathcal{L}_{\text{RM}} = -\sum_{(x, y_w, y_l)} \log \sigma\!\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
$$

This is a binary cross-entropy loss applied to preference pairs. The reward model $r_\phi$ is typically a transformer initialized from the SFT model, with a final scalar head replacing the LM head. Trained from preference data (typically tens of thousands of pairs).

**Properties of the Bradley-Terry rewards:**

- They're identified up to an additive constant: $r$ and $r + c$ give identical preference probabilities. So the reward scale is meaningless without anchoring.
- They assume **transitivity**: if humans prefer A > B and B > C, they should prefer A > C. Real human preferences violate this sometimes.
- They assume **independence of irrelevant alternatives**: the relative preference between A and B doesn't depend on whether C is in the set. Real preferences again violate this.

These violations matter for interpretation but are usually ignored in practice. Most modern preference methods use Bradley-Terry implicitly or explicitly.

> **Saying it out loud.** Bradley-Terry is the bridge from clicks to numbers. It assumes every response has a hidden quality score, and the chance a human picks one over the other is a sigmoid of the difference in those scores — the same math chess Elo has used since the 1950s. So training a reward model is just logistic regression on pairs: push the winner's score up, the loser's down. The catch worth naming is that the scores are only defined up to an additive constant, so the absolute reward number is meaningless — only gaps mean anything, which is exactly why you can't compare reward values across two different reward models.

---

## 4. RLHF: full pipeline math

> **In plain language.** RLHF is "chase the reward, but don't wander off." You let the model generate, score the generations with the reward model, push probability toward the high-scoring ones, and simultaneously penalise how far the model has moved from where it started. The math below is just those two ideas written down: a reward term and a leash term.

RLHF (Christiano et al. 2017, Ouyang et al. 2022) trains the policy to maximize expected reward subject to a KL constraint:

$$
\max_{\pi_\theta}\ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}
\left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]
$$

Three components:

- $r_\phi(x, y)$: reward from the trained reward model.
- $\pi_{\text{ref}}(y \mid x)$: reference policy, typically the SFT model — the "base" you don't want to drift far from.
- $\beta$: KL penalty coefficient (typically 0.01 to 0.1). Small $\beta$ = aggressive optimization; large $\beta$ = stay close to $\pi_{\text{ref}}$.

The objective rewards high-reward outputs but penalizes drifting from $\pi_{\text{ref}}$. Without the KL term, the policy would mode-collapse onto whatever maximizes the reward, often overfitting to reward model quirks (reward hacking).

> **Saying it out loud.** RLHF has three moving parts: a reward model that scores responses, a policy that tries to score well, and a KL penalty that stops the policy running off. Written out it's "maximise expected reward minus beta times how far you've drifted from the SFT model." Beta is the knob everyone tunes — typically somewhere between 0.01 and 0.1. Turn it down and you get more reward but the model starts gaming the scorer; turn it up and the model barely changes.

### Why the KL penalty is essential

The reward model is not the true reward. It's an approximation trained on a finite preference dataset. Optimizing too hard against it produces **specification gaming** / **reward hacking**: the policy finds outputs that exploit reward model errors rather than actually being good.

The KL penalty creates a budget for how far the policy can move from $\pi_{\text{ref}}$. As long as the policy stays close to $\pi_{\text{ref}}$ (which represents reasonable language and capabilities), reward hacking is bounded. Reduce $\beta$ and you increase reward at the cost of language quality and out-of-distribution behavior.

> **Saying it out loud.** The reward model isn't the truth, it's a guess at the truth fitted on a finite pile of comparisons. So if you optimise against it hard enough, you stop finding good answers and start finding its blind spots. The KL penalty is a leash: it says you can only move so far from the model you started with, and inside that radius the reward model is still roughly trustworthy. The failure mode when the leash is too loose has a name — reward hacking, or Goodhart's law: the reward number keeps climbing while human ratings quietly go down.

### The optimization: PPO

> **In plain language.** You can't backprop through "the model rolled a die and picked this token," so you use policy gradients instead: increase the log-probability of tokens that led to good outcomes. PPO adds a safety belt on top — if a single update would change a token's probability by more than about 20%, the gradient gets clipped off. That is all the scary-looking min-and-clip formula does.

The objective is non-trivial to optimize because $y$ is sampled from the policy (you can't differentiate through sampling). The standard solution is policy gradient via PPO (Schulman et al. 2017):

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}\Big[ \min\!\big(\rho_t \, \hat A_t,\ \mathrm{clip}(\rho_t,\ 1 - \epsilon,\ 1 + \epsilon)\, \hat A_t\big) \Big]
$$

where $\rho_t = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ is the importance ratio and $\hat A_t$ is the advantage estimate. The clip prevents updates that move the policy too far in one optimization step (a soft trust region).

For RLHF:

- $s_t$ = prompt + tokens generated so far.
- $a_t$ = next token.
- $\hat A_t$ = reward + KL-penalty + GAE bootstrap.
- The PPO clip is the **inner-loop** trust region; the KL term in the objective is the **outer-loop** constraint.

> **Saying it out loud.** The problem is that the response is sampled, so you can't just differentiate through it — you need a policy gradient. PPO's version says: compute the ratio of the new policy's probability to the old one, multiply by how good the outcome was, and clip that ratio to roughly 0.8 to 1.2 so no single update can lurch too far. There are two safety mechanisms and people mix them up: the clip is the per-update trust region, the KL term is the overall leash back to the SFT model. In LLM RLHF the state is the prompt plus everything generated so far, and the action is the next token.

### Why RLHF is hard to make work

- **PPO instability.** Loss spikes, value function divergence, KL blowup.
- **Reward model overfitting.** As the policy drifts, it goes off-distribution from the reward model's training data, where the reward model is unreliable.
- **Memory cost.** Need policy, reference policy, reward model, value function — 4 models in memory.
- **Sample efficiency.** Each PPO update needs fresh rollouts, which are expensive.

These problems motivated DPO and the alphabet soup of methods that followed.

> **Saying it out loud.** Honestly, the reason RLHF has a reputation for being painful is mostly engineering, not theory. You're holding four models in memory at once — policy, frozen reference, reward model, and value network — and you have to generate fresh samples for every update, so it's slow and expensive. On top of that PPO is genuinely temperamental: the value function diverges, KL spikes, loss blows up. That combination of four-models-in-memory plus instability is exactly what DPO was invented to avoid.

---

## 5. DPO: the elegant collapse

> **In plain language.** DPO's claim is that you never needed the reward model or the RL loop at all. If you solve the RLHF objective on paper, the optimal policy has a closed form, and you can rearrange it so the reward is just the log-ratio of your model to the frozen starting model. Plug that back into the preference formula and you're left with an ordinary supervised loss on (prompt, better answer, worse answer) triples.

Direct Preference Optimization (Rafailov et al. 2023) is the most-asked alignment paper of the past two years. The trick is to derive a closed-form optimal policy from the RLHF objective and use it as a loss directly, removing the need for a reward model and PPO.

### The derivation (whiteboard-ready)

> **In plain language.** Three moves: solve the KL-constrained reward objective in closed form, take logs and solve for the reward, then substitute into Bradley-Terry. The magic step is that the ugly normalising constant $Z(x)$ appears in both responses to the same prompt, so it subtracts away. Practise this on a whiteboard — interviewers ask for exactly these three moves.

Start with the RLHF objective for a single prompt-response pair:

$$
\max_\pi\ \mathbb{E}_y[r(x, y)] - \beta\, \mathrm{KL}\!\big(\pi \,\|\, \pi_{\text{ref}}\big)
$$

The closed-form solution to this constrained optimization is:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\left(\frac{r(x, y)}{\beta}\right)
$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\, \exp(r(x, y) / \beta)$ is the partition function. Take log of both sides:

$$
\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{r(x, y)}{\beta} - \log Z(x)
$$

$$
\Longrightarrow\quad r(x, y) = \beta \big[\log \pi^*(y \mid x) - \log \pi_{\text{ref}}(y \mid x)\big] + \beta \log Z(x)
$$

So the reward is the **log-ratio between the optimal policy and the reference**, plus a prompt-dependent constant. Now substitute this expression for $r$ into the Bradley-Terry preference model:

$$
P(y_w \succ y_l \mid x) = \sigma\!\Big(\beta \big[\log \pi^*(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)\big] - \beta \big[\log \pi^*(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)\big]\Big)
$$

The $\log Z(x)$ terms cancel! And we get the DPO loss by treating the trainable policy $\pi_\theta$ as the optimum $\pi^*$:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right) \right]
$$

That is the entire algorithm. **No reward model, no PPO, no rollouts. Just a supervised loss on preference pairs.**

> **Saying it out loud.** Here's the derivation in four beats. One: the KL-constrained reward objective has a closed-form optimum — the reference policy reweighted by the exponentiated reward. Two: take logs and rearrange, and the reward equals beta times the log-ratio of optimal policy to reference, plus a constant per prompt. Three: drop that into Bradley-Terry, and because both responses share a prompt, the partition function $Z(x)$ cancels. Four: what's left is a logistic loss on the log-ratio gap, which is DPO — no reward model, no rollouts, and one model in memory instead of four.

### What DPO is doing intuitively

The model implicitly defines a reward as $\beta \log(\pi_\theta / \pi_{\text{ref}})$. Higher policy probability than reference for $y_w$ and lower for $y_l$ increases the implicit reward gap, which Bradley-Terry says should match the preference probability. DPO trains the model to match preferences while implicitly defining its own reward function.

> **Saying it out loud.** The intuition is that the model *is* its own reward model. Whenever it assigns a response more probability than the frozen reference does, that's positive implicit reward; less probability, negative. So training just widens that gap in favour of the preferred answer. The thing to flag is that this is a relative signal, not an absolute one — a common gotcha is that DPO can push the chosen response's absolute log-probability *down*, as long as it pushes the rejected one down faster.

### Why DPO won (in practice)

- **Simpler.** One model in memory at a time (vs PPO's four).
- **More stable.** Supervised loss, no policy-gradient variance.
- **Comparable quality.** Empirically matches PPO-RLHF on many benchmarks at a fraction of the cost.
- **Tunable via $\beta$.** Smaller $\beta$ = more aggressive preference fitting; larger $\beta$ = closer to SFT.

> **Saying it out loud.** DPO caught on because it's cheap and it doesn't blow up. Two models in memory instead of four, a plain supervised loss instead of policy-gradient variance, and on a lot of chat benchmarks it lands close to PPO-RLHF for a fraction of the wallclock. Beta plays the same role it did in RLHF — small beta fits preferences hard, large beta keeps you near SFT. I'd add the caveat that "DPO matches PPO" is genuinely contested, not settled — see the note below.

### Why DPO might lose

- **Off-policy.** DPO learns from a fixed preference dataset. PPO collects fresh on-policy rollouts and can keep adapting. For settings where the policy drifts substantially from the data distribution (long training, exploratory tasks), PPO has an advantage.
- **No reward shaping.** The reward function is implicit. You can't inject inductive biases like "prefer safer outputs" via reward design.
- **Length bias.** DPO is empirically prone to producing longer outputs than necessary because longer responses can have higher implicit reward gaps. Mitigations: length-normalized DPO, SimPO.

> **Contested, not settled.** Whether offline DPO actually matches online RLHF is an open argument, and you should present it that way in an interview. Several 2024 papers (notably Tajwar et al., "Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data", and Xu et al., "Is DPO Superior to PPO for LLM Alignment?") report that a well-tuned PPO beats DPO on harder distributions, while the original DPO paper and many open recipes report parity. The honest summary: on chat-style preference data with one pass, DPO is competitive and far cheaper; on verifiable or exploratory tasks where the policy moves far from the preference data, on-policy methods still win. That is also why iterative/online DPO exists — it is an attempt to buy back the on-policy benefit without paying for PPO.

> **Saying it out loud.** The case against DPO is that it's off-policy: it learns from a frozen pile of comparisons, so as the model drifts away from whatever generated that data, the signal gets staler. You also lose the ability to shape reward — there's no place to hand-code "be a bit safer here," because the reward is implicit. And the named failure mode is length bias: because the implicit reward sums log-probs over tokens, longer answers can win for free, which is exactly what SimPO's length normalisation is there to fix.

### DPO vs RLHF, side by side

| Aspect | RLHF | DPO |
|---|---|---|
| Reward model | Required | Implicit |
| Optimization | PPO (RL) | Supervised |
| Models in memory | Policy + ref + RM + value | Policy + ref |
| Rollouts | Yes | No (uses pre-collected pairs) |
| Stability | Low–medium | High |
| Wallclock cost | High | Low |
| Theoretical equivalence | RLHF objective with optimal RM | Equivalent under regularity assumptions |
| Length bias | Less pronounced | More pronounced |

> **Saying it out loud.** If someone asks me to compare them in one breath: RLHF is on-policy, expensive, and expressive; DPO is off-policy, cheap, and stable. RLHF needs four models resident and fresh rollouts every step; DPO needs two models and a static dataset. They're provably the same objective under the idealised assumptions, but they diverge in practice the moment your reward model is imperfect or your policy drifts. Pick DPO when you have good preference pairs and limited compute; pick PPO or GRPO when the reward is verifiable and you can afford to generate.

---

## 6. The alphabet soup of preference methods

Each addresses a specific weakness of RLHF or DPO.

### IPO (Identity Preference Optimization, Azar et al. 2023)

> **In plain language.** IPO is DPO with a bounded loss so it can't run away. DPO's sigmoid keeps rewarding you for widening the gap between winner and loser, forever; IPO swaps in a squared loss with a target gap, so once you're far enough apart the pressure stops.

**Problem with DPO:** if the preference dataset has near-deterministic preferences ($P \approx 1$ for $y_w \succ y_l$), DPO's loss can drive the implicit reward gap to infinity. Manifests as DPO overfitting on certain examples.

**Fix:** replace the Bradley-Terry sigmoid with a squared loss directly on the implicit reward gap (Azar et al. 2023, Eq. 17):

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E}\!\left[\left(h_{\pi_\theta}(y_w, y_l) - \tfrac{\tau^{-1}}{2}\right)^2\right]
$$

where $\tau$ is the regularization strength (analog of $\beta$, but conceptually the inverse-temperature of the regularizer rather than a scaling on the gap), and

$$
h_{\pi_\theta}(y_w, y_l) = \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
$$

Bounded loss, more robust to label noise.

> **Saying it out loud.** IPO fixes a specific way DPO overfits. If your labellers were unanimous on a pair — everyone says A beats B — DPO's sigmoid never saturates, so it keeps pushing the gap toward infinity and the model contorts itself over a handful of examples. IPO replaces the sigmoid with a squared loss around a target margin, so once the gap is big enough, the gradient goes to zero. The tradeoff is that you're trading some peak preference accuracy for robustness to noisy or deterministic labels.

### KTO (Kahneman-Tversky Optimization, Ethayarajh et al. 2024)

**Problem:** DPO requires *paired* preferences. Hard to collect at scale. Most real-world feedback is just "this response was good" or "this was bad" — unpaired.

**Fix:** an objective inspired by Kahneman-Tversky prospect theory. Loss is asymmetric — penalize bad outputs more than reward good ones. Works with unpaired (just thumbs-up/thumbs-down) data.

> **Saying it out loud.** KTO exists because paired data is expensive and real feedback isn't paired. In production what you actually get is a thumbs-up or a thumbs-down on one response, not "here are two answers, rank them." KTO takes that unpaired signal and borrows prospect theory: losses hurt more than equivalent gains, so a thumbs-down pushes harder than a thumbs-up pulls. The practical tradeoff is that you can now use orders of magnitude more feedback, but you need to watch the desirable-to-undesirable ratio, because the asymmetry makes it sensitive to class imbalance.

### ORPO (Odds-Ratio Preference Optimization, Hong et al. 2024)

**Problem:** SFT and DPO are sequential. Can we combine them?

**Fix:** add a term to SFT loss that penalizes the probability of disliked responses:

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \log \sigma\!\big(\mathrm{odds\_ratio}(y_w, y_l)\big)
$$

Trains SFT and preference-fitting in one stage. Faster and competitive.

> **Saying it out loud.** ORPO's pitch is: why run two training stages when you can run one? It takes the ordinary SFT loss on the good response and bolts on an odds-ratio penalty that pushes down the bad one, so the model learns format and preference at the same time. You also drop the reference model, which saves memory. The tradeoff is control — with SFT and DPO separate you can tune them independently, and with ORPO you're stuck balancing one lambda.

### SimPO (Meng et al. 2024)

> **In plain language.** SimPO divides the log-probability by response length so long answers stop scoring higher just for being long, and it throws away the reference model entirely to halve memory. The $\gamma$ term is a fixed margin the winner has to beat the loser by.

**Problem:** DPO's reward is $\log(\pi_\theta / \pi_{\text{ref}})$, which biases toward longer responses (more terms in the sum).

**Fix:** length-normalize the implicit reward and remove the reference policy:

$$
\mathcal{L}_{\text{SimPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma\right)\right]
$$

Removes length bias; doesn't require keeping a reference model.

> **Saying it out loud.** SimPO makes two changes to DPO and both are about cost. First, divide each response's log-probability by its length, so a long rambling answer doesn't get a bigger implicit reward just for having more tokens. Second, delete the reference model entirely and add a fixed margin instead — that's one fewer model in memory. It fixes the length-bias failure mode, but you lose the KL anchor, so nothing is holding the policy near the SFT model anymore and you have to watch for capability drift.

### GRPO (Group Relative Policy Optimization, Shao et al. 2024 — introduced in DeepSeekMath, popularized in DeepSeek-R1)

**Problem:** PPO requires a value function (a separately trained critic), which is expensive and hard to stabilize for LLMs.

**Fix:** estimate the advantage from a *group* of $K$ rollouts of the same prompt instead of using a learned value function. Advantage = reward − group mean (normalized by group std).

$$
\hat A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}}}
$$

Used in DeepSeek-R1's RL pipeline. Particularly suited to verifiable-reward settings (math, code) where you can sample many candidates and grade them deterministically. Avoids the value-function instability of PPO at the cost of more rollouts per prompt.

> **Saying it out loud.** GRPO is PPO with the value network deleted. The reason PPO needs a critic is to answer "was this outcome better than expected?" — GRPO answers that empirically instead: sample $K$ answers to the *same prompt*, grade them all, and each answer's advantage is its reward minus the group mean, divided by the group's standard deviation. Note that "group" means a group of completions for one prompt — it has nothing to do with groups of people. The tradeoff is clean: you drop a whole model and its instability, and you pay for it in rollouts, which is why GRPO shines exactly where grading is cheap and deterministic, like math and code.

### GRPO objective (the full picture)

> **In plain language.** "Group" here means: take one prompt, sample $K$ answers from the current model, grade all $K$, and score each answer against the average of its own siblings. That average is the baseline a value network would have predicted — you just measure it instead of learning it. The rest of the formula is plain PPO clipping plus a KL leash.

Combine the group-relative advantage with PPO-style clipping and a per-token KL anchor:

$$
\mathcal{L}_{\mathrm{GRPO}} = -\mathbb{E}\!\left[\frac{1}{|y|}\sum_t \min\!\big(\rho_t \hat A,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat A\big)\right] + \beta\, \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})
$$

where $\rho_t = \pi_\theta(y_t|x, y_{<t}) / \pi_{\mathrm{old}}(y_t|x, y_{<t})$ and $\hat A = (r - \mu_{\mathrm{group}})/\sigma_{\mathrm{group}}$ is shared across all tokens of a sample. Each prompt $x$: sample $K$ rollouts → compute group advantages → PPO-clipped policy update + KL anchor.

> **Saying it out loud.** Walking the full objective: for each prompt you sample a group of completions, score them, z-score the scores within that group to get one advantage per completion, then hand that single number to every token in that completion. From there it's PPO — importance ratio, clip, take the min — plus a KL term back to the reference. The subtlety interviewers probe is that the advantage is constant across the sequence, so every token in a correct answer gets credited equally, even the filler ones. That's crude credit assignment, and it's exactly what token-level variants and process supervision try to sharpen.

---

## 7. The 2024-2025 frontier: post-GRPO methods

GRPO unlocked verifiable-reward RL at scale (DeepSeek-R1). The follow-up wave fixes its weaknesses.

### DAPO (ByteDance/Seed 2025) — Decoupled Clip and Dynamic Sampling Policy Optimization

DAPO is the canonical fix-up to GRPO. Four named tricks:

1. **Clip-Higher**: separate upper and lower clip ranges, $\epsilon_{\mathrm{low}} < \epsilon_{\mathrm{high}}$. Prevents "entropy collapse" — without an asymmetric clip, low-probability tokens get clipped before they can explore enough.
2. **Dynamic Sampling**: drop prompts whose $K$ rollouts all succeeded or all failed (group has zero advantage variance — no learning signal). Re-sample harder prompts.
3. **Token-level policy gradient loss**: average the loss over **tokens** instead of over **samples** (so long responses don't get diluted). Fixes a length bias in vanilla GRPO.
4. **Overlong-reward shaping**: soft length penalty for responses near max length, instead of a hard truncation that gives noisy signal.

DAPO reproduced DeepSeek-R1-quality results on AIME and is the new default for verifiable-reward RL.

> **Saying it out loud.** DAPO is basically GRPO with four bug fixes, and they're worth memorising by name. Clip-higher makes the upper clip looser than the lower one so rare tokens don't get squashed before they can be explored — that's the entropy-collapse fix. Dynamic sampling throws away prompts where all $K$ rollouts passed or all failed, because a group with zero variance has zero advantage and contributes literally nothing. Then token-level loss averaging so long answers aren't diluted, and a soft length penalty instead of hard truncation. Together those reproduced R1-level AIME results in the open.

### Dr. GRPO (Liu et al. 2025) — bias-free GRPO

> **In plain language.** Dr. GRPO says two of GRPO's normalisations quietly distort the gradient: dividing by the group's standard deviation, and averaging over tokens. Drop both — subtract the group mean only, and sum over tokens — and the estimator becomes unbiased.

Identifies two biases in GRPO: (a) the per-token mean reduction implicitly weights longer responses *less*; (b) the std normalization $\sigma_{\mathrm{group}}$ encourages over-confident easy prompts. Both fixed by **dropping** the std normalization and switching to a sum-not-mean over tokens:

$$
\hat A_i = r_i - \mu_{\mathrm{group}} \quad\text{(no }\sigma\text{)}; \qquad \mathcal{L} = -\sum_t \min(\rho_t \hat A,\ \mathrm{clip}(\cdot)\hat A)
$$

Cleaner, slightly better empirically.

> **Saying it out loud.** Dr. GRPO is the "your estimator is biased" paper. Dividing by the group standard deviation sounds harmless, but it inflates gradients on easy prompts where every rollout agrees and the std is tiny, and averaging the loss per token systematically under-weights long responses. Fix is one line each: subtract the group mean without dividing, and sum over tokens instead of averaging. Small change, and it removes a bias that was quietly favouring short answers on easy problems.

### RLOO (REINFORCE Leave-One-Out, Ahmadian et al. 2024)

> **In plain language.** RLOO is the stripped-down version: no critic, no clipping, just "was this answer better than the average of its siblings?" Leaving the sample itself out of its own baseline is what keeps the estimate unbiased.

The simplest baseline among them. For $K$ rollouts per prompt: each sample's advantage is its reward minus the *mean of the other $K-1$*. No critic, no per-token complexity, no PPO clipping needed at small $K$:

$$
\hat A_i = r_i - \frac{1}{K-1}\sum_{j \neq i} r_j
$$

Surprisingly competitive with PPO/GRPO for RLHF in small-budget settings. Often used when the rollout budget per prompt is limited.

> **Saying it out loud.** RLOO is the "did we actually need all this machinery?" result. Sample a handful of answers per prompt, and score each one against the mean of its siblings *excluding itself* — that leave-one-out bit is what keeps the baseline unbiased. No critic, no clipping, no value function. It's surprisingly competitive with PPO for RLHF when your rollout budget is small, say $K$ of 2 to 4, and it degrades relative to GRPO/DAPO when you scale rollouts way up.

### REINFORCE++ (2024 community recipes)

Plain REINFORCE with: (a) reward whitening, (b) baseline subtraction, (c) gradient clipping, (d) careful KL anchor. Demonstrates that "vanilla" REINFORCE — once you tune it — can match PPO/GRPO without the PPO complexity. Used in some Tülu / Ai2 recipes.

> **Saying it out loud.** REINFORCE++ is the reminder that most of PPO's benefit comes from variance reduction, not from the clipping. Take plain REINFORCE, whiten the rewards, subtract a baseline, clip the gradients, keep a KL anchor — and you get PPO-class results with far less code. It's used in some of the Ai2 Tülu recipes. The tradeoff is that you're now hand-tuning the variance reduction yourself instead of getting it from a critic, so it's less forgiving if your reward scale shifts mid-run.

### RLVR (RL with Verifiable Rewards) — the unifying framework

Less an algorithm than a setup. Instead of a learned reward model, the reward is a *programmatic verifier*:

- Math: did the final answer match the gold answer?
- Code: did the unit tests pass?
- Format: is the output valid JSON, did it call the right tools, etc.?

Eliminates reward hacking (you can't game an exact-match check) and reward-model drift. Combined with GRPO/DAPO it's the recipe behind o1, R1, Qwen-QwQ, and frontier reasoning systems. **The dominant alignment paradigm for capability-pushing tasks in 2025.**

> **Saying it out loud.** RLVR isn't really an algorithm, it's a decision about where reward comes from. Instead of a learned reward model that can be gamed, you use a program: did the answer match the gold answer, did the unit tests pass, is the JSON valid. That kills reward hacking and reward-model drift in one move, because there's no approximation left to exploit. The limitation is scope — you can verify math and code, you can't verify "was this a kind and helpful reply," so RLVR pushes capability while preference methods still carry helpfulness and harmlessness.

### TDPO / Token-level DPO (Zeng et al. 2024)

> **In plain language.** DPO hands one verdict to a whole paragraph, so every token in a rejected answer gets pushed down even the ones that were fine. TDPO spreads the credit across tokens with a per-token KL term, which matters most when the winner and loser differ by only a word or two.

DPO assigns a single preference label to an entire response. TDPO breaks this down to per-token:

$$
\mathcal{L}_{\mathrm{TDPO}} = -\log \sigma\!\big(u(x, y_w) - u(x, y_l) - \delta(x, y_w, y_l)\big)
$$

with $u$ a sequence-level utility and $\delta$ a per-token KL regularization term. Reduces DPO's "all-or-nothing" attribution problem and improves stability when responses differ in only a few tokens.

> **Saying it out loud.** TDPO attacks DPO's credit-assignment problem. If a rejected answer was 200 tokens and only 5 of them were bad, DPO still pushes down all 200, which drags perfectly good text with it. TDPO adds a per-token KL term so the pressure concentrates where the two responses actually diverge. It's most valuable when your pairs are near-identical edits, and it costs you some of DPO's simplicity.

### Step-DPO (Lai et al. 2024) — process-level preferences

For reasoning tasks: collect preference pairs at the *step* level instead of the response level. The policy learns "this reasoning step was better than that one" rather than "this whole answer was better." Combines the simplicity of DPO with process-supervision-style density. Strong on math.

> **Saying it out loud.** Step-DPO applies the same idea to reasoning chains. Instead of "this whole solution was better," you label "this step was better than that step," which tells the model exactly where the derivation went wrong. Think of it as marking the line in a proof where the error happened rather than just stamping the whole page. It's strong on math, and the tradeoff is labelling cost — step-level annotations are much more expensive than outcome labels.

### Self-Rewarding LMs (Yuan et al. 2024)

The model is *both* policy and judge. Iteratively:
1. Generate responses to prompts.
2. Self-score them using the model's LLM-judge ability (built via SFT).
3. Form preference pairs from the scores; apply DPO.
4. Loop.

Each iteration improves both the policy and the judge. Removes external reward models from the loop. Risk: judge biases compound across iterations.

> **Saying it out loud.** Self-rewarding models make the model its own grader: generate answers, score them with the same model acting as judge, turn the scores into preference pairs, run DPO, repeat. The appeal is that as the policy gets better the judge gets better too, so the ceiling isn't fixed by a frozen reward model. The obvious danger is that a judge grading its own work amplifies its own blind spots — biases compound with each round, so in practice you cap the number of iterations and periodically check against human or external judges.

### SPIN (Self-Play Fine-tuning, Chen et al. 2024)

Treats SFT data as the "expert" and the model's current generations as the "learner." Train the model to distinguish (and prefer) expert responses over its own current generations via a DPO-like loss. Iterates: each round, the gap shrinks. Pure self-improvement from existing SFT data — no new labels required.

> **Saying it out loud.** SPIN squeezes more out of SFT data you already have. Each round you treat the human-written responses as the winners and the model's own current generations as the losers, and train with a DPO-style loss to prefer the human ones. It's self-play: as the model improves, its own outputs become harder negatives. The natural stopping point is when the model's generations are indistinguishable from the SFT data — after that there's no signal left, and no amount of extra rounds helps.

### Iterative / Online DPO (Tülu 2/3, Llama 3 paper)

DPO is off-policy. To approximate online RL benefits without full PPO: alternate (a) sample fresh responses from the current policy, (b) judge them (human or AI), (c) form new preference pairs, (d) apply DPO again. Multiple rounds bridge the on-policy gap. Used in production Tülu and Llama recipes.

> **Saying it out loud.** Iterative DPO is the pragmatic middle ground and it's what most production recipes actually do. Run DPO, then generate fresh responses from the *new* model, have a judge rank those, and run DPO again on the new pairs. Each round re-centres the training data on where the policy currently lives, which is the main thing plain DPO lacks. Llama 3 and Tülu both do a small number of rounds — typically 2 or 3 — because the gain per round drops off fast and each round costs a full generation-plus-judging pass.

### NLHF (Nash Learning from Human Feedback, Munos et al. 2024)

Game-theoretic reframing. Human preferences may not satisfy Bradley-Terry transitivity. NLHF instead seeks the **Nash equilibrium of the preference game**: the policy is one whose generations beat any other policy's generations on average. Yields an algorithm based on regret minimization (mirror descent on the preference operator). Less common in practice; theoretically more principled when preferences are intransitive.

> **Saying it out loud.** NLHF drops the assumption that preferences are consistent. Bradley-Terry needs transitivity — if you like A over B and B over C, you must like A over C — and real human preferences break that all the time, the same way rock-paper-scissors has no best move. So instead of fitting a single score, NLHF looks for the policy that wins on average against every other policy, which is a Nash equilibrium of the preference game. It's more principled when preferences are intransitive, but it's rare in production because it's harder to implement and the gains over plain RLHF haven't been large.

### Inference-time alignment: Best-of-N

The simplest "alignment" doesn't change the policy at all: at inference, sample $N$ responses, score them with a reward model or verifier, return the highest-scoring. Trade compute for quality. Used in production for verifiable-reward tasks (e.g., math: sample 16 solutions, return the one that compiles + matches expected). Often combined with RL: the RL policy already concentrates probability on good responses; Best-of-N polishes the tail.

> **Saying it out loud.** Best-of-N doesn't touch the weights at all — you sample $N$ answers at inference, score them, and return the best one. It's the cheapest possible alignment in engineering terms and the most expensive in serving terms, since $N$ of 16 means 16 times the generation cost. It's a strong baseline that people forget to mention, and it stacks on top of RL. The failure mode to name is that it inherits reward-model overoptimisation: crank $N$ high enough and you're just searching for the reward model's blind spots, which is why quality often peaks and then declines past a few hundred samples.

### When to use which (the 2025 decision tree)

| Setting | Method |
|---|---|
| Verifiable reward (math/code) + plenty of compute | DAPO or Dr. GRPO with RLVR |
| Verifiable reward + tight compute budget | RLOO with verifier |
| Preference pairs available, single iteration | DPO or SimPO |
| Preference pairs + iterating allowed | Iterative / online DPO |
| Unpaired thumbs-up/down feedback | KTO |
| One-stage SFT + preference learning | ORPO |
| Need fine-grained step-level signal | Step-DPO or process-supervised RL |
| Trust the model as judge | Self-Rewarding |
| No new labels, pure self-improvement | SPIN |
| Inference-time only, no training | Best-of-N + reward / verifier |

> **Saying it out loud.** If someone asks "which one would you use," the honest answer is that it's decided by two things: can you verify the answer, and how much compute do you have. Verifiable reward plus real compute means GRPO-family with RLVR — DAPO or Dr. GRPO today. Verifiable but tight budget means RLOO. If all you have is preference pairs, it's DPO or SimPO for one pass, iterative DPO if you can afford rounds, and KTO if your feedback is unpaired thumbs-up/down. And if you can't train at all, Best-of-N with a verifier gets you surprisingly far.

---

## 8. Constitutional AI and RLAIF

### Constitutional AI (Bai et al. 2022, Anthropic)

Replace human preferences for harmlessness with **AI-generated** critiques and rewrites guided by a written "constitution" (a list of principles). Process:

1. Generate a response to a potentially harmful prompt.
2. Use the model to critique its own response against constitutional principles.
3. Rewrite the response per the critique.
4. Use these (original, rewritten) pairs as preference data.

Then RLHF/DPO on the AI-generated pairs. Reduces dependence on expensive human red-teaming for harmlessness.

> **Saying it out loud.** Constitutional AI replaces human harmlessness labellers with the model critiquing itself against a written set of principles. The loop is: produce a response, ask the model to criticise it against the constitution, ask it to rewrite, and use the original-versus-rewrite pair as preference data. It works because critiquing is easier than generating — the model can spot a problem it wouldn't have avoided on its own. The big win is that harmlessness data no longer requires exposing human red-teamers to a stream of awful content, and the principles are written down where anyone can audit them.

### RLAIF (RL from AI Feedback)

Generalizes Constitutional AI: use an AI judge (often a stronger model) instead of humans to generate preference labels. Cheaper, more scalable. Works as long as the judge model is sufficiently capable; gets risky when the judge has the same biases as the policy.

> **Saying it out loud.** RLAIF is just RLHF with the human replaced by a model judge. It's dramatically cheaper and scales to millions of comparisons, and on helpfulness it lands close to human-labelled RLHF. The condition is that the judge has to be at least as good as the labellers at the thing you're judging. The failure mode to name is correlated bias: if the judge and the policy come from the same family, they share blind spots, and you'll happily optimise into them — that's why people keep a human-labelled eval set outside the loop.

### Process supervision vs outcome supervision (Lightman et al., OpenAI)

For reasoning tasks: do you reward the final answer or the reasoning steps?

- **Outcome supervision:** reward only based on final answer correctness. Easy but sparse — the model must figure out which intermediate steps mattered.
- **Process supervision:** reward correctness of each reasoning step. Denser signal; better for math/logic. Requires step-level labels (expensive).

OpenAI's PRM800K showed process supervision substantially outperforms outcome supervision for math reasoning. This is a major topic in interview questions about reasoning models.

> **Saying it out loud.** The question is whether you grade the final answer or the working. Outcome supervision is cheap — just check the answer — but the signal is sparse, so a model that got the right answer through two errors that cancelled gets rewarded for both. Process supervision grades each step, which is much denser and stops that, but it needs step-level human labels. OpenAI's "Let's Verify Step by Step" with PRM800K is the number to cite: process supervision clearly beat outcome supervision on MATH, and that's the intellectual ancestor of today's reasoning models.

---

## 9. Failure modes and how to detect them

### Reward hacking / specification gaming

The policy finds outputs that score high under the reward model but aren't actually good. Examples:

- Length bias (longer answers score higher because reward model trained on longer-is-better).
- Style mimicry (responses that sound authoritative regardless of accuracy).
- Repetition (some reward models score highly when key phrases recur).

**Detection:** compare RL-policy outputs to SFT outputs on held-out prompts. Use a different (held-out) reward model to grade. Have humans grade — if RM and humans disagree, something's wrong.

> **Saying it out loud.** Reward hacking is when the model gets great at the score and worse at the job. Classic examples: it learns longer answers score higher, or that confident-sounding phrasing scores higher regardless of whether it's right, or it repeats a phrase the reward model happens to like. The tell is a widening gap between reward-model score and human judgement — reward keeps climbing while win-rate flattens or drops. That's Goodhart's law, and the standard detection is to grade with a held-out reward model the policy never trained against.

### KL divergence blowup

The policy moves too far from $\pi_{\text{ref}}$. Symptoms: gibberish outputs, mode collapse on a few responses, capability loss. Often a sign of $\beta$ too small or RM exploiting OOD regions.

**Detection:** monitor $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ on a held-out prompt set. Should stay below ~10 for healthy RLHF; KL > 30 usually means trouble.

> **Saying it out loud.** KL blowup is the policy slipping the leash. You see it as degraded language, weirdly repetitive phrasing, and capability loss on things the reward never covered. Usually it means beta is too small, or the reward model is handing out high scores in a region it was never trained on. The practical thing to say is that you monitor KL to the reference on a fixed held-out prompt set every run — under about 10 nats is healthy, past 30 you're almost certainly broken and should raise beta or stop early.

### Mode collapse

The policy collapses onto a few responses regardless of prompt. Symptoms: low entropy, similar outputs across diverse prompts.

**Detection:** measure response entropy and diversity. Compare to SFT baseline. If entropy halves while reward goes up, you may be in mode collapse.

> **Saying it out loud.** Mode collapse is when the model finds one answer the reward model loves and starts giving you versions of it no matter what you asked. The reward curve looks great, because it genuinely is scoring well — it's just stopped being a distribution. The way you catch it is by tracking output entropy and diversity alongside reward, not reward alone. Rule of thumb: if entropy drops sharply — say it halves — while reward rises, you're collapsing, and the fixes are a stronger KL anchor or an explicit entropy bonus.

### Sycophancy

The policy learns to agree with whatever the user implies. Symptoms: contradictory answers depending on phrasing of the question.

**Detection:** evaluate on prompts that bias toward wrong answers (e.g. "The capital of Australia is Sydney, right?"). If the model agrees, sycophancy is creeping in.

> **Saying it out loud.** Sycophancy is the model learning that agreeing with you scores better than being right. It comes straight from the preference data — human raters reward answers that validate them, so the reward model does too. You spot it by asking the same factual question two ways, one of them leading: "The capital of Australia is Sydney, right?" If the answer flips with the framing, you have it. It's a good example of the reward model faithfully learning something real about human preference that you nonetheless don't want.

### Overoptimization / reward model drift

The longer you train against a reward model, the further the policy goes off the RM's training distribution, and the less the RM is reliable. Eventually true reward (human evaluation) starts decreasing even as RM-reward keeps going up — this gap is the hallmark of overoptimization.

**Detection:** Goodhart curves — plot human win-rate against KL distance from $\pi_{\text{ref}}$. There's typically a sweet spot.

> **Saying it out loud.** Overoptimisation is the general form of all of this. The longer you train, the further the policy drifts from where the reward model was fitted, and the less that reward model knows what it's talking about. So proxy reward keeps rising while true quality peaks and then falls. The picture to draw is the Goodhart curve from Gao et al. — human win-rate on the y-axis against square-root KL on the x-axis — and it's a hump, not a line. The practical consequence is that early stopping on the KL budget is a real technique, not a hack.

---

## 10. The KL anchor: more than a regularization term

Why is KL regularization to $\pi_{\text{ref}}$ so important?

**Capability preservation.** Pretraining gave the model broad capabilities. The preference dataset only covers a narrow slice of behaviors. Without the KL anchor, the policy "forgets" capabilities that aren't being rewarded — including the ones humans aren't currently asking about but expect to work.

**OOD robustness.** Reward models are unreliable far from their training distribution. The KL anchor keeps the policy in distribution where the RM can be trusted.

**Calibration.** The pretrained policy already has well-calibrated probabilities. Aggressive RL can destroy calibration by sharpening the distribution. KL anchor preserves it.

The KL coefficient $\beta$ is the master knob in RLHF. Tuning it well is most of the work in stabilizing an RL run.

**Interview question:** "What does $\beta$ control in RLHF?" The trade-off between matching the reward (low $\beta$) and staying close to the SFT model (high $\beta$). Empirically, $\beta = 0.01$ to $0.1$ works well for most settings.

> **Saying it out loud.** People call the KL term regularisation, but it's doing three jobs. It preserves capabilities the preference data never covers, so you don't lose coding ability while optimising for politeness. It keeps you inside the region where the reward model was actually trained and is therefore trustworthy. And it protects calibration, because aggressive RL sharpens the distribution and makes models overconfident. Beta is the master knob — 0.01 to 0.1 in most recipes — and tuning it is genuinely most of the work in getting an RL run to behave.

---

## 11. Reward model design subtleties

### Bias from labelers

The RM inherits biases from the humans who provided preferences. Length bias, formatting bias, certainty bias (humans prefer confident-sounding responses regardless of correctness), and authority bias all show up.

**Mitigation:** diverse labeling pool, calibration training, residual debias techniques.

> **Saying it out loud.** A reward model is a compression of whoever labelled the data, biases included. The ones that show up reliably are length bias, formatting bias — markdown and bullet points score well — and certainty bias, where confident phrasing beats hedged phrasing even when the hedged answer is more accurate. Since the policy optimises against the reward model, any bias in the labellers gets amplified, not averaged out. The mitigations are a diverse labelling pool, explicit debiasing terms, and measuring length correlation on your preference set before you train on it.

### Single RM vs ensemble

A single RM can be exploited by the policy. Ensembles of RMs give a distribution of rewards; the policy is less able to find adversarial responses that fool all of them.

**Mitigation:** train multiple RMs with different initializations / architectures / data subsets. Use mean (or worst) of ensemble as reward. Or use uncertainty estimates: penalize responses where RM ensemble disagrees.

> **Saying it out loud.** One reward model has exactly one set of blind spots, and gradient descent is very good at finding them. An ensemble helps because the policy now has to fool several models that were initialised differently and saw different data, and their disagreement is itself a useful signal — where they disagree, you're off-distribution. You can take the mean for a normal signal or the minimum if you want to be conservative. The tradeoff is straightforward cost: $N$ reward models is $N$ times the inference during rollout scoring, so people usually cap it around 3 to 5.

### Reward shaping

Adding auxiliary terms to the reward beyond the learned RM:

- **Length penalty.** Subtract a term for very long responses to combat length bias.
- **Repetition penalty.** Penalize repeated n-grams.
- **Refusal correction.** Reward correct refusals on harmful prompts; penalize incorrect refusals on benign ones.

Reward shaping is part art, part craft. Must be done carefully to avoid introducing new failure modes.

> **Saying it out loud.** Reward shaping is where you hand-add terms the learned reward model can't express — a length penalty, a repetition penalty, a bonus for correctly refusing a harmful prompt and a penalty for refusing a benign one. It's the one place in the pipeline where you get direct control over behaviour. The warning is that every shaping term is a new thing to game: add a hard length penalty and the model gives clipped, unhelpful answers instead of concise ones. So you shape lightly and re-check the failure modes after each term you add.

### Outcome vs preference reward

- **Outcome rewards** (math correctness, code passing tests): verifiable, no RM needed.
- **Preference rewards** (helpfulness, harmlessness): need RM trained on human comparisons.

Frontier reasoning systems (o1, DeepSeek-R1) increasingly use outcome rewards on verifiable tasks, which avoids reward model issues entirely. This is a major shift from the "RLHF for everything" era.

> **Saying it out loud.** There are really two kinds of reward and they have different problems. Verifiable outcome rewards — the tests pass, the answer matches — can't be gamed and need no reward model, but they only exist for a narrow set of tasks. Learned preference rewards cover everything else, helpfulness, tone, harmlessness, but they're approximations and therefore hackable. The shift worth naming is that frontier reasoning systems moved capability training onto verifiable rewards and left preference models to handle style and safety, rather than doing RLHF for everything.

---

## 12. Online vs offline RL for alignment

**Online RL (PPO, GRPO).** Sample fresh rollouts from the current policy. Each gradient step uses on-policy data. More expensive per step but adapts to policy drift. Standard for RLHF.

**Offline RL (DPO, IPO, KTO).** Use a fixed preference dataset. No rollouts. Cheaper per step but suffers if the policy drifts far from the data distribution.

**Iterated offline (Tülu 2, Llama 3 paper).** Apply DPO. Sample new responses from the updated policy. Have a judge (human or AI) generate fresh preference pairs from the new responses. Apply DPO again. Bridge the on-policy gap without requiring full RL.

**Mixed.** Some recent recipes interleave SFT, DPO, and a small amount of online RL. The "best" stack is still being figured out at frontier labs.

> **Saying it out loud.** Online versus offline comes down to one question: does your training data come from the model you're currently training? Online — PPO, GRPO — generates fresh rollouts every step, so the data always matches the policy, and you pay for that in generation cost. Offline — DPO, IPO, KTO — reuses a fixed dataset, which is far cheaper but goes stale as the policy drifts away from it. The middle ground everyone actually ships is iterative offline: run DPO, regenerate, re-judge, repeat for a couple of rounds. Which end is better is still argued about, so I'd frame it as a compute-versus-drift tradeoff rather than a settled winner.

---

## 13. Evaluation: how do you know it worked?

Hardest part of post-training. Loss curves don't tell you the policy is good.

### Standard offline benchmarks

- **MMLU, GSM8K, HumanEval, MATH:** capability preservation. RL shouldn't tank these.
- **AlpacaEval 2, MT-Bench, Arena-Hard:** judge-based win-rate against a baseline (e.g. GPT-4). Common but susceptible to judge biases.

> **Saying it out loud.** Offline evals split into two jobs. Capability benchmarks like MMLU, GSM8K and HumanEval aren't there to show improvement — they're there to show you didn't break anything, so you want them flat within a couple of points. Then judge-based benchmarks like AlpacaEval 2 and Arena-Hard measure whether people prefer the new model. The caveat to name is that LLM judges have their own biases, especially toward length and toward their own family's style, which is why AlpacaEval 2 added a length-controlled score.

### Online human evaluations

- Side-by-side comparisons. Slow, expensive, gold standard.
- Engagement / preference data from production. High signal but lagging.

> **Saying it out loud.** Human side-by-side comparison is still the gold standard and everything else is a proxy for it. It's slow and expensive, so you use it to calibrate your cheap evals rather than to run every experiment. Production engagement data is high-signal but lagging and confounded — people click for reasons that aren't quality. The practical setup is: automatic evals for iteration speed, a periodic human eval to check that your automatic evals still correlate with what humans actually want.

### Capability vs alignment trade-offs

RL on harmlessness can cost capability (the "alignment tax"). Track both. The ideal post-training pipeline maintains MMLU/HumanEval/etc. within a few points of the SFT baseline while substantially improving preference-based metrics.

> **Saying it out loud.** The alignment tax is what you pay in raw capability for better behaviour. Training hard on harmlessness or on a narrow preference set makes the model refuse more and reason a bit worse, because the KL budget you spend on style is budget you're not spending on staying near the pretrained model. So you always report both axes — capability benchmarks and preference win-rate — never just the one that improved. A reasonable bar to state: preference metrics up substantially while MMLU and HumanEval stay within a couple of points of the SFT baseline.

### Calibration

A well-aligned model should know what it doesn't know. Test by asking factual questions and checking confidence calibration. Heavily-RLed models often become overconfident.

> **Saying it out loud.** Calibration is whether the model's confidence matches how often it's right — if it says 70% it should be right about 70% of the time. Base models are usually decently calibrated straight out of pretraining, and RLHF reliably wrecks it, because optimising for preference sharpens the distribution and humans prefer confident-sounding answers. That's a documented result in the GPT-4 report: calibration was good pre-RLHF and clearly worse after. So if you're building anything that routes on confidence, measure calibration before and after and expect to have to fix it.

---

## 14. Loss functions in code (whiteboardable in 5 min each)

You'll be asked to implement these. Below are minimal, idiomatic versions that fit on a whiteboard.

### SFT loss (next-token cross-entropy with prompt masking)

```python
def sft_loss(logits, labels, prompt_mask):
    """
    logits: [B, L, V]; labels: [B, L]; prompt_mask: [B, L] (1 = response, 0 = prompt).
    Loss is cross-entropy on response tokens only.
    """
    log_probs = F.log_softmax(logits, dim=-1)                # [B, L, V]
    nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, L]
    return (nll * prompt_mask).sum() / prompt_mask.sum()
```

### Reward model loss (Bradley-Terry NLL on preference pairs)

```python
def rm_loss(r_chosen, r_rejected):
    """r_chosen, r_rejected: [B] scalar rewards from the RM."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

### DPO loss

```python
def dpo_loss(logp_chosen, logp_rejected,
             logp_chosen_ref, logp_rejected_ref, beta=0.1):
    """
    logp_*: [B] sum of per-token log-probs of the chosen/rejected response under
            policy / reference. (Sum over response tokens only — prompt masked.)
    """
    pi_logratio  = logp_chosen      - logp_rejected
    ref_logratio = logp_chosen_ref  - logp_rejected_ref
    logits = beta * (pi_logratio - ref_logratio)              # implicit reward gap
    return -F.logsigmoid(logits).mean()
```

The whole DPO algorithm fits in 4 lines once you have log-probs.

### IPO loss (bounded variant)

```python
def ipo_loss(logp_chosen, logp_rejected,
             logp_chosen_ref, logp_rejected_ref, tau=0.1):
    h = (logp_chosen - logp_chosen_ref) - (logp_rejected - logp_rejected_ref)
    return ((h - 1.0 / (2 * tau)) ** 2).mean()    # squared loss → bounded
```

### SimPO loss (length-normalized, no reference)

```python
def simpo_loss(logp_chosen, logp_rejected, len_c, len_r, beta=2.0, gamma=1.0):
    """logp_*: sum of log-probs; len_*: response lengths."""
    margin = beta * (logp_chosen / len_c - logp_rejected / len_r) - gamma
    return -F.logsigmoid(margin).mean()
```

### KTO loss (unpaired thumbs-up/down)

```python
def kto_loss(logp, logp_ref, label, beta=0.1, lam_pos=1.0, lam_neg=1.0):
    """
    logp, logp_ref: [B] sequence log-probs under policy / reference.
    label: [B] in {+1, -1} for desirable / undesirable.
    Asymmetric: penalize undesirable harder than reward desirable (Kahneman-Tversky).
    """
    z = beta * (logp - logp_ref)                  # implicit reward
    KL = (logp - logp_ref).mean().detach()        # batch KL anchor
    pos = lam_pos * (1 - torch.sigmoid(z - KL))   # for +1 examples
    neg = lam_neg * (1 - torch.sigmoid(KL - z))   # for -1 examples
    return torch.where(label > 0, pos, neg).mean()
```

### PPO clipped surrogate (RLHF inner loop)

```python
def ppo_loss(logp_new, logp_old, advantages, eps=0.2):
    """
    logp_new, logp_old: [B, L] per-token log-probs from new / old policy.
    advantages: [B, L] from GAE on (reward - beta * KL_per_token).
    """
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return -torch.min(surr1, surr2).mean()         # negate for ascent
```

For the full RLHF inner loop you also need: token-level reward = $r(x,y)$ at end-of-sequence + $-\beta \log(\pi_\theta/\pi_{\mathrm{ref}})$ per token, then GAE.

### GRPO loss (DeepSeekMath, R1)

```python
def grpo_loss(logp_new, logp_old, rewards, eps=0.2, beta_kl=0.04, logp_ref=None):
    """
    logp_new, logp_old: [B, K, L] — K rollouts per prompt.
    rewards:            [B, K]    — scalar reward per rollout (e.g., 1 if math correct).
    logp_ref:           [B, K, L] — reference log-probs for KL anchor (optional).
    """
    # 1. Group-relative advantage (per-prompt z-score, then broadcast over tokens)
    mu    = rewards.mean(dim=1, keepdim=True)
    sigma = rewards.std(dim=1, keepdim=True) + 1e-8
    A     = ((rewards - mu) / sigma).unsqueeze(-1)            # [B, K, 1]

    # 2. PPO clip
    ratio = torch.exp(logp_new - logp_old)                    # [B, K, L]
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * A
    pg    = -torch.min(surr1, surr2).mean()

    # 3. KL anchor (token-level, to reference)
    if logp_ref is not None:
        kl = (logp_new - logp_ref).mean()
        return pg + beta_kl * kl
    return pg
```

### DAPO additions to GRPO (the four tricks)

```python
def dapo_loss(logp_new, logp_old, rewards, response_mask,
              eps_low=0.2, eps_high=0.28,                     # (1) Clip-Higher
              beta_kl=0.0):                                    # often 0 — KL handled via ref-policy snapshot
    """
    response_mask: [B, K, L] — 1 on response tokens (excluding prompt).
    Implements all four DAPO tricks. (Dynamic sampling is done at the data layer:
    drop prompts where rewards.std(dim=1) == 0.)
    """
    # Group-relative advantage
    mu   = rewards.mean(dim=1, keepdim=True)
    A    = (rewards - mu).unsqueeze(-1)                        # (Dr. GRPO style: no σ)

    # (1) Clip-Higher: asymmetric clip
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * A

    # (3) Token-level loss: average over tokens, not samples
    token_loss = -torch.min(surr1, surr2)                      # [B, K, L]
    return (token_loss * response_mask).sum() / response_mask.sum()
```

The 4 DAPO tricks in code:
1. **Clip-Higher** → asymmetric `eps_low`, `eps_high` (line 1).
2. **Dynamic sampling** → filter `rewards.std(dim=1) > 0` at data prep (not loss).
3. **Token-level loss** → `sum / sum` over response tokens (last line).
4. **Overlong-reward shaping** → applied to `rewards` before the loss (e.g., `r -= alpha * max(0, len - L_target)`).

### RLOO loss (REINFORCE Leave-One-Out)

```python
def rloo_loss(logp, rewards):
    """
    logp:    [B, K, L] — policy log-probs of each rollout.
    rewards: [B, K]    — scalar reward per rollout.
    """
    K = rewards.shape[1]
    # Each rollout's baseline = mean of OTHER K-1 rollouts
    sum_r = rewards.sum(dim=1, keepdim=True)                   # [B, 1]
    baseline = (sum_r - rewards) / (K - 1)                     # [B, K]
    A = (rewards - baseline).unsqueeze(-1)                     # [B, K, 1]
    return -(logp.sum(dim=-1) * A.squeeze(-1)).mean()          # REINFORCE with leave-one-out baseline
```

### Step-DPO loss (process-level)

Same as DPO but applied to (good_step, bad_step) pairs at each reasoning step instead of full responses. Implementation: split each response by step delimiter (e.g., newline), apply DPO loss per step pair, sum.

> **Saying it out loud.** If you're asked to code one of these on a whiteboard, the thing to internalise is that they're all a few lines once you have sequence log-probs. DPO is: subtract reference log-ratios, scale by beta, take `logsigmoid`, negate. GRPO is: z-score the rewards within each prompt's group, broadcast that scalar across the tokens, then PPO clip. The bug interviewers watch for is the masking — log-probs must be summed over response tokens only, never over the prompt, and forgetting that silently rewards the model for the user's own text.

---

## 15. The 12 most-asked alignment interview questions

(Brief answers; full grilling in `INTERVIEW_GRILL.md`.)

1. **Walk me through the RLHF pipeline.** SFT → reward model on preference pairs → PPO with KL penalty.
2. **Why do we need a KL penalty?** Bound reward hacking; preserve capabilities; stay in RM's training distribution.
3. **Walk me through DPO derivation.** Solve the RLHF objective in closed form; substitute optimal-policy form into Bradley-Terry; partition function cancels; supervised loss.
4. **DPO vs PPO trade-offs?** DPO simpler, more stable, off-policy. PPO on-policy, more expressive, harder to train.
5. **What's reward hacking?** Policy exploits RM errors instead of being good. Bound by KL anchor; detect by gap with held-out RM or human judges.
6. **What's GRPO?** Replace PPO's value function with group-mean baseline from $K$ rollouts. DeepSeek-R1's algorithm.
7. **What's Constitutional AI?** Use AI critique against written principles to generate preference pairs without human labelers for harmlessness.
8. **What's process vs outcome supervision?** Reward each reasoning step (process) vs only the final answer (outcome). Process is denser, harder to label, often better for reasoning.
9. **Why is the reward model not equal to true human preference?** Trained on finite pairs, has biases (length, certainty, style). Goodhart's law.
10. **What's KTO?** Preference learning from unpaired thumbs-up / thumbs-down feedback. Asymmetric loss (Kahneman-Tversky inspired).
11. **What's the alignment tax?** Capability loss from RL training. Mitigated by KL anchor, careful $\beta$, mixed-data SFT.
12. **What's IPO?** Bounded variant of DPO that doesn't blow up on near-deterministic preferences.

---

## 16. Recommended drill plan

1. Master the RLHF objective and the Bradley-Terry preference model.
2. Whiteboard the DPO derivation end-to-end (closed-form policy → log substitution → $Z$ cancels → loss).
3. Master the KL term: why it's there, how $\beta$ controls behavior, what happens at extremes.
4. Know GRPO's value-function-replacement trick.
5. Know Constitutional AI / RLAIF concept and motivation.
6. Know the failure modes by name: reward hacking, KL blowup, mode collapse, sycophancy, overoptimization.
7. Drill `INTERVIEW_GRILL.md`.

---

## 17. Further reading

- Christiano et al., "Deep reinforcement learning from human preferences" (2017).
- Stiennon et al., "Learning to summarize with human feedback" (2020).
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022).
- Bai et al., "Constitutional AI" (Anthropic, 2022).
- Rafailov et al., "Direct Preference Optimization" (DPO, 2023).
- Schulman et al., "Proximal Policy Optimization" (PPO, 2017).
- Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO, 2023).
- Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization" (2024).
- Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model" (2024).
- Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward" (2024).
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO, 2024).
- DeepSeek-R1 paper (2025) — most prominent demonstration of GRPO at scale.
- Lightman et al., "Let's Verify Step by Step" (process supervision, 2023).
- Gao et al., "Scaling Laws for Reward Model Overoptimization" (Goodhart curves, 2023).
- Tülu 2 / 3 papers (Allen AI) — modern open recipes.
- Lambert et al., "RewardBench" — RM evaluation.

**2024–2025 frontier:**

- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (ByteDance Seed, 2025) — Clip-Higher, Dynamic Sampling, token-level loss, overlong-reward shaping.
- Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO, 2025).
- Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (RLOO, 2024).
- Yuan et al., "Self-Rewarding Language Models" (Meta, 2024).
- Chen et al., "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models" (SPIN, 2024).
- Lai et al., "Step-DPO: Step-wise Preference Optimization" (2024).
- Zeng et al., "Token-level Direct Preference Optimization" (TDPO, 2024).
- Munos et al., "Nash Learning from Human Feedback" (NLHF, DeepMind 2024).
- DeepSeek-R1, Qwen QwQ, Kimi K1.5 — recent frontier reasoning system papers using GRPO/DAPO.
- Allen AI Tülu 3 paper (2024) — modern open recipes including iterative DPO.

If you internalize this document, post-training stops being a black box and becomes a coherent algorithmic stack.
