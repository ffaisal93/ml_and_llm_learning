# Training and alignment

**Foundation explainer:** Core material · **Priority:** Very important

Very important: connect objectives, gradients, data and verification.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f021"></a>

## F021. Distinguish pretraining, supervised fine-tuning and preference optimization.

**Start here.** They change the training signal and data distribution, rather than automatically changing the model architecture.

**Work it through.**

1. Pretraining predicts tokens over broad text to learn general representations and continuation behavior.
2. SFT applies supervised likelihood to demonstrations, often masking prompt targets so answers receive the training signal.
3. Preference optimization changes relative response desirability using comparisons or rewards. Evaluate task performance, behavior and regressions separately after each stage.

**Small example.** A correct SFT response demonstrates one answer; a preference pair says one answer is preferred to another, without requiring either to be an ideal demonstration.

**Common trap.** Claiming preference training guarantees truthfulness or that SFT itself retrieves fresh facts.

**Follow-up.** Which stage would you use for a new output schema? Often SFT or constrained decoding first, depending on the need.

---

<a id="f022"></a>

## F022. Implement masked cross-entropy with label smoothing.

**Start here.** Smoothing mixes the true-label target with a uniform distribution; masking controls which examples contribute to the average.

**Work it through.**

1. Compute stable log probabilities by subtracting each row maximum before log-sum-exp.
2. For each row use loss=(1-epsilon)*(-logp_true)+epsilon*(-mean(logp)). This corresponds to target_true=1-epsilon+epsilon/C.
3. Average included rows only; return 0 when the mask selects none. Validate target ranges and use the executable implementation in the lab.

**Small example.** For uniform predictions over 4 classes, the loss is log(4) for any smoothing coefficient because every class has the same log probability.

**Common trap.** Dividing by the original batch size after excluding examples, or using epsilon/(C-1), which is a different smoothing convention.

**Follow-up.** What changes for token losses? Flatten valid prediction/target positions or preserve their dimensions while applying the same masked reduction.

---

<a id="f023"></a>

## F023. Pack SFT examples without corrupting targets.

**Start here.** Packing reduces padding, but positions, attention visibility and target selection are three separate concerns.

**Work it through.**

1. For the practice contract, lay out prompt+answer+separator; mark prompt positions 0 and answer/separator positions 1.
2. Sort layouts longest first with stable ties. Place each into the first sample with enough capacity; skip oversize examples and pad the rest. Restart position ids per example.
3. During training, shift the target loss mask consistently with next-token labels. If examples must be independent, also use segment-aware attention or separate sequences.

**Small example.** Prompt [5,6], answer [7], separator 99 gives ids [5,6,7,99], mask [0,0,1,1] and positions [0,1,2,3].

**Common trap.** Assuming a loss mask or reset position ids prevent one packed example from attending to another.

**Follow-up.** What is the simple first-fit-decreasing implementation's worst-case placement cost? O(number_of_examples²), plus token copying.

---

<a id="f024"></a>

## F024. Explain LoRA and calculate its trainable parameter count.

**Start here.** LoRA represents a weight update as the product of two thin matrices while leaving the original weight frozen.

**Work it through.**

1. For W shaped (d_out,d_in), use delta_W=(alpha/r)*B@A with A:(r,d_in) and B:(d_out,r).
2. Train A and B, usually initializing one factor to zero so the initial update is zero without making both gradients permanently zero.
3. The trainable count is r*(d_in+d_out), versus d_in*d_out for the full matrix. Include all targeted matrices when estimating the whole model.

**Small example.** A 4096-by-4096 matrix with r=8 has 65536 adapter parameters instead of 16777216 full parameters, excluding biases.

**Common trap.** Saying the rank r alone is the parameter count, or that frozen base weights remove forward activation-memory costs.

**Follow-up.** Can the adapter be merged? Usually yes for a compatible linear layer; quantized storage and multiple adapters need extra care.

---

<a id="f025"></a>

## F025. Why use AdamW, warmup and gradient clipping?

**Start here.** They address different training problems: adaptive update scale, early-step instability and unusually large gradients.

**Work it through.**

1. Adam maintains first and second moment estimates, with bias correction after zero initialization.
2. AdamW applies decoupled weight decay rather than treating decay exactly as another adaptively scaled loss gradient.
3. Warm up learning rate over initial steps and clip the global gradient norm when appropriate. Inspect logs and ablations; these are tuning tools, not correctness fixes.

**Small example.** A gradient [3,4] has norm 5. Clipping its norm to 1 yields [0.6,0.8], preserving direction before the optimizer's own transformation.

**Common trap.** Using clipping to hide exploding activations, bad data or incorrect loss scaling.

**Follow-up.** Why unscale mixed-precision gradients before clipping? Otherwise the threshold is applied to artificially scaled gradients.

---

<a id="f026"></a>

## F026. Compare PPO, DPO and GRPO using their training requirements.

**Start here.** Choose by available feedback, exploration needs and memory budget rather than treating the methods as interchangeable names.

**Work it through.**

1. PPO uses current-policy rollouts, advantages and a clipped probability-ratio objective; common RLHF setups add a reward model, critic and reference-policy regularization.
2. DPO trains on preferred/rejected pairs with a frozen reference and a logistic loss on differences of sequence log-probability ratios; no explicit reward model is required in that objective.
3. GRPO compares several sampled answers for a prompt using group-relative rewards, avoiding a learned value critic in its usual form. Reward and variant details still matter.

**Small example.** Use preference pairs with modest compute for a DPO baseline; programmatically checked math/code rewards are a natural setting for group-relative online training.

**Common trap.** Saying DPO requires online rollouts at every step, or that critic-free training removes the need for a reliable reward signal.

**Follow-up.** When could PPO still fit? When online exploration and learned state-dependent value estimates are useful and resources permit them.

---

<a id="f027"></a>

## F027. Write and interpret the DPO loss.

**Start here.** DPO increases the preferred response's relative probability advantage over the rejected response, measured against a reference policy.

**Work it through.**

1. Let a=log pi_theta(y_w|x)-log pi_ref(y_w|x) and b=log pi_theta(y_l|x)-log pi_ref(y_l|x).
2. Minimize L=-log sigmoid(beta*(a-b)); use a stable log-sigmoid or softplus implementation.
3. Sum token log probabilities over each response, excluding prompt and padding targets under a consistent convention. Keep the reference frozen.

**Small example.** If beta*(a-b)=0, the loss is log(2)=0.693. If the margin is 1, the loss is about 0.313.

**Common trap.** Using raw response probabilities rather than log-ratio differences, reversing winner/loser signs, or training the reference accidentally.

**Follow-up.** What can go wrong despite decreasing loss? Preference-label errors, length artifacts, distribution shift and excessive movement from desired behavior.

---

<a id="f028"></a>

## F028. Explain a clipped PPO update in plain language.

**Start here.** PPO rewards actions with positive estimated advantage while limiting the incentive for a large probability-ratio change within an update.

**Work it through.**

1. Form r=pi_new(action|state)/pi_old(action|state) from the rollout policy, with rollout quantities treated as fixed.
2. Maximize mean(min(r*A,clip(r,1-epsilon,1+epsilon)*A)). A common minimized total loss adds value error and KL regularization and subtracts an entropy bonus.
3. Estimate advantages from returns and a value baseline, then monitor policy KL, reward, value error and clipping fraction.

**Small example.** For A=2, epsilon=0.2 and r=1.5, the positive surrogate is capped at 1.2*2=2.4 rather than 3.

**Common trap.** Confusing the old rollout policy in the ratio with the reference policy used for behavioral regularization.

**Follow-up.** Does clipping enforce a strict KL constraint? No; it is a surrogate, so empirical KL monitoring still matters.

---

<a id="f029"></a>

## F029. Calculate group-relative advantages and diagnose zero-variance rewards.

**Start here.** GRPO-style normalization asks which sampled answers did better than their siblings for the same prompt.

**Work it through.**

1. Sample G responses, score each, and compute A_i=(r_i-mean(r))/(std(r)+epsilon) under the chosen convention.
2. Use these fixed advantages in the policy-ratio update, with any reference penalty and token reduction explicitly specified.
3. If every reward is equal, centered advantages are zero. Investigate sampling diversity, task difficulty and reward informativeness rather than dividing by zero.

**Small example.** Rewards [0,0,1,1] have mean 0.5 and population standard deviation 0.5, giving advantages [-1,-1,1,1] before epsilon.

**Common trap.** Treating G answers from one prompt as G independent prompts when estimating evaluation uncertainty.

**Follow-up.** What if reward variance is mostly noise? Relative normalization can amplify unreliable comparisons; improve the reward and inspect sensitivity.

---

<a id="f030"></a>

## F030. Train with mixed precision without losing small gradients.

**Start here.** Lower-precision matrix operations can improve throughput, while sensitive reductions and optimizer state may need higher precision.

**Work it through.**

1. Choose supported autocast dtypes and keep numerically sensitive loss calculations in an appropriate accumulator precision.
2. For FP16, loss scaling can protect small gradients; unscale before clipping, detect overflow and skip invalid updates. BF16 usually needs less range protection but still has limited mantissa precision.
3. Compare convergence and output quality against a higher-precision baseline; measure actual speed and memory rather than assuming benefits.

**Small example.** FP16 has a narrower exponent range than BF16, so a finite FP32 activation can overflow FP16 while remaining representable in BF16.

**Common trap.** Casting every operation and optimizer state to the smallest dtype, or assuming BF16 is universally as accurate as FP32.

**Follow-up.** Why can a reduction overflow when individual elements look reasonable? The accumulated sum may exceed the representable range.

---
