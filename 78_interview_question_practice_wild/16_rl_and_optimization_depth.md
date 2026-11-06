# Reinforcement learning and optimization depth

**Foundation explainer:** Further depth · **Priority:** Important — second pass

Important: move from algorithm names to states, objectives and observable feedback.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f151"></a>

## F151. Implement value iteration for a finite MDP.

**Start here.** Repeated Bellman optimality backups improve the estimate of long-run discounted return.

**Work it through.**

1. Specify transition probabilities, immediate rewards, terminal states and discount gamma<1 for the standard contraction setting.
2. Update V_new(s)=max_a sum_s' P(s'|s,a)*(R(s,a,s')+gamma*V(s')), with terminal treatment explicit.
3. Stop on a convergence criterion or iteration cap, then derive the greedy policy from the final values.

**Small example.** An action giving reward 1 forever at gamma=0.9 has value 1/(1-0.9)=10.

**Common trap.** Bootstrapping from a terminal state as if it continues generating rewards.

**Follow-up.** Why can gamma=1 be harder? Convergence needs additional episodic or problem-specific assumptions.

---

<a id="f152"></a>

## F152. Compare policy iteration with value iteration.

**Start here.** Policy iteration alternates evaluating a whole policy and improving it; value iteration interleaves shorter optimality updates.

**Work it through.**

1. For a fixed policy, solve or iteratively approximate V_pi=R_pi+gamma*P_pi*V_pi.
2. Improve by choosing actions greedy with respect to the evaluated values.
3. Repeat until the policy stabilizes, explaining evaluation accuracy and tie rules; compare computational cost per iteration as well as iteration count.

**Small example.** A policy that always chooses the immediate largest reward can improve once evaluation reveals a better delayed-return path.

**Common trap.** Comparing only the number of outer iterations while ignoring the cost of policy evaluation.

**Follow-up.** What is modified policy iteration? A middle ground with partial policy evaluation between improvement steps.

---

<a id="f153"></a>

## F153. Implement an epsilon-greedy multi-armed bandit.

**Start here.** Balance learning about uncertain actions with exploiting the best current reward estimate.

**Work it through.**

1. Maintain each arm's count and mean reward, initializing under an explicit policy.
2. With probability epsilon choose a random arm; otherwise choose the greatest estimated mean with a specified tie rule.
3. Update a chosen arm's mean incrementally as mean+=(reward-mean)/count. Evaluate regret over repeated independent runs.

**Small example.** An arm with rewards 1,0,1 has running mean 2/3 after three pulls.

**Common trap.** Updating every arm from one arm's observed reward or evaluating only the final estimated winner.

**Follow-up.** Why might epsilon decay? To reduce long-run exploration, while preserving enough early information under the stated assumptions.

---

<a id="f154"></a>

## F154. Explain an upper-confidence-bound bandit policy.

**Start here.** UCB gives less-sampled arms an uncertainty bonus instead of exploring uniformly at random.

**Work it through.**

1. Pull each arm initially or define an infinite score for untried arms.
2. Select the arm maximizing empirical_mean+c*sqrt(log(t)/count), with the constant and reward assumptions explicit.
3. Update only the chosen arm and inspect behavior under nonstationary rewards, where the stationary formula can become stale.

**Small example.** Two equally rewarded arms are distinguished by counts: the less-tried one receives a larger exploration bonus.

**Common trap.** Using the score as a calibrated probability that the arm is optimal.

**Follow-up.** How might you handle drifting rewards? Sliding windows, discounting or a model explicitly designed for nonstationarity.

---

<a id="f155"></a>

## F155. Explain reward hacking in model post-training.

**Start here.** Optimizing an imperfect reward can improve the reward score while degrading the intended behavior.

**Work it through.**

1. List what the reward measures and what desirable behavior it omits.
2. Test adversarial and distribution-shifted outputs, compare reward with independent human or task-grounded evaluation, and audit high-scoring failures.
3. Improve reward coverage, constrain unsafe behavior and monitor optimization pressure rather than treating reward as ground truth.

**Small example.** An answer that repeats a preferred style phrase can score well on a weak judge without solving the user's task.

**Common trap.** Using the same reward model as the sole final evaluator.

**Follow-up.** Why can the issue worsen with training? Stronger optimization finds loopholes the reward designer did not anticipate.

---

<a id="f156"></a>

## F156. Compare KL penalties and hard behavioral constraints.

**Start here.** A KL term discourages policy movement from a reference distribution, but does not enforce a task-specific safety rule by itself.

**Work it through.**

1. Specify the KL direction, sampling distribution and coefficient in the objective.
2. Monitor achieved divergence and behavior separately; adjust coefficients or stopping policies under a defined procedure.
3. Enforce critical action permissions in the environment or application layer, outside a soft preference objective.

**Small example.** A policy can remain close in average KL while making a rare but unacceptable action more likely.

**Common trap.** Treating a small scalar KL estimate as proof of preserved behavior on every input.

**Follow-up.** Why does KL direction matter? Different directions weight support and low-probability regions differently.

---

<a id="f157"></a>

## F157. Control variance in policy-gradient estimates.

**Start here.** Subtracting a suitable baseline can reduce noise without changing the expected gradient under standard assumptions.

**Work it through.**

1. Write the estimator grad log pi(a|s)*(return-baseline(s)).
2. Ensure the baseline does not depend on the sampled action in a way that biases this form, and stop unwanted gradients through the advantage estimate.
3. Compare batch size, return normalization and value-baseline quality while checking for bias introduced by approximations.

**Small example.** A state-value baseline removes a state's typical return so actions are judged by whether they outperform that baseline.

**Common trap.** Assuming any arbitrary action-dependent subtraction leaves the estimator unbiased.

**Follow-up.** Why not normalize across unrelated prompts blindly? Different reward scales and difficulties can change the intended relative weighting.

---

<a id="f158"></a>

## F158. Distinguish optimizer stability from objective correctness.

**Start here.** A smoothly decreasing loss can still optimize the wrong target or leaked data.

**Work it through.**

1. Verify the mathematical objective, label alignment and reduction before tuning optimizer settings.
2. Use analytic special cases and tiny overfit tests to check implementation.
3. Then study learning rate, initialization, precision and clipping with controlled ablations and independent validation.

**Small example.** A perfectly stable model predicting the current token instead of the next token solves a different, usually trivial task.

**Common trap.** Using convergence as evidence that the data contract is correct.

**Follow-up.** What should you log beyond loss? Gradient norms, update norms, activation statistics, data counts and task-specific metrics.

---

<a id="f159"></a>

## F159. Design a learning-rate ablation that you can interpret.

**Start here.** Change one meaningful factor while keeping data, compute and evaluation comparable.

**Work it through.**

1. Choose a plausible logarithmic learning-rate range and fix architecture, optimizer convention, data order and budget.
2. Run multiple seeds or enough repeated evidence to distinguish instability from random variation.
3. Compare learning curves, final validation quality and divergence frequency; account for schedule and warmup interactions.

**Small example.** Comparing a short run at one rate with a much longer run at another confounds learning rate with compute budget.

**Common trap.** Selecting a rate solely by its best training loss.

**Follow-up.** What if batch size changes? Revisit gradient noise, schedule and effective step size rather than applying a scaling rule blindly.

---

<a id="f160"></a>

## F160. Choose preference learning versus supervised demonstrations.

**Start here.** Use feedback that captures the behavior you need and is feasible to collect reliably.

**Work it through.**

1. Use demonstrations when a good target response can be written or generated and verified.
2. Use preference pairs when comparative judgments are more reliable than producing an ideal answer, controlling pair quality and disagreement.
3. Compare against a strong SFT baseline; reserve a separate evaluation for both quality and regressions.

**Small example.** For a rigid JSON schema, demonstrations plus structured decoding may solve the problem more directly than a costly online reward loop.

**Common trap.** Assuming preference training is always the next necessary step after pretraining.

**Follow-up.** When can verifiable rewards be especially useful? Tasks such as executable code or well-specified mathematical answers with reliable checkers.

---
