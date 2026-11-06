# Debugging and interview reasoning

**Foundation explainer:** Core material · **Priority:** Very important

Very important: demonstrate that you can isolate failures and defend decisions.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f091"></a>

## F091. A classifier's loss does not decrease. What do you check first?

**Start here.** Use a small controlled problem to distinguish a broken training loop from a difficult learning task.

**Work it through.**

1. Check target encoding, logits/loss compatibility, preprocessing, train mode, optimizer parameters and zero_grad/backward/step order.
2. Inspect finite activations and gradients, then confirm that parameters change. Try overfitting a tiny batch.
3. Once basic learning works, sweep learning rate and compare a simple baseline; change one factor at a time.

**Small example.** Passing softmax probabilities into a loss expecting logits applies a second normalization and can produce weak or misleading gradients.

**Common trap.** Blaming insufficient model size before proving that the optimizer is updating the intended parameters.

**Follow-up.** If only validation loss is flat, what changes? Inspect generalization, split mismatch and validation preprocessing rather than only optimizer plumbing.

---

<a id="f092"></a>

## F092. How do you find a silent training-data leak?

**Start here.** A suspiciously good score should trigger a search for information unavailable at real prediction time.

**Work it through.**

1. Map each feature to its event and availability timestamps, then inspect split overlap by entity, duplicate and time.
2. Fit preprocessing within training folds and compare with deliberately simple baselines and label-shuffle controls.
3. Remove suspicious features or enforce a stricter split, rerun the entire selection pipeline and document the score change.

**Small example.** Near-duplicate support tickets appearing in both train and test can let a model memorize answers without generalizing to new cases.

**Common trap.** Checking only exact row duplicates while missing shared users, templates or future aggregates.

**Follow-up.** What should a label-shuffle control do? Performance should fall toward the appropriate chance baseline when genuine signal is destroyed.

---

<a id="f093"></a>

## F093. How do you gradient-check a new numerical layer?

**Start here.** Compare the analytic chain-rule gradient with the change in a small scalar objective under tiny parameter perturbations.

**Work it through.**

1. Choose small float64 inputs and a scalar objective such as sum(output*upstream_gradient).
2. For each tested parameter, compute [L(theta+h)-L(theta-h)]/(2h), restoring the parameter afterward.
3. Compare absolute and relative error across several h values; avoid nondifferentiable boundaries and separately check all gradient tensors.

**Small example.** If analytic gradient is 0.25 and the central difference is 0.25000001, the difference is consistent with ordinary floating-point error.

**Common trap.** Perturbing a parameter without restoring it or testing ReLU exactly at zero.

**Follow-up.** Why not use an arbitrarily tiny h? Subtractive cancellation eventually overwhelms the signal.

---

<a id="f094"></a>

## F094. How do you debug a production latency regression?

**Start here.** Locate the changed stage and workload before rewriting the model or adding hardware.

**Work it through.**

1. Compare request mix, prompt/output lengths, concurrency and cache hit rates against the baseline period.
2. Trace queueing, preprocessing, retrieval, model execution and network time separately, including tail percentiles.
3. Check deployment changes, batching, memory pressure and retries; validate a targeted fix under matched load and keep rollback available.

**Small example.** If median latency is stable but p99 worsens, a small fraction of long requests or queueing spikes may dominate the regression.

**Common trap.** Comparing two benchmark runs with different lengths or concurrency and attributing the difference solely to code.

**Follow-up.** Why can throughput improve while user latency worsens? Larger batches can increase queueing and per-request waiting.

---

<a id="f095"></a>

## F095. Explain your most technically complex ML project.

**Start here.** A strong answer demonstrates decisions and evidence, not a list of technologies.

**Work it through.**

1. State the user problem, constraints and your own contribution separately from the team's work.
2. Describe the data, baseline, hardest technical failure, alternatives considered and why the chosen approach fit.
3. Quantify the result using a valid comparison, explain limitations and say what you would change next. Use real experiences rather than invented metrics.

**Small example.** “I reduced p95 latency from 800 to 500 ms on a matched workload while keeping task accuracy within the predefined margin” states both benefit and guardrail.

**Common trap.** Saying “we built an LLM system” without explaining your choices or how success was measured.

**Follow-up.** What if the project failed? Explain the evidence, what you stopped doing and what the failure taught you.

---

<a id="f096"></a>

## F096. Defend a paper's contribution in a research interview.

**Start here.** Separate the claimed contribution from the evidence that actually establishes it.

**Work it through.**

1. Restate the problem, baseline and proposed mechanism in your own words.
2. Inspect ablations, compute/data parity, uncertainty, benchmark contamination and whether improvements survive meaningful slices.
3. Propose one discriminating experiment and discuss reproduction cost and deployment constraints. Be explicit about what you have not verified.

**Small example.** A new architecture trained on twice the data cannot attribute its full gain to architecture without a matched-data comparison.

**Common trap.** Reciting the abstract or a leaderboard number instead of identifying the causal evidence for the claim.

**Follow-up.** Which ablation would you run first? The cheapest experiment that separates the proposed mechanism from a plausible alternative explanation.

---

<a id="f097"></a>

## F097. Plan a four-hour ML take-home exercise.

**Start here.** Budget the work so a reproducible baseline and honest evaluation exist before optional improvements.

**Work it through.**

1. Spend an initial block clarifying the target, split, metric, data quality and a minimal runnable pipeline.
2. Build and evaluate a baseline, then run one or two prioritized improvements with tracked seeds and configurations.
3. Reserve time for error analysis, clean reruns and a short write-up of assumptions, results and remaining limitations.

**Small example.** A simple model with a valid temporal split and reproducible commands is stronger evidence than an elaborate model evaluated on leaked data.

**Common trap.** Using the final hour for one unvalidated architecture change and leaving no runnable result.

**Follow-up.** What belongs in the presentation? The decision, baseline, evidence, failure slices and the highest-value next experiment.

---

<a id="f098"></a>

## F098. How do you handle disagreement over an ML design?

**Start here.** Make the disputed assumptions testable and connect the decision to the shared objective.

**Work it through.**

1. Restate the other person's concern fairly and identify whether the disagreement concerns facts, constraints or preferences.
2. Propose a small experiment or decision table with agreed success criteria, cost and deadline.
3. Document the decision, support execution after it is made and revisit only when meaningful new evidence appears.

**Small example.** A debate over retrieval complexity can become a measured comparison of answer quality, p95 latency and maintenance cost on a fixed query set.

**Common trap.** Describing a colleague as incompetent or presenting “I won the argument” as the result.

**Follow-up.** What if experimentation is impossible? Make assumptions and reversibility explicit, then choose using the available evidence and accountable decision process.

---

<a id="f099"></a>

## F099. How do you explain a tradeoff under interview pressure?

**Start here.** Give the recommendation first, then the mechanism, the price paid and the condition that would change your mind.

**Work it through.**

1. Name the objective and dominant constraint: quality, latency, memory, cost or operational simplicity.
2. Compare a baseline with one meaningful alternative using concrete quantities or clearly stated assumptions.
3. Choose, acknowledge a failure mode and propose a validation step. Stop before turning the answer into a catalog of unrelated methods.

**Small example.** “I would start with retrieval plus a small reranker because quality needs exact evidence and the latency budget permits it; I would remove reranking if measured gains are negligible.”

**Common trap.** Saying “it depends” without identifying what it depends on or making a provisional decision.

**Follow-up.** What if the interviewer changes a constraint? Update the recommendation transparently rather than defending the original answer at all costs.

---

<a id="f100"></a>

## F100. How do you practice an unfamiliar ML coding problem?

**Start here.** Turn the prompt into a contract, implement a clear baseline and test the cases that distinguish a correct solution from a plausible one.

**Work it through.**

1. Restate inputs, outputs, shapes, reduction rules and constraints; ask only about ambiguities that affect correctness.
2. Work a tiny example by hand, write the simplest valid algorithm and explain its time and memory cost.
3. Test boundaries, invariants, numerical extremes and an independent reference; optimize only after correctness is established.

**Small example.** For masked loss, “all examples masked” and “very large logits” are more revealing than ten random ordinary batches.

**Common trap.** Memorizing one vectorized expression without being able to derive shapes or handle changed conventions.

**Follow-up.** How should you recover from a bug? State the violated invariant, isolate the smallest failing case and fix that cause.

---
