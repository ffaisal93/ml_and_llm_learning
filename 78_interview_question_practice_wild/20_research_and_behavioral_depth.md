# Research and behavioral depth

<span class="practice-page"></span>

**Foundation explainer:** Further depth · **Priority:** Important — second pass

Important: prepare evidence-based stories and research judgments rather than scripts.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f191"></a>

## F191. Present your background in ninety seconds.

**Start here.** Connect your real experience to the role through a small number of relevant capabilities and outcomes.

**Work it through.**

1. Start with your current focus and the kinds of problems you solve.
2. Choose one or two concrete projects, naming your role, difficult decision and measured result.
3. Finish with why the target work fits your interests and what you want to contribute, without inventing credentials or metrics.

**Small example.** An MLE introduction can connect data-quality work, reliable serving and one measured deployment result instead of listing every library used.

**Common trap.** Reciting the entire resume chronologically or making claims you cannot defend under follow-up.

**Follow-up.** What evidence should you prepare? A real baseline, your contribution, tradeoffs and enough technical detail for deeper questions.

---

<a id="f192"></a>

## F192. Explain a project failure constructively.

**Start here.** Show ownership, evidence and changed behavior, not a story in which someone else is always at fault.

**Work it through.**

1. State the goal, your responsibility and what failed in observable terms.
2. Explain which assumption or decision was wrong and how you detected it.
3. Describe recovery, what changed afterward and whether the fix actually helped. Keep confidential details appropriately abstract.

**Small example.** A failed evaluation due to duplicate leakage can become a concrete lesson about entity-aware splits and dataset audits.

**Common trap.** Inventing a “failure” that is really only a disguised success or blaming a teammate without examining your own actions.

**Follow-up.** What would you do earlier next time? Name a specific diagnostic or process change.

---

<a id="f193"></a>

## F193. Discuss the data strategy behind an ML project.

**Start here.** Data choices can explain more of a project's outcome than the architecture name.

**Work it through.**

1. Describe collection, permissions, label generation, cleaning and the intended deployment distribution.
2. Explain split design, duplicates, hard examples, reviewer quality and versioning.
3. Connect one data intervention to a controlled result and identify unresolved gaps.

**Small example.** Improving annotation guidelines while holding the model fixed can show a clearer causal contribution than changing data and architecture together.

**Common trap.** Calling the dataset “clean” without saying what was checked.

**Follow-up.** What would you collect next? Prioritize examples that address measured failure modes and important missing slices.

---

<a id="f194"></a>

## F194. Defend a reproduction that does not match a paper's result.

**Start here.** A mismatch is evidence to investigate, not immediate proof that either the paper or your implementation is wrong.

**Work it through.**

1. Check data versions, preprocessing, evaluation protocol, model configuration and compute budget.
2. Run simple baselines and implementation checks, then repeat enough seeds to assess variability.
3. Report the gap, verified matches and remaining differences, and propose the cheapest experiment that distinguishes explanations.

**Small example.** A reported score using a different test split is not a direct reproduction comparison.

**Common trap.** Quietly changing the evaluation until the desired number appears.

**Follow-up.** What belongs in a useful report? Exact commands/configuration, environment, evidence, discrepancies and limitations.

---

<a id="f195"></a>

## F195. Translate a research technique into a production decision.

**Start here.** A benchmark gain must survive the application's latency, data, maintenance and failure constraints.

**Work it through.**

1. Restate the proposed mechanism and the baseline it would replace.
2. Estimate integration cost and run a representative offline or shadow comparison with fixed success criteria.
3. Evaluate operational behavior, regressions, monitoring and rollback before broader adoption.

**Small example.** A reranker improving NDCG slightly may still be unsuitable if it doubles p99 latency beyond the product budget.

**Common trap.** Assuming state-of-the-art benchmark status implies production suitability.

**Follow-up.** When would you defer adoption? When the claimed benefit is smaller than uncertainty or operational cost under the actual workload.

---

<a id="f196"></a>

## F196. Handle harsh technical feedback.

**Start here.** Separate the useful content of feedback from its delivery and verify what needs to change.

**Work it through.**

1. Listen, summarize the specific concern and ask for concrete examples when necessary.
2. Check the evidence, acknowledge valid mistakes and propose a bounded correction with a follow-up date.
3. If disagreement remains, explain it respectfully using the shared goal and evidence, and seek the appropriate resolution channel.

**Small example.** Feedback that a report “is not reproducible” can lead to adding fixed configs, dependency notes and a clean-run command.

**Common trap.** Claiming you always accept every criticism or always prove the critic wrong.

**Follow-up.** How do you show growth? Describe a real later situation where the changed practice prevented the same problem.

---

<a id="f197"></a>

## F197. Resolve a policy-versus-efficiency tradeoff.

**Start here.** Distinguish a mandatory constraint from a process that can be improved through an authorized change.

**Work it through.**

1. Clarify the purpose, owner and actual requirements of the policy.
2. Find compliant alternatives or propose a reviewed exception with evidence about benefits, risks and controls.
3. Document the decision and measure the result; do not frame secretly bypassing a requirement as good ownership.

**Small example.** A slow data-access process may be improved with a scoped approved dataset rather than copying unrestricted private data to a convenient location.

**Common trap.** Treating speed as justification for ignoring an applicable security or privacy constraint.

**Follow-up.** What if the policy is ambiguous? Escalate the specific ambiguity with a concrete proposed path.

---

<a id="f198"></a>

## F198. Prepare a research experiment with limited compute.

**Start here.** Spend the budget on experiments that most clearly distinguish hypotheses.

**Work it through.**

1. State the hypothesis, expected observable effect and a simple falsifying result.
2. Choose a small faithful setting, strong baseline and controlled ablation before scaling.
3. Track compute, seeds and failures, and reserve budget for confirmation rather than spending everything on a broad unreplicated search.

**Small example.** Testing a new loss on a small task with known failure modes can be more informative than one expensive full-model run with many changes.

**Common trap.** Confounding architecture, data, optimizer and compute while claiming evidence for one mechanism.

**Follow-up.** What if a small-scale result fails to transfer? Identify which scaling assumptions changed and test them explicitly.

---

<a id="f199"></a>

## F199. Explain when small-model lessons transfer to large models.

**Start here.** Mechanisms may transfer, but hyperparameters, bottlenecks and emergent failure patterns need fresh evidence at scale.

**Work it through.**

1. Separate general correctness facts, such as target alignment, from empirical tuning choices.
2. Study how activation statistics, optimization, data mixture, compute and communication change with scale.
3. Use staged scaling experiments and comparable evaluation rather than extrapolating one small-model ranking blindly.

**Small example.** A data-cleaning fix may transfer broadly, while the best learning rate or parallelization layout can change substantially.

**Common trap.** Claiming a cheap ablation proves the same effect size at a much larger scale.

**Follow-up.** Which signal would you monitor while scaling? Stability, loss-versus-compute curves and representative downstream evaluations.

---

<a id="f200"></a>

## F200. Choose the next interview practice task from your mistakes.

**Start here.** Practice should target recall and reasoning gaps rather than the topics that already feel comfortable.

**Work it through.**

1. Attempt a question closed-book, explaining assumptions and writing any required equations or code.
2. Score correctness, clarity, edge cases and ability to answer a follow-up; record one concrete gap.
3. Reattempt after a delay with changed numbers or constraints, and advance only when you can explain the mechanism without reading.

**Small example.** If you repeatedly forget cached causal-mask offsets, practice prefix-equivalence tests rather than rereading the entire attention chapter.

**Common trap.** Measuring progress by pages read or recognizing a memorized answer.

**Follow-up.** What is a useful final mock? A timed mix of conceptual explanation, derivation, coding and a system-design constraint change.

---
