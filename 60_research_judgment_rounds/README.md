# Topic 60: Research Judgment Rounds

## What You'll Learn

This topic focuses on the interview segment where you are shown a result, claim, or experiment and asked whether it is convincing.

You will practice:
- judging evidence quality
- asking for missing ablations
- identifying leakage and confounds
- checking metric validity
- evaluating robustness and variance
- proposing the next experiment

## Why This Matters

This is where many strong coders underperform.

They describe methods well, but they do not interrogate evidence well.

A research scientist interview expects you to say:
- what is missing
- what is weak
- what you would run next
- what conclusion is and is not justified

## Core Intuition

Research judgment is the ability to separate:
- the result that was measured
- the claim being made
- the causal explanation being implied

Those are not the same thing.

A paper, experiment, or benchmark can show a real improvement while still not proving the story attached to it.

That is why good research answers sound careful but not vague.

You are trying to identify:
- what the evidence supports
- what alternative explanations remain
- what experiment would reduce that uncertainty next

## Files in This Topic

- [rounds.md](/Users/faisal/Projects/ml_and_llm_learning/60_research_judgment_rounds/rounds.md): scenario-based research rounds
- [judgment_checklist.md](/Users/faisal/Projects/ml_and_llm_learning/60_research_judgment_rounds/judgment_checklist.md): compact interview checklist

## Technical Details Interviewers Often Want

### Fair Comparison

Whenever a result improves, ask:
- what changed?
- what stayed fixed?

Important hidden confounds include:
- more data
- more compute
- different decoding
- different retrieval setup
- different prompt template
- better hyperparameter tuning

If multiple things changed, causal attribution is weak.

### Metric Fit

A metric can be computed correctly and still be the wrong metric.

Examples:
- perplexity for a task where factual grounding matters more
- recall@k without checking whether generation used the evidence
- average score hiding catastrophic failures on important slices

The question is not just "did the metric improve?"

It is "does this metric track the behavior we actually care about?"

### Variance and Slicing

Average improvement is often not enough.

Interviewers want to know whether you would ask for:
- multiple seeds
- confidence intervals
- subgroup analysis
- failure-category analysis
- long-tail examples

This is especially important for LLM systems where improvements may be concentrated in narrow prompt families.

### Next-Experiment Logic

A good follow-up experiment should reduce uncertainty about the main claim.

That means the next experiment should be chosen because it tests a specific alternative explanation, not because it is merely interesting.

## Common Failure Modes

### 1. Confusing Correlation with Mechanism

A method may correlate with improvement without being the true cause if other variables moved at the same time.

### 2. Accepting Headline Averages Too Quickly

Average gain can hide:
- instability
- regression on rare cases
- benchmark artifacts
- narrow prompt-template gains

### 3. Asking for More Experiments Without Prioritizing

In interviews, it is stronger to name the most decision-relevant next experiment than to list ten vague possibilities.

### 4. Ignoring Baseline Strength

If the baseline is weak or badly tuned, the reported gain says less than it appears to.

## Edge Cases and Follow-Up Questions

### What if the new method is better on average but worse on critical slices?

Then the deployment conclusion depends on the application.

A good answer should say that aggregate improvement may still be unacceptable if the regressions hit safety-critical or high-value cases.

### What if one seed is much better than the others?

Then average performance and variance both matter.

You should avoid overinterpreting a lucky run.

### What if the method improves offline metrics but slows serving dramatically?

Then the true system trade-off includes latency and cost, not only quality.

### What if the gain disappears after controlling for compute?

Then the original claim should be weakened.

The right conclusion might be "better compute allocation" rather than "better algorithm."

## Standard Response Pattern

When shown a result:

1. Clarify the claim.
2. Ask what changed.
3. Ask what stayed fixed.
4. Check metric fit.
5. Ask for variance and slices.
6. State the strongest conclusion justified by the evidence.

## What to Practice Saying Out Loud

1. What exact claim is this experiment trying to support?
2. What changed, and what must be held fixed for this comparison to be fair?
3. Which metric could improve while the true user outcome gets worse?
4. What is the single best next experiment to reduce uncertainty?
5. What conclusion is justified, and what conclusion is still too strong?
