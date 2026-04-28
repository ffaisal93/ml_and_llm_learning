# Topic 49: Generalization and Evaluation

> 🔥 **For interviews, read these first:**
> - **`GENERALIZATION_DEEP_DIVE.md`** — frontier-lab deep dive: data leakage (4 types), calibration (ECE, Platt/isotonic/temperature), distribution shift (covariate/label/concept), class imbalance, double descent, cross-validation done right, ablations, metric uncertainty.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions.

## What You'll Learn

This topic is about the questions that separate "I trained a model" from "I know whether the model actually works."

You will learn:
- Train / validation / test roles
- Overfitting and underfitting
- Data leakage
- Class imbalance
- Calibration
- Slice-based error analysis
- Distribution shift
- Ablations and experiment interpretation
- Metric confidence intervals

## Why This Matters for Research Scientist Interviews

A strong research answer is rarely just:

"My model got a better score."

A strong answer is:

"The gain appears on the right slices, survives repeated runs, uses the correct metric, and is not explained by leakage or evaluation artifacts."

That is the mindset this topic is meant to build.

## Core Intuition

### 1. Train, Validation, Test

- **Train**: fit model parameters
- **Validation**: tune choices
- **Test**: final unbiased evaluation

Easy rule:

If the test set affects your design choices, it is no longer a real test set.

### 2. Overfitting and Underfitting

#### Underfitting

- training error high
- validation error high
- model too weak or optimization poor

#### Overfitting

- training error low
- validation error high
- model memorizes training patterns that do not transfer

Interview explanation:

"Overfitting is not just a big model. It is a mismatch between fit to observed training data and ability to generalize to unseen data."

### 3. Data Leakage

Leakage is one of the highest-value topics in interviews because many candidates ignore it.

Leakage means information from the future, the label, or the evaluation set sneaks into training or feature creation.

Common forms:
- fitting preprocessing on all data before splitting
- duplicate records across train and test
- time leakage from future features
- target leakage hidden inside engineered columns

Good answer:

"Before trusting any result, I would check split logic, duplication, feature generation timestamps, and whether any preprocessing was fit outside the training partition."

### 4. Class Imbalance

Accuracy can be misleading.

If 99% of examples are negative, a dumb classifier that always predicts negative gets 99% accuracy.

That is why interviews often ask:
- precision
- recall
- F1
- ROC-AUC
- PR-AUC

Rule of thumb:
- use PR-focused metrics when positives are rare and important
- use recall when missing positives is costly
- use precision when false positives are costly

### 5. Calibration

Calibration asks:

"When the model says 0.8 confidence, is it right about 80% of the time?"

This matters a lot in decision systems.

A model can rank examples well but still be poorly calibrated.

Important distinction:
- discrimination: can it rank good vs bad?
- calibration: do probabilities mean what they claim?

### 6. Slice-Based Evaluation

Average performance can hide serious failures.

Always ask:
- Does the model fail on long inputs?
- Does it fail on rare classes?
- Does it fail on low-resource languages?
- Does it fail on certain customer segments?

Research scientist interviews often reward this kind of thinking.

### 7. Distribution Shift

Performance can collapse when train and deployment distributions differ.

Common shifts:
- covariate shift: input distribution changes
- label shift: class frequencies change
- concept shift: relationship between input and label changes

Useful answer:

"I would compare feature distributions, error slices, and calibration before and after deployment periods. Then I would check whether the shift is in inputs, labels, or the mapping itself."

### 8. Ablations

Ablations answer:

"Which part of the system caused the gain?"

Good ablations:
- remove one change at a time
- keep compute and data consistent
- report the base model clearly
- show failure cases, not just the best metric

### 9. Confidence Intervals for Metrics

If your metric moves from 84.1 to 84.4, that may or may not matter.

Bootstrap is often the easiest way to estimate uncertainty for:
- accuracy
- F1
- recall@k
- exact match

That is especially useful in research discussions where exact analytic variance is awkward.

## Common Failure Modes

### 1. Choosing a Convenient Metric Instead of the Right Metric

Accuracy may look better than recall or PR-AUC on an imbalanced problem, but that does not make it the right metric.

### 2. Hidden Leakage in the Pipeline

Leakage often happens outside the model:
- preprocessing fit on full data
- duplicate rows across splits
- future information in historical features
- target-derived engineered columns

### 3. Treating Average Performance as Complete Evidence

Average metrics can hide failure on:
- rare classes
- long inputs
- specific languages or user groups
- safety-critical slices

### 4. Ignoring Calibration

A model can rank examples well and still assign misleading confidence values.

That matters whenever downstream decisions use probabilities.

### 5. Believing Tiny Metric Differences Without Uncertainty Estimates

A small improvement may be real, or it may be noise.

Without repeated runs or confidence intervals, you should be cautious.

## Edge Cases and Follow-Up Questions

### What if validation improves but test does not?

Possible explanations include:
- overfitting to the validation set
- accidental tuning on the test set earlier
- shift between validation and test
- metric instability

### What if calibration is poor but accuracy is strong?

Then the model may still be risky in decision systems where confidence values affect actions.

You might discuss temperature scaling or recalibration.

### What if the positive class is extremely rare?

Then accuracy becomes even less informative, and PR-focused metrics usually become more relevant.

### What if one slice regresses badly while the overall average improves?

Then the deployment decision depends on the application.

For many real systems, a slice regression can matter more than the global average gain.

## Boilerplate Code

See [diagnostics.py](/Users/faisal/Projects/ml_and_llm_learning/49_generalization_and_evaluation/diagnostics.py) for:

- Binary confusion matrix
- Accuracy / precision / recall / F1
- Expected calibration error (ECE)
- Bootstrap confidence interval for any metric
- Slice accuracy
- Simple ablation deltas

These are the kinds of compact utilities that help during coding rounds and during your own experiment analysis.

## What to Practice Saying Out Loud

1. Why is accuracy a bad metric under class imbalance?
2. How would you detect leakage in an offline pipeline?
3. What does calibration measure that ROC-AUC does not?
4. Why should ablations keep compute and data fixed?
5. If validation gets better but test does not, what are your first hypotheses?

## Next Steps

After this topic:
- Use Topic 50 for fast coding-round patterns
- Use Topic 51 for LLM-specific research interview prep
