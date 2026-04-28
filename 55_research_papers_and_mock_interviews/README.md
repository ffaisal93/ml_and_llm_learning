# Topic 55: Research Papers and Mock Interviews

## What You'll Learn

This topic is the final layer for LLM research scientist interview prep.

It focuses on:
- paper discussion
- experiment criticism
- research proposal thinking
- mock oral interview questions
- probabilistic interview questions like distribution membership

## Why This Matters

Research interviews are often conversational.

The interviewer may ask:
- "What do you think of this result?"
- "What experiment would you run next?"
- "How would you tell which distribution this point came from?"
- "What could invalidate this benchmark?"

This topic is meant to help you practice those transitions.

## Core Intuition

Research interviews are often less about recalling a definition and more about making a defensible judgment from incomplete evidence.

The interviewer is testing whether you can:
- formalize a vague question
- make assumptions explicit
- choose a sensible first-principles method
- notice where the method can fail
- suggest what evidence would make the answer stronger

That is why this topic mixes probability questions, paper criticism, and experiment design.

All three are really the same skill:

turn a fuzzy problem into a structured reasoning process.

## The Distribution-Membership Interview Question

A very common interview question is:

"You have two arrays sampled from two different distributions. A new value arrives. How do you decide which distribution it most likely came from?"

### Good Interview Answer

Start simple and structured:

1. Estimate each distribution from its samples.
2. Compute the likelihood of the new value under each distribution.
3. If class priors differ, multiply by priors.
4. Choose the distribution with the larger posterior score.

In math:

`choose class 1 if p(x | D1) * P(D1) > p(x | D2) * P(D2)`

That is just Bayes classification.

### Important Follow-Up

Then say:

"This requires an assumption about the form of the distributions. If I assume Gaussian, I can estimate mean and variance. If I do not want a parametric assumption, I can use KDE or a nearest-neighbor density estimate."

That answer is much stronger than simply saying "compare the means."

### What the Interviewer Is Testing

They are often testing whether you understand:
- likelihood
- priors
- uncertainty
- model assumptions
- how to adapt if the distribution form is unknown

## Technical Details Interviewers Often Want

### Parametric Version of the Two-Distribution Problem

If you assume each array comes from a Gaussian distribution, you can estimate:
- mean
- variance

Then for a new value `x`, compute:

`p(x | class_1)` and `p(x | class_2)`

If class priors are equal, predict the class with larger likelihood.

If priors differ, compare posterior scores:

`p(x | class_i) P(class_i)`

The important follow-up is that the mean alone is not enough. Variance matters because a value far from the mean of a wide distribution may still be more plausible than a moderately distant value under a very narrow distribution.

### Nonparametric Version

If you do not want to assume a Gaussian, you can estimate density using:
- KDE
- k-nearest-neighbor density ideas
- histogram-style approximations

The trade-off is that nonparametric methods make fewer shape assumptions, but they need more data and can be sensitive to bandwidth or neighborhood choice.

### What the Question Is Really About

This type of interview problem is often a disguised classifier question.

The interviewer wants to see whether you recognize that:
- the arrays are training samples
- the new number is a test point
- the answer is based on comparing class-conditional densities, not just summary statistics

### Paper Critique Pattern

For paper discussion, a strong response usually covers:
- what changed relative to the baseline
- whether the comparison is fair
- what variable may be confounded
- which ablation or slice is missing
- how strong the conclusion really is

That structure is more valuable than a long summary of the method.

## Common Failure Modes

### 1. Comparing Means Only

For the distribution-membership question, many candidates say "see which mean is closer."

That ignores variance and can be badly wrong.

### 2. Forgetting Priors

If one class is much more common than the other, equal-likelihood comparisons can be misleading.

This is exactly why Bayes classification includes priors.

### 3. Ignoring Sample Size

Small arrays may produce noisy parameter estimates.

A good answer should mention uncertainty, especially if the arrays are short or heavy-tailed.

### 4. Overstating a Paper Conclusion

Candidates sometimes see a benchmark gain and immediately say the method is better.

A stronger answer distinguishes:
- observed metric improvement
- likely explanation
- causal claim that still needs more evidence

### 5. Proposing New Experiments Without Fixing the Baseline

Before asking for exotic follow-up experiments, make sure the basic comparison is fair and reproducible.

## Edge Cases and Follow-Up Questions

### What if the two distributions overlap heavily?

Then some points are inherently ambiguous.

A good answer is not to pretend classification is always certain. You can talk about posterior confidence or abstention if the scores are very close.

### What if one distribution is multimodal?

A single Gaussian may be a bad fit.

That is a strong reason to switch to KDE, mixture models, or another nonparametric density estimate.

### What if variances are nearly zero?

Then Gaussian likelihoods can become numerically unstable or overly sharp.

In practice you may need variance smoothing or a minimum variance floor.

### What if the point lies far outside both observed ranges?

Then both models may assign very low probability.

A good answer is to say you would still compare relative scores, but you might also treat the sample as out-of-distribution.

### What if the interviewer asks how to generalize beyond one dimension?

Then the same idea becomes multivariate density estimation or generative classification.

For Gaussian assumptions, that means estimating mean vectors and covariance matrices.

## Files in This Topic

- [research_judgment.py](/Users/faisal/Projects/ml_and_llm_learning/55_research_papers_and_mock_interviews/research_judgment.py): compact helpers for experiment comparison and the two-distribution question
- [mock_interview_questions.md](/Users/faisal/Projects/ml_and_llm_learning/55_research_papers_and_mock_interviews/mock_interview_questions.md): research-style interview prompts

The code is intentionally simple. It is meant to help you practice the reasoning path, not to serve as a production-grade statistical library.

## Practice Habit

Take one question at a time and answer in this structure:

1. Assumptions
2. First-principles method
3. Edge cases
4. What you would verify experimentally

That structure works very well in live interviews.

## What to Practice Saying Out Loud

1. Why is "closest mean" not a sufficient answer to the two-distribution question?
2. When do priors change the final prediction?
3. How would you answer if you do not want to assume Gaussian distributions?
4. What evidence is missing before you believe a paper's claimed improvement?
5. What is the strongest conclusion justified by the current experiment?
