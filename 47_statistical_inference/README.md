# Topic 47: Statistical Inference

> 🔥 **For interviews, read these first:**
> - **`STATISTICAL_INFERENCE_DEEP_DIVE.md`** — frontier-lab deep dive: estimators (unbiased/consistent/efficient + CRLB), MLE asymptotics, CIs (Wald/bootstrap/credible), hypothesis testing, multiple testing (Bonferroni/BH), Bayesian updates with conjugate priors, MAP-as-regularization.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions.

## What You'll Learn

This topic fills an important interview gap: how to reason about data, estimators, uncertainty, and evidence.

You will learn:
- Population vs sample
- Bias, variance, and mean squared error
- Maximum likelihood estimation (MLE)
- Confidence intervals
- Hypothesis testing and p-values
- Bootstrap intuition
- Simple Bayesian updating with conjugate priors

## Why This Matters for Interviews

Many ML interviews are not really asking, "Can you memorize formulas?"

They are asking:
- Do you understand what an estimate means?
- Do you know when a metric difference is probably noise?
- Can you explain uncertainty without hand-waving?
- Can you connect likelihood, loss functions, and model fitting?

For research scientist interviews, this matters because you need to justify conclusions from experiments. If a model improves by 0.3 points, is that real or random? If validation loss changes, what does that say about the data or the estimator?

## Core Intuition

### 1. Population vs Sample

- **Population**: the full data-generating process you care about
- **Sample**: the finite subset you actually observed

Example:
- Population question: "What is the true average click-through rate?"
- Sample answer: "In this experiment, the observed mean CTR was 3.1%."

The key point is that your sample statistic is only an estimate of the true population quantity.

### 2. Estimators

An **estimator** is a rule for turning data into a number.

Examples:
- Sample mean estimates population mean
- Sample variance estimates population variance
- Positive-rate estimates Bernoulli probability

### 3. Bias and Variance

When an interviewer asks about bias and variance, use this mental picture:

- **Bias**: how far the average estimate is from the truth
- **Variance**: how much the estimate changes across different samples

An estimator can fail in two ways:
- It is consistently wrong: high bias
- It is unstable across samples: high variance

The standard decomposition is:

`MSE = Bias^2 + Variance + Irreducible Noise`

That equation is useful because it explains why larger models can overfit:
- low bias
- high variance

It also explains why stronger regularization can help:
- slightly more bias
- much lower variance

### 4. Maximum Likelihood Estimation (MLE)

MLE chooses parameters that make the observed data most probable.

#### Bernoulli Example

Suppose each label is 0 or 1, and you model it as `Bernoulli(p)`.

If you observe:

`[1, 0, 1, 1, 0]`

then the MLE of `p` is just the sample mean:

`p_hat = number_of_ones / n`

That is why logistic regression and binary classification are so tied to probability modeling.

#### Gaussian Example

For a Gaussian with unknown mean and variance, MLE gives:

- `mu_hat = sample mean`
- `sigma^2_hat = average squared deviation from mu_hat`

Notice that the MLE variance uses division by `n`, not `n - 1`.

Interview detail:
- divide by `n` for MLE
- divide by `n - 1` for the unbiased sample variance estimator

### 5. Confidence Intervals

A confidence interval gives a range of plausible values for a parameter.

For a sample mean, the most common large-sample form is:

`mean +/- z * standard_error`

where:

`standard_error = sample_std / sqrt(n)`

Easy interpretation:
- Larger sample size means smaller standard error
- Higher variance means wider intervals

Important interview point:

A 95% confidence interval does **not** mean:
"There is a 95% probability the true mean is inside this interval."

The frequentist meaning is:
"If we repeated this whole sampling procedure many times, 95% of those intervals would contain the true parameter."

### 6. Hypothesis Testing

The usual structure is:

- Null hypothesis `H0`: no effect / no difference
- Alternative hypothesis `H1`: effect exists

Then you compute a test statistic and a p-value.

Easy way to explain a p-value:

"Assuming the null hypothesis were true, how surprising is the observed result or something more extreme?"

Common mistake:
- A p-value is not the probability that the null is true

### 7. Bootstrap

Bootstrap is useful when formulas are messy.

Idea:
1. Resample your dataset with replacement
2. Recompute the statistic many times
3. Use the empirical distribution of that statistic

Why interviewers like it:
- It is practical
- It works for complicated metrics
- It shows statistical maturity without requiring closed-form derivations

### 8. Bayesian Updating

Bayesian thinking is often easier to explain in interviews than people expect.

The recipe is:

`posterior proportional to likelihood times prior`

Simple and very useful example:

- Prior: `Beta(alpha, beta)`
- Observations: Bernoulli coin flips
- Posterior: `Beta(alpha + heads, beta + tails)`

Interpretation:
- `alpha - 1` acts like prior successes
- `beta - 1` acts like prior failures

This is easy to explain under pressure because the update rule is simple and intuitive.

## Common Failure Modes

### 1. Confusing an Estimate with the Truth

Candidates often talk about a sample statistic as if it were the population parameter.

A better answer explicitly separates:
- the unknown quantity you care about
- the finite-sample estimate you observed

### 2. Misinterpreting p-values

One of the most common mistakes is saying:

"The p-value is the probability that the null hypothesis is true."

That is not the frequentist meaning.

### 3. Mixing Up `n` and `n - 1`

Interviewers often use this to test whether you understand the difference between:
- maximum-likelihood estimation
- unbiased variance estimation

### 4. Using Large-Sample Formulas Without Checking Assumptions

Normal approximations can be fine for large samples, but not always for:
- small `n`
- heavy-tailed data
- highly skewed distributions

### 5. Reporting Confidence Intervals Without Saying What Procedure Generated Them

A confidence interval only makes sense relative to a sampling procedure and a method.

If you do not mention assumptions or the resampling method, the interval can sound more certain than it really is.

## Edge Cases and Follow-Up Questions

### What if the sample size is very small?

Then large-sample approximations may be weak.

A stronger answer might mention:
- t-based intervals
- bootstrap with caution
- stronger distributional assumptions if justified

### What if the data are heavy-tailed or contain outliers?

Then the sample mean and variance may be unstable.

You might discuss robust alternatives, trimmed means, or bootstrap-based uncertainty summaries.

### What if two models differ by a tiny amount on a noisy metric?

Then you should talk about uncertainty, repeated runs, bootstrap intervals, or significance testing instead of treating the difference as obviously meaningful.

### What if the interviewer asks for a Bayesian answer instead?

Then reframe the problem in terms of prior, likelihood, and posterior.

That is often enough to show flexibility even without a long derivation.

## What to Practice Saying Out Loud

You should be able to explain these clearly:

1. Why divide variance by `n` for MLE but `n - 1` for the unbiased estimator?
2. Why does regularization reduce variance?
3. When would you use bootstrap instead of a closed-form confidence interval?
4. What does a p-value mean, and what does it not mean?
5. How does MLE connect to cross-entropy loss?

## Boilerplate Code

See [statistical_inference.py](/Users/faisal/Projects/ml_and_llm_learning/47_statistical_inference/statistical_inference.py) for easy interview-style code covering:

- Sample mean and variance
- Bernoulli and Gaussian MLE
- Normal-approximation confidence interval
- Bootstrap confidence interval
- Two-sample t-test
- Beta-Bernoulli posterior update

The code is intentionally short and pressure-friendly:
- small functions
- no unnecessary abstractions
- direct math

## Interview Tips

- Start with the estimator before the formula
- Explain assumptions before quoting the result
- If exact math is hard, say what quantity is random and what quantity is fixed
- If you forget a test, fall back to bootstrap logic

## Next Steps

After this topic:
- Use Topic 48 for optimization and matrix calculus
- Use Topic 49 for generalization, evaluation, and experiment diagnosis
