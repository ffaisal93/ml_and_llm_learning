# Topic 37: MLE and MAP Estimation

> 🔥 **For interviews, read these first:**
> - **`MLE_MAP_DEEP_DIVE.md`** — frontier-lab deep dive: full MLE derivations (Bernoulli/Gaussian/Poisson/multinomial/linreg/logreg), asymptotic theory (consistency/Fisher info/CRLB), MAP-as-regularization (ridge from Gaussian prior, lasso from Laplace), conjugate priors catalog, MLE = forward KL, RLHF/DPO connections.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions.

## What You'll Learn

This topic covers maximum likelihood and Bayesian estimation in detail:
- Maximum Likelihood Estimation (MLE) - detailed derivation
- Maximum A Posteriori (MAP) - detailed derivation
- Connection between MLE and MAP
- Bayesian vs Frequentist perspective
- L1/L2 Regularization as Bayesian Priors
- Intuitive explanations with examples
- When to use each approach

## Why We Need This

### Interview Importance
- **Common questions**: "Derive MLE", "Explain MAP", "MLE vs MAP"
- **Fundamental concepts**: Foundation of many ML algorithms
- **Bayesian understanding**: Essential for advanced topics

### Real-World Application
- **Parameter estimation**: How models learn from data
- **Regularization**: Understanding why regularization works
- **Uncertainty**: Bayesian methods provide uncertainty estimates

## Overview

**MLE (Maximum Likelihood Estimation):**
- Frequentist approach
- Find parameters that maximize probability of observed data
- No prior beliefs

**MAP (Maximum A Posteriori):**
- Bayesian approach
- Find parameters that maximize posterior probability
- Incorporates prior beliefs

**Key Insight:**
MAP = MLE + Prior
Regularization = MAP estimation

**Additional Topics:**
- **L1/L2 Priors**: Bayesian interpretation of regularization
  - L2 Regularization = Gaussian Prior (Ridge)
  - L1 Regularization = Laplace Prior (Lasso)
  - Detailed explanations and derivations

See `mle_map_derivations.md` for complete mathematical derivations!
See `regularization_priors.md` for detailed L1/L2 priors explanation!

## Core Intuition

MLE and MAP are two closely related ways to estimate parameters from data.

### MLE

MLE asks:

"Which parameter value makes the observed data most likely?"

It uses only the likelihood from the observed data.

### MAP

MAP asks:

"Which parameter value is most plausible after combining the data likelihood with a prior belief?"

That is why MAP is often summarized as:
- data fit
- plus prior preference

## Technical Details Interviewers Often Want

### Why MAP Connects to Regularization

This is one of the most important follow-ups.

- Gaussian prior -> L2-style penalty
- Laplace prior -> L1-style penalty

That is why regularization has a Bayesian interpretation.

### MLE vs MAP in Small Data

When data is limited, the prior in MAP can matter a lot because it stabilizes estimation.

With lots of data, the likelihood often dominates.

## Common Failure Modes

- describing MAP as totally different from MLE instead of closely related
- forgetting that MLE does not include prior beliefs
- not seeing the connection between priors and regularization
- saying Bayesian methods always outperform frequentist ones

## Edge Cases and Follow-Up Questions

1. Why is MAP often more stable than MLE with small data?
2. Why does L2 regularization correspond to a Gaussian prior?
3. Why does L1 correspond to a Laplace prior?
4. Why do MLE and MAP often become similar with enough data?
5. What is the practical meaning of the prior in MAP?

## What to Practice Saying Out Loud

1. The conceptual difference between MLE and MAP
2. Why regularization has a Bayesian interpretation
3. Why priors matter most when data is limited
