# Topic 37: MLE and MAP Estimation

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

