# Discriminative vs Generative — Interview Grill

> 40 questions on the D vs G distinction, Naive Bayes, LDA/QDA, sample complexity, modern generative models. Drill until you can answer 28+ cold.

---

## A. The distinction

**1. Discriminative model — what does it estimate?**
$p(y | x)$ directly.

**2. Generative model — what does it estimate?**
$p(x, y) = p(x|y) p(y)$. Inference via Bayes: $p(y|x) \propto p(x|y) p(y)$.

**3. Examples of discriminative classifiers?**
Logistic regression, SVM, decision tree, random forest, kNN, neural network classifier.

**4. Examples of generative classifiers?**
Naive Bayes, LDA / QDA, Hidden Markov Model, Gaussian discriminant analysis.

**5. Bayes optimal classifier?**
$\hat{y}(x) = \arg\max_c p(y=c|x)$. Minimum 0-1 loss; achieves Bayes error.

**6. Bayes error?**
Irreducible error: $1 - \max_c p(c|x)$ averaged over $x$. Cannot be beaten.

---

## B. Naive Bayes

**7. Naive assumption?**
Features conditionally independent given class: $p(x|y) = \prod_j p(x_j|y)$.

**8. NB inference rule?**
$\hat{y} = \arg\max_y \log p(y) + \sum_j \log p(x_j|y)$.

**9. NB for text — what's $p(x_j|y)$?**
Multinomial / categorical — $p(x_j = v|y) = \mathrm{count}(v, y)/\mathrm{count}(y)$, with Laplace smoothing.

**10. Why Laplace smoothing?**
Avoid zero probabilities for unseen feature values, which would make all class probabilities zero.

**11. NB for continuous features?**
Model each $p(x_j|y)$ as Gaussian with class-specific mean/variance. Equivalent to special-case GDA with diagonal covariance.

**12. Why does NB work despite the naive assumption?**
Doesn't need correct probabilities — just correct *ranking* of classes. Often robust to dependence violations.

**13. NB strengths?**
Cheap, scales to high dimensions, strong text-classification baseline, works with little data.

**14. NB weaknesses?**
Miscalibrated probabilities. Can't capture feature interactions. Beaten by discriminative methods at scale.

---

## C. GDA / LDA / QDA

**15. GDA assumption?**
Each class's feature distribution is multivariate Gaussian.

**16. LDA — what's the additional assumption?**
All classes share a single covariance matrix: $\Sigma_c = \Sigma$.

**17. LDA decision boundary shape?**
Linear in $x$. Same form as logistic regression.

**18. QDA decision boundary?**
Quadratic. Class-specific covariances → quadratic terms in $x$.

**19. LDA derivation key step?**
$\log \frac{p(y=1|x)}{p(y=0|x)} = (\mu_1 - \mu_0)^\top \Sigma^{-1} x + \mathrm{const}$. Linear in $x$.

**20. LDA vs logistic regression — same model?**
Same linear functional form. Different parameter estimation: LDA fits Gaussian per class; logistic regression directly fits the conditional.

**21. Ng & Jordan result?**
For Naive Bayes vs Logistic Regression specifically: NB converges to its asymptote with $O(\log d)$ samples (in feature dimension $d$); LR needs $O(d)$. NB wins for small data when the independence assumption is reasonable; LR wins asymptotically and when the assumption is wrong.

---

## D. Sample complexity and trade-offs

**22. When prefer generative?**
Small dataset; reasonable distributional assumption; want to generate samples; anomaly detection.

**23. When prefer discriminative?**
Large dataset; complex feature distribution; primary goal is classification accuracy.

**24. Why is generative more sample-efficient when right?**
Uses parametric structure of $p(x|y)$; fewer effective parameters. Discriminative ignores $p(x)$ entirely.

**25. Why is discriminative more robust?**
Doesn't depend on getting $p(x|y)$ right. Just needs the conditional boundary correct.

---

## E. Hidden Markov Models

**26. HMM — what does it model?**
Joint distribution over observed sequence $x_{1:T}$ and hidden states $z_{1:T}$.

**27. HMM Markov assumption?**
$z_t$ depends only on $z_{t-1}$. $x_t$ depends only on $z_t$.

**28. HMM training algorithm?**
Baum-Welch (special case of EM).

**29. HMM inference — most likely state sequence?**
Viterbi algorithm.

**30. HMM inference — marginal $p(z_t|x_{1:T})$?**
Forward-backward algorithm.

**31. Why are HMMs less used now?**
Replaced by RNN/transformer encoder-decoders for most tasks. Still niche in some signal processing.

---

## F. Modern generative models

**32. VAE — what does it estimate?**
$p(x) = \int p(x|z)p(z)dz$ via amortized inference $q(z|x)$. Trained with ELBO.

**33. GAN — explicit density?**
No. Implicit generator; samples from $p(x)$ but no density evaluation.

**34. Diffusion — what does it model?**
Forward noising → reverse denoising. Score-based: learns $\nabla \log p(x_t)$.

**35. LLM as generative?**
Yes. $p(x_{1:T}) = \prod_t p(x_t|x_{<t})$ via chain rule. Each conditional is autoregressive.

**36. Are LLMs technically discriminative on per-token level?**
Each token prediction is a softmax classification. But the full model produces a distribution over sequences — generative.

---

## G. Subtleties

**37. Why doesn't discriminative training give you $p(x)$?**
Discriminative models $p(y|x)$ — doesn't require knowing $p(x)$. Marginalizing back gives nothing useful.

**38. Why does generative help with missing features?**
With $p(x|y)$ known, missing $x_j$ can be marginalized out. Discriminative struggles unless trained with imputation.

**39. Semi-supervised learning?**
Generative naturally uses unlabeled $x$ to refine $p(x)$. Helps when labels are scarce.

**40. Anomaly detection?**
Low $p(x)$ = anomaly. Generative naturally gives this. Discriminative requires explicit "outlier" class.

---

## Quick fire

**41.** *Logistic regression — D or G?* D.
**42.** *Naive Bayes — D or G?* G.
**43.** *LDA — D or G?* G.
**44.** *SVM — D or G?* D.
**45.** *VAE — D or G?* G.
**46.** *Bayes optimal classifier?* $\arg\max p(y|x)$.
**47.** *NB feature assumption?* Conditional independence given class.
**48.** *LDA boundary?* Linear.
**49.** *QDA boundary?* Quadratic.
**50.** *LLM — D or G?* G (generative; chain rule of conditionals).

---

## Self-grading

If you can't answer 1-15, you don't know D vs G. If you can't answer 16-30, you'll struggle on classifier theory questions. If you can't answer 31-40, frontier-lab questions on probabilistic modeling will go past you.

Aim for 30+/50 cold.
