# Discriminative vs Generative Models — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

The discriminative-vs-generative distinction is one of the most-asked classifier questions in interviews because it forces you to understand what a model is actually estimating. Many candidates conflate them; a clean answer earns points fast.

---

## 1. The fundamental distinction

A **classifier** maps $x \to y$. Two ways to do it:

### Discriminative

Model the conditional $p(y | x)$ directly. Optimize for the classification boundary.

Examples: logistic regression, SVMs, decision trees, neural networks, k-NN.

### Generative

Model the joint $p(x, y) = p(x | y) p(y)$. At inference, derive $p(y|x) \propto p(x|y) p(y)$ via Bayes' rule.

Examples: Naive Bayes, Gaussian Discriminant Analysis (GDA / LDA / QDA), Hidden Markov Models, GANs (sort of), VAEs (sort of), modern LLMs (text generative models).

### What's modeled
- Discriminative models the *boundary*; what makes class A vs B at $x$.
- Generative models the *data*; what each class's distribution looks like.

---

## 2. Naive Bayes — canonical generative classifier

Assumes features conditionally independent given the class:

$$
p(x | y) = \prod_j p(x_j | y)
$$

(The "naive" assumption.)

### Inference

$$
\hat{y} = \arg\max_y p(y) \prod_j p(x_j | y)
$$

Take logs to avoid underflow:

$$
\hat{y} = \arg\max_y \log p(y) + \sum_j \log p(x_j | y)
$$

### Training

For categorical features (e.g., text):

$$
p(x_j = v | y = c) = \frac{\#\{i : x_{i,j} = v, y_i = c\}}{\#\{i : y_i = c\}}
$$

Plus Laplace smoothing to avoid zero probabilities for unseen feature values.

For continuous features: model each $p(x_j | y)$ as Gaussian with class-specific mean and variance.

### Why it works despite the naive assumption

- Even when features aren't independent, *relative* class probabilities can still be ordered correctly.
- Especially good for high-dim sparse data (text classification with bag-of-words).
- Strong baseline; cheap; interpretable.

### Limitations
- Probabilities miscalibrated when independence violated.
- Can't learn feature interactions.
- Beaten by discriminative methods given enough data.

---

## 3. Gaussian Discriminant Analysis (GDA / LDA / QDA)

Assumes class-conditional distributions are Gaussian:

$$
p(x | y = c) = \mathcal{N}(x | \mu_c, \Sigma_c)
$$

Plus class prior $p(y = c) = \pi_c$.

### LDA (Linear Discriminant Analysis)

Special case where all classes share covariance: $\Sigma_c = \Sigma$. Decision boundary is **linear** in $x$.

Bayes rule gives:

$$
\log \frac{p(y=1|x)}{p(y=0|x)} = (\mu_1 - \mu_0)^\top \Sigma^{-1} x + \mathrm{const}
$$

Linear in $x$! LDA is a linear classifier *equivalent in form* to logistic regression — but trained differently.

### QDA (Quadratic Discriminant Analysis)

Class-specific covariance $\Sigma_c$. Decision boundary is **quadratic**.

### LDA vs logistic regression

Both produce linear classifiers. Different trained models. **Ng & Jordan (2002)** is the canonical reference but their analysis was on **Naive Bayes vs logistic regression** specifically (not LDA — same generative-vs-discriminative spirit but different model pair). Their result:
- The generative model (NB) has higher asymptotic error if its independence assumption is wrong.
- The generative model converges to its asymptote with $O(\log d)$ samples (where $d$ is feature dimension); the discriminative model (LR) needs $O(d)$.
- Discriminative wins asymptotically (large $n$); generative wins for small $n$ or when the assumption is approximately right.

LDA-vs-LR shares the same flavor: LDA is generative, requires the Gaussian assumption to be optimal, and is more sample-efficient when correct.

---

## 4. Bayes optimal classifier

The classifier that minimizes 0-1 loss is:

$$
\hat{y}(x) = \arg\max_c p(y = c | x)
$$

This is the *theoretical best* — no classifier can do better in expectation. Discriminative models target this directly. Generative models reach it via Bayes' rule + prior.

The error of the Bayes classifier is the **Bayes error** — the irreducible error in the problem (unless features distinguish classes perfectly).

---

## 5. Sample complexity comparison

For learning to error $\epsilon$:
- Discriminative: $O(d/\epsilon^2)$ samples (linear in feature dim).
- Generative (with correct distributional assumption): $O(\log d / \epsilon^2)$ — logarithmic in $d$.

But: generative requires the assumption to be correct. When wrong, asymptotic error is higher. Discriminative is robust to model misspecification.

### Practical rule
- Small data + reasonable distributional assumption → generative.
- Large data + want robustness → discriminative.
- Modern deep learning → almost always discriminative (or generative in the LLM sense, which is different).

---

## 6. Hidden Markov Models — sequence generative

HMM models a sequence of observations $x_{1:T}$ via hidden states $z_{1:T}$:

$$
p(x_{1:T}, z_{1:T}) = p(z_1) \prod_t p(z_t | z_{t-1}) \prod_t p(x_t | z_t)
$$

Generative — models the joint of observations and latents.

Used for: speech recognition (pre-deep-learning), POS tagging, gene finding. Trained with EM (Baum-Welch). Inferred with Viterbi (max) or forward-backward (marginal).

Modern equivalents: encoder-decoder transformers replaced HMMs in most tasks.

---

## 7. Modern generative models — VAEs, GANs, diffusion, LLMs

### VAEs

Model $p(x) = \int p(x|z) p(z) dz$ with neural decoder $p(x|z)$ and prior $p(z) = \mathcal{N}(0, I)$. Trained with ELBO (variational lower bound).

### GANs

Generator + discriminator. Generator learns to sample from $p(x)$ implicitly, no explicit density.

### Diffusion

Forward noising + reverse denoising. Implicit density via score matching.

### LLMs

Next-token prediction $p(x_t | x_{<t})$. Models the joint $p(x_{1:T}) = \prod_t p(x_t | x_{<t})$ via chain rule. Generative in the strict sense.

### Discriminative interpretation of LLMs?

Each token prediction is a discriminative classification (which token next?). But the *model* is generative because it factorizes the full distribution.

---

## 8. When does each win?

### Discriminative wins
- Large data.
- Don't need to generate $x$.
- $p(x)$ is high-dim and complex.
- Robustness to model misspecification.

### Generative wins
- Small data + good distributional assumption.
- Want to generate samples (image gen, language gen).
- Anomaly detection (low $p(x)$ = outlier).
- Semi-supervised learning (use unlabeled $x$ to refine $p(x)$).
- Missing data handling (marginalize easily).

### Modern deep learning landscape
- Classification / regression: almost always discriminative.
- Image / text / audio synthesis: generative (diffusion, LLMs).
- Embedding / contrastive: somewhere in between (learn representations that can be used for either).

---

## 9. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Naive Bayes is what kind of model? | Discriminative | Generative — models $p(x|y)$ and uses Bayes to get $p(y|x)$ |
| Logistic regression is generative? | Yes | Discriminative — directly models $p(y|x)$ |
| LDA is the same as logistic regression? | Yes | Same form (linear), different training; LDA is generative |
| Why does Naive Bayes work despite the assumption? | Independence is OK | Relative class probabilities can be correctly ordered even with violations |
| Modern LLM is discriminative? | Yes | Generative — models $p(x_{1:T})$ via chain rule |
| Bayes optimal classifier — what does it do? | Predicts perfectly | Achieves Bayes error (irreducible); no classifier can beat it |
| Generative needs more data? | Always | Less data when assumption correct; more when wrong |

---

## 10. Eight most-asked interview questions

1. **Discriminative vs generative — define both.** ($p(y|x)$ vs $p(x, y)$.)
2. **Is logistic regression generative?** (No — discriminative.)
3. **LDA vs logistic regression?** (Same linear boundary, different training, generative vs discriminative.)
4. **Walk through Naive Bayes for text classification.** (Multinomial $p(x|y)$, Bayes for $p(y|x)$, log probs, Laplace smoothing.)
5. **When does Naive Bayes outperform logistic regression?** (Small data, when independence assumption isn't too violated.)
6. **What's the Bayes optimal classifier?** ($\arg\max_c p(c|x)$; achieves Bayes error.)
7. **Sample complexity: generative vs discriminative?** (Generative $O(\log d)$ if correct; discriminative $O(d)$ but robust.)
8. **Why are modern image / text models generative?** (To produce samples; image gen, language gen needs to model $p(x)$.)

---

## 11. Drill plan

- For each of: Naive Bayes, GDA, LDA, QDA, logistic regression, SVM — recite generative or discriminative + key assumption.
- Derive Naive Bayes log-likelihood for text classification.
- Show LDA decision boundary is linear under shared covariance.
- Recite Ng & Jordan's result on discriminative vs generative sample complexity.
- For each of: VAE, GAN, diffusion, LLM — explain how they model $p(x)$.

---

## 12. Further reading

- Ng & Jordan (2002), *On Discriminative vs Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes.*
- Bishop, *Pattern Recognition and Machine Learning*, ch. 4 — discriminative classifiers; ch. 8 — generative.
- Murphy, *Machine Learning: A Probabilistic Perspective*, ch. 7–8.
- Hastie, Tibshirani, Friedman, *Elements of Statistical Learning*, ch. 4 — LDA, logistic regression, comparison.
