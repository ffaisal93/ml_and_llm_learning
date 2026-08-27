# Discriminative vs Generative Models — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

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

> **Saying it out loud.** The short version is that a discriminative model learns where the line between the classes goes, and a generative model learns what each class actually looks like. Logistic regression just asks "given this email, spam or not?" and never bothers to describe what spam is. Naive Bayes goes the other way: it builds a picture of spam emails and a picture of normal emails, then asks which picture this one fits better, using Bayes' rule to flip it back into a class probability. Both end up giving you $p(y|x)$, but only the generative one can turn around and make you a new fake email. The tradeoff is that describing the whole data distribution is a much harder job than drawing one boundary, so you pay for it whenever your assumptions about that distribution are wrong.

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

> **Saying it out loud.** Naive Bayes works despite an assumption that's obviously false because it only has to get the *ranking* of the classes right, not the actual probabilities. Words in an email are clearly not independent — "credit" and "card" travel together — so Naive Bayes double-counts that evidence and produces wildly overconfident numbers. But if it over-counts in the direction of the correct class, the argmax is still correct and you classify fine. That's why it's a great baseline for text and a terrible choice when you need calibrated probabilities. The failure mode to name is exactly that: correlated features give you accuracy that's fine and confidence scores that are near 0 or 1 and meaningless.

### Limitations
- Probabilities miscalibrated when independence violated.
- Can't learn feature interactions.
- Beaten by discriminative methods given enough data.

> **Saying it out loud.** Naive Bayes is the textbook generative classifier: model each class's word distribution, multiply by the class prior, pick the winner. You do it in log space because multiplying thousands of small probabilities underflows to zero, and you add Laplace smoothing because one unseen word would otherwise zero out an entire class. Training is basically counting — no gradient descent, one pass over the data — which is why it's still the thing you reach for when you need a spam filter by lunchtime. It's cheap, interpretable, and surprisingly strong on high-dimensional sparse text. The limitation to name is that it can never learn feature interactions, so once you have plenty of labeled data, logistic regression or a transformer will beat it.

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

> **Saying it out loud.** Here's the fun one: LDA and logistic regression both give you a straight-line decision boundary, but they're not the same model. LDA gets there by assuming each class is a Gaussian blob and that the two blobs share the same covariance — under that assumption the log-odds work out to be linear in $x$, which falls straight out of Bayes' rule. Logistic regression just declares the log-odds linear and fits the weights to maximize conditional likelihood, never saying anything about how $x$ is distributed. So LDA is the more efficient estimator when the Gaussian story is true, and the more biased one when it isn't. The number to quote is Ng and Jordan's: the generative model reaches its asymptotic error in about $O(\log d)$ samples where the discriminative one needs $O(d)$ — generative wins in the small-data regime, discriminative wins as $n$ grows.

---

## 4. Bayes optimal classifier

The classifier that minimizes 0-1 loss is:

$$
\hat{y}(x) = \arg\max_c p(y = c | x)
$$

This is the *theoretical best* — no classifier can do better in expectation. Discriminative models target this directly. Generative models reach it via Bayes' rule + prior.

The error of the Bayes classifier is the **Bayes error** — the irreducible error in the problem (unless features distinguish classes perfectly).

> **Saying it out loud.** The Bayes optimal classifier is just "always predict the most probable class given the features," assuming you magically knew the true $p(y|x)$. It's not a model you can build — it's the ceiling everything else is measured against. And the key point people miss is that its error usually isn't zero. If two genuinely different classes produce identical feature vectors sometimes, no classifier on earth can separate them, and that leftover is the Bayes error. So when someone says their model is at 8% error, the real question is how close that is to the irreducible floor — if the Bayes error is 7%, you're done and more data won't help you.

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

> **Saying it out loud.** Sample complexity is where the generative-versus-discriminative debate stops being philosophy. A generative model with the right assumptions needs roughly logarithmic samples in the feature dimension, because it's estimating a handful of per-class parameters rather than searching a big hypothesis space. A discriminative model needs samples linear in the dimension, but it doesn't care whether your distributional story is right. So the rule of thumb is: tiny dataset and a believable assumption, go generative; lots of data or no idea what the distribution looks like, go discriminative. The tradeoff has a name — bias versus variance in disguise: the generative model's assumption is a strong prior that cuts variance and adds bias you can't train away.

---

## 6. Hidden Markov Models — sequence generative

HMM models a sequence of observations $x_{1:T}$ via hidden states $z_{1:T}$:

$$
p(x_{1:T}, z_{1:T}) = p(z_1) \prod_t p(z_t | z_{t-1}) \prod_t p(x_t | z_t)
$$

Generative — models the joint of observations and latents.

Used for: speech recognition (pre-deep-learning), POS tagging, gene finding. Trained with EM (Baum-Welch). Inferred with Viterbi (max) or forward-backward (marginal).

Modern equivalents: encoder-decoder transformers replaced HMMs in most tasks.

> **Saying it out loud.** An HMM is the generative story for sequences: there's a hidden state marching along in a Markov chain, and at each step it emits an observation. Think of it as someone in another room flipping between a few biased coins according to a fixed transition table, and you only see the coin flips, not which coin. Because it models the joint over observations and hidden states, you can run it in both directions — Viterbi for the single most likely state path, forward-backward for per-step marginals, EM to learn the parameters without ever seeing a label. It ruled speech and POS tagging for two decades. The failure mode that killed it is the Markov assumption itself: state $t$ only sees state $t-1$, so it can't carry long-range context, which is precisely what attention gave us.

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

> **Saying it out loud.** Modern generative models all try to capture $p(x)$, but they differ in whether they'll actually show you the number. A VAE gives you an explicit but approximate density through the ELBO. A GAN gives you no density at all — just a sampler, trained by a discriminator that can't tell real from fake. Diffusion learns the score, the gradient of the log-density, and turns noise into data one small denoising step at a time. An LLM is the honest one: it factors the joint over a sequence into a product of next-token conditionals by the chain rule, so it gives you an exact likelihood. And that's the trap in the interview — people say LLMs are discriminative because each step is a classification over the vocabulary. Each step is, but the model is generative, because those conditionals multiply into the full $p(x_{1:T})$.

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

> **Saying it out loud.** In practice, if the task is "put a label on this," you go discriminative, full stop — that's basically every classifier and regressor shipped today. You go generative when you need something the boundary can't give you: actual samples, anomaly detection where a low $p(x)$ flags the outlier, semi-supervised learning where unlabeled data still teaches you about $p(x)$, or graceful handling of missing features because you can marginalize them out. That last set is the real argument for generative modeling, not accuracy. The tradeoff to name is compute and data: modeling the full distribution of high-dimensional inputs is far more expensive than drawing one boundary through them, which is why nobody trains a generative model just to do classification anymore.

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

> **Saying it out loud.** Most of the wrong answers here come from one confusion: people classify a model by what it outputs rather than by what it estimates. Naive Bayes outputs a class, so it feels discriminative, but it's generative because it models $p(x|y)$ and flips it. An LLM outputs one token at a time, which feels like classification, but it's generative because those conditionals compose into a joint. The clean test to say out loud is: could this model, on its own, produce a plausible new $x$? If yes, it's generative. And the other trap is "generative needs more data" — it's the opposite when the assumption holds; generative is the sample-efficient one, it just has a higher error floor when the assumption is wrong.

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
