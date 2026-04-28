# Information Theory: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Information theory is the language ML uses to talk about loss functions, compression, generalization, and divergences. Strong candidates can move fluidly between cross-entropy as a loss, KL as a divergence, mutual information as a model objective, and the connections among them. This document is the bridge.

---

## 1. Entropy: the central quantity

For a discrete distribution $p$ over outcomes:

$$
H(p) = -\sum_x p(x) \log p(x) \qquad (\text{log base 2 = bits, log base } e = \text{nats})
$$

Intuition: the average number of bits (or nats) needed to encode an outcome drawn from $p$. Equivalently, the average "surprise" of an outcome.

### Properties

**Max entropy = uniform.** $H(p) \leq \log |\mathcal{X}|$, with equality iff $p$ is uniform.

**Min entropy = deterministic.** $H(p) \geq 0$, with equality iff $p$ is a point mass.

**Concave in $p$.** If you average two distributions, you get higher entropy than the average of their entropies.

**Additivity for independent variables.** $H(X, Y) = H(X) + H(Y)$ if $X \perp Y$.

### What "entropy" means in different contexts

- **Statistics.** Spread of a distribution.
- **Coding.** Lower bound on average code length (Shannon's source coding theorem).
- **Physics.** Disorder; thermodynamic entropy.
- **ML.** How "uncertain" a model is.

---

## 2. Cross-entropy

For two distributions $p$ (true) and $q$ (model):

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

Average code length when encoding samples from $p$ using a code optimal for $q$. Bounded below by $H(p)$ (you can't do better than the entropy of the true distribution).

### The cross-entropy = entropy + KL identity

$$
H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q)
$$

$H(p)$ is fixed (it's a property of the data). Minimizing $H(p, q)$ over $q$ is equivalent to minimizing $\mathrm{KL}(p \,\|\, q)$. **This is why cross-entropy is the standard ML loss** — it's KL up to a constant.

### Cross-entropy in deep learning

For one-hot labels ($p$ is a delta on the true class):

$$
H(p, q) = -\log q(\text{true class})
$$

This is exactly the negative log-likelihood. So "cross-entropy loss" = "NLL" = "MLE" — three names for the same loss in the discrete-label case. Different generative assumptions give different losses (Gaussian → MSE), but for classification, cross-entropy is mandated by maximum likelihood under the categorical distribution.

---

## 3. KL divergence

$$
\mathrm{KL}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}[\log p(x) - \log q(x)] = H(p, q) - H(p)
$$

Measures how $q$ differs from $p$ "from $p$'s perspective."

### Properties

**Non-negative.** $\mathrm{KL}(p \,\|\, q) \geq 0$, with equality iff $p = q$. Direct consequence of Jensen's inequality applied to $-\log$.

**Asymmetric.** $\mathrm{KL}(p \,\|\, q) \neq \mathrm{KL}(q \,\|\, p)$ in general. **Not a distance.**

**Not a metric.** Triangle inequality fails. Don't think of KL as a distance; it's a divergence.

**Coordinate-invariant.** Reparameterize $x \to f(x)$ for invertible $f$; KL is unchanged. Important for deriving properties of distributions.

### Forward vs reverse KL: why direction matters

**Forward KL $\mathrm{KL}(p \,\|\, q)$** ("mean-seeking"). Penalizes $q$ heavily where $p$ has mass and $q$ doesn't. Encourages $q$ to cover all modes of $p$. If $q$ is restricted to a simpler family (e.g. unimodal Gaussian fitting a multimodal $p$), forward KL spreads $q$ to cover everything — high entropy, mean-seeking.

**Reverse KL $\mathrm{KL}(q \,\|\, p)$** ("mode-seeking"). Penalizes $q$ where $q$ has mass but $p$ doesn't. Encourages $q$ to fit one mode of $p$ well, ignoring others. Low entropy, mode-seeking.

**Why this matters for ML:**

- MLE / cross-entropy training is **forward KL**: $\mathrm{KL}(\text{data} \,\|\, \text{model}) = \mathrm{KL}(p \,\|\, q)$. Makes the model cover the data distribution. Models trained this way often produce "average-looking" outputs.
- Variational inference / RL with KL regularization is sometimes **reverse KL**: $\mathrm{KL}(\text{model} \,\|\, \text{prior})$. Makes the model concentrate on a mode.
- **GANs** approximately minimize Jensen-Shannon (a symmetric average of forward and reverse KL).

Frontier-lab interview gotcha: "Why does an MLE-trained model tend to produce average outputs?" Forward-KL is mean-seeking.

---

## 4. Mutual information

$$
I(X; Y) = \mathrm{KL}\!\big(P(X, Y) \,\|\, P(X)\, P(Y)\big) = H(X) + H(Y) - H(X, Y) = H(Y) - H(Y \mid X) = H(X) - H(X \mid Y)
$$

How much knowing $Y$ reduces uncertainty about $X$ (and vice versa).

### Properties

- $I(X; Y) \geq 0$.
- $I(X; Y) = 0$ iff $X \perp Y$.
- $I(X; X) = H(X)$.
- Symmetric: $I(X; Y) = I(Y; X)$.

### Why it matters in ML

- **Information bottleneck:** train representations $Z$ that maximize $I(Y; Z)$ (predictive of label) while minimizing $I(X; Z)$ (compressing input). A theoretical framework for understanding "good" representations.
- **Self-supervised learning.** Many SSL objectives (InfoNCE, contrastive losses) are lower bounds on mutual information.
- **Disentanglement.** Maximizing $I$ between latent dimensions and meaningful factors.

### InfoNCE (van den Oord et al. 2018)

The standard contrastive loss:

$$
\mathcal{L} = -\mathbb{E}\!\left[\log \frac{\exp f(x, y_+)}{\sum_i \exp f(x, y_i)}\right]
$$

where $y_+$ is the positive (correct) pair and $y_i$ are negatives. This is a lower bound on $I(X; Y_+)$. Used in CLIP, MoCo, SimCLR, and modern embedding models.

---

## 5. Conditional and joint entropy

$$
H(X \mid Y) = -\sum_{x, y} p(x, y) \log p(x \mid y) \qquad \text{(conditional entropy)}
$$

$$
H(X, Y) = -\sum_{x, y} p(x, y) \log p(x, y) \qquad \text{(joint entropy)}
$$

**Chain rule:** $H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)$.

Conditional entropy is the average uncertainty about $X$ given that $Y$ is known. Always between 0 and $H(X)$.

These are useful for decomposing information flow in models. E.g., $H(\text{target} \mid \text{input})$ is the **irreducible noise** any model must contend with — a lower bound on cross-entropy loss.

---

## 6. KL in machine learning

KL appears in many places.

### Maximum likelihood = forward KL minimization

Already covered. $\arg\min_\theta H(p_{\text{data}}, p_\theta) = \arg\min_\theta \mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)$.

### Variational inference / VAE

The Evidence Lower Bound (ELBO):

$$
\log p(x) \geq \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - \mathrm{KL}\!\big(q(z \mid x) \,\|\, p(z)\big)
$$

The first term is the reconstruction; the second is a KL penalty against the prior. This is why VAEs have a "KL term."

### RLHF / PPO regularization

The RLHF objective:

$$
\max_\pi \mathbb{E}[r(x, y)] - \beta \cdot \mathrm{KL}\!\big(\pi \,\|\, \pi_{\text{ref}}\big)
$$

The KL anchor prevents the policy from drifting too far from the reference. Same idea in TRPO, PPO with KL formulation.

### DPO derivation

The closed-form solution to the KL-regularized RL objective, which becomes the basis for DPO. See `08_training_techniques/ALIGNMENT_DEEP_DIVE.md`.

### Knowledge distillation

Train a student model to match a teacher's distribution by minimizing $\mathrm{KL}(p_{\text{teacher}} \,\|\, p_{\text{student}})$. The student inherits the teacher's confidence pattern, not just hard predictions.

---

## 7. Other divergences

KL is one of many.

### Jensen-Shannon (JS) divergence

$$
\mathrm{JS}(p, q) = \tfrac{1}{2} \mathrm{KL}(p \,\|\, M) + \tfrac{1}{2} \mathrm{KL}(q \,\|\, M), \qquad M = \tfrac{p + q}{2}
$$

Symmetric. Bounded $\mathrm{JS} \in [0, \log 2]$. Square root of JS is a metric.

### f-divergences

General family $D_f(p \,\|\, q) = \sum_x q(x)\, f(p(x)/q(x))$. KL: $f(t) = t \log t$. JS, $\chi^2$, total variation are all f-divergences with different $f$.

### Wasserstein distance

A different family entirely (optimal transport). Considers the geometry of the underlying space (not just distribution mass). Used in WGAN, optimal transport, distribution matching.

### Total variation

$$
\mathrm{TV}(p, q) = \tfrac{1}{2} \sum_x |p(x) - q(x)|
$$

The maximum probability of distinguishing $p$ and $q$ by any test. **Pinsker's inequality:** $\mathrm{TV}(p, q) \leq \sqrt{\mathrm{KL}(p \,\|\, q) / 2}$ — bounding TV by KL.

---

## 8. Cross-entropy in detail

For a softmax classifier with logits $z$:

$$
p_\theta(\text{class} \mid \text{input}) = \mathrm{softmax}(z) = \frac{\exp(z)}{\sum_j \exp(z_j)}
$$

$$
\mathcal{L} = -\log p_\theta(\text{true class}) = -z_{\text{true}} + \log \sum_j \exp(z_j)
$$

The $\log \sum_j \exp(z_j)$ is the log-partition function (also called log-sum-exp). Numerically computed via:

$$
\mathrm{LSE}(z) = \max(z) + \log \sum_j \exp(z_j - \max(z))
$$

### Gradient w.r.t. logits

$$
\frac{\partial \mathcal{L}}{\partial z} = \mathrm{softmax}(z) - \mathbf{1}_{\text{true class}} = p_\theta - y
$$

This is the famous "logits minus targets" gradient. It's the canonical-link gradient for the categorical distribution in GLM theory. Same form as logistic regression's $(\sigma - y)$ extended to $K$ classes.

---

## 9. Perplexity

$$
\mathrm{PPL} = \exp(H(p, q)) = \exp(\text{cross-entropy})
$$

Geometric inverse of average per-token probability. Lower perplexity = better model.

### Bounds

- Lower bound: $\exp(H(p))$ (true entropy of the data). A perfect LM would have $\mathrm{PPL} \approx \exp(H_{\text{data}})$.
- Upper bound: $|V|$ (vocabulary size, if the model is uniform random).

### Tokenizer dependence

Perplexity depends on tokenization. Same text, different tokenizer, different PPL. Cannot directly compare across tokenizers — see `03_evaluation_metrics/EVALUATION_METRICS_DEEP_DIVE.md`.

---

## 10. Information bottleneck

A theoretical framework (Tishby et al. 2000) proposing that good representations $Z$ of input $X$ for predicting label $Y$:

- Maximize $I(Y; Z)$ (predictive of label).
- Minimize $I(X; Z)$ (compress input — "throw away irrelevant information").

$$
\mathcal{L}_{\text{IB}} = I(Y; Z) - \beta \cdot I(X; Z)
$$

Empirically, deep networks trained with cross-entropy seem to (approximately) follow this trajectory: early layers compress the input; later layers preserve task-relevant information. Whether IB is the *right* explanation for deep learning's success is debated.

---

## 11. Source coding theorem (Shannon)

The minimum average bits per symbol needed to losslessly encode samples from $p$ is $H(p)$. **You cannot compress below entropy.**

Practical relevance for ML:

- Cross-entropy $H(p, q)$ is the **average code length** if you use a code optimal for $q$ to encode samples from $p$. Always $\geq H(p)$.
- Minimizing cross-entropy = building a near-optimal compressor for the data.
- LLMs are essentially lossy compressors of their training data. Better LM → better compression.

A very recent line of research (Deletang et al., "Language Modeling is Compression") makes this explicit: SOTA LLMs can compress text *better* than gzip.

---

## 12. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "Is KL a distance?" | No. Asymmetric, doesn't satisfy triangle inequality. It's a divergence. |
| "Why minimize cross-entropy?" | It's MLE under categorical. Equivalently, it's $\mathrm{KL}(\text{data} \,\|\, \text{model})$ up to the data entropy constant. |
| "Forward vs reverse KL?" | Forward ($\mathrm{KL}(p \,\|\, q)$): mean-seeking; $q$ covers $p$. Reverse ($\mathrm{KL}(q \,\|\, p)$): mode-seeking; $q$ fits one mode. MLE = forward. |
| "What's the KL between identical distributions?" | 0. $\mathrm{KL}(p \,\|\, p) = 0$. |
| "Can KL be infinite?" | Yes. $\mathrm{KL}(p \,\|\, q) = \infty$ if there's a region where $p > 0$ but $q = 0$. (You're "infinitely surprised" by a sample assigned probability 0.) |
| "What's mutual information?" | KL between joint and product of marginals. Measures statistical dependence. |
| "When are KL and cross-entropy the same?" | When $H(p)$ is fixed (i.e., during training, where the data distribution doesn't change), minimizing cross-entropy = minimizing KL. |
| "What's perplexity?" | $\exp(\text{cross-entropy})$. Inverse geometric average per-token probability. Tokenizer-dependent. |

---

## 13. The 10 most-asked information theory interview questions

1. **Define entropy.** $H(p) = -\sum p \log p$. Average surprise / coding length.
2. **Define cross-entropy.** $H(p, q) = -\sum p \log q$. Coding length using $q$-optimal code on samples from $p$.
3. **Cross-entropy = entropy + KL.** $H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q)$. Why minimizing cross-entropy = minimizing KL.
4. **Define KL divergence.** $\mathrm{KL}(p \,\|\, q) = \sum p \log(p/q)$. Non-negative, asymmetric, not a metric.
5. **Forward vs reverse KL.** Forward: mean-seeking. Reverse: mode-seeking.
6. **Mutual information.** $I(X; Y) = H(X) + H(Y) - H(X, Y)$. Statistical dependence.
7. **Why is MLE = cross-entropy?** Cross-entropy is the negative log-likelihood under categorical; MLE is $\arg\max \log P(\text{data} \mid \theta) = \arg\min H(p_{\text{data}}, p_\theta)$.
8. **Perplexity?** $\exp(\text{cross-entropy})$. Tokenizer-dependent.
9. **KL in RLHF?** Penalty $\beta \cdot \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ prevents policy from drifting from reference.
10. **What's the source coding theorem?** Average code length $\geq$ entropy. Cross-entropy is the loss because it's compressibility under the model.

---

## 14. Drill plan

1. Whiteboard $H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q)$ derivation.
2. Walk through forward vs reverse KL with multimodal-vs-unimodal example.
3. Show MI = $\mathrm{KL}(\text{joint} \,\|\, \text{marginals product})$.
4. Connect cross-entropy to MLE under categorical.
5. Drill `INTERVIEW_GRILL.md`.

---

## 15. Further reading

- Cover & Thomas, *Elements of Information Theory* (the textbook).
- Shannon, "A Mathematical Theory of Communication" (1948) — the founding paper.
- Tishby et al., "The Information Bottleneck Method" (2000).
- van den Oord et al., "Representation Learning with Contrastive Predictive Coding" (InfoNCE, 2018).
- Deletang et al., "Language Modeling is Compression" (2023).
