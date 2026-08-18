# Information Theory: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Information theory is the language ML uses to talk about loss functions, compression, generalization, and divergences. Strong candidates can move fluidly between cross-entropy as a loss, KL as a divergence, mutual information as a model objective, and the connections among them. This document is the bridge.

---

## 1. Entropy: the central quantity

**In plain terms.** Entropy is a number that says how surprised you should expect to be. If you already know what's going to happen, it's zero. If everything is equally likely, it's as big as it can get. The formula below is just the average surprise, where surprise is measured in yes/no questions.

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

> **Saying it out loud.** Entropy is the average number of yes-or-no questions you'd need to pin down an outcome. A fair coin is one bit, because one question does it. A coin that lands heads ninety-nine percent of the time is only about eight hundredths of a bit, because you'd guess heads and almost always be right — there's nothing to ask. The analogy I like is packing a suitcase: entropy is how much space the outcome genuinely takes up once you've packed it optimally, and no clever encoding gets you below that. Two anchors worth remembering: entropy is maximized by the uniform distribution and it's zero for a certainty, so it's really just a measure of how spread out your beliefs are.

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

> **Saying it out loud.** Cross-entropy is what it costs you to encode reality using the wrong beliefs. You built your compression scheme assuming distribution q, but the world actually produces samples from p, so your messages come out longer than they had to. The gap is exactly the KL divergence, and since the true entropy of the data is a fixed constant you can't touch, minimizing cross-entropy over your model is the same thing as minimizing KL. That identity is the reason cross-entropy is the default loss in all of machine learning — you're not choosing it for convenience, it's maximum likelihood written in information-theoretic notation. And with a one-hot label it collapses to just the negative log of the probability you assigned to the right answer, which is why the loss punishes confident mistakes so brutally.

---

## 3. KL divergence

**In plain terms.** KL divergence measures how much worse off you are for believing q when the truth is p. It's zero only when they match, it's never negative, and it's not symmetric — being wrong in one direction is not the same as being wrong in the other. It's a penalty, not a distance.

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

> **Saying it out loud.** KL divergence is the price of being wrong, measured in extra bits. If the world runs on p and you plan for q, KL tells you how much longer your messages get — zero when you're exactly right, and never negative. The thing every interviewer probes is that it's asymmetric, so it is emphatically not a distance: mixing up the order gives you a different number and different behavior. Forward KL, the direction maximum likelihood uses, punishes you for putting no probability where the data actually appears, so your model spreads out to cover everything — that's mean-seeking, and it's why an MLE-trained model asked to fit two peaks will sit unhappily in the valley between them. Reverse KL punishes you for putting probability where the data isn't, so it clamps onto a single mode and ignores the rest. That one asymmetry explains why MLE models produce bland average outputs and why variational methods produce overconfident narrow ones.

---

## 4. Mutual information

**In plain terms.** Mutual information asks: how much does learning one thing tell you about another? Zero means knowing Y tells you nothing about X. The larger it is, the more of your uncertainty about X gets erased once you see Y.

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

> **Saying it out loud.** Mutual information is the amount of uncertainty about one variable that gets erased when you learn another. Formally it's the KL divergence between the true joint distribution and what the joint would look like if the two were independent — so it's literally measuring how far from independent they are. Unlike correlation, it catches any kind of dependence, including non-linear ones that correlation reads as zero. In ML it shows up in two big places: the information bottleneck, which says a good representation keeps what predicts the label and throws away everything else about the input, and contrastive learning, where InfoNCE — the loss behind CLIP and SimCLR — is provably a lower bound on mutual information. The honest caveat is that mutual information is notoriously hard to estimate from samples in high dimensions, and InfoNCE's bound is capped at the log of the batch size, which is exactly why contrastive methods want enormous batches.

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

> **Saying it out loud.** Conditional entropy is how uncertain you still are about X after you've been told Y. The chain rule just says total uncertainty about a pair equals uncertainty about the first plus whatever's left about the second — you can decompose it in either order and get the same total. The reason this matters practically is that conditional entropy of the target given the input is the irreducible noise in your problem: it's the part no model can ever predict, so it's a hard floor on your cross-entropy loss. When your loss plateaus, the question is always whether you've hit that floor or whether your model just isn't good enough, and knowing the distinction saves you from chasing an accuracy number that's mathematically unreachable.

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

> **Saying it out loud.** Once you see KL you start seeing it everywhere, and being able to list the places is a strong senior signal. Ordinary supervised training is forward-KL minimization between the data distribution and the model. The VAE objective has an explicit KL term pulling the encoder toward the prior, which is what keeps the latent space usable rather than a lookup table. RLHF adds a KL penalty against the reference model, and that anchor is the only thing stopping the policy from reward-hacking its way into gibberish that scores highly. DPO comes from solving that KL-regularized objective in closed form. And distillation is KL from teacher to student, which is why the student inherits the teacher's uncertainty pattern and not just its top-1 answers. Same quantity, five very different jobs.

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

> **Saying it out loud.** KL is the famous one but it isn't the only option, and the differences matter when distributions barely overlap. Jensen-Shannon is the symmetric version — compare both to their average — and it's bounded by log two, which is exactly why it doesn't blow up when supports are disjoint. That boundedness is also its problem: for two distributions with no overlap, JS is pegged at its maximum and its gradient is flat, which is the vanishing-gradient story behind early GAN training. Wasserstein is a genuinely different animal from optimal transport — it asks how much earth you'd have to move, so it respects the geometry of the space and still gives you a useful gradient when the distributions don't overlap, which is what WGAN exploited. And Pinsker's inequality is the bridge worth remembering: total variation is bounded by the square root of half the KL, so controlling KL controls how distinguishable two distributions are.

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

> **Saying it out loud.** If you differentiate cross-entropy with respect to the logits, everything collapses to predicted probability minus the one-hot target. That's the whole result, and it's worth being able to say why: softmax is the canonical link for the categorical distribution, so the derivative of the activation exactly cancels the derivative of the loss. The practical consequences are two. First, gradients are large exactly when you're confidently wrong, which is the behavior you want and the opposite of what you'd get pairing softmax with squared error. Second, the log-sum-exp term needs the max subtracted before exponentiating or you overflow, which is why frameworks fuse softmax and cross-entropy into a single numerically-stable op instead of letting you compose them.

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

> **Saying it out loud.** Perplexity is just cross-entropy exponentiated, and the intuitive reading is "how many options is the model effectively choosing between at each token." A perplexity of twenty means the model is about as confused as if it were picking uniformly among twenty words. That makes it a nicer number to talk about than a loss in nats. Two bounds anchor it: the ceiling is your vocabulary size, which is what a completely uniform model scores, and the floor is the true entropy of the text, which nobody can beat. The trap to always name is tokenizer dependence — the same model on the same text gets different perplexity under a different tokenizer, so comparing perplexity across models with different vocabularies is meaningless.

---

## 10. Information bottleneck

A theoretical framework (Tishby et al. 2000) proposing that good representations $Z$ of input $X$ for predicting label $Y$:

- Maximize $I(Y; Z)$ (predictive of label).
- Minimize $I(X; Z)$ (compress input — "throw away irrelevant information").

$$
\mathcal{L}_{\text{IB}} = I(Y; Z) - \beta \cdot I(X; Z)
$$

Empirically, deep networks trained with cross-entropy seem to (approximately) follow this trajectory: early layers compress the input; later layers preserve task-relevant information. Whether IB is the *right* explanation for deep learning's success is debated.

> **Saying it out loud.** The information bottleneck says a good representation is one that's forgotten as much as possible while still predicting the label. You want to maximize the information the representation carries about the target and simultaneously minimize what it carries about the raw input — compression and prediction traded off by a coefficient. It's an appealing story for what deep networks do: early layers throw away pixels and phrasing, later layers keep whatever the task needs. The honest caveat, and worth saying because it shows you've read the debate, is that the empirical evidence for the famous compression phase is contested — it seems to depend on the activation function used to measure it, and it doesn't reproduce for ReLU networks. So I'd present it as a useful lens, not as an established explanation for why deep learning works.

---

## 11. Source coding theorem (Shannon)

The minimum average bits per symbol needed to losslessly encode samples from $p$ is $H(p)$. **You cannot compress below entropy.**

Practical relevance for ML:

- Cross-entropy $H(p, q)$ is the **average code length** if you use a code optimal for $q$ to encode samples from $p$. Always $\geq H(p)$.
- Minimizing cross-entropy = building a near-optimal compressor for the data.
- LLMs are essentially lossy compressors of their training data. Better LM → better compression.

A very recent line of research (Deletang et al., "Language Modeling is Compression") makes this explicit: SOTA LLMs can compress text *better* than gzip.

> **Saying it out loud.** Shannon's source coding theorem says you cannot compress below the entropy, full stop. That's what makes entropy a physical limit rather than a definition — it's the true size of the information. The connection to machine learning is direct and worth making explicit: cross-entropy is the average code length you'd get using your model's beliefs to compress the real data, so training a model to minimize cross-entropy is literally training a compressor. That's the sense in which a language model is a compression algorithm, and it's not a metaphor — recent work shows large models compress text better than gzip does. If you want a one-liner: better prediction and better compression are the same thing.

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

> **Saying it out loud.** These gotchas mostly test one thing: do you know KL isn't a distance. It's asymmetric and it fails the triangle inequality, so it's a divergence, and swapping the arguments changes the answer *and* the behavior — forward is mean-seeking, reverse is mode-seeking. Two other quick ones. KL can be infinite, which happens whenever the data puts mass somewhere your model assigned exactly zero probability — that's the infinitely-surprised case, and it's why smoothing exists. And cross-entropy versus KL: they differ by the entropy of the data, which is constant during training, so minimizing one minimizes the other — but they are not the same quantity, and the difference matters the moment your data distribution changes.

---

## 13. The 10 most-asked information theory interview questions

1. **Define entropy.** $H(p) = -\sum p \log p$. Average surprise / coding length.

   > **Saying it out loud.** Entropy is average surprise, or equivalently the average number of yes-or-no questions you'd need to nail down an outcome. A coin that always lands heads has zero entropy — no questions needed. A fair coin has one bit. Uniform over sixteen options has four bits. The two anchors to state are that entropy is maximized by the uniform distribution and zero for a certainty, and the reason it's exactly the log-based formula rather than any other spread measure is Shannon's theorem: it's the hard floor on how far you can compress.

2. **Define cross-entropy.** $H(p, q) = -\sum p \log q$. Coding length using $q$-optimal code on samples from $p$.

   > **Saying it out loud.** Cross-entropy is what it costs to encode data from p using a code you designed for q. If your beliefs are right, it equals the entropy and you can't do better. If your beliefs are wrong, you pay more, and how much more is exactly the KL divergence. That's the framing to lead with, because it immediately explains why it's the standard loss: you're measuring how badly your model's beliefs describe reality, in bits.

3. **Cross-entropy = entropy + KL.** $H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q)$. Why minimizing cross-entropy = minimizing KL.

   > **Saying it out loud.** Cross-entropy decomposes into the data's own entropy plus the KL divergence from data to model. The first term depends only on the data, so during training it's a constant you can't touch. That means minimizing cross-entropy over your parameters is exactly minimizing KL — same argmin, different offset. That's the one-line justification for why the standard classification loss is the principled choice and not a convenience. It also tells you your loss has a floor: you can never get below the data's own entropy, which is the irreducible noise.

4. **Define KL divergence.** $\mathrm{KL}(p \,\|\, q) = \sum p \log(p/q)$. Non-negative, asymmetric, not a metric.

   > **Saying it out loud.** KL divergence is the expected number of extra bits you pay for using q when reality is p. It's non-negative, which follows from Jensen's inequality, and it's zero only when the two distributions are identical. The thing to say before they ask: it is not a distance. It's asymmetric and it violates the triangle inequality, so calling it a distance will get you corrected. It can also be infinite, whenever p puts mass somewhere q assigns zero.

5. **Forward vs reverse KL.** Forward: mean-seeking. Reverse: mode-seeking.

   > **Saying it out loud.** Forward KL — data first, model second — punishes the model for assigning near-zero probability where the data actually appears. So the model spreads out to cover everything, and when it's too simple to fit a multimodal target it parks itself in the middle, which is why MLE-trained models produce bland average outputs. Reverse KL punishes the model for putting mass where the data has none, so it collapses onto one mode and ignores the rest — confident and narrow. Maximum likelihood is forward; variational inference and KL-regularized RL are typically reverse. The one-sentence version: forward covers, reverse concentrates.

6. **Mutual information.** $I(X; Y) = H(X) + H(Y) - H(X, Y)$. Statistical dependence.

   > **Saying it out loud.** Mutual information is how much learning Y reduces your uncertainty about X. You can write it three ways — entropy of X minus entropy of X given Y, or the sum of the marginal entropies minus the joint, or as the KL divergence between the joint and the product of marginals. That last one is the most illuminating, because it says mutual information is literally measuring how far the two variables are from independent. It's symmetric, it's zero exactly when they're independent, and unlike correlation it detects non-linear dependence.

7. **Why is MLE = cross-entropy?** Cross-entropy is the negative log-likelihood under categorical; MLE is $\arg\max \log P(\text{data} \mid \theta) = \arg\min H(p_{\text{data}}, p_\theta)$.

   > **Saying it out loud.** They're the same thing written in different notation. Maximum likelihood maximizes the log-probability the model assigns to the observed data. Cross-entropy is the negative expected log-probability under the empirical data distribution. Flip the sign, divide by the number of samples, and you have the identical objective — so minimizing cross-entropy is maximizing likelihood. The reason this pairing produces different losses for different tasks is the assumed output distribution: categorical gives you cross-entropy, Gaussian gives you mean squared error, Poisson gives you Poisson loss. Same principle, different likelihood.

8. **Perplexity?** $\exp(\text{cross-entropy})$. Tokenizer-dependent.

   > **Saying it out loud.** Perplexity is the exponential of cross-entropy, and it reads as "the effective number of choices the model is deciding among per token." A perplexity of twenty means the model is about as uncertain as if it were picking uniformly from twenty words. The ceiling is the vocabulary size, the floor is the true entropy of the text. And the caveat that scores points: perplexity is tokenizer-dependent, so a model with a bigger vocabulary can post a different number on the same text without being any better. Comparisons across tokenizers are meaningless.

9. **KL in RLHF?** Penalty $\beta \cdot \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ prevents policy from drifting from reference.

   > **Saying it out loud.** RLHF maximizes reward minus beta times the KL divergence between the current policy and the frozen reference model. The KL term is an anchor: without it, the policy discovers that the reward model has blind spots and rushes into them, producing text that scores fantastically and reads like nonsense — that's reward hacking. Beta sets how far you'll let it roam; too small and you get degenerate outputs, too large and the model barely changes from the reference. Watching the KL number during training is the standard diagnostic, and a sudden spike means you're about to hack the reward.

10. **What's the source coding theorem?** Average code length $\geq$ entropy. Cross-entropy is the loss because it's compressibility under the model.

   > **Saying it out loud.** Shannon's source coding theorem says the average number of bits per symbol in any lossless code is at least the entropy of the source, and you can get arbitrarily close to it. So entropy isn't a convention, it's a physical limit on compression. The ML connection is that cross-entropy is exactly the code length you'd achieve using your model's distribution, so training a model to minimize cross-entropy is training a compressor. That's why the claim "language modeling is compression" is literal rather than poetic — large models genuinely compress text better than gzip.


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
