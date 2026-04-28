# Regularization: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Regularization is where most candidates can recite the names but not the geometry, the Bayesian interpretation, or the failure modes. Interviewers probe: "Why is L1 sparse and L2 not?", "What's dropout actually doing at inference?", "Why does data augmentation work?". Strong answers require understanding what regularization *means* — constraining the hypothesis class — not just listing techniques.

---

## 1. The unifying principle

**Regularization = imposing prior structure on the model to reduce variance at the cost of some bias.**

Every regularization technique is one of:

1. **Penalty on parameters** — L2, L1, weight decay.
2. **Penalty on the function** — spectral norm, sharpness penalty.
3. **Stochastic perturbation** — dropout, data augmentation, noise injection.
4. **Implicit constraint via training** — early stopping, SGD noise, learning-rate schedules.
5. **Architectural constraint** — convolutional weight sharing, attention sparsity.

The deep idea: with infinite data you don't need regularization; you fit any function and the data tells you which is right. With finite data, multiple functions fit equally well; regularization picks the "simpler" one. Bayesian rephrasing: regularization = prior; data + prior = posterior.

---

## 2. The bias-variance decomposition

For squared error on a fresh test point $x$:

$$
\mathbb{E}\!\left[(y - \hat f(x))^2\right] = \text{noise}^2 + \text{bias}(\hat f)^2 + \text{var}(\hat f)
$$

**Bias:** systematic error of the average prediction.
**Variance:** how much $\hat f$ changes across training sets.
**Noise:** irreducible.

Regularization **trades variance for bias**. A simpler model has lower variance (small dataset → similar function each time) but higher bias (can't fit complex truth). The regularized optimum is the sweet spot.

This is the central interview talking point. Strong candidates frame every regularization technique as "increases bias slightly, reduces variance substantially."

---

## 3. L2 (Ridge) regularization

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \|w\|^2
$$

### Geometry

L2's level sets are **circles** (in 2D) or hyperspheres. The penalized minimum is where a contour line of the unregularized loss is tangent to a circle of constant $\|w\|^2$. The result is **proportional shrinkage**: every coefficient pulled toward zero by a fraction.

### Bayesian interpretation

L2 regularization corresponds to a **Gaussian prior** $w \sim \mathcal{N}(0, \sigma^2 I)$ and finding the **MAP** (maximum a posteriori) estimate:

$$
\arg\max_w P(w \mid \text{data}) \propto \arg\max_w P(\text{data} \mid w) \cdot P(w)
$$

Taking negative log: $-\log P(\text{data} \mid w) + \frac{1}{2\sigma^2} \|w\|^2$. The first term is the data loss; the second is L2 with $\lambda = 1/\sigma^2$.

### Effect on optimization

For linear regression, L2 has a closed-form solution $(X^\top X + \lambda I)^{-1} X^\top y$. The $+\lambda I$ prevents $X^\top X$ from being singular (which it would be if features are collinear). L2 is one of the standard fixes for multicollinearity.

### Effect on conditioning

Adding $\lambda I$ to the Hessian shifts every eigenvalue up by $\lambda$. Improves conditioning, makes optimization easier, reduces variance.

### When to use

Almost always. L2 is the default. If you're not sure what regularization to use, use L2 with $\lambda \in [10^{-4}, 10^{-1}]$.

---

## 4. L1 (Lasso) regularization

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \|w\|_1 = \mathcal{L}_{\text{data}} + \lambda \sum_i |w_i|
$$

### Geometry

L1's level sets are **diamonds** (in 2D) or cross-polytopes — corners on the axes. The penalized minimum often lands at a corner, which means one or more coordinates are exactly zero.

### Why this gives sparsity

At a corner of the diamond, the gradient of the data loss has to balance against a non-smooth penalty. The subgradient of $|w|$ is $\operatorname{sign}(w)$ for $w \neq 0$ and $[-1, 1]$ at $w = 0$. So if $|\partial \mathcal{L}_{\text{data}} / \partial w_j| < \lambda$, the optimum is $w_j = 0$ exactly (the gradient can't push past the penalty).

### Bayesian interpretation

L1 corresponds to a **Laplace (double-exponential) prior**: $P(w_j) \propto \exp(-|w_j| / b)$. Compared to Gaussian, Laplace has a **sharper peak at 0** (and heavier tails far from 0), encoding "most weights should be exactly 0, but a few may be large."

### When to use

- Feature selection: when you suspect most features are irrelevant.
- Interpretability: sparse models are easier to explain.
- High-dimensional, low-sample: $d \gg N$.

### Failure modes

- Among correlated features, L1 picks one arbitrarily — unstable across data subsets.
- Convex but not strictly convex along the zero-axes; multiple optima possible.
- Optimization is non-trivial (not differentiable at 0); standard tools: subgradient, proximal gradient (ISTA/FISTA), coordinate descent.

---

## 5. Elastic Net

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \alpha \lambda \|w\|_1 + (1 - \alpha) \cdot \frac{\lambda}{2} \|w\|^2
$$

Combines L1 and L2. The $\alpha \in [0, 1]$ controls the mix.

### When to use

- Correlated features: L1 alone arbitrarily picks one of a correlated pair; elastic net groups them.
- Default-ish for high-dim regression in practice (`glmnet` is the standard tool).

### Bayesian interpretation

Mixture of Laplace and Gaussian priors. Heavier tails near 0 (sparsity) plus smooth quadratic shrinkage (stability).

---

## 6. Dropout

The most famous deep learning regularizer. Hinton et al. 2014.

### Mechanism (training)

For each forward pass, randomly zero out each activation with probability $p$ (typically 0.1–0.5). Scale remaining activations by $1/(1-p)$ so the expected value is unchanged.

```python
mask = torch.bernoulli(torch.full_like(h, 1 - p))
h_drop = h * mask / (1 - p)         # scale to keep expected value
```

Equivalently, with $m_i \sim \operatorname{Bernoulli}(1-p)$ and $\tilde h_i = h_i \cdot m_i / (1 - p)$:

$$
\mathbb{E}[\tilde h_i] = h_i \cdot \frac{(1-p)}{1-p} = h_i
$$

### Mechanism (inference)

**No dropout.** All activations active. The $1/(1-p)$ scaling during training ensures activation magnitudes match.

### Why it works (multiple stories)

**Story 1: ensemble of subnetworks.** Each forward pass uses a random subnetwork; over time, the model averages exponentially many subnetworks. Standard inference (no dropout, weight scaling) approximates the *geometric mean* of these subnetworks deterministically. **Monte Carlo dropout** — sampling masks at inference and averaging predictions — is a different procedure that approximates Bayesian model averaging (Gal & Ghahramani 2016).

**Story 2: prevents co-adaptation.** Neurons can't rely on specific other neurons to be present, so each must be useful in many contexts. Forces redundant representations.

**Story 3: noise injection.** Adds multiplicative Bernoulli noise to activations, which acts like data augmentation in feature space.

### Common choices

- $p = 0.1$–$0.3$ for hidden layers in moderate-size networks.
- $p = 0.5$ was original choice; rarely used today.
- Transformers usually use small dropout (0.0–0.1) in pre-training, sometimes more in fine-tuning.
- Modern LLMs at scale often **don't use dropout at all** because data is plentiful and dropout slows learning.

### Variants

- **DropPath / Stochastic Depth:** drop entire residual blocks (probabilistically). Used in some vision transformers.
- **DropConnect:** drop weights instead of activations.
- **Variational dropout:** principled Bayesian view; same mask within a sequence (for RNNs).
- **Spatial dropout:** drop entire feature maps in CNNs.

### Failure modes

- Reduces effective model capacity → may hurt if you're already underfitting.
- Inference scaling error: forgetting $1/(1-p)$ at training time silently breaks inference.
- Inconsistent train/eval mode: forgetting `model.eval()` keeps dropout active at inference.

---

## 7. Early stopping

### Mechanism

Monitor validation loss during training. Stop when it stops improving (or starts increasing). Restore the weights from the best validation checkpoint.

### Why it regularizes

Training loss continues decreasing while validation loss starts increasing — overfitting. Stopping at the validation minimum prevents the model from fitting training noise. **Equivalent in spirit** to having a smaller effective model capacity.

### Connection to L2

For squared loss with gradient flow, early stopping at time $t$ is approximately equivalent to L2 with $\lambda \propto 1/t$. Stopping early = strong L2; running long = weak L2. **Friedman's "early stopping is L2 in disguise"** is one of the most beautiful results in this area.

### Practical issues

- Need a reliable validation set (no leakage).
- Patience: how many epochs without improvement before stopping?
- Restoration: keep the best-validation weights, not just the last.

---

## 8. Data augmentation

### Why it regularizes

Augmentation increases the effective training set size by transforming inputs in ways that preserve the label. The model sees more variation, generalizes better. Equivalent to enforcing **invariance**: the function should give the same answer regardless of the augmentation applied.

### Common augmentations

- **Image:** rotations, crops, flips, color jitter, MixUp, CutMix.
- **Text:** synonym replacement, back-translation, dropout-style masking.
- **Audio:** time stretching, pitch shifting, noise addition.

### MixUp (Zhang et al. 2018)

$$
x_{\text{mix}} = \alpha \cdot x_1 + (1 - \alpha) \cdot x_2
$$

$$
y_{\text{mix}} = \alpha \cdot y_1 + (1 - \alpha) \cdot y_2
$$

Linearly interpolate two examples and their labels. Encourages the model to behave linearly between training examples — strong implicit regularization, often improves calibration.

### CutMix

Splice a rectangular region from one image into another; mix labels by area fraction. Locally cleaner than MixUp; often beats MixUp on classification.

### Why does augmentation help even on near-IID data?

Because the model can memorize specific training points but not their continuum of perturbations. Forces learning the underlying invariances rather than the specific examples.

---

## 9. Label smoothing

$$
y_{\text{smooth}} = (1 - \varepsilon) \cdot y_{\text{one-hot}} + \frac{\varepsilon}{K}
$$

Replace hard one-hot label $y = [0, 0, 1, 0, 0]$ with soft $y_{\text{smooth}} = [\varepsilon/K, \varepsilon/K, 1 - \varepsilon(K-1)/K, \varepsilon/K, \varepsilon/K]$. Typically $\varepsilon = 0.1$.

### Why it helps

- Prevents the model from learning to push logits to $\pm \infty$ for confident predictions.
- Improves calibration (the model knows it's not 100% sure even on training data).
- Acts as a soft regularizer on the output distribution.

### Connection to regularization

Cross-entropy with one-hot labels can be made arbitrarily small only by pushing logits to extremes. Cross-entropy with smoothed labels has a non-zero floor — encourages bounded logits.

### Used in

Many vision and NLP recipes (label smoothing 0.1 is common in transformer pretraining). LLM pretraining often uses it (or doesn't, depending on the recipe).

---

## 10. Weight decay (revisited)

For SGD, weight decay = L2 regularization. For Adam, they differ; AdamW decouples them. See `02_gradient_descent/LEARNING_RATE_DEEP_DIVE.md` for the full story.

In LLM training: weight decay $0.1$ is typical. Decoupled (AdamW) so that high-gradient parameters are still penalized uniformly.

---

## 11. Sharpness-Aware Minimization (SAM)

Foret et al. 2020. Recent regularizer based on the loss landscape.

$$
\min_w \max_{\|\varepsilon\| < \rho} L(w + \varepsilon)
$$

Find weights where the **maximum nearby loss** is small — i.e., flat regions of the loss landscape. Practical implementation:

1. Compute gradient of $L$ at $w$.
2. Take a step in that direction to find $w + \varepsilon$ (the "sharpest direction").
3. Compute gradient of $L$ at $w + \varepsilon$.
4. Apply that gradient as the actual update at $w$.

Empirically improves generalization on vision tasks. **Doubles training time** (two forward-backward passes per step). For LLMs, mostly research-stage; production runs use cheaper alternatives.

---

## 12. Implicit regularization

Modern deep learning's biggest insight: **the optimizer itself regularizes**, even without an explicit penalty.

### SGD's implicit regularization

SGD with mini-batch noise has stochastic updates with magnitude $\eta \cdot g + \eta \cdot \xi$, where $\xi$ is the per-batch noise. The noise scale $\eta/B$ biases SGD toward **flat minima** — regions where small perturbations don't increase loss much. Flat minima generalize better than sharp ones.

### Implicit bias of overparameterized networks

With more parameters than data points, there are infinitely many functions that perfectly fit. SGD does not pick a random one — it picks the one closest to the initialization in some sense. This is why double descent happens: as model size grows past interpolation, generalization eventually improves because the optimizer's bias selects "nicer" solutions.

### Why this matters

The whole "deep learning works" story is implicit-regularization-driven. Explicit regularization (weight decay, dropout) helps but isn't the main thing. Frontier-lab interviews often probe this: do you understand that SGD is implicitly biasing toward generalizable solutions?

---

## 13. Inductive bias as regularization

Architecture choices regularize by restricting the function class:

- **Convolutions** enforce translation equivariance and locality.
- **Attention** enforces token-wise mixing without inherent ordering.
- **Recurrence** enforces sequential processing.
- **Pooling** enforces invariance to small spatial perturbations.

These are stronger than explicit L2 penalties for many tasks because they encode domain knowledge directly into the model structure.

---

## 14. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "L1 is sparse, L2 is not — why?" | Geometry. L1 diamonds have corners on axes; the optimum often lands at a corner = sparse weights. L2 spheres have no corners. |
| "Are L2 and weight decay the same?" | For SGD yes; for Adam no (the preconditioning weakens L2). AdamW decouples. |
| "What does dropout do at inference?" | Nothing. Just disable it. The $1/(1-p)$ scaling during training ensures activations match. |
| "Why does early stopping help?" | Equivalent to L2 with $\lambda \propto 1/t$. Stops fitting noise once validation loss stops improving. |
| "Why does data augmentation work?" | Increases effective dataset size; enforces invariance to the augmentations. |
| "What's MixUp?" | Linear interpolation of $(x, y)$ pairs. Encourages linear behavior between training examples. |
| "L2 corresponds to what prior?" | Gaussian prior $\mathcal{N}(0, 1/\lambda)$. MAP = L2-regularized MLE. |
| "L1 corresponds to what prior?" | Laplace (double-exponential) prior. Heavier tails near zero → sparsity. |
| "Why does SGD generalize better than full-batch GD?" | Implicit regularization toward flat minima. The noise scale $\eta/B$ matters. |
| "When does dropout hurt?" | When the model is underfitting (reduces effective capacity). When data is plentiful (LLMs at scale). |

---

## 15. The 10 most-asked regularization interview questions

1. **What's the bias-variance trade-off?** Squared error decomposes as $\text{bias}^2 + \text{variance} + \text{noise}$. Regularization trades variance for bias.
2. **L1 vs L2 difference?** L1 sparsity (diamond corners), L2 shrinkage (spherical). L1 = Laplace prior, L2 = Gaussian.
3. **What does dropout do?** Randomly zero activations during training; scale to compensate. At inference: no dropout.
4. **Why does early stopping work?** Stops fitting noise; equivalent to L2 with $\lambda \propto 1/t$.
5. **What's data augmentation doing mathematically?** Enforcing invariance under the augmentation; expanding effective dataset.
6. **What's label smoothing?** Soft labels prevent overconfident extreme logits; improves calibration.
7. **What's MixUp?** Linear interpolation of $(x, y)$ pairs. Strong augmentation, often improves calibration.
8. **What's the implicit regularization of SGD?** Biases toward flat minima via noise scale $\eta/B$.
9. **AdamW vs Adam+L2?** AdamW decouples weight decay from preconditioning, recovering uniform regularization.
10. **What's SAM?** Sharpness-Aware Minimization — minimize the max-loss in a small neighborhood. Targets flat minima.

---

## 16. Drill plan

1. Master L1 vs L2 geometry and the Bayesian priors.
2. Walk through dropout's training and inference with the $1/(1-p)$ scaling.
3. Explain early stopping ≈ L2 connection.
4. Know the implicit regularization of SGD ($\eta/B$ story).
5. Drill `INTERVIEW_GRILL.md`.
