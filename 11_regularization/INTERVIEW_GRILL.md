# Regularization — Interview Grill

> 45 questions on regularization. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. What's the bias-variance trade-off?**
For squared error: $\mathbb{E}[(y - \hat f)^2] = \text{noise}^2 + \text{bias}(\hat f)^2 + \text{var}(\hat f)$. Bias = systematic error of the average prediction; variance = how much the prediction changes with training data. Regularization trades variance for bias — increases bias slightly to substantially reduce variance.

**2. Why do we need regularization?**
With finite data, multiple functions fit equally well; we want the "simplest" / lowest-variance one. With infinite data, regularization is unnecessary (data tells you which fit is right). Regularization encodes prior structure on the model.

**3. The five categories of regularization?**
Penalty on parameters (L1, L2). Penalty on the function (spectral norm, sharpness). Stochastic perturbation (dropout, augmentation). Implicit constraint (early stopping, SGD noise). Architectural constraint (convolutions, attention).

**4. Bayesian framing of regularization?**
Regularization = prior on parameters. MAP estimation = $\arg\max P(w \mid \text{data}) = \arg\max P(\text{data} \mid w) \cdot P(w)$. Negative log gives $\text{loss} + \text{regularizer}$, where the regularizer is $-\log P(w)$. L2 = Gaussian prior; L1 = Laplace prior.

---

## B. L1, L2, Elastic Net

**5. State L2 regularization.**
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + (\lambda/2) \|w\|^2$. Quadratic penalty on weights. Equivalent to Gaussian prior $\mathcal{N}(0, 1/\lambda)$.

**6. State L1 regularization.**
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \|w\|_1 = \mathcal{L}_{\text{data}} + \lambda \sum_i |w_i|$. Sum of absolute values. Equivalent to Laplace prior.

**7. Why does L1 produce sparse weights but L2 doesn't?**
Geometry. L1's level sets are diamonds with corners on the axes. The penalized minimum often lands at a corner where one or more coordinates are exactly 0. L2's level sets are circles with no corners; minima don't naturally land on the axes.

**8. Mathematically: when does L1 produce zero weights?**
The subgradient of $\lambda |w_j|$ at 0 is $[-\lambda, \lambda]$. If $|\partial \mathcal{L}_{\text{data}} / \partial w_j| < \lambda$ at $w_j = 0$, the data gradient can't push past the penalty's subgradient and the optimum stays at exactly 0.

**9. Bayesian interpretation of L2?**
Gaussian prior $w \sim \mathcal{N}(0, \sigma^2 I)$ with $\lambda = 1/\sigma^2$. MAP estimate equals MLE on $-\log P(\text{data} \mid w) - \log P(w) = \text{loss} + (1/2\sigma^2) \|w\|^2$.

**10. Bayesian interpretation of L1?**
Laplace (double-exponential) prior $P(w_j) \propto \exp(-|w_j|/b)$. Heavier tails near zero than Gaussian → encodes "most weights should be zero."

**11. Effect of L2 on optimization?**
Adds $\lambda I$ to the Hessian, shifting eigenvalues up. Improves conditioning, reduces variance. For linear regression, $(X^\top X + \lambda I)^{-1} X^\top y$ is the closed-form solution.

**12. Why is L2 the default if L1 is more interpretable?**
L2 is differentiable (easy optimization), gives smooth solutions, handles correlated features well. L1's sparsity comes at the cost of optimization complexity (subgradient methods) and arbitrary choice among correlated features.

**13. What's Elastic Net?**
$\mathcal{L}_{\text{data}} + \alpha \lambda \|w\|_1 + (1 - \alpha) (\lambda/2) \|w\|^2$. Combines L1 and L2. $\alpha \in [0, 1]$ controls mix. Useful when features are correlated (L1 alone arbitrarily picks one of a correlated pair; elastic net groups them).

**14. When to use L1, L2, Elastic Net?**
L2: default; smooth solutions; multicollinearity-robust. L1: feature selection; sparse models needed. Elastic Net: high-dim, correlated features.

---

## C. Weight decay (vs L2)

**15. For SGD, are L2 and weight decay the same?**
Yes. Gradient of $(\lambda/2) \|w\|^2$ is $\lambda w$, so SGD with explicit decay $w \leftarrow w - \eta g - \eta \lambda w$ is identical to SGD with L2 added to the loss.

**16. For Adam, are L2 and weight decay the same?**
No. Adam's $1/\sqrt{\hat v}$ preconditioning divides L2's $\lambda w$ term, weakening regularization where gradient variance is high. Decay strength becomes non-uniform across parameters.

**17. What does AdamW do?**
Decouples weight decay: $\theta \leftarrow \theta - \eta \cdot \hat m / (\sqrt{\hat v} + \varepsilon) - \eta \cdot \lambda \cdot \theta$. Decay applied directly to parameters, not added to gradient. Recovers uniform regularization.

**18. Typical weight decay for LLM pretraining?**
$\lambda = 0.1$ is the modern default. Earlier and smaller models used $0.01$–$0.05$. SFT and DPO usually use $0.0$ or near-zero.

---

## D. Dropout

**19. What is dropout?**
For each forward pass, randomly zero each activation with probability $p$; scale remaining by $1/(1-p)$ to preserve expectations. Applied during training only; inference uses all activations.

**20. Why scale by $1/(1-p)$?**
Expected activation magnitude during training $= (1-p) \cdot h + p \cdot 0 = (1-p) \cdot h$. Multiplying by $1/(1-p)$ recovers $h$, so train and inference activations have matching scales without changing inference code.

**21. Why does dropout work? (multiple stories)**
(a) Ensemble: each forward pass is a random subnetwork; training averages over subnetworks. (b) Prevents co-adaptation: neurons can't rely on specific others to be present. (c) Noise injection in feature space, like data augmentation.

**22. Typical dropout rates?**
$0.1$–$0.3$ for hidden layers in moderate-size networks. Original paper used $0.5$ (rare today). Transformers in pretraining: $0.0$–$0.1$. Modern LLMs at scale often use **no dropout** because data is plentiful.

**23. When does dropout hurt?**
Underfitting models (reduces effective capacity). LLMs with abundant data (slows learning). Tasks requiring all features (rare).

**24. What's DropPath / Stochastic Depth?**
Randomly drop entire residual blocks during training. Used in some vision transformers (ConvNeXt, DeiT). Like dropout but applied at the block level rather than activation.

**25. Train vs eval mode in PyTorch?**
`model.train()` enables dropout (and BN running stats). `model.eval()` disables dropout (uses BN running averages). Forgetting to switch modes is a classic bug source — leaves dropout active at inference, results are inconsistent.

---

## E. Early stopping, label smoothing, augmentation

**26. What is early stopping?**
Train while monitoring validation loss; stop when validation loss stops improving (or starts increasing); restore weights from best validation checkpoint. Prevents fitting noise.

**27. Connection between early stopping and L2?**
For squared loss with gradient flow: early stopping at time $t$ is approximately equivalent to L2 with $\lambda \propto 1/t$. Stopping early = strong L2; running long = weak L2. Friedman's classic result.

**28. What's label smoothing?**
Replace one-hot labels with $y_{\text{smooth}} = (1 - \varepsilon) \cdot y_{\text{one-hot}} + \varepsilon/K$. Typical $\varepsilon = 0.1$. Prevents the model from learning to push logits to $\pm \infty$.

**29. Why does label smoothing help?**
Pushes the model toward calibrated confidence. With one-hot labels, cross-entropy can only be 0 by pushing logits to extremes (overconfident). With smoothed labels, there's a non-zero floor — bounded logits are optimal.

**30. What's data augmentation doing mathematically?**
Increases effective training set size by transforming inputs in ways that preserve the label. Equivalent to enforcing invariance under those transforms. Reduces overfitting by exposing the model to more "variations" of the underlying concepts.

**31. What's MixUp?**
$x_{\text{mix}} = \alpha \cdot x_1 + (1 - \alpha) \cdot x_2$, $y_{\text{mix}} = \alpha \cdot y_1 + (1 - \alpha) \cdot y_2$. Linearly interpolate two examples and labels. Encourages the model to behave linearly between training examples. Often improves calibration.

**32. What's CutMix?**
Splice a rectangular region from image 1 into image 2; mix labels by area fraction. Locally cleaner than MixUp; often beats it on image classification.

---

## F. Implicit regularization

**33. What's the implicit regularization of SGD?**
Mini-batch noise biases SGD toward flat minima — regions where small perturbations don't increase loss much. Flat minima generalize better. The noise scale $\eta/B$ controls this implicit regularization strength.

**34. Why does this matter?**
Modern deep learning's success is largely due to implicit regularization, not explicit penalties. Overparameterized networks have many functions that perfectly fit training data; SGD picks "nice" ones (close to init, flat in loss landscape). Without this, deep learning wouldn't work as well as it does.

**35. What's the connection between batch size, learning rate, and generalization?**
Implicit regularization scale $\propto \eta/B$. Larger batches = less noise = less implicit regularization. To preserve generalization when scaling up batch size, scale up $\eta$ proportionally (linear scaling rule for SGD; sqrt for Adam).

**36. What's double descent?**
Test loss decreases as model size grows past the interpolation threshold (where training loss = 0). Goes against classical bias-variance: more capacity can be better, not worse. Modern deep learning lives in this overparameterized regime.

---

## G. Modern and frontier topics

**37. What's Sharpness-Aware Minimization (SAM)?**
$\min_w \max_{\|\varepsilon\| < \rho} L(w + \varepsilon)$. Find weights where the maximum nearby loss is small (flat regions). Practical: 2 forward-backward passes per step (find sharpest direction, then take step there). Costly but improves generalization.

**38. What's spectral normalization?**
Constrain each weight matrix's largest singular value to 1. Used in GANs (Spectral Norm GAN) for stability. Bounds the Lipschitz constant of the network.

**39. What's batch normalization's regularization effect?**
BN's batch statistics inject noise into activations (different batches give different stats). Acts as a mild stochastic regularizer. Some networks combine BN with explicit dropout; others find BN alone is enough.

**40. Why is dropout less common in modern LLMs?**
At scale, LLMs are not data-limited (or barely so). Implicit regularization from SGD/AdamW noise is enough. Dropout slows learning. Many modern recipes use 0 dropout.

---

## H. Misc and gotchas

**41. Can L2 alone cause underfitting?**
Yes if $\lambda$ is too high. Strong L2 forces small weights; the model can't fit the data. Should pick $\lambda$ by validation, not by intuition.

**42. What's wrong with $\lambda = 0$?**
No regularization. With overparameterized models on finite data, can lead to overfitting and weight blowup (especially on separable data — see logistic regression). Practical: use small but non-zero $\lambda$.

**43. How do you tune $\lambda$?**
Validation. Train with multiple $\lambda$ values; pick the one with best validation metric. Log-scale grid ($10^{-5}$ to $10^{-1}$) is standard.

**44. Why does data augmentation generalize even on near-IID data?**
The model can memorize specific points but not their continuum of transformations. Augmentation forces learning the underlying invariances (rotation, color, etc.) rather than the specific examples.

**45. Why doesn't regularization replace good data?**
Regularization picks the "simplest" function consistent with the data. If the data is misleading, regularization doesn't fix it. More and better data dominates regularization in most regimes.

---

## I. Quick fire

**46.** *L2 is equivalent to which prior?* Gaussian.
**47.** *L1 is equivalent to which prior?* Laplace.
**48.** *Default dropout rate for original paper?* $0.5$.
**49.** *Default label smoothing $\varepsilon$?* $0.1$.
**50.** *Default LLM weight decay?* $0.1$.

---

## Self-grading

If you can't answer 1-15, you don't know basic regularization. If you can't answer 16-35, you'll struggle with serious ML interviews. If you can't answer 36-50, frontier-lab interviews will go past you.

Aim for 30+/45 cold.
