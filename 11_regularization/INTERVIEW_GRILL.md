# Regularization — Interview Grill

> 45 questions on regularization. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. What's the bias-variance trade-off?**
For squared error: $\mathbb{E}[(y - \hat f)^2] = \text{noise}^2 + \text{bias}(\hat f)^2 + \text{var}(\hat f)$. Bias = systematic error of the average prediction; variance = how much the prediction changes with training data. Regularization trades variance for bias — increases bias slightly to substantially reduce variance.

> **Saying it out loud.** Your error breaks into three pieces: bias, which is how wrong you'd be on average across all possible training sets; variance, which is how much your model jumps around when the training set changes; and noise you can never remove. A tiny model is consistently wrong the same way — high bias, low variance. A huge unconstrained model fits each dataset perfectly and differently — low bias, high variance. Regularization buys a small amount of bias to sell a large amount of variance, and being able to say that sentence about any specific technique is what scores.

**2. Why do we need regularization?**
With finite data, multiple functions fit equally well; we want the "simplest" / lowest-variance one. With infinite data, regularization is unnecessary (data tells you which fit is right). Regularization encodes prior structure on the model.

> **Saying it out loud.** Because with finite data there are many different functions that fit your training set perfectly and disagree everywhere else, and nothing in the data alone tells you which one to pick. Regularization is how you say what you'd prefer when the data is silent — smaller weights, fewer active features, less dependence on any one neuron. The Bayesian phrasing is the crispest one: it's a prior. And the corollary worth stating is that with infinite data you'd need none of it, which is exactly why frontier LLMs use so little explicit regularization.

**3. The five categories of regularization?**
Penalty on parameters (L1, L2). Penalty on the function (spectral norm, sharpness). Stochastic perturbation (dropout, augmentation). Implicit constraint (early stopping, SGD noise). Architectural constraint (convolutions, attention).

> **Saying it out loud.** I'd group them by what they constrain. Penalties on the parameters — L1 and L2. Penalties on the function itself — spectral norm, sharpness. Injected randomness — dropout, augmentation, noise. Constraints from how you train — early stopping, SGD noise, the learning-rate schedule. And constraints baked into the architecture — convolutions, attention, pooling. The last one is usually the strongest and the one people forget to mention, because a convolution restricts the hypothesis class far more than any weight-decay value you could pick.

**4. Bayesian framing of regularization?**
Regularization = prior on parameters. MAP estimation = $\arg\max P(w \mid \text{data}) = \arg\max P(\text{data} \mid w) \cdot P(w)$. Negative log gives $\text{loss} + \text{regularizer}$, where the regularizer is $-\log P(w)$. L2 = Gaussian prior; L1 = Laplace prior.

> **Saying it out loud.** In the Bayesian view, the regularizer is just the negative log of your prior over weights. You start from MAP estimation — maximise the likelihood times the prior — take the negative log, and the product turns into a sum: data loss plus penalty. A Gaussian prior gives you a squared term, which is L2; a Laplace prior gives you an absolute-value term, which is L1. That's the elegant answer, and it explains why L1 is sparse in one line: the Laplace density has a spike at zero, so zero is genuinely the most probable value a priori.

---

## B. L1, L2, Elastic Net

**5. State L2 regularization.**
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + (\lambda/2) \|w\|^2$. Quadratic penalty on weights. Equivalent to Gaussian prior $\mathcal{N}(0, 1/\lambda)$.

> **Saying it out loud.** L2 adds half lambda times the sum of squared weights to your loss. The gradient of that is just lambda times the weight, so every step pulls each weight proportionally toward zero — shrinkage, not selection. It corresponds to a Gaussian prior with variance one over lambda, so a bigger lambda is a tighter prior. And it's the default for a reason: it's smooth, it's differentiable everywhere, and it improves the conditioning of the problem as a side effect.

**6. State L1 regularization.**
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \|w\|_1 = \mathcal{L}_{\text{data}} + \lambda \sum_i |w_i|$. Sum of absolute values. Equivalent to Laplace prior.

> **Saying it out loud.** L1 adds lambda times the sum of absolute values. The key difference from L2 is that the gradient magnitude is constant — it's lambda no matter how small the weight is — whereas L2's pull shrinks as the weight shrinks. That constant pull is what actually drives weights to exactly zero rather than asymptotically toward it. It corresponds to a Laplace prior, which has a spike at zero, so it's the formal way of saying "I believe most of these should be exactly nothing.\"

**7. Why does L1 produce sparse weights but L2 doesn't?**
Geometry. L1's level sets are diamonds with corners on the axes. The penalized minimum often lands at a corner where one or more coordinates are exactly 0. L2's level sets are circles with no corners; minima don't naturally land on the axes.

> **Saying it out loud.** Two answers, and I'd give both. Geometrically, the L1 constraint region is a diamond with corners sitting on the axes, and a loss contour expanding outward is most likely to first touch a corner — a corner means a coordinate is exactly zero. L2's region is a sphere, which has no corners, so the touch point almost surely has every coordinate non-zero. The algebraic answer is stronger: L1's pull toward zero has constant magnitude lambda, while L2's pull is proportional to the weight and so fades to nothing as the weight shrinks. Constant pull reaches zero; fading pull never does.

**8. Mathematically: when does L1 produce zero weights?**
The subgradient of $\lambda |w_j|$ at 0 is $[-\lambda, \lambda]$. If $|\partial \mathcal{L}_{\text{data}} / \partial w_j| < \lambda$ at $w_j = 0$, the data gradient can't push past the penalty's subgradient and the optimum stays at exactly 0.

> **Saying it out loud.** Exactly when the data gradient at zero is smaller in magnitude than lambda. The absolute value isn't differentiable at zero — its subgradient there is the whole interval from minus lambda to plus lambda — so if the data's push on that coordinate falls inside that interval, zero is a genuine optimum and nothing can dislodge it. That's the precise statement, and it's much better than hand-waving about diamond corners. It also tells you the tuning behaviour: raise lambda and more coordinates fall inside the interval, so sparsity increases smoothly.

**9. Bayesian interpretation of L2?**
Gaussian prior $w \sim \mathcal{N}(0, \sigma^2 I)$ with $\lambda = 1/\sigma^2$. MAP estimate equals MLE on $-\log P(\text{data} \mid w) - \log P(w) = \text{loss} + (1/2\sigma^2) \|w\|^2$.

> **Saying it out loud.** L2 is a Gaussian prior on the weights, centred at zero with variance one over lambda. You get there by writing MAP estimation, taking the negative log, and noticing that the log of a Gaussian density is a squared term. So a large lambda means a narrow prior — you strongly believe the weights are small — and lambda going to zero recovers plain maximum likelihood with a flat prior. The nice consequence to mention: this tells you lambda should scale with how much data you have, since a fixed prior gets outvoted by more likelihood terms.

**10. Bayesian interpretation of L1?**
Laplace (double-exponential) prior $P(w_j) \propto \exp(-|w_j|/b)$. Heavier tails near zero than Gaussian → encodes "most weights should be zero."

> **Saying it out loud.** L1 is a Laplace prior — the double-exponential, proportional to $e^{-|w|/b}$. Compared to a Gaussian it has a sharp spike right at zero and heavier tails further out, which encodes exactly the belief you want for sparsity: most coefficients are truly zero, but the few that aren't may be large. A Gaussian says the opposite — it says everything is smallish and nothing is exactly zero. That contrast is the cleanest one-line explanation of why one selects features and the other only shrinks them.

**11. Effect of L2 on optimization?**
Adds $\lambda I$ to the Hessian, shifting eigenvalues up. Improves conditioning, reduces variance. For linear regression, $(X^\top X + \lambda I)^{-1} X^\top y$ is the closed-form solution.

> **Saying it out loud.** L2 adds lambda to every eigenvalue of the Hessian, which is a big deal for conditioning. If two features are collinear, $X^\top X$ is singular and the unregularized solution doesn't exist — adding $\lambda I$ makes it invertible, which is where ridge regression's closed form comes from. In the deep learning case the same effect shows up as a better-behaved loss surface and less sensitivity to the learning rate. So L2 isn't purely a generalisation tool; it's also a numerical stability tool, which is worth saying.

**12. Why is L2 the default if L1 is more interpretable?**
L2 is differentiable (easy optimization), gives smooth solutions, handles correlated features well. L1's sparsity comes at the cost of optimization complexity (subgradient methods) and arbitrary choice among correlated features.

> **Saying it out loud.** Because L2 is easier in every practical way. It's differentiable everywhere, so plain gradient descent handles it and there's nothing special to implement. It handles correlated features gracefully, shrinking them together instead of picking one. And in deep learning, sparsity in the weights isn't actually useful — you don't get a speedup from scattered zeros without structured sparsity support, and nobody is interpreting individual weights of a transformer. So L1's benefit is mostly a classical-statistics benefit, and its costs — non-smooth optimisation and unstable selection among correlated features — are real everywhere.

**13. What's Elastic Net?**
$\mathcal{L}_{\text{data}} + \alpha \lambda \|w\|_1 + (1 - \alpha) (\lambda/2) \|w\|^2$. Combines L1 and L2. $\alpha \in [0, 1]$ controls mix. Useful when features are correlated (L1 alone arbitrarily picks one of a correlated pair; elastic net groups them).

> **Saying it out loud.** Elastic net is L1 plus L2 with a mixing weight. You reach for it when features are correlated, because L1 alone will keep one of a correlated group essentially arbitrarily and zero the rest, and that choice flips when you resample the data. Adding the L2 term makes the objective strictly convex, so correlated features get shrunk together — that's the grouping effect. You keep most of the sparsity and gain stability, and you pay with an extra hyperparameter.

**14. When to use L1, L2, Elastic Net?**
L2: default; smooth solutions; multicollinearity-robust. L1: feature selection; sparse models needed. Elastic Net: high-dim, correlated features.

> **Saying it out loud.** L2 is the default and you should say so — smooth, stable, good with correlated features. L1 is when you actually want feature selection or a genuinely sparse, interpretable model, typically with far more features than examples. Elastic net is when you want L1's selection but your features are correlated enough that L1 alone is unstable. And for deep learning specifically it's basically always L2 as weight decay, because sparsity in a neural network's weights doesn't buy you anything you can use.

---

## C. Weight decay (vs L2)

**15. For SGD, are L2 and weight decay the same?**
Yes. Gradient of $(\lambda/2) \|w\|^2$ is $\lambda w$, so SGD with explicit decay $w \leftarrow w - \eta g - \eta \lambda w$ is identical to SGD with L2 added to the loss.

> **Saying it out loud.** Yes, exactly the same for SGD. The gradient of the squared-norm penalty is lambda times the weight, so adding L2 to the loss and subtracting lambda-times-weight after the step produce an identical update — it's just algebra. They only come apart when there's a preconditioner sitting between the gradient and the update. That's the setup for the AdamW question, which is almost always what the interviewer is actually driving at.

**16. For Adam, are L2 and weight decay the same?**
No. Adam's $1/\sqrt{\hat v}$ preconditioning divides L2's $\lambda w$ term, weakening regularization where gradient variance is high. Decay strength becomes non-uniform across parameters.

> **Saying it out loud.** No, and this is the classic gotcha. With Adam, L2 goes into the gradient and then gets divided by the square root of the second moment along with everything else — so a parameter with big noisy gradients has a big denominator and its regularisation is quietly weakened. You end up with decay strength varying across parameters in a way you never chose and can't see. Loshchilov and Hutter's point is that this is why Adam had a reputation for generalising worse than SGD, and it was fixable by reordering two operations.

**17. What does AdamW do?**
Decouples weight decay: $\theta \leftarrow \theta - \eta \cdot \hat m / (\sqrt{\hat v} + \varepsilon) - \eta \cdot \lambda \cdot \theta$. Decay applied directly to parameters, not added to gradient. Recovers uniform regularization.

> **Saying it out loud.** AdamW applies weight decay directly to the parameters after the Adam step, instead of folding it into the gradient beforehand. So the decay bypasses the adaptive denominator entirely, and every weight shrinks by exactly eta times lambda regardless of its gradient history. The word to use is decoupled. It's a tiny code change and it's the reason AdamW rather than Adam is the default for essentially every large model trained today.

**18. Typical weight decay for LLM pretraining?**
$\lambda = 0.1$ is the modern default. Earlier and smaller models used $0.01$–$0.05$. SFT and DPO usually use $0.0$ or near-zero.

> **Saying it out loud.** 0.1 is the modern default for LLM pretraining and you'll see it in nearly every published recipe. Older and smaller models used 0.01 to 0.05. For fine-tuning — SFT, DPO — you generally use zero or something tiny, because you're taking few steps on limited data and shrinking weights just drags you away from a checkpoint you already liked. And the detail that impresses: you exclude embeddings, LayerNorm parameters, and biases from decay, because they get sparse or one-dimensional gradients and uniform shrinkage damages them.

---

## D. Dropout

**19. What is dropout?**
For each forward pass, randomly zero each activation with probability $p$; scale remaining by $1/(1-p)$ to preserve expectations. Applied during training only; inference uses all activations.

> **Saying it out loud.** Dropout randomly zeroes each activation with probability $p$ on every training forward pass, and scales the survivors by $1/(1-p)$ so the expected magnitude is unchanged. Different mask every batch, so the network never sees the same architecture twice. At inference you turn it off completely and use every unit. The single most important thing to get right is that the scaling happens at training time, which is what lets inference be a plain unmodified forward pass.

**20. Why scale by $1/(1-p)$?**
Expected activation magnitude during training $= (1-p) \cdot h + p \cdot 0 = (1-p) \cdot h$. Multiplying by $1/(1-p)$ recovers $h$, so train and inference activations have matching scales without changing inference code.

> **Saying it out loud.** Because if you zero a fraction $p$ of the units, the expected sum feeding the next layer drops to $(1-p)$ of what it should be — so downstream layers would see a systematically smaller signal at training than at inference. Dividing by $(1-p)$ restores the expectation. It's called inverted dropout, and the choice to do the correction during training rather than at inference is deliberate: it keeps the deployment path simple and free. If you forget it, nothing crashes — training looks fine and inference is silently miscalibrated, which is the nastiest kind of bug.

**21. Why does dropout work? (multiple stories)**
(a) Ensemble: each forward pass is a random subnetwork; training averages over subnetworks. (b) Prevents co-adaptation: neurons can't rely on specific others to be present. (c) Noise injection in feature space, like data augmentation.

> **Saying it out loud.** Three stories and they're all partly true. The ensemble story: each mask defines a different subnetwork, so you're training exponentially many of them with shared weights, and the scaled full network at inference approximates their average. The co-adaptation story: no neuron can count on any specific other neuron being present, so features have to be individually useful rather than part of a fragile committee. And the noise story: multiplicative Bernoulli noise on activations is data augmentation in feature space. The co-adaptation one is Hinton's original framing and usually lands best in interviews.

**22. Typical dropout rates?**
$0.1$–$0.3$ for hidden layers in moderate-size networks. Original paper used $0.5$ (rare today). Transformers in pretraining: $0.0$–$0.1$. Modern LLMs at scale often use **no dropout** because data is plentiful.

> **Saying it out loud.** 0.1 to 0.3 for hidden layers in moderate networks. The original paper's 0.5 is far too aggressive by modern standards and you rarely see it. Transformers in pretraining sit at 0 to 0.1. The observation worth ending on is that large LLMs frequently use zero dropout, because when you're training on trillions of tokens and seeing most data roughly once, you aren't overfitting at all — so dropout is pure slowdown.

**23. When does dropout hurt?**
Underfitting models (reduces effective capacity). LLMs with abundant data (slows learning). Tasks requiring all features (rare).

> **Saying it out loud.** Whenever you're not actually overfitting. It reduces effective capacity, so if your training loss is already too high, dropout makes both numbers worse. Large-data regimes are the big case — a model doing a single pass over trillions of tokens has no opportunity to memorise, so there's nothing for dropout to prevent. It also interacts badly with batch norm, since you've got two noise sources fighting, which is why the standard ResNet recipe uses batch norm and no dropout. The diagnostic is simple: dropout is for a train-eval gap, and if there's no gap, remove it.

**24. What's DropPath / Stochastic Depth?**
Randomly drop entire residual blocks during training. Used in some vision transformers (ConvNeXt, DeiT). Like dropout but applied at the block level rather than activation.

> **Saying it out loud.** Stochastic depth drops entire residual blocks instead of individual activations — with some probability the block is skipped and only the identity path carries through. Because a residual network is already a sum over many paths, dropping whole blocks is a much coarser and more architectural perturbation than dropping units. It effectively trains a network of random depth and gives you the full depth at inference. It's standard in deep vision transformers, usually with the drop probability ramping up linearly with depth.

**25. Train vs eval mode in PyTorch?**
`model.train()` enables dropout (and BN running stats). `model.eval()` disables dropout (uses BN running averages). Forgetting to switch modes is a classic bug source — leaves dropout active at inference, results are inconsistent.

> **Saying it out loud.** `model.train()` turns dropout on and makes batch norm use the current batch's statistics while updating its running averages. `model.eval()` turns dropout off and makes batch norm use the stored running averages. Forgetting to call eval is one of the most common bugs in practice, and the symptom is nasty — evaluation results that change between runs and are worse than they should be, with nothing raising an error. And the reverse bug exists too: leaving the model in eval during training means batch norm never updates its statistics.

---

## E. Early stopping, label smoothing, augmentation

**26. What is early stopping?**
Train while monitoring validation loss; stop when validation loss stops improving (or starts increasing); restore weights from best validation checkpoint. Prevents fitting noise.

> **Saying it out loud.** Early stopping means watching validation loss and halting when it stops improving, then restoring the best checkpoint rather than the last one. It works because models fit the broad signal early and the noise late, so cutting training short means the noise never gets fitted. It's the cheapest regularizer available since you were computing validation loss anyway. Two practical details: you need a patience window, because validation loss is noisy and stopping on the first uptick is usually premature, and your validation set has to be genuinely held out — early stopping on leaked data is just slow overfitting.

**27. Connection between early stopping and L2?**
For squared loss with gradient flow: early stopping at time $t$ is approximately equivalent to L2 with $\lambda \propto 1/t$. Stopping early = strong L2; running long = weak L2. Friedman's classic result.

> **Saying it out loud.** They're approximately the same thing. For gradient descent on a squared loss, you can show that stopping at step $t$ gives roughly the same solution as ridge regression with lambda proportional to $1/t$. The intuition is that gradient descent picks up the high-curvature directions of the problem first and the low-curvature ones slowly — so stopping early is precisely leaving the small-eigenvalue directions unfitted, which is exactly what ridge suppresses. The consequence worth stating: your training duration is a regularisation hyperparameter, so tuning weight decay and training length independently is a bit of a fiction.

**28. What's label smoothing?**
Replace one-hot labels with $y_{\text{smooth}} = (1 - \varepsilon) \cdot y_{\text{one-hot}} + \varepsilon/K$. Typical $\varepsilon = 0.1$. Prevents the model from learning to push logits to $\pm \infty$.

> **Saying it out loud.** Label smoothing replaces a hard one-hot target with a slightly softened one — typically 0.9 on the true class and the remaining 0.1 spread over the other classes. Epsilon of 0.1 is the standard value. The reason is that with a hard target, cross-entropy is only truly minimised by driving the correct logit to infinity, so the model learns unbounded confidence. Softening the target puts a floor under the achievable loss, and therefore a ceiling on the logits.

**29. Why does label smoothing help?**
Pushes the model toward calibrated confidence. With one-hot labels, cross-entropy can only be 0 by pushing logits to extremes (overconfident). With smoothed labels, there's a non-zero floor — bounded logits are optimal.

> **Saying it out loud.** Because it caps how confident the model is allowed to become. With one-hot targets there's always more loss to shave off by pushing logits further apart, so the model ends up claiming 99.99% on things it gets wrong — badly calibrated. With smoothed targets the optimal logit gap is finite, so confidence stays in a sane range and calibration improves measurably. The tradeoff worth naming: it slightly degrades representations for transfer and it makes distillation harder, since you've deliberately blurred the fine structure in the output distribution that a student would learn from.

**30. What's data augmentation doing mathematically?**
Increases effective training set size by transforming inputs in ways that preserve the label. Equivalent to enforcing invariance under those transforms. Reduces overfitting by exposing the model to more "variations" of the underlying concepts.

> **Saying it out loud.** Augmentation is you telling the model what shouldn't matter. Rotating a cat doesn't make it not a cat, so showing rotated versions encodes rotation invariance without touching the architecture. So it's not merely "more data" — it's a claim about the symmetries of the problem, which is why an augmentation that breaks the label, like mirroring a digit, actively hurts. Formally it's like averaging the loss over a group of transformations, which shrinks the effective hypothesis class. And it's usually the highest-return regulariser you have, ahead of anything you'd do with a penalty term.

**31. What's MixUp?**
$x_{\text{mix}} = \alpha \cdot x_1 + (1 - \alpha) \cdot x_2$, $y_{\text{mix}} = \alpha \cdot y_1 + (1 - \alpha) \cdot y_2$. Linearly interpolate two examples and labels. Encourages the model to behave linearly between training examples. Often improves calibration.

> **Saying it out loud.** MixUp blends two training images pixel-wise and blends their labels in the same proportion — 70% cat plus 30% dog gives you a target that's 0.7 and 0.3. It sounds absurd because the blended image isn't a real image of anything, and it works anyway. What it enforces is linear behaviour between training points, so the model can't build wildly confident sharp decision regions in the space between examples. The consistent finding is that it improves calibration as much as accuracy, which is the detail that shows you've actually used it.

**32. What's CutMix?**
Splice a rectangular region from image 1 into image 2; mix labels by area fraction. Locally cleaner than MixUp; often beats it on image classification.

> **Saying it out loud.** CutMix pastes a rectangular patch from one image into another and mixes the labels by the area of the patch. The advantage over MixUp is that every pixel remains a real pixel from a real image rather than a ghostly blend, so local statistics stay natural — which matters for convolutional features. It also has a localisation benefit, since the model has to recognise a partially occluded object. On image classification it usually edges out MixUp, and in practice recipes often use both, sampling one or the other per batch.

---

## F. Implicit regularization

**33. What's the implicit regularization of SGD?**
Mini-batch noise biases SGD toward flat minima — regions where small perturbations don't increase loss much. Flat minima generalize better. The noise scale $\eta/B$ controls this implicit regularization strength.

> **Saying it out loud.** Mini-batch noise. Every step you're using a random subset, so the gradient you follow is the true gradient plus noise, and the size of that noise scales with learning rate over batch size. That noise makes sharp minima unstable — you get shaken out of a narrow valley but you stay in a wide one — so SGD systematically lands in flat regions, which generalise better. It's implicit because nobody wrote a penalty term; it falls out of the algorithm. The practical consequence: learning rate and batch size are regularisation knobs, not just speed knobs.

**34. Why does this matter?**
Modern deep learning's success is largely due to implicit regularization, not explicit penalties. Overparameterized networks have many functions that perfectly fit training data; SGD picks "nice" ones (close to init, flat in loss landscape). Without this, deep learning wouldn't work as well as it does.

> **Saying it out loud.** Because it's the answer to why deep learning works at all. A model with far more parameters than data points should be free to fit the training set in an infinite number of ways, most of them garbage — and yet gradient descent reliably picks a good one. It doesn't sample uniformly: it stays near initialisation and prefers flat minima. So the effective capacity is set by the optimiser's bias, not by the parameter count, which is why counting parameters tells you almost nothing about overfitting. That's also the setup for double descent.

**35. What's the connection between batch size, learning rate, and generalization?**
Implicit regularization scale $\propto \eta/B$. Larger batches = less noise = less implicit regularization. To preserve generalization when scaling up batch size, scale up $\eta$ proportionally (linear scaling rule for SGD; sqrt for Adam).

> **Saying it out loud.** They're linked through the noise scale, roughly learning rate over batch size. Bigger batches mean less gradient noise, so less implicit regularisation, which is why very large batch training often generalises worse unless you compensate. Compensating means raising the learning rate — linearly for SGD, closer to square-root for Adam — plus longer warmup so the bigger steps don't destabilise early training. The framing that scores is that neither number matters on its own; it's the ratio that sets the implicit regularisation strength.

**36. What's double descent?**
Test loss decreases as model size grows past the interpolation threshold (where training loss = 0). Goes against classical bias-variance: more capacity can be better, not worse. Modern deep learning lives in this overparameterized regime.

> **Saying it out loud.** Double descent is the observation that test error goes down, then up, then down again as you increase model size. The first descent and the rise are the classical bias-variance story, and the peak sits right at the interpolation threshold, where the model has just barely enough capacity to fit the training data exactly and therefore has exactly one way to do it — a brittle one. Push past that and there are many perfect fits available, so the optimiser's implicit bias gets to pick a nice one, and error falls again. There's an epoch-wise version too, and the practical upshot is the counterintuitive advice that if you're at the peak, going bigger can help more than going smaller.

---

## G. Modern and frontier topics

**37. What's Sharpness-Aware Minimization (SAM)?**
$\min_w \max_{\|\varepsilon\| < \rho} L(w + \varepsilon)$. Find weights where the maximum nearby loss is small (flat regions). Practical: 2 forward-backward passes per step (find sharpest direction, then take step there). Costly but improves generalization.

> **Saying it out loud.** SAM chases flat minima on purpose. The reasoning is that a sharp minimum is fragile — nudge the weights or shift the test distribution slightly and the loss jumps — while a flat one is robust, and robustness to perturbation is close to what generalisation means. So instead of minimising the loss where you are, you minimise the worst loss within a small ball around you. Implementation is two passes per step: step uphill to find the nastiest nearby point, take the gradient there, apply it back at your original weights. The tradeoff to end on is blunt — it doubles training cost, which is why it's common in vision research and essentially absent from LLM pretraining.

**38. What's spectral normalization?**
Constrain each weight matrix's largest singular value to 1. Used in GANs (Spectral Norm GAN) for stability. Bounds the Lipschitz constant of the network.

> **Saying it out loud.** Spectral normalisation divides each weight matrix by its largest singular value, so the layer can never amplify an input by more than a factor of one. Stack those and you've bounded the Lipschitz constant of the whole network, which means small input changes can't produce huge output changes. That's why it took hold in GANs — an unbounded discriminator gives the generator explosive gradients and training collapses. In practice you estimate the top singular value with one or two power-iteration steps per forward pass, so it's cheap.

**39. What's batch normalization's regularization effect?**
BN's batch statistics inject noise into activations (different batches give different stats). Acts as a mild stochastic regularizer. Some networks combine BN with explicit dropout; others find BN alone is enough.

> **Saying it out loud.** Batch norm normalises using the current batch's mean and variance, and those statistics are random because the batch is random — so each example's activations depend on whichever other examples happened to be in the batch with it. That's noise injection, and it's a genuine regularisation effect, which is a large part of why batch-normalised networks often need less dropout. The catch is that it makes the effect depend on batch size: very small batches give noisy statistics and unstable training, which is exactly why layer norm took over for transformers, where sequence lengths and batch composition vary.

**40. Why is dropout less common in modern LLMs?**
At scale, LLMs are not data-limited (or barely so). Implicit regularization from SGD/AdamW noise is enough. Dropout slows learning. Many modern recipes use 0 dropout.

> **Saying it out loud.** Because modern LLMs aren't in the overfitting regime. When you make roughly one pass over trillions of tokens, the model never sees an example twice, so there's nothing to memorise and nothing for dropout to prevent. What you get instead is a slower fit for the same compute, which at pretraining scale is an expensive thing to pay for nothing. The implicit regularisation from gradient noise and weight decay covers what's needed. Dropout does come back for fine-tuning, where you're doing many epochs over a small dataset and overfitting is real again.

---

## H. Misc and gotchas

**41. Can L2 alone cause underfitting?**
Yes if $\lambda$ is too high. Strong L2 forces small weights; the model can't fit the data. Should pick $\lambda$ by validation, not by intuition.

> **Saying it out loud.** Yes, straightforwardly — lambda is a dial and you can turn it too far. Strong L2 forces every weight toward zero, and in the limit you get a constant predictor. The diagnostic is that training loss and validation loss are both high and close together, which means underfitting, whereas a big gap between them means overfitting. So you read the direction to move lambda off that gap, not off intuition, and you pick the exact value by validation sweep.

**42. What's wrong with $\lambda = 0$?**
No regularization. With overparameterized models on finite data, can lead to overfitting and weight blowup (especially on separable data — see logistic regression). Practical: use small but non-zero $\lambda$.

> **Saying it out loud.** Usually not much, in deep learning — implicit regularisation carries a lot of the load, and plenty of good models train with very little explicit decay. Where it genuinely breaks is separable data with logistic regression: there's no finite optimum, so the weights grow without bound as the model chases ever-more-confident predictions, and you get numerical trouble along with terrible calibration. The safe habit is a small non-zero lambda by default, since the cost of a slightly-too-large value is mild and the cost of unbounded weights isn't.

**43. How do you tune $\lambda$?**
Validation. Train with multiple $\lambda$ values; pick the one with best validation metric. Log-scale grid ($10^{-5}$ to $10^{-1}$) is standard.

> **Saying it out loud.** By validation sweep on a log scale — powers of ten from about $10^{-5}$ to $10^{-1}$ — then refining around whatever wins. Log scale matters because the effect is multiplicative; trying 0.01, 0.02, 0.03 is wasted effort when the interesting range spans orders of magnitude. Two things to watch: lambda interacts with learning rate and with training length, so re-tune it if you change either, and at LLM scale you can't afford this sweep, which is why everyone just uses 0.1 from published recipes.

**44. Why does data augmentation generalize even on near-IID data?**
The model can memorize specific points but not their continuum of transformations. Augmentation forces learning the underlying invariances (rotation, color, etc.) rather than the specific examples.

> **Saying it out loud.** Because memorisation doesn't survive transformation. A model can store a specific image and the answer that goes with it, but it can't store every crop, rotation and colour shift of that image — there are effectively infinitely many, so the cheapest way to get them all right is to learn the actual invariance. So you're not just enlarging the dataset, you're making the memorisation shortcut unaffordable relative to the general solution. That's why augmentation helps even when your data is already IID with the test set.

**45. Why doesn't regularization replace good data?**
Regularization picks the "simplest" function consistent with the data. If the data is misleading, regularization doesn't fix it. More and better data dominates regularization in most regimes.

> **Saying it out loud.** Because regularization only chooses among functions that fit the data you gave it — if the data is biased or missing a whole regime, every candidate function is wrong there and picking the simplest wrong one doesn't help. Regularization controls variance; it does nothing about bias in the sample itself. And empirically the ordering is stark: more and better data usually beats any amount of hyperparameter tuning, which is why frontier labs spend far more effort on data curation than on regularisation schemes. The way I'd put it: regularization is how you spend a limited data budget well, not a substitute for having one.

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
