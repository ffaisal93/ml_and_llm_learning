# Gradient Descent & Learning Rate — Interview Grill

> **How to use this file.** Cover the answer with your hand. Read the question out loud. Speak your answer out loud, completely, before checking. If your answer is more than 25% off, mark it; come back tomorrow. The answers below are interview-ready: the level of detail you should be able to produce on a whiteboard or in a Zoom screen-share, not just the gist.

---

## A. Foundations (warmup; you must get all of these)

**1. Write down the SGD update rule.**
$\theta_{t+1} = \theta_t - \eta \cdot g_t$, where $g_t = \nabla L(\theta_t; \text{batch}_t)$ is the stochastic gradient computed on a mini-batch. $\eta$ is the learning rate.

> **Saying it out loud.** New weights equal old weights minus the learning rate times the gradient. That's it — the gradient points uphill on the loss, so you step the other way, and $\eta$ decides how far. The word 'stochastic' just means you computed that gradient on a mini-batch instead of the whole dataset, so it's a noisy estimate of the true direction. And that noise isn't purely a cost — it's a big part of why SGD generalizes.

**2. What is the difference between batch, stochastic, and mini-batch gradient descent?**
Batch GD computes the gradient over the entire training set per step — expensive but exact in expectation. SGD uses a single sample per step — cheap, very noisy. Mini-batch uses $B$ samples per step (typical 32–8192) and is the practical default because it amortizes per-step overhead, gives reasonable gradient estimates, and matches accelerator hardware. The variance of the mini-batch gradient is roughly $\sigma^2 / B$, where $\sigma^2$ is the per-sample variance.

> **Saying it out loud.** They're the same update; the only difference is how many examples you look at before taking a step. Full batch gives you the exact gradient but you get one step per pass over the data, which is hopeless. Pure SGD gives you a step per example but the direction is wildly noisy. Mini-batch is the compromise everyone actually uses, and the reason is partly statistical — variance falls as $\sigma^2/B$, so you get diminishing returns from bigger batches — and partly hardware, since a GPU wants a few thousand examples at once to stay busy.

**3. What does the learning rate control?**
The step size in parameter space. For a quadratic loss with Hessian $H$, GD converges if $0 < \eta < 2/\lambda_{\max}(H)$. Above that, iterates diverge. The optimal rate for a quadratic is $\eta = 2/(\lambda_{\max} + \lambda_{\min})$, and convergence rate is governed by the condition number $\kappa = \lambda_{\max} / \lambda_{\min}$.

> **Saying it out loud.** The learning rate is how big a step you take, and there's a hard ceiling on it set by the curvature. For a quadratic bowl you're stable only while $\eta$ is below $2/\lambda_{\max}$ — past that, the sharpest direction overshoots and amplifies every step, and you diverge. The uncomfortable part is that the same $\eta$ has to serve the flattest direction too, so how fast you converge is governed by the condition number, the ratio of sharpest to flattest curvature. That single fact is why we invented momentum, Adam, and everything after them.

**4. What happens if your learning rate is too high?**
Updates overshoot in the sharpest curvature direction, iterates oscillate with growing amplitude, loss becomes NaN within steps to hundreds of steps. In Adam, you may also see "loss spikes" that recover — symptoms of operating past the stability edge.

> **Saying it out loud.** Too high and you overshoot in the sharpest direction, and the overshoot compounds. Each step lands further up the opposite wall than where you started, so the oscillation grows and the loss goes to NaN — sometimes in ten steps, sometimes in a few hundred. With Adam the picture is muddier because you'll often see loss spikes that recover, which means you're bouncing around the stability edge rather than past it. The diagnostic I'd reach for is the gradient norm over time: a clean exponential blow-up means LR, not data.

**5. What happens if your learning rate is too low?**
Convergence is slow; you may run out of compute budget before reaching a useful loss. You may also fail to escape saddle points or shallow local minima. Per-layer update-to-weight ratio drops below $10^{-4}$.

> **Saying it out loud.** Too low and nothing is broken, which is why it's the more expensive mistake. Loss goes down, just too slowly, and you burn your entire compute budget getting to a mediocre place. You also lose the noise that helps you escape saddles and shallow basins, so you can sit near a critical point for a long time. The number I'd check is the update-to-weight ratio: if it's down at $10^{-5}$ or $10^{-6}$ instead of around $10^{-3}$, you're barely moving.

**6. What's the relationship between gradient descent and Newton's method?**
Newton's method is $\theta_{t+1} = \theta_t - H_t^{-1} g_t$. It uses second-order curvature information to choose step direction *and* size. GD is the first-order approximation: $H_t^{-1} \approx \eta \cdot I$, i.e. assume the loss is locally a sphere and use a fixed scalar. Newton converges quadratically near a minimum; GD converges linearly with rate determined by $\kappa$. Adam is a coarse *diagonal Fisher* approximation: $\mathbb{E}[g g^\top]$ estimates the Fisher information matrix, which equals the Hessian only at a stationary point of negative log-likelihood. So calling Adam "diagonal Newton" is loose — strictly it's diagonal natural gradient.

> **Saying it out loud.** Newton uses the actual curvature to pick both the direction and the step size — multiply the gradient by the inverse Hessian and, on a quadratic, you land on the minimum in one step. Gradient descent is the cheap approximation where you pretend the Hessian is $\eta$ times the identity, meaning you assume the loss looks like a perfect sphere. That's why GD suffers when the condition number is bad: the sphere assumption is wrong. And Adam sits in between — the honest framing is that it's a diagonal natural-gradient method, not a diagonal Newton, because it estimates the Fisher and the Fisher only equals the Hessian at a stationary point.

---

## B. Learning rate behavior

**7. Why is no single learning rate good for the whole network?**
Because the curvature of the loss varies across layers, across parameters within a layer, and across training phases. Embedding tables and early-layer features are typically ill-conditioned compared to later layers. A global $\eta$ is necessarily a compromise — too small for flat directions, too large for sharp ones. This is exactly the problem Adam, AdamW, LARS, LAMB, and muP all attempt to fix.

> **Saying it out loud.** Because different parts of the network live in wildly different curvature. An embedding row that only gets touched when its token appears has a totally different gradient scale from a late-layer weight that gets a signal every step. One global $\eta$ has to be small enough not to blow up the sharpest direction, which means it's far too small for everything flat — so most of the network is under-trained so that one part doesn't explode. That single observation is the motivation for Adam, LARS, LAMB, and muP; they're all different ways of giving each parameter or layer its own effective step size.

**8. Why does the same model on the same data sometimes need different learning rates on different hardware?**
Because batch size depends on memory and parallelism, and the optimal LR depends on batch size. Linear scaling rule: scaling batch by $k$ requires scaling $\eta$ by $k$ (SGD) or roughly $\sqrt{k}$ (Adam) to maintain trajectory. Different hardware setups also produce slightly different numerics (fp16/bf16/fp32 mix), which can change the stable LR range.

> **Saying it out loud.** Because the hardware determines your batch size, and the right learning rate depends on the batch size. More GPUs means a bigger batch means a less noisy gradient, and a less noisy gradient can support a bigger step — linearly for SGD, closer to square root for Adam. So the identical code on eight GPUs versus sixty-four is genuinely a different optimization problem unless you rescale. Numerics play a smaller role too: a bf16 run and an fp32 run have different stable LR ranges, which is why a recipe ported across clusters sometimes just diverges.

**9. Walk me through how you'd find the right learning rate from scratch.**
LR finder (Leslie Smith): start $\eta$ very low ($10^{-7}$), multiply by ~1.1 per step, plot loss vs. $\log \eta$. Pick $\eta$ an order of magnitude below the divergence point — i.e. where loss is still descending steeply. Verify by checking update-to-weight ratio is around $10^{-3}$ per layer in the first few thousand steps. Adjust schedule from there.

> **Saying it out loud.** I'd run an LR range test before committing anything. Start absurdly low, like $10^{-7}$, and multiply by about 1.1 every step while plotting loss against log LR — you get a curve that's flat, then descends steeply, then blows up. Pick something roughly an order of magnitude below the blow-up point, in the steep part. Then I'd sanity check it by looking at the per-layer update-to-weight ratio for the first few thousand steps and confirming it's around $10^{-3}$, because the range test gives you a starting point, not a schedule.

**10. Why is the update-to-weight ratio ($\|\eta \cdot \text{update}\| / \|\theta\|$) a better diagnostic than the loss curve?**
Because it works per layer and reveals heterogeneity that a global loss curve hides. A loss curve might look fine while one layer is stuck at $10^{-6}$ and another is exploding at $10^{-1}$. A healthy training run has the ratio around $10^{-3}$ for most layers. Karpathy's heuristic, widely cited.

> **Saying it out loud.** Because the loss curve is one number for the whole model and it hides everything. You can be at a perfectly healthy-looking loss while your embeddings are frozen at a ratio of $10^{-6}$ and one attention block is being blown apart at $10^{-1}$ — the average looks fine and the model is quietly broken. The update-to-weight ratio is per-layer, dimensionless, and has a known healthy value around $10^{-3}$, so it tells you *where* the problem is rather than just that there is one. It's Karpathy's heuristic and it's the first thing I'd log on a new training run.

**11. What is the edge of stability phenomenon?**
**Loss spikes can be a feature, not a bug.** Cohen et al. 2021: in deep-net training, the largest Hessian eigenvalue drifts upward until it pins near $2/\eta$ — exactly the GD stability boundary. Loss bounces non-monotonically but globally descends. Classical convergence theory (smooth descent to minimum) doesn't describe real training.

> **Saying it out loud.** The classical picture says you pick $\eta$ below $2/\lambda_{\max}$ and descend smoothly. What actually happens in deep nets is the reverse: the sharpness climbs during training until it pins right at $2/\eta$ and then hovers there. So the model finds its way to the edge of the stability boundary and stays there, which means the loss bounces around non-monotonically while still trending down. The practical upshot is huge — occasional loss spikes are what training at a good learning rate *looks like*, and reflexively lowering LR to smooth the curve just wastes compute.

**12. If your loss spikes occasionally during pretraining but training overall is healthy, what do you do?**
Don't panic. Add gradient clipping at norm 1.0 if not already present. Investigate the spike batch (sometimes a single bad sequence). Don't reflexively lower the global LR — that may actually slow you below the edge of stability and waste compute. The spike is often a feature of operating at the optimal LR. Lower LR only if spikes are getting worse over time or producing NaNs.

> **Saying it out loud.** Nothing dramatic. Make sure gradient clipping is on at norm 1.0, then go look at the batch that caused it — surprisingly often it's one pathological document. What I would *not* do is immediately drop the global learning rate, because edge-of-stability says spikes are expected when you're running at a good LR, and lowering it costs you real training progress. The trigger for actually intervening is a trend: spikes getting more frequent or larger over time, or one that produces a NaN instead of recovering.

**13. Why do you need learning-rate warmup?**
Three reasons: (a) Adam's second-moment $\hat v$ is noisy and biased low for the first few hundred steps, so updates are oversized; (b) residual streams in transformers are not yet calibrated near init, gradients are oversized; (c) the loss landscape near random init can be pathologically sharp. Without warmup, transformers near-deterministically diverge at modern scales. Typical warmup is 0.5–5% of total steps; 2000 steps for LLM pretraining is common.

> **Saying it out loud.** Warmup exists because the first few hundred steps are the most dangerous ones. Adam's second-moment estimate starts at zero and is noisy and unreliable early, so the updates it produces are effectively oversized; the network at random init has pathologically sharp curvature; and in a transformer the residual stream isn't calibrated yet. Ramping the LR from near zero over a couple thousand steps gets you past all three. Without it, large transformers don't just train worse, they reliably diverge — it's usually 0.5 to 5 percent of total steps, and it's not optional at scale.

**14. What's the linear scaling rule?**
Goyal et al. 2017: when you multiply batch size by $k$, multiply $\eta$ by $k$ to maintain training dynamics — for SGD, with warmup. The intuition is that per-epoch progress depends on $\eta \cdot k$. It works on ResNets up to batch size ~8192, and breaks at very large batches (the "critical batch size"). For Adam, sqrt scaling is closer.

> **Saying it out loud.** Double the batch, double the learning rate. The intuition is that a batch twice as large gives a gradient with half the variance, so you can afford to trust it twice as much — and you're taking half as many steps per epoch, so you need each one to count double. Goyal's ImageNet-in-an-hour result showed this holds for ResNets up to about batch 8192 with warmup. Two caveats worth naming: it breaks past the critical batch size, and for Adam the empirical scaling is closer to square root because Adam already normalizes by gradient magnitude.

**15. What is the critical batch size?**
**The batch size where doubling stops paying off.** McCandlish et al. 2018: beyond this point, even with optimal LR rescaling, you don't get proportional speedups. Determined by the gradient noise scale (mean gradient magnitude / gradient covariance trace). Importantly, it *grows during training*, so the right batch size is not static. Frontier labs track this because it determines data-parallel ROI.

> **Saying it out loud.** It's the batch size past which buying more GPUs stops buying you speed. Below it, doubling the batch roughly halves the number of steps you need; above it, you're spending twice the compute for almost no reduction in steps, because your gradient was already accurate enough. What sets it is the gradient noise scale — how big the gradient's variance is relative to its mean. The detail people miss is that it grows during training, since the gradient gets smaller and noisier as you converge, so the optimal batch size early on is much smaller than late on, and that's why big runs ramp batch size.

**16. How would you transfer a learning rate from a 1B-parameter model to a 70B-parameter model?**
Use muP (Yang & Hu 2022), which scales initialization, embedding, and per-layer learning rates so the optimal LR is width-invariant. Sweep $\eta$ cheaply on small models, scale up. Without muP, large models often need lower $\eta$, and the ratio depends on width, depth, and details of the architecture in a non-trivial way.

> **Saying it out loud.** With muP, and honestly without it the answer is 'you guess and pray'. Standard parameterization makes the optimal learning rate drift as you widen a model, so what you tuned at 1B is simply wrong at 70B, and nobody can afford an LR sweep at 70B. muP rescales the initialization, the per-layer learning rates, and the embedding multipliers so that the optimal LR becomes width-invariant. Then you sweep cheaply on a small proxy model and transfer the number directly — that's the whole value proposition, and it's why every serious lab uses some form of it.

---

## C. Schedules

**17. What's the standard LR schedule for LLM pretraining?**
Linear warmup over ~0.5–2% of steps, then cosine decay to ~10% of peak. Some recent runs use linear or trapezoidal decay instead of cosine. Pure constant LR is rare in pretraining; pure step decay is dated.

> **Saying it out loud.** Linear warmup for the first percent or two of steps, then cosine decay down to about ten percent of the peak. That's the recipe almost everything from GPT-3 onward has used. There's been some movement lately toward trapezoidal or linear decay, mostly because cosine forces you to commit to a total step count up front and a constant-then-decay shape lets you extend a run. Constant LR alone is basically never used for pretraining, and step decay is a relic from the ImageNet era.

**18. Why cosine decay specifically?**
Smooth — no sudden drops that shock the optimizer. Spends roughly half the budget at relatively high LR (where most progress happens) and gradually transitions to fine-grained polishing. Empirically dominates step decay on most large-scale tasks. Linear decay is competitive and easier to reason about for compute-budget scaling.

> **Saying it out loud.** Cosine wins mostly by being smooth and by spending its budget in the right places. It sits at high LR for a good chunk of training, where most of the actual progress happens, then eases down gradually instead of dropping off a cliff the way step decay does — and sudden drops shock the optimizer's moment estimates. Empirically it just beats step decay at scale, and the theoretical stories are post-hoc. The honest caveat is that linear decay is competitive and much easier to reason about when you want to change your compute budget mid-run.

**19. What's `ReduceLROnPlateau` and when is it useful?**
A schedule that drops $\eta$ (typically by 10x) when validation loss has not improved for $N$ epochs/steps. Useful for fine-tuning, transfer learning, and any setting where you can't predict the optimal total schedule in advance. Less common in pretraining where compute budget is fixed and cosine works well.

> **Saying it out loud.** It's the reactive schedule: watch validation loss, and when it stops improving for a few evaluations, cut the LR by 10x. That's useful exactly when you can't predict the right schedule in advance — fine-tuning, transfer learning, small datasets where you don't know how long you'll train. In pretraining it's rare because you know your compute budget up front, so a planned cosine works better and doesn't waste steps on a plateau you had to detect first. The failure mode is that it's laggy by construction — you only cut after you've already wasted the patience window.

**20. Should $\eta_{\min}$ at the end of cosine be zero or positive?**
Positive, typically $0.1 \cdot \eta_{\max}$. Pure zero LR at the tail provides no gradient signal and may cause optimizer state to drift. Recent practice keeps $\eta_{\min}$ non-trivial for stability.

> **Saying it out loud.** Positive — typically about ten percent of the peak. Decaying all the way to zero means the last stretch of training does essentially nothing, so you've paid for steps that don't move the weights, and the optimizer's moment estimates go stale. Keeping a floor also leaves you in a better place if you want to continue training later from that checkpoint. It's a small detail that shows up in every modern recipe precisely because people learned it the hard way.

---

## D. Optimizers

**21. What does Adam do that SGD does not?**
Tracks per-parameter first moment $m_t$ (gradient mean) and second moment $v_t$ (gradient squared mean). Updates use $\hat m_t / (\sqrt{\hat v_t} + \varepsilon)$, which gives per-parameter step sizes adapted to per-parameter gradient magnitudes. This makes Adam much more tolerant to ill-conditioned problems and removes most of the LR-tuning sensitivity that SGD has. It also incorporates momentum.

> **Saying it out loud.** Adam gives every parameter its own step size. It keeps two running averages — the mean gradient, which is momentum, and the mean squared gradient, which measures how big that parameter's gradients typically are — and then divides one by the square root of the other. The effect is that a parameter with consistently tiny gradients gets scaled up and one with huge gradients gets scaled down, so a badly conditioned problem stops being a problem. That's why it dominates for transformers, and it's why Adam is far less sensitive to your LR choice than SGD.

**22. Walk me through Adam with bias correction.**
**Verbal story**: "Adam = momentum (first moment) + per-parameter scaling by RMS gradient (second moment) + a small correction so the first few steps aren't biased toward zero."

**Math**:
$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad &\text{(first moment / momentum)} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad &\text{(second moment / RMS)} \\
\hat m_t &= m_t / (1 - \beta_1^t),\quad \hat v_t = v_t / (1 - \beta_2^t) \quad &\text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \eta \cdot \hat m_t / (\sqrt{\hat v_t} + \varepsilon)
\end{aligned}
$$

Defaults: $\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}$. Bias correction matters for the first few thousand steps; afterward $1 - \beta^t \approx 1$.

> **Saying it out loud.** Three ingredients. First, momentum: an exponential moving average of the gradient, so you keep going in a consistent direction and average out the noise. Second, an exponential moving average of the *squared* gradient, which is a per-parameter estimate of gradient magnitude — you divide by its square root, so every parameter ends up taking a similarly sized step in relative terms. Third, bias correction, because both averages start at zero and would be biased small early on, so you divide by $1 - \beta^t$ to undo that. Defaults are 0.9 and 0.999, and the epsilon in the denominator keeps you from dividing by zero.

**23. Why does Adam need bias correction?**
The moving averages are initialized at zero. With $\beta_1 = 0.9$, $m_t$ is biased toward zero for the first ~10 steps; with $\beta_2 = 0.999$, $v_t$ is biased toward zero for the first ~1000. Without correction, the early effective LR is wrong: small $\hat v$ would inflate the update, but small $\hat m$ would deflate it. Bias correction $1/(1 - \beta^t)$ recovers the intended estimate magnitude.

> **Saying it out loud.** Because both moving averages start at zero, so early on they're dragged toward zero by the initialization rather than by the data. With $\beta_2$ at 0.999, the second moment takes on the order of a thousand steps to forget its zero start — so without correction, $\hat v$ is far too small and the update is far too big at exactly the moment the model is most fragile. Dividing by $1 - \beta^t$ rescales each estimate to what it would be if you'd started from an unbiased average. It only matters for the first few thousand steps, after which the correction factor is essentially one, but those are the steps where runs die.

**24. What does $\varepsilon$ in Adam actually do?**
Numerical floor: prevents division by zero when $\hat v_t$ is tiny, and caps the maximum effective per-parameter LR. When $\sqrt{\hat v} \ll \varepsilon$, the update is roughly $(\eta / \varepsilon) \cdot \hat m$, so dimensions with very small gradients still get sensible updates. Some recipes set $\varepsilon = 10^{-3}$ or higher (especially for embeddings) to dampen aggressive updates on rarely-touched parameters.

> **Saying it out loud.** Epsilon does two jobs. The obvious one is preventing a divide-by-zero when a parameter's gradients have been near zero for a while. The subtler and more important one is that it caps the maximum effective step size — once $\sqrt{\hat v}$ falls below epsilon, the update saturates at roughly $\eta/\varepsilon$ times the momentum, instead of exploding. That's why some recipes deliberately raise epsilon to $10^{-3}$ for embeddings: rare tokens have tiny second moments, and without a bigger floor Adam would slam them with enormous updates the few times they appear.

**25. How is AdamW different from Adam with L2 regularization?**
Naive L2 adds $\lambda \theta$ to the gradient. Adam then divides everything by $\sqrt{\hat v}$, so the L2 contribution gets weakened wherever $\hat v$ is large — i.e. regularization is no longer uniform across parameters, contrary to intent. AdamW decouples weight decay: $\theta_{t+1} = \theta_t - \eta \cdot \hat m / (\sqrt{\hat v} + \varepsilon) - \eta \cdot \lambda \cdot \theta_t$. Weight decay is now a uniform fractional shrinkage independent of gradient statistics, recovering the regularization behavior people thought they were getting.

> **Saying it out loud.** The difference is what the preconditioner does to your regularizer. If you add L2 to the loss, that $\lambda\theta$ term rides through Adam's division by $\sqrt{\hat v}$ along with everything else — so parameters with big gradients get *less* decay than parameters with small ones, which is exactly backwards from what you wanted. AdamW pulls the decay out of the gradient path and applies it directly to the weights, so it's a uniform fractional shrinkage regardless of gradient statistics. That one change is worth a real accuracy improvement, which is why every LLM recipe says AdamW and not Adam.

**26. For SGD, are L2 and weight decay the same thing?**
Yes. The gradient of $(\lambda/2)\|\theta\|^2$ is $\lambda \theta$, so SGD with explicit weight decay is identical to SGD with L2 added to the loss. They diverge only when there is preconditioning (Adam, RMSProp, etc.).

> **Saying it out loud.** Yes, exactly the same — the gradient of $(\lambda/2)\|\theta\|^2$ is $\lambda\theta$, so adding it to the loss and subtracting it from the weights are literally the same arithmetic. The distinction only appears when something rescales the gradient before it reaches the weights. That's what preconditioning does, and it's why Adam, RMSProp, and Adagrad all need the decoupled version while plain SGD doesn't care. It's a nice test question because people memorize 'weight decay and L2 are different' without knowing the condition under which that's true.

**27. What's RMSProp and how does it relate to Adam?**
RMSProp = Adam's second moment without the first. $v_t = \beta v_{t-1} + (1 - \beta) g_t^2$, then update by $\eta g_t / \sqrt{v_t}$. It adapts per-parameter LR by gradient variance but doesn't use momentum. Adam = RMSProp + momentum + bias correction. RMSProp is a useful building block to remember; Adam dominates in practice.

> **Saying it out loud.** RMSProp is Adam minus momentum. You keep the running average of squared gradients and divide by its square root, so you get the per-parameter step-size adaptation, but you use the raw gradient rather than a smoothed one for direction. Adam is basically RMSProp with momentum added and the bias correction bolted on. It's worth knowing because it makes clear that Adam's two ideas are separable — the adaptive scaling and the momentum are doing different jobs.

**28. What's Lion and when might you prefer it?**
Lion (Chen et al. 2023, *Symbolic Discovery of Optimization Algorithms*) uses $\mathrm{sign}(\text{momentum})$ instead of $\hat m / \sqrt{\hat v}$. It's roughly half the memory of Adam (no second moment) and sometimes matches or beats AdamW on language modeling. The trade-off: it's more sensitive to LR and weight decay choice; it works best at smaller LR values (3–10x smaller than Adam). Frontier-lab interviews increasingly ask about it.

> **Saying it out loud.** Lion drops the second moment entirely and just takes the sign of the momentum, so every parameter moves by the same fixed magnitude and only the direction varies. The immediate payoff is memory: no $v$ buffer means roughly half the optimizer state, which at 70B parameters is tens of gigabytes you get back. It came out of a symbolic search over optimizer programs, and it matches or beats AdamW on some language modeling runs. The tradeoff to name is that because every step is the same size, it's much more sensitive to LR and weight decay — you typically need an LR 3 to 10x smaller than Adam's.

**29. Why does SGD with momentum sometimes generalize better than Adam?**
Several stories, none fully settled. (a) Adam's per-parameter rescaling can overfit to the noise in $\hat v$, biasing toward sharper minima. (b) SGD's gradient noise is a stronger implicit regularizer toward flat minima. (c) Adam's effective trajectory differs from SGD's, and the geometric properties of the resulting solutions differ. Result: for many vision tasks, well-tuned SGD+momentum wins on validation. For large language models, AdamW dominates because of conditioning issues SGD can't handle.

> **Saying it out loud.** Nobody has fully settled this, and saying so is the right answer. The leading stories are that Adam's per-parameter rescaling amplifies directions where the gradient is small and noisy, which biases it toward sharper minima, and that SGD's raw gradient noise is a stronger implicit push toward flat solutions that generalize. The empirical picture is that well-tuned SGD with momentum still wins on many vision benchmarks. But for language models it's not close — the conditioning is so bad across embeddings and layers that SGD can't be tuned into competitiveness, so AdamW wins regardless of the generalization argument.

**30. What's the "Adam generalization gap" and how is it usually mitigated?**
The empirical observation that Adam achieves equal or lower training loss than SGD but worse validation accuracy on some tasks. Mitigations: AdamW (decoupled decay), longer training, explicit regularization, switching to SGD for the final phase (the "AdamSwitch" trick).

> **Saying it out loud.** It's the observation that Adam gets to a lower *training* loss than SGD but a worse validation number on certain tasks — mostly vision. So it's not that Adam optimizes worse, it's that where it lands generalizes worse. The main practical fix turned out to be AdamW, since a lot of the original gap was really a broken weight-decay implementation. Beyond that: train longer, add explicit regularization, or switch to SGD for the final phase to get Adam's fast early progress and SGD's endpoint. On LLMs the gap doesn't really show up, so it's mostly a vision-era concern.

---

## E. Stability, scaling, and noise

**31. What is gradient clipping and when do you use it?**
$g \leftarrow g \cdot \min(1, c / \|g\|)$. Caps gradient norm at $c$ (clip-by-norm; clip-by-value also exists). Use it (a) to absorb occasional gradient spikes from bad batches, (b) almost always for RNNs (which have exploding-gradient issues), (c) standard in LLM pretraining at $c = 1.0$. Different from LR: clipping targets outliers; LR scales everything.

> **Saying it out loud.** Clipping rescales the gradient whenever its norm exceeds a threshold, so the direction is preserved but the magnitude is capped. The point is that it targets outliers only — on a normal step it does nothing at all, which is what makes it different from just lowering the learning rate, which shrinks every step. You use it because one pathological batch can produce a gradient a hundred times normal and destroy a run that was otherwise fine. It's essentially mandatory for RNNs and standard at norm 1.0 in every LLM pretraining recipe.

**32. Why does gradient noise help generalization?**
Implicit regularization: SGD biases toward flat minima. Three views — escape mechanism (steep walls in sharp minima get rejected by noise), Bayesian view (SGD samples from a posterior with temperature $\eta/B$), explicit view (SGD $\approx$ GD on $L + (\eta/4B) \cdot \|\nabla L\|^2$, an extra penalty on high-gradient regions). The relevant quantity is $\eta/B$; raising batch size without raising LR removes this regularization.

> **Saying it out loud.** Noise keeps you out of narrow crevices. A sharp minimum has steep walls, so a noisy step is likely to bounce you out of it, while a wide flat basin is stable under perturbation — so SGD preferentially ends up in flat minima, which tend to generalize better. There's a Bayesian version of the same story where SGD is sampling from a posterior at temperature $\eta/B$, and an analytic version where SGD behaves like GD on the loss plus a penalty on the gradient norm. The number that matters in all three is $\eta/B$, and the practical warning is that if you scale up batch size without scaling LR, you silently turn this regularization off.

**33. Why is large-batch training hard?**
Several reasons compound: (a) optimal $\eta$ scaling is non-trivial (linear for SGD, sqrt for Adam, both break at very large batches); (b) implicit regularization weakens as $\eta/B$ shrinks; (c) per-step gradient is closer to the true gradient, so the optimizer makes more deterministic progress along sharp directions, which can hurt generalization. McCandlish's critical batch size puts a hard ceiling on practical parallelism.

> **Saying it out loud.** Large batches run into three problems at once. The LR rescaling that keeps dynamics intact stops working past a point — linear for SGD, sqrt for Adam, and both break eventually. The implicit regularization scales with $\eta/B$, so a bigger batch without a bigger LR means less of the noise that helps generalization. And the gradient becomes so accurate that the optimizer marches confidently down sharp directions it would previously have been jostled out of. Underneath it all is the critical batch size: past that, extra data parallelism buys you basically nothing, which is a hard ceiling on how much you can parallelize one run.

**34. What is the gradient noise scale?**
Roughly $\mathrm{tr}(\Sigma) / \|\mu\|^2$, where $\Sigma$ is the gradient covariance and $\mu$ is the mean gradient. Large noise scale → stochasticity dominates → small batch is fine. Small noise scale → gradient is well-estimated → larger batch helps. Critical batch size is approximately the noise scale. Importantly, noise scale grows during training, so optimal batch size grows as you train.

> **Saying it out loud.** It's a measure of how noisy your gradient is relative to its size — roughly the trace of the gradient covariance divided by the squared norm of the mean gradient. Intuitively, it tells you how many samples you'd need before the direction stops being dominated by noise. Big noise scale means a small batch is fine because you're drowning in variance either way; small noise scale means the gradient is well-determined and you can profitably average more. It's essentially the critical batch size, and it grows through training, which is why the optimal batch size isn't a constant.

**35. What is muP and why does it matter for frontier labs?**
Maximal update parameterization (Yang & Hu 2022). A specific scaling of initialization, learning rate per layer, and embedding multipliers such that the optimal LR is *invariant under model width*. Practically: tune hyperparameters on a small model under muP, scale up to 70B with the same $\eta$. Without muP, hyperparameters sweep at 70B is prohibitively expensive and small-model hyperparameters don't transfer.

> **Saying it out loud.** muP is a parameterization that makes the optimal learning rate independent of model width. Under standard init, activation and gradient magnitudes drift as you widen, so the best LR shifts and hyperparameters found on a small model are wrong on a big one. muP fixes the scaling of initialization, per-layer learning rates, and the embedding and output multipliers so that the training dynamics are the same at any width. Why frontier labs care: an LR sweep at 70B is unaffordable, so being able to tune on a 40M proxy and transfer directly is worth a very large amount of money.

**36. What's the difference between LARS, LAMB, and muP?**
LARS/LAMB enforce a layer-wise trust ratio at runtime: $\eta_{\text{layer}} = \eta \cdot \|\theta_{\text{layer}}\| / \|\text{update}_{\text{layer}}\|$. They make large-batch training stable by keeping update-to-weight ratio constant per layer. muP changes the parameterization (init scales and per-layer LR factors) so that the right thing happens automatically, without runtime trust-ratio computation. muP is more elegant; LAMB is older and explicit.

> **Saying it out loud.** LARS and LAMB fix the problem at runtime; muP fixes it at design time. LARS and LAMB compute, for each layer, the ratio of weight norm to update norm and rescale the step so every layer moves by the same relative amount — that's what made batch sizes of 32K trainable for BERT and ResNet. muP instead changes the parameterization itself, so the correct per-layer scaling falls out of the initialization and LR multipliers with no runtime computation and no extra norms. muP is the cleaner idea and it also gives you hyperparameter transfer across widths, which trust ratios don't.

---

## F. Diagnostics and debugging (stuff that wins offers)

**37. Loss is flat. Is this a learning-rate problem?**
Maybe. Check: gradient norm. If $\|g\| \to 0$, you're stuck at a critical point; consider warm restart or perturbation. If $\|g\|$ is healthy but loss isn't moving, your $\eta$ is too small. If $\|g\|$ is huge but loss isn't moving, you're oscillating across a sharp ridge — $\eta$ is too large in that direction. Per-layer update-to-weight ratio narrows it down.

> **Saying it out loud.** Maybe, and the gradient norm tells you which. If the gradient norm has collapsed toward zero, you're sitting at a critical point and the LR isn't the issue — you need a restart or a perturbation. If the gradient norm looks healthy but the loss isn't moving, your steps are too small, so raise the LR. And if the gradient norm is enormous while the loss stays flat, you're ping-ponging across a sharp ravine and the LR is too big. Then I'd break it down per layer with the update-to-weight ratio, because 'flat loss' is often one broken layer, not a global problem.

**38. Loss diverges to NaN at step 1.**
Almost always one of: LR way too high, fp16 overflow in the forward pass (not optimizer-related; check activation magnitudes), bad initialization, or division-by-zero somewhere (softmax of all-$-\infty$, or $\sqrt{0}$ without $\varepsilon$). Lower LR by 10x first; if still NaN, suspect non-LR causes.

> **Saying it out loud.** NaN at step one is almost never subtle. In order of likelihood: the learning rate is wildly too high, you've got an fp16 overflow in the forward pass, the initialization is broken, or there's a genuine divide-by-zero — a softmax over an all-masked row, or a square root of zero without an epsilon. The fast triage is to drop the LR by 10x and rerun: if it still NaNs immediately, it's not the optimizer, and I'd go print activation magnitudes layer by layer to find where the forward pass blows up.

**39. Loss diverges to NaN around step 100–500.**
Classic warmup-end signature. The peak $\eta$ is too high; add or extend warmup. Also possible: a single bad batch with extreme gradient norm — gradient clipping fixes it. Also possible: Adam $\hat v$ accumulating an outlier — same fix.

> **Saying it out loud.** NaN in the low hundreds of steps is the classic end-of-warmup signature — you survived the ramp and died the moment you hit peak LR, which means the peak is too high or the warmup was too short. The other two candidates are a single toxic batch producing a huge gradient, and Adam's second moment absorbing an outlier and then mis-scaling for hundreds of steps afterward. Gradient clipping at 1.0 handles both of those. So the fix order is: turn on clipping, extend warmup, then lower the peak LR if it still dies.

**40. Loss decreases on training but eval loss increases (overfitting).**
Not directly an LR question, but interviewers often ask whether LR can fix it. Lower LR may help by under-fitting; better answers add regularization (weight decay, dropout), early stopping, or data augmentation. The cleanest answer: LR shouldn't be your first lever for overfitting.

> **Saying it out loud.** That's overfitting, and the learning rate is the wrong lever for it. You can technically make the gap smaller by training worse, but that's just trading one problem for another. The real answers are more regularization — weight decay, dropout, augmentation — early stopping on the validation curve, or more data. The one honest LR connection is that a *higher* LR is mildly regularizing through gradient noise, so the naive instinct to lower it can actually make overfitting worse.

**41. Different layers learn at very different rates. What do you do?**
Per-layer LR (LARS/LAMB or muP-style scaling). For Adam, increase $\varepsilon$ for the lagging layer; for SGD, give it a higher per-layer multiplier. In practice, "BERT layer-wise decay" (lower LR for earlier layers during fine-tuning) is a related, simpler fix.

> **Saying it out loud.** Give them different learning rates — that's what LARS, LAMB, and muP all do, just with different machinery. LARS and LAMB compute a per-layer trust ratio at runtime so every layer's relative step is equalized; muP bakes the right multipliers into the parameterization. For a quick fix within Adam, raising epsilon on a layer that's over-updating damps it, and a per-group LR multiplier is the blunt instrument. In fine-tuning there's a simpler version everyone uses: layer-wise LR decay, where earlier layers get a smaller LR because you want to disturb the pretrained features less.

**42. The loss spikes once and recovers. Should I lower the LR?**
Probably not as a first response. Add or tighten gradient clipping. Investigate the spike batch (sometimes a single token sequence). Edge of stability predicts that operating at the optimal $\eta$ *will* produce occasional spikes. Lower the LR only if spikes get worse over time or actually NaN.

> **Saying it out loud.** Probably not. A single spike that recovers is what training at a good learning rate actually looks like — edge of stability says the sharpness settles right at $2/\eta$, so you're expected to bounce. The first response is to make sure gradient clipping is on and to go look at what was in that batch, because a single weird sequence is a common culprit. You lower the LR when the pattern changes: spikes getting bigger or more frequent over time, or one that goes to NaN instead of recovering.

**43. I changed batch size and now training is unstable. What's the first thing you check?**
That $\eta$ was scaled appropriately. For Adam, sqrt scaling. For SGD, linear scaling. Then check warmup length — bigger batch = need longer warmup because each step is bigger. Then check for fp16 issues if batch size change increased pre-softmax scores.

> **Saying it out loud.** Whether the learning rate got rescaled with it. That's the first thing, every time — bigger batch means a less noisy gradient means you should be taking bigger steps, roughly linearly for SGD and roughly square root for Adam. Second is warmup length, because a bigger batch means fewer, larger steps and you need to ramp longer to get through the fragile phase. Third, if the change also increased sequence packing or activation sizes, check for fp16 overflow in the forward pass — that shows up as instability that has nothing to do with the optimizer.

**44. How do you debug "training is fine but slow"?**
First confirm "slow" by measuring per-step time and per-token loss decrease. If per-step time is the bottleneck, it's a systems issue (data loading, GPU utilization). If per-step loss decrease is too slow, your $\eta$ is probably too low (check update-to-weight ratio) or your batch size is below the noise scale (using too many noisy gradients per parameter update).

> **Saying it out loud.** First I'd figure out which kind of slow it is, because they have completely different fixes. Measure wall-clock per step and loss reduction per step separately: if the step time is bad, it's a systems problem — data loading, GPU utilization, a synchronization stall — and no optimizer change will help. If steps are fast but each one accomplishes little, that's an optimization problem, and I'd check the update-to-weight ratio for a too-small LR, or check whether the batch size is well below the noise scale so you're spending steps on pure noise. Confusing those two is the most common way people waste a week.

**45. How do you decide between Adam and SGD?**
SGD with momentum if (a) you can afford to tune LR carefully, (b) the problem is well-conditioned (CNNs on vision), (c) generalization is paramount. Adam(W) if (a) the problem is ill-conditioned (transformers, embeddings), (b) you need robustness to LR choice, (c) you have heterogeneous gradients across parameters. For LLM training the answer is always AdamW or a successor (Lion, Sophia, etc.).

> **Saying it out loud.** AdamW unless you have a specific reason not to. SGD with momentum is a genuine contender when the problem is well-conditioned — convnets on images — you can afford a careful LR sweep, and you care most about the last half point of validation accuracy. Adam is the answer when the gradients are wildly heterogeneous across parameters, which is exactly what transformers and embedding tables give you, and when you need robustness to getting the LR slightly wrong. For anything language-model-shaped there is no real debate: AdamW or one of its successors.

---

## G. Advanced and frontier topics

**46. What's Sophia, and why might it matter for LLM training?**
Sophia (Liu et al. 2023) approximates the diagonal of the Hessian using Hutchinson's estimator and uses it as a preconditioner instead of $\sqrt{\hat v}$. Empirically converges in fewer steps than Adam on language modeling — a real speedup in compute terms. Cost: extra estimator calls per step. Whether it dominates AdamW at scale is still being established.

> **Saying it out loud.** Sophia tries to be genuinely second-order without paying second-order prices. Instead of dividing by the root of the squared-gradient average like Adam, it estimates the diagonal of the Hessian directly with a Hutchinson estimator and uses that as the preconditioner — so it's scaling by actual curvature rather than by gradient magnitude, which is a proxy. The claimed payoff is reaching the same loss in meaningfully fewer steps on language modeling. The cost is extra Hessian-vector products, done every few steps to amortize, and whether it holds up at frontier scale is still genuinely unresolved.

**47. What is the relationship between learning rate and batch size in the "constant noise scale" view?**
The implicit regularization strength scales as $\eta/B$. To keep the same regularization when changing batch size, change $\eta$ proportionally — equivalently, increase $B$ and $\eta$ together to speed up training without changing implicit regularization. "Don't decay the LR, increase the batch size" (Smith et al. 2018) exploits this: instead of decaying $\eta$, increase $B$ over time, getting equivalent dynamics with better hardware utilization.

> **Saying it out loud.** The strength of SGD's implicit regularization scales with $\eta/B$, so the learning rate and the batch size aren't independent knobs — they're one knob. That means decaying the LR by a factor of two and doubling the batch size produce nearly identical training dynamics. Smith's paper exploited exactly this: instead of decaying the LR at the end of training, increase the batch size on the same schedule, and you get equivalent results with much better hardware utilization since bigger batches are more efficient. The limit is that you run out of memory or hit the critical batch size, and at that point you have to decay the LR after all.

**48. What does "second-order Adam" or natural gradient buy you?**
Natural gradient uses the Fisher information matrix as a preconditioner, accounting for the geometry of the parameter space under the model's output distribution. It corresponds to "shortest distance in distribution space" rather than parameter space. K-FAC is a tractable approximation. For LLMs, full natural gradient is prohibitive; Adam's diagonal approximation is what we settle for. Sophia is one step closer to second-order at affordable cost.

> **Saying it out loud.** Natural gradient asks what step is small in *function* space rather than in parameter space. Two weight configurations can be numerically close while producing completely different output distributions, or far apart and behave identically, so plain gradient descent's notion of distance is somewhat arbitrary. Preconditioning by the Fisher information fixes that, and the payoff is that convergence stops depending on how you happened to parameterize the model. The problem is cost: the Fisher is parameter-count squared, so we approximate — K-FAC with block structure, Adam with a crude diagonal, Sophia with a Hessian diagonal. Adam is the cheapest possible version of this idea, which is worth knowing when someone asks why it works.

**49. What's the difference between LR for embedding tables and the rest of the model?**
Embedding tables are sparse (only the rows for sampled tokens get gradient signal per step), and their $\hat v$ estimates are heavily skewed toward frequent tokens. Naive Adam can over-update rare-token embeddings (because their $\hat v$ is small) and under-update common tokens. Common fixes: per-row state, larger $\varepsilon$ for embeddings, or specialized sparse Adam. Frontier-lab pretraining recipes often have a separate LR multiplier for embeddings.

> **Saying it out loud.** Embeddings are sparse, and that breaks Adam's assumptions. A given row only gets a gradient on steps where its token actually appeared, so for a rare token the second moment is tiny — and dividing by a tiny number means when it finally does appear, it gets a huge update. Meanwhile frequent tokens have large second moments and get damped. So naive Adam over-updates the rare rows and under-updates the common ones, which is exactly backwards. The standard fixes are a bigger epsilon for the embedding group, a separate LR multiplier, or a sparse Adam that only advances the moment estimates for rows that were touched.

**50. Walk me through the math of why Adam is sensitive to $\beta_2$ choice.**
With $\beta_2 = 0.999$, the effective horizon of the second-moment estimate is $1/(1 - \beta_2) = 1000$ steps. If you have a gradient outlier at step $t$, its contribution to $v_t$ decays as $\beta_2^k$ after $k$ steps — half-life ~700 steps. That means a single bad batch at step 10000 still influences the update at step 10700. This is why some recipes use $\beta_2 = 0.95$ for robustness or $\beta_2 = 0.99$ for very long pretraining. Mismatched $\beta_2$ is a real source of training instability after long training runs.

> **Saying it out loud.** Because $\beta_2$ sets how long Adam remembers a gradient outlier. At 0.999 the effective window is a thousand steps and the half-life of any single contribution is about 700, so a spike at step 10,000 is still distorting your step size at step 10,700 — and while $\hat v$ is inflated, that parameter's effective learning rate is suppressed the whole time. Set it too low and the estimate gets noisy and you lose the conditioning benefit; too high and you get these long memory artifacts. That's why long pretraining runs often use 0.95 for robustness, and mismatched $\beta_2$ is a real and under-diagnosed source of instability late in training.

---

## H. Quick-fire (say each answer in under 10 seconds)

**51.** *Default Adam betas?* $0.9, 0.999$.
**52.** *Default Adam epsilon?* $10^{-8}$.
**53.** *Default LLM clipping norm?* $1.0$.
**54.** *Typical pretraining peak LR for a 7B?* $\sim 3 \times 10^{-4}$.
**55.** *Typical fine-tuning LR for full SFT?* $\sim 2 \times 10^{-5}$.
**56.** *Typical DPO LR?* $\sim 5 \times 10^{-7}$.
**57.** *Warmup duration as fraction of training?* $0.5$–$2\%$.
**58.** *Cosine $\eta_{\min}$ as fraction of peak?* $\sim 10\%$.
**59.** *AdamW weight decay default?* $0.01$–$0.1$, often $0.1$ for LLMs.
**60.** *Update-to-weight ratio target?* $\sim 10^{-3}$ per layer.

---

## I. Self-grading

If you can't answer questions 1–10 cold, you don't know SGD. If you can't answer 11–20, you don't know modern optimization. If you can't answer 21–30, you can't pass an MLE round. If you can't answer 31–50, you'll struggle in frontier-lab applied scientist screens.

Aim for 40+ correct out of 50 before walking into an interview. Re-grill on the misses.
