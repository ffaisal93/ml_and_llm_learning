# Normalization — Interview Grill

> 40 questions on normalization in deep learning. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. What does normalization do mathematically?**
Standardize activations to (approximately) zero mean and unit variance, then apply a learned affine transform $\gamma \cdot x_{\text{norm}} + \beta$. The standardization is the "normalization"; the affine transform restores capacity.

> **Saying it out loud.** It's two steps that people usually blur together. First you standardize — subtract the mean and divide by the standard deviation, so the activations sit around zero with unit spread. Then you immediately give the model a learned scale and shift back, $\gamma$ and $\beta$, so it can undo that if it wants to. It sounds circular, but it isn't: the first step is what fixes the optimization, and the second step is what stops you from throwing away representational capacity. Standardize to help the optimizer, then re-parameterize so nothing is lost.

**2. Why do we normalize?**
To improve optimization. Best current understanding: normalization smooths the loss landscape (smaller gradient Lipschitz constant), enabling larger learning rates and faster training. Equivalent in spirit to per-feature preconditioning.

> **Saying it out loud.** Because it makes the loss surface easier to walk on. Without normalization, different directions in parameter space have wildly different curvature, so any learning rate big enough to make progress in the flat directions blows up in the steep ones. Normalization compresses that range, which means you can use a much larger learning rate and train much faster. The honest framing is that it's a preconditioner you get for almost free. The practical number is that BatchNorm let ResNets use learning rates around ten times larger than they could otherwise.

**3. What was the original justification for BN, and why is it wrong?**
Ioffe & Szegedy (2015) proposed BN to fix "internal covariate shift" — the changing distribution of layer inputs during training. Santurkar et al. (2018) empirically showed BN doesn't actually reduce covariate shift much. The current believed explanation is loss landscape smoothing.

> **Saying it out loud.** The original story was internal covariate shift — as earlier layers update, the distribution feeding later layers keeps moving, and normalizing pins it down. Great story, and it turned out to be basically wrong. Santurkar and colleagues in 2018 deliberately injected noise after BatchNorm to make the covariate shift worse, and training still improved, which kills the causal claim. What they found instead is that BN smooths the loss landscape — it bounds how fast gradients change, so larger steps stay safe. It's a nice reminder that a technique can be enormously useful while its stated mechanism is fiction.

**4. What's the role of $\gamma$ and $\beta$?**
Standardization sets every dimension to mean 0, variance 1. The affine transform $\gamma \cdot x_{\text{norm}} + \beta$ allows the model to recover any other distribution if useful. So normalization is "invertible" in principle; in practice, $\gamma$ and $\beta$ learn values that work better than the original.

> **Saying it out loud.** They're the escape hatch. Standardization forces every feature to mean zero and variance one, which is a strong constraint the model never asked for — maybe a sigmoid activation actually wants inputs centered somewhere else. So you hand back a learned scale and shift, and the model can recover any mean and variance it likes. Crucially the optimization benefit survives, because the gradient still flows through the normalization step. In practice $\gamma$ tends to settle well below one in deep transformers, which is the model quietly telling you it wants a smaller residual contribution per layer.

**5. What's $\varepsilon$ for in normalization?**
Numerical stability. Prevents division by zero when variance is near zero. Standard values: $10^{-5}$ to $10^{-6}$. If $\varepsilon$ is too small, can cause NaNs in fp16; if too large, weakens normalization. Some recipes use $10^{-3}$ for embeddings or low-precision training.

> **Saying it out loud.** It's the tiny number added under the square root so you never divide by zero. If a token's features happen to be nearly identical, the variance is nearly zero, and without $\varepsilon$ you'd get an enormous or infinite output. Typical values are $10^{-5}$ or $10^{-6}$. It's not a free knob — in fp16 a too-small $\varepsilon$ underflows and you get NaNs, and a too-large one starts actually weakening the normalization. That's why low-precision recipes sometimes bump it to $10^{-3}$, and why a mysterious NaN a few thousand steps into a half-precision run is worth checking here.

---

## B. The four normalizers

**6. Walk me through BatchNorm.**
For a 4D activation $[B, C, H, W]$, normalize per-channel across batch and spatial dims:

$$
\mu_c = \mathrm{mean}(X[:, c, :, :]), \qquad \sigma_c^2 = \mathrm{var}(X[:, c, :, :])
$$

$$
\hat x = (x - \mu_c) / \sqrt{\sigma_c^2 + \varepsilon}, \qquad \text{output} = \gamma_c \cdot \hat x + \beta_c
$$

Per-channel $\gamma, \beta$. Maintains running averages of $\mu, \sigma^2$ for inference.

> **Saying it out loud.** BatchNorm normalizes each channel using statistics gathered across the whole batch. For an image tensor, you take one channel — say the edge detector — and compute its mean and variance over every image in the batch and every pixel position, then standardize it and apply that channel's learned scale and shift. The key thing is the statistics come from other examples, which means your prediction for one image depends on what else was in the batch. At inference you can't do that, so BN keeps a running average from training and uses that instead — and that train-test gap is the source of most BatchNorm bugs.

**7. Walk me through LayerNorm.**
For a 3D activation $[B, N, d]$, normalize per-sample per-token across features:

$$
\mu = \mathrm{mean}(X[b, n, :]), \qquad \sigma^2 = \mathrm{var}(X[b, n, :])
$$

$$
\hat x = (x - \mu) / \sqrt{\sigma^2 + \varepsilon}, \qquad \text{output} = \gamma \cdot \hat x + \beta
$$

Single $\gamma, \beta$ (per-feature). No running averages needed.

> **Saying it out loud.** LayerNorm computes the mean and variance across the feature dimension of a single token, and nothing else. No other tokens, no other examples in the batch. So each token's vector gets standardized on its own, then scaled and shifted by learned parameters shared across positions. That independence is the whole reason it fits transformers: it behaves identically at batch size one and batch size a thousand, it doesn't care that your sequences are different lengths, and it needs no running averages, so training and inference are literally the same computation.

**8. What's RMSNorm?**
LayerNorm without mean subtraction:

$$
\mathrm{RMSNorm}(x) = \gamma \cdot x / \mathrm{RMS}(x), \qquad \mathrm{RMS}(x) = \sqrt{\mathrm{mean}(x^2) + \varepsilon}
$$

Just unit-variance normalization. ~30% cheaper. Empirically as good as LN for transformers. Used in LLaMA, Gemma, Mistral.

> **Saying it out loud.** RMSNorm is LayerNorm with the mean subtraction deleted. You just divide by the root mean square of the vector and apply a learned scale — no centering, no $\beta$ shift. The bet is that the rescaling was doing all the work and the centering was decorative, and empirically for deep transformers that bet holds up. You save one pass over the feature dimension, which is roughly 10 to 30 percent of the normalization cost depending on your kernel, and normalization runs several times per layer per token, so it adds up. LLaMA, Mistral and Gemma all use it, which is about as strong a vote as you get.

**9. What's GroupNorm?**
Compromise between LN and BN: normalize per-sample across groups of channels. For 4D $[B, C, H, W]$, split channels into $G$ groups, normalize each group separately:

$$
\mu = \mathrm{mean}(X[b, \text{group}, :, :]), \qquad \sigma^2 = \mathrm{var}(\cdots)
$$

Used in vision when batch size is small. Some diffusion models use it.

> **Saying it out loud.** GroupNorm sits between the other two. You split the channels into groups — 32 is the usual default — and normalize within each group for each sample separately. Because it never touches the batch dimension it works fine at batch size one, but unlike LayerNorm it doesn't force all channels into one shared statistic, which matters for convolutional features where different channels genuinely have different scales. It's the standard answer for vision tasks like detection and segmentation where images are huge and batch size per GPU is two or four. Diffusion U-Nets use it for the same reason.

**10. BN vs LN: tabular comparison.**

| Aspect | BN | LN |
|---|---|---|
| Norm dim | batch + spatial per channel | features per token per sample |
| Per-batch statistics | Yes | No |
| Train/test discrepancy | Yes (running averages) | No |
| Works at batch=1 | No (variance = 0) | Yes |
| Works for variable seq lengths | No (padding issues) | Yes |
| Used in | CNNs, vision | Transformers |

> **Saying it out loud.** If I had to compress the table into a sentence: BatchNorm normalizes across examples, LayerNorm normalizes within one example, and everything else follows from that. Normalizing across examples means you need a decent batch, you get different behavior at train and test time, and padding poisons your statistics. Normalizing within one example means none of that applies — batch size one is fine, variable sequence lengths are fine, and eval mode is identical to train mode. That's the whole reason vision went one way and transformers went the other.

---

## C. Why transformers don't use BN

**11. Why don't transformers use BatchNorm?**
Three reasons. (a) Variable sequence lengths cause padding, which corrupts BN statistics. (b) BN's running averages can drift due to distribution shift between training and inference. (c) BN needs reasonable batch size for stable statistics; transformer training often has many short sequences with heterogeneous structure.

> **Saying it out loud.** Three reasons stack up. Padding is the ugliest one: sequences have different lengths, so a batch is full of pad tokens, and BatchNorm happily averages those zeros into its statistics and corrupts them. Then there's the train-test gap — the running averages you accumulate during training are a poor match for inference, especially when your inference distribution differs from your training mix. And BatchNorm needs a decent per-device batch, which large-model training rarely has, since you're usually at one or two long sequences per GPU with everything else eaten by activations. LayerNorm has none of these problems, so there's nothing to trade off.

**12. Why does LayerNorm work where BN fails?**
LN is purely per-token, per-sample. No batch dimension. Identical at train and test. Works at batch=1. Handles variable sequence lengths trivially. None of BN's pathologies apply.

> **Saying it out loud.** Because it never looks at any other token or any other example. Every one of BatchNorm's problems traces back to depending on batch composition, and LayerNorm simply doesn't. Same computation at batch one and batch 1024, same computation in training and in eval, no running statistics to drift, and pad tokens can't contaminate anything because each token is normalized alone. It's a case where the simpler operation is also strictly the more robust one.

**13. Could BN be made to work for transformers in principle?**
With workarounds (mask out padding, large enough effective batch size, careful train-test handling) yes — but the workarounds are ugly and provide no benefit over LN. So in practice nobody does it.

> **Saying it out loud.** In principle yes — you'd mask padding out of the statistics, make sure the effective batch is big enough, maybe sync across GPUs, and handle the train-test switch carefully. People have tried it, PowerNorm being the best-known attempt. The problem is you're taking on all that machinery to end up roughly where LayerNorm already is for free, and every one of those workarounds is a place for a bug to hide. So the answer is that it's possible and nobody does it, which is usually the right answer when a workaround has no upside.

**14. SyncBN — what and why?**
For multi-GPU training with small per-GPU batch sizes, BN's per-GPU statistics are noisy. SyncBN aggregates statistics across GPUs (extra communication cost) for stable normalization. Common in vision; not relevant for transformers.

> **Saying it out loud.** When you're training on many GPUs, each one only sees its own slice of the batch, so if that slice is two or four images the BatchNorm statistics are extremely noisy. SyncBN does an all-reduce so every device computes statistics over the global batch instead. It genuinely helps for detection and segmentation, where small per-device batches are unavoidable. The cost is a synchronization point in every normalization layer, which is real communication overhead — typically 10 to 20 percent slower — and it's irrelevant for transformers, since they use LayerNorm and never had the problem.

---

## D. Pre-LN vs post-LN

**15. What's post-LN?**
The original transformer (Vaswani 2017): $x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. Norm comes after the residual addition.

> **Saying it out loud.** Post-LN is the original 2017 transformer: you run the sublayer, add the residual, and then normalize the sum. So the normalization sits on the main path — everything flowing forward gets renormalized at every block. That sounds tidy, and it's exactly what makes deep post-LN models hard to train, because there's no clean unnormalized path from input to output for gradients to travel down.

**16. What's pre-LN?**
Modern: $x \leftarrow x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$. Norm comes before the sublayer; the residual is unnormed.

> **Saying it out loud.** Pre-LN moves the normalization inside the branch: you normalize, run the sublayer, and add the result to the untouched residual stream. The residual path is now a pure identity from the first layer to the last, which is what gradients want. Every large model since roughly GPT-2 does this. The one real cost is that the residual stream grows in magnitude with depth, since you keep adding to it without ever renormalizing, which is why pre-LN architectures put one final normalization before the output head.

**17. Why does pre-LN train more stably?**
Two reasons. (a) Gradient flow: in pre-LN, the residual path is identity through LayerNorm, so gradients flow unchanged. In post-LN, every block applies LN to the gradient on the way back, which can attenuate signal. (b) Residual stream magnitude: pre-LN preserves the unnormed stream; post-LN renormalizes after every block, amplifying any perturbations.

> **Saying it out loud.** It comes down to the gradient's path home. In pre-LN the residual stream is untouched identity, so a gradient from the loss can reach layer one without passing through a single normalization — it arrives at full strength. In post-LN every block's LayerNorm sits on that path and scales the gradient on the way back, and those factors compound multiplicatively across depth, so a 48-layer model can attenuate or amplify catastrophically. The practical evidence is stark: post-LN transformers past about 12 layers won't train without careful warmup, while pre-LN models train from step one with no warmup at all.

**18. What did post-LN need for stability?**
Elaborate warmup schedules. The original paper uses warmup over thousands of steps with a specific formula $\text{lr} \propto d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$. Without it, post-LN diverges at modern scales.

> **Saying it out loud.** Warmup, and a lot of it. The original paper ramps the learning rate up over 4,000 steps with that inverse-square-root schedule, and that's not a nicety — remove it and a post-LN transformer diverges in the first few hundred steps. What's happening is that early gradients are large and badly scaled, and without the ramp the first few updates destroy the initialization. Pre-LN mostly removes the need, which is a nice illustration that a lot of deep learning folklore is really compensation for an architectural choice.

**19. Are there any modern uses of post-LN?**
Some encoder-only models still use it (BERT). For decoder-only LLMs at any meaningful scale, pre-LN is universal.

> **Saying it out loud.** BERT and its descendants are post-LN, and they're still everywhere in production for embeddings and classification — but those are 12 or 24 layers, which is shallow enough for the instability not to bite. For any decoder-only LLM at real scale, pre-LN is universal, no exceptions worth mentioning. There's also a small line of work like DeepNet arguing post-LN gives slightly better final quality if you can tame it with careful residual scaling, but nobody is betting a frontier run on that.

**20. What's a sandwich norm?**
LayerNorm both before AND after each sublayer. Used in some recent architectures (GPT-J variants). More compute, occasionally helps stability. Sometimes called "double norm."

> **Saying it out loud.** Sandwich norm means normalizing both before the sublayer and again on its output before the residual add. The motivation is that pre-LN lets the residual stream grow unchecked with depth, and the second norm caps how much any single block can inject. It costs you an extra normalization per sublayer, so a few percent of throughput, and the payoff is stability at very large scale — Gemma 2 uses it. It's the kind of thing you reach for when you're seeing loss spikes at 70B and you'd rather pay 3 percent throughput than restart from a checkpoint.

**21. What's QK-norm?**
Apply LayerNorm separately to $Q$ and $K$ before computing attention scores. Stabilizes attention at very large model scales. Used in some recent papers as a stability fix.

> **Saying it out loud.** QK-norm normalizes the query and key vectors before you take their dot product. The problem it fixes is that attention logits are a dot product of two learned vectors, so nothing bounds their magnitude — and if they drift large, the softmax saturates, gradients vanish through the attention weights, and you get a loss spike. Normalizing $Q$ and $K$ first caps the logits by construction. It costs almost nothing and it's become fairly standard in recent large models, precisely because attention-logit blowup is one of the most common causes of a training run dying at scale.

---

## E. RMSNorm specifics

**22. Why does RMSNorm work without mean subtraction?**
Empirically: the variance normalization is doing most of the work; removing the mean is redundant given that $\gamma, \beta$ can recenter if needed. For deep transformers with pre-LN structure, mean-subtraction is largely redundant.

> **Saying it out loud.** Mostly because the rescaling was doing the real work. Two things make the centering redundant in a deep transformer: the learned parameters can absorb any offset the model actually wants, and in high dimensions the mean of a feature vector is a single number out of thousands, so removing it barely changes the geometry. Empirically the ablation is a wash — same loss curves, same downstream numbers. Where it's not a wash is shallow networks or architectures with asymmetric activations, where the centering does earn its keep.

**23. What does RMSNorm save?**
~30% compute over LayerNorm (one fewer reduction across the feature dimension). For LLM pretraining, where LayerNorm is computed many times per token per layer, this adds up.

> **Saying it out loud.** You skip one full reduction over the feature dimension, so roughly 10 to 30 percent of the normalization cost, and normalization runs two or three times per layer per token. Sounds trivial until you notice that normalization is memory-bandwidth-bound, not compute-bound — you're reading and writing the whole activation tensor, and the arithmetic is nothing. So the saving is a real fraction of wall-clock, not just a FLOP count. Over a multi-week pretraining run at billion-parameter scale, single-digit percentages of throughput are worth real money.

**24. Are there cases where LN beats RMSNorm?**
Yes — in some shallow networks or non-transformer architectures, mean-subtraction matters more. But for modern deep transformers, RMSNorm is fine.

> **Saying it out loud.** Yes, though they're mostly outside the transformer world. Shallow networks, recurrent models, and architectures where feature means genuinely drift benefit from the centering, because there the mean carries information about scale drift rather than being noise. There's also some evidence that RMSNorm is more sensitive to activation outliers, since without centering a few enormous channels dominate the root-mean-square. For modern deep transformers the difference is within noise, which is why the field took the cheaper one.

**25. Why is variance computed in fp32 even with bf16 activations?**
Variance computation in low precision can lose significant figures (subtraction of similar numbers, summation of squares). Computing in fp32 and casting back is standard. Same for the affine transform's $\gamma$ multiplication.

> **Saying it out loud.** Because variance is a subtraction of two similar large numbers, and that's the classic way to lose all your precision. bf16 has about three decimal digits of mantissa, so summing thousands of squared values in bf16 accumulates error fast, and if the mean and the mean-of-squares are close you can end up with a negative variance — then you take a square root and get NaN. So you upcast, reduce in fp32, and cast back. It's cheap because the tensor is already in registers, and it's the difference between a run that survives and a run that dies at hour twenty for no visible reason.

---

## F. Affine transform and inference

**26. Doesn't $\gamma$ and $\beta$ just undo the normalization?**
The optimizer can learn $\gamma = \sigma_{\text{original}}, \beta = \mu_{\text{original}}$ to recover the un-normalized activation in principle. But the optimization landscape with normalization is fundamentally different — gradients flow differently, learning rate sensitivity changes. The point isn't that activations are different; the optimization dynamics are improved.

> **Saying it out loud.** They could in principle, and that's exactly the point of the question. The answer is that even if the forward pass ended up identical, the backward pass wouldn't be — normalization changes how gradients scale, because the division by the standard deviation is part of the computation graph and gets differentiated too. So you've reparameterized the problem: the same function is now reachable through a much better-conditioned landscape. It's not about what values the activations take, it's about how the optimizer moves through the space to find them.

**27. What if $\gamma$ is initialized at 0?**
A common trick: zero-init $\gamma$ in the LN before residual addition. Effect: at init, the sublayer contributes nothing. The model starts as a pure residual identity stream and the layers gradually become non-trivial as $\gamma$ grows. Helps stability at very large scales.

> **Saying it out loud.** Then at initialization every block outputs exactly zero and the whole network is a pure identity function from input to output. That sounds useless, but it's a deliberate and very effective trick: the model starts in a state that's guaranteed stable, and each layer switches itself on gradually as $\gamma$ grows away from zero. It means you're never fighting a badly-conditioned deep stack in the first thousand steps, which is when most large runs die. Variants of this are everywhere — ResNet's zero-init on the last BN of each block is the same idea, and it's worth a point or so of ImageNet accuracy for free.

**28. Layer at inference: what's different?**
LN: nothing. Identical to training. RMSNorm: same. **BN: significantly different.** BN at inference uses running averages of $\mu$ and $\sigma$ accumulated during training, not batch statistics. PyTorch's `model.eval()` flag controls this. Forgetting to set it is a common production bug.

> **Saying it out loud.** For LayerNorm and RMSNorm, absolutely nothing — same computation, train or test. For BatchNorm it's a completely different computation: at inference it uses running averages accumulated during training instead of the current batch's statistics. That's why forgetting `model.eval()` is such a classic production bug. The symptom is distinctive and worth memorizing — your model scores well in validation and then behaves erratically in production, with predictions that change depending on what else happened to be in the request batch.

**29. Why is BN's train-test discrepancy a problem?**
Subtle bugs: (a) inference batch composition differs from training, so running averages may be inaccurate; (b) single-sequence inference uses normalization that depends on running averages which may drift; (c) interaction with dropout (different active state at train vs test) compounds.

> **Saying it out loud.** Because the running averages are a summary of your training distribution, and inference rarely matches it. Serve a model on data that has drifted, or with a batch composition unlike training, and the normalization is now using statistics that describe a different world. It also interacts badly with dropout: during training the activations include dropout's zeros, so the running averages describe the with-dropout distribution, while at inference there's no dropout and the activations are systematically different. None of these crash — they just quietly cost you accuracy, which is the worst kind of bug.

---

## G. Edge cases and gotchas

**30. BN with batch size 1?**
Variance is 0 (single sample), normalization divides by $\varepsilon$, output is dominated by $\gamma \cdot 0 + \beta = \beta$. Useless. LN works fine at batch=1.

> **Saying it out loud.** It breaks completely. With one sample the variance across the batch is exactly zero, so you divide by essentially $\varepsilon$ and the output collapses to just $\beta$ — the input is gone, the layer outputs a constant. So the network is dead. This is precisely why memory-hungry vision tasks like detection, where you can only fit two images per GPU, moved to GroupNorm, and it's one more reason transformers never wanted BatchNorm.

**31. LN with sequence length 1?**
Fine. LN is per-token. Sequence length doesn't matter.

> **Saying it out loud.** Totally fine, because LayerNorm never looks along the sequence axis. It normalizes across the feature dimension of one token, so whether that token has neighbors is irrelevant. This is exactly the property that makes autoregressive decoding work — during generation you're processing a single token at a time, and the normalization has to produce the same result it would have inside a full sequence. BatchNorm cannot make that promise, and LayerNorm gets it for free.

**32. How does dropout interact with normalization?**
Dropout zeros some activations during training, changing their distribution. BN computes statistics including the dropout effect, so its running averages reflect the "with dropout" distribution. At inference (no dropout), the active activations are larger (no zeros), causing slight mismatch with running averages. Modern recipes either: (a) apply dropout AFTER normalization to avoid this, or (b) accept the small discrepancy.

> **Saying it out loud.** The tricky combination is dropout before BatchNorm. Dropout zeros a random subset of activations and scales the rest up, so the distribution BatchNorm sees during training has a different variance from what it sees at inference when dropout is off — and the running averages bake in the wrong one. Li and colleagues called this the variance shift, and it's the documented reason some networks got worse when both were used together. The fixes are to put dropout after the normalization, or just to skip dropout entirely, which is what most large models do since they rely on data scale for regularization instead.

**33. What's "pre-LN with parallel sublayers"?**
GPT-J style: compute attention and FFN in **parallel** from the same input rather than sequentially:

$$
x' = x + \mathrm{Attention}(\mathrm{LN}(x)) + \mathrm{FFN}(\mathrm{LN}(x))
$$

Saves one dependency in the dataflow graph (slight speedup). Empirically comparable to standard pre-LN.

> **Saying it out loud.** Normally attention runs, then the feed-forward network runs on attention's output — strictly sequential. The parallel variant, from GPT-J, runs both from the same normalized input and adds both results to the residual. Since neither waits for the other, you can fuse their matrix multiplications and shorten the critical path, which is worth something like 15 percent throughput at scale. Quality is roughly a wash at large sizes, slightly worse at small ones, so it's a throughput trade — PaLM and Falcon took it, most others didn't.

**34. WeightNorm?**
Salimans & Kingma 2016. Reparameterize each weight as $W = (g / \|v\|) \cdot v$. Decouples direction from magnitude. Less popular today but historically important. Mostly superseded by LN/BN.

> **Saying it out loud.** WeightNorm normalizes the weights instead of the activations: you write each weight vector as a learned magnitude times a unit-norm direction, so the optimizer can adjust how big a filter is separately from what it detects. That decoupling helps conditioning, and it's data-independent, so there's no batch dependence and no train-test gap at all. It was reasonably popular around 2016 and then faded, because in practice activation normalization gave more benefit for the same effort. Worth knowing mainly as a data point that normalization is about conditioning, not about activations specifically.

**35. Why is normalization sensitive to extreme outliers?**
Activation outliers (a few channels with 100x typical magnitude) disturb the variance estimate. RMSNorm is more affected than LN (no mean subtraction to attenuate). SmoothQuant deals with this for quantization; for fp32 training, it's usually fine but can require larger $\varepsilon$ or gradient clipping.

> **Saying it out loud.** Because the scale factor is a mean over the feature dimension, and one channel that's 100 times larger than the rest dominates it — so everything else gets squashed toward zero. Large transformers reliably develop these outlier channels, a handful of dimensions with enormous magnitude, and RMSNorm feels it more than LayerNorm since there's no centering to take the edge off. It's mostly survivable in fp32 training, and it's a serious problem for quantization, because those outliers force you to use a dynamic range that wastes almost all your bits. That's exactly what SmoothQuant exists to fix, by migrating the outlier scale from activations into the weights.

---

## H. Quick fire

**36.** *BN paper?* Ioffe & Szegedy 2015.
**37.** *LN paper?* Ba, Kiros, Hinton 2016.
**38.** *RMSNorm paper?* Zhang & Sennrich 2019.
**39.** *Default $\varepsilon$ in PyTorch LN?* $10^{-5}$.
**40.** *Default $\gamma, \beta$ init?* $\gamma = 1, \beta = 0$.

---

## Self-grading

If you can't answer 1-10, you don't know normalization. If you can't answer 11-25, you can't pass an architecture round. If you can't answer 26-40, you'll struggle with deep follow-ups.

Aim for 30+/40 cold.
