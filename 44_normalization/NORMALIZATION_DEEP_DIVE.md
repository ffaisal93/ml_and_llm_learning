# Normalization: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Normalization is one of those topics that "everyone knows" — and where most candidates blow simple questions because they don't know the *why*. Why does BN help optimization? Why doesn't BN work for transformers? Why pre-LN over post-LN? What's the difference between LayerNorm and RMSNorm and which do modern LLMs use? This document is the answer.

---

## 1. Why normalize at all?

The honest answer is: **we don't fully know, but it works.** Three motivations are commonly given, in roughly increasing order of how seriously researchers take them:

> **Saying it out loud.** The honest answer is that normalization helps optimization and we're still arguing about exactly why. The story that holds up best is that it smooths the loss surface — without it, different directions in parameter space have wildly different curvature, so any step size big enough to move in the flat directions explodes in the steep ones. Normalization compresses that range so you can use a much larger learning rate. What I'd avoid saying is the original internal-covariate-shift story, because that one was tested and didn't survive. The number that makes the case is that BatchNorm let ResNets train at roughly ten times the learning rate they could otherwise.

### 1. Internal Covariate Shift (the original BN claim, now mostly debunked)

Ioffe & Szegedy (2015) proposed BN to fix "internal covariate shift" — the changing distribution of layer inputs during training. Stabilize this distribution and training becomes easier.

**Problem:** Santurkar et al. (2018, "How Does Batch Normalization Help Optimization?") showed empirically that BN doesn't actually reduce covariate shift much. So the original justification is wrong.

> **Saying it out loud.** Covariate shift was the original pitch: as earlier layers update, the distribution arriving at later layers keeps moving, so each layer is chasing a target that won't sit still. Normalize and you pin the target down. It's a compelling story and it's essentially wrong — Santurkar's group deliberately injected noise after the BatchNorm layer to make the shift worse, and training still improved. So the mechanism can't be what the paper claimed. Good thing to have ready, because interviewers ask it specifically to see whether you've read past the abstract.

### 2. Smoothing the loss landscape

Santurkar et al. proposed instead: BN makes the loss landscape smoother (smaller gradient Lipschitz constant). Fewer cliffs, easier optimization, larger learning rates feasible. This is the better-supported story.

> **Saying it out loud.** The replacement explanation is that normalization makes the loss surface less jagged — formally it lowers the Lipschitz constant of the gradient, which just means the gradient doesn't change direction as violently as you move. Practically that means the gradient you measured at your current point stays valid over a longer step, so you can take bigger steps safely. That's the whole benefit: the same landscape, but with the cliffs sanded down. It's the best-supported story we have, and it lines up with the observed fact that the main thing normalization buys you is a much larger usable learning rate.

### 3. Implicit conditioning / preconditioning

Normalization is essentially an implicit form of per-feature rescaling. It removes scale variation across features, which improves the conditioning of gradient descent — same effect as Adam's per-parameter rescaling, but applied to activations rather than parameters.

The honest interview answer: **"normalization helps optimization, with current best understanding being smoothing of the loss landscape and implicit preconditioning. The original 'covariate shift' story is no longer believed."**

> **Saying it out loud.** The third framing is that normalization is a preconditioner. If one feature ranges over thousands and another over hundredths, the loss surface is a long narrow valley and plain gradient descent zigzags down it. Rescaling each feature to comparable magnitude turns the valley into something closer to a bowl. That's exactly what Adam does for parameters with its per-parameter scaling — normalization does it for activations. Two different attacks on the same conditioning problem, which is why they stack rather than making each other redundant.

---

## 2. The four canonical normalizers

> **In plain language.** All four of these do the same two things — subtract a mean, divide by a spread, then apply a learned scale and shift. The only thing that changes between them is which slice of the tensor you compute those statistics over. The formulas below are the same formula four times with different subscripts.

Each computes mean and variance over a different set of dimensions, then standardizes.

### Batch Normalization (BN, Ioffe & Szegedy 2015)

For each feature channel, normalize across the **batch and spatial** dimensions.

For a 4D activation $X[B, C, H, W]$ (vision), for each channel $c$:

$$
\mu_c = \mathrm{mean}(X[:, c, :, :]), \qquad \sigma_c^2 = \mathrm{var}(X[:, c, :, :])
$$

$$
\hat X[:, c, :, :] = \frac{X[:, c, :, :] - \mu_c}{\sqrt{\sigma_c^2 + \varepsilon}}
$$

$$
\text{output}[:, c, :, :] = \gamma_c \cdot \hat X[:, c, :, :] + \beta_c
$$

The learned parameters $\gamma, \beta$ per channel (affine transform) restore representational capacity that pure standardization removes.

> **Saying it out loud.** BatchNorm normalizes one channel at a time using statistics from the entire batch. Take the edge-detector channel, look at its values across every image in the batch and every pixel location, get the mean and variance, standardize, then apply that channel's learned scale and shift. The part that causes all the trouble is that your output for one image depends on which other images were in the batch. At inference there is no batch, so it falls back on running averages collected during training — and that train-test asymmetry is the source of essentially every BatchNorm bug you'll ever hit.

### Layer Normalization (LN, Ba et al. 2016)

For each sample, normalize across the **feature** dimension.

For a 3D activation $X[B, N, d]$ (sequence), for each $(b, n)$:

$$
\mu = \mathrm{mean}(X[b, n, :]), \qquad \sigma^2 = \mathrm{var}(X[b, n, :])
$$

$$
\hat X[b, n, :] = \frac{X[b, n, :] - \mu}{\sqrt{\sigma^2 + \varepsilon}}
$$

$$
\text{output}[b, n, :] = \gamma \cdot \hat X[b, n, :] + \beta
$$

Per-token, per-sample normalization. **No batch dimension**. Crucial for transformers.

> **Saying it out loud.** LayerNorm computes its statistics across the feature dimension of a single token and nothing else — no other tokens, no other examples. So the transformation of any given token is completely independent of what else is in the batch. That's the property that makes it right for transformers: identical behavior at batch size one and batch size a thousand, no problem with variable sequence lengths, no running averages, and the exact same computation at training and inference time. It's a strictly more robust operation than BatchNorm, and it happens to be simpler too.

### RMSNorm (Zhang & Sennrich 2019)

LayerNorm without mean subtraction:

$$
\mathrm{RMSNorm}(x) = \gamma \cdot \frac{x}{\mathrm{RMS}(x)}, \qquad \mathrm{RMS}(x) = \sqrt{\mathrm{mean}(x^2) + \varepsilon}
$$

Just unit-variance normalization. ~30% cheaper (one fewer reduction). Empirically as good as LayerNorm for transformers. **Used in LLaMA family, Gemma, Mistral, etc.**

> **Saying it out loud.** RMSNorm is LayerNorm with the centering deleted — divide by the root mean square, apply a learned scale, done. The bet is that the rescaling was doing all the work and the mean subtraction was decoration, and for deep transformers that bet has held. You save one reduction over the feature dimension, worth something like 10 to 30 percent of the normalization cost, and since normalization is memory-bandwidth-bound rather than compute-bound, that shows up in wall-clock. LLaMA, Mistral and Gemma all use it, which is a strong practical endorsement.

### GroupNorm (Wu & He 2018)

Compromise between LN and BN: normalize across groups of channels. For each $(b, \text{group } g)$:

$$
\mu_g = \mathrm{mean}(X[b, \text{channels in } g, :, :]), \qquad \sigma_g^2 = \mathrm{var}(\cdots)
$$

Used in vision when batch size is small (BN fails) but per-sample LN is too noisy. Some diffusion models use GroupNorm.

> **Saying it out loud.** GroupNorm splits the channels into groups — usually 32 — and normalizes within each group per sample. It's the middle ground: no batch dependence, so it works at batch size one, but it doesn't force every channel into one shared statistic the way LayerNorm would, which matters when different convolutional channels genuinely operate at different scales. That's why it's the default for detection and segmentation, where images are big and you can only fit two or four per GPU. Diffusion U-Nets use it for exactly the same reason.

### Visualization

|              | BN                  | LN                          | RMSNorm                     | GroupNorm           |
|---           |---                  |---                          |---                          |---                  |
| Norm dim     | batch + spatial     | features (per token, per sample) | features (per token, per sample) | feature groups |
| Per-batch    | yes                 | no                          | no                          | no                  |
| Per-token    | no                  | yes                         | yes                         | no                  |

---

## 3. Why BN fails for transformers (and sequences in general)

### Variable sequence lengths

Real batches have padding. BN's per-channel statistics are corrupted by zero-padding tokens. Workarounds (mask out padding) are messy.

> **Saying it out loud.** This is the least glamorous reason BatchNorm fails on text and maybe the most decisive. Sequences have different lengths, so batches are full of padding, and BatchNorm cheerfully folds those pad positions into its per-channel mean and variance. Now your statistics depend on how much padding happened to be in the batch, which depends on how you bucketed your data. You can mask it out, but you're threading a mask through every normalization layer to fix a problem LayerNorm never has.

### Train-test discrepancy

BN uses batch statistics during training and running averages at inference. Subtle bugs:

- Single-sequence inference uses a different normalization than training.
- Stability depends on the running averages being good estimates of the true mean/var.
- Distribution shift at inference time (different batch composition) leaks into the running estimate.

> **Saying it out loud.** The core issue is that BatchNorm computes one thing during training and a different thing at inference. During training it uses the live batch statistics; at inference it uses running averages frozen from training. So the function you validated isn't literally the function you deployed. The classic production incident is forgetting `model.eval()` — the model then normalizes using whatever requests happened to be batched together, and predictions become dependent on unrelated traffic. It doesn't crash, it just gets quietly worse, which is why it survives to production.

### Statistics are unreliable for small batches

BN needs batch size at least ~16–32 for stable statistics. LLM training often uses many short sequences per batch, but the relevant statistics are per-position, which can have small effective sample size.

> **Saying it out loud.** BatchNorm is estimating a mean and variance from a sample, so a small batch means a noisy estimate, and you want something like 16 to 32 examples before it settles down. Large-model training basically never has that per device — you're at one or two long sequences per GPU because activations eat the memory. So you'd be normalizing with statistics estimated from a couple of samples, which adds noise instead of removing it. This is the same problem that pushed high-resolution vision tasks toward GroupNorm.

### LayerNorm has no such issues

LN is purely per-sample. No batch dependency. Identical at train and test. Works at batch size 1. **This is the dominant reason transformers use LN, not BN.**

> **Saying it out loud.** Every one of BatchNorm's failure modes traces back to depending on other examples, and LayerNorm simply doesn't. No batch dependence means no padding contamination, no small-batch noise, no running averages, no train-test gap, no eval-mode bug. It's the same computation whether you're training on 1,024 sequences or decoding a single token. That robustness, not any quality advantage, is the real reason transformers use it.

### Side note: SyncBN

For multi-GPU vision training, sync-BN computes statistics across all GPUs. Necessary for small per-GPU batches. Adds communication cost. Not relevant for transformers.

> **Saying it out loud.** SyncBN is what you do when per-GPU batches are too small for stable statistics: all-reduce the mean and variance across devices so BatchNorm sees the global batch. It genuinely helps detection and segmentation. The cost is a synchronization point inside every normalization layer, typically 10 to 20 percent throughput, and it doesn't help transformers at all since they use LayerNorm and never had the problem.

---

## 4. The affine transform: gamma and beta

Standard normalization sets every dimension to mean 0, variance 1. This **removes representational capacity**: any pattern that needed a non-standard scale or offset is gone.

The affine transform $\gamma \cdot \hat x + \beta$ restores this:

- $\gamma$ scales the output back up if needed.
- $\beta$ shifts it.

The model can learn $\gamma = \sigma_{\text{original}}, \beta = \mu_{\text{original}}$ to recover the un-normalized activation. So the normalization is **invertible** in principle; in practice the learned $\gamma, \beta$ settle on values that work better than the un-normalized activation.

**Interview gotcha.** "Doesn't the affine transform undo the normalization?" Subtle: the optimizer can choose any $\gamma, \beta$, and the optimization landscape with the normalization-then-affine is fundamentally different from the un-normalized landscape. The key is that the *optimization dynamics* are improved, not that the activations are necessarily different.

> **Saying it out loud.** Standardizing forces every dimension to mean zero and variance one, which is a constraint the model never asked for — maybe some feature genuinely wants a large scale or a nonzero offset. So you hand back a learned scale and shift and the model can recover whatever distribution it likes. The obvious objection is that this undoes the normalization, and the answer is that it doesn't undo the part that mattered: the division by the standard deviation is still in the computation graph, so it still reshapes the gradients. You've changed the coordinate system the optimizer walks through, not the set of functions it can reach. In practice $\gamma$ tends to settle well below one in deep transformers, which is the model asking for a smaller contribution per layer.

---

## 5. Pre-LN vs post-LN

> **In plain language.** This is about where you put the normalization relative to the residual connection — inside the branch or on the main path. Two equations that differ by the position of one function call, and that difference decides whether a deep transformer trains at all.

The defining architectural choice for transformers.

### Post-LN (original, 2017)

$$
x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
$$

Norm comes after the residual addition.

> **Saying it out loud.** Post-LN is the 2017 original: run the sublayer, add the residual, then normalize the sum. The normalization sits directly on the main path, so everything flowing forward gets renormalized at every single block. It looks tidier, and it's exactly what makes deep post-LN models hard to train, because there's no unbroken path from output back to input for the gradient to travel.

### Pre-LN (modern)

$$
x \leftarrow x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))
$$

Norm comes before the sublayer; the residual is the unnormed input.

> **Saying it out loud.** Pre-LN moves the normalization inside the branch — you normalize a copy, run the sublayer on it, and add the result to the untouched residual stream. Now the residual path is pure identity from layer one to the output, which is exactly what gradients want. Everything since roughly GPT-2 does this. The one real cost is that the stream keeps growing in magnitude with depth since nothing renormalizes it, which is why pre-LN models put a final normalization before the output head.

### Why pre-LN trains more stably

Two arguments. First, **gradient flow**: in pre-LN, the residual stream is identity through the LayerNorm gates — gradients flow unchanged through the residual path. In post-LN, every block applies LayerNorm to the gradient on the way back, which can attenuate or amplify signal.

Second, **residual stream magnitude**: in post-LN, the stream is renormalized after every block, so any perturbation to the layer output is rescaled and propagated. In pre-LN, the unnormed stream preserves its magnitude; the sublayer reads a normed input but writes to the unnormed stream, decoupling read and write magnitudes.

Empirical evidence: post-LN transformers need elaborate warmup schedules and are unstable at modern scales. Pre-LN transformers are stable without elaborate warmup. Almost every modern LLM uses pre-LN or RMSNorm.

> **Saying it out loud.** It's about the gradient's route home. In pre-LN the residual stream is untouched identity, so a gradient can travel from the loss to layer one without passing through a single normalization — it arrives at full strength. In post-LN there's a LayerNorm on that path in every block, each scaling the gradient a bit, and those factors multiply across depth, so 48 layers can attenuate catastrophically. The empirical evidence is blunt: post-LN past about 12 layers won't train without a carefully tuned warmup schedule, while pre-LN trains from step one with essentially none.

### Sandwich norms / parallel norms

Some recent architectures (e.g., GPT-J, Gemma 2) use modifications:

- **Sandwich norm:** norm before AND after each sublayer. More compute, occasionally helps stability.
- **Parallel attention/FFN:** compute attention and FFN in parallel from the same input rather than sequentially. Saves a dependency.
- **QK-norm:** normalize Q and K separately before computing attention scores. Helps stability of very large models.

These are all variations on the pre-LN theme.

> **Saying it out loud.** These are all small patches on the pre-LN theme, each buying stability or throughput at very large scale. Sandwich norm adds a second normalization on the sublayer's output, capping how much any block can inject into the residual stream — Gemma 2 uses it, and it costs a few percent throughput. Parallel attention and FFN run both branches from the same input so you can fuse their matmuls, worth around 15 percent speed with roughly neutral quality. QK-norm is the one I'd single out: it normalizes queries and keys before the dot product, which bounds the attention logits and prevents softmax saturation — one of the most common ways a big training run dies.

---

## 6. RMSNorm: why subtract the mean isn't necessary

LayerNorm: $(x - \mu) / \sigma$. Two reductions.
RMSNorm: $x / \mathrm{RMS}(x)$. One reduction.

> **Saying it out loud.** The whole difference is one reduction. LayerNorm passes over the feature dimension twice, once for the mean and once for the variance around it; RMSNorm passes once, computing the root mean square directly. Since normalization is bound by memory bandwidth rather than arithmetic, cutting a pass is a real saving in wall-clock, not just in FLOP counts. And in deep transformers the quality difference is within noise, so the field took the cheaper one.

### The argument

Zhang & Sennrich (2019) argued that the mean-subtraction in LayerNorm doesn't add much: the variance normalization is doing most of the work, and re-centering is redundant given the affine transform $\gamma, \beta$ afterward (which can recenter if needed).

Empirically: RMSNorm matches LayerNorm performance on transformers, and it's faster. So modern LLMs prefer it.

> **Saying it out loud.** The claim is that the rescaling does the work and the centering is redundant. Two things support that: the learned scale can absorb whatever offset the model wants, and in a few-thousand-dimensional vector the mean is one number that barely moves the geometry. Zhang and Sennrich showed the ablation is a wash on transformers — same loss curves, same downstream numbers, less compute. It's a rare case where the simplification is free, which is why every recent open model adopted it.

### Where mean-subtraction matters

In some models (especially small or shallow ones), removing the mean keeps activations bounded around zero, which can stabilize training. For deep transformers, the residual stream and pre-LN structure make mean-subtraction redundant.

> **Saying it out loud.** Centering earns its keep when activations can actually drift off zero and stay there — shallow networks, recurrent models, architectures with asymmetric activation functions. In a deep pre-LN transformer the residual stream stays roughly centered on its own, so there's nothing for the mean subtraction to fix. There's also a known sensitivity: without centering, a few enormous outlier channels dominate the root mean square more than they would dominate a centered variance, which is one reason RMSNorm models can be slightly harder to quantize.

### Variance computation precision

Both LN and RMSNorm compute statistics in fp32 even when the activations are bf16, because variance computation in low precision can lose precision. This is a subtle but important implementation detail.

> **Saying it out loud.** Variance is a subtraction of two similar large numbers, which is the textbook way to lose precision, so you compute it in fp32 even when the activations are bf16. bf16 carries about three decimal digits, so summing thousands of squares accumulates real error, and if the mean and mean-of-squares come out close you can get a negative variance — square-root that and you have NaN. Upcast, reduce, cast back; it's nearly free because the data is already in registers. This is one of the most common causes of a mixed-precision run dying with no other symptom.

---

## 7. Initialization and norm interaction

The choice of normalization interacts with initialization in non-trivial ways.

> **Saying it out loud.** The short version is that normalization makes you much less sensitive to how you initialize, because it rescales activations to a fixed magnitude regardless of what the weights were. Before normalization, getting the initialization scale wrong by a factor of two would blow up or vanish across depth, which is why He and Xavier initialization were such a big deal. With normalization you can be sloppy about scale and still train. What you can't be sloppy about is anything the normalization doesn't touch — the residual stream's accumulation with depth, and the per-step update size, which is where muP comes in.

### Why pre-LN models can use larger initialization

The normalization rescales activations to unit variance regardless of init magnitude. So you can initialize with larger weights and the model still trains stably. Compared to plain residual networks that need careful init scaling (He init, etc.).

> **Saying it out loud.** Because the normalization erases the scale before anything downstream sees it. Double every weight and the pre-activation doubles, but then you divide by a standard deviation that also doubled, so the output is unchanged. That makes the forward pass largely invariant to initialization scale, which is a huge robustness win compared to plain residual networks where init scale had to be tuned per-depth. The thing it doesn't make invariant is the effective learning rate — bigger weights mean the same gradient step is a smaller relative change, so scale still affects training dynamics even when it doesn't affect the forward pass.

### muP (maximal update parameterization)

In muP, initialization is rescaled per layer width such that activations stay $O(1)$ regardless of width. The normalization layers cooperate: they ensure activations stay normalized; muP ensures the per-step updates also stay $O(1)$.

> **Saying it out loud.** muP is a set of rules for how initialization and learning rates should scale with model width so that training dynamics stay the same as the model grows. The payoff is enormous in practice: you tune hyperparameters on a small model and transfer them directly to a huge one, instead of guessing or burning compute on sweeps you can't afford at scale. Normalization and muP address different halves of the problem — normalization keeps activations $O(1)$ in the forward pass, muP keeps the updates $O(1)$. You need both, and the practical win is skipping a hyperparameter sweep at 70B, which is worth millions.

### LayerNorm's init effect on transformers

At initialization, LayerNorm acts as a strong regularizer of the residual stream magnitude. Without it, activations would explode after a few layers due to residual accumulation.

> **Saying it out loud.** The thing worth internalizing is that a residual stream accumulates. Every block adds its output to the stream and nothing removes anything, so the magnitude grows roughly with the square root of depth, and by layer 80 the activations would be far outside a sane range. LayerNorm is what keeps the input to each sublayer at a fixed scale despite that growth, so the sublayer sees the same distribution at layer 1 and layer 80. The zero-init trick on $\gamma$ takes it further — every block starts as a no-op and switches itself on gradually, so the model begins as a guaranteed-stable identity function.

---

## 8. The test-time / inference subtleties

### LayerNorm at inference

Identical to training. No statistics tracking. No mode switch. **This is one of LN's biggest practical advantages over BN.**

> **Saying it out loud.** There's nothing to say, and that's the point. Same computation, no mode switch, no statistics to track, no state to get stale. When you serve a LayerNorm model you're running exactly the function you validated. That reliability is worth more than it sounds, because BatchNorm's inference behavior is the source of a whole genre of production bugs that don't announce themselves.

### BatchNorm at inference

Uses running averages of mean and variance accumulated during training. Bugs and instabilities here have caused production incidents at major companies. Common pitfalls:

- Forgetting to put the model in `eval()` mode (PyTorch).
- Running averages stale because of distribution shift.
- Dropout interacting with BN (different magnitudes between train and eval).

> **Saying it out loud.** At inference BatchNorm swaps live batch statistics for running averages collected during training, and everything that can go wrong here does go wrong at some point. Forget `eval()` and your predictions depend on whatever else was in the request batch — the model is technically leaking information between users. Let your data drift and the averages describe a distribution you're no longer serving. And dropout compounds it, because the averages were collected with dropout on and inference runs with it off. The signature is always the same: nothing crashes, accuracy is just quietly worse than validation promised.

### When to freeze BN

For fine-tuning, BN running averages may not match the new data distribution. Common practice: freeze BN parameters during fine-tuning. Some libraries default to this.

> **Saying it out loud.** When you fine-tune on a small or different dataset, the running averages from the original training run are usually better estimates than anything you'd compute from your handful of new batches. So the standard move is to freeze BatchNorm — keep the statistics fixed and let only the rest of the network adapt. It matters most exactly when fine-tuning matters most: small datasets and small batches, where recomputing statistics adds noise instead of information. Forgetting this is a common reason a fine-tune mysteriously underperforms the model it started from.

---

## 9. Failure modes interviewers like to probe

**1. BN with batch size 1.** Statistics are degenerate (variance = 0 within a single sample). LN doesn't have this problem.

**2. LN at the very last position of a long sequence.** Should be fine because LN is per-token. If something looks weird, it's not LN's fault.

**3. RMSNorm with extreme outliers.** A single huge activation can dominate the RMS. Some models use clipping or a larger $\varepsilon$ to prevent this.

**4. Affine parameters initialized incorrectly.** Common bug: $\gamma = 0$ everywhere or $\beta \neq 0$ at init. Most frameworks default to $\gamma = 1, \beta = 0$, which is correct. But custom initializations sometimes break this.

**5. Training-inference discrepancy in BN due to dropout.** Dropout changes the variance of activations. BN's running averages are computed with dropout active; inference computes without. Subtle scaling errors can result.

**6. LN before vs after residual mismatch.** Mixing pre-LN and post-LN in the same architecture is a common bug source. Stick to one.

> **Saying it out loud.** If I had to rank these, batch size one with BatchNorm is the one to have instantly: the variance within a single sample is zero, so the output collapses to just $\beta$ and the layer stops passing information — that's why small-batch vision moved to GroupNorm. The next most useful is the dropout-before-BatchNorm variance shift, because it explains why two individually sensible regularizers can hurt when combined. And on the affine parameters, be precise: $\beta = 0$ at init is correct and standard, and $\gamma = 0$ is not always a bug either — zero-initializing the last $\gamma$ in each block is a deliberate stability trick that makes the network start as an identity function. What is a bug is a custom init that leaves $\gamma$ at zero everywhere, which kills the network outright.

---

## 10. The 8 most-asked normalization interview questions

1. **Why does normalization help?** Smooths the loss landscape, improves conditioning. Original "covariate shift" story is wrong (Santurkar et al.).
2. **BN vs LN?** BN normalizes across batch+spatial per channel; LN per-sample per token across features. LN works at batch=1 and has no train/test discrepancy.
3. **Why don't transformers use BN?** Variable sequence lengths, train-test discrepancy, padding issues. LN handles all cleanly.
4. **What's RMSNorm?** LN without mean subtraction. Cheaper, equally effective for transformers.
5. **Pre-LN vs post-LN?** Pre-LN trains stably without elaborate warmup; post-LN needs careful schedules. All modern LLMs use pre-LN.
6. **What does $\gamma, \beta$ do?** Restore representational capacity that pure standardization removes. Affine transform after normalization.
7. **What's the role of $\varepsilon$?** Numerical stability — prevents division by zero when variance is tiny. Default $10^{-5}$ to $10^{-6}$.
8. **Train-test mode for BN?** Train uses batch statistics; eval uses running averages. LN doesn't have this distinction.

---

## 11. Drill plan

1. Memorize the four-normalizer table (BN/LN/RMSNorm/GroupNorm) including which dimensions each normalizes.
2. Whiteboard LN's forward pass with $\gamma, \beta$.
3. Explain pre-LN vs post-LN training stability story.
4. Explain why BN fails for transformers (3 reasons).
5. Drill [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

---

## 12. Further reading

- Ioffe & Szegedy, "Batch Normalization" (2015).
- Ba, Kiros, Hinton, "Layer Normalization" (2016).
- Wu & He, "Group Normalization" (2018).
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (RMSNorm, 2019).
- Santurkar et al., "How Does Batch Normalization Help Optimization?" (2018).
- Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020) — pre-LN vs post-LN.
- Salimans & Kingma, "Weight Normalization" (2016).
- Henry et al., "Query-Key Normalization for Transformers" (QK-Norm, 2020).
