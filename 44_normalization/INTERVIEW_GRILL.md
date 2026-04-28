# Normalization — Interview Grill

> 40 questions on normalization in deep learning. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. What does normalization do mathematically?**
Standardize activations to (approximately) zero mean and unit variance, then apply a learned affine transform $\gamma \cdot x_{\text{norm}} + \beta$. The standardization is the "normalization"; the affine transform restores capacity.

**2. Why do we normalize?**
To improve optimization. Best current understanding: normalization smooths the loss landscape (smaller gradient Lipschitz constant), enabling larger learning rates and faster training. Equivalent in spirit to per-feature preconditioning.

**3. What was the original justification for BN, and why is it wrong?**
Ioffe & Szegedy (2015) proposed BN to fix "internal covariate shift" — the changing distribution of layer inputs during training. Santurkar et al. (2018) empirically showed BN doesn't actually reduce covariate shift much. The current believed explanation is loss landscape smoothing.

**4. What's the role of $\gamma$ and $\beta$?**
Standardization sets every dimension to mean 0, variance 1. The affine transform $\gamma \cdot x_{\text{norm}} + \beta$ allows the model to recover any other distribution if useful. So normalization is "invertible" in principle; in practice, $\gamma$ and $\beta$ learn values that work better than the original.

**5. What's $\varepsilon$ for in normalization?**
Numerical stability. Prevents division by zero when variance is near zero. Standard values: $10^{-5}$ to $10^{-6}$. If $\varepsilon$ is too small, can cause NaNs in fp16; if too large, weakens normalization. Some recipes use $10^{-3}$ for embeddings or low-precision training.

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

**7. Walk me through LayerNorm.**
For a 3D activation $[B, N, d]$, normalize per-sample per-token across features:

$$
\mu = \mathrm{mean}(X[b, n, :]), \qquad \sigma^2 = \mathrm{var}(X[b, n, :])
$$

$$
\hat x = (x - \mu) / \sqrt{\sigma^2 + \varepsilon}, \qquad \text{output} = \gamma \cdot \hat x + \beta
$$

Single $\gamma, \beta$ (per-feature). No running averages needed.

**8. What's RMSNorm?**
LayerNorm without mean subtraction:

$$
\mathrm{RMSNorm}(x) = \gamma \cdot x / \mathrm{RMS}(x), \qquad \mathrm{RMS}(x) = \sqrt{\mathrm{mean}(x^2) + \varepsilon}
$$

Just unit-variance normalization. ~30% cheaper. Empirically as good as LN for transformers. Used in LLaMA, Gemma, Mistral.

**9. What's GroupNorm?**
Compromise between LN and BN: normalize per-sample across groups of channels. For 4D $[B, C, H, W]$, split channels into $G$ groups, normalize each group separately:

$$
\mu = \mathrm{mean}(X[b, \text{group}, :, :]), \qquad \sigma^2 = \mathrm{var}(\cdots)
$$

Used in vision when batch size is small. Some diffusion models use it.

**10. BN vs LN: tabular comparison.**

| Aspect | BN | LN |
|---|---|---|
| Norm dim | batch + spatial per channel | features per token per sample |
| Per-batch statistics | Yes | No |
| Train/test discrepancy | Yes (running averages) | No |
| Works at batch=1 | No (variance = 0) | Yes |
| Works for variable seq lengths | No (padding issues) | Yes |
| Used in | CNNs, vision | Transformers |

---

## C. Why transformers don't use BN

**11. Why don't transformers use BatchNorm?**
Three reasons. (a) Variable sequence lengths cause padding, which corrupts BN statistics. (b) BN's running averages can drift due to distribution shift between training and inference. (c) BN needs reasonable batch size for stable statistics; transformer training often has many short sequences with heterogeneous structure.

**12. Why does LayerNorm work where BN fails?**
LN is purely per-token, per-sample. No batch dimension. Identical at train and test. Works at batch=1. Handles variable sequence lengths trivially. None of BN's pathologies apply.

**13. Could BN be made to work for transformers in principle?**
With workarounds (mask out padding, large enough effective batch size, careful train-test handling) yes — but the workarounds are ugly and provide no benefit over LN. So in practice nobody does it.

**14. SyncBN — what and why?**
For multi-GPU training with small per-GPU batch sizes, BN's per-GPU statistics are noisy. SyncBN aggregates statistics across GPUs (extra communication cost) for stable normalization. Common in vision; not relevant for transformers.

---

## D. Pre-LN vs post-LN

**15. What's post-LN?**
The original transformer (Vaswani 2017): $x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. Norm comes after the residual addition.

**16. What's pre-LN?**
Modern: $x \leftarrow x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$. Norm comes before the sublayer; the residual is unnormed.

**17. Why does pre-LN train more stably?**
Two reasons. (a) Gradient flow: in pre-LN, the residual path is identity through LayerNorm, so gradients flow unchanged. In post-LN, every block applies LN to the gradient on the way back, which can attenuate signal. (b) Residual stream magnitude: pre-LN preserves the unnormed stream; post-LN renormalizes after every block, amplifying any perturbations.

**18. What did post-LN need for stability?**
Elaborate warmup schedules. The original paper uses warmup over thousands of steps with a specific formula $\text{lr} \propto d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$. Without it, post-LN diverges at modern scales.

**19. Are there any modern uses of post-LN?**
Some encoder-only models still use it (BERT). For decoder-only LLMs at any meaningful scale, pre-LN is universal.

**20. What's a sandwich norm?**
LayerNorm both before AND after each sublayer. Used in some recent architectures (GPT-J variants). More compute, occasionally helps stability. Sometimes called "double norm."

**21. What's QK-norm?**
Apply LayerNorm separately to $Q$ and $K$ before computing attention scores. Stabilizes attention at very large model scales. Used in some recent papers as a stability fix.

---

## E. RMSNorm specifics

**22. Why does RMSNorm work without mean subtraction?**
Empirically: the variance normalization is doing most of the work; removing the mean is redundant given that $\gamma, \beta$ can recenter if needed. For deep transformers with pre-LN structure, mean-subtraction is largely redundant.

**23. What does RMSNorm save?**
~30% compute over LayerNorm (one fewer reduction across the feature dimension). For LLM pretraining, where LayerNorm is computed many times per token per layer, this adds up.

**24. Are there cases where LN beats RMSNorm?**
Yes — in some shallow networks or non-transformer architectures, mean-subtraction matters more. But for modern deep transformers, RMSNorm is fine.

**25. Why is variance computed in fp32 even with bf16 activations?**
Variance computation in low precision can lose significant figures (subtraction of similar numbers, summation of squares). Computing in fp32 and casting back is standard. Same for the affine transform's $\gamma$ multiplication.

---

## F. Affine transform and inference

**26. Doesn't $\gamma$ and $\beta$ just undo the normalization?**
The optimizer can learn $\gamma = \sigma_{\text{original}}, \beta = \mu_{\text{original}}$ to recover the un-normalized activation in principle. But the optimization landscape with normalization is fundamentally different — gradients flow differently, learning rate sensitivity changes. The point isn't that activations are different; the optimization dynamics are improved.

**27. What if $\gamma$ is initialized at 0?**
A common trick: zero-init $\gamma$ in the LN before residual addition. Effect: at init, the sublayer contributes nothing. The model starts as a pure residual identity stream and the layers gradually become non-trivial as $\gamma$ grows. Helps stability at very large scales.

**28. Layer at inference: what's different?**
LN: nothing. Identical to training. RMSNorm: same. **BN: significantly different.** BN at inference uses running averages of $\mu$ and $\sigma$ accumulated during training, not batch statistics. PyTorch's `model.eval()` flag controls this. Forgetting to set it is a common production bug.

**29. Why is BN's train-test discrepancy a problem?**
Subtle bugs: (a) inference batch composition differs from training, so running averages may be inaccurate; (b) single-sequence inference uses normalization that depends on running averages which may drift; (c) interaction with dropout (different active state at train vs test) compounds.

---

## G. Edge cases and gotchas

**30. BN with batch size 1?**
Variance is 0 (single sample), normalization divides by $\varepsilon$, output is dominated by $\gamma \cdot 0 + \beta = \beta$. Useless. LN works fine at batch=1.

**31. LN with sequence length 1?**
Fine. LN is per-token. Sequence length doesn't matter.

**32. How does dropout interact with normalization?**
Dropout zeros some activations during training, changing their distribution. BN computes statistics including the dropout effect, so its running averages reflect the "with dropout" distribution. At inference (no dropout), the active activations are larger (no zeros), causing slight mismatch with running averages. Modern recipes either: (a) apply dropout AFTER normalization to avoid this, or (b) accept the small discrepancy.

**33. What's "pre-LN with parallel sublayers"?**
GPT-J style: compute attention and FFN in **parallel** from the same input rather than sequentially:

$$
x' = x + \mathrm{Attention}(\mathrm{LN}(x)) + \mathrm{FFN}(\mathrm{LN}(x))
$$

Saves one dependency in the dataflow graph (slight speedup). Empirically comparable to standard pre-LN.

**34. WeightNorm?**
Salimans & Kingma 2016. Reparameterize each weight as $W = (g / \|v\|) \cdot v$. Decouples direction from magnitude. Less popular today but historically important. Mostly superseded by LN/BN.

**35. Why is normalization sensitive to extreme outliers?**
Activation outliers (a few channels with 100x typical magnitude) disturb the variance estimate. RMSNorm is more affected than LN (no mean subtraction to attenuate). SmoothQuant deals with this for quantization; for fp32 training, it's usually fine but can require larger $\varepsilon$ or gradient clipping.

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
