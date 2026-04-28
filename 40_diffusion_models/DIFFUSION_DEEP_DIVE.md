# Diffusion Models: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Diffusion is the dominant paradigm for image, video, and 3D generation, and it's increasingly applied beyond. Interviewers probe: forward/reverse processes, why we predict noise, classifier-free guidance, latent vs pixel-space diffusion, flow matching. This document covers the math without the dense Bayesian notation.

---

## 1. The big picture

A diffusion model has two processes:

**Forward (fixed):** progressively add noise to data over $T$ steps, transforming $x_0$ (data) into $x_T$ (pure Gaussian noise).

**Reverse (learned):** progressively denoise, starting from $x_T$ (pure noise) and producing $x_0$ (data sample).

The model learns the reverse process. At sampling time, run the reverse process with a fresh random Gaussian; you get a fresh sample from the learned data distribution.

**Why this works:** the forward process has a simple form (add Gaussian noise; tractable mathematically). The reverse process — which is what generates data — can be learned by training the model to undo each forward step.

---

## 2. The forward process

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\big(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t I\big)
$$

At each step, mix $x_{t-1}$ with Gaussian noise of variance $\beta_t$. Iterate $T$ steps. $\beta_t$ follows a **schedule** (linear, cosine, etc.) — typically small early, larger later.

### Closed form

A key property: you can sample $x_t$ directly from $x_0$ without iterating:

$$
q(x_t \mid x_0) = \mathcal{N}\!\big(x_t;\, \sqrt{\bar\alpha_t}\, x_0,\, (1 - \bar\alpha_t) I\big)
$$

where $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. So:

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

This direct sampling is critical: during training, you don't need to iterate the forward process — you sample a random $t$ and a random $\varepsilon$ and compute $x_t$ directly.

### Variance schedule

**Linear:** $\beta_t$ linearly interpolated from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ over $T = 1000$ steps. Original DDPM choice.

**Cosine** (Nichol & Dhariwal 2021): $\bar\alpha_t = \cos^2(\cdot)$. Smoother decay; better for high-resolution images.

**Variance-preserving (VP) vs variance-exploding (VE):** different parameterizations of the diffusion process. VP keeps $\operatorname{Var}(x_t)$ near 1; VE lets it grow. VP (DDPM-style) is the more common choice.

---

## 3. The reverse process

We want to learn $p_\theta(x_{t-1} \mid x_t)$. The true posterior $q(x_{t-1} \mid x_t, x_0)$ is also Gaussian (Bayes on the forward Markov chain), with mean derivable in terms of $x_0$ and $x_t$. The training trick: parameterize the model to predict either $x_0$, $\varepsilon$ (the noise), or the score (gradient of log-density).

### Predicting noise (the standard choice)

DDPM (Ho et al. 2020) parameterizes the reverse mean as:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \varepsilon_\theta(x_t, t) \right)
$$

where $\varepsilon_\theta$ is the model's prediction of the noise that was added to get to $x_t$.

The training loss simplifies dramatically:

$$
\mathcal{L} = \mathbb{E}_{t,\, x_0,\, \varepsilon}\!\left[ \big\| \varepsilon - \varepsilon_\theta(x_t, t) \big\|^2 \right]
$$

**MSE between predicted and actual noise.** That's the entire training objective.

### Why predict noise specifically?

Mathematically equivalent options:

- Predict $x_0$ directly.
- Predict $\varepsilon$.
- Predict the **score** $\nabla_x \log p_t(x)$ (Tweedie's formula).

Empirically, predicting $\varepsilon$ works best because:

- The target has constant scale (normalized).
- Loss is well-conditioned across all timesteps.
- Easier to train than $x_0$ prediction (which needs to span full data range).

### Score matching connection

Predicting $\varepsilon$ is equivalent to predicting the **score** (gradient of log density):

$$
\varepsilon \approx -\sigma_t\, \nabla_x \log p_t(x)
$$

So diffusion models are **score-based generative models** — they learn to follow gradients of log-density. This is the Song & Ermon line of work; DDPM (Ho et al.) and score-matching (Song & Ermon) are equivalent up to parameterization.

---

## 4. Sampling

Once trained, sample from $p(x_0)$ by running the reverse process:

1. Start with $x_T \sim \mathcal{N}(0, I)$.
2. For $t$ from $T$ down to $1$:
    - Predict noise: $\hat\varepsilon = \varepsilon_\theta(x_t, t)$.
    - Compute mean: $\mu = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\hat\varepsilon\right)$.
    - Add Gaussian noise: $x_{t-1} = \mu + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$.
3. Return $x_0$.

For $T = 1000$, this is 1000 model forward passes per sample. **Slow.**

### DDIM: deterministic sampling with fewer steps

DDIM (Song et al. 2021) reformulates the reverse process to be deterministic and to allow skipping steps:

$$
x_{t-k} = \sqrt{\bar\alpha_{t-k}}\, \frac{x_t - \sqrt{1 - \bar\alpha_t}\,\hat\varepsilon}{\sqrt{\bar\alpha_t}} + \sqrt{1 - \bar\alpha_{t-k}}\,\hat\varepsilon
$$

Same model, different sampling. With DDIM you can sample in 50–100 steps with quality comparable to 1000-step DDPM. **Standard for production diffusion.**

### Even faster sampling

- **DPM-Solver, DPM-Solver++ (Lu et al.):** ODE/SDE solvers that exploit the structure of the diffusion ODE. ~20 steps with strong quality.
- **Consistency Models (Song et al. 2023):** distill diffusion into a model that goes from noise to data in 1–2 steps. Sacrifices some quality for speed.
- **Rectified Flow / Flow Matching:** learn a straighter trajectory; sample in fewer steps.

The active research direction is reducing sampling steps from 1000 to 1–4 while preserving quality.

---

## 5. The ELBO and the loss derivation

For interview-grade understanding (often asked):

The ELBO for diffusion models:

$$
\log p_\theta(x_0) \geq \mathbb{E}_q[\log p_\theta(x_0 \mid x_1)] - \sum_{t > 1} \mathbb{E}_q\!\big[\mathrm{KL}\!\big(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\big)\big] - \mathrm{KL}\!\big(q(x_T \mid x_0) \,\|\, p(x_T)\big)
$$

After algebra (dropping irrelevant constants), each KL term reduces to:

$$
\mathcal{L}_t = \mathbb{E}_{x_0, \varepsilon}\!\left[\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar\alpha_t)}\, \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]
$$

DDPM uses the **simplified loss** (drop the prefactor):

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\!\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]
$$

Empirically, the simplified loss works better than the weighted ELBO. The prefactor would over-weight some timesteps.

---

## 6. Classifier-free guidance (CFG)

A critical technique for conditional generation (text-to-image, etc.).

### Setup

The model is trained jointly:

- Conditional: $\varepsilon_\theta(x_t, t, c)$ where $c$ is the conditioning (e.g., text embedding).
- Unconditional: $\varepsilon_\theta(x_t, t, \emptyset)$ — replace $c$ with a null embedding 10–20% of the time during training.

### At sampling

Combine the two predictions:

$$
\hat\varepsilon_{\text{guided}} = \varepsilon_\theta(x_t, t, \emptyset) + w \cdot \big(\varepsilon_\theta(x_t, t, c) - \varepsilon_\theta(x_t, t, \emptyset)\big)
$$

$w$ is the **guidance scale** (typically 1.5–7.5). $w = 1$ means no guidance (just use conditional). $w > 1$ amplifies the conditional signal.

### Why this works

The difference $\varepsilon_{\text{cond}} - \varepsilon_{\text{uncond}}$ is a direction that "points toward the condition" in score space. Amplifying it pushes the sample more strongly toward the condition.

### Trade-offs

- **High $w$:** stronger adherence to condition, but may produce overexposed / oversaturated images. Sample diversity drops.
- **Low $w$:** more diverse samples, weaker adherence to condition.
- Stable Diffusion typically uses $w = 7.5$.

CFG is ubiquitous in text-to-image. Almost every paper since 2022 uses it.

### Classifier guidance (older)

The original conditioning method (Dhariwal & Nichol 2021): use a separate classifier's gradient $\nabla_x \log p(c \mid x_t)$ to push samples toward the condition. Replaced by CFG which doesn't need a separate classifier.

---

## 7. Latent diffusion (Stable Diffusion)

### The problem

Pixel-space diffusion is expensive. A 512×512 RGB image has 786K pixels. Forward/reverse passes through a UNet on this is slow.

### The fix

Latent diffusion (Rombach et al. 2022, Stable Diffusion):

1. Encode image to a smaller latent $z$ via a pretrained autoencoder (4–8x downsampling).
2. Run diffusion in the latent space $z$ (much smaller, faster).
3. Decode back to pixels at the end.

$$
\text{encode: } x_{\text{pixel}} \to z_{\text{latent}} \quad \text{(VAE encoder)}
$$
$$
\text{diffuse + denoise } z_{\text{latent}}
$$
$$
\text{decode: } z_{\text{latent}} \to x_{\text{pixel}} \quad \text{(VAE decoder)}
$$

### Why it works

- Most "perceptual" content (textures, semantics) is captured in the latent.
- Diffusion in latent space is 4–8x cheaper.
- Final image quality limited by the VAE's reconstruction quality, but in practice this is fine.

### Stable Diffusion family

SD 1.x, SD 2.x, SDXL, SD 3 — all latent diffusion. Differences: VAE quality, UNet vs Transformer (DiT), training data, conditioning model (CLIP vs T5), schedules.

---

## 8. Architecture: UNet vs DiT

### UNet (DDPM, Stable Diffusion 1.x/2.x)

Convolutional U-shape with skip connections. Down-sampling encoder + up-sampling decoder. Cross-attention layers for conditioning. Standard for diffusion until ~2023.

### DiT (Diffusion Transformer, Peebles & Xie 2022)

Replace the UNet with a transformer over image patches. Same idea as ViT. Better scaling properties; SD 3, FLUX use DiT.

### Why DiT wins at scale

Transformers scale predictably with parameters and data. Convolutional UNets have hand-crafted inductive biases that limit scalability. As diffusion models grow, DiT-style architectures dominate.

---

## 9. Flow Matching and Rectified Flow (recent)

A reformulation of diffusion that's becoming dominant:

> Learn a velocity field $v_\theta(x_t, t)$ that transforms noise → data along a continuous path.

Key ideas:

- **Straighter paths:** flow matching produces ODEs with straighter trajectories than diffusion. Fewer sampling steps for equivalent quality.
- **Simpler training:** the loss is similar to noise prediction but conceptually cleaner.
- **Same model in practice:** the trained network is equivalent to a diffusion network, but the training objective and sampling are different.

Used in Stable Diffusion 3, FLUX, recent video models. **Likely to replace pure diffusion as the dominant paradigm.**

---

## 10. Conditioning

Diffusion models are conditioned in many ways:

### Text (CLIP / T5)

Embed text prompt with CLIP or T5; inject into UNet/DiT via cross-attention. Modern models often use multiple text encoders combined.

### Image (image conditioning)

For image-to-image, inpainting, super-resolution: concatenate noisy latent with condition image latent at every step.

### ControlNet

Add an auxiliary network that processes structural conditions (depth, segmentation, edges) and injects them via additional cross-attention. Lightweight; widely used for spatial control.

### LoRA / DreamBooth

Fine-tune diffusion models on small datasets to add new concepts (a person, a style). LoRA-style updates dominate for personalization.

---

## 11. Diffusion in NLP

Mostly research-stage. Pure diffusion for text is hard because:

- Text is discrete; diffusion is naturally continuous.
- Workarounds: diffuse in embedding space, or use special discrete diffusion processes.

Recent: SEDD (Score Entropy Discrete Diffusion), Diffusion-LM. Promising but not at frontier-LLM scale yet.

For text generation, autoregressive models still dominate.

---

## 12. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "Why predict noise instead of $x_0$?" | MSE loss is well-conditioned across timesteps. $\varepsilon$ targets have constant scale; $x_0$ targets span full data range. |
| "Is diffusion training computationally expensive?" | Each step: one forward pass on a noisy image. Many steps over time but each is parallelizable. Comparable to other generative models. |
| "Why is sampling slow?" | Need many denoising steps (1000 for DDPM, 50–100 for DDIM). Recent: consistency models can do it in 1–4 steps. |
| "What's CFG?" | Combine conditional and unconditional predictions during sampling; amplify the conditional direction. Standard for text-to-image. |
| "Why latent diffusion?" | Diffuse in compressed latent space (via VAE encoder), much cheaper than pixel space. Stable Diffusion innovation. |
| "DiT vs UNet?" | DiT (transformer) scales better than UNet (conv). Modern flagship models use DiT. |
| "What's flow matching?" | Reformulation with straighter trajectories; fewer sampling steps. Used in SD3, FLUX. Likely to replace pure diffusion. |
| "Diffusion vs GANs?" | Diffusion: stable training, no mode collapse, slower sampling. GANs: fast sampling, harder training, mode collapse. Diffusion has won. |
| "Is diffusion an MLE?" | Approximately, via the ELBO. Simplified loss is not exactly MLE but works better empirically. |

---

## 13. The 10 most-asked diffusion interview questions

1. **What's the forward process?** Add Gaussian noise over $T$ steps. Closed-form: $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon$.
2. **What's the reverse process?** Learned denoising. Predict noise $\hat\varepsilon = \varepsilon_\theta(x_t, t)$ and use it to compute $\mu$ for $x_{t-1}$.
3. **Why predict noise not data?** Better-conditioned loss; constant scale across timesteps.
4. **What's DDIM?** Deterministic sampler that allows fewer steps (50–100 vs 1000). Same trained model.
5. **What's classifier-free guidance?** Train conditional + unconditional jointly; combine at sampling: $\hat\varepsilon = \varepsilon_{\text{unc}} + w \cdot (\varepsilon_{\text{cond}} - \varepsilon_{\text{unc}})$. Standard for text-to-image.
6. **Why latent diffusion?** Diffuse in compressed latent space; 4–8x cheaper than pixel space.
7. **DiT vs UNet?** DiT (transformer) scales better; modern flagship models use it.
8. **What's flow matching?** Reformulation with straighter paths; fewer sampling steps. Likely future of diffusion.
9. **What's the ELBO for diffusion?** Sum of KL terms across timesteps. Simplified MSE loss works better empirically.
10. **Connection between score matching and diffusion?** Equivalent. Predicting $\varepsilon \approx$ predicting $-\sigma \nabla \log p$. Diffusion models are score-based generative models.

---

## 14. Drill plan

1. Memorize the closed-form $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon$.
2. Memorize the simplified DDPM loss: MSE on noise prediction.
3. Know CFG: train cond+uncond, combine at sampling.
4. Know latent diffusion's role (Stable Diffusion).
5. Know DiT and flow matching as the modern direction.
6. Drill `INTERVIEW_GRILL.md`.

---

## 15. Further reading

- Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015) — original diffusion idea.
- Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models" (DDPM, 2020).
- Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution" (score-based, 2019).
- Song et al., "Denoising Diffusion Implicit Models" (DDIM, 2021).
- Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (classifier guidance, 2021).
- Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022).
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion, 2022).
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT, 2023).
- Lipman et al., "Flow Matching for Generative Modeling" (2023).
- Liu et al., "Rectified Flow" (2022).
