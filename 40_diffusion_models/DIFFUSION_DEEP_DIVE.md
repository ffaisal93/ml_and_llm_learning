# Diffusion Models: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Diffusion is the dominant paradigm for image, video, and 3D generation, and it's increasingly applied beyond. Interviewers probe: forward/reverse processes, why we predict noise, classifier-free guidance, latent vs pixel-space diffusion, flow matching. This document covers the math without the dense Bayesian notation.

---

## 1. The big picture

A diffusion model has two processes:

**Forward (fixed):** progressively add noise to data over $T$ steps, transforming $x_0$ (data) into $x_T$ (pure Gaussian noise).

**Reverse (learned):** progressively denoise, starting from $x_T$ (pure noise) and producing $x_0$ (data sample).

The model learns the reverse process. At sampling time, run the reverse process with a fresh random Gaussian; you get a fresh sample from the learned data distribution.

**Why this works:** the forward process has a simple form (add Gaussian noise; tractable mathematically). The reverse process — which is what generates data — can be learned by training the model to undo each forward step.

> **Saying it out loud.** Picture a photograph slowly dissolving into television static. Going that direction is trivial — you just keep adding noise, and there's nothing to learn. The whole trick of diffusion is that the model learns to run that film backwards, one frame at a time, and once it can do that you can start from real static and end up with an image that never existed. The reason it beats trying to generate a picture in one shot is that each individual step is an easy, well-posed denoising problem, and you get hundreds of them to gradually commit to a coherent image. The tradeoff is baked in from the start: one network evaluation per step means generation is inherently slow.

---

## 2. The forward process

*In plain language:* the forward process is the destruction half, and nothing here is learned. You take a clean image and repeatedly mix in a little Gaussian noise until only noise is left. The two formulas below are the per-step version and — much more useful — the shortcut that lets you jump from the clean image to any noise level in one line.

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

> **Saying it out loud.** The closed form is the single most load-bearing equation in diffusion, so it's worth saying slowly. Because a Gaussian plus a Gaussian is a Gaussian, all $t$ noising steps compose analytically into one: the noisy image at step $t$ is just $\sqrt{\bar\alpha_t}$ times the clean image plus $\sqrt{1-\bar\alpha_t}$ times a single fresh noise draw. So during training you never simulate the chain — you roll a random timestep, draw one noise vector, and construct the input directly. That makes every training example independent and the whole thing perfectly parallel. Which sets up diffusion's defining asymmetry: training parallelizes completely, sampling is strictly sequential.

### Variance schedule

**Linear:** $\beta_t$ linearly interpolated from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ over $T = 1000$ steps. Original DDPM choice.

**Cosine** (Nichol & Dhariwal 2021): $\bar\alpha_t = \cos^2(\cdot)$. Smoother decay; better for high-resolution images.

**Variance-preserving (VP) vs variance-exploding (VE):** different parameterizations of the diffusion process. VP keeps $\mathrm{Var}(x_t)$ near 1; VE lets it grow. VP (DDPM-style) is the more common choice.

> **Saying it out loud.** The schedule decides how fast you destroy the image, and it's more consequential than it sounds. DDPM's linear ramp from 0.0001 to 0.02 over a thousand steps turned out to destroy information too early — by two-thirds of the way through, the image is already effectively pure noise, so a big chunk of your timesteps teach the model nothing. The cosine schedule keeps signal around longer and matters most at high resolution. And VP versus VE is just a bookkeeping choice: VP shrinks the signal as it adds noise so total variance stays near one, VE piles noise on top and lets variance grow. VP wins in practice because a constant input scale is much friendlier to a neural network.

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

> **Saying it out loud.** Here's the thing that surprises people: after all the probability theory, the training loop is four lines. Take a clean image, pick a random timestep, add a known amount of noise, ask the network what noise it sees, and penalize the squared difference. That's it — no adversary, no sampling during training, no unstable dynamics. The reason it's legal is that the true reverse of each small forward step is itself Gaussian with a known variance, so all the model has to supply is a mean, and predicting the noise determines the mean algebraically. The stability of that objective is why diffusion beat GANs: it's regression, and regression scales.

### Why predict noise specifically?

Mathematically equivalent options:

- Predict $x_0$ directly.
- Predict $\varepsilon$.
- Predict the **score** $\nabla_x \log p_t(x)$ (Tweedie's formula).

Empirically, predicting $\varepsilon$ works best because:

- The target has constant scale (normalized).
- Loss is well-conditioned across all timesteps.
- Easier to train than $x_0$ prediction (which needs to span full data range).

> **Saying it out loud.** The reason for predicting noise rather than the clean image is conditioning of the loss. If your target is the clean image, then at low noise levels the task is nearly trivial and at high noise levels you're asking the network to invent an entire image out of static — so the loss magnitude swings enormously across timesteps and training is badly behaved. The noise target is a unit Gaussian at every timestep, so the statistics never change and each timestep contributes comparably to the gradient. It's worth naming the third option too: $v$-prediction interpolates between the two and is what people reach for at high guidance scales and for distillation, precisely because pure noise prediction gets poorly conditioned at the very last denoising steps.

### Score matching connection

Predicting $\varepsilon$ is equivalent to predicting the **score** (gradient of log density):

$$
\varepsilon \approx -\sigma_t\, \nabla_x \log p_t(x)
$$

So diffusion models are **score-based generative models** — they learn to follow gradients of log-density. This is the Song & Ermon line of work; DDPM (Ho et al.) and score-matching (Song & Ermon) are equivalent up to parameterization.

> **Saying it out loud.** This is the observation that merged two separate research programs. The score is the gradient of the log-density — from wherever you are, it points toward more probable data. And predicting the added noise turns out to be the same quantity up to a sign and a scale factor, which makes intuitive sense, because the noise *is* the direction you'd move to make the image more plausible. So Ho's DDPM and Song's score matching are one model in two notations. The payoff isn't aesthetic: once you see it as a score, you can write the whole thing as a continuous-time SDE, derive the equivalent probability-flow ODE, and hand it to any off-the-shelf ODE solver — which is where every fast sampler comes from.

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

> **Saying it out loud.** Sampling is the loop that costs you. Start from pure noise, ask the model what noise it sees, subtract most of it, add a little fresh noise back, repeat — and the noise you add back is what gives you diversity across seeds. The number to quote is a thousand forward passes per image for vanilla DDPM, strictly sequential, since each step needs the previous one. DDIM cuts that to fifty or a hundred by making the reverse process deterministic, which turns it into an ODE you're allowed to take big steps along, with no retraining. Better solvers get you to about twenty, and distilled consistency models to one to four. What you give up as you shorten the chain is fine detail and sample diversity.

---

## 5. The ELBO and the loss derivation

*In plain language:* this section explains where that simple noise-prediction MSE actually comes from. You can't compute the likelihood of an image under the model, so you bound it from below, and because the noising process is a chain of small Gaussian steps, that bound breaks into one comparison per step. Every comparison turns out to be "how far off was your noise guess," which is why the whole apparatus collapses into a squared error. The equations below are that collapse, written out.

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

> **Saying it out loud.** If they ask you to derive it, the arc is short. You can't compute $\log p_\theta(x_0)$, so you write a variational lower bound. Because the forward process is a Markov chain, that bound decomposes into one KL divergence per timestep between the true posterior and your model's reverse step. Then the lucky part: both of those are Gaussians with the same known variance, and the KL between two equal-variance Gaussians is just the squared distance between their means. Rewrite the means in terms of noise and each term is a weighted MSE on noise prediction. DDPM then deletes the weights, and this is the honest bit — the weighted version is the correct ELBO and the unweighted version makes better pictures, because the ELBO's weights pour effort into low-noise steps where the remaining error is imperceptible instead of the high-noise steps that decide the image's structure.

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

> **Saying it out loud.** Classifier-free guidance is the trick that made text-to-image actually obey the prompt. During training you drop the caption maybe 10 to 20 percent of the time, so a single network learns both a conditional and an unconditional model. At sampling you run it twice and extrapolate: the difference between the two predictions is a pure "more of what the prompt asked for" direction, since everything generic cancels, and you overshoot along it by a factor $w$. Formally you're sampling from a sharpened conditional. Which explains the failure mode exactly — push $w$ past about 15 and you get blown-out, oversaturated images and near-zero diversity, because you've concentrated all the mass on a few modes. Stable Diffusion's 7.5 is the tuned compromise, and you pay for all of it with two forward passes per step.

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

> **Saying it out loud.** Latent diffusion is what put image generation on consumer hardware. Rather than denoising 512 by 512 pixels, you compress the image with a pretrained VAE encoder into a 64 by 64 latent, run the entire diffusion process in there, and decode once at the end. That's roughly 48 times fewer values, and since attention and convolution scale with spatial size, it's something like a 40 to 60 times reduction in compute per step. It works because most pixel-level detail is perceptually redundant — the VAE strips it out and the diffusion model gets to spend its capacity on semantics. The tradeoff to name is that the VAE is a hard ceiling: whatever the decoder can't reconstruct, the model can't generate, which is precisely why Stable Diffusion has always been bad at small text and fine facial detail.

---

## 8. Architecture: UNet vs DiT

### UNet (DDPM, Stable Diffusion 1.x/2.x)

Convolutional U-shape with skip connections. Down-sampling encoder + up-sampling decoder. Cross-attention layers for conditioning. Standard for diffusion until ~2023.

### DiT (Diffusion Transformer, Peebles & Xie 2022)

Replace the UNet with a transformer over image patches. Same idea as ViT. Better scaling properties; SD 3, FLUX use DiT.

### Why DiT wins at scale

Transformers scale predictably with parameters and data. Convolutional UNets have hand-crafted inductive biases that limit scalability. As diffusion models grow, DiT-style architectures dominate.

> **Saying it out loud.** The UNet was the natural first architecture because denoising takes an image in and returns an image of the same size, and skip connections preserve fine detail across the bottleneck. But every choice in it — how many resolution levels, where to place attention — is hand-designed, and that's what caps it. DiT throws the U-shape away and treats the latent as a flat sequence of patches, one resolution throughout, with conditioning injected through adaptive layer norm. The argument that won is a scaling law: DiT quality improves smoothly and predictably with training FLOPs where the UNet plateaus. Plus you inherit the entire transformer tooling ecosystem for free, which is why SD3 and FLUX are both DiTs.

---

## 9. Flow Matching and Rectified Flow (recent)

A reformulation of diffusion that's becoming dominant:

> Learn a velocity field $v_\theta(x_t, t)$ that transforms noise → data along a continuous path.

Key ideas:

- **Straighter paths:** flow matching produces ODEs with straighter trajectories than diffusion. Fewer sampling steps for equivalent quality.
- **Simpler training:** the loss is similar to noise prediction but conceptually cleaner.
- **Same model in practice:** the trained network is equivalent to a diffusion network, but the training objective and sampling are different.

Used in Stable Diffusion 3, FLUX, recent video models. **Likely to replace pure diffusion as the dominant paradigm.**

> **Saying it out loud.** Flow matching asks a simpler question. Draw a straight line from a noise sample to a data sample, pick a random point along it, and train the network to predict the direction of travel — the velocity. Same MSE loss, but no noise schedule to design, no Markov chain, no ELBO. Sampling means integrating that velocity field with an ODE solver. The reason it matters is purely geometric: diffusion's trajectory from noise to data is curved, and a solver taking big steps along a curve accumulates error fast, whereas a straight path is exactly what a coarse solver approximates well. In practice that's good samples in 20 to 30 steps instead of 50 to 100. It isn't a different model family — noise, score, and velocity prediction are reparameterizations of each other — it's a better-conditioned choice of path.

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

> **Saying it out loud.** Conditioning is layered, and it's worth naming the mechanism for each. Text enters through cross-attention, where image features are queries and the prompt provides keys and values, so the model consults the prompt at every layer and every denoising step rather than once at the start. Image conditioning for inpainting and super-resolution is usually just concatenated onto the input channels. ControlNet clones the encoder, feeds the copy a depth map or pose skeleton, and merges it back through zero-initialized convolutions — the zero init is the key detail, since it means training starts out as a no-op and the control fades in without damaging the base model. And LoRA won personalization over full DreamBooth fine-tuning purely on logistics: a few megabytes instead of gigabytes, stackable, and shareable.

---

## 11. Diffusion in NLP

Mostly research-stage. Pure diffusion for text is hard because:

- Text is discrete; diffusion is naturally continuous.
- Workarounds: diffuse in embedding space, or use special discrete diffusion processes.

Recent: SEDD (Score Entropy Discrete Diffusion), Diffusion-LM. Promising but not at frontier-LLM scale yet.

For text generation, autoregressive models still dominate.

> **Saying it out loud.** The obstacle is that diffusion is built on adding a little continuous Gaussian noise, and there's no such thing as slightly noising the word "cat." Workarounds either diffuse in a continuous embedding space and round at the end, or define genuinely discrete corruption processes like progressive masking, which is what SEDD does. The upside would be real — parallel generation of an entire sequence instead of one token at a time, plus natural infilling and editing. But autoregressive models are far ahead on quality per unit of compute, partly because next-token prediction gives an exact likelihood and partly because a decade of infrastructure is built around it. Interesting research, not yet a threat.

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

> **Saying it out loud.** The traps here are mostly about conflating training cost with sampling cost. Training diffusion is cheap and fully parallel, because the closed form lets any example jump to any timestep independently — it's sampling that's slow, and slow for a structural reason, namely that each denoising step needs the previous one. Second trap: noise prediction versus $x_0$ prediction isn't arbitrary, it's about keeping the loss well-conditioned across timesteps. Third: diffusion isn't exactly maximum likelihood — the simplified loss deliberately throws away the ELBO's weights and does better on perceptual quality while doing worse on bits-per-dimension. Being able to say that last one, that the principled objective and the good-looking objective genuinely came apart, is what reads as having actually worked with these models.

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
