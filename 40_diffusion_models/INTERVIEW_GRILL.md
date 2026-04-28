# Diffusion Models — Interview Grill

> 40 questions on diffusion models. Drill until you can answer 28+ cold.

---

## A. Foundations

**1. What's the basic idea of diffusion?**
Two processes. Forward: progressively add Gaussian noise to data over $T$ steps until it's pure noise. Reverse (learned): denoise step by step from noise back to data. Sample from the data distribution by running the reverse process from a random Gaussian.

**2. Forward process equations?**
**Intuition**: add a tiny bit of Gaussian noise each step; the closed form lets us jump from clean image to step-$t$ noisy image *in one shot* — no need to iterate during training.

**Math**: per-step $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$. Closed form: $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t) I)$ where $\bar\alpha_t = \prod_s (1-\beta_s)$. Direct sample: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ — this is *the* identity used in training.

**3. Why is the closed-form direct sampling important?**
During training, you don't iterate the forward process — you sample a random $t$ and a random $\varepsilon$, then directly compute $x_t$. Makes training tractable.

**4. What's the variance schedule?**
A function $\beta_t$ controlling noise per step. Linear (DDPM original): $\beta$ linear from $10^{-4}$ to $0.02$ over 1000 steps. Cosine (Nichol & Dhariwal): smoother, better for high-res images.

**5. What's the reverse process?**
Learn $p_\theta(x_{t-1} \mid x_t)$. The model predicts a Gaussian's mean (and optionally variance) for the previous step. Standard parameterization: predict the noise $\varepsilon$.

---

## B. Training

**6. What's the simplified DDPM loss?**
$\mathcal{L} = \mathbb{E}_{t, x_0, \varepsilon}[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2]$. MSE between actual added noise and predicted noise. Drops weighted prefactors from the proper ELBO; works better empirically.

**7. Why predict $\varepsilon$ instead of $x_0$?**
Constant target scale across timesteps ($\varepsilon \sim \mathcal{N}(0, I)$). Better-conditioned loss. Empirically better than predicting $x_0$.

**8. Connection to score matching?**
Predicting $\varepsilon$ is equivalent to predicting the score (gradient of log density) up to a scaling: $\varepsilon \approx -\sigma \cdot \nabla \log p_t(x)$. Diffusion models are score-based generative models.

**9. Walk through the diffusion loss derivation.**
**One-liner**: "It's an MSE on noise prediction — derived from the ELBO, but DDPM drops the per-timestep weights because empirically that works better."

**Whiteboard version**: Start with ELBO on $\log p_\theta(x_0)$ → decomposes into per-timestep KLs $\mathrm{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$. Both $q$ and $p_\theta$ are Gaussians with the same variance schedule, so each KL reduces to a weighted MSE between predicted and true noise. DDPM drops the weights → final loss is $\mathbb{E}_{t, x_0, \epsilon} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$.

**10. What does the model architecture look like?**
UNet (DDPM, SD 1.x/2.x) or Diffusion Transformer (DiT, used in SD3, FLUX). Time $t$ injected via embedding (sinusoidal + MLP) added to layers. Conditioning $c$ injected via cross-attention.

---

## C. Sampling

**11. How does DDPM sampling work?**
1. Sample $x_T \sim \mathcal{N}(0, I)$. 2. For $t$ from $T$ down to 1: predict $\hat\varepsilon$; compute mean $\mu$; add Gaussian noise. 3. Return $x_0$. $T = 1000$ typical → 1000 model forward passes per sample. Slow.

**12. What's DDIM?**
Deterministic sampler that allows skipping steps. Same trained model. 50–100 DDIM steps $\approx$ 1000 DDPM steps in quality. Standard for production.

**13. How can you sample faster than DDIM?**
DPM-Solver/DPM-Solver++ (~20 steps), Consistency Models (1–4 steps after distillation), Flow Matching (straighter trajectories, fewer steps).

**14. What's the consistency model trick?**
Distill the diffusion process so a single forward pass goes from any $x_t$ directly to $x_0$. Trade some quality for ~100x sampling speedup.

**15. Why is sampling slow?**
Each timestep is one forward pass. $T = 1000$ native; 50–100 with DDIM. Cannot parallelize across timesteps (each depends on previous). Compute-bound at sampling time.

---

## D. Conditioning and CFG

**16. How is conditioning injected?**
Cross-attention layers. Text embedding from CLIP/T5; image embedding for image-to-image; structural condition for ControlNet. The condition steers the denoising process toward conditioned outputs.

**17. What's classifier-free guidance (CFG)?**
Train jointly: conditional $\varepsilon_\theta(x_t, t, c)$ and unconditional $\varepsilon_\theta(x_t, t, \emptyset)$ (with $c$ dropped 10–20% of training). At sampling: $\hat\varepsilon_{\text{guided}} = \varepsilon_{\text{unc}} + w \cdot (\varepsilon_{\text{cond}} - \varepsilon_{\text{unc}})$. $w$ = guidance scale.

**18. Why does CFG work?**
**The gap between conditional and unconditional predictions points toward the prompt — we just amplify that direction.** Mechanically: $\varepsilon_{\text{cond}} - \varepsilon_{\text{unc}}$ is a score-space vector toward the condition; multiplying by $w > 1$ pushes samples more strongly into the conditional distribution.

**19. What's a typical guidance scale?**
$w = 7.5$ for Stable Diffusion. Higher: stronger condition adherence but oversaturated outputs and lower diversity. Lower: more diverse but condition may be ignored.

**20. CFG vs classifier guidance?**
Classifier guidance (older): use a separate classifier's gradient to push samples toward class. Requires training the classifier. CFG: no separate classifier, just train cond+unc jointly. CFG dominates.

---

## E. Latent diffusion

**21. What's latent diffusion?**
Run diffusion in a compressed latent space (via VAE encoder/decoder) instead of pixel space. 4–8x cheaper. Stable Diffusion's key innovation.

**22. Why does latent diffusion work?**
Most "perceptual" content (semantics, textures) is captured in the latent. Diffusion in this space is much cheaper. Final quality is bounded by VAE reconstruction quality, which is fine in practice.

**23. How is the VAE trained?**
Separately, before diffusion training. Encoder + decoder reconstruct images. Includes adversarial loss (LPIPS, GAN-style) for perceptual quality. Frozen during diffusion training.

**24. What's SDXL's improvement over SD 1.5?**
Larger model. Refiner stage (second diffusion model for high-res details). Multiple text encoders (CLIP-L + CLIP-G concatenated). Better noise schedule for high-res.

**25. What's SD 3's improvement over SDXL?**
DiT-based architecture. Flow matching instead of pure DDPM. Three text encoders (CLIP-L, CLIP-G, T5-XXL). Multimodal DiT (joint image-text attention).

---

## F. Architecture: UNet vs DiT

**26. What's a UNet for diffusion?**
Convolutional U-shape. Down-sampling encoder + up-sampling decoder + skip connections. Cross-attention layers for conditioning. Standard for diffusion until ~2023.

**27. What's DiT?**
Diffusion Transformer (Peebles & Xie 2022). Replace UNet with transformer over patches (like ViT). Better scaling. Used in SD3, FLUX.

**28. Why is DiT replacing UNet?**
Transformers scale predictably with parameters/data. Conv-UNet's hand-crafted inductive biases limit scaling. As models grow, DiT-style dominates. Same lesson as ViT replacing CNNs in vision.

---

## G. Recent: Flow Matching

**29. What's flow matching?**
Train a velocity field $v_\theta(x, t)$ that transforms noise → data along a continuous path. Learn the velocity by regressing on the path's tangent vector.

**30. Why flow matching over diffusion?**
Straighter trajectories → fewer sampling steps for equivalent quality. Conceptually cleaner training objective. SD 3, FLUX use it.

**31. Are flow matching models the same as diffusion models?**
Mathematically related — both predict velocities (or noise/score, equivalently). The training objective and sampling procedure differ, but the trained network is similar.

---

## H. Other models and applications

**32. Diffusion vs GANs?**
Diffusion: stable training, no mode collapse, slow sampling. GANs: fast sampling, mode collapse risk, harder to train. Diffusion has largely won for image generation.

**33. Diffusion in NLP?**
Mostly research-stage. Text is discrete; diffusion is naturally continuous. SEDD, Diffusion-LM are recent attempts. Autoregressive models still dominate text generation.

**34. Video diffusion?**
Apply diffusion to video frames jointly (3D UNet) or extend image diffusion temporally (Sora, VideoCrafter). Memory-intensive; many active research directions.

**35. ControlNet — what does it do?**
Add an auxiliary network conditioned on structural input (depth, segmentation, edges). Injected via additional cross-attention. Lightweight; widely used for spatial control.

**36. DreamBooth / LoRA for diffusion?**
Fine-tune diffusion models on small datasets to add new concepts (a person, a style). LoRA dominates for personalization (small adapter, easy to share).

---

## I. Subtleties

**37. What's variance-preserving (VP) vs variance-exploding (VE)?**
VP: $\mathrm{Var}(x_t) \approx 1$ for all $t$ (DDPM standard). VE: $\mathrm{Var}(x_t)$ grows with $t$ (Song & Ermon original). Different parameterizations of the same idea. VP is more common in modern usage.

**38. Why does the simplified DDPM loss work better than the weighted ELBO?**
The ELBO prefactor weights early timesteps very heavily (where noise is small and predictions are easy) and late timesteps very lightly. Empirically, this distorts training. The simplified loss treats all timesteps equally and works better.

**39. What's "image super-resolution" via diffusion?**
Train a diffusion model conditioned on a low-resolution image. The reverse process generates the high-resolution image. Used in Imagen, SDXL refiner.

**40. What's classifier guidance (older)?**
Use $\nabla \log p(c \mid x_t)$ from a separate classifier to push samples toward the condition. Requires training a classifier on noisy data. Replaced by CFG.

---

## Quick fire

**41.** *DDPM paper?* Ho, Jain, Abbeel 2020.
**42.** *DDIM paper?* Song et al. 2021.
**43.** *Stable Diffusion paper?* Rombach et al. 2022.
**44.** *DiT paper?* Peebles & Xie 2022.
**45.** *CFG scale typical?* $7.5$ for SD.

---

## Self-grading

If you can't answer 1-15, you don't know diffusion. If you can't answer 16-30, you'll struggle on generative-modeling interviews. If you can't answer 31-45, frontier-lab interviews on diffusion will go past you.

Aim for 28+/45 cold.
