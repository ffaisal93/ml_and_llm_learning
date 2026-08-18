# Diffusion Models — Interview Grill

> 40 questions on diffusion models. Drill until you can answer 28+ cold.

---

## A. Foundations

**1. What's the basic idea of diffusion?**
Two processes. Forward: progressively add Gaussian noise to data over $T$ steps until it's pure noise. Reverse (learned): denoise step by step from noise back to data. Sample from the data distribution by running the reverse process from a random Gaussian.

> **Saying it out loud.** Think of a photograph slowly dissolving into TV static — that's the forward process, and it needs no learning at all, you just keep adding a little Gaussian noise until nothing is left. The whole model is learning to run that film backwards one frame at a time. Then to generate, you start from actual static and let it walk back to an image. The reason it works so much better than trying to conjure a picture in one shot is that each individual step is easy — "this is slightly noisy, clean it up a bit" — and you get a thousand of them. The tradeoff is right there: you need one network evaluation per step, which is why sampling is slow.

**2. Forward process equations?**
**Intuition**: add a tiny bit of Gaussian noise each step; the closed form lets us jump from clean image to step-$t$ noisy image *in one shot* — no need to iterate during training.

**Math**: per-step $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$. Closed form: $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t) I)$ where $\bar\alpha_t = \prod_s (1-\beta_s)$. Direct sample: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ — this is *the* identity used in training.

> **Saying it out loud.** Each forward step shrinks the image slightly and adds a little noise — the shrink factor is what keeps the total variance from blowing up. The beautiful part is that because Gaussians compose, you never have to actually iterate. There's a closed form that jumps straight from the clean image to step $t$: take $\sqrt{\bar\alpha_t}$ times the original plus $\sqrt{1-\bar\alpha_t}$ times a fresh Gaussian. That single identity is what makes training possible at all, and it's the one equation I'd make sure to write on the board.

**3. Why is the closed-form direct sampling important?**
During training, you don't iterate the forward process — you sample a random $t$ and a random $\varepsilon$, then directly compute $x_t$. Makes training tractable.

> **Saying it out loud.** Without the closed form, generating one training example at timestep 800 would mean running 800 sequential noise additions, and training would be hopeless. With it, you pick a random timestep, draw one noise vector, and construct the noisy image in a single line. That also means every training step is independent — no sequential dependency anywhere in training, so it parallelizes perfectly across a batch. The asymmetry is worth naming: training is fully parallel, sampling is strictly sequential, and that's the whole reason diffusion is cheap to train and expensive to run.

**4. What's the variance schedule?**
A function $\beta_t$ controlling noise per step. Linear (DDPM original): $\beta$ linear from $10^{-4}$ to $0.02$ over 1000 steps. Cosine (Nichol & Dhariwal): smoother, better for high-res images.

> **Saying it out loud.** The schedule decides how fast the image gets destroyed, and it matters more than it sounds. The original DDPM used a linear ramp from 0.0001 to 0.02 over a thousand steps. The problem people found is that a linear schedule destroys the image too early — by the time you're two-thirds through, it's already essentially pure noise, so the last third of your training steps are learning nothing useful. The cosine schedule fixes that by keeping information around longer, and it matters more at high resolution because there's more redundancy in the pixels to destroy.

**5. What's the reverse process?**
Learn $p_\theta(x_{t-1} \mid x_t)$. The model predicts a Gaussian's mean (and optionally variance) for the previous step. Standard parameterization: predict the noise $\varepsilon$.

> **Saying it out loud.** The reverse process is the only part that's learned. The key fact that makes it tractable is that when each forward step adds a small enough amount of noise, the reverse of that step is also approximately Gaussian — so the network only has to output a mean, not an arbitrary distribution. And in practice it doesn't even predict the mean directly; it predicts the noise that was added, and you algebraically recover the mean from that. If the steps were large, the true reverse would be multimodal and the Gaussian assumption would break, which is why you need many small steps rather than a few big ones.

---

## B. Training

**6. What's the simplified DDPM loss?**
$\mathcal{L} = \mathbb{E}_{t, x_0, \varepsilon}[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2]$. MSE between actual added noise and predicted noise. Drops weighted prefactors from the proper ELBO; works better empirically.

> **Saying it out loud.** After all the variational machinery, the loss you actually implement is embarrassingly simple: take a clean image, pick a random timestep, add a known amount of noise, and train the network to predict exactly the noise you added, with plain mean squared error. That's four lines of code. The subtlety worth mentioning is that the proper ELBO comes with per-timestep weights and DDPM just throws them away — the simplified, unweighted version trains better in practice. It's a case where the principled objective and the one that works came apart, and the empirical one won.

**7. Why predict $\varepsilon$ instead of $x_0$?**
Constant target scale across timesteps ($\varepsilon \sim \mathcal{N}(0, I)$). Better-conditioned loss. Empirically better than predicting $x_0$.

> **Saying it out loud.** It's a conditioning argument. If you predict the clean image, the difficulty of the task varies wildly with the timestep — at $t=1$ it's trivial, at $t=999$ you're asking the network to hallucinate an entire image from static, so the loss scale swings by orders of magnitude across timesteps and training is badly conditioned. The noise target is always a unit Gaussian regardless of $t$, so the target statistics never change and every timestep contributes comparably. There's a third option worth naming, $v$-prediction, which interpolates between the two and is what people use for high guidance scales and distillation.

**8. Connection to score matching?**
Predicting $\varepsilon$ is equivalent to predicting the score (gradient of log density) up to a scaling: $\varepsilon \approx -\sigma \cdot \nabla \log p_t(x)$. Diffusion models are score-based generative models.

> **Saying it out loud.** This is the connection that unified two research lines that looked completely different. The score is the gradient of the log density — it points in the direction of "more probable data" from wherever you are. And it turns out predicting the noise is the same thing up to a minus sign and a scale factor, which makes sense intuitively: the noise is exactly the direction you'd move to make the image more plausible. So Ho's DDPM and Song's score matching are the same model in different notation, and once you see that, the continuous-time SDE formulation and all the fancy ODE solvers become available to you.

**9. Walk through the diffusion loss derivation.**
**One-liner**: "It's an MSE on noise prediction — derived from the ELBO, but DDPM drops the per-timestep weights because empirically that works better."

**Whiteboard version**: Start with ELBO on $\log p_\theta(x_0)$ → decomposes into per-timestep KLs $\mathrm{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$. Both $q$ and $p_\theta$ are Gaussians with the same variance schedule, so each KL reduces to a weighted MSE between predicted and true noise. DDPM drops the weights → final loss is $\mathbb{E}_{t, x_0, \epsilon} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$.

> **Saying it out loud.** If they want the derivation, the shape is: you can't compute the data likelihood so you bound it with an ELBO, and because the forward process is a Markov chain the bound decomposes into one KL divergence per timestep. Then the piece of luck that makes everything collapse — both distributions in each KL are Gaussians with known, fixed variances, and the KL between two Gaussians with equal variance is just the squared distance between their means. Rewrite the means in terms of the noise and each term becomes a weighted MSE on noise prediction. DDPM then drops the weights because uniform weighting trains better. So: intractable likelihood, ELBO, per-step KLs, Gaussian-to-Gaussian, weighted MSE, drop the weights, done.

**10. What does the model architecture look like?**
UNet (DDPM, SD 1.x/2.x) or Diffusion Transformer (DiT, used in SD3, FLUX). Time $t$ injected via embedding (sinusoidal + MLP) added to layers. Conditioning $c$ injected via cross-attention.

> **Saying it out loud.** The network takes a noisy image and a timestep and outputs a prediction the same shape as the image, which is why UNets were the natural first choice — same in, same out, with skip connections preserving fine detail. The timestep goes in as a sinusoidal embedding, exactly like positional encoding in a transformer, pushed through a small MLP and added into every block, because the model genuinely needs to know how noisy its input is. Text conditioning enters through cross-attention. Since 2023 the field has been migrating to Diffusion Transformers, which treat the latent as a sequence of patches, for the same reason ViTs beat CNNs: transformers scale more predictably.

---

## C. Sampling

**11. How does DDPM sampling work?**
1. Sample $x_T \sim \mathcal{N}(0, I)$. 2. For $t$ from $T$ down to 1: predict $\hat\varepsilon$; compute mean $\mu$; add Gaussian noise. 3. Return $x_0$. $T = 1000$ typical → 1000 model forward passes per sample. Slow.

> **Saying it out loud.** Sampling is a loop. Start with pure noise, ask the model what noise it sees, subtract most of it, add back a little fresh noise, repeat. The extra noise you add back is what makes it a stochastic process rather than a deterministic descent, and it's what gives you diversity across samples. The number to say out loud is a thousand: DDPM's original sampler needs a thousand sequential network evaluations to produce one image, and they cannot be parallelized because each depends on the last. That's minutes per image, and it's the practical problem that everything in the next question exists to solve.

**12. What's DDIM?**
Deterministic sampler that allows skipping steps. Same trained model. 50–100 DDIM steps $\approx$ 1000 DDPM steps in quality. Standard for production.

> **Saying it out loud.** DDIM's insight is that you can define a *deterministic* reverse process that has the same marginals as the stochastic one — meaning the same trained model still works, no retraining. And once it's deterministic, you're solving an ODE rather than simulating an SDE, so you're allowed to take big steps. Fifty to a hundred steps get you roughly the quality that took a thousand before, a ten-to-twentyfold speedup for free. Determinism also gives you a bonus: the same seed always gives the same image, and you can interpolate smoothly in noise space, which is what makes image editing workflows possible.

**13. How can you sample faster than DDIM?**
DPM-Solver/DPM-Solver++ (~20 steps), Consistency Models (1–4 steps after distillation), Flow Matching (straighter trajectories, fewer steps).

> **Saying it out loud.** Three routes, and they attack different things. Better ODE solvers — DPM-Solver and friends — exploit the specific semi-linear structure of the diffusion ODE and get you to good samples in about twenty steps with no retraining at all, which is the cheapest win available. Distillation trains a student to take the shortcuts a teacher needed many steps for, which gets you down to one to four steps at some cost in diversity and fine detail. And flow matching changes the training objective so the trajectories from noise to data are straighter to begin with, meaning a coarse solver has less curvature to approximate. Modern systems combine them.

**14. What's the consistency model trick?**
Distill the diffusion process so a single forward pass goes from any $x_t$ directly to $x_0$. Trade some quality for ~100x sampling speedup.

> **Saying it out loud.** Consistency models train the network so that every point along a given trajectory maps to the same endpoint — that's the consistency property. If that holds, then from any noise level you can jump straight to the final image in one evaluation, because every point on the path already knows where it's going. That's roughly a hundredfold speedup and it's what makes real-time image generation possible. What you give up is diversity and fine detail: one-step samples are noticeably blurrier and less varied than fifty-step samples, which is why people often use two or four steps as the compromise.

**15. Why is sampling slow?**
Each timestep is one forward pass. $T = 1000$ native; 50–100 with DDIM. Cannot parallelize across timesteps (each depends on previous). Compute-bound at sampling time.

> **Saying it out loud.** The fundamental problem is that the reverse process is strictly sequential — step $t-1$ needs the output of step $t$, so no amount of hardware helps you within a single image. Contrast that with training, which is fully parallel because the closed-form forward process lets every example jump to any timestep independently. That asymmetry is diffusion's defining tradeoff versus GANs: a GAN generates in one forward pass and diffusion needs anywhere from twenty to a thousand, which is why so much research effort has gone into step reduction rather than quality.

---

## D. Conditioning and CFG

**16. How is conditioning injected?**
Cross-attention layers. Text embedding from CLIP/T5; image embedding for image-to-image; structural condition for ControlNet. The condition steers the denoising process toward conditioned outputs.

> **Saying it out loud.** Conditioning goes in through cross-attention: the image features are the queries and the text embedding provides keys and values, so at every layer and every denoising step the model can look back at the prompt and ask "what am I supposed to be making here?" It's not a one-time nudge at the start — the prompt is consulted continuously through the entire trajectory. The text encoder choice matters a lot; SD3 uses three of them, CLIP-L, CLIP-G, and T5-XXL, because CLIP embeddings are good at visual concepts and weak at compositional structure, which T5 handles better.

**17. What's classifier-free guidance (CFG)?**
Train jointly: conditional $\varepsilon_\theta(x_t, t, c)$ and unconditional $\varepsilon_\theta(x_t, t, \emptyset)$ (with $c$ dropped 10–20% of training). At sampling: $\hat\varepsilon_{\text{guided}} = \varepsilon_{\text{unc}} + w \cdot (\varepsilon_{\text{cond}} - \varepsilon_{\text{unc}})$. $w$ = guidance scale.

> **Saying it out loud.** Classifier-free guidance is the trick that made text-to-image actually follow the prompt. During training you randomly drop the caption maybe 10 to 20 percent of the time, so one network learns both the conditional and the unconditional model. At sampling you run it twice, once with the prompt and once without, and then extrapolate: take the unconditional prediction and push it past the conditional one along the difference between them. The cost is that every sampling step is now two forward passes, so you've doubled your inference bill for prompt adherence — and everyone pays it, which tells you how much it's worth.

**18. Why does CFG work?**
**The gap between conditional and unconditional predictions points toward the prompt — we just amplify that direction.** Mechanically: $\varepsilon_{\text{cond}} - \varepsilon_{\text{unc}}$ is a score-space vector toward the condition; multiplying by $w > 1$ pushes samples more strongly into the conditional distribution.

> **Saying it out loud.** The intuition is a tug of war. The unconditional prediction says "make this look like some image." The conditional says "make this look like an image of a corgi." Subtract them and what's left is a pure direction vector meaning "more corgi" with everything generic cancelled out. Then you don't just follow that direction, you overshoot it by a factor $w$. In score terms you're sampling from a distribution proportional to the conditional raised to the power $w$ — a sharpened version of the true conditional. Which is exactly why the named failure mode is oversaturation: crank $w$ past about 15 and you get blown-out contrast and every sample from the same prompt starts looking identical, because you've sharpened the distribution onto its modes.

**19. What's a typical guidance scale?**
$w = 7.5$ for Stable Diffusion. Higher: stronger condition adherence but oversaturated outputs and lower diversity. Lower: more diverse but condition may be ignored.

> **Saying it out loud.** 7.5 is the Stable Diffusion default and it's a genuinely tuned compromise. Below about 3, the model wanders off and ignores half your prompt. Above about 15, you get that overcooked, oversaturated look with blown highlights, and diversity collapses so every seed gives you nearly the same picture. The reason is that guidance is sharpening the conditional distribution, so a high scale concentrates all the probability on a few modes. It's a genuine precision-recall tradeoff — high guidance improves prompt fidelity and destroys sample variety — and models trained with flow matching or $v$-prediction generally tolerate higher scales before breaking.

**20. CFG vs classifier guidance?**
Classifier guidance (older): use a separate classifier's gradient to push samples toward class. Requires training the classifier. CFG: no separate classifier, just train cond+unc jointly. CFG dominates.

> **Saying it out loud.** Classifier guidance came first: train a separate image classifier, and during sampling nudge the image along the gradient of the classifier's log-probability for your target class. It works, but it has a nasty practical requirement — the classifier has to be trained on *noisy* images at every noise level, because that's what it sees during sampling, so you can't just grab an off-the-shelf model. Classifier-free guidance gets the same effect with no second network at all, just occasional caption dropout during training, and it generalizes to arbitrary text prompts rather than a fixed label set. That last point is why it completely displaced the older method.

---

## E. Latent diffusion

**21. What's latent diffusion?**
Run diffusion in a compressed latent space (via VAE encoder/decoder) instead of pixel space. 4–8x cheaper. Stable Diffusion's key innovation.

> **Saying it out loud.** Latent diffusion is the idea that made image generation run on a consumer GPU. Instead of denoising a 512-by-512 pixel image, you first squash it with a pretrained VAE encoder down to a 64-by-64 latent — an eightfold reduction per side, so about 48 times fewer values — run the entire diffusion process there, and decode once at the very end. Since attention and convolution costs scale with spatial size, that's a roughly 40 to 60 times reduction in compute per step. That single change is the difference between a research-lab model and Stable Diffusion running on a laptop.

**22. Why does latent diffusion work?**
Most "perceptual" content (semantics, textures) is captured in the latent. Diffusion in this space is much cheaper. Final quality is bounded by VAE reconstruction quality, which is fine in practice.

> **Saying it out loud.** It works because pixel space is enormously redundant, and most of that redundancy is imperceptible detail rather than content. The VAE strips out the high-frequency noise that a diffusion model would otherwise spend most of its capacity modeling, leaving the semantic structure that actually matters. So the diffusion model gets to work on the hard part and skip the busywork. The tradeoff you must name is that the VAE is a hard ceiling — the diffusion model can never produce anything the decoder can't reconstruct, which is exactly why Stable Diffusion is famously bad at small text and fine facial detail: the VAE destroyed those before diffusion ever started.

**23. How is the VAE trained?**
Separately, before diffusion training. Encoder + decoder reconstruct images. Includes adversarial loss (LPIPS, GAN-style) for perceptual quality. Frozen during diffusion training.

> **Saying it out loud.** The VAE is trained first, on its own, and then frozen — two completely separate stages, which is what keeps the pipeline manageable. And it isn't trained with plain reconstruction loss, because pure MSE gives you blurry decodes. It uses a perceptual loss like LPIPS plus a GAN discriminator, so the decoder is rewarded for producing something that *looks* right rather than something that's pixel-wise close. The KL term is deliberately weighted very low, so it's really more of a regularized autoencoder than a true VAE — the goal is a good compressed space, not a well-behaved probabilistic prior.

**24. What's SDXL's improvement over SD 1.5?**
Larger model. Refiner stage (second diffusion model for high-res details). Multiple text encoders (CLIP-L + CLIP-G concatenated). Better noise schedule for high-res.

> **Saying it out loud.** SDXL is mostly scale plus a set of practical fixes. Bigger UNet, around 2.6 billion parameters versus under a billion. Two text encoders concatenated, because one CLIP model's text understanding was the bottleneck on prompt adherence. A second refiner model that takes the base output and adds high-frequency detail. And the underrated one: conditioning on image size and crop coordinates during training, which let them train on non-square images without the model learning that everything should be centered and cropped. That last trick fixed a lot of the weird framing artifacts in SD 1.5.

**25. What's SD 3's improvement over SDXL?**
DiT-based architecture. Flow matching instead of pure DDPM. Three text encoders (CLIP-L, CLIP-G, T5-XXL). Multimodal DiT (joint image-text attention).

> **Saying it out loud.** SD3 changes three things at once. The backbone becomes a transformer instead of a UNet, because transformers scale more predictably. The objective becomes rectified flow matching, which gives straighter noise-to-data paths and therefore needs fewer sampling steps. And it adds T5-XXL as a third text encoder, which is what fixed the long-standing weakness on compositional prompts and rendering readable text — CLIP embeddings are good at objects and bad at "the red cube *on top of* the blue sphere." The architectural detail worth naming is MMDiT, where image and text tokens attend jointly with separate weights rather than text only entering through cross-attention.

---

## F. Architecture: UNet vs DiT

**26. What's a UNet for diffusion?**
Convolutional U-shape. Down-sampling encoder + up-sampling decoder + skip connections. Cross-attention layers for conditioning. Standard for diffusion until ~2023.

> **Saying it out loud.** A UNet downsamples the image through a few resolution levels, does its heaviest processing at the smallest resolution, then upsamples back, with skip connections carrying fine detail across from the encoder side to the decoder side. That shape is a natural fit for denoising because input and output are the same size and you need both global structure and local detail. Attention blocks get inserted at the lower resolutions, where the sequence is short enough to afford them, and conditioning enters there through cross-attention. Its limitation is that all the architectural choices — where to put attention, how many levels — are hand-designed, which is exactly what makes it hard to scale.

**27. What's DiT?**
Diffusion Transformer (Peebles & Xie 2022). Replace UNet with transformer over patches (like ViT). Better scaling. Used in SD3, FLUX.

> **Saying it out loud.** DiT throws away the convolutional U-shape and treats the latent as a flat sequence of patches, exactly like a ViT treats an image. Uniform blocks, one resolution throughout, no skip connections. Timestep and conditioning are injected through adaptive layer norm — the conditioning predicts scale and shift parameters for each block. The finding that made it matter is that DiT quality improves smoothly and predictably with compute, measured in GFLOPs, in a way the UNet did not. That's a scaling law, and once you have one you know how to spend your next training budget.

**28. Why is DiT replacing UNet?**
Transformers scale predictably with parameters/data. Conv-UNet's hand-crafted inductive biases limit scaling. As models grow, DiT-style dominates. Same lesson as ViT replacing CNNs in vision.

> **Saying it out loud.** It's the bitter lesson again. A UNet's inductive biases — locality, the multi-resolution hierarchy — are genuinely helpful when data and compute are limited, and they become a constraint once they aren't. A transformer has almost no built-in assumptions, so it needs more data, and in exchange it keeps improving as you scale where the UNet plateaus. Add the practical argument: all the infrastructure for training huge transformers already exists, so you get flash attention, tensor parallelism, and every optimization the LLM world built, for free. That's why SD3 and FLUX are both DiTs.

---

## G. Recent: Flow Matching

**29. What's flow matching?**
Train a velocity field $v_\theta(x, t)$ that transforms noise → data along a continuous path. Learn the velocity by regressing on the path's tangent vector.

> **Saying it out loud.** Flow matching asks a simpler question than diffusion does. Draw a straight line from a noise sample to a data sample, pick a random point on that line, and train the network to predict the direction you'd travel along it — the velocity. That's it, and the loss is again just MSE. There's no noise schedule to design, no forward Markov chain, no ELBO. Sampling means starting at noise and integrating that velocity field forward with an ODE solver. It's a cleaner formulation of the same underlying idea.

**30. Why flow matching over diffusion?**
Straighter trajectories → fewer sampling steps for equivalent quality. Conceptually cleaner training objective. SD 3, FLUX use it.

> **Saying it out loud.** The whole argument is trajectory curvature. Diffusion's path from noise to data curves through the space, and a solver taking large steps along a curved path accumulates error fast, which is why you need many steps. Flow matching with a linear interpolation path is trained toward straight trajectories, and a straight line is exactly what a coarse solver approximates perfectly — in the ideal limit you could jump the whole way in one step. In practice that translates to good samples in something like 20 to 30 steps instead of 50 to 100. Same network, same MSE loss, different geometry.

**31. Are flow matching models the same as diffusion models?**
Mathematically related — both predict velocities (or noise/score, equivalently). The training objective and sampling procedure differ, but the trained network is similar.

> **Saying it out loud.** They're the same family viewed through different lenses. Diffusion in continuous time is an SDE with a corresponding probability-flow ODE, and flow matching is directly training the vector field of an ODE. Noise prediction, score prediction, and velocity prediction are all related by simple algebraic reparameterizations, so you can convert between them. The real differences are practical: flow matching uses a straight interpolation path instead of the variance-preserving one, which gives less curvature, and it has no schedule to tune. Calling them different model classes overstates it — it's a change of parameterization with real practical consequences.

---

## H. Other models and applications

**32. Diffusion vs GANs?**
Diffusion: stable training, no mode collapse, slow sampling. GANs: fast sampling, mode collapse risk, harder to train. Diffusion has largely won for image generation.

> **Saying it out loud.** GANs generate an image in one forward pass, so they're a hundred to a thousand times faster at inference, and that's their only remaining advantage. Everything else favors diffusion. A GAN is a minimax game between two networks, which is unstable, sensitive to hyperparameters, and prone to mode collapse where the generator finds a handful of outputs that fool the discriminator and stops exploring. Diffusion is a plain regression problem — predict the noise, minimize MSE — so training is stable, it scales monotonically, and it covers the full data distribution because the maximum-likelihood-style objective punishes leaving modes uncovered. The tradeoff is spelled out in one line: mode coverage and stable training, paid for with sequential sampling.

**33. Diffusion in NLP?**
Mostly research-stage. Text is discrete; diffusion is naturally continuous. SEDD, Diffusion-LM are recent attempts. Autoregressive models still dominate text generation.

> **Saying it out loud.** The mismatch is that diffusion is built on adding continuous Gaussian noise, and you can't add a little bit of Gaussian noise to the word "cat." The workarounds either diffuse in a continuous embedding space and round at the end, or define discrete corruption processes like masking, which is what SEDD does. There's real appeal — you'd get parallel generation of a whole sequence instead of one token at a time, plus natural infilling. But autoregressive models remain far ahead on quality per unit of compute, largely because next-token prediction has an exact likelihood and a decade of tooling behind it. Worth watching, not yet worth betting on.

**34. Video diffusion?**
Apply diffusion to video frames jointly (3D UNet) or extend image diffusion temporally (Sora, VideoCrafter). Memory-intensive; many active research directions.

> **Saying it out loud.** Video is the same machinery with a time axis bolted on, and the whole difficulty is that memory scales with frames times resolution. The dominant approach is spatiotemporal attention — factorized so you attend within a frame and then across frames separately, because full 3D attention is quadratic in the total token count and unaffordable. Sora's framing is the useful one to cite: treat video as a sequence of spacetime patches and run a DiT over them, which unifies images and video as the same problem at different patch counts. The failure mode that dominates the field is temporal consistency — objects that morph, flicker, or fail to persist across frames.

**35. ControlNet — what does it do?**
Add an auxiliary network conditioned on structural input (depth, segmentation, edges). Injected via additional cross-attention. Lightweight; widely used for spatial control.

> **Saying it out loud.** ControlNet gives you spatial control that text can't express — you can say "a knight in a forest," but you can't say "with his arm at exactly this angle." So you clone the encoder half of the UNet, feed the copy a structural map like edges, depth, or a pose skeleton, and add its outputs into the frozen original through zero-initialized convolutions. The zero initialization is the essential detail: at the start of training the ControlNet contributes exactly nothing, so the base model's behavior is untouched and the control fades in gradually instead of wrecking it. The base weights stay frozen, so you can train one on a modest dataset and swap them freely.

**36. DreamBooth / LoRA for diffusion?**
Fine-tune diffusion models on small datasets to add new concepts (a person, a style). LoRA dominates for personalization (small adapter, easy to share).

> **Saying it out loud.** Both teach the model a new concept from a handful of images — your face, your brand's style. DreamBooth fine-tunes the whole model and binds the concept to a rare token, with a prior-preservation loss to stop the model from deciding that all people now look like you, which is the named failure mode called language drift. LoRA instead freezes everything and trains small low-rank update matrices on the attention layers. LoRA won on logistics rather than quality: a full DreamBooth checkpoint is gigabytes, a LoRA is a few megabytes, so you can share them, stack several at once, and adjust their strength at inference.

---

## I. Subtleties

**37. What's variance-preserving (VP) vs variance-exploding (VE)?**
VP: $\mathrm{Var}(x_t) \approx 1$ for all $t$ (DDPM standard). VE: $\mathrm{Var}(x_t)$ grows with $t$ (Song & Ermon original). Different parameterizations of the same idea. VP is more common in modern usage.

> **Saying it out loud.** These are two conventions for the same process. Variance-preserving scales the signal down as it adds noise, so the total variance stays around one at every timestep — that's DDPM, and it's convenient because your network always sees inputs on roughly the same scale. Variance-exploding leaves the signal alone and just piles noise on top, so the variance grows to something enormous by the end — that's the original score-matching formulation. Song's continuous-time paper showed they're both SDEs differing in their drift and diffusion coefficients, and you can convert between them. VP dominates in practice because constant input scale is friendlier to a neural network.

**38. Why does the simplified DDPM loss work better than the weighted ELBO?**
The ELBO prefactor weights early timesteps very heavily (where noise is small and predictions are easy) and late timesteps very lightly. Empirically, this distorts training. The simplified loss treats all timesteps equally and works better.

> **Saying it out loud.** The honest answer is that the ELBO is optimizing the wrong thing for what we care about. Its weights pour effort into low-noise timesteps, where the task is nearly trivial and the remaining error is imperceptible high-frequency detail, and it barely weights the high-noise steps where the model decides the global structure of the image. Since we're judging on perceptual quality rather than likelihood, that's backwards. Dropping the weights entirely reweights effort toward the steps that determine whether the picture makes sense. It's a clean example of the likelihood-versus-perceptual-quality gap, and it's why diffusion models with worse bits-per-dim can produce much better-looking images.

**39. What's "image super-resolution" via diffusion?**
Train a diffusion model conditioned on a low-resolution image. The reverse process generates the high-resolution image. Used in Imagen, SDXL refiner.

> **Saying it out loud.** Super-resolution is just conditional diffusion where the condition is the low-resolution image, usually upsampled and concatenated to the noisy input. The model learns to invent plausible high-frequency detail consistent with what it's given. The reason it's architecturally important is cascading: rather than training one enormous model at 1024 pixels, Imagen generates at 64 and then runs two super-resolution diffusion models to reach 1024, which is far cheaper because the expensive semantic work happens at the smallest resolution. The failure mode to name is hallucinated detail — the model will happily invent text or facial features that were never in the source, because it's generating, not recovering.

**40. What's classifier guidance (older)?**
Use $\nabla \log p(c \mid x_t)$ from a separate classifier to push samples toward the condition. Requires training a classifier on noisy data. Replaced by CFG.

> **Saying it out loud.** Classifier guidance was the first method that got diffusion models to actually respect a condition, and it's worth knowing because it explains why CFG looks the way it does. You add the gradient of a classifier's log-probability to the score during sampling, which mathematically shifts you toward a sharpened conditional distribution. Two problems killed it. The classifier has to be trained on noisy images at every noise level, so no off-the-shelf model works. And it only handles whatever fixed label set the classifier knows, which is useless for open-ended text prompts. CFG achieves the same distributional sharpening with one network and arbitrary conditions.

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
