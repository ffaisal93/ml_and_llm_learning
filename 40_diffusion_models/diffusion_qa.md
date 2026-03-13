# Diffusion Models: Comprehensive Interview Q&A

## Q1: What are diffusion models? How do they work?

**Answer:**

**Diffusion Models:**
- Generative models that learn to reverse a gradual noising process
- Work by iteratively removing noise from data, starting from pure noise
- Achieve state-of-the-art results in image generation (DALL-E, Stable Diffusion)

**How They Work:**

**1. Forward Process (Fixed):**
- Gradually add Gaussian noise to data
- q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
- After T steps, data becomes pure noise N(0, I)

**2. Reverse Process (Learned):**
- Learn to remove noise step by step
- p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
- Neural network predicts how to denoise

**3. Training:**
- Predict the noise that was added
- Loss: L = E[||ε - ε_θ(x_t, t)||²]
- Model learns: given noisy data x_t and timestep t, predict noise ε

**4. Generation:**
- Start from pure noise x_T ~ N(0, I)
- Iteratively apply reverse process: x_T → x_{T-1} → ... → x_0
- End with clean, generated sample

**Key Insight:**
- Break down complex generation into many simple denoising steps
- Each step only needs to remove small amount of noise
- Much easier to learn than generating directly

---

## Q2: What is the mathematical formulation of diffusion models?

**Answer:**

**Forward Diffusion Process:**

Given data x₀, define sequence x₁, x₂, ..., x_T where:
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
```

Where:
- β_t is variance schedule (0 < β_t < 1)
- √(1-β_t) preserves signal
- β_t adds noise

**Closed-Form Expression:**

Can sample x_t directly from x₀:
```
q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
```

Where:
- α_t = 1 - β_t
- ᾱ_t = ∏_{s=1}^t α_s (cumulative product)

**Reverse Diffusion Process:**

Learn to reverse the process:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**Training Objective:**

Predict noise that was added:
```
L = E_{t,x_0,ε} [||ε - ε_θ(x_t, t)||²]
```

Where:
- x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε (noisy data)
- ε_θ(x_t, t) is predicted noise
- ε is actual noise

**Sampling:**

Given predicted noise ε_θ, compute:
```
μ_θ = (1/√(α_t))(x_t - (β_t/√(1-ᾱ_t))ε_θ)
x_{t-1} ~ N(μ_θ, Σ_t)
```

---

## Q3: How do you train a diffusion model?

**Answer:**

**Training Algorithm:**

**1. Setup:**
- Define variance schedule β_t (linear or cosine)
- Precompute α_t, ᾱ_t for efficiency

**2. Training Loop:**
```
For each batch:
  a. Sample data: x_0 ~ q(x_0)
  b. Sample timestep: t ~ Uniform({1, 2, ..., T})
  c. Sample noise: ε ~ N(0, I)
  d. Create noisy data: x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
  e. Predict noise: ε_pred = ε_θ(x_t, t)
  f. Compute loss: L = ||ε - ε_pred||²
  g. Update: θ ← θ - α∇_θ L
```

**Key Points:**
- Randomly sample timesteps (not sequential)
- Model sees all noise levels during training
- Simple MSE loss (predict noise)

**Best Practices:**
- Learning rate: 1e-4 to 1e-3
- Use learning rate scheduling (cosine annealing)
- Gradient clipping (norm = 1.0)
- Monitor loss and generate samples during training
- More timesteps = better quality but slower

**Variance Schedule:**
- Linear: β_t = (β_max - β_min) * (t/T) + β_min
- Cosine: ᾱ_t = cos²(π/2 * (t/T)) (often better)

---

## Q4: How do you generate samples from a diffusion model?

**Answer:**

**Generation Algorithm:**

**1. Start from Noise:**
```
x_T ~ N(0, I)  # Pure noise
```

**2. Reverse Diffusion:**
```
For t = T, T-1, ..., 1:
  a. Predict noise: ε_t = ε_θ(x_t, t)
  b. Compute mean: μ_t = (1/√(α_t))(x_t - (β_t/√(1-ᾱ_t))ε_t)
  c. Sample: x_{t-1} ~ N(μ_t, Σ_t)
```

**3. Return:**
```
x_0  # Generated sample
```

**Step-by-Step:**

**At step t:**
- Model predicts noise ε_t that was added
- Use this to compute predicted mean μ_t of x_{t-1}
- Sample x_{t-1} from N(μ_t, Σ_t)
- This gives slightly less noisy version
- Repeat until clean data x_0

**Variance:**
- Can be fixed: Σ_t = β_t I
- Or learned: Σ_t = Σ_θ(x_t, t)

**Conditional Generation:**
- Condition on text, class, etc.
- Model becomes: ε_θ(x_t, t, c)
- Same process but with conditioning

---

## Q5: What are discrete diffusion models? How do they work for NLP?

**Answer:**

**The Challenge:**
- Standard diffusion works on continuous data (images)
- Text is discrete (tokens), need adaptation

**Discrete Forward Process:**

Instead of Gaussian noise, use transition matrix:
```
q(x_t | x_{t-1}) = Categorical(x_t; Q_t x_{t-1})
```

Where Q_t is transition matrix defining how tokens are corrupted.

**Common Approaches:**

**1. Absorbing State:**
- Have special [MASK] token
- At each step, tokens transition to [MASK] with probability β_t
- After T steps, all tokens become [MASK]

**2. Uniform Transition:**
- Tokens can transition to any other token uniformly
- More general but harder to learn

**Discrete Reverse Process:**

Learn to predict original token:
```
p_θ(x_{t-1} | x_t) = Categorical(x_{t-1}; p_θ(x_t, t))
```

Where p_θ(x_t, t) is probability distribution over vocabulary.

**Training:**
- Loss: Cross-entropy between predicted and original tokens
- L = E[-log p_θ(x_0 | x_t, t)]

**Advantages for NLP:**
- Non-autoregressive (can generate in parallel)
- Better for editing tasks (text inpainting)
- More flexible control

**Example: Text Inpainting**
- Mask some tokens: "The [MASK] sat on the [MASK]"
- Use reverse diffusion to fill in masked tokens
- Better than autoregressive for editing

---

## Q6: How do you evaluate diffusion models?

**Answer:**

**For Images:**

**1. FID (Frechet Inception Distance):**
- Measures quality and diversity
- Lower is better
- Compares feature distributions from Inception network

**2. IS (Inception Score):**
- Measures quality and diversity
- Higher is better (typically 1-10)
- Based on Inception network predictions

**3. Reconstruction Error:**
- Test if model can recover original from noisy version
- Lower is better
- Measures denoising capability

**For Text:**

**1. BLEU Score:**
- Measures n-gram overlap with reference
- Higher is better (0-1)
- Standard NLP metric

**2. Perplexity:**
- Measures how well model predicts tokens
- Lower is better
- Standard language modeling metric

**3. Diversity Metrics:**
- Distinct-n: Ratio of unique n-grams
- Self-BLEU: Average BLEU between generated samples
- Higher distinct = more diverse

**Diffusion-Specific:**

**1. Denoising Accuracy:**
- Test accuracy at each timestep
- Measures how well model denoises
- Should improve with training

**2. Sample Quality:**
- Visual inspection (for images)
- Human evaluation (for text)
- Compare with real data

**Best Practices:**
- Use multiple metrics
- Compare with baselines
- Generate samples during training
- Monitor metrics over time

---

## Q7: What are the advantages and disadvantages of diffusion models?

**Answer:**

**Advantages:**

**1. High Quality:**
- State-of-the-art results in image generation
- Better than GANs in many cases
- Very realistic samples

**2. Stable Training:**
- More stable than GANs
- No mode collapse
- Simple loss function (MSE)

**3. Flexible:**
- Can condition on various inputs
- Good for editing tasks
- Non-autoregressive (for discrete)

**4. Theoretical Foundation:**
- Well-grounded in probability theory
- Clear mathematical formulation
- Understandable process

**Disadvantages:**

**1. Slow Generation:**
- Requires many steps (1000-4000)
- Much slower than GANs or autoregressive models
- Each step requires forward pass

**2. Memory:**
- Need to store intermediate states
- Can be memory intensive
- Especially for high-resolution images

**3. Training:**
- Can be slow (many timesteps)
- Need to sample random timesteps
- More complex than some methods

**4. Discrete Data:**
- Need special handling for discrete data
- Transition matrices can be complex
- Less natural than continuous

**Comparison:**

| Aspect | Diffusion | GANs | Autoregressive |
|--------|-----------|------|----------------|
| Quality | Excellent | Very Good | Good |
| Speed | Slow | Fast | Medium |
| Training | Stable | Unstable | Stable |
| Flexibility | High | Medium | Low |

---

## Q8: What are use cases of diffusion models in NLP?

**Answer:**

**1. Non-Autoregressive Text Generation:**
- Generate all tokens in parallel
- Faster than autoregressive models
- Better for controlled generation

**2. Text Inpainting:**
- Fill in masked tokens
- Edit specific parts of text
- Example: "The [MASK] sat on the [MASK]" → "The cat sat on the mat"

**3. Text-to-Image:**
- DALL-E, Stable Diffusion
- Generate images from text descriptions
- Multimodal understanding

**4. Text Editing:**
- Style transfer
- Paraphrasing
- Rewriting with constraints

**5. Controllable Generation:**
- Generate with specific attributes
- Control length, style, topic
- More flexible than autoregressive

**6. Text Completion:**
- Complete partial text
- Better than simple next-token prediction
- Can consider full context

**7. Data Augmentation:**
- Generate training data
- Improve model robustness
- Handle rare patterns

**8. Multimodal Tasks:**
- Text-to-image
- Image-to-text
- Cross-modal generation

**Industry Examples:**
- DALL-E: Text-to-image generation
- Stable Diffusion: Open-source text-to-image
- Research: Non-autoregressive text generation
- Editing tools: Text inpainting and rewriting

---

## Q9: Compare diffusion models with autoregressive models (GPT) for text generation.

**Answer:**

**Generation Process:**

**Autoregressive (GPT):**
- Generate left-to-right, one token at a time
- Each token depends on all previous tokens
- Sequential: t₁ → t₂ → t₃ → ...

**Diffusion:**
- Generate all tokens in parallel (discrete diffusion)
- Iteratively refine all tokens together
- Parallel: All tokens refined simultaneously

**Advantages:**

**Autoregressive:**
- Faster single-pass generation
- Simpler implementation
- Better for long sequences
- More established for text

**Diffusion:**
- Non-autoregressive (parallel)
- Better for editing tasks
- More flexible control
- Can edit specific parts

**Disadvantages:**

**Autoregressive:**
- Sequential (can't parallelize)
- Hard to edit specific parts
- Less flexible control

**Diffusion:**
- Slower (many iterations)
- More complex
- Less established for text
- May need more training data

**When to Use:**

**Autoregressive:**
- Standard text generation
- Long sequences
- When speed is important
- Established use cases

**Diffusion:**
- Text editing/inpainting
- Controlled generation
- When need parallel generation
- Research/experimental

**Current State:**
- Autoregressive (GPT) dominates text generation
- Diffusion better for images
- Discrete diffusion promising for text
- Active area of research

---

## Q10: How do you implement a simple diffusion model from scratch?

**Answer:**

**Key Components:**

**1. Variance Schedule:**
```python
def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)
```

**2. Forward Diffusion:**
```python
def q_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x_0)
    return sqrt_alphas_cumprod[t] * x_0 + sqrt_one_minus_alphas_cumprod[t] * noise
```

**3. Model (Noise Predictor):**
```python
class DiffusionModel(nn.Module):
    def forward(self, x_t, t):
        # Predict noise given noisy data and timestep
        return noise_pred
```

**4. Training:**
```python
# Sample timestep and noise
t = torch.randint(0, timesteps, (batch_size,))
noise = torch.randn_like(x_0)
x_t = q_sample(x_0, t, ...)

# Predict and compute loss
noise_pred = model(x_t, t)
loss = F.mse_loss(noise_pred, noise)
```

**5. Sampling:**
```python
# Start from noise
x = torch.randn(shape)

# Reverse diffusion
for t in reversed(range(timesteps)):
    noise_pred = model(x, t)
    x = denoise_step(x, noise_pred, t)
```

**See `diffusion_code.py` for complete implementation!**

---

## Summary

Diffusion models are powerful generative models that learn to reverse a gradual noising process. They achieve state-of-the-art results in image generation and are increasingly applied to NLP tasks through discrete diffusion. Key advantages include high quality, stable training, and flexibility, though they are slower than alternatives. For NLP, discrete diffusion enables non-autoregressive generation, text inpainting, and better control over generation, making them valuable for editing and controlled generation tasks.

