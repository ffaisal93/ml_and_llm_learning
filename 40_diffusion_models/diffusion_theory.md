# Diffusion Models: Complete Theoretical Foundation

## Overview

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process. They have achieved state-of-the-art results in image generation (DALL-E, Stable Diffusion) and are increasingly being applied to NLP tasks. This document provides a comprehensive theoretical foundation for understanding diffusion models.

---

## Part 1: Core Concept and Intuition

### What is a Diffusion Model?

A diffusion model is a generative model that learns to reverse a forward diffusion process. The key idea is to train a model to gradually remove noise from data, starting from pure noise and ending with a clean sample. This is analogous to how an artist might start with a blank canvas and gradually add details, but in reverse: we start with noise and gradually remove it to reveal the structure.

The forward process is a fixed, predefined process that gradually adds Gaussian noise to data. After many steps (typically 1000-4000 steps), the data becomes indistinguishable from pure Gaussian noise. The reverse process is what the model learns: given noisy data at step t, predict what the data looked like at step t-1 (less noisy). By iteratively applying this reverse process, we can generate new samples from pure noise.

### Why Diffusion Models Work

Diffusion models work because they break down the complex problem of generating data into many simpler problems of removing small amounts of noise. Instead of learning to generate complex data directly, the model learns to make small, incremental improvements to noisy data. This is easier to learn because each step only needs to remove a small amount of noise, making the learning problem more tractable.

The forward process ensures that the data distribution at each step is close to the distribution at the previous step, making the reverse process learnable. The model doesn't need to learn complex mappings between very different distributions; it only needs to learn how to make small denoising steps, which is a much simpler problem.

---

## Part 2: Mathematical Foundation

### Forward Diffusion Process

The forward diffusion process is a fixed Markov chain that gradually adds Gaussian noise to data. Given a data sample x₀ from the data distribution q(x₀), we define a sequence of increasingly noisy versions x₁, x₂, ..., x_T, where x_T is approximately pure Gaussian noise.

**Mathematical Formulation:**

For each step t, we add a small amount of Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
```

Where:
- β_t is a variance schedule (0 < β_t < 1) that controls how much noise is added at step t
- √(1-β_t) is a scaling factor that ensures the signal doesn't explode
- I is the identity matrix (assuming independent noise per dimension)

**Properties:**
- β_t is typically small (e.g., 0.0001 to 0.02) and increases with t
- The process is designed so that after T steps, x_T ≈ N(0, I) (pure noise)
- Each step adds a small amount of noise, making the transition smooth

**Closed-Form Expression:**

A key insight is that we can sample x_t directly from x₀ without going through all intermediate steps:

```
q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
```

Where:
- α_t = 1 - β_t (the amount of signal preserved)
- ᾱ_t = ∏_{s=1}^t α_s (cumulative product)
- This allows efficient sampling during training

**Intuition:**
- At t=0: x_0 is clean data
- At t=T: x_T is approximately pure noise N(0, I)
- The variance schedule β_t controls how quickly we add noise

### Reverse Diffusion Process

The reverse diffusion process is what the model learns. Given noisy data x_t at step t, the model learns to predict x_{t-1} (less noisy version). This is the generative process: we start from pure noise x_T ~ N(0, I) and iteratively apply the reverse process to generate clean data x₀.

**Mathematical Formulation:**

The reverse process is parameterized by a neural network:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

Where:
- μ_θ(x_t, t) is the predicted mean (learned by neural network)
- Σ_θ(x_t, t) is the predicted variance (can be learned or fixed)
- θ represents the model parameters

**Key Insight:**

Instead of predicting x_{t-1} directly, we can predict the noise ε that was added. This is easier because:
- The noise is simpler to predict than the complex data structure
- We can use the closed-form expression: x_{t-1} = (1/√(α_t))(x_t - (β_t/√(1-ᾱ_t))ε)

**Prediction Target:**

The model predicts the noise ε that was added to get from x₀ to x_t:

```
ε_θ(x_t, t) ≈ ε
```

Where ε ~ N(0, I) is the noise that was added.

---

## Part 3: Training Objective

### Loss Function

The training objective is to minimize the difference between the predicted noise and the actual noise that was added. This is done using a simple mean-squared error loss.

**Mathematical Formulation:**

```
L = E_{t,x_0,ε} [||ε - ε_θ(x_t, t)||²]
```

Where:
- t is uniformly sampled from {1, 2, ..., T}
- x_0 is a sample from the data distribution
- ε ~ N(0, I) is the noise added
- x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε is the noisy data (using closed-form)
- ε_θ(x_t, t) is the predicted noise

**Intuition:**
- At each training step, we:
  1. Sample a data point x₀
  2. Sample a timestep t
  3. Sample noise ε
  4. Create noisy data x_t = √(ᾱ_t)x₀ + √(1-ᾱ_t)ε
  5. Train the model to predict ε given x_t and t
- The model learns to predict what noise was added, which allows it to reverse the process

### Training Procedure

**Algorithm:**

1. **Sample data**: x₀ ~ q(x₀)
2. **Sample timestep**: t ~ Uniform({1, 2, ..., T})
3. **Sample noise**: ε ~ N(0, I)
4. **Create noisy data**: x_t = √(ᾱ_t)x₀ + √(1-ᾱ_t)ε
5. **Predict noise**: ε_pred = ε_θ(x_t, t)
6. **Compute loss**: L = ||ε - ε_pred||²
7. **Update parameters**: θ ← θ - α∇_θ L

**Key Points:**
- We randomly sample timesteps during training (not sequential)
- This allows the model to learn denoising at all noise levels
- The model sees all noise levels during training, making it robust

---

## Part 4: Sampling/Generation Process

### Generation Algorithm

To generate new samples, we start from pure noise and iteratively apply the reverse process:

**Algorithm:**

1. **Sample initial noise**: x_T ~ N(0, I)
2. **For t = T, T-1, ..., 1:**
   - **Predict noise**: ε_t = ε_θ(x_t, t)
   - **Predict mean**: μ_t = (1/√(α_t))(x_t - (β_t/√(1-ᾱ_t))ε_t)
   - **Sample**: x_{t-1} ~ N(μ_t, Σ_t)
3. **Return**: x₀ (generated sample)

**Step-by-Step:**

At each step t:
- The model predicts the noise ε_t that was added
- We use this to compute the predicted mean μ_t of x_{t-1}
- We sample x_{t-1} from N(μ_t, Σ_t)
- This gives us a slightly less noisy version
- We repeat until we get clean data x₀

**Variance Schedule:**

The variance Σ_t can be:
- **Fixed**: Σ_t = β_t I (simple, works well)
- **Learned**: Σ_t = Σ_θ(x_t, t) (more flexible, harder to train)

**Intuition:**
- We start with pure noise (no structure)
- Each step removes a small amount of noise (adds structure)
- After T steps, we have clean, structured data

---

## Part 5: Discrete Diffusion for NLP

### The Challenge

Standard diffusion models work on continuous data (images, audio). Text is discrete (tokens), so we need adaptations. Discrete diffusion models extend diffusion to discrete data.

### Discrete Forward Process

Instead of adding Gaussian noise, we use a transition matrix that corrupts tokens:

**Mathematical Formulation:**

```
q(x_t | x_{t-1}) = Categorical(x_t; Q_t x_{t-1})
```

Where:
- Q_t is a transition matrix that defines how tokens are corrupted
- Each row of Q_t defines the probability distribution for corrupting a token
- Common choices: uniform transition, absorbing state, etc.

**Absorbing State:**
- One common approach is to have an "absorbing" token [MASK]
- At each step, tokens can transition to [MASK] with probability β_t
- After T steps, all tokens become [MASK]

**Uniform Transition:**
- Tokens can transition to any other token uniformly
- More general but harder to learn

### Discrete Reverse Process

The reverse process learns to predict the original token given the corrupted version:

**Mathematical Formulation:**

```
p_θ(x_{t-1} | x_t) = Categorical(x_{t-1}; p_θ(x_t, t))
```

Where:
- p_θ(x_t, t) is a probability distribution over vocabulary (learned by model)
- The model predicts which token should be at position i at step t-1

**Training Objective:**

Similar to continuous case, but with cross-entropy loss:

```
L = E_{t,x_0,x_t} [-log p_θ(x_0 | x_t, t)]
```

Or predict the corruption:

```
L = E_{t,x_0,x_t} [CrossEntropy(x_0, p_θ(x_t, t))]
```

### Advantages for NLP

**Non-Autoregressive:**
- Can generate all tokens in parallel
- Faster than autoregressive models
- Better for controlled generation

**Flexible:**
- Can edit specific parts of text
- Can do text inpainting (fill in masked tokens)
- Better control over generation

---

## Part 6: Variance Schedules

### Linear Schedule

**Definition:**
```
β_t = (β_max - β_min) * (t / T) + β_min
```

**Properties:**
- Simple and commonly used
- Linear increase in noise
- β_min ≈ 0.0001, β_max ≈ 0.02

### Cosine Schedule

**Definition:**
```
ᾱ_t = cos²(π/2 * (t/T))
```

**Properties:**
- Adds noise more slowly at the beginning
- Faster at the end
- Often works better than linear

### Custom Schedules

**Polynomial:**
- β_t = (t/T)^p for some power p
- Allows control over noise schedule

**Learnable:**
- Can learn optimal schedule during training
- More complex but potentially better

---

## Part 7: Model Architecture

### U-Net for Images

**Standard Architecture:**
- U-Net with skip connections
- Time embedding (sinusoidal or learned)
- Attention mechanisms
- Residual connections

**Time Embedding:**
- Encodes timestep t into vector
- Added to each layer
- Allows model to condition on noise level

### Transformer for Text

**Architecture:**
- Standard transformer encoder
- Time embedding added to input
- Predicts token distribution at each position
- Can be non-autoregressive

**Conditioning:**
- Can condition on text prompts
- Can condition on partial text (inpainting)
- Flexible conditioning mechanisms

---

## Part 8: Advanced Topics

### Classifier-Free Guidance

**Concept:**
- Train model with and without conditioning
- At inference, use guidance to increase conditioning strength
- Improves quality and control

**Mathematical Formulation:**

```
ε_θ(x_t, t, c) = (1 + w) * ε_θ(x_t, t, c) - w * ε_θ(x_t, t)
```

Where:
- c is the condition (e.g., text prompt)
- w is the guidance weight
- Higher w = stronger conditioning

### Latent Diffusion

**Concept:**
- Apply diffusion in latent space (not pixel space)
- Use VAE to encode/decode
- Much more efficient

**Advantages:**
- Faster training and inference
- Lower memory usage
- Better quality (latent space is more structured)

### Multimodal Diffusion

**Concept:**
- Apply diffusion to multiple modalities
- Can generate text and images together
- Cross-modal conditioning

**Applications:**
- Text-to-image (DALL-E, Stable Diffusion)
- Image-to-text
- Text-to-audio

---

## Part 9: Comparison with Other Generative Models

### vs. Autoregressive Models (GPT)

**Diffusion:**
- Non-autoregressive (parallel generation)
- Iterative refinement
- Better for editing tasks

**Autoregressive:**
- Sequential generation
- Faster single-pass generation
- Better for long sequences

### vs. GANs

**Diffusion:**
- More stable training
- Better mode coverage
- Slower generation

**GANs:**
- Faster generation
- Can have mode collapse
- Harder to train

### vs. VAEs

**Diffusion:**
- Better quality
- More complex training
- Slower generation

**VAEs:**
- Faster generation
- Lower quality
- Simpler training

---

## Summary

Diffusion models are powerful generative models that learn to reverse a gradual noising process. They work by:
1. **Forward process**: Gradually add noise to data
2. **Reverse process**: Learn to remove noise iteratively
3. **Training**: Predict the noise that was added
4. **Generation**: Start from noise and denoise to generate samples

For NLP, discrete diffusion extends this to tokens, enabling non-autoregressive text generation with better control and editing capabilities. The key advantages are parallel generation, flexible conditioning, and the ability to edit specific parts of text.

