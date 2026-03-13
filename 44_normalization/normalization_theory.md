# Normalization Techniques: Complete Theoretical Foundation

## Overview

Normalization techniques are crucial for training deep neural networks. They stabilize training, accelerate convergence, and enable training of very deep networks. The two most important normalization techniques are Batch Normalization and Layer Normalization, each with distinct properties and use cases.

---

## Part 1: Batch Normalization

### What is Batch Normalization?

Batch Normalization is a technique that normalizes the activations of a layer by adjusting and scaling them using statistics computed across the batch dimension. The key insight is that by normalizing the inputs to each layer, we can reduce internal covariate shift, which is the change in the distribution of layer inputs during training. This normalization makes the training process more stable and allows for higher learning rates, leading to faster convergence.

The process works by computing the mean and variance of activations across the batch dimension for each feature. These statistics are then used to normalize the activations, followed by learnable scale and shift parameters that allow the network to learn the optimal distribution for each layer. During training, batch statistics are used, but during inference, running averages of these statistics are used to ensure consistent behavior regardless of batch size.

### Mathematical Formulation

For a batch of activations x with shape (batch_size, features), Batch Normalization computes:

**Step 1: Compute batch statistics**
```
μ_B = (1/m) ∑_{i=1}^m x_i
σ²_B = (1/m) ∑_{i=1}^m (x_i - μ_B)²
```

Where m is the batch size, and the statistics are computed across the batch dimension (first dimension).

**Step 2: Normalize**
```
x̂ = (x - μ_B) / √(σ²_B + ε)
```

Where ε is a small constant (typically 1e-5) added for numerical stability.

**Step 3: Scale and shift**
```
y = γ * x̂ + β
```

Where γ (gamma) and β (beta) are learnable parameters that allow the network to learn the optimal scale and shift for each feature.

### Why Batch Normalization Works

Batch Normalization works through several mechanisms. First, it reduces internal covariate shift by keeping the distribution of layer inputs relatively stable during training. This stability allows for higher learning rates because the gradients are more consistent. Second, it acts as a form of regularization by adding noise through the batch statistics, which can help prevent overfitting. Third, it makes the optimization landscape smoother, making it easier for gradient-based optimizers to find good solutions.

The normalization also helps with the vanishing gradient problem in deep networks. By keeping activations in a normalized range, gradients can flow more easily through the network during backpropagation. Additionally, Batch Normalization makes the network less sensitive to weight initialization, as the normalization compensates for poor initializations to some extent.

### Training vs Inference

During training, Batch Normalization uses the statistics from the current batch to normalize activations. This means the normalization depends on the batch composition, which can vary between batches. However, during inference, we want consistent behavior regardless of batch size or composition. To achieve this, Batch Normalization maintains running averages of the mean and variance computed during training. These running averages are updated using exponential moving average with a momentum parameter, typically around 0.1.

The running statistics are computed as:
```
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
running_var = (1 - momentum) * running_var + momentum * batch_var
```

During inference, these running statistics are used instead of batch statistics, ensuring that the normalization is consistent and doesn't depend on the current batch.

### Limitations of Batch Normalization

Batch Normalization has several limitations that make it unsuitable for certain scenarios. First, it requires a batch size greater than one to compute meaningful statistics. With batch size one, the variance would be zero, making normalization impossible. Second, the normalization depends on batch statistics, which means the behavior can vary between training and inference if the batch statistics differ significantly. Third, Batch Normalization doesn't work well with small batches because the statistics become noisy and unreliable.

For sequence models like transformers, Batch Normalization has additional problems. The sequence length can vary between samples, making it difficult to normalize across the sequence dimension. Additionally, in many NLP applications, batch sizes are often small, making batch statistics unreliable. These limitations led to the development of Layer Normalization, which addresses these issues.

---

## Part 2: Layer Normalization

### What is Layer Normalization?

Layer Normalization is a technique that normalizes activations across the feature dimension rather than the batch dimension. Unlike Batch Normalization, which computes statistics across the batch, Layer Normalization computes statistics for each sample independently. This makes it independent of batch size and allows it to work with any batch size, including batch size one.

The key difference is in which dimension the statistics are computed. In Batch Normalization, we compute mean and variance across the batch dimension (first dimension), while in Layer Normalization, we compute them across the feature dimension (last dimension). This means each sample is normalized independently based on its own feature statistics, making the normalization independent of other samples in the batch.

### Mathematical Formulation

For activations x with shape (batch_size, features) or (batch_size, seq_len, features), Layer Normalization computes:

**Step 1: Compute per-sample statistics**
```
μ_L = (1/d) ∑_{i=1}^d x_i
σ²_L = (1/d) ∑_{i=1}^d (x_i - μ_L)²
```

Where d is the number of features, and the statistics are computed across the feature dimension (last dimension) for each sample independently.

**Step 2: Normalize**
```
x̂ = (x - μ_L) / √(σ²_L + ε)
```

**Step 3: Scale and shift**
```
y = γ * x̂ + β
```

Where γ and β are learnable parameters, same as in Batch Normalization.

### Why Layer Normalization Works

Layer Normalization works by normalizing each sample independently, which makes it robust to batch size variations. This independence from batch statistics means that Layer Normalization behaves the same way during training and inference, eliminating the need for running statistics. The normalization across features helps stabilize the training by keeping the feature activations in a normalized range, similar to Batch Normalization, but without the dependency on batch composition.

For sequence models, Layer Normalization is particularly effective because it normalizes across the feature dimension at each position independently. This means that regardless of sequence length, each position is normalized based on its own features, making it suitable for variable-length sequences. This property is crucial for transformers, where sequences can vary significantly in length.

### Advantages Over Batch Normalization

Layer Normalization has several advantages over Batch Normalization, especially for certain types of models. First, it works with any batch size, including batch size one, making it suitable for online learning and inference scenarios where batch size might be small or variable. Second, it has the same behavior during training and inference, eliminating the need for running statistics and making the implementation simpler. Third, it's more suitable for sequence models where sequence lengths vary, as it normalizes per position independently.

For transformers specifically, Layer Normalization is ideal because it can handle variable sequence lengths, works with small batch sizes common in NLP, and provides stable normalization regardless of batch composition. These properties make it the standard choice for transformer architectures.

---

## Part 3: Why Transformers Use Layer Normalization

### The Transformer Architecture Requirements

Transformers have specific requirements that make Layer Normalization more suitable than Batch Normalization. First, transformers process sequences of variable length, and we need a normalization technique that can handle this variability. Batch Normalization would require normalizing across sequences of different lengths, which is problematic. Layer Normalization, on the other hand, normalizes across features at each position independently, making it naturally suited for variable-length sequences.

Second, in many NLP applications, especially during training or fine-tuning, batch sizes can be small. Batch Normalization relies on batch statistics, which become unreliable with small batches. Layer Normalization, by computing statistics per sample, is robust to small batch sizes and works equally well with batch size one as with larger batches.

Third, transformers often need to process sequences one at a time during inference, especially in autoregressive generation. With Batch Normalization, this would require using running statistics, which might not accurately represent the single sample being processed. Layer Normalization, with its per-sample normalization, works naturally in this scenario.

### The Pre-LayerNorm vs Post-LayerNorm Debate

In transformers, there's a choice of where to place Layer Normalization: before the attention/FFN layers (Pre-LayerNorm) or after them (Post-LayerNorm). Pre-LayerNorm, where normalization is applied before the sub-layers, has become the standard in modern transformers because it provides more stable gradients and allows for deeper networks. Post-LayerNorm, used in the original Transformer paper, applies normalization after the sub-layers and can have gradient issues in very deep networks.

The Pre-LayerNorm architecture applies normalization to the input before passing it through attention or feed-forward layers, then adds the normalized output to the original input through a residual connection. This arrangement ensures that the inputs to each sub-layer are normalized, leading to more stable training.

---

## Part 4: Comparison and When to Use Each

### Key Differences

**Normalization Dimension:**
- BatchNorm: Normalizes across batch dimension (first dimension)
- LayerNorm: Normalizes across feature dimension (last dimension)

**Batch Size Dependency:**
- BatchNorm: Requires batch_size > 1, statistics depend on batch
- LayerNorm: Works with any batch size, independent of batch

**Training vs Inference:**
- BatchNorm: Different behavior (uses batch stats vs running stats)
- LayerNorm: Same behavior (always uses per-sample stats)

**Running Statistics:**
- BatchNorm: Needs running mean/variance for inference
- LayerNorm: No running statistics needed

**Use Cases:**
- BatchNorm: CNNs, image classification, large batches
- LayerNorm: Transformers, RNNs, NLP, variable batch sizes

### When to Use Batch Normalization

Batch Normalization is ideal for scenarios with large, consistent batch sizes, such as image classification with CNNs. It works well when the batch statistics are reliable and representative of the data distribution. It's particularly effective in convolutional networks where spatial dimensions provide additional samples for computing statistics.

### When to Use Layer Normalization

Layer Normalization is ideal for sequence models, especially transformers, where sequence lengths vary and batch sizes might be small. It's also preferred when you need consistent behavior between training and inference, or when you need to process samples one at a time. For NLP applications, Layer Normalization is almost always the better choice.

---

## Summary

Batch Normalization and Layer Normalization are both powerful techniques for stabilizing neural network training, but they differ fundamentally in how they compute normalization statistics. Batch Normalization normalizes across the batch dimension, making it dependent on batch composition, while Layer Normalization normalizes across the feature dimension, making it independent of batch size. For transformers and many modern NLP models, Layer Normalization is the preferred choice due to its robustness to variable batch sizes, sequence lengths, and its consistent behavior between training and inference.

