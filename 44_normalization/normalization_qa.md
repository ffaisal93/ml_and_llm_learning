# Normalization Techniques: Interview Q&A

## Q1: What is Batch Normalization? How does it work?

**Answer:**

Batch Normalization is a technique that normalizes the activations of a neural network layer by computing statistics across the batch dimension. The process involves three steps: first, computing the mean and variance of activations across all samples in the batch for each feature; second, normalizing the activations by subtracting the mean and dividing by the standard deviation; and third, applying learnable scale and shift parameters that allow the network to learn the optimal distribution.

Mathematically, for a batch of activations x with shape (batch_size, features), Batch Normalization computes the batch mean μ_B and batch variance σ²_B across the batch dimension, then normalizes each activation as (x - μ_B) / √(σ²_B + ε), where ε is a small constant for numerical stability. Finally, it applies learnable parameters γ (scale) and β (shift) to get the output y = γ * x̂ + β.

The key insight is that by normalizing activations, Batch Normalization reduces internal covariate shift, which is the change in distribution of layer inputs during training. This stabilization allows for higher learning rates and faster convergence. During training, batch statistics are used, but during inference, running averages of these statistics are maintained to ensure consistent behavior regardless of batch size.

---

## Q2: What is Layer Normalization? How does it differ from Batch Normalization?

**Answer:**

Layer Normalization is a technique that normalizes activations across the feature dimension rather than the batch dimension. Unlike Batch Normalization, which computes statistics across all samples in a batch, Layer Normalization computes statistics independently for each sample across its features. This makes Layer Normalization independent of batch size and allows it to work with any batch size, including batch size one.

The mathematical formulation is similar to Batch Normalization, but the statistics are computed differently. For activations x with shape (batch_size, features), Layer Normalization computes the mean μ_L and variance σ²_L across the feature dimension (last dimension) for each sample independently. The normalization and scaling steps are the same: x̂ = (x - μ_L) / √(σ²_L + ε), followed by y = γ * x̂ + β.

The key differences are: Batch Normalization normalizes across the batch dimension (first dimension), making it dependent on batch composition, while Layer Normalization normalizes across the feature dimension (last dimension), making it independent of batch size. Batch Normalization requires batch_size > 1 and uses different statistics during training and inference, while Layer Normalization works with any batch size and has the same behavior in both training and inference.

---

## Q3: Why do transformers use Layer Normalization instead of Batch Normalization?

**Answer:**

Transformers use Layer Normalization for several critical reasons. First, transformers process sequences of variable length, and Batch Normalization would require normalizing across sequences of different lengths, which is problematic. Layer Normalization, by normalizing across features at each position independently, naturally handles variable-length sequences.

Second, in many NLP applications, especially during training or fine-tuning, batch sizes can be small. Batch Normalization relies on batch statistics, which become unreliable with small batches. Layer Normalization, by computing statistics per sample, is robust to small batch sizes and works equally well with batch size one as with larger batches.

Third, transformers often need to process sequences one at a time during inference, especially in autoregressive generation. With Batch Normalization, this would require using running statistics, which might not accurately represent the single sample being processed. Layer Normalization, with its per-sample normalization, works naturally in this scenario.

Additionally, Layer Normalization has the same behavior during training and inference, eliminating the need for running statistics and making the implementation simpler. The normalization across features at each position independently makes it ideal for the transformer architecture, where each position in the sequence is processed similarly regardless of sequence length.

---

## Q4: Explain the mathematical formulation of Batch Normalization in detail.

**Answer:**

Batch Normalization involves computing statistics across the batch dimension and using them to normalize activations. For a batch of activations x with shape (batch_size, features), the process is:

**Step 1: Compute batch statistics**
```
μ_B = (1/m) ∑_{i=1}^m x_i
σ²_B = (1/m) ∑_{i=1}^m (x_i - μ_B)²
```

Where m is the batch size, and the mean and variance are computed across the batch dimension (first dimension) for each feature independently. This gives us a mean and variance for each feature across all samples in the batch.

**Step 2: Normalize**
```
x̂ = (x - μ_B) / √(σ²_B + ε)
```

This centers the activations around zero and scales them to have unit variance. The small constant ε (typically 1e-5) prevents division by zero and ensures numerical stability.

**Step 3: Scale and shift**
```
y = γ * x̂ + β
```

Where γ (gamma) and β (beta) are learnable parameters that allow the network to learn the optimal scale and shift for each feature. If the network determines that the original distribution was optimal, it can learn γ = √(σ²_B) and β = μ_B to recover the original activations.

During inference, running averages of μ_B and σ²_B are used instead of computing them from the current batch, ensuring consistent behavior regardless of batch size or composition.

---

## Q5: Explain the mathematical formulation of Layer Normalization in detail.

**Answer:**

Layer Normalization computes statistics across the feature dimension for each sample independently. For activations x with shape (batch_size, features) or (batch_size, seq_len, features), the process is:

**Step 1: Compute per-sample statistics**
```
μ_L = (1/d) ∑_{i=1}^d x_i
σ²_L = (1/d) ∑_{i=1}^d (x_i - μ_L)²
```

Where d is the number of features, and the mean and variance are computed across the feature dimension (last dimension) for each sample independently. This means each sample in the batch gets its own mean and variance computed from its features.

**Step 2: Normalize**
```
x̂ = (x - μ_L) / √(σ²_L + ε)
```

This normalizes each sample independently, centering it around zero and scaling to unit variance based on its own feature statistics.

**Step 3: Scale and shift**
```
y = γ * x̂ + β
```

Where γ and β are learnable parameters, same as in Batch Normalization. These allow the network to learn the optimal distribution for each feature.

The key difference from Batch Normalization is that statistics are computed per sample (across features) rather than per feature (across batch). This makes Layer Normalization independent of batch size and ensures the same behavior during training and inference.

---

## Q6: What are the advantages and disadvantages of Batch Normalization?

**Answer:**

**Advantages of Batch Normalization:**

Batch Normalization provides several benefits. First, it reduces internal covariate shift by keeping the distribution of layer inputs relatively stable during training, which allows for higher learning rates and faster convergence. Second, it acts as a form of regularization by adding noise through batch statistics, which can help prevent overfitting. Third, it makes the optimization landscape smoother, making it easier for gradient-based optimizers to find good solutions. Fourth, it helps with the vanishing gradient problem by keeping activations in a normalized range, allowing gradients to flow more easily. Finally, it makes networks less sensitive to weight initialization.

**Disadvantages of Batch Normalization:**

Batch Normalization has several limitations. First, it requires a batch size greater than one to compute meaningful statistics. With batch size one, the variance would be zero, making normalization impossible. Second, the normalization depends on batch statistics, which means behavior can vary between training and inference if batch statistics differ significantly. Third, it doesn't work well with small batches because statistics become noisy and unreliable. Fourth, for sequence models, it's problematic because sequence lengths can vary, making it difficult to normalize across the sequence dimension. Finally, it requires maintaining running statistics for inference, adding complexity to the implementation.

---

## Q7: What are the advantages and disadvantages of Layer Normalization?

**Answer:**

**Advantages of Layer Normalization:**

Layer Normalization has several key advantages. First, it works with any batch size, including batch size one, making it suitable for online learning and inference scenarios. Second, it has the same behavior during training and inference, eliminating the need for running statistics and making implementation simpler. Third, it's ideal for sequence models where sequence lengths vary, as it normalizes per position independently. Fourth, it's more suitable for small batch sizes common in NLP applications. Fifth, it works naturally in autoregressive generation where samples are processed one at a time.

**Disadvantages of Layer Normalization:**

Layer Normalization has some limitations. First, it doesn't benefit from the regularization effect of batch statistics that Batch Normalization provides. Second, for large batches, Batch Normalization might provide better statistics and potentially better performance. Third, in some cases, especially with very large batches and consistent data, Batch Normalization might converge faster. However, for most modern applications, especially in NLP and transformers, these limitations are outweighed by the advantages.

---

## Q8: How do Batch Normalization and Layer Normalization differ in terms of which dimension they normalize?

**Answer:**

The fundamental difference is in which dimension the normalization statistics are computed. Batch Normalization computes statistics across the batch dimension (first dimension), while Layer Normalization computes statistics across the feature dimension (last dimension).

For a tensor with shape (batch_size, features), Batch Normalization computes mean and variance across the batch dimension, giving one mean and one variance per feature across all samples. Layer Normalization computes mean and variance across the feature dimension, giving one mean and one variance per sample across all features.

For a 3D tensor with shape (batch_size, seq_len, features), Batch Normalization would normalize across the batch dimension (and potentially sequence dimension), while Layer Normalization normalizes across the feature dimension at each position independently. This difference is crucial: Batch Normalization makes the normalization dependent on other samples in the batch, while Layer Normalization makes it independent of batch composition.

This dimensional difference leads to all the other differences: batch size dependency, training vs inference behavior, and suitability for different architectures. Understanding this fundamental difference is key to understanding when to use each technique.

---

## Summary

Batch Normalization and Layer Normalization are both powerful normalization techniques, but they differ fundamentally in how they compute statistics. Batch Normalization normalizes across the batch dimension, making it dependent on batch composition but potentially providing better statistics with large batches. Layer Normalization normalizes across the feature dimension, making it independent of batch size and ideal for sequence models and transformers. The choice between them depends on the architecture, batch size, and specific requirements of the application.

