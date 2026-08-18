# Normalization Techniques: Interview Q&A

## Q1: What is Batch Normalization? How does it work?

**Answer:**

Batch Normalization is a technique that normalizes the activations of a neural network layer by computing statistics across the batch dimension. The process involves three steps: first, computing the mean and variance of activations across all samples in the batch for each feature; second, normalizing the activations by subtracting the mean and dividing by the standard deviation; and third, applying learnable scale and shift parameters that allow the network to learn the optimal distribution.

Mathematically, for a batch of activations x with shape (batch_size, features), Batch Normalization computes the batch mean μ_B and batch variance σ²_B across the batch dimension, then normalizes each activation as (x - μ_B) / √(σ²_B + ε), where ε is a small constant for numerical stability. Finally, it applies learnable parameters γ (scale) and β (shift) to get the output y = γ * x̂ + β.

The key insight is that by normalizing activations, Batch Normalization reduces internal covariate shift, which is the change in distribution of layer inputs during training. This stabilization allows for higher learning rates and faster convergence. During training, batch statistics are used, but during inference, running averages of these statistics are maintained to ensure consistent behavior regardless of batch size.

> **Saying it out loud.** BatchNorm takes one feature at a time and rescales it using the mean and variance measured across the whole batch, then hands the model back a learned scale and shift so nothing is really lost. The reason it helps is that it makes the loss surface easier to walk on, so you can use a much bigger learning rate — around ten times bigger for ResNets, which is where its reputation comes from. One correction worth making out loud: the internal-covariate-shift explanation above is the original 2015 story, and Santurkar and colleagues tested it in 2018 by injecting noise after the BatchNorm layer to make covariate shift worse — training still improved, so that can't be the mechanism. The current best explanation is loss-landscape smoothing, and saying so is a quick way to show you've read past the abstract.

---

## Q2: What is Layer Normalization? How does it differ from Batch Normalization?

**Answer:**

Layer Normalization is a technique that normalizes activations across the feature dimension rather than the batch dimension. Unlike Batch Normalization, which computes statistics across all samples in a batch, Layer Normalization computes statistics independently for each sample across its features. This makes Layer Normalization independent of batch size and allows it to work with any batch size, including batch size one.

The mathematical formulation is similar to Batch Normalization, but the statistics are computed differently. For activations x with shape (batch_size, features), Layer Normalization computes the mean μ_L and variance σ²_L across the feature dimension (last dimension) for each sample independently. The normalization and scaling steps are the same: x̂ = (x - μ_L) / √(σ²_L + ε), followed by y = γ * x̂ + β.

The key differences are: Batch Normalization normalizes across the batch dimension (first dimension), making it dependent on batch composition, while Layer Normalization normalizes across the feature dimension (last dimension), making it independent of batch size. Batch Normalization requires batch_size > 1 and uses different statistics during training and inference, while Layer Normalization works with any batch size and has the same behavior in both training and inference.

> **Saying it out loud.** LayerNorm normalizes within a single example instead of across the batch — it takes one token's feature vector, standardizes it against its own mean and variance, and never looks at anything else. That one change removes every problem BatchNorm has: it works at batch size one, it doesn't care about variable sequence lengths, and it's the exact same computation at training and inference, so there are no running averages and no eval-mode bug. The tradeoff is that you lose the mild regularization BatchNorm got from noisy batch statistics — which is a real effect in vision, and irrelevant in language modeling where you have far more data than you can overfit.

---

## Q3: Why do transformers use Layer Normalization instead of Batch Normalization?

**Answer:**

Transformers use Layer Normalization for several critical reasons. First, transformers process sequences of variable length, and Batch Normalization would require normalizing across sequences of different lengths, which is problematic. Layer Normalization, by normalizing across features at each position independently, naturally handles variable-length sequences.

Second, in many NLP applications, especially during training or fine-tuning, batch sizes can be small. Batch Normalization relies on batch statistics, which become unreliable with small batches. Layer Normalization, by computing statistics per sample, is robust to small batch sizes and works equally well with batch size one as with larger batches.

Third, transformers often need to process sequences one at a time during inference, especially in autoregressive generation. With Batch Normalization, this would require using running statistics, which might not accurately represent the single sample being processed. Layer Normalization, with its per-sample normalization, works naturally in this scenario.

Additionally, Layer Normalization has the same behavior during training and inference, eliminating the need for running statistics and making the implementation simpler. The normalization across features at each position independently makes it ideal for the transformer architecture, where each position in the sequence is processed similarly regardless of sequence length.

> **Saying it out loud.** Because everything BatchNorm depends on is broken in a transformer. Sequences have different lengths, so batches are full of padding and BatchNorm folds those pad positions straight into its statistics. Large-model training runs at one or two sequences per GPU, which is far below the 16 to 32 you'd want for stable batch statistics. And at generation time you're decoding one token for one user, so there's no batch to normalize over at all — you'd be leaning entirely on running averages. LayerNorm has none of these problems because it never looks past the single token it's normalizing, so it's not a close call; there's no tradeoff being made here.

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

> **Saying it out loud.** Three steps, and the third is the one people forget. Compute the mean and variance of each feature across the batch, subtract and divide to standardize, then apply a learned scale and shift so the model can recover any distribution it wants. The $\varepsilon$ under the square root is just there so a near-constant feature doesn't blow up. The detail worth calling out at the end is that this whole computation changes at inference — there's no batch, so you use running averages accumulated during training, which means the deployed function isn't literally the one you validated. That gap is the source of essentially every BatchNorm production incident, starting with forgetting to call `eval()`.

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

> **Saying it out loud.** It's the same three steps as BatchNorm — standardize, then scale and shift — with one index changed: you average over the feature dimension instead of the batch dimension. So each token computes its own mean and variance from its own thousand-odd features. The consequence of that single change is everything that matters: no dependence on batch composition, no running averages, and an identical computation at train and test time. If you can only say one sentence, say that BatchNorm normalizes across examples and LayerNorm normalizes within one, and every other difference follows from that.

---

## Q6: What are the advantages and disadvantages of Batch Normalization?

**Answer:**

**Advantages of Batch Normalization:**

Batch Normalization provides several benefits. First, it reduces internal covariate shift by keeping the distribution of layer inputs relatively stable during training, which allows for higher learning rates and faster convergence. Second, it acts as a form of regularization by adding noise through batch statistics, which can help prevent overfitting. Third, it makes the optimization landscape smoother, making it easier for gradient-based optimizers to find good solutions. Fourth, it helps with the vanishing gradient problem by keeping activations in a normalized range, allowing gradients to flow more easily. Finally, it makes networks less sensitive to weight initialization.

**Disadvantages of Batch Normalization:**

Batch Normalization has several limitations. First, it requires a batch size greater than one to compute meaningful statistics. With batch size one, the variance would be zero, making normalization impossible. Second, the normalization depends on batch statistics, which means behavior can vary between training and inference if batch statistics differ significantly. Third, it doesn't work well with small batches because statistics become noisy and unreliable. Fourth, for sequence models, it's problematic because sequence lengths can vary, making it difficult to normalize across the sequence dimension. Finally, it requires maintaining running statistics for inference, adding complexity to the implementation.

> **Saying it out loud.** The advantage is a much larger usable learning rate and faster convergence, plus a bit of free regularization because the batch statistics are noisy and that noise acts like a mild dropout. The disadvantages all come from one root cause: your output for a given input depends on which other inputs happened to be in the batch. That means it breaks at batch size one, gets noisy below roughly 16 samples, behaves differently at inference, and gets corrupted by padding in sequence models. Worth being precise about the batch-size-one case: the within-sample variance is exactly zero, so the output collapses to $\beta$ and the layer stops passing any information at all.

---

## Q7: What are the advantages and disadvantages of Layer Normalization?

**Answer:**

**Advantages of Layer Normalization:**

Layer Normalization has several key advantages. First, it works with any batch size, including batch size one, making it suitable for online learning and inference scenarios. Second, it has the same behavior during training and inference, eliminating the need for running statistics and making implementation simpler. Third, it's ideal for sequence models where sequence lengths vary, as it normalizes per position independently. Fourth, it's more suitable for small batch sizes common in NLP applications. Fifth, it works naturally in autoregressive generation where samples are processed one at a time.

**Disadvantages of Layer Normalization:**

Layer Normalization has some limitations. First, it doesn't benefit from the regularization effect of batch statistics that Batch Normalization provides. Second, for large batches, Batch Normalization might provide better statistics and potentially better performance. Third, in some cases, especially with very large batches and consistent data, Batch Normalization might converge faster. However, for most modern applications, especially in NLP and transformers, these limitations are outweighed by the advantages.

> **Saying it out loud.** The advantages are all robustness: any batch size, identical behavior at train and test, no running statistics, no trouble with variable-length sequences, and it works during single-token autoregressive decoding. The one genuine thing you give up is the regularization noise from batch statistics, which mattered in the small-data vision era and matters very little when you're training on trillions of tokens. Claims that BatchNorm converges faster with large consistent batches are true in vision benchmarks and don't transfer to transformers. The practical evidence is that nobody has managed to make BatchNorm competitive for language models despite people trying — PowerNorm being the best-known attempt.

---

## Q8: How do Batch Normalization and Layer Normalization differ in terms of which dimension they normalize?

**Answer:**

The fundamental difference is in which dimension the normalization statistics are computed. Batch Normalization computes statistics across the batch dimension (first dimension), while Layer Normalization computes statistics across the feature dimension (last dimension).

For a tensor with shape (batch_size, features), Batch Normalization computes mean and variance across the batch dimension, giving one mean and one variance per feature across all samples. Layer Normalization computes mean and variance across the feature dimension, giving one mean and one variance per sample across all features.

For a 3D tensor with shape (batch_size, seq_len, features), Batch Normalization would normalize across the batch dimension (and potentially sequence dimension), while Layer Normalization normalizes across the feature dimension at each position independently. This difference is crucial: Batch Normalization makes the normalization dependent on other samples in the batch, while Layer Normalization makes it independent of batch composition.

This dimensional difference leads to all the other differences: batch size dependency, training vs inference behavior, and suitability for different architectures. Understanding this fundamental difference is key to understanding when to use each technique.

> **Saying it out loud.** One sentence: BatchNorm normalizes across examples, LayerNorm normalizes within an example. Picture the activation tensor as a grid with samples down one axis and features across the other — BatchNorm computes statistics down a column, LayerNorm across a row. Every practical difference is downstream of that geometry, including batch-size sensitivity, the train-test gap, and which architectures each suits. If you want to show depth, mention that GroupNorm sits between them by slicing the feature axis into groups, which is what vision uses when the batch is too small for BatchNorm but a single shared statistic per sample is too coarse.

---

## Summary

Batch Normalization and Layer Normalization are both powerful normalization techniques, but they differ fundamentally in how they compute statistics. Batch Normalization normalizes across the batch dimension, making it dependent on batch composition but potentially providing better statistics with large batches. Layer Normalization normalizes across the feature dimension, making it independent of batch size and ideal for sequence models and transformers. The choice between them depends on the architecture, batch size, and specific requirements of the application.

