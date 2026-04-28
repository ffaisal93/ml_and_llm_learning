# Topic 44: Normalization Techniques (Batch Norm & Layer Norm)

> 🔥 **For interviews, read these first:**
> - **`NORMALIZATION_DEEP_DIVE.md`** — frontier-lab interview deep dive: BN/LN/RMSNorm/GroupNorm, why BN fails for transformers, pre-LN vs post-LN, the affine transform, the loss-landscape-smoothing argument (and why "internal covariate shift" is wrong).
> - **`INTERVIEW_GRILL.md`** — 40 active-recall questions with strong answers.

## What You'll Learn

This topic teaches you normalization techniques comprehensively:
- Batch Normalization (BatchNorm)
- Layer Normalization (LayerNorm)
- Mathematical formulations
- Differences and when to use each
- Why transformers use LayerNorm
- Implementation details

## Why We Need This

### Interview Importance
- **Common question**: "Explain BatchNorm vs LayerNorm"
- **Transformer understanding**: Why transformers use LayerNorm
- **Implementation**: May ask to implement from scratch

### Real-World Application
- **CNNs**: Often use BatchNorm
- **Transformers**: Use LayerNorm
- **Training stability**: Critical for deep networks
- **Convergence**: Helps training converge faster

## Industry Use Cases

### 1. **Batch Normalization**
**Use Case**: CNNs, image classification
- Normalizes across batch dimension
- Requires batch statistics
- Works well with large batches

### 2. **Layer Normalization**
**Use Case**: Transformers, RNNs, NLP
- Normalizes across feature dimension
- Independent of batch size
- Works with any batch size

## Core Intuition

Normalization helps training by controlling activation scale and making optimization more stable.

The intuition is not just "make values smaller."

It is:
- keep activations in a reasonable range
- reduce sensitivity to scale changes across layers
- make optimization easier and more stable

### BatchNorm

BatchNorm uses statistics across examples in the batch.

That makes it work well in settings like CNNs where:
- batch statistics are meaningful
- batch sizes are usually large enough

### LayerNorm

LayerNorm uses statistics within each example across its feature dimension.

That makes it useful when:
- batch size is small or variable
- sequence models need consistent behavior per token/sample

This is why transformers use LayerNorm much more naturally than BatchNorm.

## Technical Details Interviewers Often Want

### Why Transformers Prefer LayerNorm

Transformers often use variable sequence lengths, small effective batches, and token-wise computations.

LayerNorm is attractive because it:
- does not depend on batch statistics
- behaves consistently across training and inference
- fits sequence modeling well

### Why BatchNorm Can Be Awkward in NLP

BatchNorm depends on batch-level statistics, which can be less stable or less natural in autoregressive and sequence-heavy settings, especially with variable lengths or small batches.

### Learnable Parameters Matter

Both BatchNorm and LayerNorm typically use learnable scale and shift parameters.

Why?
- after normalization, the model still needs flexibility to represent useful scales and offsets

## Common Failure Modes

- explaining normalization only as "faster training" without mechanism
- forgetting that BatchNorm behaves differently at training and inference
- ignoring batch-size dependence in BatchNorm
- not being able to explain why LayerNorm is common in transformers

## Edge Cases and Follow-Up Questions

1. Why does BatchNorm depend on batch size?
2. Why is LayerNorm more natural for transformers?
3. Why do normalized activations still need learnable scale and bias?
4. Why can BatchNorm become awkward with very small batches?
5. Why is training-time vs inference-time behavior different for BatchNorm?

## What to Practice Saying Out Loud

1. The difference between normalizing across batch vs across features
2. Why LayerNorm is standard in transformer architectures
3. Why normalization is really about optimization stability, not just value scaling

## Theory

### Batch Normalization

**What it is:**
- Normalizes activations across the batch dimension
- Uses batch statistics (mean, variance)
- Helps with training stability and convergence

**Mathematical Formulation:**
```
μ_B = (1/m) ∑ x_i
σ²_B = (1/m) ∑ (x_i - μ_B)²
x̂ = (x - μ_B) / √(σ²_B + ε)
y = γ * x̂ + β
```

### Layer Normalization

**What it is:**
- Normalizes activations across the feature dimension
- Uses per-sample statistics
- Independent of batch size

**Mathematical Formulation:**
```
μ_L = (1/d) ∑ x_i
σ²_L = (1/d) ∑ (x_i - μ_L)²
x̂ = (x - μ_L) / √(σ²_L + ε)
y = γ * x̂ + β
```

## Industry-Standard Boilerplate Code

See detailed files for complete implementations:
- `normalization_implementations.py`: Complete implementations from scratch
- `normalization_theory.md`: Detailed theoretical explanations
- `normalization_qa.md`: Comprehensive interview Q&A

## Exercises

1. Implement BatchNorm from scratch
2. Implement LayerNorm from scratch
3. Compare BatchNorm vs LayerNorm
4. Understand why transformers use LayerNorm
5. Test with different batch sizes

## Next Steps

- Review transformer architecture
- Understand training dynamics
- Explore other normalization techniques
