# Topic 44: Normalization Techniques (Batch Norm & Layer Norm)

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

