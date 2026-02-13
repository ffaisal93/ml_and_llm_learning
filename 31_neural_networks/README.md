# Topic 31: Neural Networks from Scratch

## What You'll Learn

This topic teaches you to build and train neural networks from scratch:
- Forward pass
- Backpropagation (detailed mathematical explanation)
- Activation functions
- Loss functions
- Training loop
- Simple implementations in pure Python/NumPy

## Why We Need This

### Interview Importance
- **Common question**: "Implement backpropagation from scratch"
- **Deep understanding**: Shows you understand fundamentals
- **Foundation**: All deep learning builds on this

### Real-World Application
- **Understanding**: Know how neural networks work internally
- **Debugging**: Understand what's happening during training
- **Customization**: Build custom architectures

## Detailed Theory

### Forward Pass

**Mathematical Formulation:**

For a simple 2-layer neural network:

```
Input: x (n_features,)
Layer 1: h1 = activation(W1 @ x + b1)
Layer 2: h2 = activation(W2 @ h1 + b2)
Output: y = h2
```

**Step-by-step:**
1. **Input layer**: Raw features x
2. **Hidden layer 1**: 
   - Linear transformation: z1 = W1 @ x + b1
   - Apply activation: h1 = σ(z1) where σ is activation function
3. **Hidden layer 2**:
   - Linear transformation: z2 = W2 @ h1 + b2
   - Apply activation: h2 = σ(z2)
4. **Output**: Final prediction y = h2

**Why activation functions?**
- Without activation, neural network is just linear transformation
- Activation introduces non-linearity
- Enables learning complex patterns

### Backpropagation (Detailed Explanation)

**Backpropagation is the algorithm to compute gradients of loss with respect to all parameters.**

**Mathematical Foundation:**

We want to compute: ∂L/∂W and ∂L/∂b for all layers

**Chain Rule:**
If y = f(g(x)), then dy/dx = (df/dg) × (dg/dx)

**Step-by-step Backpropagation:**

**Step 1: Forward Pass**
```
x → z1 = W1 @ x + b1 → h1 = σ(z1) → z2 = W2 @ h1 + b2 → h2 = σ(z2) → L
```

**Step 2: Compute Output Layer Gradients**

For output layer (layer 2):
- Loss gradient w.r.t. output: ∂L/∂h2
- This depends on loss function (e.g., MSE: ∂L/∂h2 = 2(h2 - y_true))
- Gradient w.r.t. z2: ∂L/∂z2 = (∂L/∂h2) × (∂h2/∂z2) = (∂L/∂h2) × σ'(z2)
- Gradient w.r.t. W2: ∂L/∂W2 = (∂L/∂z2) @ h1^T
- Gradient w.r.t. b2: ∂L/∂b2 = ∂L/∂z2

**Step 3: Backpropagate to Hidden Layer**

For hidden layer (layer 1):
- Gradient w.r.t. h1: ∂L/∂h1 = W2^T @ (∂L/∂z2)
- Gradient w.r.t. z1: ∂L/∂z1 = (∂L/∂h1) × σ'(z1)
- Gradient w.r.t. W1: ∂L/∂W1 = (∂L/∂z1) @ x^T
- Gradient w.r.t. b1: ∂L/∂b1 = ∂L/∂z1

**Why it's called "backpropagation":**
- Gradients flow backwards from output to input
- Each layer uses gradients from next layer
- Computationally efficient (one forward + one backward pass)

### Activation Functions

**Sigmoid:**
- Formula: σ(x) = 1 / (1 + e^(-x))
- Range: (0, 1)
- Derivative: σ'(x) = σ(x)(1 - σ(x))
- **Use**: Output layer for binary classification
- **Problem**: Vanishing gradients (derivative → 0 for large |x|)

**ReLU (Rectified Linear Unit):**
- Formula: ReLU(x) = max(0, x)
- Derivative: 1 if x > 0, else 0
- **Use**: Hidden layers (most common)
- **Advantage**: Solves vanishing gradient problem
- **Problem**: Dead ReLU (outputs 0 forever if input < 0)

**Tanh:**
- Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- Range: (-1, 1)
- **Use**: Hidden layers (centered around 0)
- **Better than sigmoid**: Stronger gradients

### Loss Functions

**Mean Squared Error (MSE):**
- Formula: L = (1/n) Σ(y_pred - y_true)²
- **Use**: Regression
- **Derivative**: ∂L/∂y_pred = 2(y_pred - y_true)

**Cross-Entropy:**
- Formula: L = -Σ y_true × log(y_pred)
- **Use**: Classification
- **Derivative**: ∂L/∂y_pred = -y_true / y_pred

## Industry-Standard Boilerplate Code

See `neural_network.py` for complete implementation.

## Exercises

1. Implement forward pass
2. Implement backpropagation
3. Train on simple dataset
4. Visualize training process

## Next Steps

- **Topic 32**: Isolation Forest and Anomaly Detection
- Review neural network fundamentals

