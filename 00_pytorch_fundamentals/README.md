# Topic 0: PyTorch Fundamentals

## What You'll Learn

This topic covers all essential PyTorch concepts you need to write code in this repository:
- Tensors (creation, operations, indexing)
- Autograd (automatic differentiation)
- Neural Network Layers
- Loss Functions
- Optimizers
- Training Loops
- Data Loading
- Device Management (CPU/GPU)
- Simple, clear examples

## Why We Need This

### Foundation for All Topics
- **Neural networks**: All use PyTorch
- **Training**: Need to understand training loops
- **Gradients**: Backpropagation uses autograd
- **Reference**: Come back here when you need PyTorch syntax

### Interview Importance
- **Common questions**: "Implement training loop in PyTorch"
- **Practical knowledge**: Shows you can use PyTorch
- **Code writing**: Need to write PyTorch code in interviews

## Core Intuition

PyTorch gives you the core building blocks of deep learning in a way that is easy to compose:
- tensors hold data
- autograd computes gradients
- modules organize learnable computation
- optimizers update parameters

If you understand those pieces, most PyTorch code becomes understandable instead of feeling like boilerplate magic.

### Tensors

Tensors are just arrays with extra capabilities:
- GPU support
- datatype control
- gradient tracking compatibility

### Autograd

Autograd is what makes backprop practical in modern frameworks.

The key idea is:
- define the forward computation
- let the framework build the graph
- call backward to get gradients

### Modules and Parameters

An `nn.Module` bundles:
- parameters
- submodules
- forward logic

That means PyTorch models are really compositions of reusable parameterized functions.

## Technical Details Interviewers Often Want

### Why `zero_grad()` Matters

PyTorch accumulates gradients by default.

That is useful for gradient accumulation, but a bug in ordinary training loops if you forget to clear gradients.

### Why `train()` vs `eval()` Matters

Some layers behave differently during training and inference:
- dropout
- batch normalization

If you do not set the correct mode, model behavior and metrics can be wrong in subtle ways.

### Why `torch.no_grad()` Matters

During inference, you usually do not want gradient tracking.

Turning it off:
- saves memory
- speeds execution
- avoids unnecessary graph construction

## Common Failure Modes

- forgetting `optimizer.zero_grad()`
- forgetting `model.train()` or `model.eval()`
- mismatching tensor devices (CPU vs GPU)
- using the wrong tensor shape for the loss
- tracking gradients during inference unnecessarily

## Edge Cases and Follow-Up Questions

1. Why does PyTorch accumulate gradients by default?
2. Why do dropout and BatchNorm need different train and eval behavior?
3. Why can code run but still fail when tensors live on different devices?
4. What is the conceptual difference between `view` and `reshape`?
5. Why is autograd useful but not free?

## What to Practice Saying Out Loud

1. The standard PyTorch training loop
2. What autograd is doing conceptually
3. Why mode switching and gradient control matter

## Core Concepts

### 1. Tensors

**What are Tensors?**
Tensors are multi-dimensional arrays, similar to NumPy arrays but with GPU support and automatic differentiation.

**Creating Tensors:**
```python
import torch

# From Python list
x = torch.tensor([1, 2, 3])

# From NumPy
import numpy as np
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)

# Zeros, ones, random
x = torch.zeros(3, 4)  # 3x4 tensor of zeros
x = torch.ones(2, 3)  # 2x3 tensor of ones
x = torch.randn(2, 3)  # 2x3 tensor from normal distribution

# With specific dtype
x = torch.tensor([1, 2, 3], dtype=torch.float32)
```

**Tensor Operations:**
```python
# Basic operations (element-wise)
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b  # [5, 7, 9]
c = a * b  # [4, 10, 18]

# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.matmul(A, B)  # or A @ B

# Reshaping
x = torch.randn(2, 3, 4)
y = x.view(6, 4)  # Reshape to 6x4
y = x.reshape(6, 4)  # Same as view
```

**Indexing and Slicing:**
```python
x = torch.randn(5, 3)

# Indexing
first_row = x[0]  # First row
first_col = x[:, 0]  # First column
element = x[0, 1]  # Element at row 0, col 1

# Slicing
first_two_rows = x[:2]  # First 2 rows
last_col = x[:, -1]  # Last column
```

### 2. Autograd (Automatic Differentiation)

**What is Autograd?**
Autograd automatically computes gradients (derivatives) of tensors. This is what makes backpropagation work.

**How it works:**
```python
# Create tensor with requires_grad=True
x = torch.tensor([2.0], requires_grad=True)

# Define computation
y = x ** 2  # y = x²

# Compute gradient
y.backward()  # Computes dy/dx

# Access gradient
print(x.grad)  # Should be 4.0 (dy/dx = 2x = 2*2 = 4)
```

**Why requires_grad?**
- `requires_grad=True`: Track operations for gradient computation
- `requires_grad=False`: Don't track (saves memory, faster)

**Common Pattern:**
```python
# During training: track gradients
x = torch.randn(3, 4, requires_grad=True)

# During inference: no gradients needed
with torch.no_grad():
    output = model(x)  # Faster, no gradient tracking
```

### 3. Neural Network Layers

**Linear Layer (Fully Connected):**
```python
import torch.nn as nn

# Linear layer: y = xW^T + b
# Input size: 10, Output size: 5
linear = nn.Linear(10, 5)

x = torch.randn(32, 10)  # Batch of 32, 10 features
output = linear(x)  # Shape: (32, 5)
```

**Activation Functions:**
```python
# ReLU
relu = nn.ReLU()
output = relu(x)  # max(0, x)

# Sigmoid
sigmoid = nn.Sigmoid()
output = sigmoid(x)  # 1 / (1 + exp(-x))

# Tanh
tanh = nn.Tanh()
output = tanh(x)

# Can also use functional
import torch.nn.functional as F
output = F.relu(x)
output = F.sigmoid(x)
```

**Building a Simple Network:**
```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Usage
model = SimpleNet(10, 20, 5)
x = torch.randn(32, 10)
output = model(x)  # Shape: (32, 5)
```

### 4. Loss Functions

**Common Loss Functions:**
```python
# Mean Squared Error (for regression)
criterion = nn.MSELoss()
pred = torch.randn(10, 1)
target = torch.randn(10, 1)
loss = criterion(pred, target)

# Cross Entropy (for classification)
criterion = nn.CrossEntropyLoss()
pred = torch.randn(10, 3)  # 10 samples, 3 classes
target = torch.randint(0, 3, (10,))  # Class indices
loss = criterion(pred, target)

# Binary Cross Entropy (for binary classification)
criterion = nn.BCELoss()
pred = torch.sigmoid(torch.randn(10, 1))  # Probabilities
target = torch.randint(0, 2, (10, 1)).float()
loss = criterion(pred, target)
```

### 5. Optimizers

**Common Optimizers:**
```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (most common)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW (better weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Using optimizer
optimizer.zero_grad()  # Clear gradients
loss.backward()  # Compute gradients
optimizer.step()  # Update weights
```

### 6. Training Loop

**Complete Training Loop:**
```python
model = SimpleNet(10, 20, 5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()  # Set to training mode
    
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

**Why zero_grad()?**
- Gradients accumulate by default
- `zero_grad()` clears gradients from previous iteration
- Must call before each `backward()`

### 7. Device Management (CPU/GPU)

**Moving to GPU:**
```python
# Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model = model.to(device)

# Move data to device
x = x.to(device)
y = y.to(device)

# Or create directly on device
x = torch.randn(10, 5).to(device)
```

**Best Practice:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet(10, 20, 5).to(device)

for batch_x, batch_y in dataloader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    # ... rest of training
```

### 8. Data Loading

**Dataset and DataLoader:**
```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset
dataset = MyDataset(X, y)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,  # Shuffle for training
    num_workers=2  # Parallel data loading
)

# Use in training
for batch_x, batch_y in dataloader:
    # batch_x shape: (32, features)
    # batch_y shape: (32,)
    pass
```

## Common Patterns

### Pattern 1: Training with Validation

```python
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch_x, batch_y in train_loader:
        # ... training code
    
    # Validation
    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # No gradients needed
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            # ... compute metrics
```

### Pattern 2: Saving and Loading Models

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = SimpleNet(10, 20, 5)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Pattern 3: Gradient Clipping

```python
# Prevent gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Quick Reference

See `pytorch_basics.py` for complete code examples.

## Exercises

1. Create tensors and perform operations
2. Build a simple neural network
3. Write a complete training loop
4. Use GPU if available

## Next Steps

- Use these concepts in all neural network topics
- Reference this when writing PyTorch code
