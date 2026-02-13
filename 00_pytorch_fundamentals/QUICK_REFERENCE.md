# PyTorch Quick Reference

Quick reference for common PyTorch operations you'll need in this repository.

## Creating Tensors

```python
# From list
x = torch.tensor([1, 2, 3])

# Zeros, ones, random
x = torch.zeros(3, 4)
x = torch.ones(2, 3)
x = torch.randn(2, 3)  # Normal distribution

# From NumPy
x = torch.from_numpy(np_array)
```

## Operations

```python
# Element-wise
c = a + b
c = a * b

# Matrix multiplication
C = A @ B  # or torch.matmul(A, B)

# Reshape
y = x.view(6, 4)
y = x.reshape(6, 4)
```

## Autograd

```python
# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)

# Compute
y = x ** 2

# Backward
y.backward()

# Access gradient
print(x.grad)  # 4.0
```

## Neural Network

```python
# Linear layer
linear = nn.Linear(10, 5)
output = linear(x)

# Activation
relu = nn.ReLU()
output = relu(x)

# Simple network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)
```

## Loss Functions

```python
# MSE (regression)
loss = nn.MSELoss()(pred, target)

# Cross Entropy (classification)
loss = nn.CrossEntropyLoss()(pred, target)

# BCE (binary classification)
loss = nn.BCELoss()(pred, target)
```

## Optimizer

```python
# Create
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use
optimizer.zero_grad()  # Clear gradients
loss.backward()  # Compute gradients
optimizer.step()  # Update weights
```

## Training Loop

```python
model.train()
for batch_x, batch_y in dataloader:
    # Forward
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Device (GPU)

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
```

## No Grad (Inference)

```python
model.eval()
with torch.no_grad():
    outputs = model(x)
```

## Common Mistakes to Avoid

1. **Forgetting zero_grad()**: Always call before backward()
2. **Using requires_grad in inference**: Use torch.no_grad()
3. **Not setting model.train()/eval()**: Affects dropout, batch norm
4. **Wrong device**: Make sure model and data on same device

