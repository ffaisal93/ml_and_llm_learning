"""
Linear Regression using PyTorch
Simple PyTorch implementation for comparison
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearRegressionTorch(nn.Module):
    """
    Linear Regression using PyTorch
    Simple neural network with one linear layer
    """
    
    def __init__(self, input_size: int = 1):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x)


# Usage Example
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 1).astype(np.float32)
    y = (2 * X.flatten() + 1 + 0.1 * np.random.randn(100)).astype(np.float32)
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).unsqueeze(1)
    
    # Create model
    model = LinearRegressionTorch(input_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training
    n_epochs = 1000
    for epoch in range(n_epochs):
        # Forward pass
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    # Get learned parameters
    weight = model.linear.weight.data.item()
    bias = model.linear.bias.data.item()
    
    print(f"\nLearned weight: {weight:.4f}")
    print(f"Learned bias: {bias:.4f}")

