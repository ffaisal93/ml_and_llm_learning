"""
Logistic Regression using PyTorch
Simple PyTorch implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogisticRegressionTorch(nn.Module):
    """
    Logistic Regression using PyTorch
    """
    
    def __init__(self, input_size: int = 2):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))


# Usage Example
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Class 0
    X0 = np.random.randn(n_samples//2, 2).astype(np.float32) + np.array([-2, -2])
    y0 = np.zeros(n_samples//2, dtype=np.float32)
    
    # Class 1
    X1 = np.random.randn(n_samples//2, 2).astype(np.float32) + np.array([2, 2])
    y1 = np.ones(n_samples//2, dtype=np.float32)
    
    #explain vstack and hstack with same example
    # X0 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    # X1 = [[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
    #y0 = [0, 0, 0, 0, 0]
    #y1 = [1, 1, 1, 1, 1]
    # np.vstack([X0, X1]) = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
    # np.hstack([X0, X1]) = [[1, 2, 11, 12], [3, 4, 13, 14], [5, 6, 15, 16], [7, 8, 17, 18], [9, 10, 19, 20]]
    # np.vstack is used to stack the arrays vertically
    # np.hstack is used to stack the arrays horizontally
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    #explain unsqueeze with same example
    # X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    # y = [0, 0, 0, 0, 0]
    # torch.from_numpy(y).unsqueeze(1) = [[0], [0], [0], [0], [0]]
    # torch.from_numpy(y).unsqueeze(1) is used to add a dimension to the array to make it a column vector
    y_tensor = torch.from_numpy(y).unsqueeze(1)
    
    # Create model
    model = LogisticRegressionTorch(input_size=2)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
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
        
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    # Evaluate
    with torch.no_grad():
        predictions = model(X_tensor)
        predicted_classes = (predictions >= 0.5).float()
        accuracy = (predicted_classes == y_tensor).float().mean()
        print(f"\nAccuracy: {accuracy.item():.4f}")

