"""
PyTorch Fundamentals: Complete Reference
All concepts you need to write PyTorch code in this repository
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==================== 1. TENSORS ====================

def tensor_basics():
    """
    Creating and manipulating tensors
    """
    print("1. Tensor Basics")
    print("=" * 60)
    
    # Create from list
    x = torch.tensor([1, 2, 3])
    print(f"From list: {x}")
    
    # Create from NumPy
    arr = np.array([1, 2, 3])
    x = torch.from_numpy(arr)
    print(f"From NumPy: {x}")
    
    # Zeros, ones, random
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    rand = torch.randn(2, 3)  # Normal distribution
    
    print(f"\nZeros (2x3):\n{zeros}")
    print(f"Ones (2x3):\n{ones}")
    print(f"Random (2x3):\n{rand}")
    
    # With dtype
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    print(f"\nFloat32 tensor: {x}, dtype: {x.dtype}")

def tensor_operations():
    """
    Tensor operations
    """
    print("\n2. Tensor Operations")
    print("=" * 60)
    
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    b = torch.tensor([4, 5, 6], dtype=torch.float32)
    
    # Element-wise operations
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a ** 2 = {a ** 2}")
    
    # Matrix multiplication
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    C = A @ B  # or torch.matmul(A, B)
    print(f"\nMatrix multiplication: A(3x4) @ B(4x5) = C({C.shape})")
    
    # Reshaping
    x = torch.randn(2, 3, 4)
    y = x.view(6, 4)  # Reshape to 6x4
    z = x.reshape(6, 4)  # Same as view
    print(f"\nReshape: {x.shape} -> {y.shape}")

def tensor_indexing():
    """
    Indexing and slicing
    """
    print("\n3. Tensor Indexing")
    print("=" * 60)
    
    x = torch.randn(5, 3)
    print(f"Tensor shape: {x.shape}")
    
    # Indexing
    first_row = x[0]
    first_col = x[:, 0]
    element = x[0, 1]
    
    print(f"First row: {first_row}")
    print(f"First column: {first_col}")
    print(f"Element [0,1]: {element}")

# ==================== 2. AUTOGRAD ====================

def autograd_example():
    """
    Automatic differentiation
    """
    print("\n4. Autograd (Automatic Differentiation)")
    print("=" * 60)
    
    # Create tensor with gradient tracking
    x = torch.tensor([2.0], requires_grad=True)
    
    # Define computation: y = x²
    y = x ** 2
    
    # Compute gradient: dy/dx = 2x
    y.backward()
    
    print(f"x = {x.item()}")
    print(f"y = x² = {y.item()}")
    print(f"dy/dx = {x.grad.item()}")  # Should be 4.0 (2 * 2)
    
    # Multiple variables
    x1 = torch.tensor([1.0], requires_grad=True)
    x2 = torch.tensor([2.0], requires_grad=True)
    y = x1 * x2 + x1 ** 2
    
    y.backward()
    print(f"\nMultiple variables:")
    print(f"y = x1*x2 + x1²")
    print(f"∂y/∂x1 = {x1.grad.item()}")  # Should be 4.0 (x2 + 2*x1 = 2 + 2*1)
    print(f"∂y/∂x2 = {x2.grad.item()}")  # Should be 1.0 (x1)

def no_grad_example():
    """
    When to disable gradient tracking
    """
    print("\n5. torch.no_grad()")
    print("=" * 60)
    
    x = torch.randn(10, 5, requires_grad=True)
    
    # With gradients (slower, uses memory)
    y1 = x ** 2
    print(f"With grad: requires_grad = {y1.requires_grad}")
    
    # Without gradients (faster, saves memory)
    with torch.no_grad():
        y2 = x ** 2
        print(f"Without grad: requires_grad = {y2.requires_grad}")
    
    print("\nUse torch.no_grad() for:")
    print("  - Inference (making predictions)")
    print("  - Validation")
    print("  - When you don't need gradients")

# ==================== 3. NEURAL NETWORK LAYERS ====================

def linear_layer_example():
    """
    Linear (fully connected) layer
    """
    print("\n6. Linear Layer")
    print("=" * 60)
    
    # Linear layer: y = xW^T + b
    # Input size: 10, Output size: 5
    linear = nn.Linear(10, 5)
    
    x = torch.randn(32, 10)  # Batch of 32, 10 features
    output = linear(x)  # Shape: (32, 5)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight shape: {linear.weight.shape}")  # (5, 10)
    print(f"Bias shape: {linear.bias.shape}")  # (5,)

def activation_functions():
    """
    Activation functions
    """
    print("\n7. Activation Functions")
    print("=" * 60)
    
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # ReLU: max(0, x)
    relu = nn.ReLU()
    print(f"ReLU: {relu(x)}")
    
    # Sigmoid: 1 / (1 + exp(-x))
    sigmoid = nn.Sigmoid()
    print(f"Sigmoid: {sigmoid(x)}")
    
    # Tanh
    tanh = nn.Tanh()
    print(f"Tanh: {tanh(x)}")
    
    # Functional API
    print(f"F.relu: {F.relu(x)}")
    print(f"F.sigmoid: {F.sigmoid(x)}")

def simple_network():
    """
    Building a simple neural network
    """
    print("\n8. Simple Neural Network")
    print("=" * 60)
    
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.fc1(x)  # Linear transformation
            x = self.relu(x)  # Activation
            x = self.fc2(x)  # Linear transformation
            return x
    
    # Create model
    model = SimpleNet(input_size=10, hidden_size=20, output_size=5)
    
    # Forward pass
    x = torch.randn(32, 10)  # Batch of 32
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# ==================== 4. LOSS FUNCTIONS ====================

def loss_functions():
    """
    Common loss functions
    """
    print("\n9. Loss Functions")
    print("=" * 60)
    
    # Mean Squared Error (regression)
    mse = nn.MSELoss()
    pred = torch.randn(10, 1)
    target = torch.randn(10, 1)
    loss_mse = mse(pred, target)
    print(f"MSE Loss: {loss_mse.item():.4f}")
    
    # Cross Entropy (classification)
    ce = nn.CrossEntropyLoss()
    pred = torch.randn(10, 3)  # 10 samples, 3 classes
    target = torch.randint(0, 3, (10,))  # Class indices
    loss_ce = ce(pred, target)
    print(f"Cross Entropy Loss: {loss_ce.item():.4f}")
    
    # Binary Cross Entropy (binary classification)
    bce = nn.BCELoss()
    pred = torch.sigmoid(torch.randn(10, 1))  # Probabilities [0,1]
    target = torch.randint(0, 2, (10, 1)).float()
    loss_bce = bce(pred, target)
    print(f"BCE Loss: {loss_bce.item():.4f}")

# ==================== 5. OPTIMIZERS ====================

def optimizers_example():
    """
    Optimizers
    """
    print("\n10. Optimizers")
    print("=" * 60)
    
    model = nn.Linear(10, 1)
    
    # SGD
    sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("SGD optimizer created")
    
    # Adam (most common)
    adam = optim.Adam(model.parameters(), lr=0.001)
    print("Adam optimizer created")
    
    # AdamW (better weight decay)
    adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    print("AdamW optimizer created")
    
    # Using optimizer
    criterion = nn.MSELoss()
    x = torch.randn(10, 10)
    y = torch.randn(10, 1)
    
    # Training step
    optimizer = adam
    optimizer.zero_grad()  # Clear gradients
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
    
    print(f"Loss after one step: {loss.item():.4f}")

# ==================== 6. TRAINING LOOP ====================

def complete_training_loop():
    """
    Complete training loop
    """
    print("\n11. Complete Training Loop")
    print("=" * 60)
    
    # Model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    
    # Training
    model.train()  # Set to training mode
    num_epochs = 5
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ==================== 7. DEVICE MANAGEMENT ====================

def device_management():
    """
    CPU/GPU device management
    """
    print("\n12. Device Management (CPU/GPU)")
    print("=" * 60)
    
    # Check if GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model = nn.Linear(10, 5).to(device)
    print(f"Model on device: {next(model.parameters()).device}")
    
    # Move data to device
    x = torch.randn(10, 10).to(device)
    y = model(x)
    print(f"Data and output on device: {x.device}, {y.device}")
    
    # Create directly on device
    x = torch.randn(10, 10, device=device)
    print(f"Created directly on device: {x.device}")

# ==================== 8. DATA LOADING ====================

class SimpleDataset(Dataset):
    """
    Custom dataset
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def data_loading():
    """
    Dataset and DataLoader
    """
    print("\n13. Data Loading")
    print("=" * 60)
    
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    
    # Create dataset
    dataset = SimpleDataset(X, y)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows, 2-4 for Linux/Mac
    )
    
    # Iterate through batches
    print("\nBatches:")
    for i, (batch_x, batch_y) in enumerate(dataloader):
        print(f"  Batch {i+1}: x shape = {batch_x.shape}, y shape = {batch_y.shape}")
        if i >= 2:  # Show first 3 batches
            break

# ==================== 9. COMMON PATTERNS ====================

def training_with_validation():
    """
    Training with validation pattern
    """
    print("\n14. Training with Validation Pattern")
    print("=" * 60)
    
    model = nn.Sequential(nn.Linear(10, 5))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Dummy data
    train_X = torch.randn(80, 10)
    train_y = torch.randint(0, 5, (80,))
    val_X = torch.randn(20, 10)
    val_y = torch.randint(0, 5, (20,))
    
    for epoch in range(3):
        # Training
        model.train()
        train_outputs = model(train_X)
        train_loss = criterion(train_outputs, train_y)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()  # Set to evaluation mode
        with torch.no_grad():  # No gradients needed
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss.item():.4f}, "
              f"Val Loss = {val_loss.item():.4f}")

def model_saving_loading():
    """
    Save and load models
    """
    print("\n15. Model Saving and Loading")
    print("=" * 60)
    
    # Create and train model
    model = nn.Sequential(nn.Linear(10, 5))
    
    # Save model
    torch.save(model.state_dict(), 'model_temp.pth')
    print("Model saved to model_temp.pth")
    
    # Load model
    new_model = nn.Sequential(nn.Linear(10, 5))
    new_model.load_state_dict(torch.load('model_temp.pth'))
    new_model.eval()
    print("Model loaded from model_temp.pth")
    
    # Clean up
    import os
    if os.path.exists('model_temp.pth'):
        os.remove('model_temp.pth')

def gradient_clipping():
    """
    Gradient clipping to prevent explosion
    """
    print("\n16. Gradient Clipping")
    print("=" * 60)
    
    model = nn.Sequential(nn.Linear(10, 5))
    optimizer = optim.Adam(model.parameters())
    
    x = torch.randn(10, 10)
    y = torch.randn(10, 5)
    
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    
    # Clip gradients to max norm of 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    print("Gradients clipped to max_norm=1.0")

# ==================== MAIN ====================

if __name__ == "__main__":
    print("PyTorch Fundamentals: Complete Reference")
    print("=" * 60)
    
    tensor_basics()
    tensor_operations()
    tensor_indexing()
    autograd_example()
    no_grad_example()
    linear_layer_example()
    activation_functions()
    simple_network()
    loss_functions()
    optimizers_example()
    complete_training_loop()
    device_management()
    data_loading()
    training_with_validation()
    model_saving_loading()
    gradient_clipping()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Tensors: Multi-dimensional arrays with GPU support")
    print("  - Autograd: Automatic differentiation for backpropagation")
    print("  - nn.Module: Base class for neural networks")
    print("  - Loss functions: Measure prediction error")
    print("  - Optimizers: Update model parameters")
    print("  - Training loop: Forward → Loss → Backward → Update")
    print("  - Device: Move to GPU for faster computation")
    print("  - DataLoader: Efficient batch loading")
    print("\nUse this as reference when writing PyTorch code!")

