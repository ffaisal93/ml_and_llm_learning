"""
Neural Network from Scratch
Complete implementation with detailed backpropagation
"""
import numpy as np
from typing import List, Tuple

class NeuralNetwork:
    """
    Simple 2-layer neural network from scratch
    
    Architecture:
    Input → Hidden Layer (with activation) → Output Layer (with activation)
    
    Forward Pass:
    z1 = W1 @ x + b1
    h1 = activation(z1)
    z2 = W2 @ h1 + b2
    y = activation(z2)
    
    Backpropagation:
    Computes gradients using chain rule
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        
        # Store activations for backpropagation
        self.z1 = None
        self.h1 = None
        self.z2 = None
        self.h2 = None
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
        
        Why: Introduces non-linearity, outputs in (0, 1)
        Problem: Vanishing gradients for large |x|
        """
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid: σ'(x) = σ(x)(1 - σ(x))
        
        This is used in backpropagation to compute gradients
        """
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation: ReLU(x) = max(0, x)
        
        Why: Solves vanishing gradient problem
        Advantage: Fast computation, strong gradients
        Problem: Dead ReLU (outputs 0 if input < 0)
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU: 1 if x > 0, else 0
        """
        return (x > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network
        
        Detailed steps:
        1. Input X (batch_size, input_size)
        2. Layer 1: z1 = W1 @ X^T + b1 → (hidden_size, batch_size)
        3. Activation: h1 = σ(z1)
        4. Layer 2: z2 = W2 @ h1 + b2 → (output_size, batch_size)
        5. Activation: h2 = σ(z2) → (output_size, batch_size)
        6. Return h2^T → (batch_size, output_size)
        """
        # Store for backpropagation
        self.X = X.T  # (input_size, batch_size)
        
        # Layer 1: Linear transformation
        # z1 = W1 @ X + b1
        # W1: (hidden_size, input_size), X: (input_size, batch_size)
        # Result: (hidden_size, batch_size)
        self.z1 = self.W1 @ self.X + self.b1
        
        # Activation
        self.h1 = self.relu(self.z1)  # (hidden_size, batch_size)
        
        # Layer 2: Linear transformation
        # z2 = W2 @ h1 + b2
        # W2: (output_size, hidden_size), h1: (hidden_size, batch_size)
        # Result: (output_size, batch_size)
        self.z2 = self.W2 @ self.h1 + self.b2
        
        # Output activation
        self.h2 = self.sigmoid(self.z2)  # (output_size, batch_size)
        
        return self.h2.T  # (batch_size, output_size)
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        """
        Backpropagation: Compute gradients using chain rule
        
        Detailed explanation:
        
        We want to compute: ∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2
        
        Using chain rule:
        
        For output layer (Layer 2):
        1. Loss gradient: ∂L/∂h2 = 2(h2 - y) for MSE
        2. Through activation: ∂L/∂z2 = (∂L/∂h2) × σ'(z2)
        3. Through linear: ∂L/∂W2 = (∂L/∂z2) @ h1^T
        4. Bias: ∂L/∂b2 = ∂L/∂z2 (sum over batch)
        
        For hidden layer (Layer 1):
        1. Backpropagate: ∂L/∂h1 = W2^T @ (∂L/∂z2)
        2. Through activation: ∂L/∂z1 = (∂L/∂h1) × ReLU'(z1)
        3. Through linear: ∂L/∂W1 = (∂L/∂z1) @ X^T
        4. Bias: ∂L/∂b1 = ∂L/∂z1 (sum over batch)
        """
        m = X.shape[0]  # Batch size
        
        # Output layer gradients
        # Loss: MSE = (1/m) Σ(y_pred - y_true)²
        # ∂L/∂h2 = 2(h2 - y) / m
        dL_dh2 = 2 * (output - y) / m  # (batch_size, output_size)
        dL_dh2 = dL_dh2.T  # (output_size, batch_size)
        
        # Through sigmoid activation
        # ∂L/∂z2 = (∂L/∂h2) × σ'(z2)
        dL_dz2 = dL_dh2 * self.sigmoid_derivative(self.z2)  # (output_size, batch_size)
        
        # Gradients for W2 and b2
        # ∂L/∂W2 = (∂L/∂z2) @ h1^T
        # dL_dz2: (output_size, batch_size), h1: (hidden_size, batch_size)
        # Result: (output_size, hidden_size)
        dL_dW2 = dL_dz2 @ self.h1.T
        
        # ∂L/∂b2 = sum of dL_dz2 over batch dimension
        dL_db2 = np.sum(dL_dz2, axis=1, keepdims=True)
        
        # Hidden layer gradients
        # Backpropagate through W2
        # ∂L/∂h1 = W2^T @ (∂L/∂z2)
        # W2^T: (hidden_size, output_size), dL_dz2: (output_size, batch_size)
        # Result: (hidden_size, batch_size)
        dL_dh1 = self.W2.T @ dL_dz2
        
        # Through ReLU activation
        # ∂L/∂z1 = (∂L/∂h1) × ReLU'(z1)
        dL_dz1 = dL_dh1 * self.relu_derivative(self.z1)  # (hidden_size, batch_size)
        
        # Gradients for W1 and b1
        # ∂L/∂W1 = (∂L/∂z1) @ X^T
        # dL_dz1: (hidden_size, batch_size), X: (input_size, batch_size)
        # Result: (hidden_size, input_size)
        dL_dW1 = dL_dz1 @ self.X.T
        
        # ∂L/∂b1 = sum of dL_dz1 over batch dimension
        dL_db1 = np.sum(dL_dz1, axis=1, keepdims=True)
        
        # Update weights using gradient descent
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Mean Squared Error loss
        
        L = (1/m) Σ(y_pred - y_true)²
        """
        m = y_pred.shape[0]
        return np.mean((y_pred - y_true)**2)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000,
              batch_size: int = None, verbose: bool = True):
        """
        Training loop
        
        Steps:
        1. Forward pass: Compute predictions
        2. Compute loss
        3. Backward pass: Compute gradients
        4. Update weights
        5. Repeat for all epochs
        """
        if batch_size is None:
            batch_size = X.shape[0]
        
        losses = []
        
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Backward pass
                self.backward(X_batch, y_batch, output)
            
            # Compute loss on full dataset
            output = self.forward(X)
            loss = self.compute_loss(output, y)
            losses.append(loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.forward(X)


# Usage Example
if __name__ == "__main__":
    print("Neural Network from Scratch")
    print("=" * 60)
    
    # Generate sample data (XOR problem)
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR
    
    print("Training Data (XOR problem):")
    print("  Input | Output")
    for i in range(len(X)):
        print(f"  {X[i]} | {y[i][0]}")
    print()
    
    # Create and train network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    
    print("Network Architecture:")
    print(f"  Input: 2 features")
    print(f"  Hidden: 4 neurons (ReLU activation)")
    print(f"  Output: 1 neuron (Sigmoid activation)")
    print()
    
    print("Training...")
    losses = nn.train(X, y, epochs=1000, verbose=True)
    
    print("\nPredictions:")
    predictions = nn.predict(X)
    for i in range(len(X)):
        print(f"  Input: {X[i]}, True: {y[i][0]}, Predicted: {predictions[i][0]:.4f}")
    
    print(f"\nFinal Loss: {losses[-1]:.4f}")

