"""
Regularization from Scratch
Interview question: "Explain L1 vs L2 regularization"
"""
import numpy as np

# ==================== L1 Regularization (Lasso) ====================

def l1_regularization_loss(weights: np.ndarray, lambda_reg: float) -> float:
    """
    L1 Regularization: lambda * sum(|w|)
    
    Effect: Many weights become exactly 0 (sparsity)
    Use: Feature selection, interpretability
    """
    return lambda_reg * np.sum(np.abs(weights))

def l1_gradient(weights: np.ndarray, lambda_reg: float) -> np.ndarray:
    """Gradient of L1 regularization"""
    return lambda_reg * np.sign(weights)

# ==================== L2 Regularization (Ridge) ====================

def l2_regularization_loss(weights: np.ndarray, lambda_reg: float) -> float:
    """
    L2 Regularization: lambda * sum(w^2)
    
    Effect: Shrinks weights toward 0 (but not exactly 0)
    Use: Most common, improves generalization
    """
    return lambda_reg * np.sum(weights**2)

def l2_gradient(weights: np.ndarray, lambda_reg: float) -> np.ndarray:
    """Gradient of L2 regularization"""
    return 2 * lambda_reg * weights

# ==================== Elastic Net ====================

def elastic_net_loss(weights: np.ndarray, lambda_l1: float, 
                    lambda_l2: float) -> float:
    """
    Elastic Net: Combines L1 and L2
    lambda_l1 * |w| + lambda_l2 * w^2
    """
    return lambda_l1 * np.sum(np.abs(weights)) + lambda_l2 * np.sum(weights**2)

# ==================== Dropout ====================

def dropout(x: np.ndarray, dropout_rate: float, training: bool = True) -> np.ndarray:
    """
    Dropout: Randomly zero out activations during training
    
    Args:
        x: Input activations
        dropout_rate: Probability of dropping (0.0 to 1.0)
        training: If False, scale by (1-dropout_rate) but don't drop
    """
    if not training:
        # At inference: scale by (1-dropout_rate) to maintain expected value
        return x * (1 - dropout_rate)
    
    # During training: randomly drop and scale
    mask = np.random.binomial(1, 1 - dropout_rate, x.shape).astype(np.float32)
    return x * mask / (1 - dropout_rate)  # Scale to maintain expected value

# ==================== Early Stopping ====================

class EarlyStopping:
    """
    Early Stopping: Stop training when validation loss stops improving
    Prevents overfitting
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def should_stop(self, val_loss: float) -> bool:
        """Check if should stop training"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


# Usage Example
if __name__ == "__main__":
    print("Regularization Techniques")
    print("=" * 60)
    
    # Example weights
    weights = np.array([0.5, -0.3, 1.2, -0.8, 0.1])
    lambda_reg = 0.1
    
    print(f"Weights: {weights}")
    print()
    
    # L1 regularization
    l1_loss = l1_regularization_loss(weights, lambda_reg)
    l1_grad = l1_gradient(weights, lambda_reg)
    print(f"L1 Regularization:")
    print(f"  Loss: {l1_loss:.4f}")
    print(f"  Gradient: {l1_grad}")
    print(f"  Effect: Promotes sparsity (many weights → 0)")
    
    print()
    
    # L2 regularization
    l2_loss = l2_regularization_loss(weights, lambda_reg)
    l2_grad = l2_gradient(weights, lambda_reg)
    print(f"L2 Regularization:")
    print(f"  Loss: {l2_loss:.4f}")
    print(f"  Gradient: {l2_grad}")
    print(f"  Effect: Shrinks weights toward 0")
    
    print()
    
    # Dropout
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Dropout (rate=0.5):")
    print(f"  Input: {x}")
    np.random.seed(42)
    dropped = dropout(x, dropout_rate=0.5, training=True)
    print(f"  After dropout (training): {dropped}")
    print(f"  After dropout (inference): {dropout(x, dropout_rate=0.5, training=False)}")
    
    print()
    
    # Early stopping
    early_stop = EarlyStopping(patience=3)
    val_losses = [0.5, 0.4, 0.35, 0.34, 0.33, 0.33, 0.33]
    print("Early Stopping:")
    for i, loss in enumerate(val_losses):
        should_stop = early_stop.should_stop(loss)
        print(f"  Epoch {i+1}, Val Loss: {loss:.3f}, Stop: {should_stop}")
        if should_stop:
            print(f"  → Stopped at epoch {i+1}")
            break

