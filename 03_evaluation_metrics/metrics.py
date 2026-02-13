"""
Evaluation Metrics from Scratch
Interview question: "Implement precision, recall, F1 from scratch"
"""
import numpy as np
from typing import Tuple, Optional

# ==================== Classification Metrics ====================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy: (TP + TN) / (TP + TN + FP + FN)
    Simple: correct predictions / total predictions
    """
    return np.mean(y_true == y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray, 
              positive_label: int = 1) -> float:
    """
    Precision: TP / (TP + FP)
    Of all positive predictions, how many were correct?
    """
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true: np.ndarray, y_pred: np.ndarray,
           positive_label: int = 1) -> float:
    """
    Recall: TP / (TP + FN)
    Of all actual positives, how many did we catch?
    """
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             positive_label: int = 1) -> float:
    """
    F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    Harmonic mean of precision and recall
    """
    prec = precision(y_true, y_pred, positive_label)
    rec = recall(y_true, y_pred, positive_label)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion Matrix
    Rows = true labels, Columns = predicted labels
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for true_cls, pred_cls in zip(y_true, y_pred):
        true_idx = class_to_idx[true_cls]
        pred_idx = class_to_idx[pred_cls]
        cm[true_idx, pred_idx] += 1
    
    return cm

# ==================== Regression Metrics ====================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error
    Penalizes large errors more
    """
    return np.mean((y_true - y_pred)**2)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error
    Equal weight to all errors
    """
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error
    Same units as target variable
    """
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² Score (Coefficient of Determination)
    1 - (SS_res / SS_tot)
    Proportion of variance explained
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error
    Percentage-based error metric
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Classification example
    print("Classification Metrics:")
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    
    print(f"Accuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {precision(y_true, y_pred):.4f}")
    print(f"Recall: {recall(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    
    print("\n" + "="*50 + "\n")
    
    # Regression example
    print("Regression Metrics:")
    y_true = np.array([3, -0.5, 2, 7, 4.2])
    y_pred = np.array([2.5, 0.0, 2.1, 7.8, 5.3])
    
    print(f"MSE: {mse(y_true, y_pred):.4f}")
    print(f"MAE: {mae(y_true, y_pred):.4f}")
    print(f"RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"MAPE: {mape(y_true, y_pred):.4f}%")

