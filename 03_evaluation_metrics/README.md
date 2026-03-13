# Topic 3: Evaluation Metrics

## What You'll Learn

This topic teaches you to implement all common evaluation metrics from scratch:
- Classification metrics (Accuracy, Precision, Recall, F1)
- Regression metrics (MSE, MAE, R²)
- Ranking metrics (NDCG, MAP)
- Theory and when to use each

## Why We Need This

### Interview Importance
- **Common question**: "Implement precision/recall from scratch"
- **Understanding**: Know what metrics mean
- **Application**: Choose right metric for problem

### Real-World Application
- **Model evaluation**: Measure model performance
- **Problem-specific**: Different problems need different metrics
- **Debugging**: Understand model weaknesses

## Industry Use Cases

### 1. **Classification Metrics**
**Use Case**: Binary/multi-class classification
- Spam detection (Precision important)
- Medical diagnosis (Recall important)
- Balanced problems (F1 score)

### 2. **Regression Metrics**
**Use Case**: Continuous value prediction
- House prices (MSE, MAE)
- Model comparison (R²)

### 3. **Ranking Metrics**
**Use Case**: Recommendation systems
- Search engines (NDCG)
- Recommendations (MAP)

## Industry-Standard Boilerplate Code

### Classification Metrics (Pure Python)

```python
"""
Classification Metrics from Scratch
Interview question: "Implement precision, recall, F1"
"""
import numpy as np
from typing import Tuple

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy: (TP + TN) / (TP + TN + FP + FN)"""
    return np.mean(y_true == y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision: TP / (TP + FP)"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall: TP / (TP + FN)"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 Score: 2 * (Precision * Recall) / (Precision + Recall)"""
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Confusion Matrix"""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return cm
```

### Regression Metrics (Pure Python)

```python
"""
Regression Metrics from Scratch
"""
import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((y_true - y_pred)**2)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² Score: 1 - (SS_res / SS_tot)"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
```

## Theory

### When to Use Which Metric

**Classification:**
- **Accuracy**: Balanced classes
- **Precision**: When false positives are costly
- **Recall**: When false negatives are costly
- **F1**: Balance between precision and recall

**Regression:**
- **MSE**: Penalizes large errors more
- **MAE**: Equal weight to all errors
- **R²**: Proportion of variance explained

## Exercises

1. Implement multi-class metrics
2. Implement weighted metrics
3. Calculate metrics from confusion matrix
4. Compare different metrics

## Perplexity: Detailed Guide

**New Comprehensive Content:**

- **`perplexity_detailed.md`**: Complete theoretical guide
  - What is perplexity and intuitive understanding
  - Mathematical formulations
  - Connection to entropy and cross-entropy
  - Interpretation and typical values
  - Computing perplexity step-by-step
  - Perplexity variants (word, character, byte-level)
  - Perplexity in practice (training, evaluation, comparison)
  - Limitations and best practices
  - Related concepts (entropy, KL divergence, bits per token)
  - Applications

- **`perplexity_code.py`**: Complete implementations
  - Basic perplexity computation
  - Perplexity from logits
  - Language model perplexity
  - Per-token perplexity
  - Character-level perplexity
  - Bits per token
  - Normalized perplexity
  - Model comparison utilities

**Key Concepts:**
- Perplexity = exp(average negative log-likelihood)
- Lower perplexity = better model
- Typical values: 10-50 for good language models
- Connection: PP = 2^H (perplexity = 2^entropy)
- BPT = log₂(PP) (bits per token)

## Next Steps

- **Topic 4**: Transformers
- **Topic 5**: Attention mechanisms

