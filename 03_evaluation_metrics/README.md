# Topic 3: Evaluation Metrics

> 🔥 **For interviews, read these first:**
> - **`EVALUATION_METRICS_DEEP_DIVE.md`** — frontier-lab interview deep dive: classification (precision/recall/F1/AUROC/PR-AUC), regression (MSE/MAE/R²/quantile loss), ranking (MAP/NDCG/MRR), LLM-specific (PPL, pass@k, BLEU, LLM-as-judge biases), calibration (Brier/ECE/temperature scaling), Goodhart's Law and methodology pitfalls.
> - **`INTERVIEW_GRILL.md`** — 50 active-recall questions with strong answers.

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

## Core Intuition

Metrics are not just for reporting a number after training.

They define what "good" means for the problem.

That is why interviewers care so much about them: if you choose the wrong metric, you can optimize the wrong behavior.

### Classification

For classification, different metrics care about different kinds of mistakes.

- **Accuracy** treats all mistakes equally
- **Precision** asks: when I predict positive, how often am I right?
- **Recall** asks: among true positives, how many did I recover?
- **F1** balances precision and recall

### Regression

For regression, the main question is how errors are penalized.

- **MSE** punishes large errors more strongly
- **MAE** treats errors linearly
- **R2** measures variance explained relative to predicting the mean

### Ranking

Ranking metrics care about order, not just set membership.

That is why search and recommendation systems need metrics like NDCG or MAP rather than plain classification accuracy.

## Technical Details That Commonly Get Missed

### Accuracy Can Be Misleading

If positives are rare, accuracy can look great even for a useless model.

Example:
- 99% negative data
- always predict negative
- 99% accuracy
- terrible recall for the positive class

### Precision vs Recall Trade-Off

You often improve one at the expense of the other by changing the threshold.

That means the metric is not just about the model. It is also about:
- threshold choice
- business cost
- tolerance for false positives vs false negatives

### R2 Edge Case

`R2 = 1 - SS_res / SS_tot`

Important edge case:
- if `SS_tot = 0`, the target has no variance
- then R2 is not informative in the normal way

### Ranking Metrics Need Position Sensitivity

NDCG is useful because relevant items near the top matter more than relevant items buried lower in the list.

That is usually what you want in retrieval and recommendation.

## Common Failure Modes

- using accuracy for heavy class imbalance
- reporting F1 without saying threshold
- comparing regression metrics across differently scaled targets without context
- using perplexity or loss as if it directly captured downstream usefulness
- forgetting confidence intervals for small evaluation sets

## Edge Cases and Follow-Up Questions

1. What if the positive class is only 0.1%?
2. What if false negatives are much more costly than false positives?
3. Why can precision rise when recall falls?
4. Why might two models have similar accuracy but very different usefulness?
5. Why is NDCG better than accuracy for search ranking?

## What to Practice Saying Out Loud

1. Why metric choice is really objective choice
2. Why threshold matters for classification metrics
3. Why MSE and MAE can disagree about which model is better
4. Why ranking metrics need position sensitivity

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
