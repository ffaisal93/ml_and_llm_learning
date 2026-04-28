# Topic 32: Anomaly Detection & Isolation Forest

> 🔥 **For interviews, read these first:**
> - **`ANOMALY_DETECTION_DEEP_DIVE.md`** — frontier-lab deep dive: statistical methods (z-score, Mahalanobis), density-based (KDE, LOF), Isolation Forest score derivation, One-Class SVM, autoencoder reconstruction, embedding-based AD with foundation models, time-series anomalies (point/contextual/collective), evaluation (AUPRC > AUC for imbalance).
> - **`INTERVIEW_GRILL.md`** — 50 active-recall questions.

## What You'll Learn

This topic covers anomaly detection in detail:
- Isolation Forest (detailed explanation)
- How it works mathematically
- When to use it
- Comparison with other methods
- Simple implementation

## Why We Need This

### Interview Importance
- **Common question**: "Explain Isolation Forest"
- **Practical knowledge**: Used in fraud detection, system monitoring
- **Understanding**: Shows knowledge of unsupervised learning

### Real-World Application
- **Fraud detection**: Identify unusual transactions
- **System monitoring**: Detect anomalies in metrics
- **Quality control**: Find defective products
- **Network security**: Detect intrusions

## Core Intuition

Anomaly detection is about finding points that do not fit the usual pattern of the data.

The challenge is that anomalies are often:
- rare
- diverse
- poorly labeled

That is why unsupervised approaches like Isolation Forest are useful.

### Why Isolation Forest Is Different

Many anomaly methods ask:
- how far is this point from normal data?
- how low is the local density?

Isolation Forest asks:
- how quickly can random partitioning isolate this point?

That difference in viewpoint is the core idea.

## Isolation Forest: Detailed Explanation

### What is Isolation Forest?

**Isolation Forest is an anomaly detection algorithm based on the principle that anomalies are easier to isolate (separate) than normal points.**

**Key Insight:**
- Normal points are in dense regions (hard to isolate)
- Anomalies are in sparse regions (easy to isolate)
- Anomalies have shorter path lengths in isolation trees

## Technical Details Interviewers Often Want

### Why Short Path Length Signals Anomaly

If a point is unusual, random splits tend to separate it quickly because there are fewer nearby points keeping it inside a dense region.

### Why the Forest Helps

A single random tree is noisy.

Using many trees:
- reduces variance
- stabilizes the score
- avoids over-interpreting one random partition

### What Contamination Controls

Contamination mainly affects the threshold for calling something anomalous.

It is not the anomaly detector itself.

### How It Works (Step-by-Step)

**Step 1: Build Isolation Trees**

An isolation tree is a binary tree built by randomly selecting a feature and split value:

```
Algorithm: Build Isolation Tree
1. Randomly select a feature
2. Randomly select a split value between min and max of that feature
3. Split data: left if value < split, right if value >= split
4. Recursively build left and right subtrees
5. Stop when: tree reaches max depth OR only one sample remains
```

**Why random splits?**
- Normal points need many splits to isolate (long path)
- Anomalies need few splits to isolate (short path)
- Random splits ensure anomalies are isolated quickly

**Step 2: Compute Path Length**

Path length = number of edges from root to leaf

- **Normal point**: Long path (many splits needed)
- **Anomaly**: Short path (few splits needed)

**Step 3: Anomaly Score**

The anomaly score is computed as:

```
s(x, n) = 2^(-E(h(x)) / c(n))

Where:
- h(x): Average path length across all trees
- E(h(x)): Expected path length
- c(n): Normalization constant (average path length for unsuccessful search)
- n: Number of samples
```

**Interpretation:**
- s ≈ 1: Anomaly (short path, easy to isolate)
- s ≈ 0: Normal (long path, hard to isolate)
- s ≈ 0.5: Borderline

**Step 4: Ensemble**

Build multiple isolation trees (forest):
- Each tree uses random subset of data
- Average path lengths across all trees
- More robust than single tree

### Mathematical Details

**Normalization Constant c(n):**

```
c(n) = 2H(n-1) - 2(n-1)/n

Where H(n) is the harmonic number:
H(n) = 1 + 1/2 + 1/3 + ... + 1/n ≈ ln(n) + γ

(γ is Euler-Mascheroni constant ≈ 0.577)
```

**Why this normalization?**
- Adjusts for different tree sizes
- Makes scores comparable across different datasets
- Expected path length for normal points ≈ c(n)

**Anomaly Score Formula Explained:**

```
s(x, n) = 2^(-E(h(x)) / c(n))

If E(h(x)) << c(n): s → 1 (anomaly)
If E(h(x)) ≈ c(n): s → 0.5 (normal)
If E(h(x)) >> c(n): s → 0 (very normal)
```

### Why Isolation Forest Works

**Advantages:**
1. **No need for labeled data**: Unsupervised
2. **Handles high dimensions**: Works well even with many features
3. **Fast**: O(n log n) complexity
4. **Handles multiple anomaly types**: Works for various anomaly patterns
5. **Interpretable**: Can see which features contribute to isolation

**How it differs from other methods:**

**vs. K-Means:**
- K-Means assumes clusters, Isolation Forest doesn't
- Isolation Forest better for high dimensions

**vs. DBSCAN:**
- DBSCAN needs density, Isolation Forest doesn't
- Isolation Forest faster for large datasets

**vs. One-Class SVM:**
- One-Class SVM needs to define "normal" region
- Isolation Forest doesn't need this assumption

### When to Use Isolation Forest

**Use When:**
- You have unlabeled data (unsupervised)
- Anomalies are rare (< 5% of data)
- High-dimensional data
- Need fast detection
- Anomalies are different from normal (not just outliers)

**Don't Use When:**
- Anomalies are common (> 10% of data)
- Need to understand why something is anomalous (use other methods)
- Data has clear clusters (K-Means might be better)

## Common Failure Modes

- assuming anomaly detection works well when anomalies are common
- treating contamination as ground truth rather than a thresholding choice
- using Isolation Forest when interpretability of the anomaly cause is the main requirement
- treating rare-but-valid behavior as automatically anomalous

## Edge Cases and Follow-Up Questions

1. Why are anomalies easier to isolate than normal points?
2. Why does averaging many trees help?
3. What does the contamination parameter really do?
4. Why is anomaly detection difficult when anomalies are common?
5. Why is anomaly detection often unsupervised in practice?

## What to Practice Saying Out Loud

1. The intuition behind path length in Isolation Forest
2. Why dense regions make normal points harder to isolate
3. Why anomaly detection remains hard even with a simple algorithm

### Parameters

**n_estimators:**
- Number of isolation trees
- More trees = more stable, but slower
- Typical: 100-200

**max_samples:**
- Number of samples per tree
- Smaller = faster, but less stable
- Typical: 256 or 'auto'

**contamination:**
- Expected proportion of anomalies
- Used to set threshold
- Typical: 0.1 (10%) or 'auto'

**max_features:**
- Number of features to consider per split
- Typical: All features or sqrt(n_features)

## Industry Use Cases

### 1. Fraud Detection

**Problem**: Detect fraudulent credit card transactions

**Why Isolation Forest:**
- Fraud is rare (< 1% of transactions)
- High-dimensional features (amount, location, time, merchant, etc.)
- Need fast real-time detection
- Unlabeled data (don't know which are fraud)

**Implementation:**
```python
# Features: transaction amount, location, time, merchant, etc.
# Isolation Forest identifies unusual patterns
# Flag transactions with high anomaly score
```

### 2. System Monitoring

**Problem**: Detect anomalies in server metrics (CPU, memory, network)

**Why Isolation Forest:**
- Anomalies are rare (system usually normal)
- Many metrics (high-dimensional)
- Need to detect quickly
- Unlabeled data

**Implementation:**
```python
# Features: CPU usage, memory, network traffic, disk I/O, etc.
# Monitor in real-time
# Alert when anomaly score > threshold
```

### 3. Quality Control

**Problem**: Detect defective products in manufacturing

**Why Isolation Forest:**
- Defects are rare
- Many quality metrics
- Need fast detection on production line

## Comparison with Other Methods

| Method | Type | When to Use | Complexity |
|--------|------|-------------|------------|
| **Isolation Forest** | Unsupervised | Rare anomalies, high-dim | O(n log n) |
| **K-Means** | Unsupervised | Clear clusters | O(n) |
| **DBSCAN** | Unsupervised | Density-based anomalies | O(n²) |
| **One-Class SVM** | Unsupervised | Need to define normal region | O(n²) |
| **Autoencoder** | Unsupervised | Can learn complex patterns | O(n) training |

## Exercises

1. Implement Isolation Forest from scratch
2. Compare with other anomaly detection methods
3. Apply to real dataset
4. Tune parameters

## Next Steps

- Review anomaly detection methods
- Practice on real datasets
