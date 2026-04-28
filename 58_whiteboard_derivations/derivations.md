# Whiteboard Derivations

---

## 1. Logistic Regression Gradient

### Setup

Prediction:

`p_i = sigmoid(z_i)`

where:

`z_i = x_i^T w + b`

Binary cross-entropy loss:

`L = -(1/n) * sum[y_i log p_i + (1 - y_i) log(1 - p_i)]`

### Goal

Derive:

- `dL/dz_i = p_i - y_i`
- `grad_w = X^T (p - y) / n`
- `grad_b = mean(p - y)`

### Key Steps

1. Differentiate BCE with respect to `p_i`
2. Use derivative of sigmoid:

`dp_i/dz_i = p_i (1 - p_i)`

3. Simplify the expression

The cancellation gives:

`dL/dz_i = p_i - y_i`

Then:

`dL/dw = (1/n) * sum[(p_i - y_i) x_i]`

which in matrix form is:

`grad_w = X^T (p - y) / n`

### What to Say in the Interview

"The important simplification is that BCE combined with sigmoid gives a very clean derivative with respect to the logits: `p - y`."

---

## 2. Softmax + Cross-Entropy

### Setup

Softmax:

`p_j = exp(z_j) / sum_k exp(z_k)`

Cross-entropy for one-hot target `y`:

`L = -sum_j y_j log p_j`

### Result

The gradient with respect to logits is:

`dL/dz = p - y`

### Why This Matters

This is one of the most important interview derivations because it appears in almost every classification model.

### What to Say

"Just like sigmoid plus BCE in binary classification, softmax plus cross-entropy gives a very clean gradient: predicted probabilities minus target distribution."

---

## 3. Bernoulli MLE

### Setup

If `x_i` are Bernoulli samples with parameter `p`, then:

`P(x_i | p) = p^{x_i} (1 - p)^{1 - x_i}`

Likelihood:

`L(p) = product_i p^{x_i} (1 - p)^{1 - x_i}`

Log-likelihood:

`log L(p) = sum_i [x_i log p + (1 - x_i) log(1 - p)]`

### Differentiate

Set derivative to zero and solve:

`p_hat = mean(x)`

### What to Say

"The MLE for a Bernoulli parameter is just the empirical fraction of ones."

---

## 4. Gaussian MLE

### Setup

Assume `x_i ~ N(mu, sigma^2)`.

The MLEs are:

- `mu_hat = sample mean`
- `sigma^2_hat = (1/n) * sum (x_i - mu_hat)^2`

### Important Detail

This variance estimator divides by `n`, not `n - 1`.

### What to Say

"For Gaussian MLE, the variance uses division by `n`. The unbiased estimator uses `n - 1`, which is a different objective."

---

## 5. Why `n - 1` for Sample Variance?

### Core Intuition

Once we estimate the sample mean from the same data, one degree of freedom is used up.

If you subtract the sample mean, the centered values must sum to zero, so only `n - 1` of them are free to vary independently.

### Interview Answer

"The correction is there to remove the downward bias in the naive variance estimate after using the sample mean estimated from the same sample."

---

## 6. Confidence Interval for a Mean

### Standard Form

`mean +/- critical_value * standard_error`

where:

`standard_error = sample_std / sqrt(n)`

### What to Say

"The standard error shrinks like `1/sqrt(n)`, which is why larger sample sizes give tighter intervals."

---

## 7. Attention Shapes

### Setup

If:
- `Q` has shape `(seq_len, d_k)`
- `K` has shape `(seq_len, d_k)`
- `V` has shape `(seq_len, d_v)`

then:

`QK^T` has shape `(seq_len, seq_len)`

After softmax over the key dimension:

`attention_weights @ V` gives shape `(seq_len, d_v)`

### What to Say

"The attention matrix is a token-to-token relevance matrix, so its shape is sequence length by sequence length."

---

## 8. Bias-Variance Intuition

### Standard Decomposition

`MSE = Bias^2 + Variance + Noise`

### What to Say

"Bias is average systematic error. Variance is sensitivity to the training sample. More flexible models often reduce bias and increase variance."
