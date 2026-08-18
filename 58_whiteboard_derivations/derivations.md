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

> **Saying it out loud (narrate while you write).** "Let me set up the pieces first — I'll write `z equals x transpose w plus b`, then `p equals sigmoid of z`, and underneath, the binary cross-entropy loss. I want `dL/dw`, so I'll go through `z` by the chain rule: `dL/dw` is `dL/dp` times `dp/dz` times `dz/dw`. Taking the first factor, differentiating the loss with respect to `p` gives me `minus y over p plus one minus y over one minus p` — I'll put that over a common denominator, so it's `p minus y` on top and `p times one minus p` underneath. Now the second factor: the sigmoid has that lovely property that its derivative is `p times one minus p` — I'll write that here. And you can see those cancel exactly" — draw the strike-through — "so `dL/dz` is just `p minus y`. The last factor `dz/dw` is `x_i`, so stacking over the batch gives me `X transpose times p minus y, divided by n`, and the bias gradient is the same thing without the `X`, just the mean." Then land it: that cancellation is the whole point, and it's why you never pair sigmoid with MSE — there the `p times one minus p` survives and kills the gradient exactly where you're most confidently wrong.

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

> **Saying it out loud (narrate while you write).** "This is the same story as the binary case, one dimension up. I'll write the softmax, then cross-entropy against a one-hot target — and I'll note that since `y` is one-hot, the sum collapses to a single term, `minus log p` of the true class. Now the derivative of the softmax has two cases, so let me write them both: when I differentiate `p_j` with respect to its own logit I get `p_j times one minus p_j`, and with respect to a different logit `z_k` I get `minus p_j p_k`. Chaining through and summing over the classes, the true-class term contributes `p minus 1` and every other class contributes just its own `p`" — and that's exactly the vector `p minus y`. Close it: same shape of answer as sigmoid plus BCE, and it's not a coincidence — it's what you always get pairing an exponential-family likelihood with its natural link. Practical consequence: the gradient is bounded between minus one and one per class, which is why fused softmax-cross-entropy is both stable and fast.

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

> **Saying it out loud (narrate while you write).** "Standard MLE recipe, four moves, and I'll say them as I go. Write the likelihood for one sample — `p` to the `x`, times `one minus p` to the `one minus x`, which is just a compact way of saying `p` if it's a one and `one minus p` if it's a zero. Multiply over independent samples. Take logs, because the product becomes a sum and the maximizer doesn't move" — write the log-likelihood. "Differentiate with respect to `p`: I get `the number of ones over p` minus `the number of zeros over one minus p`. Set that to zero, cross-multiply" — and you land on `p hat equals the sample mean`. Then say why it's satisfying: the most likely bias for the coin is exactly the fraction of heads you saw, which is what everyone would have guessed, and MLE just proved it. Worth adding the caveat: with three flips and three heads, MLE says the coin never lands tails — that's the failure mode a prior or Laplace smoothing exists to fix.

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

> **Saying it out loud (narrate while you write).** "Same recipe as Bernoulli, two parameters instead of one. I'll write the Gaussian log-likelihood — and immediately note the useful shape: it's `minus n over two log sigma squared`, minus the sum of squared deviations over `two sigma squared`, plus a constant I can drop. Differentiate with respect to `mu` first: only the squared-error term depends on it, and setting it to zero gives the sample mean — so maximizing the likelihood is the same as minimizing squared error, which is the reason least squares and Gaussian noise are the same assumption. Now substitute that back and differentiate with respect to `sigma squared`" — and you land on the average squared deviation, dividing by `n`. Then say the part they're fishing for: this MLE is biased downward, because you used the same data to estimate `mu`, and the fix is `n minus 1`. Two different objectives — MLE maximizes likelihood, the `n minus 1` version is unbiased — and you cannot have both.

---

## 5. Why `n - 1` for Sample Variance?

### Core Intuition

Once we estimate the sample mean from the same data, one degree of freedom is used up.

If you subtract the sample mean, the centered values must sum to zero, so only `n - 1` of them are free to vary independently.

### Interview Answer

"The correction is there to remove the downward bias in the naive variance estimate after using the sample mean estimated from the same sample."

> **Saying it out loud (narrate while you write).** "Here's the intuition without any algebra. I'll write the `n` deviations from the sample mean — and now notice this" — write `sum of (x_i minus x-bar) equals 0`. "They're forced to sum to zero. So if you tell me `n minus 1` of them, I can compute the last one exactly; it isn't free. That's what a degree of freedom is, and I've spent one estimating the mean from the same data I'm now measuring spread on." Then the deeper reason: the sample mean is, by construction, the point that minimizes the sum of squared deviations, so deviations from it are systematically smaller than deviations from the true mean would be. Dividing by `n` therefore underestimates the variance every single time, and `n minus 1` is exactly the correction that makes it unbiased. The number to land on: with `n` equal to 2 the correction is a factor of two, and by `n` equal to 100 it's one percent and nobody cares.

---

## 6. Confidence Interval for a Mean

### Standard Form

`mean +/- critical_value * standard_error`

where:

`standard_error = sample_std / sqrt(n)`

### What to Say

"The standard error shrinks like `1/sqrt(n)`, which is why larger sample sizes give tighter intervals."

> **Saying it out loud (narrate while you write).** "Every confidence interval has the same three parts, so let me lay them out: an estimate, a measure of how noisy that estimate is, and a multiplier for how confident I want to be. The estimate is the sample mean. The noise is the standard error, which is the sample standard deviation over root `n` — and the key thing to say is that this is the spread of the *mean*, not of the data; the data doesn't get tighter as you collect more, your estimate of its center does. The multiplier is about 1.96 for 95 percent under a normal, or a `t` value with `n minus 1` degrees of freedom when `n` is small and I've estimated the standard deviation." Then the punchline: the interval shrinks like one over root `n`, so cutting your error bar in half costs four times the data — that's the number to quote. And the interpretation caveat, because interviewers probe it: the 95 percent refers to the procedure across repeated experiments, not to the probability that this particular interval contains the truth.

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

> **Saying it out loud (narrate while you write).** "I'll do this by shapes, left to right, because shapes are where the bugs live. `Q` is sequence length by `d_k`, `K` is the same, so `Q` times `K` transpose is sequence by sequence" — write the box and say what it *is*: "that's a token-to-token relevance matrix, row `i` telling me how much token `i` cares about every other token. I divide by root `d_k` so the scores don't blow up with dimension, then softmax along the row — over keys, the last axis — so each row sums to one. Multiplying by `V`, which is sequence by `d_v`, contracts the sequence dimension away and leaves me sequence by `d_v`: same number of tokens I started with, each one now a weighted blend of all the values." Land on the cost: that middle matrix is the quadratic term, `n` squared in both time and memory, which is the entire reason FlashAttention, sliding windows, and state-space models exist.

---

## 8. Bias-Variance Intuition

### Standard Decomposition

`MSE = Bias^2 + Variance + Noise`

### What to Say

"Bias is average systematic error. Variance is sensitivity to the training sample. More flexible models often reduce bias and increase variance."

> **Saying it out loud (narrate while you write).** "Let me build this up rather than quote it. Fix a test point, and imagine retraining the model on many different samples from the same distribution. Bias is the gap between the *average* prediction across all those models and the truth — a systematic miss that more data won't cure, because it's a limitation of the model class. Variance is how much those predictions scatter around their own average — that's sensitivity to which particular sample you happened to get. And noise is the irreducible part, the floor you can't get under no matter what." Draw the three-term decomposition and say the crossing point: as you make the model more flexible, bias falls and variance rises, so test error traces a U and the goal is the bottom, not the left edge. Then add the honest modern footnote: for hugely over-parameterized networks the U keeps going and test error descends a second time, so the classical picture is a good intuition and an incomplete law.

