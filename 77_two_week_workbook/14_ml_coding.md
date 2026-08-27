# The ML coding round

Forty-five minutes, a shared editor, and a request to implement something small from scratch: softmax, cross-entropy, attention, k-means, k-NN, a training loop. The maths is not the difficulty. The difficulty is producing correct, running code under observation, with numerical stability handled and the shapes right the first time. Candidates lose this round by reaching for a clever vectorised one-liner, getting a shape error, and spending fifteen minutes debugging it. Write the loop. It runs, it is readable, and you can optimise it afterwards if asked.

## The equations

**Softmax with the max-subtraction trick**

$$\sigma(x)_i = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}, \qquad m = \max_j x_j$$

$x$ is the logit vector and $m$ its maximum; subtracting $m$ leaves the result unchanged because the constant cancels between numerator and denominator, but the largest exponent becomes $e^0 = 1$ so nothing overflows.

**Log-sum-exp**

$$\log \sum_j e^{x_j} = m + \log \sum_j e^{x_j - m}$$

The same shift applied to the normaliser; this is the single identity that every stable softmax, cross-entropy, and log-likelihood implementation is built on.

**Cross-entropy for one example, from logits**

$$L = -\log \sigma(x)_{y} = \Big(m + \log \sum_j e^{x_j - m}\Big) - x_y$$

$y$ is the true class index; the loss is the log-normaliser minus the true class logit, which is why you never need to form the probabilities at all.

**Gradient of softmax cross-entropy**

$$\frac{\partial L}{\partial x_i} = \sigma(x)_i - \mathbb{1}[i = y]$$

The gradient with respect to the logits is just "predicted probability minus one-hot target", which is why fusing softmax and cross-entropy is both faster and more stable than composing them.

**Sigmoid and its derivative**

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$

$z$ is a scalar pre-activation; the derivative is expressible in the output alone, which is the reason it was cheap enough to dominate early networks, and its maximum value of $0.25$ is the reason deep sigmoid stacks lose gradient.

**Scaled dot-product attention**

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V$$

$Q$ is $(T_q, d_k)$, $K$ is $(T_k, d_k)$, $V$ is $(T_k, d_v)$; $\sqrt{d_k}$ keeps the dot products at unit scale so the softmax does not saturate, and $M$ is the mask holding $-\infty$ wherever attention is forbidden.

**k-means objective**

$$J = \sum_{i=1}^{n} \big\| x_i - \mu_{c(i)} \big\|_2^2$$

$c(i)$ is the cluster assigned to point $i$ and $\mu_c$ the centroid of cluster $c$; the assign step minimises $J$ over $c$ with $\mu$ fixed and the update step minimises $J$ over $\mu$ with $c$ fixed, so $J$ decreases monotonically and this quantity is what sklearn calls `inertia_`.

## Code from memory

**1. Numerically stable softmax.** The one function you will be asked for most.

```python
import numpy as np

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)                 # subtract the max: shifts the largest exponent to 0
    shifted = x - m
    exps = np.zeros_like(shifted)
    for i in range(len(shifted)):
        exps[i] = np.exp(shifted[i])
    total = 0.0
    for i in range(len(exps)):
        total += exps[i]
    return exps / total

p = softmax([1.0, 2.0, 3.0])
print("probs:", np.round(p, 6), "sum:", p.sum())
print("big inputs:", softmax([1000.0, 1001.0, 1002.0]))
print("naive on big inputs:", np.exp([1000.0, 1001.0, 1002.0]) / np.exp([1000.0, 1001.0, 1002.0]).sum())
```

Output — the stable version gives the same answer for small and huge inputs, the naive version overflows to `inf/inf`:

```
probs: [0.090031 0.244728 0.665241] sum: 0.9999999999999999
big inputs: [0.09003057 0.24472847 0.66524096]
naive on big inputs: [nan nan nan]
```

This matches `torch.softmax` to floating-point tolerance.

**2. Cross-entropy from logits, with the causal-LM label shift.** No probabilities are ever formed.

```python
import numpy as np, torch

def cross_entropy_from_logits(logits, targets):
    # logits: (N, V) raw scores. targets: (N,) integer class ids.
    N = logits.shape[0]
    total = 0.0
    for i in range(N):
        row = logits[i]
        m = np.max(row)                                  # stability shift
        logZ = m + np.log(np.sum(np.exp(row - m)))       # log-sum-exp
        total += logZ - row[targets[i]]                  # -log p[target]
    return total / N

## causal LM shift: predict token t+1 from position t
tokens = np.array([5, 2, 9, 1, 7])
logits = np.random.default_rng(0).normal(size=(len(tokens), 12))
shift_logits = logits[:-1]     # drop the LAST position (it predicts nothing)
shift_labels = tokens[1:]      # drop the FIRST token (nothing predicts it)
loss = cross_entropy_from_logits(shift_logits, shift_labels)
ref = torch.nn.functional.cross_entropy(torch.tensor(shift_logits),
                                        torch.tensor(shift_labels)).item()
print("mine: %.6f  torch: %.6f  agree: %s" % (loss, ref, abs(loss - ref) < 1e-9))
```

Output — agreement with `torch.nn.functional.cross_entropy` to within 1e-9:

```
mine: 2.133003  torch: 2.133003  agree: True
```

**3. Scaled dot-product attention with a causal mask.** Explicit loops over query and key positions.

```python
import numpy as np, torch

def causal_attention(Q, K, V):
    T, d = Q.shape
    scores = np.zeros((T, T))
    # score every query against every key, scaled by sqrt(d)
    for i in range(T):
        for j in range(T):
            scores[i, j] = np.dot(Q[i], K[j]) / np.sqrt(d)
    # causal mask: a query at i may not see a key at j > i
    for i in range(T):
        for j in range(i + 1, T):
            scores[i, j] = -np.inf
    out = np.zeros((T, V.shape[1]))
    for i in range(T):
        row = scores[i] - np.max(scores[i])
        w = np.exp(row) / np.sum(np.exp(row))
        out[i] = w @ V
    return out

rng = np.random.default_rng(1); T, d = 4, 3
Q, K, V = rng.normal(size=(T, d)), rng.normal(size=(T, d)), rng.normal(size=(T, d))
mine = causal_attention(Q, K, V)
ref = torch.nn.functional.scaled_dot_product_attention(
    torch.tensor(Q), torch.tensor(K), torch.tensor(V), is_causal=True).numpy()
print(np.round(mine, 4))
print("matches torch SDPA:", np.allclose(mine, ref))
```

Output — agrees with `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`:

```
[[-2.7112 -1.889  -0.1748]
 [-2.0257 -1.2593 -0.0574]
 [-0.253  -1.0415 -0.1508]
 [ 0.5307 -0.3312  0.1622]]
matches torch SDPA: True
```

Row 0 equals `V[0]` exactly, because the first query can attend to nothing but the first key. That is the free correctness check for a causal mask.

**4. k-means with explicit assign and update loops.** Best of several random initialisations.

```python
import numpy as np
from sklearn.cluster import KMeans

def kmeans_once(X, k, seed, iters=50):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        for i in range(len(X)):                       # assign step
            best, best_d = 0, np.inf
            for c in range(k):
                d = np.sum((X[i] - centers[c]) ** 2)
                if d < best_d:
                    best, best_d = c, d
            labels[i] = best
        for c in range(k):                            # update step
            members = X[labels == c]
            if len(members) > 0:
                centers[c] = members.mean(axis=0)
    inertia = sum(np.sum((X[i] - centers[labels[i]]) ** 2) for i in range(len(X)))
    return labels, centers, inertia

def kmeans(X, k, n_init=10):
    # random init can land two centres in one blob, so keep the best of several runs
    runs = [kmeans_once(X, k, seed) for seed in range(n_init)]
    return min(runs, key=lambda r: r[2])

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(0, .5, (40, 2)), rng.normal(4, .5, (40, 2)), rng.normal([0, 4], .5, (40, 2))])
labels, centers, inertia = kmeans(X, 3)
sk = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("mine inertia: %.6f   sklearn inertia: %.6f" % (inertia, sk.inertia_))
```

Output — my inertia is 59.521564 and `sklearn.cluster.KMeans` reports 59.521564, an exact match on this data:

```
mine inertia: 59.521564   sklearn inertia: 59.521564
```

A single random restart on this same data gave inertia 362.29, because two centroids initialised inside one blob. That is the honest reason `n_init` exists.

**5. k-nearest-neighbours with an explicit distance loop.**

```python
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for i in range(len(X_test)):
        # explicit distance loop over the training set
        dists = []
        for j in range(len(X_train)):
            d = 0.0
            for f in range(X_train.shape[1]):
                d += (X_test[i, f] - X_train[j, f]) ** 2
            dists.append((d, y_train[j]))
        dists.sort(key=lambda t: t[0])          # nearest first
        votes = Counter(lab for _, lab in dists[:k])
        preds.append(votes.most_common(1)[0][0])
    return np.array(preds)

X, y = load_iris(return_X_y=True)
Xtr, ytr, Xte, yte = X[::2], y[::2], X[1::2], y[1::2]
mine = knn_predict(Xtr, ytr, Xte, k=5)
ref = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr).predict(Xte)
print("accuracy: %.4f   agreement with sklearn: %.4f" % ((mine == yte).mean(), (mine == ref).mean()))
```

Output — 100 percent agreement with `sklearn.neighbors.KNeighborsClassifier` on all 75 test points:

```
accuracy: 0.9867   agreement with sklearn: 1.0000
```

Squared distance is used, not the square root, because the ordering is identical and the square root costs time for nothing.

**6. Minimal PyTorch training loop.** The order of `zero_grad`, `backward`, `step` is the point.

```python
import torch, torch.nn as nn

torch.manual_seed(0)
X = torch.randn(200, 4)
y = (X @ torch.tensor([1.0, -2.0, 0.5, 3.0]) > 0).long()

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()          # takes LOGITS, not probabilities

for epoch in range(200):
    model.train()
    opt.zero_grad()                      # 1. clear gradients from the last step
    logits = model(X)                    # 2. forward
    loss = loss_fn(logits, y)            # 3. loss on logits
    loss.backward()                      # 4. accumulate gradients
    opt.step()                           # 5. apply the update
    if epoch % 50 == 0:
        print("epoch %3d  loss %.4f" % (epoch, loss.item()))

model.eval()
with torch.no_grad():                    # no grad at evaluation time
    acc = (model(X).argmax(1) == y).float().mean().item()
print("final loss %.4f  train accuracy %.4f" % (loss.item(), acc))
```

Output:

```
epoch   0  loss 0.7725
epoch  50  loss 0.1043
epoch 100  loss 0.0417
epoch 150  loss 0.0256
final loss 0.0172  train accuracy 1.0000
```

Note the model has no softmax at the end. `nn.CrossEntropyLoss` applies `log_softmax` internally, so adding one would apply it twice.

## Questions

### Q1. Why does softmax subtract the maximum, and what overflows without it?

Because `exp` overflows fast in float64. The largest finite float64 is about 1.8e308, and `exp(x)` passes that at roughly `x = 709`. In float32 the limit is around `x = 88`. Logits of that size are ordinary in a large model with unnormalised outputs, so a naive `exp(x) / sum(exp(x))` gives `inf / inf`, which is `nan`, and the `nan` then propagates through the whole backward pass. Subtracting the maximum is exact, not an approximation: multiplying numerator and denominator by $e^{-m}$ leaves the ratio unchanged. After the shift the largest exponent is $e^0 = 1$ and every other is in $(0, 1]$, so nothing can overflow. Underflow can still happen for very negative shifted values, but those terms round to zero, which is the correct limit and harmless. The demonstration above shows the naive form producing `nan` on logits of 1000 while the shifted form is fine.

> **Say it.** Because exp overflows. In float64 exp blows past the largest finite value at about 709, in float32 at about 88, and unnormalised logits reach that easily. Then you get inf over inf, which is nan, and the nan spreads through the backward pass. Subtracting the max is exact, not an approximation, because the constant cancels in the ratio. After the shift the biggest exponent is e to the zero, which is one, so nothing overflows. Small terms underflow to zero, which is the right limit.

### Q2. Why is cross-entropy implemented from logits rather than from probabilities?

Two reasons, one numerical and one about gradients. Numerically, going logits to probabilities to log throws away precision twice. A true probability of 1e-40 underflows to zero in float32, and `log(0)` is `-inf`, so a confident wrong prediction produces an infinite loss instead of a large finite one. Computing the loss directly as $\log\sum_j e^{x_j} - x_y$ with the log-sum-exp shift never forms that tiny number, so the loss stays finite and correct. Second, the fused gradient is exactly $\sigma(x) - \text{onehot}(y)$, one clean subtraction, whereas composing softmax then a separate log then a negative-log-likelihood makes the framework chain three Jacobians and hit the same catastrophic cancellation in the backward pass. That is why `nn.CrossEntropyLoss` and `F.cross_entropy` take logits, and why adding a softmax before them is a bug that silently degrades training rather than raising an error.

> **Say it.** Two reasons. Numerically, logits to probabilities to log loses precision twice — a probability of 1e-40 underflows to zero in float32 and log of zero is minus infinity, so a confidently wrong prediction gives infinite loss. Computing log-sum-exp minus the true logit never forms that number. And the fused gradient is just predicted probability minus the one-hot target, one subtraction, instead of chaining three Jacobians. That is why torch's cross-entropy takes logits, and putting a softmax in front of it is a real bug.

### Q3. What is the difference between logits and log_softmax, and where does numerical stability actually come from?

Logits are the raw unnormalised network outputs. They have no constraint: any real value, any scale, and adding a constant to all of them changes nothing about the resulting distribution. `log_softmax` is the normalised log-probability, $x_i - \log\sum_j e^{x_j}$, so the values are all non-positive and `exp` of them sums to one. Stability does not come from either representation on its own. It comes from the log-sum-exp identity used inside the conversion: pull the maximum out before exponentiating. Both `softmax` and `log_softmax` do that internally. The practical consequences are: `nn.CrossEntropyLoss` expects logits and applies `log_softmax` itself; `nn.NLLLoss` expects `log_softmax` output already; and `log(softmax(x))` computed as two separate steps is the unstable version you should never write, because it forms the probabilities first and then takes a log that can be `-inf`.

> **Say it.** Logits are raw unnormalised outputs, any scale, shift-invariant. log_softmax is the normalised log-probability — non-positive values whose exp sums to one. Stability comes from neither of them by itself; it comes from the log-sum-exp trick applied inside the conversion, pulling the max out before exponentiating. Practically: CrossEntropyLoss takes logits and does log_softmax internally, NLLLoss takes log_softmax output. Writing log of softmax as two separate calls is the unstable version, because it forms the probabilities first.

### Q4. What is the correct label shift for causal language modelling, and what is the classic off-by-one?

The model at position $t$ predicts the token at position $t+1$. So you drop the last position from the logits, because it predicts a token beyond the end of the sequence that you do not have, and you drop the first token from the labels, because nothing precedes it to predict it. That is `logits[:, :-1, :]` against `labels[:, 1:]`, giving $T-1$ prediction targets from a sequence of $T$ tokens. The classic off-by-one is aligning `logits[t]` with `labels[t]`, which asks the model to predict the token it was just given. That task is trivial with a causal mask, so the loss drops to near zero almost immediately and looks like spectacular training. The signal that this happened is a loss far below $\log V$ within the first few steps, and generation that produces nothing coherent. Some libraries do the shift for you when you pass `labels`, so check rather than assume, and never shift twice.

> **Say it.** Position t predicts token t plus one. So logits lose the last position, labels lose the first — logits[:, :-1] against labels[:, 1:], giving T minus one targets. The classic mistake is aligning logits[t] with labels[t], which asks the model to predict the token it was just handed. Under a causal mask that is trivial, so the loss collapses to near zero and looks like great training while generation is garbage. The tell is loss far below log of vocabulary size in the first few steps. And some libraries shift for you, so check.

### Q5. Why do you call zero_grad, and what happens if you forget?

Because PyTorch accumulates into `.grad` rather than overwriting it. Every `backward()` adds the new gradient to whatever is already stored. That behaviour is deliberate: it is how you implement gradient accumulation across micro-batches to simulate a larger batch, and how you sum gradients from several loss terms. If you forget `zero_grad`, step $n$ uses the sum of the gradients from all steps 1 through $n$. The effective gradient magnitude grows without bound, so the updates get larger and larger, and training diverges to `nan` — usually within tens of steps, sometimes after a slow drift that looks like a learning-rate problem. It goes in the loop before `backward()`. Placing it after `backward()` and before `step()` erases the gradient you just computed, and the model then never learns at all. That failure is easy to spot: the loss is completely flat.

> **Say it.** Because PyTorch accumulates into dot-grad instead of overwriting. That is deliberate — it is how gradient accumulation across micro-batches works. If you forget it, step n uses the sum of every gradient so far, the effective magnitude grows without bound, and training diverges to nan, often within tens of steps. It goes before backward. If you put it between backward and step you wipe out the gradient you just computed and the loss stays perfectly flat, which is the easier bug to spot.

### Q6. How would you debug a training loop where the loss does not move?

I work from cheapest to most expensive. First check the loss is even connected: does it change at all across steps, and is the initial value what theory predicts — for $V$ balanced classes it should start near $\log V$, so about 2.30 for ten classes. A flat loss at exactly the initial value usually means `zero_grad` after `backward`, a detached tensor, `requires_grad` false, or `torch.no_grad` around the forward pass. Second, print gradient norms per layer; all-zero says the graph is broken, all-`nan` says a numerical problem earlier. Third, overfit a single batch of 8 examples. A correct model must drive that to near-zero loss in a few hundred steps. If it cannot, the bug is in the model or the loss, not in the data or the schedule. Only if it can do I look at learning rate, data pipeline, shuffling, and label alignment. That ordering isolates the cause instead of changing several things at once.

> **Say it.** Cheapest checks first. Is the initial loss what theory says — around log of the number of classes? Does it change at all? A perfectly flat loss usually means zero_grad in the wrong place, a detached tensor, or no_grad around the forward. Then I print per-layer gradient norms: all zeros means a broken graph, nan means a numerical problem upstream. Then I overfit one batch of eight examples; a correct model must reach near-zero. If it cannot, the bug is the model or the loss. Only then do I touch learning rate or data.

### Q7. How do you check a gradient implementation numerically?

Central finite differences. For each parameter $\theta_i$, perturb it by a small $\epsilon$ in both directions and compute $(f(\theta + \epsilon e_i) - f(\theta - \epsilon e_i)) / 2\epsilon$, then compare against the analytic gradient. Central differences have error $O(\epsilon^2)$ where forward differences have $O(\epsilon)$, so the central form is worth the extra function evaluation. Use float64, because in float32 the rounding noise swamps the signal. Pick $\epsilon$ around 1e-5: too large and the truncation error dominates, too small and subtractive cancellation does. Compare with a relative error, $|a - n| / \max(|a|, |n|, 1)$, and expect below roughly 1e-7 for a correct implementation in float64. Two practical warnings. Check only a random sample of coordinates, because the cost is two forward passes per parameter. And turn off dropout and any other stochastic layer first, or the two evaluations are of different functions and the check is meaningless. `torch.autograd.gradcheck` does exactly this.

> **Say it.** Central finite differences: perturb one parameter by plus and minus epsilon, take the difference over two epsilon, compare to the analytic gradient. Central is order epsilon squared, forward is order epsilon, so it is worth the extra evaluation. Float64, epsilon around 1e-5, relative error below about 1e-7. I sample a few coordinates rather than all of them, because it costs two forward passes each. And I turn off dropout first, otherwise the two evaluations are different functions. torch.autograd.gradcheck automates it.

### Q8. Loops or vectorisation — which does the interviewer want?

I start with the explicit loop, say out loud that it is $O(n k d)$ and that I will vectorise if there is time, and get it running and tested first. Then I offer the vectorised version. The reasons are practical. A loop is far more likely to be correct on the first attempt under time pressure, because broadcasting bugs produce silently wrong shapes rather than errors — a `(n, 1)` against `(1, n)` gives you an `(n, n)` matrix and no complaint from NumPy. A loop is also readable to the interviewer, so they can follow the algorithm rather than decode index gymnastics. And it demonstrates that I know the algorithm rather than an API. The exception is when the question is explicitly about performance, or when the interviewer says "now make it fast", or when the data size in the prompt makes an $O(n^2)$ loop obviously infeasible. Then vectorising is the task, and I do it having already established a correct reference to check against.

> **Say it.** I write the loop first, say the complexity out loud, and get it correct and tested. Then I offer to vectorise. A loop is much more likely to be right first time under pressure, because broadcasting bugs are silent — the shapes just come out wrong and NumPy does not complain. It is also readable, so the interviewer can follow the algorithm. I vectorise when they ask, or when the stated data size makes the loop infeasible, and then I have a correct reference to check the fast version against.

### Q9. How would you test an ML function?

With four layers. First, known-value tests: cases where I can compute the answer by hand. Softmax of `[0, 0]` is `[0.5, 0.5]`; cross-entropy of a uniform distribution over $V$ classes is $\log V$; attention over one position returns `V[0]` exactly. Second, invariants and properties: softmax output sums to one and is invariant to adding a constant to the input; k-means inertia decreases monotonically across iterations; a causal attention output at position $t$ must not change when I edit inputs after $t$. Third, reference comparison against a library — `torch.nn.functional.cross_entropy`, `sklearn.cluster.KMeans`, `KNeighborsClassifier` — using `np.allclose` with a stated tolerance. Fourth, edge cases: empty input, a single element, all-equal values, extreme magnitudes like 1000 and -1000, and an empty cluster in k-means. The causal invariance test in the third layer is the one that catches real masking bugs, and almost nobody writes it.

> **Say it.** Four layers. Known values I can compute by hand — softmax of two zeros is a half each, uniform cross-entropy over V classes is log V. Then invariants: probabilities sum to one, softmax is shift-invariant, k-means inertia decreases every iteration, causal attention at position t is unchanged when I edit later inputs. Then comparison against a library reference with allclose and a stated tolerance. Then edge cases — empty, single element, all-equal, magnitudes of a thousand, empty clusters. The causal invariance test is the one that catches real mask bugs.

### Q10. What are the common time sinks in this round, and how do you avoid them?

Four, in order of how much time they cost. Broadcasting and shape debugging, which is why I write loops and print shapes after each step rather than reasoning about them. Rewriting from scratch because a clever approach failed, which is why the first version I write is the boring one that will definitely run. Silence, because if I think for two minutes without speaking, the interviewer cannot help me and cannot score my reasoning. And blank-page paralysis at the start. My defence for the last one is a fixed opening: restate the input and output shapes, write the function signature with a docstring naming those shapes, write a test case with the expected answer, then fill in the body. That takes about ninety seconds and it converts an open question into a small concrete one. I also keep the numerical stability step in muscle memory, so subtracting the max is automatic rather than something I remember when the interviewer asks.

> **Say it.** Four things. Shape and broadcasting debugging — I avoid it by writing loops and printing shapes at each step. Rewriting after a clever approach fails — so my first version is deliberately boring. Silence, because if I think for two minutes without talking the interviewer cannot help me or score me. And blank-page paralysis. My fix for that is a fixed opening: restate the input and output shapes, write the signature and docstring, write one test case with the expected answer, then fill in the body. Ninety seconds, and the problem becomes concrete.

### Q11. Why does k-means need multiple initialisations, and does it converge?

It converges, but only to a local minimum. Each iteration is coordinate descent on $J = \sum_i \|x_i - \mu_{c(i)}\|^2$: the assign step minimises $J$ over assignments with centroids fixed, and the update step minimises $J$ over centroids with assignments fixed. Both steps are exact minimisations, so $J$ never increases, and since the number of possible assignments is finite the algorithm terminates. It does not find the global optimum. Which local minimum you land in depends entirely on the initial centroids. In the block above, one random restart on three well-separated blobs gave inertia 362.29 because two centroids initialised inside the same blob, while the best of ten restarts gave 59.52, matching sklearn exactly. That is a factor of six from initialisation alone on easy data. The fixes are running `n_init` restarts and keeping the lowest inertia, and using k-means++ initialisation, which picks each new centroid with probability proportional to squared distance from the nearest existing one.

> **Say it.** It converges, but to a local minimum. Each iteration is exact coordinate descent on the inertia — assign minimises over labels, update minimises over centroids — so the objective never increases and the assignments are finite, therefore it terminates. Where it lands depends entirely on the initial centroids. On my three-blob test one restart gave inertia 362 because two centroids started in the same blob, and best-of-ten gave 59.5, matching sklearn. Factor of six from initialisation alone. So: multiple restarts, or k-means++.

### Q12. What choices matter in k-NN, and what breaks it?

Three choices. The value of $k$ trades variance for bias: $k = 1$ has zero training error and high variance, and large $k$ over-smooths toward the majority class, so I pick it by cross-validation and use an odd number for binary problems to avoid ties. The distance metric must match the data: Euclidean for continuous features on a comparable scale, cosine when direction matters more than magnitude, as with embeddings. Feature scaling is the thing that actually breaks it — an unscaled feature measured in thousands dominates every distance and the other features stop mattering, so standardise first. Beyond that, k-NN has no training phase but $O(nd)$ prediction cost per query, which is the reason approximate nearest-neighbour indexes exist. And it degrades in high dimensions, because distances between all pairs of points concentrate, so "nearest" stops carrying information. Ties in the vote need a stated rule: nearest neighbour wins, or distance-weighted voting.

> **Say it.** Three choices. k trades variance against bias — one has high variance, large k over-smooths, so I cross-validate and use an odd k for binary. The metric has to match the data: Euclidean for scaled continuous features, cosine for embeddings. And feature scaling is what actually breaks it, because an unscaled large-magnitude feature dominates every distance. There is no training cost but prediction is O of n times d per query, which is why approximate indexes exist, and it degrades in high dimensions because distances concentrate.

### Q13. In attention, why divide by the square root of the key dimension?

Because the dot product's variance grows with dimension. If the components of $q$ and $k$ are independent with mean zero and unit variance, then $q \cdot k$ is a sum of $d_k$ independent products, so it has mean zero and variance $d_k$, giving a standard deviation of $\sqrt{d_k}$. With $d_k = 64$ the raw scores have a spread of about 8, and with $d_k = 128$ about 11. Feeding scores with that spread into softmax pushes it toward a one-hot distribution: one weight near 1, the rest near 0. That is bad early in training, because the gradient of softmax is proportional to $p(1-p)$ and therefore vanishes when $p$ saturates at 0 or 1. Dividing by $\sqrt{d_k}$ restores unit variance regardless of head dimension, so the softmax stays in its responsive range and gradients flow. The mask is applied as $-\infty$ before the softmax, so those positions get exactly zero weight rather than a small one.

> **Say it.** Because the dot product variance scales with the key dimension. With zero-mean unit-variance components the dot product has variance d_k, so standard deviation root d_k — about 8 for a head dimension of 64. Softmax over scores that spread saturates toward one-hot, and softmax gradients go like p times one minus p, so they vanish when it saturates. Dividing by root d_k restores unit variance whatever the head size, keeping softmax responsive. And the mask is minus infinity applied before the softmax, so masked positions get exactly zero weight.

## Done when

- You can write stable softmax, cross-entropy from logits, and causal attention from a blank file in under 15 minutes total, and all three match the torch reference on the first run.
- You can state, without looking, that float32 `exp` overflows near 88 and float64 near 709, and explain why subtracting the max is exact rather than approximate.
- You can write the k-means assign and update loops and reproduce sklearn's `inertia_` to at least six significant figures.
- You can type the five lines of a PyTorch step in the correct order and say what breaks for each of the two ways of misplacing `zero_grad`.
