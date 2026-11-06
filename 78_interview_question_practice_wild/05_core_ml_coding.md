# Core ML coding

<span class="practice-page"></span>

**Foundation explainer:** Core material · **Priority:** Very important

Very important: these reusable numerical patterns support many interview questions.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f041"></a>

## F041. Implement stable softmax and log-softmax.

**Start here.** A common logit shift leaves the probability distribution unchanged, so subtract a maximum before exponentiation.

**Work it through.**

1. Convert finite logits to a floating array and choose the class axis explicitly.
2. Set z=x-max(x), then softmax=exp(z)/sum(exp(z)). Compute log-softmax directly as z-log(sum(exp(z))).
3. Test shift invariance, normalization and very large logits. Define a separate policy for negative-infinity masks rather than silently accepting an all-masked row.

**Small example.** Logits [1000,1001] give the same probabilities as [0,1], approximately [0.269,0.731].

**Common trap.** Computing log(softmax) when tiny probabilities may underflow to zero.

**Follow-up.** What is the softmax Jacobian? diag(p)-p*pᵀ; with cross-entropy it simplifies to predicted minus target probabilities.

---

<a id="f042"></a>

## F042. Implement LayerNorm forward and backward.

**Start here.** Each row is centered and rescaled independently across its features; the backward pass must include the effects of both statistics.

**Work it through.**

1. Compute mean, centered values, population variance and inv_std=1/sqrt(var+eps). Set xhat=centered*inv_std and out=gamma*xhat+beta.
2. For g=dout*gamma, use dx=inv_std*(g-mean(g)-xhat*mean(g*xhat)), with means over features.
3. Sum dout*xhat and dout over the batch for dgamma and dbeta. Verify with central finite differences in float64.

**Small example.** A constant row has xhat=0, so its forward output equals beta; epsilon keeps the inverse standard deviation finite.

**Common trap.** Using mean(x²)-mean(x)² for large-offset data, or forgetting that epsilon is inside the square root.

**Follow-up.** Why can constant upstream gradients give zero dx for gamma=1? A normalized row is invariant to a common input shift.

---

### Detailed derivation and worked reasoning

### Derive the normalization and its gradient

For row $x_i\in\mathbb R^D$, compute population statistics across **features**, independently of the other rows:

$$
\mu_i=\frac1D\sum_jx_{ij},\quad
v_i=\frac1D\sum_j(x_{ij}-\mu_i)^2,\quad
s_i=(v_i+\epsilon)^{-1/2},\quad
\hat x_{ij}=(x_{ij}-\mu_i)s_i.
$$

Then $y_{ij}=\gamma_j\hat x_{ij}+\beta_j$. The epsilon goes inside the square root. Compute variance from centered deviations, not $E[x^2]-E[x]^2$: the latter subtracts two huge nearly equal numbers when inputs share a large offset.

Let $G_{ij}=\partial L/\partial y_{ij}$ and $U_{ij}=G_{ij}\gamma_j$. The chain rule has three paths: a direct input path, the row mean path and the row variance path. Combining them gives

$$
\frac{\partial L}{\partial x_{ij}}=
 s_i\left(U_{ij}-\overline U_i-
 \hat x_{ij}\,\overline{U_i\hat x_i}\right),
$$

where each overline averages across the $D$ features of that row. The parameter gradients sum over rows:

$$
\frac{\partial L}{\partial\gamma_j}=\sum_iG_{ij}\hat x_{ij},\qquad
\frac{\partial L}{\partial\beta_j}=\sum_iG_{ij}.
$$

For $[1,2,3,4]$, the mean is $2.5$ and variance $1.25$. With unit scale and zero bias, the result is approximately $[-1.3416,-0.4472,0.4472,1.3416]$. A constant row normalizes to zero, so its output equals the bias. Its input derivative is not universally zero: it depends on the upstream pattern and epsilon.

The forward cache stores normalized inputs, inverse standard deviations and scale. Do not mutate these before backward. Validate $x$, $\gamma$ and $\beta$ derivatives independently with float64 central differences; checking only output shape cannot catch omitted statistic paths. Both passes cost $O(ND)$.


<a id="f043"></a>

## F043. Write a two-hidden-layer MLP and derive backpropagation.

**Start here.** Cache local intermediate values, then apply the chain rule in the reverse order of the forward computation.

**Work it through.**

1. Use Z1=XW1+b1, A1=ReLU(Z1), Z2=A1W2+b2, A2=ReLU(Z2), Yhat=A2W3+b3.
2. For mean squared error over all N*C outputs, dY=2*(Yhat-Y)/(N*C). Compute dW3=A2ᵀ*dY and db3=sum(dY), then propagate through W3 and the ReLU masks.
3. Repeat for the earlier layers. Keep parameter updates separate from gradient calculation and check a small network numerically.

**Small example.** For a dense layer X:(4,3), W:(3,2), dY:(4,2), dW has shape (3,2) and dX has shape (4,3).

**Common trap.** Dividing by batch size twice, or updating a weight before using its original value in an earlier gradient.

**Follow-up.** Why avoid finite-difference points at ReLU zero? The derivative is not uniquely defined there.

---

### Detailed derivation and worked reasoning

### Walk backward through the exact scalar objective

Use row-major examples and explicitly name every activation:

$$
Z_1=XW_1+b_1,\ A_1=\operatorname{ReLU}(Z_1),\quad
Z_2=A_1W_2+b_2,\ A_2=\operatorname{ReLU}(Z_2),\quad
\hat Y=A_2W_3+b_3.
$$

For $N$ examples and $C$ target coordinates, define $L=\|\hat Y-Y\|_F^2/(NC)$. The output derivative is therefore

$$
G_3=\frac{2}{NC}(\hat Y-Y),\qquad
\nabla_{W_3}L=A_2^\top G_3,\quad
\nabla_{b_3}L=\sum_iG_{3,i,:}.
$$

Propagate through the matrix and activation in reverse order:

$$
G_2=(G_3W_3^\top)\odot\mathbf1[Z_2>0],\qquad
\nabla_{W_2}L=A_1^\top G_2,
$$

then repeat for layer one. The ReLU mask belongs to the **preactivation** of the corresponding layer. Bias derivatives sum across reused batch positions, and no additional mean should be introduced after the $1/(NC)$ already in $G_3$.

For $X$ shaped $(4,3)$ and $W_1$ shaped $(3,5)$, $Z_1$ is $(4,5)$, $\nabla W_1$ is $(3,5)$ and $\nabla X$ is $(4,3)$. Check these shapes before compacting expressions.

Compute all gradients before updating weights, and compare each parameter tensor with a central-difference reference on a tiny non-boundary example. Shape agreement alone misses wrong transpose/reduction choices. A constant or symmetric initialization can also prevent useful hidden-unit differentiation; a correct backward pass does not guarantee an effective training configuration.


<a id="f044"></a>

## F044. Implement k-means with convergence and empty clusters.

**Start here.** Alternate assigning points to their closest centers and moving each center to the mean of its assigned points.

**Work it through.**

1. Initialize k distinct data indices under an explicit random seed, requiring 1<=k<=N.
2. Compute squared distances, assign closest centers and recompute non-empty means. For an empty cluster, retain its center or explicitly reseed it.
3. Stop on a center-shift tolerance or iteration cap, then recompute labels for the returned centers. Compare objective values, not center ordering, across implementations.

**Small example.** For [0,1,9,10] with k=2 and sensible initialization, centers converge to 0.5 and 9.5.

**Common trap.** Taking the mean of an empty slice, stopping with labels from old centers, or assuming a local optimum is globally best.

**Follow-up.** How does k-means++ help? It spreads initial centers probabilistically, often reducing poor local solutions.

---

<a id="f045"></a>

## F045. Implement k-nearest-neighbor classification.

**Start here.** Store the training examples; for a query, predict from the labels of the nearest k examples under a chosen distance.

**Work it through.**

1. Fit feature transformations on training data only and apply the same transformation to queries.
2. Compute query-to-training squared distances, choose k neighbors, and count their labels or use defined distance weights.
3. Specify tie-breaking and k bounds. For many queries use batching or a search index; the brute-force baseline costs O(ND) per query before selection overhead.

**Small example.** Neighbors with labels [A,A,B] predict A for k=3. With k=2 and labels [A,B], a deterministic tie rule is required.

**Common trap.** Comparing unscaled features with incompatible units or including the test point as its own training neighbor.

**Follow-up.** Why can high-dimensional KNN weaken? Distances become less discriminative and search structures can lose their advantage.

---

<a id="f046"></a>

## F046. Compute pairwise Euclidean distances between two tensor batches.

**Start here.** Make the desired pairwise shape explicit before using broadcasting.

**Work it through.**

1. For X:(N,D), Y:(M,D), the result is (N,M). A direct reference uses X[:,None,:]-Y[None,:,:].
2. A lower-memory algebraic version computes squared distances as norm(X)²+norm(Y)²-2X@Yᵀ, clamping tiny negative roundoff to zero.
3. Take square roots only if actual distances are needed. Block the computation when N*M is too large; no formula removes the output-size cost.

**Small example.** The distance between [0,0] and [3,4] is 5; the squared distance is 25.

**Common trap.** Returning elementwise distances between matching rows when the question asks for every pair.

**Follow-up.** When is the direct difference more accurate? For nearly equal large vectors, the norm identity can suffer cancellation.

---

<a id="f047"></a>

## F047. Implement symmetric InfoNCE loss.

**Start here.** A matched query/key pair should score above other examples in the batch, in both retrieval directions.

**Work it through.**

1. Normalize q and k along the embedding dimension and require temperature>0.
2. Compute logits=q@kᵀ/temperature; the correct key for query i is index i. Use stable cross-entropy across each row.
3. Average query-to-key and key-to-query cross-entropies. Keep the computation differentiable in a training implementation and define how duplicate positives are treated.

**Small example.** For B=1 there are no in-batch negatives and the loss is zero, which does not mean the embedding model is useful.

**Common trap.** Treating all same-batch non-diagonal examples as valid negatives when some are actually semantically positive.

**Follow-up.** What changes with distributed negatives? Gather representations with the intended gradient semantics and adjust positive indices by worker offset.

---

### Detailed derivation and worked reasoning

### Derive the diagonal contrastive targets

For matched query/key rows $(q_i,k_i)$, normalize each nonzero vector and form a batch similarity matrix

$$
S_{ij}=\frac{\tilde q_i^\top\tilde k_j}{\tau},\qquad
L_{q\to k}=-\frac1B\sum_i\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}.
$$

The key-to-query loss uses $S^\top$; the symmetric objective is $(L_{q\to k}+L_{k\to q})/2$. Temperature controls logit sharpness and gradient scale. Normalizing first makes this interpretation independent of arbitrary positive embedding scale.

A batch of one has zero loss because the only key is the positive. That is a warning about absent negatives, not evidence of perfect semantic retrieval. Likewise, repeated or alternative true matches can be wrongly penalized as negatives by a diagonal-only target.

Use stable log-softmax/cross-entropy rather than manually exponentiating large scores. A NumPy implementation is a forward numerical reference; training needs the equivalent differentiable operations so gradients flow into both intended branches. A deliberately detached momentum branch is a different architecture decision, not an accidental conversion to NumPy.

Test swapping $q$ and $k$, positive row scaling, small temperature and zero-norm input policy. If negatives are gathered across workers, adjust each positive index and specify whether gradients cross that collective. More negatives can help discrimination, but label correctness and negative distribution are at least as important as the raw batch count.


<a id="f048"></a>

## F048. Implement numerically stable binary focal loss.

**Start here.** Focal loss reduces the contribution of already-easy examples by multiplying binary cross-entropy by (1-p_true)^gamma.

**Work it through.**

1. Given a binary label y and logit z, compute stable BCE as logaddexp(0,z)-y*z.
2. Let log_p_true=-BCE and compute 1-p_true stably as -expm1(log_p_true).
3. Multiply BCE by that factor raised to gamma and by alpha for positives or 1-alpha for negatives, if alpha weighting is specified. State the reduction.

**Small example.** At gamma=0, focal modulation is one, so the result reduces to ordinary BCE with the chosen alpha weighting.

**Common trap.** Applying focal loss twice to already-probability-transformed scores or confusing it with calibration improvement.

**Follow-up.** Why can it hurt? It can overemphasize mislabeled hard examples; inspect data quality and held-out metrics.

---

<a id="f049"></a>

## F049. Calculate average calibration error from confidence bins.

**Start here.** Calibration compares predicted confidence with observed correctness; it is different from classification accuracy.

**Work it through.**

1. Convert predicted-class log probabilities to confidence with exp(logprob), then assign equal-width bins in [0,1].
2. Put confidence exactly 1 in the last bin. For each non-empty bin compute abs(mean_accuracy-mean_confidence).
3. For this ACE definition, average gaps equally across non-empty bins and return zero for no examples. A sample-weighted version is a different metric.

**Small example.** Bins with gaps 0.05 and 0.025 give ACE=0.0375 regardless of their different populations under the equal-bin convention.

**Common trap.** Calling a population-weighted metric and an equal-bin metric interchangeable, or counting empty bins as perfect calibration.

**Follow-up.** Why can ACE be noisy? Small changes can move examples across bin boundaries; inspect bin counts and reliability plots.

---

### Detailed derivation and worked reasoning

### Specify which calibration error you mean

Confidence here is the probability of the **predicted** class, not necessarily the true class. If its stored value is log confidence $\ell_i$, recover $c_i=e^{\ell_i}$ and let $a_i\in\{0,1\}$ indicate correctness.

Split $[0,1]$ into $M$ equal-width bins, placing exactly $1$ in the final bin. For nonempty bins $\mathcal B_+$, this exercise defines

$$
\operatorname{ACE}=\frac1{|\mathcal B_+|}\sum_{b\in\mathcal B_+}
\left|\frac{\sum_{i\in b}a_i}{n_b}-\frac{\sum_{i\in b}c_i}{n_b}\right|.
$$

Every nonempty bin receives equal weight. A population-weighted ECE instead weights each gap by $n_b/N$ and is a different statistic. Empty bins are omitted rather than counted as zero-error bins.

For confidence values $[0.9,0.95,1,0.5,0.55]$ and correctness $[1,1,1,0,1]$ at five bins, the first group has accuracy $1$ and mean confidence $0.95$, gap $0.05$; the second has accuracy $0.5$ and confidence $0.525$, gap $0.025$. ACE is $(0.05+0.025)/2=0.0375$.

Show bin counts alongside the number. Small bins and examples crossing a boundary can make the curve spiky even without a large change in the model. Use independent calibration/evaluation data and inspect proper scores such as log loss too; one binned metric is not a complete account of probability quality.


<a id="f050"></a>

## F050. Clean noisy annotations using rater agreement.

**Start here.** Estimate which annotators agree with a provisional consensus, but recognize that the consensus itself can be biased.

**Work it through.**

1. Group annotations by prompt; form a majority label, marking ties as unresolved.
2. For each rater, measure agreement only on resolved prompts. Assign reliability tiers or continuous weights, then filter or aggregate annotations according to the specified thresholds.
3. Train identical baseline and cleaned-data models and compare on independent gold labels. Include a leave-one-rater-out consensus variant to reduce self-inclusion bias.

**Small example.** If A and B vote 1 and C votes 0 on two prompts, C has agreement zero with their majority; that does not prove A and B are correct.

**Common trap.** Evaluating on labels generated by the same consensus used for cleaning, or assuming minority annotators are necessarily careless.

**Follow-up.** How do you handle systematic group bias? Use gold audits, subgroup checks, guideline review and richer annotator models.

---
