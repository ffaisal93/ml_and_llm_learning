# Depth and breadth questions

**Breadth** questions sample widely — twenty topics, thirty seconds each. The interviewer is checking coverage, so the right answer is crisp and stops.
**Depth** questions pick one thing you claimed to know and ask "why" until you hit bottom. The interviewer is checking whether your knowledge is a memorized surface or an actual model.
Most people prepare only for breadth, and get eliminated on the third rung of a depth ladder. Both sections below exist for that reason.

---

## Depth ladders

Each ladder is one opening question and the follow-ups an interviewer actually asks. Read the rungs downward. If you can't answer rung 4, that's the topic to review tonight.

**How to survive a ladder.** Answer the rung you were asked, not the one after it — volunteering rung 4 early reads as rehearsed and invites them to go to rung 6. When you reach your limit, name the boundary out loud ("I know the mechanism; I haven't derived the bound") and then reason forward from what you do know. Interviewers score the honest boundary far higher than a confident wrong answer, and every ladder is designed to end in one — hitting bottom is the expected outcome, not the failure.

---

### Ladder 1 — Logistic regression

**Q1. Why do we use log loss instead of MSE for logistic regression?**
Two reasons. MSE on a sigmoid output is non-convex in the weights, so gradient descent can stall in local minima. And its gradient carries a $\sigma'(z)$ factor that vanishes when the model is confidently wrong — exactly when you most need a large update. Log loss is the negative log-likelihood of the Bernoulli model, which is convex and whose gradient is the plain residual.

**Q2. Derive the gradient.**
$z = Xw$, $p = \sigma(z)$, $\mathcal{L} = -\sum[y\log p + (1-y)\log(1-p)]$.
$\partial\mathcal{L}/\partial p = \frac{p-y}{p(1-p)}$, and $\partial p/\partial z = p(1-p)$. They cancel: $\partial\mathcal{L}/\partial z = p - y$. Then $\nabla_w = X^\top(p-y)$ — the same form as linear regression with MSE. That's not a coincidence: both are GLMs with their canonical link, and the canonical link is defined so the link derivative cancels the loss curvature.

**Q3. Why is it convex?**
The Hessian is $X^\top S X$ where $S = \text{diag}(p_i(1-p_i))$. Every $p_i(1-p_i) > 0$, so $S \succ 0$, so $X^\top S X \succeq 0$ — positive semidefinite everywhere, which is the definition of convex. It's strictly convex iff $X$ has full column rank.

**Q4. What happens on linearly separable data?**
The MLE doesn't exist. For any separating $w$, scaling to $2w$ makes every prediction more confident and strictly lowers the loss, so the optimizer pushes $\|w\| \to \infty$ chasing an infimum it never reaches. Practically: weights blow up, probabilities saturate at 0 and 1, and the model is wildly overconfident on anything near the boundary. sklearn hides this because `LogisticRegression` regularizes by default.

**Q5. How does regularization fix it, and what does it converge to?**
Adding $\lambda\|w\|_2^2$ makes the objective strictly convex and coercive, so a finite unique minimum exists. Geometrically, the penalty stops $\|w\|$ from growing and picks the finite-margin solution. Worth adding: unregularized gradient descent on separable data has an implicit bias — the *direction* of $w$ converges to the max-margin (hard SVM) solution, just very slowly, at $O(1/\log t)$. So explicit regularization mostly buys you speed and a well-conditioned problem.

**Q6. L1 or L2 here — does it matter?**
L2 keeps all features with shrunken coefficients and has a unique solution even under collinearity. L1 drives coefficients exactly to zero because the $\ell_1$ ball has corners on the axes, so the constrained optimum lands on one. Choose L1 when you want selection or a sparse deployable model, L2 when features are correlated and you want stability — L1 arbitrarily picks one of a correlated pair and zeros the rest, which makes the selected set unstable across resamples. Elastic net when you want both.

---

### Ladder 2 — Attention

**Q1. What does attention compute?**
A weighted average of value vectors, where the weights come from a softmax over query-key similarities. Each token emits a query, every token offers a key and a value, and the output for a token is the mixture of values whose keys its query matched. It's content-based routing, and it's permutation-equivariant, which is why you need positional information.

**Q2. Why divide by $\sqrt{d_k}$?**
If $q$ and $k$ have i.i.d. entries with unit variance, $q \cdot k$ has variance $d_k$, so logits grow like $\sqrt{d_k}$. Large logits saturate the softmax; its Jacobian $p_i(\delta_{ij}-p_j)$ goes to zero and gradients die. Dividing by $\sqrt{d_k}$ normalizes the logit variance back to 1 at initialization.

**Q3. What's the complexity, and when does it actually matter?**
$O(n^2 d)$ time and $O(n^2)$ memory for the score matrix, versus $O(nd^2)$ for the projections. So attention only dominates once $n \gtrsim d$ — for a 4096-wide model at 512 tokens, the MLPs are the cost. The memory term is what FlashAttention fixes: it tiles the computation and never materializes the $n \times n$ matrix, giving $O(n)$ memory. The $O(n^2)$ FLOPs remain; FlashAttention is an IO-complexity win, not an asymptotic FLOP win.

**Q4. At inference time, what dominates memory?**
The KV cache. You cache each layer's keys and values so decoding step $t$ doesn't recompute steps $1..t-1$. Size is $2 \times L \times n_{kv} \times d_{head} \times \text{tokens} \times \text{bytes}$, times batch size. It grows linearly with context and batch, and unlike weights it's per-user — at long context and high batch it exceeds the weights. It also makes decoding memory-bandwidth-bound rather than compute-bound.

**Q5. So why MQA and GQA?**
Because $n_{kv}$ is the only free term in that formula. MQA uses one KV head shared across all query heads — up to a 64× cache cut, but it degrades quality and is unstable to train. GQA is the interpolation: $g$ groups of query heads share a KV head. Llama-2 70B uses 64 query heads and 8 KV heads, an 8× reduction with essentially no quality loss, and GQA is the default in current open-weight models. DeepSeek takes a different route with MLA, compressing KV into a low-rank latent that's re-expanded per head.

**Q6 (if they keep going). Why does the cache shrink but the compute barely change?**
You still run all $h$ query heads; you just broadcast the shared K/V. FLOPs are nearly identical. The win is memory and, more importantly, memory bandwidth — which is the actual bottleneck during decode.

---

### Ladder 3 — Overfitting

**Q1. How do you detect overfitting?**
Training loss keeps falling while validation loss flattens then rises. The gap is the signal, not the absolute level. Learning curves as a function of training-set size are the cleaner diagnostic: overfitting shows as a persistent train/val gap that narrows as you add data.

**Q2. What causes it?**
Model capacity large relative to the information in the data, so the model fits sampling noise. Concretely: too many parameters for too few examples, too many training steps, leakage-adjacent features that only correlate in this sample, or a validation set that's been tuned against so many times it's effectively training data.

**Q3. How do you fix it?**
In rough order of leverage: more or better data; augmentation; reduce capacity; regularize (L2, dropout, early stopping); ensemble; transfer learning from a pretrained model. For a specific model there's a specific answer — max depth and min samples per leaf for trees, $\lambda$ for linear models, dropout and weight decay for nets.

**Q4. Why does more data help — mechanically?**
Variance in the bias-variance decomposition falls roughly as $1/n$ while bias is unchanged. Equivalently: the empirical risk converges to the true risk at $O(1/\sqrt{n})$, so the gap between what you optimize and what you care about shrinks. More data doesn't make the model smarter; it makes the training objective a better estimate of the real one.

**Q5. When does more data not help?**
Four cases. (1) You're underfitting — bias-dominated, so adding data moves nothing and you need capacity instead. (2) The new data is from a different distribution than deployment. (3) You're at the irreducible-noise floor: label noise $\sigma^2$ doesn't shrink with $n$. (4) The new data is redundant — near-duplicates add no information, which is why dedup matters more than raw token count in LLM pretraining. The learning-curve test settles it: if train and val loss have already converged to each other and both are high, it's bias, and data won't help.

**Q6. Your val loss is lower than your train loss. What's going on?**
Usually not a miracle. Common causes: dropout and other regularization are active during training but off at eval, so train loss is measured on a handicapped model; train loss is averaged over the epoch while val is measured at the end, after the model has improved; or the validation split is simply easier or smaller and noisier. Genuine leakage from val into train produces the opposite signature — implausibly good val performance that collapses on a fresh test set.

---

### Ladder 4 — Backpropagation

**Q1. What is backprop?**
Reverse-mode automatic differentiation on the computation graph. Forward pass caches activations; backward pass applies the chain rule from the loss backward, reusing each layer's upstream gradient. It costs about the same as the forward pass because it computes all parameter gradients in one sweep — which is why reverse mode and not forward mode, given one scalar output and millions of inputs.

**Q2. Why do gradients vanish or explode in deep nets?**
The gradient through $L$ layers is a product of $L$ Jacobians. Products of matrices with spectral norm consistently below 1 decay exponentially; above 1, they blow up. With sigmoid activations, $\sigma' \le 0.25$, so even in the best case you lose a factor of 4 per layer. Explosion is the same mechanism with the inequality flipped, and it's easier to fix — just clip.

**Q3. How do residual connections help?**
$y = x + F(x)$ makes the Jacobian $I + \partial F/\partial x$. The identity term guarantees a path of gradient magnitude 1 straight back to any earlier layer, so the product can't decay to zero through that path. Framing it as "learning the residual rather than the full map" is the same statement — the identity is now the default, and the block only has to learn the deviation.

**Q4. Where does normalization fit in?**
It controls the scale of the Jacobians so their product stays near 1. It also removes the dependence of the gradient on the weight scale: with normalization downstream, scaling $W$ by $c$ leaves the output unchanged and scales the gradient by $1/c$, which auto-corrects bad initialization. That's most of why it smooths the loss landscape and lets you use larger learning rates — the internal-covariate-shift story from the original paper is largely not the mechanism.

**Q5. Pre-norm or post-norm, and why?**
Pre-norm: $x + \text{Attn}(\text{Norm}(x))$. The residual stream is never normalized, so there's a completely clean identity path from loss to embeddings, and deep stacks train without learning-rate warmup. Post-norm normalizes after the addition, breaking that path, which is why the original transformer needed careful warmup. Pre-norm is the standard now. The cost is that the residual stream's magnitude grows with depth, which is why models add a final norm before the output head.

---

### Ladder 5 — Bias and variance

**Q1. State the decomposition.**
For squared loss, expected error at a point $= \text{bias}^2 + \text{variance} + \sigma^2$. Bias is how far the *average* model over training sets is from truth; variance is how much the model wobbles between training sets; $\sigma^2$ is label noise.

**Q2. Where does the noise sit, and can you ever reduce it?**
$\sigma^2$ is inherent to $P(y \mid x)$ given your features — the same $x$ maps to different $y$. No model or amount of data reduces it. The one thing that does is changing $x$: adding a feature that explains the variation moves noise into signal. So "irreducible" is relative to the feature set, not absolute, and that's the nuance worth voicing.

**Q3. Does deep learning obey this?**
Not in the classical U-shape. Massively overparameterized networks interpolate the training data — zero training error, capacity far exceeding $n$ — and still generalize. The decomposition is still algebraically true; what's false is the assumption that variance rises monotonically with parameter count.

**Q4. So explain double descent.**
Test error follows the classical U up to the interpolation threshold (roughly parameters $\approx$ examples), spikes there, then *decreases again* as you keep growing the model. At the threshold there's exactly one interpolating solution and it's forced to be jagged. Past it there are infinitely many, and SGD's implicit bias picks a low-norm, smooth one. It also appears in epochs (epoch-wise double descent) and in data size (more data can transiently hurt near the threshold).

**Q5. Practical implication?**
Don't stop at the first sign of degradation when scaling capacity — you might be sitting on the interpolation peak. And "reduce model size to reduce overfitting" is classical-regime advice; in the modern regime the answer is usually more data, more regularization, or a *bigger* model, not a smaller one.

**Q6. If capacity isn't what controls generalization, what does?**
Effective capacity under the training procedure, not parameter count. The optimizer's implicit bias (SGD finds low-norm, flat solutions), the regularizers, the data augmentation, and the architecture's inductive bias all constrain which of the many interpolating functions you land on. The honest current statement is that parameter count is a poor complexity measure for deep nets, classical uniform-convergence bounds are vacuous at this scale, and the field does not have a fully satisfying replacement.

---

### Ladder 6 — Embeddings

**Q1. What is an embedding?**
A learned dense vector for a discrete item, positioned so that geometry encodes relatedness. It replaces one-hot (sparse, orthogonal, no notion of similarity) with a low-dimensional space where distance means something.

**Q2. How does word2vec actually learn them?**
Skip-gram with negative sampling: predict context words from a center word, but instead of a full softmax over the vocabulary, do binary classification of the true pair against $k$ sampled negatives. Negatives are drawn from the unigram distribution raised to the 3/4 power, which upweights rare words. The objective is implicitly factorizing a shifted PMI matrix.

**Q3. Why do static embeddings fail, and what replaced them?**
One vector per word type, so "bank" gets a single point that's an average of all senses. Contextual models (ELMo, then transformers) produce a vector per *token occurrence*, conditioned on the sentence. Today's retrieval embeddings are transformer encoders trained contrastively — mean-pooled or CLS-pooled, then InfoNCE against in-batch negatives.

**Q4. Why cosine similarity and not dot product or Euclidean?**
Cosine normalizes away magnitude, which in these models correlates with token frequency rather than meaning. If you L2-normalize, cosine and dot product are equivalent and Euclidean distance is a monotone function of both — so the choice only matters for unnormalized vectors. The real requirement is to use the same metric the model was *trained* with; a model trained with cosine InfoNCE should be queried with cosine.

**Q5. Why does ANN search work, and what does it cost you?**
Exact search is $O(nd)$ per query, fine at a million vectors and not at a billion. HNSW builds a navigable small-world graph and gets logarithmic-ish search at high memory cost; IVF-PQ partitions and quantizes, trading recall for a large memory reduction. The cost is that you're now approximate — recall@k below 1 — and the recall/latency knob (`efSearch`, `nprobe`) is a real hyperparameter that silently degrades your retrieval quality if set wrong.

**Q6. Why does hybrid search beat dense retrieval alone?**
Dense embeddings are lossy compression of meaning, so they systematically miss exact-match needs: product SKUs, error codes, rare proper nouns, negation. BM25 nails those and misses paraphrase. Fusing the two ranked lists (reciprocal rank fusion, or a cross-encoder reranker over the union) recovers both. Recall goes up; the reranker is where you spend the latency.

---

### Ladder 7 — BatchNorm vs LayerNorm

**Q1. What's the difference?**
Which axis the statistics come from. BatchNorm normalizes each feature across the batch; LayerNorm normalizes each example across its features. Same formula, different reduction axis.

**Q2. Why do transformers use LayerNorm?**
Three reasons. Sequences have variable length, so batch statistics per position are ill-defined. Batch sizes per device are small in large-scale training, making BN's estimates noisy and forcing cross-device syncs. And BN behaves differently at train (batch stats) and test (running averages), which is a real source of train/serve skew — LayerNorm is identical in both modes and has no dependence on other examples in the batch.

**Q3. Then why does BatchNorm survive in vision?**
Convnets have large spatial dimensions, so BN's statistics are computed over batch × height × width — plenty of samples even with a modest batch. It also has a genuine regularizing effect from batch noise. When batch size is forced small (detection, segmentation), people switch to GroupNorm, which is the interpolation between LayerNorm and per-channel normalization.

**Q4. What breaks if you use BatchNorm with batch size 1, or at inference?**
Batch size 1 gives zero variance per feature, so the normalized output is garbage divided by $\epsilon$. At inference BN uses running averages accumulated during training — if your deployment distribution differs from training, those stats are wrong and predictions shift, with no error raised. This is the classic silent BN bug: fine-tuning with frozen BN in train mode keeps updating running stats on the new data.

**Q5. Why did RMSNorm win in LLMs?**
It drops the mean subtraction and the bias, keeping only the scale: $x/\text{RMS}(x) \odot \gamma$. Empirically re-centering contributes almost nothing, and removing it saves a reduction pass and a bit of memory — a few percent of runtime at scale, for free. It's now near-universal in current open-weight models (Llama, Qwen, Gemma, DeepSeek).

**Q6. Normalization is scale-invariant, so what is $\gamma$ actually for?**
Normalizing forces every layer's output to unit scale, which is a constraint the network didn't ask for — $\gamma$ (and $\beta$ where it exists) lets it undo that, up to and including recovering the identity. Practically it lets each feature choose its own operating scale while the *optimization* still benefits from the normalized parameterization. It also has a side effect worth knowing: weight decay on layers followed by a norm doesn't shrink the function at all, it only changes the effective learning rate, which is why people exclude norm and bias parameters from weight decay.

---

### Ladder 8 — Metric choice under class imbalance

**Q1. 1% of transactions are fraud. Your model gets 99% accuracy. Assessment?**
It may be predicting "not fraud" for everything. Accuracy is uninformative here because the majority class baseline already hits 99%. I'd look at the confusion matrix first, then precision and recall on the positive class.

**Q2. You report ROC-AUC of 0.97. Is that good?**
Suspiciously easy to get. FPR is $FP/(FP+TN)$, and with 99% negatives, $TN$ is huge — so hundreds of false positives barely move FPR. ROC-AUC is invariant to class balance by construction, which sounds like a virtue and here hides that your alert queue is 95% noise. PR-AUC is the honest curve at this prevalence, because precision has $FP$ in the denominator with $TP$, not $TN$.

**Q3. What's the baseline PR-AUC?**
The positive prevalence — 0.01. So a PR-AUC of 0.3 is a 30× lift, which is a real result, even though 0.3 looks bad next to an AUC of 0.97. Always state PR-AUC relative to prevalence.

**Q4. Should you resample to fix the imbalance?**
Usually no, as a first move. Resampling or class weights change the decision threshold implicitly and distort your predicted probabilities, which matters if downstream logic uses them. The cleaner approach is to train on the natural distribution with a proper scoring rule and then tune the threshold on validation against the actual cost matrix. If you do resample, recalibrate afterward (Platt or isotonic), because the shift is a known, correctable prior shift.

**Q5. So what's the metric you'd actually report to the business?**
Something tied to the operating point and the cost asymmetry. Precision at the recall you're required to hit, or recall at the precision your review capacity allows, or expected cost per transaction with real dollar values for a missed fraud versus a blocked good customer. A single scalar with no threshold attached is the wrong deliverable for a deployed classifier.

**Q6. How do you pick the threshold, concretely?**
Sweep it on validation and pick the point minimizing expected cost: $\mathbb{E}[\text{cost}] = C_{FP}\cdot FP(t) + C_{FN}\cdot FN(t)$. If costs are unknown, pick the operating constraint instead — the reviewers can handle 500 alerts a day, so take the top 500 scores and report the recall you get. Never ship 0.5; it's the default only because it's the midpoint of a range, not because it optimizes anything.

---

### Ladder 9 — RAG vs fine-tuning

**Q1. When do you use each?**
RAG for knowledge — facts that change, are private, are large, or need citation. Fine-tuning for behavior — format, tone, task structure, tool-calling conventions, domain jargon. Rough heuristic: if the failure is "the model doesn't know that," retrieve. If it's "the model knows it but won't answer the way I need," fine-tune. They compose, and production systems usually do both.

**Q2. Fine-tuning fails to add facts — why?**
Facts injected by fine-tuning are learned slowly, generalize poorly to paraphrased queries, and increase hallucination: you teach the model the *style* of confidently stating things in this domain, which it then does for facts it never saw. It's also unauditable — you can't cite it, can't delete a record, can't tell whether a given answer came from the new data or from pretraining.

**Q3. Where do RAG systems actually break?**
Retrieval, almost always, not generation. Chunking that splits an answer across boundaries; embeddings that miss lexical matches (IDs, part numbers) that BM25 would catch; a top-k too small to contain the answer; no reranker, so the right chunk sits at rank 40. Then generation failures: the answer is in context and the model ignores it, especially in the middle of a long context. Evaluate retrieval separately with recall@k — if the answer isn't in the retrieved set, no prompt fixes it.

**Q4. How do you evaluate the whole thing?**
Two layers. Retrieval: recall@k, MRR, NDCG against labeled query-document pairs. Generation: faithfulness (is every claim supported by the retrieved context), answer relevance, and context precision. LLM-as-judge is standard for the generation layer but needs a human-labeled calibration set, because judges have known biases — position, length, self-preference.

**Q5. If you have to fine-tune, what's the cheapest thing that works?**
LoRA or QLoRA. Freeze the base weights, learn a low-rank update $W + BA$ with rank 8-64, training well under 1% of the parameters. It fits on a single GPU, adapters are swappable per task, and quality is close to full fine-tuning for style and task adaptation. Before that, though: a better prompt and a few good few-shot examples solve a surprising share of the cases people reach for fine-tuning on.

**Q6. What's the failure mode of fine-tuning that people don't anticipate?**
Catastrophic forgetting and alignment regression. A narrow SFT run on a few thousand domain examples measurably degrades general instruction-following and safety behavior, because you're moving weights that encoded both. Mitigations: mix in a slice of general instruction data, keep the learning rate low, use LoRA so the base weights are untouched, and always evaluate on a general held-out suite alongside your domain eval — not just the task you fine-tuned for.

---

### Ladder 10 — Transformer scaling

**Q1. What do scaling laws say?**
Loss falls as a smooth power law in parameters, data, and compute: $L = L_\infty + A/N^\alpha + B/D^\beta$. Smooth and predictable over many orders of magnitude, which is why labs can forecast a large run's loss from small ones.

**Q2. What's Chinchilla-optimal?**
For a fixed compute budget $C \approx 6ND$, the loss-minimizing split is roughly 20 training tokens per parameter, scaling $N$ and $D$ together. GPT-3 was badly undertrained by this standard — 175B parameters on 300B tokens, about 1.7:1 — and Chinchilla at 70B on 1.4T tokens beat it. Epoch AI's 2024 replication found the paper's fitted constants were wrong but the ~20:1 policy holds.

**Q3. Then why is nobody training Chinchilla-optimal today?**
Chinchilla optimizes *training* compute only. Once you serve a model, inference dominates lifetime cost, and inference cost scales with $N$, not $D$. So you deliberately over-train a smaller model far past 20:1 — a 7B on trillions of tokens — accepting worse training efficiency for permanently cheaper serving. Modern releases are commonly at hundreds of tokens per parameter.

**Q4. Where does the compute go — give me the arithmetic.**
$6N$ FLOPs per token for training: $2N$ forward, $4N$ backward, one multiply-add per parameter per direction. Attention adds about $12Ldn$ per token, which is negligible until context length approaches model width. For a dense decoder, $N \approx 12Ld^2$ plus embeddings. For MoE, FLOPs use *active* parameters while memory uses *total* — which is the whole point of MoE and the thing people get backwards.

**Q5. What breaks the smooth curve?**
Data. Power laws assume fresh unique tokens; repeated data gives diminishing returns after roughly four epochs, and past that adding repeats is close to worthless. That's the binding constraint driving synthetic data, aggressive dedup, and multimodal corpora. Also worth saying: the laws predict *pretraining loss*, and downstream capability is a noisy, sometimes discontinuous function of loss — so smooth loss curves do not imply smooth benchmark curves.

**Q6. Why is inference the harder engineering problem now?**
Prefill and decode have opposite bottlenecks. Prefill processes the whole prompt in parallel and is compute-bound. Decode emits one token at a time, so every step re-reads all weights plus the whole KV cache from HBM for a single token of output — arithmetic intensity near zero, memory-bandwidth-bound. Everything in a serving stack follows from that: continuous batching to raise intensity, paged KV to stop fragmentation, GQA/MLA to shrink the bytes read, speculative decoding to get more tokens per weight-read, and prefix caching to skip prefill entirely on shared prompts.

---

## Breadth rapid-fire

Thirty seconds each. Answer, then stop talking.

### Classical ML

**Supervised vs unsupervised vs self-supervised?** Supervised has labels; unsupervised finds structure without them; self-supervised manufactures labels from the data itself (next token, masked token) and is the basis of all pretraining.

**Bias-variance tradeoff in one sentence?** Simple models are consistently wrong (bias), complex models are inconsistently right (variance), and total error is their sum plus irreducible noise.

**Generative vs discriminative?** Discriminative models $P(y\mid x)$ directly; generative models $P(x,y)$ and derives it. Generative can sample and handle missing features; discriminative usually wins on pure classification accuracy.

**Why does Naive Bayes work despite the independence assumption being false?** Correlated features distort the probability estimates but often not the argmax, and classification only needs the ranking to be right.

**Bagging vs boosting?** Bagging trains independent models on bootstrap samples in parallel and averages — reduces variance. Boosting trains sequentially, each model fitting the previous residuals — reduces bias. Bagging is hard to overfit; boosting is easy to overfit.

**Random forest vs gradient boosting — which do you reach for?** GBM (XGBoost/LightGBM/CatBoost) for maximum tabular accuracy; random forest when you want a strong baseline with almost no tuning and no risk of overfitting from too many trees.

**How does a decision tree choose a split?** It maximizes impurity reduction — Gini or entropy for classification, variance reduction for regression — greedily over all features and thresholds.

**Gini vs entropy?** Nearly identical results. Gini is cheaper (no log) and is the default.

**What does the kernel trick do?** Computes inner products in a high-dimensional feature space without ever constructing the mapping, so you get a nonlinear boundary at the cost of a kernel evaluation.

**What is a support vector?** A training point on or inside the margin. Only these determine the boundary — remove any other point and the solution is unchanged.

**How does k-means work and what's its weakness?** Alternate assigning points to nearest centroid and recomputing centroids. It assumes spherical, equal-size clusters, is sensitive to initialization (use k-means++), and needs $k$ chosen in advance.

**How do you choose $k$?** Elbow on inertia, silhouette score, or gap statistic — but usually downstream utility, since the elbow is often ambiguous.

**PCA in one sentence?** Project onto the top eigenvectors of the covariance matrix — the directions of maximum variance, which are also the directions minimizing reconstruction error.

**Must you scale before PCA?** Yes, if features have different units, otherwise the largest-variance feature dominates the components purely because of its scale.

**PCA vs t-SNE vs UMAP?** PCA is linear, deterministic, invertible, and preserves global structure. t-SNE and UMAP are nonlinear, preserve local neighborhoods, and distort global distances — visualization only, never as features.

**What's the curse of dimensionality?** As dimensions grow, volume grows exponentially, data becomes sparse, and all pairwise distances converge — so distance-based methods stop discriminating.

**What is collinearity and why care?** Correlated predictors make $X^\top X$ near-singular, so coefficients become huge and unstable with flipped signs. Predictions are fine; interpretation is not. Ridge fixes it.

**When is a linear model the right choice?** Small data, need for interpretability, genuinely linear-ish relationship, or as the baseline you must beat before anything else is justified.

**What is the ROC curve made of?** TPR versus FPR as you sweep the decision threshold across all values.

**Cross-validation — when do you not use k-fold?** Time series (use forward-chaining), grouped data (use GroupKFold so a group never spans folds), and very large data where a single held-out set is enough.

### Deep learning

**Why non-linear activations?** Without them, a stack of linear layers collapses to a single linear layer — depth buys nothing.

**Why did ReLU beat sigmoid?** Constant gradient of 1 for positive inputs, so no vanishing; sparse activations; and it's just a comparison, so it's fast.

**What is a dying ReLU and how do you avoid it?** A unit stuck with negative pre-activation for all inputs gets zero gradient forever. Avoid with lower learning rate, better init, or a leaky variant / GELU.

**Why GELU or SiLU over ReLU in transformers?** Smooth and non-monotone near zero, giving nonzero gradient for slightly-negative inputs, and empirically a small but consistent quality win. SwiGLU (a gated SiLU) is the current standard FFN.

**How do you initialize weights?** Xavier/Glorot for tanh-like ($\text{Var} = 2/(n_{in}+n_{out})$), He for ReLU ($2/n_{in}$). The goal is to keep activation and gradient variance stable across layers.

**Why can't you initialize all weights to zero?** Every unit in a layer computes the same thing and receives the same gradient, so symmetry never breaks.

**What does the learning rate actually control, and how do you pick it?** Step size in the loss landscape. Pick with an LR range test (sweep exponentially, look for steepest descent), then use warmup plus cosine decay.

**Why warmup?** Adam's second-moment estimates are noisy for the first hundreds of steps; a large LR then produces huge, badly-scaled updates. Warmup lets the estimates settle.

**Batch size effects?** Large batches give less gradient noise, better hardware utilization, and often slightly worse generalization; scale the LR roughly linearly with batch size, and use gradient accumulation to simulate large batches on small hardware.

**Adam vs SGD with momentum?** Adam converges faster with less tuning and is standard for transformers. SGD+momentum often generalizes marginally better on convnets and is still used there.

**What does gradient clipping do?** Rescales the gradient when its norm exceeds a threshold, preventing a single bad batch from destroying the weights. Norm 1.0 is the usual value for LLM training.

**Dropout at train vs inference?** Train: zero units with probability $1-p$ and divide by $p$. Inference: nothing at all — dropout is disabled. Forgetting `model.eval()` is a classic bug.

**Why don't we use dropout much in large transformers?** Pretraining on trillions of unique tokens is essentially single-epoch, so there's little to overfit; dropout mostly just slows convergence. It reappears during fine-tuning on small datasets.

**What is a residual connection for?** A gradient highway — the identity term in the Jacobian keeps gradients from vanishing through depth, making 100+ layer stacks trainable.

**CNN vs transformer for vision?** CNNs bake in locality and translation equivariance, so they're more sample-efficient on small data. ViTs have weaker inductive bias but scale better with data and are dominant at scale.

**What does a convolution's receptive field mean?** The region of the input a given output unit depends on. It grows with depth, kernel size, and dilation.

**Why did RNNs lose to transformers?** Sequential dependency prevents parallel training over the sequence, and the fixed-size hidden state is an information bottleneck over long ranges.

**What is teacher forcing?** Feeding ground-truth tokens rather than model predictions during training. It stabilizes training but creates exposure bias, since at inference the model conditions on its own outputs.

**Mixed precision — what and why?** Compute in bf16/fp16 with an fp32 master copy of weights. Roughly 2× faster and half the memory. bf16 is preferred now because its exponent range matches fp32, removing the need for loss scaling.

**Gradient checkpointing?** Discard intermediate activations in the forward pass and recompute them in the backward pass. Trades roughly 30% more compute for a large memory reduction.

**Data parallel vs tensor parallel vs pipeline parallel?** Data: replicate the model, split the batch. Tensor: split individual matrices across devices (needs fast interconnect). Pipeline: split layers across devices. Large runs combine all three; ZeRO/FSDP shards optimizer states, gradients, and parameters across data-parallel ranks.

### NLP and LLMs

**What is tokenization and why BPE?** Splitting text into subword units. BPE iteratively merges the most frequent pair, giving a fixed vocabulary with no out-of-vocabulary tokens — rare words decompose into pieces.

**Why does tokenization cause weird failures?** Character-level tasks (counting letters, reversing strings) and arithmetic are hard because the model sees opaque chunks, not characters. It's also why non-English text costs more tokens.

**Encoder-only, decoder-only, encoder-decoder — when each?** Encoder-only (BERT) for classification and embeddings, bidirectional context. Decoder-only (GPT/Llama) for generation, causal mask. Encoder-decoder (T5) for seq2seq like translation. Decoder-only dominates because it scales and unifies tasks.

**Why do transformers need positional encodings?** Self-attention is permutation-equivariant — without position information, a shuffled sentence gives an identical output.

**What is RoPE and why did it win?** Rotary embeddings rotate Q and K by a position-dependent angle, so the attention score depends only on relative distance. It needs no extra parameters and extrapolates further, and it's the standard now.

**How do you extend a model's context window?** Interpolate or rescale the RoPE base frequency (NTK-aware scaling, YaRN) plus a short fine-tune on long sequences. Architecturally, sliding-window or interleaved local/global attention keeps the cost down.

**Pretraining, SFT, RLHF — what does each do?** Pretraining learns language and world knowledge from raw text. SFT teaches the instruction-following format from demonstrations. RLHF/DPO aligns to preferences — helpfulness, harmlessness, tone — using comparison data.

**DPO vs PPO?** PPO trains a separate reward model and optimizes against it with RL, which is powerful and finicky. DPO reparameterizes so you optimize preferences directly with a classification-style loss — no reward model, no sampling loop, much simpler, and it's the common default.

**What causes hallucination?** The training objective rewards fluent plausible continuations, not truth; the model has no mechanism to represent "I don't know"; and RLHF can push toward confident answers. Mitigations: retrieval grounding, citation requirements, abstention training, and consistency checks.

**Greedy vs beam vs sampling?** Greedy is deterministic and repetitive. Beam is better for tasks with one right answer (translation), bad for open generation. Top-p/temperature sampling is standard for open-ended text.

**What is top-p (nucleus) sampling?** Sample from the smallest set of tokens whose cumulative probability exceeds $p$ — an adaptive cutoff, unlike top-k's fixed one.

**What is speculative decoding?** A small draft model proposes several tokens; the large model verifies them in one parallel forward pass and accepts the longest correct prefix. Output distribution is unchanged; latency drops 2-3×.

**What is quantization and what does it cost?** Storing weights (and sometimes activations) in fewer bits — int8, int4. Roughly linear memory savings. Weight-only int8 is near-lossless; int4 costs a little quality; activation quantization is harder because of outlier channels.

**LoRA — what is it?** Freeze the base model and learn a low-rank update $\Delta W = BA$ with rank typically 8-64. Under 1% of parameters trained, adapters are small and swappable, quality close to full fine-tuning for task adaptation.

**What is a mixture-of-experts model?** Each FFN is replaced by many experts with a router activating a few per token. Total parameters grow while per-token FLOPs stay flat — more capacity at fixed compute, at the cost of memory and routing/load-balance complexity.

**What's actually in a modern open-weight LLM?** Decoder-only, pre-norm RMSNorm, SwiGLU FFN, RoPE, grouped-query attention, often MoE at larger sizes, frequently sliding-window or QK-norm variations. That stack describes the current Llama/Qwen/Gemma/DeepSeek family.

**Chain-of-thought — why does it help?** It gives the model serial computation depth: intermediate tokens act as scratch memory, so multi-step problems aren't forced through a single forward pass.

**What is prompt injection?** Untrusted content in the context (a retrieved document, a webpage, a tool result) that the model treats as instructions. There's no complete fix — mitigations are privilege separation, treating retrieved text as data, output filtering, and confirmation for consequential actions.

### Probability and statistics

**Bayes' theorem, and the classic trap?** Posterior $\propto$ likelihood × prior. The trap is base rates: a 99%-accurate test for a 1-in-10,000 disease still yields mostly false positives.

**MLE vs MAP?** MLE maximizes likelihood; MAP adds a prior. MAP with a Gaussian prior is L2 regularization; with a Laplace prior it's L1. They converge as data grows.

**What is a p-value — precisely?** The probability of observing data at least this extreme *if the null hypothesis were true*. It is not the probability that the null is true, and not the probability the result was chance.

**Type I vs Type II error?** Type I is a false positive (reject a true null), rate $\alpha$; Type II is a false negative (fail to reject a false null), rate $\beta$; power is $1-\beta$.

**Why correct for multiple comparisons?** Testing 20 hypotheses at $\alpha=0.05$ gives about a 64% chance of at least one false positive. Bonferroni controls the family-wise rate conservatively; Benjamini-Hochberg controls the false discovery rate with more power.

**Central limit theorem?** Sample means of i.i.d. variables with finite variance approach a normal distribution as $n$ grows, whatever the underlying distribution.

**What is the law of large numbers versus the CLT?** LLN says the sample mean converges to the true mean; CLT describes the *shape and scale* of the remaining error, $\mathcal{N}(0,\sigma^2/n)$.

**Confidence interval vs credible interval?** A 95% CI is a procedure that covers the true parameter 95% of the time across repeated experiments. A credible interval says the parameter lies in that range with 95% posterior probability. Only the Bayesian one means what people think CIs mean.

**What is a bootstrap?** Resample with replacement from your data, recompute the statistic, repeat. The spread of those values estimates the sampling distribution — no distributional assumptions needed.

**Covariance vs correlation?** Covariance is unnormalized and unit-dependent; correlation is covariance scaled to $[-1,1]$ by the standard deviations.

**Does zero correlation imply independence?** No — correlation only measures linear association ($y = x^2$ on symmetric $x$ has zero correlation). It does imply independence for jointly Gaussian variables.

**Explain Simpson's paradox.** A trend present in every subgroup reverses when the groups are pooled, because group sizes are confounded with the effect. It's why you disaggregate before concluding anything.

**What is a confounder?** A variable that causes both treatment and outcome, creating association without causation. Randomization removes it; otherwise you must control for it.

**Expected value of a fair die, and of the max of two?** 3.5; and 4.47 for the max of two — compute via $P(\max \le k) = (k/6)^2$.

### Evaluation

**What's the first thing you check on a new model?** The confusion matrix and a handful of actual errors. Aggregate metrics hide the failure mode.

**Precision or recall — how do you decide?** By the cost asymmetry. Cancer screening: recall, a miss is fatal. Spam filtering: precision, a false positive loses real mail. Say the cost, then the metric.

**Why is F1 sometimes the wrong summary?** It weights precision and recall equally, which is almost never the true cost ratio, and it ignores true negatives entirely. Use $F_\beta$ or a cost-weighted metric when you know the asymmetry.

**Macro vs micro vs weighted average for multiclass?** Macro averages per-class metrics equally, so rare classes matter as much as common ones. Micro aggregates counts first, so it's dominated by frequent classes and equals accuracy in single-label settings. Weighted averages by support.

**What is calibration and why care?** Whether a predicted 0.7 corresponds to a 70% empirical rate. It matters whenever the probability feeds a downstream decision — expected-value thresholds, pricing, triage. Fix with Platt scaling or isotonic regression on a held-out set.

**How do you evaluate a generative model with no ground truth?** Task-grounded automatic metrics where they exist, LLM-as-judge with a human-calibrated rubric, pairwise human preference on a stratified sample, and behavioral regression tests on known failure cases.

**What's wrong with LLM-as-judge?** Position bias, verbosity bias, self-preference for its own family's outputs, and poor calibration on borderline cases. Mitigate by randomizing order, using a rubric, and validating against human labels.

**Offline metric improved, online metric didn't. What happened?** Distribution shift between logged and live data, feedback loops (your model changes what data you collect), a proxy metric that isn't the business metric, or leakage inflating the offline number.

**How do you design an A/B test for a model?** Define the primary metric and guardrails up front, power the test for the minimum detectable effect, randomize at the right unit (usually user, not request), run at least a full weekly cycle, and don't peek without sequential correction.

**What's a good baseline?** Whatever's in production, plus something trivially simple — the majority class, a heuristic rule, logistic regression on ten features. If you can't beat those, nothing else you report matters.

### Data and features

**Why split before scaling?** Fitting the scaler on all data leaks test-set statistics (mean, variance) into training. Fit on train, transform test.

**How do you handle missing data?** Understand the mechanism first (MCAR/MAR/MNAR). Then: drop if rare and random, impute with median/mode plus a missingness indicator, model-based imputation, or use a model that handles it natively (LightGBM, XGBoost).

**Is a missingness indicator worth it?** Often yes — the fact that a value is missing is frequently predictive (an unfilled optional field correlates with user behavior).

**How do you encode categoricals?** One-hot for low cardinality; target/mean encoding with out-of-fold computation for high cardinality; learned embeddings for very high cardinality in a neural model; ordinal only when order is real.

**What's the danger of target encoding?** Direct leakage — the target appears in the feature. You must compute it out-of-fold or with smoothing on training data only.

**Standardization vs normalization?** Standardize to zero mean and unit variance (default, and required for PCA/SVM/regularized linear). Min-max normalize to $[0,1]$ when you need bounded inputs. Trees need neither.

**How do you detect data leakage?** Suspiciously high validation performance, a single feature with outsize importance, anything computed after the prediction time, and IDs or timestamps that encode the target. Ask of every feature: would I have this at prediction time?

**How do you handle outliers?** Determine whether they're errors or real. Errors: fix or drop. Real: winsorize, log-transform, or use a robust loss (Huber, MAE) — don't silently delete real signal.

**What is feature importance and what's the catch?** Impurity-based importance is biased toward high-cardinality features and is computed on training data. Permutation importance on held-out data is more honest; SHAP gives per-prediction attributions but is expensive and assumes feature independence in its common form.

**How do you deal with imbalanced training data?** Class weights first, then threshold tuning. Resampling (SMOTE, undersampling) if needed, applied inside the CV fold only. Recalibrate afterward.

**What is covariate shift versus concept drift?** Covariate shift: $P(x)$ changes, $P(y\mid x)$ doesn't — often fixable by reweighting. Concept drift: $P(y\mid x)$ itself changes, which requires retraining. Monitor both.

### ML systems

**How do you serve a model?** Batch scoring to a table for non-latency-critical uses; a real-time endpoint behind a feature store otherwise. The hard part is guaranteeing training and serving compute features identically.

**What is training/serving skew and how do you prevent it?** The same feature computed differently in the pipeline and the server. Prevent with a shared feature-computation library or a feature store, plus logging serving features and comparing distributions against training.

**What do you monitor in production?** Input distributions, prediction distributions, latency and error rates, and delayed ground-truth metrics when labels arrive. Prediction drift is the early warning; metric degradation is the confirmation.

**When do you retrain?** On a schedule matched to drift velocity, or triggered by a monitoring threshold. Whichever it is, the pipeline should be automated and the new model must beat the old one on a frozen eval before promotion.

**How do you roll out a new model safely?** Shadow mode first (serve old, log new), then a small-percentage canary with guardrail metrics, then a ramped A/B with automatic rollback.

**How do you reduce inference latency?** Quantize, distill to a smaller model, batch requests, cache (including prompt/prefix caching for LLMs), use a compiled runtime (TensorRT, ONNX, vLLM), and cut the parts of the pipeline that aren't the model.

**What is knowledge distillation?** Train a small student on the teacher's soft output distribution. The soft targets carry inter-class similarity information that hard labels don't, so the student typically beats one trained from scratch on the same data.

**What is a feature store for?** A single definition of each feature, served consistently online and offline, with point-in-time-correct historical lookups so training data doesn't leak future values.

**How do you version an ML system?** Code, data, features, model artifact, and configuration — all four, because reproducing a result requires all of them. Model registry plus data versioning (DVC, LakeFS, or immutable snapshots).

**How would you build an LLM app that must not make things up?** Ground every claim in retrieval, require citations that you programmatically verify against the retrieved text, allow and reward abstention, keep temperature low, and add a verification pass for high-stakes outputs. Then measure faithfulness explicitly rather than assuming it.

---

## The questions people fumble

Each of these sounds like a warm-up. Each has a trap.

**1. What's the difference between parameters and hyperparameters?**
Parameters are learned from data by the optimizer (weights, biases, split thresholds). Hyperparameters are set before training and control the learning process (learning rate, depth, $\lambda$, number of layers).
*The trap:* people say "parameters are inside the model, hyperparameters are outside," which is vague. The real line is **who sets them** — the optimizer or you. Follow-up worth pre-empting: the number of parameters is itself a hyperparameter.

**2. Why do we split before scaling?**
Because fitting the scaler on the full dataset uses test-set mean and variance, which leaks information into training and inflates your validation score.
*The trap:* people say "to avoid leakage" and stop. The interviewer wants the mechanism (the *statistics* are the leak) and the follow-up: the same argument applies to imputation, target encoding, feature selection, and SMOTE — every fitted transform belongs inside the CV fold, which is why `Pipeline` exists.

**3. Does adding a feature always improve training error?**
For an unregularized model with a rich enough hypothesis class, it can never *increase* training error — the old solution is still available with a zero coefficient. So training error is non-increasing.
*The trap:* saying "yes, always improves." It's non-increasing, not strictly decreasing, and with regularization or a greedy learner (trees, early stopping) training error can genuinely go up. Test error is a different question entirely.

**4. Is a lower loss always a better model?**
No. Loss is a proxy. A lower loss on a different metric, a different data split, or with a different class balance isn't comparable. Cross-entropy can improve while accuracy at your threshold degrades, and a model can win on loss while failing the business constraint.
*The trap:* forgetting that loss and the deployed decision metric are different objects, and that losses are only comparable across models trained on identical data with an identical objective.

**5. Is more data always better?**
No. Redundant, mislabeled, or off-distribution data can hurt. If you're bias-limited, more data does nothing. And near-duplicate data actively harms LLM training.
*The trap:* people reflexively say yes. The real answer names the condition: more data reduces *variance*, so it helps only when variance is what's limiting you.

**6. Your model has 99% accuracy. Is it good?**
Unanswerable without the class balance and the baseline. At 99% negative prevalence, 99% accuracy is the do-nothing baseline.
*The trap:* congratulating yourself. Always ask for prevalence and the majority-class baseline before evaluating any accuracy number.

**7. Does correlation imply causation? — and the real version: how would you establish causation?**
No, and the useful answer is the second half: randomized experiment if possible; otherwise a natural experiment, instrumental variable, difference-in-differences, or regression discontinuity, each with its assumptions stated.
*The trap:* answering only "no." Everyone knows the slogan; the question is testing whether you know what to do instead.

**8. Why is ReLU non-linear? It's two straight lines.**
Because linearity requires $f(ax+by) = af(x)+bf(y)$ for all inputs, and ReLU fails that at the kink. Piecewise-linear is not linear. A network of ReLUs is a piecewise-linear function with exponentially many regions in depth — enough to approximate anything.
*The trap:* getting flustered. The answer is one sentence about the definition of linearity.

**9. What is the difference between a validation set and a test set?**
Validation is used repeatedly to select models and hyperparameters. Test is looked at once, at the end, to estimate generalization.
*The trap:* using the phrase "test set" for something you've tuned against fifty times. Once you make decisions based on it, it's a validation set and its score is optimistically biased. Nested CV exists for exactly this.

**10. Does a deeper network always fit better?**
No. Beyond a point, plain deep networks get *worse training* error — the degradation problem — which is optimization difficulty, not overfitting. That observation is what motivated residual connections.
*The trap:* attributing the failure of deep plain nets to overfitting. It shows up in training error, so it can't be overfitting.

**11. Why not just use accuracy for everything, given a balanced dataset?**
Even balanced, accuracy assumes symmetric error costs and throws away the model's confidence. Two models with identical accuracy can have very different calibration and very different behavior at a shifted threshold.
*The trap:* treating class balance as the only thing that makes accuracy bad. Cost asymmetry and information loss are the other two.

**12. Random forests don't overfit — true?**
Adding more trees doesn't overfit, because averaging more independent estimates only reduces variance. But an individual random forest absolutely can overfit if the trees are unconstrained and $n$ is small, and it overfits badly to noisy labels.
*The trap:* the folk claim is about tree count only. Say which knob you mean.

**13. Do transformers have $O(n^2)$ memory?**
Naively yes, but FlashAttention makes it $O(n)$ by tiling and never materializing the attention matrix. Time is still $O(n^2 d)$.
*The trap:* conflating time and memory complexity, and quoting the 2017 numbers for a 2026 implementation.

---

## Questions to ask them

Ask three or four. Each of these returns information you can't get from the job posting.

**"What does the first ninety days look like for this role?"**
A specific answer means they've thought about onboarding and have work queued. A vague one means the role is under-defined and you'll spend a quarter finding your own scope.

**"How do models get from a notebook to production here, and who owns them after?"**
This is the single most predictive question. If the answer involves a platform team, CI, and a monitoring story, you'll ship. If it's "we hand it to engineering," you'll spend your time in rewrite negotiations.

**"What's the last model that got deprecated, and why?"**
Tests whether they measure anything post-launch. A team that can't name a retired model probably doesn't monitor, which means no one knows which of their models are currently broken.

**"How do you decide what to work on — who sets the roadmap?"**
Distinguishes a team with product partnership from a team that takes tickets. Also reveals whether ML is treated as a research function or a service function, which changes the job entirely.

**"What's the split between data work, modeling, and infrastructure for someone in this role?"**
Everyone says 80% data work; ask anyway, because the *variance* in the answer between interviewers on the same team tells you whether the role is actually defined.

**"What's something the team tried that didn't work?"**
Willingness to answer honestly is a culture signal. A team that can discuss failures in detail is a team where you're allowed to have them.

**"How is success measured for this role at six and twelve months?"**
If it's "ship X" that's clear. If it's "improve the model," ask improve what by how much — and if they can't say, neither will your performance review.

**"What's the review process for a model that affects users — who has to sign off?"**
Tells you the risk posture and how much friction sits between you and a launch. Neither extreme is good, and you want to know which one you're walking into.

**To the hiring manager specifically: "What's your biggest constraint right now — headcount, data, compute, or org buy-in?"**
Names the actual problem you'd be hired to relieve, and lets you talk to it in your close.
