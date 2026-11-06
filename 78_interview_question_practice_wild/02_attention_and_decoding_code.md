# Attention and decoding code

<span class="practice-page"></span>

**Foundation explainer:** Core material · **Priority:** Very important

Very important: implement the primitives and defend their shapes and edge cases.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f011"></a>

## F011. Implement stable scaled dot-product attention.

**Start here.** Compute a normalized weighted average of values, with one probability distribution per query.

**Work it through.**

1. Project X into Q,K,V, then form QKᵀ/sqrt(d_k).
2. Replace forbidden scores with negative infinity. Subtract each row's maximum, exponentiate, and divide by the row sum.
3. Return probabilities @ V. Check mask shape and reject rows with no valid keys. See the executable attention implementation in the coding lab.

**Small example.** With only the diagonal allowed, output row i must equal value row i, regardless of the other scores.

**Common trap.** Softmax over the query axis instead of the key axis, or allowing masked keys to retain probability.

**Follow-up.** How would you test causality without inspecting the matrix? Perturb future inputs and compare earlier outputs.

---

### Detailed derivation and worked reasoning

### Derivation and a complete small example

For one token matrix $X\in\mathbb R^{n\times d_{\rm model}}$, learn projections

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

with $W_Q,W_K\in\mathbb R^{d_{\rm model}\times d_k}$ and $W_V\in\mathbb R^{d_{\rm model}\times d_v}$. The output has shape $n\times d_v$. Row $i$ asks which keys help query $i$; therefore softmax normalizes **across columns**, not across queries.

Let $X=W_Q=W_K=I_2$ and $V=\begin{bmatrix}1&2\\3&4\end{bmatrix}$. The score matrix is $I_2/\sqrt2$. The first row's probabilities are

$$
p_{11}=\frac{e^{1/\sqrt2}}{e^{1/\sqrt2}+1}\approx0.669762,
\qquad p_{12}\approx0.330238.
$$

Its output is $0.669762[1,2]+0.330238[3,4]\approx[1.660477,2.660477]$. This explicitly shows that the output is a weighted value mixture, not the selected key or a vocabulary probability.

For a boolean allowed-mask, set forbidden scores to $-\infty$ **before** softmax. Subtract each valid row maximum before exponentiation; this prevents overflow without changing the distribution. An all-forbidden row needs a declared policy because subtracting $-\infty$ from $-\infty$ creates NaN. This exercise guarantees at least one allowed key, and the reference rejects a violated contract.

**Cost.** Pairwise attention uses $O(n^2(d_k+d_v))$ arithmetic and $O(n^2)$ score storage in this straightforward implementation, in addition to the linear projections. **Checks.** Identity-only masking returns $V$ exactly; changing a forbidden future input must not change an earlier causal output.


<a id="f012"></a>

## F012. Implement grouped-query attention without losing the head mapping.

**Start here.** Several query heads share one key/value head, reducing stored K and V while retaining multiple query patterns.

**Work it through.**

1. Require Hq divisible by Hkv and let group_size=Hq/Hkv.
2. For query head h, use KV head h//group_size. Apply scaled attention independently for each query head.
3. For a simple reference implementation, repeat each KV head group_size times along the head axis; optimized kernels avoid materializing the repetitions.

**Small example.** With Hq=8 and Hkv=2, query heads 0–3 use KV head 0; heads 4–7 use head 1.

**Common trap.** Using h%Hkv, which produces alternating KV heads and a different grouping.

**Follow-up.** What are MHA and MQA here? Hkv=Hq gives MHA; Hkv=1 gives MQA.

---

### Detailed derivation and worked reasoning

### Derive the shared-head mapping

Let $H_q$ query heads share $H_{\rm KV}$ key/value heads, with $g=H_q/H_{\rm KV}$ an integer. For query head $h$, the contiguous grouping contract is

$$
k(h)=\left\lfloor\frac hg\right\rfloor,\qquad
O_h=\operatorname{softmax}\!\left(\frac{Q_hK_{k(h)}^\top}{\sqrt d}\right)V_{k(h)}.
$$

For eight query heads and two KV heads, heads $0,1,2,3$ use KV head zero and heads $4,5,6,7$ use KV head one. Mapping by $h\bmod H_{\rm KV}$ alternates heads and implements a different convention.

A clear reference loops over batch and query head, using the mapped KV head. Repeating KV tensors along the head axis can verify the result independently, but materializing repeats loses part of the memory advantage. The output still has $H_q$ heads; sharing keys does not average away query outputs.

The ideal cache storage ratio relative to otherwise equivalent ordinary MHA is $H_{\rm KV}/H_q$. With 32 query heads and eight KV heads, raw cached K/V elements drop fourfold. Query attention still computes a distribution for every query head, so do not claim all attention arithmetic also drops by four.

During backward, all query heads sharing a KV tensor contribute derivatives to it. Those contributions **sum**; retaining only the final head or averaging without an objective-based factor changes training. Test the mapping using KV values that differ unmistakably between groups.


<a id="f013"></a>

## F013. Add a KV cache and prove equivalence to full-prefix attention.

**Start here.** Previously computed keys and values do not need to be recomputed when a causal decoder receives one new token.

**Work it through.**

1. During prefill, compute and store all layerwise K and V tensors.
2. At each decode step, project only the new token, append its K,V, and attend its Q over the entire available cache. Use the correct position offset.
3. Compare each step's last-position logits against a fresh full-prefix forward pass in evaluation mode, using tolerances before comparing sampled tokens.

**Small example.** For a prefix of length 5, the new query has length 1 and the cache length becomes 6. It may attend to all six positions.

**Common trap.** Applying a top-left 1-by-6 triangular mask, which incorrectly permits only key 0; or failing to disable dropout in the comparison.

**Follow-up.** Why compare logits before samples? A tiny rounding change near a sampling boundary can change all later tokens.

---

### Detailed derivation and worked reasoning

### Why the cache preserves the computation

Causality means a previously computed token representation cannot depend on newly appended future tokens. Therefore its layerwise key and value remain reusable in evaluation mode. Cache them separately for every layer, request and compatible model/configuration.

At step $t$, compute only the new query, key and value, append the new K/V, and evaluate

$$
o_t=\operatorname{softmax}\!\left(\frac{q_t[K_1;\ldots;K_t]^\top}{\sqrt d}\right)
[V_1;\ldots;V_t].
$$

The single new query can see **all** cached prefix positions. A naive top-left triangular mask of shape $1\times t$ permits only the first key; if multiple new tokens are processed together, offset query positions by the existing cache length.

For RoPE, apply the correct absolute position to the new Q/K and do not rotate stored keys again. Compare cached last-position logits against a full-prefix run with dropout disabled. Test every prefix length before comparing sampled sequences, because tiny numerical changes near a sampling threshold can cause later samples to diverge.

For the attention interaction alone, cached one-token steps total $O(T^2d)$ across a generated length $T$, versus $O(T^3d)$ if a full growing-prefix attention calculation is redundantly recomputed each step. This excludes architecture-specific projection and prefill costs. The teaching code concatenates arrays and therefore copies memory; production caches normally use preallocation or blocks. Cache correctness is not, by itself, evidence of a particular speedup on a given machine.


<a id="f014"></a>

## F014. Implement top-k sampling from logits.

**Start here.** Keep a fixed number of candidate tokens, renormalize their probabilities and sample among them.

**Work it through.**

1. Validate 1<=k<=vocabulary_size and temperature>0.
2. Select the k greatest logits, divide retained logits by temperature, and compute stable softmax only over them.
3. Sample a local candidate index and map it back to the original vocabulary index. Use an explicit random generator for reproducible experiments.

**Small example.** For logits [1,2,0.5,4] and k=2, only token indices 1 and 3 can be returned.

**Common trap.** Returning the index inside the small candidate array instead of the original token id.

**Follow-up.** What happens at k=1? Greedy decoding, apart from the specified handling of exact ties.

---

### Detailed derivation and worked reasoning

### Define the candidate set before sampling

Sampling has three operations: choose retained tokens, normalize their scores, and draw a vocabulary id. With temperature $\tau>0$ and retained set $S$,

$$
P(i\mid i\in S)=\frac{\exp(z_i/\tau)}{\sum_{j\in S}\exp(z_j/\tau)}.
$$

Top-$k$ defines $S$ as the $k$ largest logits. Temperature changes the distribution within that fixed set but does not change its ordering for positive $\tau$. With logits $[1,2,0.5,4]$ and $k=2$, only original ids 1 and 3 may be returned. A local index zero in the two-element candidate array is not necessarily vocabulary id zero.

For top-$p$, first sort probabilities $p_{(1)}\ge\cdots\ge p_{(V)}$ and choose

$$
k^*=\min\left\{k:\sum_{j=1}^k p_{(j)}\ge p\right\}.
$$

Include the token that crosses the threshold, then renormalize. For $[0.6,0.25,0.15]$ and threshold $0.8$, retain the first two tokens, whose mass is $0.85$. Keeping only the first would fail the requested coverage.

A readable reference sorts in $O(V\log V)$; partial selection can improve top-$k$ selection cost but needs a deliberate equal-score rule. Use an explicit random generator to reproduce experiments. Test $k=1$, $k=V$, $p=1$, very small positive $p$ and tied logits. A fixed seed does not guarantee identical sequences after a change in floating-point reduction or the number/order of random draws.


<a id="f015"></a>

## F015. Implement nucleus, or top-p, sampling.

**Start here.** Keep the smallest highest-probability prefix whose cumulative mass reaches p, then renormalize and sample.

**Work it through.**

1. Compute stable softmax and sort token indices by descending probability.
2. Find the first cumulative sum >=p and include that boundary token. Always keep at least one token.
3. Sample from the retained probabilities after renormalization, then map back to vocabulary ids. Define p in (0,1].

**Small example.** For probabilities [0.6,0.25,0.15] and p=0.8, keep the first two tokens because their combined mass is 0.85.

**Common trap.** Dropping the token that first crosses the threshold, which retains less probability mass than requested.

**Follow-up.** Why does candidate count vary? A sharp distribution reaches p with fewer tokens than a flat one.

---

### Detailed derivation and worked reasoning

### Define the candidate set before sampling

Sampling has three operations: choose retained tokens, normalize their scores, and draw a vocabulary id. With temperature $\tau>0$ and retained set $S$,

$$
P(i\mid i\in S)=\frac{\exp(z_i/\tau)}{\sum_{j\in S}\exp(z_j/\tau)}.
$$

Top-$k$ defines $S$ as the $k$ largest logits. Temperature changes the distribution within that fixed set but does not change its ordering for positive $\tau$. With logits $[1,2,0.5,4]$ and $k=2$, only original ids 1 and 3 may be returned. A local index zero in the two-element candidate array is not necessarily vocabulary id zero.

For top-$p$, first sort probabilities $p_{(1)}\ge\cdots\ge p_{(V)}$ and choose

$$
k^*=\min\left\{k:\sum_{j=1}^k p_{(j)}\ge p\right\}.
$$

Include the token that crosses the threshold, then renormalize. For $[0.6,0.25,0.15]$ and threshold $0.8$, retain the first two tokens, whose mass is $0.85$. Keeping only the first would fail the requested coverage.

A readable reference sorts in $O(V\log V)$; partial selection can improve top-$k$ selection cost but needs a deliberate equal-score rule. Use an explicit random generator to reproduce experiments. Test $k=1$, $k=V$, $p=1$, very small positive $p$ and tied logits. A fixed seed does not guarantee identical sequences after a change in floating-point reduction or the number/order of random draws.


<a id="f016"></a>

## F016. Implement beam search with correct scores and end-of-sequence handling.

**Start here.** Beam search keeps several strong partial sequences; it is a search procedure rather than a random sampler.

**Work it through.**

1. Store each beam's tokens and sum of token log probabilities. Expand only unfinished beams.
2. Combine candidate expansions and retain the best beam_width sequences under the chosen ranking rule; keep finished hypotheses separately or carry them without further tokens.
3. Stop when the chosen stopping criterion is satisfied, and state whether length normalization is used. A minimal reference implementation is in the coding lab.

**Small example.** A sequence with token probabilities 0.5 and 0.5 has probability 0.25 and log score log(0.25), not the average probability 0.5.

**Common trap.** Multiplying many probabilities, appending tokens after EOS, or assuming a wider beam always improves subjective quality.

**Follow-up.** Why can raw likelihood prefer short answers? Every additional conditional probability is at most one.

---

### Detailed derivation and worked reasoning

### Track complete hypothesis state

A beam consists of token ids, accumulated log probability and completion status. Its raw score is

$$
s(y_{1:t})=\sum_{j=1}^t\log P(y_j\mid x,y_{<j}).
$$

Expand only unfinished hypotheses. Combine their continuations, compare them under the declared score and retain at most the beam width. A finished hypothesis must not acquire arbitrary tokens after EOS; it can be carried unchanged or stored in a separate completed set under a compatible stopping rule.

Multiplying probabilities directly underflows for long sequences; sums of logs avoid this problem. Raw likelihood often favors shorter completions because each added log probability is nonpositive. Length penalties change the ranking and stopping logic, so specify their formula rather than quietly dividing scores by length.

For example, two tokens with conditional probabilities $0.5$ and $0.5$ have joint probability $0.25$ and log score $\log0.25\approx-1.3863$. Their average probability is not a valid sequence likelihood. A width-one beam behaves like greedy search under the same constraints; a finite wider beam still is not exhaustive maximum-probability search.

Use a tiny callback model with known probabilities, enumerate its short possible completions, and compare behavior. Test immediate EOS, no EOS within the limit, equal scores and a prefix that initially looks weaker but later produces a stronger completion. This last case illustrates why pruning is approximate and why a larger beam is not a universal guarantee of better subjective answers.


<a id="f017"></a>

## F017. Compute softmax entropy with constant auxiliary memory.

**Start here.** Entropy can be calculated from normalization statistics without materializing the probability vector.

**Work it through.**

1. Scan by integer indexing to validate logits and find the finite maximum m; reject NaN, positive infinity, empty input and all-negative-infinity input.
2. Scan again in bounded inner blocks, accumulating Z=sum(exp(x-m)) and S=sum(exp(x-m)*(x-m)); negative infinity contributes zero.
3. Return log(Z)-S/Z. The shifted form avoids subtracting two huge nearly equal unshifted numbers. Keep only scalar accumulators.

**Small example.** For [0,0], Z=2 and S=0, so entropy is log(2). For one finite logit and the rest negative infinity, entropy is zero.

**Common trap.** Using slices or copying each block, which breaks the read-only lazy-sequence and O(1)-memory requirements.

**Follow-up.** Why skip negative infinity explicitly? Zero times negative infinity is NaN in floating-point arithmetic.

---

### Detailed derivation and worked reasoning

### Eliminate the probability vector algebraically

Choose $m=\max_i z_i$ and define shifted weights $w_i=e^{z_i-m}$ and $Z=\sum_iw_i$. Then $p_i=w_i/Z$ and

$$
H(p)=-\sum_i p_i\log p_i
=\log Z-\frac{\sum_iw_i(z_i-m)}{Z}.
$$

This identity requires only a maximum and two scalar sums. First scan the indexed input to validate values and find $m$; scan again to accumulate $Z$ and the weighted shifted sum. A block controls each inner loop's index range; no slice or block array is needed, so auxiliary memory remains $O(1)$.

For logits $[1000,999]$, shifting gives $[0,-1]$, $Z=1+e^{-1}$, and the weighted sum is $-e^{-1}$. The entropy is approximately $0.582203109$ nats. Computing huge unshifted exponentials would overflow even though the answer is ordinary-sized.

A negative-infinity logit has zero probability and should be skipped in the weighted sum: evaluating $0\cdot(-\infty)$ gives NaN. Empty input, nonpositive block size, NaN, positive infinity and an all-negative-infinity sequence violate the stated contract. Validate before publishing a result.

Test an object that supports only integer indexing and `len`: this catches accidental slicing or iteration hidden in convenient NumPy conversion. Changing block size must not change the mathematical result apart from harmless floating-point accumulation effects.


<a id="f018"></a>

## F018. Turn next-token probabilities into a binary classifier.

**Start here.** Define two mutually exclusive labels and compare their model scores under the same prompt and scoring convention.

**Work it through.**

1. For single-token labels A and B, compute P(A|A or B)=P(A)/(P(A)+P(B)), preferably from their log scores via stable normalization.
2. If a label spans multiple tokens, score its complete conditional sequence and state how termination is handled; one next-token probability is insufficient.
3. Choose the decision threshold on validation data, then measure calibration, class-specific errors and sensitivity to prompt changes.

**Small example.** If the label probabilities are 0.06 and 0.02, the conditional binary probability for the first is 0.75, even though most mass lies elsewhere.

**Common trap.** Reporting 0.06 as the binary probability without explaining unallocated token mass.

**Follow-up.** What if labels have different lengths? Sequence likelihood has length effects; choose controlled labels or explicitly validate the scoring rule.

---

<a id="f019"></a>

## F019. Debug a miniature GPT that fails to learn.

**Start here.** Localize the first broken invariant rather than changing many hyperparameters at once.

**Work it through.**

1. Check shifted labels, causal masking, tensor axes, vocabulary bounds and padding/loss masks on a tiny batch.
2. Check train/eval mode, gradient presence, optimizer steps and parameter changes. Overfit a tiny dataset with dropout disabled as a diagnostic.
3. Compare against a simple trusted baseline; then test generation and caching separately. Record one controlled change and its effect at a time.

**Small example.** For a uniform predictor over 10 tokens, the initial cross-entropy is about log(10)=2.303 nats. A wildly different baseline deserves investigation.

**Common trap.** Interpreting unusually low training loss as success before checking whether future tokens or targets leaked into inputs.

**Follow-up.** If tiny-batch overfitting works but validation fails, what changes? Investigate data splits, distribution shift and generalization rather than basic gradient plumbing.

---

<a id="f020"></a>

## F020. Implement a deterministic toy tokenizer, then explain what BPE adds.

**Start here.** A tokenizer turns text into a reproducible token sequence; BPE learns reusable merges beyond individual symbols.

**Work it through.**

1. Define the base alphabet and normalization policy, including whitespace and unknown characters.
2. For BPE training, repeatedly count adjacent pairs, choose the most frequent under a deterministic tie rule, merge non-overlapping occurrences and update the vocabulary.
3. At encoding time apply learned merge ranks, not a fresh frequency count on each document. Keep decoding and special-token handling explicit.

**Small example.** Starting from a a a a, merging pair (a,a) non-overlapping produces aa aa, not three overlapping replacements.

**Common trap.** Confusing longest-prefix matching over a fixed vocabulary with a complete BPE training algorithm.

**Follow-up.** Why do byte-level schemes help? Every byte sequence is representable without an unknown-character token, though token lengths vary.

---
