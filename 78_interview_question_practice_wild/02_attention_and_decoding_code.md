# Attention and decoding code

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
