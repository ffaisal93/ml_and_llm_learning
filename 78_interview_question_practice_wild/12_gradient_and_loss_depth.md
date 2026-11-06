# Gradient and loss depth

**Foundation explainer:** Further depth · **Priority:** Important — second pass

Important: derive compact numerical rules and test boundary conventions.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f111"></a>

## F111. Derive the softmax cross-entropy gradient.

**Start here.** The fused gradient is predicted probability minus target probability.

**Work it through.**

1. Write L=-sum(y_j*logp_j), with targets summing to one.
2. Differentiate logp_j=z_j-logsumexp(z), yielding dL/dz_i=p_i-y_i.
3. Apply the chosen batch or mask reduction afterward and preserve zero gradients for excluded examples.

**Small example.** For p=[0.2,0.3,0.5] and class 2, the per-example logit gradient is [0.2,0.3,-0.5].

**Common trap.** Applying the full softmax Jacobian again after using the already-fused p-y result.

**Follow-up.** Does label smoothing change the structure? No; replace the one-hot target with the smoothed distribution.

---

<a id="f112"></a>

## F112. Derive a dense layer's input, weight and bias gradients.

**Start here.** Each parameter receives the sum of contributions from every example that used it.

**Work it through.**

1. For Y=XW+b with X:(N,D), W:(D,C), let upstream G:(N,C).
2. Compute dX=G@Wᵀ, dW=Xᵀ@G and db=sum(G,axis=0).
3. Do not average again unless the scalar objective's reduction requires it; G may already include the mean factor.

**Small example.** For one example x=[2,3] and scalar upstream 4, the weight gradient is [8,12].

**Common trap.** Summing the bias gradient over features instead of examples or silently transposing the weight convention.

**Follow-up.** How do extra leading dimensions work? Flatten independent positions or sum their contributions consistently.

---

<a id="f113"></a>

## F113. Differentiate binary focal loss conceptually.

**Start here.** Its gradient includes both cross-entropy's gradient and the derivative of the focusing weight.

**Work it through.**

1. Write L=(1-p_t)^gamma*(-logp_t), with any class weight explicit.
2. Apply the product rule: changing the logit changes the BCE term and the modulation term.
3. Use a stable differentiable implementation and compare gradients with finite differences rather than multiplying BCE gradients by a fixed-looking weight.

**Small example.** At gamma=0, the modulation is one and its derivative vanishes, recovering weighted BCE.

**Common trap.** Detaching the focusing factor unintentionally, which changes the objective's gradient.

**Follow-up.** Why inspect mislabeled hard examples? The focusing term can make them disproportionately influential.

---

<a id="f114"></a>

## F114. Handle multiple positive pairs in a contrastive batch.

**Start here.** A diagonal-only target is wrong when several keys are valid matches for a query.

**Work it through.**

1. Construct a positive relation mask from task labels or known matches.
2. Choose a documented multi-positive objective, such as log(sum positive exp scores)-log(sum all exp scores), with stable reductions.
3. Define how queries without positives are handled and ensure evaluation measures the same relevance relation.

**Small example.** If two captions correctly describe the same image, penalizing one as a negative contradicts the task labels.

**Common trap.** Using repeated entities as negatives solely because their row indices differ.

**Follow-up.** Can different multi-positive reductions behave differently? Yes; averaging positive log probabilities and logging their summed probability are different objectives.

---

<a id="f115"></a>

## F115. Choose the denominator in weighted cross-entropy.

**Start here.** Weighting examples and choosing an average are distinct conventions with different loss scales.

**Work it through.**

1. Compute per-example stable cross-entropy and multiply by its sample/class weight.
2. Specify whether to divide by N, the sum of active weights, or the number of unmasked examples.
3. Keep the reduction consistent between implementations and account for its effect on gradient scale and learning-rate comparisons.

**Small example.** For losses [1,3] and weights [1,3], weighted-sum/weight-sum is 2.5; weighted-sum/N is 5.

**Common trap.** Comparing raw losses from different denominators as if they represent the same quantity.

**Follow-up.** When can weighting hurt probability calibration? A reweighted training distribution can change the fitted class probabilities.

---

<a id="f116"></a>

## F116. Why does zero initialization fail in a hidden layer?

**Start here.** Identical hidden units can receive identical gradients and remain redundant instead of learning different features.

**Work it through.**

1. Trace the forward activations of two units with equal incoming weights and biases.
2. Show that their backward gradients are equal under symmetric outgoing connections, so gradient descent preserves symmetry.
3. Use an initialization that breaks symmetry with a scale appropriate to activation and fan-in; zero biases can still be reasonable.

**Small example.** A layer of 100 identically initialized units can behave like copies of one unit rather than 100 distinct features.

**Common trap.** Claiming all parameters, including biases in an otherwise random network, must be nonzero.

**Follow-up.** Why can both LoRA factors not start at zero? Their product gradients both vanish at that point.

---

<a id="f117"></a>

## F117. Compute variance accurately when values have a large common offset.

**Start here.** Avoid subtracting nearly equal large moments when the variance itself is small.

**Work it through.**

1. For in-memory arrays, subtract the mean first and average squared deviations.
2. For streaming data, use Welford's update for count, mean and centered sum of squares.
3. Choose population or sample variance deliberately and validate with a constant-offset transformation test.

**Small example.** The variance of [1,2,3] and [100000001,100000002,100000003] is the same mathematically, but naive moment subtraction can lose precision.

**Common trap.** Computing mean(x²)-mean(x)² and assuming float64 always prevents cancellation.

**Follow-up.** How does this affect LayerNorm? Stable centered variance protects normalization on offset inputs.

---

<a id="f118"></a>

## F118. Backpropagate through shared KV heads in GQA.

**Start here.** When several query heads use one KV head, their gradient contributions must accumulate into the shared tensors.

**Work it through.**

1. Compute each query head's attention derivatives using its mapped KV head.
2. Sum all dK and dV contributions assigned to each shared head before propagating through the KV projections.
3. Compare the result with an explicit repeated-head reference whose repeat operation's backward sums repeated entries.

**Small example.** If query heads 0 and 1 share K0, then dK0 receives dK_from_head0+dK_from_head1.

**Common trap.** Averaging contributions without a matching objective convention or keeping only the last head's gradient.

**Follow-up.** Does forward head sharing imply fewer query outputs? No; output head count remains Hq.

---

<a id="f119"></a>

## F119. Aggregate a masked metric across workers correctly.

**Start here.** Global means require global numerators and denominators, not an unweighted average of worker means.

**Work it through.**

1. Each worker computes the sum of included losses and the count or weight sum under the metric definition.
2. Reduce both sums across workers, then divide once, with an explicit zero-denominator policy.
3. Use the same weighting for comparisons and distinguish this reporting calculation from framework-specific gradient averaging.

**Small example.** Workers with losses totaling 2 over 2 examples and 18 over 6 examples have a global mean 20/8=2.5, not (1+3)/2=2.

**Common trap.** Averaging per-batch means when batch sizes differ.

**Follow-up.** How would padding affect perplexity? Exclude padding targets before accumulating token log loss.

---

<a id="f120"></a>

## F120. Implement a Gaussian image filter with explicit boundary handling.

**Start here.** A Gaussian filter replaces a pixel with a distance-weighted neighborhood average.

**Work it through.**

1. Choose sigma, kernel radius and a boundary policy such as reflection or zero padding.
2. Construct and normalize a one-dimensional Gaussian kernel, then apply it along rows and columns for a separable 2D filter.
3. Preserve channel independence unless requested otherwise and use a suitable floating accumulator.

**Small example.** A normalized kernel leaves an interior constant image unchanged; zero padding may darken borders under a different boundary contract.

**Common trap.** Comparing two implementations with different padding rules and attributing the difference to arithmetic.

**Follow-up.** Why separable? Two length-K passes cost O(HWK) rather than O(HWK²) for a direct K-by-K kernel.

---
