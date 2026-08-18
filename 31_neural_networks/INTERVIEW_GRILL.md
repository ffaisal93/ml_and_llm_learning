# Neural Networks Fundamentals — Interview Grill

> 50 questions on MLPs, activations, init, backprop, gradient pathologies. Drill until you can answer 35+ cold.

---

## A. MLP basics

**1. What does a single layer of an MLP compute?**
$h = \sigma(W x + b)$. Affine transform followed by element-wise non-linearity.

> **Saying it out loud.** A layer does two things: it mixes and it bends. First it takes a weighted sum of everything coming in and adds a bias — that's the linear part, a matrix multiply. Then it pushes each number through a non-linear function like ReLU, which is what lets the network represent anything more interesting than a straight line. That's the whole unit, and stacking it is what gives you a deep network. The number worth knowing is that the matrix multiply is where essentially all the compute goes; the activation is nearly free.

**2. Why does an MLP need non-linearity?**
Without it, stacked layers collapse to a single affine map $W_L \cdots W_1 x + c$. No expressive gain from depth.

> **Saying it out loud.** Without a non-linearity, depth is free of charge and worth nothing. Stack two linear layers and you get a matrix times a matrix, which is just another matrix — so a hundred-layer linear network is exactly as expressive as one layer. The non-linearity is what breaks that collapse and lets each layer build on the last. That's why the choice of activation matters so much and why the very first thing you check in a broken model is whether you accidentally left one out.

**3. State the universal approximation theorem.**
A one-hidden-layer MLP with non-polynomial activation can approximate any continuous function on a compact set to arbitrary precision (Cybenko 1989, Hornik 1991).

> **Saying it out loud.** The theorem says a single hidden layer, wide enough, can approximate any continuous function on a bounded region as closely as you like. It's a statement about existence, not about learning — it says the weights exist, not that gradient descent will find them, and not that the layer will be a reasonable size. That's the key caveat to add, because interviewers ask this to see whether you'll over-claim. The width required can be exponential in the input dimension, which is exactly why nobody builds one-layer networks.

**4. If one hidden layer is enough, why use depth?**
Width may need to be exponential. Depth is more parameter-efficient for hierarchical/compositional functions (Telgarsky 2016). Also, depth induces useful inductive biases.

> **Saying it out loud.** Because "enough" can mean astronomically wide. A shallow network can represent the function, but it may need exponentially many neurons where a deep one needs a linear number — there are functions provably requiring exponential width at depth two and polynomial width at depth three. The intuition is compositional: deep nets reuse intermediate features, so early layers learn edges and later ones learn faces, instead of memorizing every combination separately. So depth buys parameter efficiency and a useful inductive bias, and the price is that deep networks are much harder to optimize.

**5. Why is depth alone hard to train?**
Vanishing/exploding gradients. Without residual connections + normalization, networks past ~10 layers struggle.

> **Saying it out loud.** The problem is that gradients have to travel through every layer, and each layer multiplies them by something. If those factors are consistently below one, the gradient shrinks geometrically and the early layers barely learn; if they're above one, it blows up to NaN. Past roughly ten layers, plain stacked layers stop training at all. The fixes that made deep learning work are residual connections, which give the gradient a direct path, plus normalization and careful initialization — and with those, thousand-layer networks are routine.

---

## B. Activations

**6. Why is sigmoid problematic in hidden layers?**
Two reasons. (1) Saturates — gradient $\sigma'(x) = \sigma(x)(1-\sigma(x)) \leq 0.25$ everywhere, vanishes for $|x| > 5$. (2) Not zero-centered — outputs in $(0,1)$ cause weight gradients to all share sign.

> **Saying it out loud.** Sigmoid kills gradients, in two separate ways. Its derivative maxes out at 0.25 and falls off a cliff once the input is past about five in absolute value, so stacking ten layers multiplies the gradient by at most 0.25 to the tenth — around one in a million. And it isn't zero-centered: since every output is positive, all the weight gradients into a neuron share the same sign, which makes the optimizer zigzag. It's fine as an output activation for binary classification, where you want a probability; it's a bad choice anywhere in the middle.

**7. Why is ReLU the default?**
Cheap (just a max), non-saturating for $x > 0$ (gradient = 1 → no decay), induces sparsity (~50% activations zero). Solved the vanishing gradient problem for deep nets.

> **Saying it out loud.** ReLU is just max of zero and x, and that simplicity is why it won. For any positive input the gradient is exactly one, so it doesn't attenuate the signal as it passes back through layers — that's what fixed vanishing gradients and made deep networks trainable. It's also essentially free to compute, and it makes about half the activations exactly zero, which is a nice sparsity property. The cost is the dead-neuron failure mode: if a unit gets pushed to always output zero, its gradient is zero forever and it never comes back.

**8. What's the dying ReLU problem?**
If a neuron's pre-activation is negative for all training data, $\text{ReLU}'(x) = 0$ → no gradient → neuron is dead permanently. Caused by large negative bias or large LR pushing weights into dead region.

> **Saying it out loud.** A dead ReLU is a neuron that outputs zero on every single input, and because ReLU's gradient is zero on the negative side, it gets no gradient and can never recover. It's permanent. What causes it is usually a learning rate large enough to blast the weights or bias deep into negative territory in one step, and once there it's stuck. The number you'll hear is that with a bad configuration ten to forty percent of your network can die, which shows up as a model that trains but plateaus well above where it should.

**9. How do you fix dying ReLU?**
Leaky ReLU ($\max(\alpha x, x)$), GELU/SiLU (smooth, non-zero gradient everywhere), better initialization, lower learning rate.

> **Saying it out loud.** There are two families of fixes: change the activation or change the optimization. Leaky ReLU gives a small slope on the negative side — usually 0.01 — so there's always some gradient to climb back on, and GELU or SiLU do the same thing smoothly. On the optimization side, He initialization and a lower learning rate stop you from killing units in the first place, and that's usually the real problem. In practice most people just use GELU or SiLU in transformers and never think about it, which is the honest answer.

**10. What's GELU?**
$x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. Smooth, has stochastic regularizer interpretation. Standard in BERT, GPT-2, GPT-3.

> **Saying it out loud.** GELU is a smooth version of ReLU. Instead of a hard cutoff at zero, it multiplies the input by the probability that a standard normal draw is below it, so small negative values get squashed toward zero gradually rather than chopped off. That smoothness means there's a non-zero gradient everywhere, so neurons don't die, and it gives slightly better results in transformers. The intuition people use is stochastic regularization — you're softly gating each input by how large it is. The cost is that it's more expensive than a max, which is why ReLU is still around.

**11. What's SwiGLU and why is it now standard in LLMs?**
$\text{SwiGLU}(x) = \text{Swish}(xW_g) \odot (xW)$. Two parallel projections with element-wise gating. Empirically beats vanilla FFN consistently (Shazeer 2020). Used in Llama, PaLM, Mistral.

> **Saying it out loud.** SwiGLU is a gated feedforward block. Instead of one projection followed by an activation, you make two projections from the same input, run one of them through a Swish activation, and multiply them element-wise — so one branch acts as a learned gate on the other. It consistently beats a plain feedforward layer by a small but reliable margin, and the honest story from the paper is that it was found empirically, not derived. Because you now have three weight matrices instead of two, you shrink the hidden dimension to about two-thirds so the parameter count stays matched. It's standard in Llama, PaLM and Mistral.

**12. ReLU vs GELU empirically?**
GELU slightly better for transformers; ReLU still competitive and cheaper. SiLU/Swish often preferred over GELU in newer LLMs.

> **Saying it out loud.** Empirically the differences are small and consistent. GELU edges out ReLU in transformers by a fraction of a percent, and the newest LLMs mostly use SiLU inside a SwiGLU block. ReLU remains competitive and is cheaper, so if you're inference-bound it's a reasonable choice. The thing to say is that activation choice is not where your model's quality is decided — data, scale and optimization matter orders of magnitude more — so I wouldn't spend a week tuning it.

---

## C. Loss functions

**13. Why pair softmax with cross-entropy?**
Cross-entropy is the negative log-likelihood under a multinomial. Gradient simplifies to $\hat{y} - y$ — clean and easy. Canonical link function of the multinomial GLM.

> **Saying it out loud.** They fit together because the math cancels beautifully. Cross-entropy is the negative log-likelihood of a multinomial, and when you differentiate it through the softmax, everything collapses to predicted minus actual. That's it — no derivative-of-the-activation factor hanging around to shrink toward zero. The consequence is that when the model is badly wrong, the gradient is large, which is exactly the behavior you want. That's why they're paired, and why libraries fuse them into one operation for numerical stability.

**14. Why pair sigmoid with BCE?**
Same reason — gradient is $\hat{y} - y$. Mismatched pairings (e.g., MSE on sigmoid output) give vanishing gradients near saturation.

> **Saying it out loud.** Same reason as softmax with cross-entropy: the gradient works out to predicted minus actual, with no saturating factor. If you instead put mean squared error on top of a sigmoid, the gradient picks up the sigmoid's derivative, which is nearly zero when the model is confidently wrong — so the most badly mistaken examples produce the smallest updates, which is exactly backwards. That's the punchline: mismatched loss and activation give you vanishing gradients precisely where you need learning most.

**15. When is MSE the right loss?**
Continuous regression with Gaussian noise assumption. Identity output activation. NOT for classification (gradient vanishes near saturation).

> **Saying it out loud.** MSE is right when you're predicting a continuous number and your noise is roughly Gaussian, with a plain linear output and no squashing. Under a Gaussian likelihood, minimizing squared error is maximum likelihood, so it's principled, not arbitrary. It's wrong for classification for two reasons: the gradient vanishes when paired with a saturating output, and it's not the right likelihood for a categorical variable. It's also fragile with outliers, since the penalty grows quadratically — which is why Huber loss exists.

**16. Multi-class vs multi-label classification?**
Multi-class: one true class, softmax + CE. Multi-label: multiple true classes, sigmoid (per class) + sum of BCE.

> **Saying it out loud.** The question is whether the classes compete. Multi-class means exactly one label is right, so you use softmax, which forces the probabilities to sum to one and makes the classes trade off against each other. Multi-label means several can be true at once — a photo with a dog and a beach — so you use an independent sigmoid per class and sum the binary cross-entropies, letting each one be high simultaneously. Using softmax for a multi-label problem is a common bug, and it shows up as the model refusing to be confident about two labels at once.

---

## D. Backpropagation

**17. What is backpropagation?**
Reverse-mode automatic differentiation: chain rule applied backward through a computational graph to compute gradient of scalar loss w.r.t. all parameters in $O(\text{forward})$ time.

> **Saying it out loud.** Backprop is the chain rule applied backward through the computation graph, organized so you compute every parameter's gradient in one pass. The reason it's cheap is that you reuse intermediate results: instead of asking "how does each weight affect the loss" separately, you propagate the sensitivity backward layer by layer. That makes the backward pass cost about the same as the forward pass — roughly two times, in practice — regardless of how many millions of parameters you have. It's not a learning algorithm, it's just an efficient way to get gradients; gradient descent is what does the learning.

**18. Walk through backprop for a 2-layer MLP.**
Forward: $z_1 = W_1 x$, $h_1 = \sigma(z_1)$, $z_2 = W_2 h_1$, $\hat{y} = \mathrm{softmax}(z_2)$. Loss: $\mathcal{L} = -\sum y \log \hat{y}$.

Backward:
- $\delta_2 = \hat{y} - y$
- $\nabla_{W_2} \mathcal{L} = \delta_2 h_1^\top$
- $\delta_1 = W_2^\top \delta_2 \odot \sigma'(z_1)$
- $\nabla_{W_1} \mathcal{L} = \delta_1 x^\top$

> **Saying it out loud.** Go forward first: multiply by W1, apply the activation, multiply by W2, softmax to get probabilities. Then backward, and the first step is the nice one — the error at the output is just predicted minus true, because softmax and cross-entropy cancel. The gradient for W2 is that error outer-producted with the hidden activations. Then you push the error back through W2 transposed and multiply element-wise by the activation's derivative to get the hidden error, and W1's gradient is that outer-producted with the input. The pattern to notice is that every weight gradient is an error signal times the input that fed that layer — which is exactly why you have to cache activations during the forward pass.

**19. Why is reverse-mode used for ML?**
Loss is scalar (1 output), parameters are millions. Reverse-mode costs $O(\text{outputs}) = O(1)$ passes. Forward-mode costs $O(\text{inputs}) = O(\text{millions})$ — infeasible.

> **Saying it out loud.** It's about the shape of the problem. Reverse mode costs one pass per output; forward mode costs one pass per input. In deep learning we have millions of inputs — the parameters — and exactly one output, the scalar loss. So reverse mode gets the whole gradient in one backward sweep, while forward mode would need millions of sweeps. The tradeoff is memory: reverse mode has to store all the intermediate activations until the backward pass reaches them, which is why activation memory, not parameter memory, is usually what makes you run out of GPU.

**20. When would forward-mode be preferred?**
When inputs are few and outputs are many (e.g., computing a Jacobian-vector product, sensitivity analysis with few parameters).

> **Saying it out loud.** Forward mode wins when the shape flips — few inputs, many outputs. If you want to know how a handful of parameters affect a large output vector, forward mode gets you the whole Jacobian-vector product in one pass with almost no memory, since nothing needs to be cached. That's typical in sensitivity analysis, some physics simulations, and computing directional derivatives. It also composes nicely for higher-order derivatives, which is why frameworks implement both and combine them for Hessian-vector products.

**21. What does `loss.backward()` actually do?**
Walks the computation graph backward from `loss` to leaf tensors, applies stored backward formulas via chain rule, *accumulates* gradients into `param.grad`. (Hence the need for `zero_grad`.)

> **Saying it out loud.** It walks the graph that got recorded during the forward pass, from the loss back to every leaf tensor that requires gradients, applying each operation's stored backward rule. The important detail is the last one: it *accumulates* into the .grad field rather than overwriting it. That's a deliberate design choice — it's what makes gradient accumulation across micro-batches possible — and it's also why forgetting to zero out gradients is one of the most common bugs in PyTorch. It also frees the graph by default, which is why calling backward twice throws an error unless you ask it to retain.

**22. Why call `optimizer.zero_grad()` before backward?**
Gradients accumulate in `.grad` — useful for gradient accumulation across mini-batches, but if you forget to zero, gradients from previous batches contaminate current ones. Classic bug.

> **Saying it out loud.** Because gradients add up rather than replace. If you don't zero them, this batch's gradient gets stacked on top of last batch's, so your update is a stale mixture and effectively an ever-growing learning rate — the loss usually looks noisy or diverges. The reason PyTorch does it this way is that accumulation is a feature: you can run several micro-batches, let the gradients pile up, and take one optimizer step, which simulates a big batch on small hardware. So zero-grad isn't a wart, it's the price of that flexibility.

**23. What are activations stored for during forward pass?**
Backward pass needs them — gradient w.r.t. weights involves the input to that layer. Without storing, you'd have to recompute (gradient checkpointing trades memory for compute by doing exactly this).

> **Saying it out loud.** The backward pass needs the layer's input to compute the weight gradient — the gradient for a weight matrix is the error signal outer-producted with whatever went into that layer. So every intermediate activation from the forward pass has to stick around until backprop reaches it. That's why memory usage scales with batch size times sequence length times depth, and why activations, not weights, usually blow up your GPU. Gradient checkpointing is the escape hatch: throw most of them away and recompute them during the backward pass, buying big memory savings for about thirty percent more compute.

---

## E. Initialization

**24. What's the goal of weight initialization?**
Preserve activation variance (and gradient variance) across layers — prevent vanishing or exploding signals.

> **Saying it out loud.** The goal is to keep the signal at a constant scale as it moves through the network — not shrinking, not exploding — in both directions. Each layer multiplies variance by something, and you want that something to be about one; otherwise after fifty layers you're at ten to the minus twenty or ten to the plus twenty. So init sets the weight variance based on how many inputs feed each neuron. Get it wrong and the network either produces NaNs in the first few steps or sits there learning nothing, and both look mysterious until you check activation statistics.

**25. Derive LeCun and Xavier (Glorot) init.**
For $z = Wx$ with $W_{ij} \sim \mathcal{N}(0, \sigma^2)$ and $x_i$ iid with variance $v$: $\text{Var}(z_j) = n_{\text{in}} \sigma^2 v$. To preserve forward variance: $\sigma^2 = 1/n_{\text{in}}$ — that's **LeCun init** (for tanh/sigmoid/SELU). To preserve *both* forward and backward variance: $\sigma^2 = 2/(n_{\text{in}} + n_{\text{out}})$ — that's **Xavier (Glorot) init**.

> **Saying it out loud.** The derivation is one line of variance algebra. If a neuron sums n inputs, each independent with variance v, and the weights have variance sigma-squared, the output variance is n times sigma-squared times v. To keep the output variance equal to the input variance you need sigma-squared equal to one over fan-in — that's LeCun. But the backward pass has the same issue in the other direction, where fan-out is what matters, and you can't satisfy both exactly, so Xavier compromises with two over fan-in plus fan-out. Both were derived assuming a roughly linear activation around zero, which is why tanh works with them and ReLU doesn't.

**26. Why does He init differ from Xavier?**
ReLU zeros out half the activations, halving the variance contribution. Compensate: $\sigma^2 = 2/n_{\text{in}}$. Xavier was derived for tanh, where this issue doesn't apply.

> **Saying it out loud.** He init exists because ReLU throws away half the signal. Since it zeros every negative pre-activation, it cuts the output variance in half, so if you use Xavier the signal decays by a factor of two per layer — over twenty layers that's a million-fold shrink. He compensates by doubling the weight variance to two over fan-in. Xavier was derived for tanh, which is roughly linear and symmetric near zero, so no correction is needed there. The practical rule is: ReLU family gets He, tanh and sigmoid get Xavier or LeCun.

**27. What init does GPT-2 use?**
$\sigma = 0.02$ (fixed, not depending on fan-in), plus a $1/\sqrt{2L}$ scaling on residual outputs. Works because LayerNorm renormalizes activations regardless.

> **Saying it out loud.** GPT-2 does something that looks lazy and works fine: a fixed standard deviation of 0.02 for all weights, ignoring fan-in entirely. It gets away with it because LayerNorm renormalizes activations at every block anyway, so the exact input scale stops mattering after the first layer. The part that does matter is the extra scaling on the residual projections by one over the square root of two times the number of layers — that keeps the residual stream from growing as you add blocks, since each block adds variance to a running sum. That residual scaling is the piece people forget, and it's what lets you stack depth without the activations drifting upward.

**28. What happens with all-zero init?**
All neurons compute the same thing → identical gradients → never break symmetry. Network never learns. Bias to zero is fine; weights need random init.

> **Saying it out loud.** Nothing happens, forever. If every weight is identical, every neuron in a layer computes the same output, receives the same gradient, and updates identically — so they stay identical for all of training and your thousand-unit layer has the expressive power of one unit. That's the symmetry-breaking problem, and it's why weights need random init. Biases are different: initializing them to zero is fine and standard, because the random weights already break the symmetry.

**29. What happens if weights are too large?**
Activations explode, gradients explode, NaN. Especially with deep networks — $\text{Var}(h^{(L)}) = (\sigma^2 n)^L$ blows up if $\sigma^2 n > 1$.

> **Saying it out loud.** You get an explosion, usually within a handful of steps. Each layer multiplies the variance by sigma-squared times fan-in, so if that product is above one, the activations grow geometrically with depth — the loss goes to NaN and training is over. Too small has the mirror problem: signal decays to nothing and gradients vanish. This is precisely why init formulas set that product to one, or two for ReLU. And when you see NaNs in the first few hundred steps of a new model, init and learning rate are the first two things to check.

---

## F. Vanishing and exploding gradients

**30. Why do gradients vanish in deep sigmoid networks?**
Each layer multiplies gradient by $\sigma'(z) \leq 0.25$. After $L$ layers: gradient scaled by $\leq 0.25^L \to 0$.

> **Saying it out loud.** Each sigmoid layer multiplies the backward signal by its derivative, which never exceeds 0.25. So the gradient reaching layer one of a ten-layer network has been multiplied by at most 0.25 to the tenth power, which is about one in a million — and that's the best case, since the derivative is much smaller for saturated units. The result is that early layers effectively don't train while late layers do, and the network looks like it's learning but plateaus early. That single fact is why ReLU, whose derivative is exactly one on the positive side, changed everything.

**31. Five fixes for vanishing gradients?**
(1) Non-saturating activations (ReLU/GELU). (2) Better init (He/Xavier). (3) Normalization (BN/LN/RMSNorm). (4) Residual connections. (5) Architectures designed for long-range gradient flow (LSTM gates, transformer attention).

> **Saying it out loud.** Five things, and modern architectures use all of them at once. Non-saturating activations, so the per-layer factor is one instead of a quarter. Proper initialization, so the scale starts right. Normalization layers, which reset the scale at every block. Residual connections, which give the gradient an identity path straight to the early layers. And architectures built for long-range flow — LSTM gates, attention. The reason we can train hundred-layer networks now isn't one breakthrough, it's the stack of all five.

**32. How do residual connections help?**
$h^{(\ell+1)} = h^{(\ell)} + F(h^{(\ell)})$. Gradient: $\partial h^{(\ell+1)} / \partial h^{(\ell)} = I + \partial F / \partial h^{(\ell)}$. Identity term ensures gradient never fully vanishes — there's always a direct gradient path.

> **Saying it out loud.** A residual connection adds the input back to the output of a block, and the gradient consequence is what matters. When you differentiate, you get identity plus the block's own Jacobian — so even if the block's contribution is tiny, the identity term guarantees the gradient reaches earlier layers undiminished. It turns a product of small numbers into a sum with a one in it. The other useful framing is that each block only has to learn a *correction* to what came before, so a block that learns nothing is harmless rather than destructive. That's what let ResNet go from about twenty layers to over a hundred.

**33. What's gradient clipping?**
Cap $\|\nabla\| \leq \tau$ (clip by norm) or $|\nabla_i| \leq \tau$ (clip by value). Prevents loss spikes from exploding gradients. Standard for transformers (typically $\tau = 1.0$).

> **Saying it out loud.** Gradient clipping is a safety valve. You compute the global norm of all gradients and, if it exceeds some threshold, scale the whole vector down to that norm — preserving direction, capping magnitude. It's there for the rare catastrophic batch that produces an enormous gradient and would otherwise blow your weights apart in one step, which shows up as a loss spike that never recovers. Clipping by global norm is preferred over clipping per-element, because per-element clipping distorts the direction of the update. The standard value for transformer training is 1.0.

**34. Why are RNNs especially prone to vanishing gradients?**
BPTT unrolls a single weight matrix $W$ across $T$ time steps. Gradient is $W^T \cdot \prod \sigma'$. If $\|W\| < 1$, gradient vanishes; if $\|W\| > 1$, explodes. LSTMs use gates to maintain a roughly identity state path (similar idea to residual).

> **Saying it out loud.** An RNN applies the *same* weight matrix at every time step, so backpropagating through a hundred steps multiplies by that matrix a hundred times. That's essentially raising it to a power, and unless its largest singular value is almost exactly one, you either vanish to zero or explode. It's a much sharper version of the depth problem, because in a deep feedforward net the layers at least have different weights that can partially compensate. LSTMs fix it by adding a cell state with additive updates gated in and out, which is the same trick as a residual connection — an identity path the gradient can flow along.

**35. Pre-LN vs Post-LN — which is more stable?**
Pre-LN ($h + F(\text{LN}(h))$) — standard in modern LLMs. Gradient flows through the residual path without going through LN first, which keeps it well-scaled. Post-LN (original transformer) is harder to train deep (requires careful warmup).

> **Saying it out loud.** Pre-LN is more stable and it's what everyone uses now. The difference is whether the normalization sits inside the residual branch or after the addition. With pre-LN, the residual path is a clean identity from input to output, so gradients flow straight to the early layers unchanged. With post-LN the gradient has to pass through a normalization layer at every block, and the scaling compounds — deep post-LN transformers often diverge without a long, carefully tuned warmup. The tradeoff is that post-LN, when you can train it, sometimes ends up slightly better, which is why some labs use hybrid schemes like sandwich norm.

---

## G. Training loop

**36. What's a typical PyTorch training loop?**
For each batch: zero gradients, forward, compute loss, backward, optionally clip gradients, optimizer step, scheduler step.

> **Saying it out loud.** The loop is five lines and the order matters. Zero the gradients, run the forward pass, compute the loss, call backward, then step the optimizer — and typically clip gradients between backward and step, and step the learning-rate scheduler after. The two classic bugs are forgetting zero-grad, which silently accumulates last batch's gradient, and clipping in the wrong place, since clipping after the optimizer step does nothing at all. And you wrap the eval pass in no-grad, or you'll quietly build a graph you never free.

**37. What's gradient checkpointing?**
Trade memory for compute — don't store activations during forward pass; recompute them during backward. Used to fit large models in memory at the cost of ~33% slowdown.

> **Saying it out loud.** Checkpointing trades compute for memory. Normally every activation from the forward pass is kept so the backward pass can use it, and that's what eats your GPU. Instead you keep only a few checkpoints — say one per layer group — throw the rest away, and recompute them on the fly during backward. The saving is large, roughly the square root of the depth in the ideal layout, and the cost is about one extra forward pass, so around thirty percent slower. You use it when memory is the binding constraint, which for large models it almost always is.

**38. What's gradient accumulation?**
Run forward+backward on multiple micro-batches without optimizer.step(), then step. Effective batch size = micro-batch × accumulation steps. Used when memory limits batch size.

> **Saying it out loud.** Gradient accumulation is how you fake a big batch on small hardware. You run forward and backward on several micro-batches without stepping the optimizer, letting the gradients pile up in .grad, then take one step. Effective batch size is micro-batch times accumulation steps. It works because gradients are additive — with the caveat that you have to scale the loss by one over the accumulation count so you get an average, not a sum. The one thing it doesn't fix is batch norm, whose statistics are still computed per micro-batch.

**39. What does mixed-precision training do?**
Forward/backward in FP16 or BF16, weights and optimizer state in FP32. Faster, less memory. BF16 is preferred over FP16 for stability (no dynamic loss scaling needed).

> **Saying it out loud.** Mixed precision keeps two copies of the numbers. The heavy matrix multiplies run in sixteen-bit — either FP16 or BF16 — which is roughly twice as fast on tensor cores and halves activation memory, while the master weights and optimizer state stay in thirty-two-bit so small updates don't get rounded away. BF16 is preferred now because it has the same exponent range as FP32, so it just doesn't overflow, whereas FP16's narrow range requires dynamic loss scaling to keep small gradients from flushing to zero. That's the practical difference: BF16 trades precision for range, and range is what training actually needs.

**40. What's a learning rate scheduler typically doing?**
Warmup (linearly increase LR from 0) + decay (cosine, linear, or constant). Warmup prevents early instability; decay refines at the end. LLMs typically use cosine decay to ~10% of peak.

> **Saying it out loud.** A scheduler does two jobs. Warmup ramps the learning rate up from near zero over the first few thousand steps, because at initialization the gradients are large and poorly conditioned and a full-size step will wreck the model — this matters especially with Adam, whose variance estimates are unreliable early. Then decay, usually cosine, brings the rate down so late training can settle into a sharper minimum instead of bouncing around it. The typical LLM recipe is a couple of thousand warmup steps, then cosine decay to about ten percent of peak.

---

## H. Modern architectures

**41. How is a CNN different from an MLP for backprop?**
Same chain-rule math, but convolution instead of matmul → weight sharing across spatial positions. Backprop convolution is convolution with flipped kernel.

> **Saying it out loud.** The math is identical — it's the same chain rule — but the structure of the weights changes what the gradient looks like. In a convolution, one small kernel is applied at every spatial position, so the gradient for that kernel is the sum of contributions from every position it touched. That's weight sharing, and it's why a CNN has far fewer parameters than an equivalent MLP. The neat fact is that backpropagating through a convolution is itself a convolution, with the kernel flipped, which is why the same fast primitives serve both passes.

**42. What's a transformer FFN block?**
A 2-layer MLP applied position-wise: $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$. Hidden dim typically 4× model dim. Modern variants use SwiGLU.

> **Saying it out loud.** The feedforward block is a two-layer MLP applied to each token independently — expand up, apply a non-linearity, project back down. The hidden dimension is conventionally four times the model dimension, so it holds about two-thirds of the parameters in a transformer, more than attention does. The way to talk about the division of labor is that attention moves information *between* tokens and the FFN does the computing *within* a token. Modern models swap the plain version for SwiGLU and shrink the hidden dimension to roughly eight-thirds of the model dimension to keep the parameter count matched.

**43. Why do CNNs typically use BN and transformers use LN?**
BN normalizes across batch — works well with large batches and image data (translation-invariant statistics). LN normalizes across features per-token — independent of batch size, works for variable sequence lengths, more stable for transformers.

> **Saying it out loud.** They normalize across different axes because their data has different structure. BatchNorm computes statistics across the batch for each channel, which works great for images where the batch gives you a decent estimate of a channel's distribution — but it ties every example's output to the other examples in its batch, and it breaks with small batches and variable-length sequences. LayerNorm normalizes across features within a single token, so it's completely independent of the batch, which is what you need for sequences that vary in length and for distributed training with tiny per-device batches. It also means training and inference behave identically, with no running-statistics bookkeeping.

**44. What's the role of dropout in modern transformers?**
Less critical than in early MLPs. Used in BERT-style training (10–20%); often removed or reduced in large LLMs that have implicit regularization from massive data + weight decay.

> **Saying it out loud.** Dropout matters much less than it used to. In the old regime you had a big model and a small dataset, so you needed aggressive regularization; BERT-style training still uses ten to twenty percent. In large LLM pretraining you're often making a single pass over a trillion tokens, so there's essentially nothing to memorize and dropout mostly just slows convergence — many large models set it to zero and rely on data scale and weight decay instead. It comes back for fine-tuning, where you're on a small dataset again and overfitting is a genuine risk.

**45. What's weight decay actually doing?**
$\ell_2$ penalty on parameters: $\mathcal{L} + \lambda \|w\|^2$. Pulls weights toward zero. In AdamW, decoupled from gradient (correct implementation); in vanilla Adam, it's coupled and weaker than intended.

> **Saying it out loud.** Weight decay pulls the weights toward zero a little at every step, which limits how large they can grow and biases the model toward simpler functions. The subtlety worth knowing is the difference between Adam and AdamW: in vanilla Adam the L2 penalty goes through the gradient and gets divided by the same per-parameter scaling as everything else, so parameters with large gradients get almost no decay — which is not what you asked for. AdamW decouples it and applies the decay directly to the weights, which is the correct implementation and why AdamW is the default now. Typical value is 0.1 for LLM pretraining, and you exclude biases and normalization parameters from it.

---

## I. Subtleties

**46. Can the loss go up during training?**
Yes — with momentum-based optimizers, large LR, or when the LR scheduler resets. Long-term trend should be down. Short-term noise is normal.

> **Saying it out loud.** Yes, and a little of it is healthy. Mini-batch gradients are noisy, so any single step can point slightly wrong; momentum carries you through regions where the loss temporarily rises; and a scheduler warm restart deliberately raises the rate and bumps the loss. What matters is the trend over a few hundred steps, not any individual spike. What isn't normal is a spike that never recovers — that's usually a bad batch plus an exploding gradient, and it's exactly the failure gradient clipping exists to prevent.

**47. What does it mean if training loss plateaus at a non-zero value?**
Model has reached a local minimum or saddle point given current capacity/data/optimizer. Possible fixes: more capacity, better optimizer, lower LR, data augmentation, regularization.

> **Saying it out loud.** A plateau means the model has stopped extracting anything new with its current capacity, data and optimization — and the first job is to figure out which of those three it is. If training loss is stuck high and validation tracks it, you're underfitting: more capacity, longer training, or a higher learning rate. If the learning rate has decayed to nearly nothing, that's the answer. The useful diagnostic is to overfit a single batch: if the model can't drive loss to zero on ten examples, the problem is a bug or the optimizer, not the dataset.

**48. Why might validation loss go up while training loss goes down?**
Overfitting. Model memorizes training data. Fixes: regularization, early stopping, more data, smaller model.

> **Saying it out loud.** That's overfitting, and the gap between the two curves is the size of it. The model has started memorizing training examples rather than learning patterns that generalize, which usually kicks in once capacity outstrips the amount of data. The fixes in order of what I'd actually try: early stopping, which is free; more data or augmentation, which is the real fix; then regularization — weight decay, dropout — and finally a smaller model. One caveat worth mentioning is that in very large models you sometimes see validation loss rise and then fall again, so it isn't always simply overfitting.

**49. What's catastrophic forgetting?**
Sequential training on task A then B → model forgets A. Common in RL, transfer learning, continual learning. Fixes: replay, EWC (elastic weight consolidation), PEFT (LoRA).

> **Saying it out loud.** Catastrophic forgetting is when you fine-tune on task B and the model loses task A, because gradient descent has no reason to preserve anything it isn't currently being scored on — the weights that encoded A just get overwritten. It shows up sharply in continual learning and in RL, and it's why aggressive fine-tuning can wreck a base model's general abilities. The mitigations are replay, meaning mix some old data back in; regularization like EWC that penalizes moving weights the old task cared about; and parameter-efficient methods like LoRA that freeze the base model entirely and learn a small adapter. LoRA is the practical answer most of the time, because the original weights are untouched by construction.

**50. Lottery ticket hypothesis?**
Frankle & Carbin (2018): dense networks contain sparse subnetworks ("winning tickets") that, trained from scratch with the same init, match the dense network's performance. Suggests over-parameterization is mostly about init/optimization landscape.

> **Saying it out loud.** The lottery ticket hypothesis says a big randomly-initialized network already contains a small subnetwork that, trained on its own from the *same* initialization, matches the full network's accuracy. The catch is that you can only find it by training the big network first, pruning, and rewinding to the original init — reinitializing the sparse network randomly doesn't work, which is the striking part. What it suggests is that over-parameterization is mostly about giving optimization many chances to find a good path, not about needing all those parameters at the end. Practically it hasn't delivered cheap training, since you still have to train the dense model to find the ticket.

---

## Quick fire

**51.** *Best init for ReLU?* He: $\sigma^2 = 2/n_{\text{in}}$.
**52.** *Best init for tanh?* LeCun: $\sigma^2 = 1/n_{\text{in}}$. (Xavier/Glorot is $2/(n_{\text{in}} + n_{\text{out}})$ — balances forward + backward; commonly used for tanh too.)
**53.** *Output activation for binary classification?* Sigmoid.
**54.** *Output activation for multi-class?* Softmax.
**55.** *Backprop time complexity?* $O(\text{forward})$ — about 2× forward.
**56.** *Why does sigmoid vanish?* $\sigma' \leq 0.25$, multiplies through depth.
**57.** *What does residual fix?* Vanishing gradients (identity path).
**58.** *Why pre-LN over post-LN?* Cleaner gradient flow through residual.
**59.** *FFN hidden dim ratio in transformers?* Typically $4d_{\text{model}}$.
**60.** *Standard gradient clip value?* 1.0 (clip by global norm).

---

## Self-grading

If you can't answer 1-15, you don't know neural networks. If you can't answer 16-35, you can't pass a deep-learning interview screen. If you can't answer 36-50, frontier-lab applied scientist interviews on training large models will go past you.

Aim for 40+/60 cold.
