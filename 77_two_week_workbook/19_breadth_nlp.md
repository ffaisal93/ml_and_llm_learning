# The breadth round: NLP and deep learning

A breadth round is rapid fire. The interviewer asks twenty short questions in forty minutes. He wants the equation, the one distinction that matters, and the reason — nothing more. Every answer on this page is written for about a minute of speech — roughly 120 to 140 spoken words. The shape is always the same: the direct answer, then the equation or the mechanism, then one sentence of consequence, then stop. Let him ask for depth. A candidate who talks for four minutes on question one fails the round even when every sentence is correct.

## Neural network basics

### Q1. Walk me through the forward pass of a linear layer, with shapes.

A linear layer is one matrix multiply plus a bias. With a batch $X$ of shape $(B, d_{\text{in}})$, a weight $W$ of shape $(d_{\text{in}}, d_{\text{out}})$ and a bias $b$ of shape $(d_{\text{out}},)$, the output is

$$Y = XW + b$$

and $Y$ has shape $(B, d_{\text{out}})$. $B$ is the batch size, $d_{\text{in}}$ the input width, $d_{\text{out}}$ the output width. The bias broadcasts across the batch. Then an activation is applied elementwise, which does not change the shape. The whole network is that pattern repeated: multiply, add, squash.

### Q2. State backpropagation in one breath.

Backpropagation is the chain rule applied in reverse over the computation graph, with results cached so no partial derivative is computed twice. The forward pass stores each layer's input. The backward pass starts from $\partial L / \partial y$ at the output and walks backwards, and at each node it multiplies the incoming gradient by that node's local Jacobian. For a linear layer with $Y = XW + b$ that gives $\partial L/\partial W = X^\top G$ and $\partial L/\partial X = G W^\top$, where $G$ is the gradient arriving from above. It costs about the same as one forward pass.

**Walk the derivation.** Three steps, and I say them in this order.

1. **The output.** Take one linear layer $z = xW + b$ followed by an elementwise activation $a = \phi(z)$. The layer above hands me $\partial L / \partial a$. That is the only thing I start with.
2. **Through the activation.** The activation acts on each entry separately, so its Jacobian is diagonal. Therefore I multiply pointwise rather than by a matrix: $\partial L/\partial z = \partial L/\partial a \odot \phi'(z)$. Call that result $G$.
3. **Through the matrix.** Now $z$ depends on $W$, $b$ and $x$. Differentiating $xW + b$ gives $\partial L/\partial W = x^\top G$, $\partial L/\partial b$ as $G$ summed over the batch, and $\partial L/\partial x = G W^\top$. That last one is what I hand to the layer below, and the loop repeats.

### Q3. Why do we need non-linearity at all?

Because a stack of linear layers is still one linear layer. If $Y = (XW_1)W_2$, then $W_1 W_2$ is a single matrix, so depth buys nothing — the model can only fit hyperplanes. A non-linear activation between layers breaks that collapse, so each layer can bend the space and the composition represents functions no single affine map can. That is the whole reason deep networks are more expressive than linear regression. The activation must be non-linear; it does not need to be complicated, which is why ReLU works.

### Q4. What activation functions do you know, and what are the benefits and weaknesses of each?

| Function | Equation | Benefit | Failure mode |
|---|---|---|---|
| Sigmoid | $\sigma(x) = 1/(1+e^{-x})$ | Output in $(0,1)$, reads as a probability | Saturates both ends, gradient vanishes; output not zero-centred |
| Tanh | $\tanh(x) = (e^{x}-e^{-x})/(e^{x}+e^{-x})$ | Zero-centred, range $(-1,1)$ | Still saturates, so still vanishing gradient |
| ReLU | $\max(0, x)$ | Cheap, no saturation for $x>0$, sparse | Dying ReLU: a unit stuck at $x<0$ has zero gradient forever |
| Leaky ReLU | $\max(\alpha x, x)$, $\alpha \approx 0.01$ | Fixes dying ReLU, still cheap | Extra hyperparameter; kink is still non-smooth |
| ELU | $x$ if $x>0$ else $\alpha(e^{x}-1)$ | Smooth, negative saturation pushes mean to zero | Exponential costs more than a max |
| GELU | $x \cdot \Phi(x)$, $\Phi$ the standard normal CDF | Smooth, self-gating, strong empirically | Costlier; non-monotonic near zero |
| SiLU / Swish | $x \cdot \sigma(x)$ | Smooth, near-GELU quality, cheaper | Non-monotonic; small negative dip |
| Softmax | $e^{z_i}/\sum_j e^{z_j}$ | Turns logits into a distribution | Only for outputs, not hidden layers; overflow without the max trick |

Modern transformers use GELU or SiLU, usually inside a gated feed-forward block, because the smooth gate passes a small useful gradient where ReLU passes exactly zero.

**With numbers.** Two failure modes, and both are a gradient going to zero.

| Case | Output | Gradient |
|---|---|---|
| Sigmoid at $z = 0$ | $0.500$ | $0.250$ |
| Sigmoid at $z = 6$ | $0.9975$ | $0.0025$ |
| ReLU at $z = -3$ | $0$ | exactly $0$ |

A saturated sigmoid unit passes one hundredth of the gradient an unsaturated one passes, so it learns a hundred times slower but does eventually recover. The dying ReLU is worse than slow: its gradient is exactly zero, so the unit can never be pushed back on.

### Q5. How do you initialise weights, and why does the scale depend on fan-in?

Xavier for symmetric activations, He for ReLU. The argument is variance. If a unit sums $n_{\text{in}}$ independent terms $w_i x_i$, the output variance is $n_{\text{in}} \cdot \mathrm{Var}(w) \cdot \mathrm{Var}(x)$. To keep activation variance constant through depth I need $\mathrm{Var}(w) = 1/n_{\text{in}}$, which is Xavier. ReLU zeroes half the inputs and so halves the variance, so He doubles it to $\mathrm{Var}(w) = 2/n_{\text{in}}$. Get this wrong and activations shrink or blow up geometrically with depth, and training never starts.

**Walk the derivation.** Three steps, and I say them in this order.

1. **One unit.** A unit computes $z = \sum_{i=1}^{n_{\text{in}}} w_i x_i$. The terms are independent and zero-mean, so their variances add: $\mathrm{Var}(z) = n_{\text{in}} \cdot \mathrm{Var}(w) \cdot \mathrm{Var}(x)$.
2. **The condition.** I want the signal to stay the same size layer after layer, so I demand $\mathrm{Var}(z) = \mathrm{Var}(x)$. That forces $\mathrm{Var}(w) = 1/n_{\text{in}}$, which is Xavier.
3. **The ReLU correction.** ReLU zeroes the negative half, so it halves the mean square of what passes through. I therefore double the weight variance to pay for it, giving $\mathrm{Var}(w) = 2/n_{\text{in}}$, which is He.

**With numbers.** Ten ReLU layers of width $512$, starting from unit mean-square activations, averaged over twenty random draws.

| Weight variance | Mean square after ten layers |
|---|---|
| $2/n_{\text{in}}$, He | $1.03$ |
| $1/n_{\text{in}}$, Xavier | $0.0010$ |
| $0.25/n_{\text{in}}$ | $9.6 \times 10^{-10}$ |

Every layer applies the same factor, so a scale that is merely a factor of two too small loses a thousandfold over ten layers and a factor of eight too small loses a billionfold. The constant is derived, not tuned, because being close is not good enough.

### Q6. What are vanishing and exploding gradients, and how do we fix them now?

The backward pass multiplies many Jacobians together. If their typical singular value is below one the product shrinks to nothing, and if it is above one the product blows up. So early layers either stop learning or produce NaNs. The modern fixes are: non-saturating activations like ReLU and GELU, careful initialisation, normalisation layers, and residual connections that give the gradient a path of multiplication by one. For explosion specifically, gradient clipping by global norm. In recurrent nets, gating — the LSTM cell path — plays the residual role.

**With numbers.** The sigmoid derivative is largest at zero, where $\sigma'(0) = 0.25$. Stack ten sigmoid layers and the most favourable possible chain of derivatives is $0.25^{10}$, which is $0.00000095$. Every unit away from zero makes it smaller still. So the first layer of a ten-layer sigmoid net receives about one millionth of the gradient the last layer receives, which is exactly why those networks did not train before ReLU and residual connections.

### Q7. State the universal approximation theorem honestly.

A feedforward network with one hidden layer and a non-polynomial activation can approximate any continuous function on a compact set to any accuracy you choose, given enough hidden units. That is all it says. It does not say the required width is reasonable — it can be exponential in the input dimension. It does not say gradient descent will find those weights. It does not say the fit will generalise to new data. So it justifies neural networks as a hypothesis class, and says nothing about optimisation or generalisation, which are the parts that actually decide whether training works.

### Q8. How many parameters does an MLP have?

Count each layer as weights plus biases. A layer from $d_{\text{in}}$ to $d_{\text{out}}$ has $d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}}$ parameters. For an MLP with widths $784 \to 256 \to 128 \to 10$ that is $784 \cdot 256 + 256 = 200{,}960$, then $256 \cdot 128 + 128 = 32{,}896$, then $128 \cdot 10 + 10 = 1{,}290$, so $235{,}146$ in total. The weights dominate; biases are a rounding error. The useful takeaway is that parameter count grows with the product of adjacent widths, so widening a layer is quadratically expensive.

**With numbers.** Same input and output, only the hidden widths change.

| Widths | Parameters |
|---|---|
| $784 \to 256 \to 128 \to 10$ | $235{,}146$ |
| $784 \to 512 \to 128 \to 10$ | $468{,}874$ |
| $784 \to 512 \to 256 \to 10$ | $535{,}818$ |

Doubling the first hidden layer nearly doubles the whole model, because that layer touches the $784$-wide input. Doubling the second adds only fourteen percent, because it sits between two narrow neighbours. So width costs most where it meets the widest neighbour, which is where I look first when trimming a model.

### Q9. What is softmax, and what is the log-sum-exp trick?

Softmax maps a vector of logits to a distribution: $p_i = e^{z_i} / \sum_j e^{z_j}$, so the entries are positive and sum to one. Computed naively it overflows, because $e^{z}$ for $z = 1000$ is infinity in float. The trick is that softmax is invariant to adding a constant to every logit, so I subtract the maximum first: $p_i = e^{z_i - m} / \sum_j e^{z_j - m}$ with $m = \max_j z_j$. Now the largest exponent is $e^{0} = 1$ and nothing overflows. The same shift stabilises $\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}$.

**With numbers.** Take logits $[1000, 999, 998]$ in float32. The exponential overflows above about $88.7$, so all three become infinity, the sum is infinity, and every probability comes back as not-a-number. Subtract the maximum and the logits become $[0, -1, -2]$, the exponentials are $[1, 0.368, 0.135]$, and the probabilities are $[0.665, 0.245, 0.090]$. The shift changes nothing mathematically, because it cancels in the numerator and the denominator. It turns a total failure into the correct answer for the cost of one subtraction.

### Q10. What is cross-entropy, and why does it pair with softmax?

Cross-entropy is the negative log probability the model assigns to the correct class: $L = -\log p_y$ for a one-hot target. It pairs with softmax for two reasons. First, it is the maximum-likelihood loss for a categorical distribution, so minimising it is fitting the correct probability model. Second, the gradient collapses beautifully: for softmax plus cross-entropy, $\partial L / \partial z = p - y$, the predicted distribution minus the one-hot target. That is one subtraction, it never saturates, and it is why libraries fuse the two into a single stable op on logits.

**Walk the derivation.** Three steps, and I say them in this order.

1. **The loss.** With a one-hot target the loss is $L = -\log p_y$, and $p_y = e^{z_y} / \sum_j e^{z_j}$. Writing the log out gives $L = -z_y + \log \sum_j e^{z_j}$, which is two simple terms.
2. **The two terms.** Differentiating the first term against logit $z_i$ gives $-1$ when $i = y$ and zero otherwise, which is exactly $-y_i$. Differentiating the log-sum-exp gives $e^{z_i} / \sum_j e^{z_j}$, which is exactly $p_i$.
3. **The result.** Adding them gives $\partial L / \partial z_i = p_i - y_i$. No softmax derivative is left in the expression, so there is no factor that can go to zero. That cancellation is why the two are paired, and why libraries fuse them into one op on logits.

### Q11. Distinguish logits, probabilities and log-probabilities.

Logits are the raw real-valued scores the last linear layer emits; they are unbounded and only their differences matter. Probabilities are logits after softmax: in $[0,1]$ and summing to one. Log-probabilities are the log of those, so they are negative and add instead of multiply. In practice I keep everything in logit or log-probability space, because multiplying many small probabilities underflows while summing log-probabilities does not. Losses take logits for numerical stability, sampling and temperature act on logits, and sequence scoring sums log-probabilities.

### Q12. What does one step of gradient descent actually do?

It moves every parameter a small distance opposite to the gradient of the loss: $\theta \leftarrow \theta - \eta \nabla_\theta L$, where $\eta$ is the learning rate. The gradient points uphill in loss, so the minus sign goes downhill. In practice the gradient is estimated on a mini-batch, which makes it noisy but cheap, and that noise is mildly helpful for escaping poor regions. The learning rate is the one hyperparameter that matters most: too large and the loss diverges, too small and training takes forever.

**With numbers.** Take the simplest loss, $L = x^2/2$, where the update is $x \leftarrow (1 - \eta)x$ starting from $x = 1$.

| Learning rate | $x$ after fifty steps |
|---|---|
| $0.1$ | $0.0052$ |
| $1.0$ | $0$, in a single step |
| $2.1$ | $117$ and rising |

The step multiplies the distance by $1 - \eta$ every time, so anything below two converges and anything above two diverges. The threshold comes from the curvature of the loss, not from the size of the problem, which is why the usable learning rate changes the moment you change the architecture.

## Training deep networks

### Q13. What does batch norm normalise, and how do train and inference differ?

Batch norm normalises each feature across the batch dimension. For feature $j$ it subtracts the batch mean and divides by the batch standard deviation, then rescales with learned $\gamma$ and $\beta$:

$$\hat{x}_j = \frac{x_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \qquad y_j = \gamma_j \hat{x}_j + \beta_j$$

At training time $\mu$ and $\sigma$ come from the current batch. At inference there is no batch, so it uses running averages collected during training. That difference is a classic bug source: forget to switch to eval mode and your predictions change with batch composition.

### Q14. Why does batch norm interact badly with small batches?

Because the statistics are estimated from the batch itself. With a batch of four, the mean and variance are noisy, so each example is normalised by a number that depends on which other examples happened to land beside it. That adds gradient noise and creates a train-inference mismatch, since the running averages no longer match what any batch sees. It also breaks under sequence-length variation and makes distributed training awkward, because statistics must be synchronised across devices. That is the main reason sequence models moved to layer norm.

**With numbers.** I draw one feature from a standard normal and estimate the mean and the deviation from the batch itself.

| Batch size | Spread of the batch mean | Average batch deviation |
|---|---|---|
| $2$ | $0.71$ | $0.57$ |
| $256$ | $0.06$ | $1.00$ |

At batch size two the estimated deviation is on average forty-three percent too small, so every activation is divided by too small a number, and the subtracted mean wanders by $0.71$ from one batch to the next. At $256$ both estimates are essentially exact. The normaliser is noise at small batch, and the running averages kept for inference match neither.

### Q15. What is layer norm, and why do sequence models use it?

Layer norm normalises across the feature dimension of a single token, not across the batch. For one vector $x \in \mathbb{R}^{d}$ it uses that vector's own mean and variance, then applies learned $\gamma$ and $\beta$. Because the statistics come from one example, there is no batch dependence at all: training and inference compute exactly the same function, batch size one works, and variable sequence lengths cause no problem. Sequence models also process tokens one at a time at generation time, where a batch statistic would be meaningless. That is why every transformer uses layer norm or a variant.

### Q16. What is RMSNorm and what does it drop?

RMSNorm drops the mean subtraction. It divides by the root mean square of the vector and applies a learned gain, with no centring and usually no bias:

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma$$

So it rescales but does not recentre. The motivation is that the rescaling is what stabilises training, while the mean subtraction costs an extra pass over the vector for little benefit. It is slightly cheaper and empirically as good, which is why most recent large language models use it in place of layer norm.

### Q17. What does dropout do at train time and at test time?

At train time it zeroes each unit independently with probability $p$ and scales the survivors by $1/(1-p)$, so the expected activation is unchanged. That stops units from co-adapting, because no unit can rely on any particular neighbour being present. At test time dropout is off entirely and the network runs deterministically — the inverted scaling during training is what makes the test-time network match the training expectation. Typical $p$ is $0.1$ for large transformers and $0.5$ for wide fully connected layers. Modern large models often use little or no dropout, because the data is large enough.

### Q18. Why do residual connections help?

A residual block computes $y = x + F(x)$ instead of $y = F(x)$. On the backward pass the derivative of the identity branch is exactly one, so the gradient reaching layer $k$ is the gradient from above plus whatever flows through $F$. That gives every layer a direct path to the loss — a gradient highway — so depth no longer multiplies many small Jacobians together. It also makes the block's default behaviour the identity, so adding layers cannot make the function harder to represent. This is what made hundred-layer networks trainable.

**Walk the derivation.** Three steps, and I say them in this order.

1. **The block.** A residual block is $y = x + F(x)$. Differentiating against the input gives $\partial y / \partial x = I + \partial F / \partial x$: an identity matrix plus whatever the branch contributes.
2. **The stack.** Chaining $L$ blocks multiplies those sums together. Expanding the product gives one term that is pure identity, plus terms that pass through one branch, two branches, and so on up to all of them.
3. **The consequence.** The pure identity term is there whatever the branches do, so the gradient arriving at the bottom layer is the gradient at the top plus corrections. Depth stops being a chain of small multiplications. That surviving term is the highway.

### Q19. When and how do you clip gradients?

I clip when the loss occasionally spikes or produces NaNs, which is common in recurrent nets and in large transformer training. The standard form is clipping by global norm: compute the norm of the whole gradient vector across all parameters, and if it exceeds a threshold $c$, rescale the entire gradient by $c / \lVert g \rVert$. Rescaling the whole vector preserves direction and only shortens the step. A typical threshold is around $1.0$. Clipping per parameter instead of globally is worse, because it distorts the descent direction.

**With numbers.** Say a loss spike produces a gradient whose global norm is $37.4$, against a threshold of $1.0$. Clipping multiplies every component by $1.0 / 37.4 = 0.0267$, so the step is thirty-seven times shorter and the direction is untouched. On an ordinary step, where the norm might be $0.4$, the rule does nothing at all. So clipping is a rare intervention on the tail of the distribution, not a change to normal training.

### Q20. Explain learning-rate warmup and cosine decay.

Warmup raises the learning rate linearly from near zero to its peak over the first few hundred or few thousand steps. Early gradients are large and the adaptive optimiser's second-moment estimates are still unreliable, so a full-size step early can wreck the initialisation. After warmup, cosine decay lowers the rate smoothly to near zero following a half cosine over the remaining steps. The high middle phase explores, the low tail refines. The combination is the default schedule for transformer pretraining because it is stable early and converges cleanly late.

### Q21. What is label smoothing and why use it?

Label smoothing replaces the one-hot target with a soft target: mass $1-\epsilon$ on the correct class and $\epsilon$ spread over the other $K-1$ classes, with $\epsilon$ typically $0.1$. Without it, cross-entropy pushes the correct logit toward infinity, since the loss keeps falling as confidence rises. That gives overconfident, poorly calibrated models and large weights. Smoothing gives the loss a finite minimum, so logits stay bounded and confidence is tempered. It usually improves calibration and often accuracy slightly. The cost is that it degrades the model as a pure likelihood estimator.

**With numbers.** Five classes with $\epsilon = 0.1$.

| Target | Correct class | Each of the other four |
|---|---|---|
| One-hot | $1.0$ | $0$ |
| Smoothed | $0.9$ | $0.025$ |

With the one-hot target the loss keeps falling as the correct logit grows, so there is no finite minimum and the logits run away. With the smoothed target the loss bottoms out at $0.464$, reached when the correct logit sits exactly $3.58$ above the others. The bounded logit gap is the mechanism; better calibration is the effect.

### Q22. What is mixed precision, and why do you need a loss scaler?

Mixed precision keeps weights and the optimiser state in float32 but runs the matrix multiplies in float16 or bfloat16, which roughly halves memory and doubles throughput on modern accelerators. The problem with float16 is its narrow range: small gradients underflow to zero. The loss scaler fixes that by multiplying the loss by a large constant before the backward pass, which scales every gradient up into representable range, then dividing the gradients by the same constant before the optimiser step. Bfloat16 has float32's exponent range, so it usually needs no scaler.

**With numbers.** Float16 reaches down to about $0.000061$ as a normal number and $0.00000006$ as a subnormal. A gradient of $0.00000001$ therefore rounds to exactly zero and is gone. Multiply the loss by $1024$ before the backward pass and the same gradient arrives as $0.0000102$, comfortably inside range; dividing by $1024$ afterwards restores its true size. The scaler changes no mathematics at all. It only moves the numbers into the part of the format that exists.

### Q23. What is gradient accumulation and when do you use it?

Gradient accumulation runs several small forward and backward passes, sums their gradients, and only then takes one optimiser step. Doing that over $k$ micro-batches of size $m$ gives the same update as one batch of size $km$, at the memory cost of $m$. I use it when the batch size I want does not fit in memory — large-batch pretraining on limited hardware, or long sequences. Two details matter: average the loss over micro-batches rather than summing, and remember that batch norm statistics are not accumulated, so they still see only the micro-batch.

### Q24. How does overfitting look different in a deep net versus a shallow model?

In a shallow model overfitting shows as the classic U: validation loss falls, then rises, and more capacity always makes it worse. In a deep net the picture is messier. The network can drive training loss to nearly zero while validation loss stays flat or even keeps improving, and adding parameters past the interpolation point often helps rather than hurts. Validation loss can rise while validation accuracy holds, because the model gets confidently wrong on a few examples. So I watch the metric I care about, not only the loss, and I use early stopping on that metric.

### Q25. Adam or SGD — how do you choose?

SGD with momentum for convolutional vision models, Adam or AdamW for transformers and anything with sparse or badly scaled gradients. Adam keeps a running mean and a running variance of each gradient and divides by the square root of the variance, so every parameter gets its own effective step size. That handles the wildly different gradient scales across embedding, attention and normalisation parameters, which plain SGD does not. The cost is two extra optimiser states per parameter, so memory triples. AdamW is the version to use, because it decouples weight decay from the adaptive scaling.

## Sequence models before transformers

### Q26. Write the RNN recurrence and say why it fails on long sequences.

A vanilla RNN carries a hidden state forward one step at a time:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

where $h_t$ is the state at step $t$ and $x_t$ the input. It fails on long sequences because the gradient from step $T$ back to step $1$ is a product of $T$ Jacobians, all involving $W_h$. If the relevant singular values are below one the signal vanishes exponentially; above one it explodes. So the model cannot learn dependencies more than a few dozen steps apart. It is also strictly sequential, so it does not parallelise over time.

**With numbers.** Take a fifty-step sequence, and suppose the relevant factor in the Jacobian is a single number repeated at every step.

| Factor per step | After fifty steps |
|---|---|
| $0.9$ | $0.0052$ |
| $1.1$ | $117$ |

A ten percent shortfall per step leaves half a percent of the gradient after fifty steps. A ten percent excess multiplies it by more than a hundred. Nothing between those two is stable, so the model has to sit on a knife edge it cannot control, and that is the argument for replacing the multiplicative path with an additive one.

### Q27. Name the LSTM gates and say what each does.

Three gates plus a candidate. The forget gate decides how much of the previous cell state to keep. The input gate decides how much of the new candidate to write. The candidate is the proposed new content, a $\tanh$ of the inputs. The output gate decides how much of the cell state to expose as the hidden state. The cell state updates as $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, which is additive, so gradients flow along the cell path without repeated matrix multiplication. That additive path is the whole reason LSTMs handle longer dependencies than vanilla RNNs.

**Walk the derivation.** Three steps, and I say them in this order.

1. **The update.** The cell state moves by $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. The old state is carried forward by an elementwise multiply, not by a weight matrix.
2. **The derivative.** So $\partial c_t / \partial c_{t-1}$ is just $f_t$, a diagonal of gate values between zero and one. The gradient across $T$ steps is the product of those gates, and nothing else appears in it.
3. **The escape.** A forget gate near one keeps that product near one across many steps, so the gradient passes through almost unchanged. The vanilla RNN had no such choice, because its factor was always $W_h$ times a $\tanh$ derivative. Being able to learn the gate is what buys the long path.

### Q28. What is a GRU and what does it merge?

A GRU is a lighter LSTM. It merges the forget and input gates into one update gate, so a single number decides how much of the old state to keep and how much new content to write — they are tied to sum to one. It also drops the separate cell state, keeping only the hidden state, and replaces the output gate with a reset gate that controls how much past state feeds the candidate. So two gates instead of three and fewer parameters. In practice it trains faster and performs about the same as an LSTM on most tasks.

### Q29. What is a bidirectional RNN and when can you use one?

A bidirectional RNN runs two independent recurrences over the same sequence, one left to right and one right to left, and concatenates the two hidden states at each position. So every position's representation sees the whole sentence, not just its prefix. That helps a lot for tagging, named entity recognition and classification. The constraint is that it needs the full sequence up front, so it cannot be used for autoregressive generation or for streaming input. This is exactly the encoder-versus-decoder distinction that later appears as bidirectional BERT versus causal GPT.

### Q30. Describe seq2seq and its fixed-vector bottleneck.

Seq2seq uses two RNNs. The encoder reads the source sequence and its final hidden state becomes a single fixed-length vector. The decoder is initialised from that vector and generates the target one token at a time. The bottleneck is that one vector, typically a few hundred dimensions, must carry the entire meaning of the source. Performance therefore degrades sharply as the source gets longer, and long sentences lose their early content because the encoder state has been overwritten. Reversing the source word order helped a little, which tells you how bad the bottleneck was.

### Q31. How did attention fix the bottleneck, historically?

Attention removed the requirement that one vector carry everything. Instead of only the final encoder state, the decoder keeps all encoder states and, at each output step, computes a weighted average of them. The weights come from a learned score between the current decoder state and each encoder state, passed through a softmax. So the decoder can look back at the specific source positions it needs right now, and the gradient reaches those positions directly. This arrived as an addition to RNN seq2seq for translation; only later did the transformer drop the recurrence and keep attention alone.

### Q32. What are teacher forcing and exposure bias?

Teacher forcing means that during training the decoder is fed the true previous token rather than its own prediction. That makes training parallel and stable, because every position has a correct prefix. Exposure bias is the mismatch it creates: at generation time the model is fed its own outputs, so once it makes an error it is in a state it never saw in training, and errors compound. Mitigations include scheduled sampling, where you sometimes feed the model's own prediction, and sequence-level fine-tuning. In practice large models reduce the problem rather than solve it.

### Q33. Beam search or greedy decoding — what is the difference?

Greedy decoding takes the highest-probability token at each step and never reconsiders, so it is fast but can be trapped by an early mistake. Beam search keeps the $k$ highest-scoring partial sequences at every step and expands all of them, scoring by summed log-probability, usually with a length penalty because otherwise short sequences win. Typical $k$ is $4$ to $10$. Beam search helps on tasks with a single right answer, such as translation and summarisation. On open-ended generation it produces bland, repetitive text, so sampling methods are used instead.

## Word representations

### Q34. What is wrong with one-hot word vectors?

A one-hot vector has one entry per vocabulary word, one at the word's index and zero elsewhere. Three problems. It is huge and sparse — dimension equals vocabulary size, often hundreds of thousands. It carries no similarity: every pair of distinct words has dot product zero, so "cat" is exactly as far from "dog" as from "tuesday". And it cannot generalise, because a model must learn each word's behaviour separately from scratch. Dense learned embeddings fix all three: a few hundred dimensions, similar words get nearby vectors, and evidence about one word transfers to its neighbours.

### Q35. Explain word2vec skip-gram and CBOW.

Both learn embeddings from a sliding context window, in opposite directions. Skip-gram takes the centre word and predicts each context word; CBOW averages the context words and predicts the centre word. The skip-gram objective maximises

$$\sum_t \sum_{-c \le j \le c,\, j \ne 0} \log p(w_{t+j} \mid w_t)$$

where $c$ is the window radius. Skip-gram works better on rare words and small corpora, because each occurrence produces several training pairs. CBOW is faster and smoother, because averaging the context reduces noise. Both give one static vector per word type.

### Q36. What is negative sampling and why is it needed?

The softmax over the whole vocabulary is the expensive part: every update would need a sum over hundreds of thousands of words. Negative sampling replaces that with a binary problem. For each true centre-context pair, draw $k$ random words as negatives and train a logistic classifier to separate the real pair from the fake ones:

$$\log \sigma(v_c^\top v_w) + \sum_{i=1}^{k} \log \sigma(-v_{n_i}^\top v_w)$$

Typical $k$ is $5$ to $20$, larger for small datasets. Negatives are drawn from the unigram distribution raised to the power $0.75$, which upsamples rare words.

**Walk the derivation.** Three steps, and I say them in this order.

1. **The objective.** Skip-gram maximises $\log p(w_{\text{context}} \mid w_{\text{centre}})$, and the obvious model for that is a softmax over the whole vocabulary, $e^{v_c^\top v_w} / \sum_j e^{v_j^\top v_w}$.
2. **The cost.** That denominator sums over every word in the vocabulary, so one gradient step touches every output vector. At a vocabulary of two hundred thousand, a single training pair costs two hundred thousand dot products.
3. **The replacement.** So I stop modelling a distribution and model a decision instead: is this pair real, or was it drawn at random? That is one logistic term for the true pair and $k$ terms for the sampled negatives, and it needs no normaliser at all.

**With numbers.** With a vocabulary of two hundred thousand and $k = 5$, the full softmax needs $200{,}000$ output dot products per training pair while negative sampling needs $6$. That is a factor of about $33{,}000$ per update. What I give up is a calibrated conditional distribution, which nobody wanted, because the embedding table is the only thing anyone keeps.

### Q37. How does GloVe differ from word2vec?

GloVe is explicitly a factorisation of the global co-occurrence matrix rather than a prediction task over sliding windows. It builds a matrix $X$ where $X_{ij}$ counts how often word $j$ appears in the context of word $i$, then fits vectors so that the dot product matches the log count:

$$\sum_{i,j} f(X_{ij}) \left( v_i^\top \tilde{v}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

$f$ is a weighting that caps the influence of very frequent pairs. So word2vec is local and online, GloVe is global and batch. The resulting embeddings are of similar quality.

### Q38. What does fastText add?

fastText represents a word as the sum of character n-gram vectors plus a vector for the whole word. "playing" contributes subwords like "pla", "lay", "ayi" and so on. Two benefits follow. It can produce an embedding for a word it never saw in training, by summing that word's n-grams, so out-of-vocabulary is no longer fatal. And it shares evidence across morphological variants, which matters enormously in morphologically rich languages like Finnish or Turkish where a lemma has hundreds of surface forms. The cost is a larger table and slower lookup.

### Q39. Static versus contextual embeddings — state the difference.

A static embedding gives one fixed vector per word type, so "bank" has a single vector that blends the river sense and the finance sense. Word2vec, GloVe and fastText are all static. A contextual embedding is produced by running the whole sentence through an encoder, so "bank" gets a different vector in "river bank" than in "bank loan". ELMo did this with stacked bidirectional LSTMs; BERT and every transformer since do it with self-attention. Contextual embeddings resolve word sense and syntax, which is why they replaced static ones for essentially every downstream task.

### Q40. Why do we use cosine similarity for embeddings?

Cosine similarity is the dot product of two vectors divided by their norms:

$$\cos(u, v) = \frac{u^\top v}{\lVert u \rVert \, \lVert v \rVert}$$

It measures angle only, so it ignores magnitude. That matters because embedding norm often tracks word frequency or token count rather than meaning — a common word can get a long vector simply from being updated more. Normalising removes that nuisance dimension and leaves direction, which is where the semantics live. As a bonus, on unit-normalised vectors cosine similarity and Euclidean distance rank identically, so a fast inner-product index gives the same neighbours.

**With numbers.** Take $a = (1, 1)$, $b = (3, 3)$ and $c = (2, 0)$.

| Pair | Dot product | Cosine |
|---|---|---|
| $a$ with $b$ | $6$ | $1.00$ |
| $a$ with $c$ | $2$ | $0.71$ |

By raw dot product $b$ scores three times higher than $c$. But $b$ is only $a$ made longer; it points in exactly the same direction, and cosine says so. Length, not direction, is doing the ranking in the dot product, and length is the part that tracks frequency rather than meaning.

### Q41. The king minus man plus woman analogy — how much should you claim?

I would state it carefully. Embedding spaces do show linear offsets that line up with some relations, so the nearest vector to $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}}$ is often "queen". But the standard demonstrations exclude the three input words from the candidate list, and without that exclusion the answer is frequently just "king". The effect is strong for frequent, well-attested relations like country-capital and much weaker elsewhere. So it is real evidence that some structure is linear, not evidence that the space encodes a general algebra of meaning.

## Transformers, breadth level

### Q42. Explain self-attention in one breath.

Every token produces three vectors by linear projection: a query, a key and a value. Each token scores every other token by the dot product of its query with their keys, softmaxes those scores into weights, and outputs the weighted sum of values:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

$Q$ is $(n, d_k)$, $K$ is $(n, d_k)$, $V$ is $(n, d_v)$, and $n$ is the sequence length. So each token builds its new representation as a content-based lookup over the whole sequence, and all positions compute in parallel.

**Walk the derivation.** Three steps, and I say them in this order.

1. **The scores.** Each token projects to a query, and every token also projects to a key. The dot product $q_i^\top k_j$ is a similarity, so row $i$ of $QK^\top$ says how much token $i$ cares about each other token.
2. **The weights.** Those scores are unbounded real numbers, and what I need is a set of mixing weights. So I divide by $\sqrt{d_k}$ and take a softmax across the row, which gives non-negative weights that sum to one.
3. **The output.** Each token also projects to a value. Token $i$'s output is the sum of all the values under its weights. So the whole operation is a soft lookup: the query asks, the keys decide the match, the values are what comes back.

### Q43. Why divide by the square root of the head dimension?

Because of variance. If query and key entries are roughly independent with unit variance, their dot product over $d_k$ dimensions has variance $d_k$, so the raw scores grow like $\sqrt{d_k}$. Feed large-magnitude scores into a softmax and it saturates: almost all mass lands on one token and the gradient through the softmax goes to nearly zero. Dividing by $\sqrt{d_k}$ returns the scores to roughly unit variance, so the softmax stays in a responsive range and gradients keep flowing. It is a normalisation, not a tuned constant.

**Walk the derivation.** Three steps, and I say them in this order.

1. **The dot product.** A score is $\sum_{i=1}^{d_k} q_i k_i$. If the entries are independent with zero mean and unit variance, each product has variance one, and the variances of independent terms add. So the score has variance $d_k$.
2. **The consequence.** The standard deviation is therefore $\sqrt{d_k}$. At $d_k = 64$ that is $8$, so typical scores sit around plus or minus eight and the gaps between them are of that size. A softmax over gaps that wide puts nearly all its mass on one entry, and its gradient goes to nearly zero.
3. **The fix.** Dividing the scores by $\sqrt{d_k}$ divides their variance by $d_k$, returning it to one. The constant is whatever undoes the sum, so it follows from the dimension and is not a tuned choice.

**With numbers.** Two hundred thousand random query-key pairs at $d_k = 64$.

| Scores | Variance | Largest softmax weight in one row |
|---|---|---|
| Raw | $63.8$ | $0.9997$ |
| Divided by $\sqrt{64}$ | $1.00$ | $0.454$ |

Unscaled, the softmax has already collapsed onto a single token at initialisation, before any training, so almost no gradient reaches the others. Scaled, the row is still a real distribution the model can move.

### Q44. Why multiple heads?

One attention head produces one softmax distribution per token, so it can only average over one pattern at a time. Multi-head attention splits the model dimension into $h$ heads of size $d_k = d_{\text{model}}/h$, runs attention independently in each, concatenates, and applies an output projection. Different heads can then specialise — one tracks the previous token, one tracks syntactic dependencies, one attends to a delimiter. The cost is the same as single-head attention at full width, because the head dimension shrinks proportionally. Typical $h$ is $8$ to $128$ depending on model size.

**With numbers.** At $d_{\text{model}} = 512$ with eight heads, each head gets $d_k = 512/8 = 64$. The four projection matrices are still $512 \times 512$ in total, so the parameter count is $4 \times 512^2 = 1{,}048{,}576$ whether there is one head or eight. Eight heads therefore cost nothing over one; they only split the same budget into eight narrower subspaces. So the head count is a choice about how many patterns to track, not a compute decision.

### Q45. Why does a transformer need positional information at all?

Because self-attention is permutation-equivariant. The attention weights depend only on the content of the vectors, so shuffling the input tokens shuffles the outputs identically — the model literally cannot tell "dog bites man" from "man bites dog". Position must therefore be injected. The original transformer added fixed sinusoidal vectors to the embeddings; BERT and GPT-2 learned an absolute position table; modern models mostly use relative schemes like RoPE, which rotates queries and keys by an angle proportional to position so the attention score depends on the offset between tokens.

### Q46. Encoder-only, decoder-only, encoder-decoder — what suits what?

Encoder-only is bidirectional: every token sees the whole input. That suits understanding tasks — classification, tagging, retrieval embeddings. BERT is the example. Decoder-only is causal: every token sees only the past. That suits generation and, at scale, everything else, because the next-token objective works on any text. GPT is the example. Encoder-decoder has a bidirectional encoder over the source plus a causal decoder that cross-attends to it, which suits transduction where input and output are distinct sequences — translation, summarisation. T5 is the example.

### Q47. What is the causal mask and what does it do?

The causal mask makes attention look only backwards. Before the softmax, every score where the key position is later than the query position is set to negative infinity, so after softmax those weights are exactly zero. It is an upper-triangular mask over the $n \times n$ score matrix. This is what lets a decoder train on every position in parallel while still being a valid autoregressive model: position $t$ predicts token $t+1$ having seen only tokens up to $t$. Remove the mask and the model trivially copies the answer from the future.

### Q48. What were BERT's two pretraining objectives?

Masked language modelling and next-sentence prediction. For MLM, fifteen percent of tokens are selected; of those, eighty percent are replaced with a mask token, ten percent with a random token, ten percent left alone, and the model predicts the originals. The corruption mix exists because the mask token never appears at fine-tuning time. NSP feeds two segments and predicts whether the second truly followed the first, half the time sampled at random. NSP turned out to be weak — RoBERTa dropped it and got better results — so MLM is the part that mattered.

### Q49. What is GPT's training objective?

Plain autoregressive language modelling: maximise the log-likelihood of each token given all previous tokens,

$$L = \sum_t \log p(x_t \mid x_{<t})$$

with a causal mask so the model never sees the future. Its appeal is that it needs no labels, works on any text, and is a proper generative model, so a trained network can produce text directly. The contrast with BERT is that GPT sees only the left context, which is weaker per token for understanding, but the objective scales without limit and turned out to induce broad capability at size.

### Q50. What is the complexity of attention in sequence length?

Time and memory are both $O(n^2 d)$ for sequence length $n$ and model width $d$, because the score matrix has $n^2$ entries and each is a $d$-dimensional dot product. So doubling context quadruples the cost. That is the wall long-context work fights. FlashAttention removes the memory term by never materialising the full matrix — it stays $O(n^2)$ in time but becomes linear in memory. Sparse, sliding-window and linear-attention variants reduce the time term too, at some quality cost.

**With numbers.** One head in one layer, scores held in float32.

| Sequence length | Entries in the score matrix | Memory |
|---|---|---|
| $1{,}024$ | $1{,}048{,}576$ | $4.2$ MB |
| $8{,}192$ | $67{,}108{,}864$ | $268$ MB |

Eight times the context costs sixty-four times the score matrix. Multiply that by the heads and the layers and it is clear why the naive implementation runs out of memory long before it runs out of time, and why FlashAttention attacks the memory term first.

### Q51. What does the feed-forward block do?

It is a two-layer MLP applied to each position independently, with no mixing across tokens: expand from $d_{\text{model}}$ to a hidden width of typically $4 d_{\text{model}}$, apply GELU or a SiLU gate, then project back down. So the division of labour in a transformer block is clean — attention moves information between positions, and the feed-forward block does the per-token computation on it. There is good evidence it acts as a key-value memory storing factual associations, which is why editing facts in a model usually targets these layers.

### Q52. Where do a transformer's parameters actually live?

Roughly two thirds in the feed-forward blocks and one third in attention, plus the embedding table. Per layer, attention has four projection matrices of size $d \times d$, so about $4d^2$. The feed-forward block with an expansion factor of four has two matrices of $d \times 4d$, so about $8d^2$ — twice as much. Normalisation and biases are negligible. The embedding and output matrices are $V \times d$ for vocabulary $V$, which dominates in small models and becomes a small fraction in large ones.

**With numbers.** A model with $d = 4096$, thirty-two layers, a vocabulary of $128{,}000$ and a four-times feed-forward width.

| Part | Parameters |
|---|---|
| Attention, per layer | $4d^2 = 67$ million |
| Feed-forward, per layer | $8d^2 = 134$ million |
| All thirty-two layers | $6.44$ billion |
| Embedding table | $524$ million |

The feed-forward block is exactly two thirds of every layer, and the embedding is seven and a half percent of the total. Hold the vocabulary fixed and shrink the width to $512$ and that same table becomes thirty-nine percent of the model, which is why the split changes completely with scale.

## Classic NLP tasks and metrics

### Q53. What is named entity recognition, and what is BIO tagging?

NER finds spans of text that name entities and labels their type — person, organisation, location, and so on. It is framed as token classification, and BIO is the tagging scheme that turns spans into per-token labels. B marks the first token of an entity, I marks a continuation inside the same entity, O marks a token outside any entity. So "Barack Obama visited Paris" tags as B-PER, I-PER, O, B-LOC. The B tag exists to separate two adjacent entities of the same type, which a simple in-or-out scheme could not do. Variants add E for end and S for single-token spans.

### Q54. What is part-of-speech tagging?

POS tagging assigns each token its grammatical category — noun, verb, adjective, determiner and so on — usually from the Universal Dependencies tag set for cross-lingual work or the Penn Treebank set for English. It is a sequence labelling task, historically solved with hidden Markov models and CRFs, now with a fine-tuned encoder that reaches roughly ninety-seven percent accuracy on English. The hard part is ambiguity that only context resolves: "book" is a noun in "read the book" and a verb in "book the flight". It is now mostly a feature for downstream parsers rather than an end goal.

### Q55. Dependency versus constituency parsing, one line each.

Constituency parsing builds a tree of nested phrases — this span is a noun phrase, that span is a verb phrase — so the internal nodes are phrase categories and the words sit at the leaves. Dependency parsing instead draws a labelled directed arc from each word to its syntactic head, producing a tree over the words themselves with labels like subject, object and modifier. Dependency is more common now, because it is compact, works better for free word-order languages, and maps directly onto relation extraction. Constituency still suits questions about phrase structure.

### Q56. What is coreference resolution?

Coreference resolution groups the mentions in a document that refer to the same real-world entity — linking "Dr. Chen", "she" and "the researcher" into one cluster. It matters for information extraction, summarisation and question answering, because facts get distributed across mentions. The hard cases need world knowledge, not syntax: in "the trophy did not fit in the suitcase because it was too big", resolving "it" requires knowing what "big" implies about containers. That is the Winograd style of problem. Modern systems score mention pairs or spans with an encoder and cluster the results.

### Q57. What is BLEU, and what is wrong with it?

BLEU scores a machine translation by n-gram overlap with one or more references. It is the geometric mean of modified n-gram precisions for $n = 1$ to $4$, times a brevity penalty that punishes output shorter than the reference. Precisions are clipped so repeating a correct word cannot inflate the score. The weaknesses are well known: it is precision-based and recall-blind, it ignores meaning entirely so a correct paraphrase scores zero overlap, it is insensitive to word order beyond four tokens, and its absolute value is not comparable across languages or tokenisations.

**With numbers.** The reference is "the cat sat on the mat", six tokens.

| Candidate | Precisions at $n = 1$ to $4$ | Brevity penalty | BLEU |
|---|---|---|---|
| "the cat sat on" | $1, 1, 1, 1$ | $e^{-0.5} = 0.607$ | $0.607$ |
| "the cat sat on the floor" | $5/6, 4/5, 3/4, 2/3$ | $1$ | $0.760$ |

The first candidate is perfect on every n-gram it produced, and it still scores lower than a candidate with a wrong word, because it stopped two tokens early. The brevity penalty is the only place BLEU accounts for what was left out, and it is a blunt substitute for recall.

### Q58. What is ROUGE and when do you use it?

ROUGE is the recall-oriented counterpart to BLEU, used for summarisation. ROUGE-N measures the fraction of reference n-grams that appear in the generated summary; ROUGE-L uses the longest common subsequence, which rewards in-order overlap without demanding contiguity. Recall is the right emphasis for summarisation, because the question is whether the summary covered the source's key content. It inherits BLEU's core problem — a good abstractive summary that uses different words scores badly — so it favours extractive systems. Use it as a cheap regression check, not as the deciding metric.

### Q59. What is perplexity, and how does it relate to cross-entropy?

Perplexity is the exponential of the average per-token cross-entropy:

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{t=1}^{N} \log p(x_t \mid x_{<t})\right)$$

So they are the same quantity on different scales, and the interpretation is the effective number of equally likely choices the model faces at each token. A perplexity of $20$ means the model is as uncertain as if picking uniformly among twenty options. The catch is that it depends on the tokenizer and the corpus, so perplexity numbers are only comparable between models sharing both.

**With numbers.** Cross-entropy in nats, perplexity as its exponential.

| Cross-entropy | Perplexity |
|---|---|
| $1.5$ | $4.5$ |
| $3.0$ | $20.1$ |
| $4.6$ | $99.5$ |

Halving the cross-entropy does not halve the perplexity; it takes its square root. What does halve perplexity is a drop of $0.693$ nats, which is $\log 2$, and it halves it from any starting point. So the loss curve is the thing with the constant scale, and perplexity is the thing you can quote to somebody.

### Q60. Why do n-gram overlap metrics fail for open-ended generation?

Because they assume there is one correct surface form, and for open-ended tasks there are many. Ask for a story opening or a code explanation and two excellent answers can share almost no n-grams, so the metric scores a good output as badly as a bad one. The failure is in both directions: overlap metrics also reward fluent copying that says nothing new. So for generation I use pairwise human or model-based preference judgements, task-grounded checks such as unit tests for code or exact answer match for question answering, and reserve BLEU or ROUGE for constrained transduction tasks.

### Q61. What baselines would you name for text classification?

Three, in order of cost. First, TF-IDF features into logistic regression or a linear SVM — fast, strong on topical classification, and interpretable through the coefficients. Second, a fastText-style averaged-embedding classifier, which handles morphology and trains in seconds. Third, a fine-tuned encoder such as BERT or a smaller distilled version, which wins whenever word order and context matter. I always run the linear baseline first, because it takes minutes and often lands within a few points of the transformer, which tells me whether the task needs one at all.

### Q62. Write down TF-IDF.

TF-IDF weights a term by how often it appears in a document, discounted by how many documents contain it:

$$\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \log \frac{N}{1 + \text{df}(t)}$$

$\text{tf}(t,d)$ is the count of term $t$ in document $d$, often log-scaled; $N$ is the number of documents; $\text{df}(t)$ is the number of documents containing $t$. The inverse document frequency term is the point — a word in every document carries no discriminating information, so its weight goes to nearly zero, while a rare term gets a high weight. Vectors are then L2-normalised for cosine comparison.

**With numbers.** A collection of $1{,}000$ documents, scoring three terms in one of them.

| Term | Count in the document | Documents containing it | IDF | TF-IDF |
|---|---|---|---|---|
| "the" | $40$ | $1{,}000$ | $-0.001$ | $-0.04$ |
| "neural" | $3$ | $50$ | $2.98$ | $8.93$ |
| "transformer" | $1$ | $4$ | $5.30$ | $5.30$ |

"the" appears forty times and scores nothing. "transformer" appears once and scores five. The count is doing almost none of the work and the rarity is doing nearly all of it, which is why removing stop words barely moves a TF-IDF model.

## Tokenization, breadth level

### Q63. Why subword tokenization?

It sits between two bad extremes. Word tokens give a vocabulary that is enormous and still incomplete — any new name, typo or compound is unknown. Character tokens have a tiny vocabulary but make sequences several times longer, which is expensive under quadratic attention and forces the model to relearn spelling. Subwords keep a fixed vocabulary of typically thirty thousand to two hundred thousand pieces, represent frequent words as single tokens, and split rare words into parts. So nothing is ever out of vocabulary, sequences stay short, and morphology is partly shared across related words.

### Q64. Explain BPE in three sentences.

Byte pair encoding starts with a vocabulary of individual characters or bytes and repeatedly finds the most frequent adjacent pair in the training corpus, merging it into a new single token. It records each merge in an ordered list and stops once the vocabulary reaches the target size. At encoding time it splits the text into characters and replays the merge list in the same order, which makes the tokenisation deterministic. Byte-level BPE starts from raw bytes instead of characters, so any Unicode string is representable and there is no unknown token at all.

**Walk the derivation.** Three steps on a tiny corpus: "low" five times, "lower" twice, "newest" six times, "widest" three times.

1. **Split.** Every word becomes its characters plus an end-of-word marker. The starting vocabulary is the eleven distinct symbols and nothing more.
2. **Count.** I count every adjacent pair, weighted by how often its word occurs. The pair "e" then "s" appears in "newest" and "widest", so its count is $6 + 3 = 9$, which ties for the highest.
3. **Merge.** I add "es" to the vocabulary, rewrite every occurrence of the pair, and record it as merge number one in the ordered list. Then I recount and repeat: the next merge is "es" with "t", then "est" with the end marker.

**With numbers.** The corpus starts as $95$ symbol tokens across sixteen word occurrences. After the first merge it is $86$, after the second $77$, after the third $68$. Each merge removes exactly one token per occurrence of the pair, so nine every time. Three vocabulary entries bought a twenty-eight percent shorter sequence, and that trade — vocabulary size against sequence length — is the only dial BPE has.

### Q65. What is WordPiece?

WordPiece is BPE with a different merge criterion. Instead of merging the most frequent pair, it merges the pair that most increases the likelihood of the corpus under a unigram language model — roughly, it maximises the ratio of the pair's frequency to the product of its parts' frequencies. That prefers merges that are genuinely more than the sum of their pieces. It is the tokenizer BERT uses, and it marks word continuations with a double-hash prefix rather than marking word starts with a space.

### Q66. What is Unigram tokenization?

Unigram works subtractively. It starts from a large candidate vocabulary, fits a unigram language model over subword pieces with expectation-maximisation, then repeatedly removes the pieces whose deletion costs the least likelihood, until the target size is reached. Because it keeps a probability for every piece, one string has many possible segmentations with different scores, so it can return the most likely one or sample among them. That sampling is subword regularisation, which acts as data augmentation. It is the default in SentencePiece and is used by T5 and many multilingual models.

### Q67. How is out-of-vocabulary handled now?

Mostly it is not a problem any more. Byte-level BPE has every byte in its vocabulary, so any input string decomposes into known tokens and the unknown token never fires — a novel word simply becomes several pieces. Character-level fallback in SentencePiece does the same job. Where an unknown token still exists it is a signal of a mismatch, such as text in a script the tokenizer never saw. The real modern cost is not failure but inefficiency: unfamiliar input still tokenises, just into many more tokens than it should.

### Q68. Why does token count differ across languages, and what does that cost?

Because the tokenizer's merges are learned from its training corpus, which is usually dominated by English. English text gets efficient merges, roughly one token per three or four characters. A language written in another script, or one with rich morphology, gets far less merging, so the same meaning can take two to five times as many tokens. Three costs follow: inference and training bill by the token, the effective context window shrinks for those languages, and quality drops because meaning is spread over more, less meaningful pieces. It is a real fairness issue.

**With numbers.** The same sentence, "the weather is very nice today", counted with the cl100k tokenizer.

| Language | Tokens |
|---|---|
| English | $7$ |
| Hindi | $26$ |
| Burmese | $57$ |

Hindi costs $3.7$ times more and Burmese $8.1$ times more, for identical meaning. So a Burmese user pays roughly eight times per sentence and gets about one eighth of the usable context window. Nothing in the model causes that; it comes entirely from which corpus the merges were learned on.

### Q69. What are the special tokens and what is each for?

| Token | Purpose |
|---|---|
| `[CLS]` / `<s>` | Sequence start; its final hidden state is used as the pooled sentence representation for classification |
| `[SEP]` / `</s>` | Marks a boundary between segments and the end of the input |
| `[MASK]` | The corruption target in masked language modelling; only used during BERT-style pretraining |
| `[PAD]` | Fills short sequences to a common length in a batch; must be excluded by the attention mask |
| `[UNK]` | Stands in for anything the vocabulary cannot represent; effectively unused with byte-level BPE |
| `[BOS]` / `[EOS]` | Begin and end of a generated sequence; sampling stops when the model emits EOS |

The one to get right is `[PAD]`: padding that is not masked out silently corrupts attention and every pooled representation in the batch.

## "Why not just do the simple thing?"

The interviewer names an obvious simpler design and asks why the field does not use it. The answer always has the same shape: name what the simple thing breaks, then name what the real design buys.

### Q70. Why sine and cosine for positional encoding? Why not just use the integers 1, 2, 3, 4?

*This is the long one on the page. The first two sentences are a complete answer — stop there if he looks satisfied.*

Because a raw integer is unbounded and it is only one number, and the encoding needs to be neither. Sinusoids stay inside $[-1, 1]$ at every position, and they give a whole vector of position information instead of a single scalar. Integers fail on scale first: the model was trained on small normalised activations, so position 5000 injects a value far larger than any embedding component and swamps the signal. Dividing by the maximum length fixes the scale but ties the encoding to that maximum, so you cannot extrapolate past it, and adjacent positions become nearly identical at long lengths. The sinusoidal scheme gives many frequencies instead:

$$PE(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad PE(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i/d}}\right)$$

Here $p$ is the position, $d$ the model width and $i$ indexes the dimension pair. Each pair of dimensions is a clock hand, and the wavelength grows geometrically across dimensions from about $2\pi$ to about $10000 \cdot 2\pi$ — coarse position from the slow components, fine position from the fast ones. The property that matters most is relative offset. Because $\sin(a+b)$ and $\cos(a+b)$ expand into linear combinations of $\sin a$, $\cos a$, $\sin b$ and $\cos b$, the encoding at position $p+k$ is a fixed linear transform of the encoding at $p$, so a linear layer can learn to attend three tokens back once and have it work everywhere. Learned absolute embeddings do about as well inside the trained range but cannot extrapolate at all, and modern models mostly use RoPE, which applies the same rotation to the query and key vectors instead of adding to the input.

**With numbers.** Eight dimensions, so four frequency pairs. Positions $5$ and $6$ differ in the fast components and hardly at all in the slow ones.

| Pair of positions | Cosine between their encodings |
|---|---|
| $5$ and $6$ | $0.88$ |
| $5$ and $100$ | $0.33$ |
| $5$ and $5000$ | $0.31$ |

Every entry stays inside $[-1, 1]$, and every encoding has the same length, $2$ at this width, so position $5000$ injects no more magnitude than position $5$. That is precisely where the integer scheme fails: it would inject a value of $5000$ beside embedding components of about one.

### Q71. Why not one-hot encode words instead of using embeddings?

Because a one-hot vector is huge, expresses no similarity, and shares nothing between words. Its dimension equals the vocabulary size, often hundreds of thousands, so the input is almost entirely zeros and the first weight matrix carries a private column block for every word. Worse, every pair of distinct words is exactly equidistant — the dot product of two different one-hot vectors is zero — so "cat" is as far from "dog" as it is from "tuesday", and no notion of similarity is expressible at all. And because each word owns its own slot, the model must learn every word's behaviour from that word's own occurrences; evidence about "cat" never transfers to "kitten". A dense embedding of a few hundred dimensions fixes all three at once.

**With numbers.** A vocabulary of $50{,}000$ and a batch of $512$ tokens. As one-hot vectors that batch is $25{,}600{,}000$ float32 numbers, which is $102$ MB, and all but $512$ of them are zero. As $300$-dimensional embeddings it is $153{,}600$ numbers, which is $0.6$ MB. That is a factor of $167$ in memory, and the dense version is also the one that can express similarity at all. The sparse version costs more and says less.

### Q72. Why not just use a bigger n-gram model instead of a neural language model?

Because the counts get sparse exponentially fast in $n$. There are $V^{n-1}$ possible contexts for vocabulary $V$, so at $n = 5$ with fifty thousand words almost every context you meet at test time was never seen in training and its count is zero. Smoothing and backoff patch that, but they patch it by falling back to a shorter context, which is exactly the information you wanted. Storage explodes for the same reason, since the table grows with the number of observed n-grams. The deeper problem is that there is no generalisation at all: an n-gram model has no notion of similarity between contexts, so seeing "the cat sat on the" teaches it nothing about "the dog sat on the". A neural model shares parameters through embeddings, so it does.

**With numbers.** With a vocabulary of $50{,}000$, a trigram model has $50{,}000^2 = 2.5$ billion possible contexts, and a 5-gram model has $50{,}000^4$, which is $6.25 \times 10^{18}$. A ten-billion-token corpus can observe at most ten billion of them, so it covers under two parts in a billion of the context space. Almost every 5-gram context you meet at test time is therefore unseen, and the model has nothing to say about it except by backing off to a shorter one.

### Q73. Why not feed the whole document to an RNN instead of using attention?

Three reasons, and the third is the one that decided it. The hidden state is fixed width, so the entire prefix must be squeezed into a few hundred numbers — an information bottleneck that gets worse the longer the document is, because early content is overwritten. Gradients decay over long paths, since the signal from step $T$ back to step $1$ is a product of $T$ Jacobians, so dependencies more than a few dozen steps apart are never learned. And the computation is strictly sequential: step $t$ needs step $t-1$, so training cannot be parallelised over the sequence, which wastes the accelerator. Attention gives every position direct access to every other in a single step, with a path length of one and full parallelism over the sequence. That is what made scale possible.

### Q74. Why not use mean squared error for classification?

Because it is the wrong loss twice over. Composed with a sigmoid it is non-convex in the parameters, so the surface has flat regions that cross-entropy does not have. The worse problem is the gradient. Squared error through a sigmoid carries a factor of $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, which goes to zero when the unit saturates, so a prediction of $0.99$ against a label of $0$ produces almost no gradient — and that is exactly the case that needs the largest update. Cross-entropy has no such factor: its gradient through the softmax is simply $p - y$, the predicted probability minus the target. Confidently wrong therefore gives a gradient near one, which is the behaviour you want from a loss.

**With numbers.** Take a sigmoid output of $0.99$ against a label of $0$ — confidently wrong, the worst case there is. Squared error's gradient through the sigmoid carries the factor $\sigma'(z) = 0.99 \times 0.01 = 0.0099$, so what reaches the logit is about one hundredth of the error. Cross-entropy's gradient is $p - y = 0.99$. The example that needs the largest correction gets a full one from cross-entropy and almost none from squared error, and that ratio of a hundred is the whole argument.

### Q75. Why not just use accuracy?

Because it hides three things. It hides class imbalance: if one percent of transactions are fraud, a model that always says "not fraud" scores ninety-nine percent and catches nothing. It ignores the different costs of the two error types — missing a tumour and flagging a healthy patient are not the same mistake, and a single number that weights them equally cannot say so. And it is a threshold metric, so it throws away the model's confidence: a prediction at $0.51$ and one at $0.99$ count identically, so a model that ranks correctly but is badly calibrated can look worse than one that ranks poorly. So I report precision and recall at the operating point I care about, plus a ranking metric such as ROC-AUC or PR-AUC.

**With numbers.** Ten thousand transactions, one percent of them fraud.

| Model | Fraud caught | Accuracy |
|---|---|---|
| Always says "not fraud" | $0$ of $100$ | $99.0$ percent |
| Catches sixty percent at a five percent false alarm rate | $60$ of $100$ | $94.7$ percent |

The useful model scores four points worse than the model that does nothing at all. Accuracy is dominated by the majority class, so on an imbalanced problem it does not merely lose information — it ranks the models backwards.

### Q76. Why not remove the softmax and use the raw scores?

Because raw scores are not a distribution. They can be negative, they do not sum to one, and nothing fixes their scale, so they do not compose with a log-likelihood objective — cross-entropy is maximum likelihood only if the outputs form a proper categorical distribution. The exponential matters as much as the normalisation: it makes every value positive and makes the result depend only on differences between scores, which is what lets temperature scaling and logit arithmetic behave sensibly. And the standard implementation subtracts the largest logit before exponentiating, so nothing overflows, which is a numerical guarantee raw scores do not give you. For ranking alone, raw scores are fine. For training and for calibrated probabilities they are not.

### Q77. Why not normalise with batch statistics in a transformer instead of layer norm?

Because batch statistics depend on things that have nothing to do with the token being normalised. They vary with the batch size, and they vary with how much padding sits in the batch, so the same sentence is normalised differently depending on what happened to land beside it. They also differ between training and inference, since inference uses running averages rather than a live batch, and that mismatch is a standard source of silent bugs. And they couple the examples in a batch, which breaks for variable-length sequences and breaks completely at batch size one — the normal case when generating a single response. Layer norm uses one token's own features, so training and inference compute exactly the same function and no example depends on any other.

### Q78. Why not make the feed-forward block the same width as the model instead of four times wider?

Because the feed-forward block is where most of the per-token capacity lives, and narrowing it throws that capacity away. Attention moves information between positions but computes very little on it; the block is the part that transforms each token's representation, and there is good evidence it acts as a key-value memory holding factual associations. A wider inner dimension gives the non-linearity room to work, because more hidden units means more distinct features can be detected before the projection back down. Set the inner width to $d_{\text{model}}$ and you remove roughly two thirds of the layer's parameters. I would say plainly that the four-times ratio is an empirical default from the original paper, not a derived constant; gated variants shrink it to about two thirds of that to keep the parameter count equal.

### Q79. Why not just train on more data instead of regularising?

More data is the better fix when you can get it, and I would say that first. Representative new data reduces variance without asserting anything about the solution, so it beats every regulariser on its own terms. Regularisation is what you do when you cannot get more: labels are expensive, the domain is narrow, or the events you care about are rare by nature. But it is not only a substitute. A regulariser encodes a genuine prior belief about the solution — L1 says most features should be irrelevant, L2 says no single weight should dominate, weight sharing in a convolution says the same feature matters anywhere in the image. Data alone does not give you that, and where the prior is true it buys accuracy that data would take a long time to reach.

## The ten they ask most

1. [What activation functions do you know, and what are the benefits and weaknesses of each?](#q4-what-activation-functions-do-you-know-and-what-are-the-benefits-and-weaknesses-of-each)
2. [Explain self-attention in one breath.](#q42-explain-self-attention-in-one-breath)
3. [Why divide by the square root of the head dimension?](#q43-why-divide-by-the-square-root-of-the-head-dimension)
4. [State backpropagation in one breath.](#q2-state-backpropagation-in-one-breath)
5. [What are vanishing and exploding gradients, and how do we fix them now?](#q6-what-are-vanishing-and-exploding-gradients-and-how-do-we-fix-them-now)
6. [What is layer norm, and why do sequence models use it?](#q15-what-is-layer-norm-and-why-do-sequence-models-use-it)
7. [What is cross-entropy, and why does it pair with softmax?](#q10-what-is-cross-entropy-and-why-does-it-pair-with-softmax)
8. [Why sine and cosine for positional encoding? Why not just use the integers 1, 2, 3, 4?](#q70-why-sine-and-cosine-for-positional-encoding-why-not-just-use-the-integers-1-2-3-4)
9. [What is perplexity, and how does it relate to cross-entropy?](#q59-what-is-perplexity-and-how-does-it-relate-to-cross-entropy)
10. [Explain BPE in three sentences.](#q64-explain-bpe-in-three-sentences)
