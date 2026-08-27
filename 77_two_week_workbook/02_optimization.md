# Optimization and training dynamics

This is the section that separates people who have trained models from people who have read about training models. The interviewer asks about Adam and warmup because the follow-up questions are diagnostic: what breaks without bias correction, why pre-norm survives depth, why fp16 needs a loss scaler. The common failure is describing what an optimizer does instead of why each term exists. Every answer here should name a specific quantity and say which direction it moves.

## The equations

**Gradient descent update.**

$$\theta_{t+1} = \theta_t - \eta\,\nabla_\theta \mathcal{L}(\theta_t)$$

$\theta$ is the parameter vector, $\eta$ the learning rate, and $\nabla_\theta \mathcal{L}$ the loss gradient; each step moves against the steepest local increase, and $\eta$ sets how far you trust that local linear model.

**Momentum.**

$$v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta\,v_t$$

$v$ is a running sum of past gradients with decay $\beta$, usually 0.9; it averages out gradient noise and builds speed along directions of consistent sign, so the effective step in a stable direction is about $\frac{1}{1-\beta}$ times larger.

**RMSProp.**

$$s_t = \rho\,s_{t-1} + (1-\rho)\,g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon}\,g_t$$

$g_t$ is the gradient, $s_t$ an exponential average of its square taken elementwise, and $\epsilon$ about $10^{-8}$ stops division by zero; dividing by the root mean square gives every parameter its own step size, so rarely updated coordinates still move.

**Adam with bias correction.**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Adam is momentum plus RMSProp with defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$; the hat terms divide out the initialisation-at-zero bias, and $t$ is the step count starting at one.

**Weight decay versus L2 in adaptive optimizers.**

$$\text{L2: } g_t \leftarrow g_t + \lambda\theta_t \quad\text{(then divided by }\sqrt{\hat{v}_t}\text{)}, \qquad \text{AdamW: } \theta_{t+1} = \theta_t - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\lambda\theta_t$$

L2 puts the penalty inside the gradient, so Adam's per-parameter scaling shrinks it for high-variance coordinates; AdamW applies the decay directly to the weights, so every parameter decays at the same rate $\eta\lambda$.

**Warmup and cosine schedule.**

$$\eta_t = \eta_{\max}\frac{t}{T_w} \;\; (t \le T_w), \qquad \eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\left(1 + \cos\frac{\pi (t - T_w)}{T - T_w}\right)$$

$T_w$ is the warmup length in steps and $T$ the total; the ramp keeps early steps small while Adam's second-moment estimate is still noisy, and the cosine decays smoothly to near zero so the last steps refine rather than bounce.

**Batch norm.**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta, \qquad \mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$

Statistics are computed per feature across the $m$ examples in the batch, with $\gamma$ and $\beta$ learned; it needs running averages at inference time, so training and inference behave differently.

**Layer norm.**

$$\mu_i = \frac{1}{d}\sum_{j=1}^{d} x_{ij}, \qquad \sigma_i^2 = \frac{1}{d}\sum_{j=1}^{d}(x_{ij}-\mu_i)^2, \qquad y_{ij} = \gamma_j \frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j$$

Statistics are computed per example across its $d$ features, so there is no dependence on other examples in the batch and no train-inference mismatch.

**Gradient clipping by global norm.**

$$g \leftarrow g\,\cdot\,\min\!\left(1, \frac{c}{\|g\|_2}\right), \qquad \|g\|_2 = \sqrt{\sum_{p}\|g_p\|_2^2}$$

The norm is taken over all parameters concatenated, and $c$ is the threshold, typically 1.0; direction is preserved and only the magnitude is capped, so one bad batch cannot destroy the weights.

**Chain rule through a linear layer.**

$$y = xW + b, \qquad \frac{\partial \mathcal{L}}{\partial W} = x^{\top}\frac{\partial \mathcal{L}}{\partial y}, \qquad \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}W^{\top}, \qquad \frac{\partial \mathcal{L}}{\partial b} = \sum_{\text{batch}} \frac{\partial \mathcal{L}}{\partial y}$$

With $x$ of shape $(n, d_{\text{in}})$ and $W$ of shape $(d_{\text{in}}, d_{\text{out}})$, the weight gradient is inputs transposed against output gradients, and the input gradient is output gradients against $W$ transposed; every backward pass in a network is this identity repeated.

## Code from memory

Adam from scratch in NumPy, verified against `torch.optim.Adam` on a small quadratic.

```python
import numpy as np, torch

def adam(grad_fn, x0, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, steps=50):
    x = x0.copy()
    m = np.zeros_like(x); v = np.zeros_like(x)
    for t in range(1, steps + 1):
        g = grad_fn(x)
        m = b1 * m + (1 - b1) * g              # 1st moment
        v = b2 * v + (1 - b2) * g * g          # 2nd moment
        mh = m / (1 - b1 ** t)                 # bias correction, t starts at 1
        vh = v / (1 - b2 ** t)
        x -= lr * mh / (np.sqrt(vh) + eps)
    return x

A = np.array([3.0, 0.5])                       # loss = 0.5 * sum(A * x^2), grad = A * x
x0 = np.array([1.0, -2.0])
mine = adam(lambda x: A * x, x0.copy())

xt = torch.tensor(x0, requires_grad=True)
opt = torch.optim.Adam([xt], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
for _ in range(50):
    opt.zero_grad()
    (0.5 * (torch.tensor(A) * xt * xt).sum()).backward()
    opt.step()
print("mine ", mine, "\ntorch", xt.detach().numpy())
print("max abs diff", float(np.max(np.abs(mine - xt.detach().numpy()))))
```

Output: both give `[-0.00481822  0.07576401]` and the max absolute difference is `4.16e-17`, which is float64 rounding, so the implementations agree exactly. Note that the stiff coordinate, $A = 3$, and the flat coordinate, $A = 0.5$, end at similar magnitudes, because Adam's per-parameter scaling removes most of the curvature difference.

Layer norm forward, verified against `torch.nn.functional.layer_norm`.

```python
import numpy as np, torch

def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)             # per-example mean over features
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)            # normalise over the feature axis
    return gamma * xhat + beta                      # learned scale and shift

rng = np.random.default_rng(0)
x = rng.normal(size=(4, 8)); g = rng.normal(size=8); b = rng.normal(size=8)
mine = layer_norm(x, g, b)
ref = torch.nn.functional.layer_norm(torch.tensor(x), (8,),
        torch.tensor(g), torch.tensor(b), eps=1e-5).numpy()
print("max abs diff vs torch:", float(np.max(np.abs(mine - ref))))
```

Output: `max abs diff vs torch: 4.44e-16`, so the two match to float64 precision. The variance uses the biased $\frac{1}{d}$ denominator, which is what PyTorch uses here.

A minimal PyTorch training loop with AdamW, cosine schedule, and gradient clipping.

```python
import torch, torch.nn as nn

torch.manual_seed(0)
X = torch.randn(512, 10)
y = (X @ torch.randn(10) + 0.1 * torch.randn(512)).unsqueeze(1)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
loss_fn = nn.MSELoss()

for epoch in range(200):
    for i in range(0, len(X), 64):                                # minibatch loop
        xb, yb = X[i:i + 64], y[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # clip by global norm
        opt.step()
    sched.step()                                                  # schedule steps per epoch
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.5f}")
```

Output: loss falls from `2.6337` at epoch 0 to `0.0032` at the end, with the learning rate decaying from `0.01000` through `0.00848` and `0.00492` to `0.00141`. Order matters: `zero_grad`, forward, `backward`, clip, `step`.

## Questions

### Q1. Why does Adam need bias correction, and what happens without it?

Both moment estimates start at zero, so early on they are biased towards zero. Take the second moment: $v_t = (1-\beta_2)\sum_{i=1}^{t}\beta_2^{t-i}g_i^2$. If the gradients have roughly constant second moment, then $\mathbb{E}[v_t] \approx (1 - \beta_2^t)\,\mathbb{E}[g^2]$, so dividing by $1 - \beta_2^t$ makes it unbiased. At $t = 1$ with $\beta_2 = 0.999$, the raw $v_1$ is only $0.001\,g_1^2$, a thousand times too small. Without correction the update is $\frac{m_1}{\sqrt{v_1}} = \frac{0.1 g_1}{\sqrt{0.001 g_1^2}} \approx 3.16$, so the step is over three times the learning rate instead of about one. Worse, $\beta_1$ and $\beta_2$ decay at different speeds, so the ratio is wrong for hundreds of steps. The result is a huge, badly directed first step that can push the model into a bad region it never leaves. Bias correction is why plain Adam can start without warmup at moderate learning rates.

> **Say it.** Both moments start at zero, so early estimates are biased low. Dividing by one minus beta to the $t$ makes them unbiased. The damage is worst at step one: with $\beta_2$ at 0.999 the raw second moment is a thousand times too small, so the ratio of first to second moment gives a step about three times the learning rate instead of one. And because the two betas decay at different rates, the ratio stays wrong for hundreds of steps. Without correction you get one huge misdirected jump early, which can put you somewhere you never recover from.

### Q2. Why does AdamW decouple weight decay, and why does that matter?

In classic L2 you add $\lambda\theta$ to the gradient before the optimizer sees it. Adam then divides the whole gradient by $\sqrt{\hat{v}}$. So the actual shrinkage a parameter gets is $\frac{\eta\lambda\theta}{\sqrt{\hat{v}}+\epsilon}$, which depends on that parameter's own gradient history. Parameters with large historical gradients have large $\hat{v}$, so they get decayed the least, and parameters that barely move get decayed the most. That is backwards: the large, active weights are the ones you want to control. AdamW instead subtracts $\eta\lambda\theta$ from the weights directly, outside the adaptive term, so every parameter shrinks by the same relative amount per step. This makes the decay coefficient mean the same thing regardless of learning rate schedule and makes it tunable independently. It matters in practice because it recovered most of the generalisation gap between Adam and SGD with momentum, and it is the default for transformer training. Also exclude biases and norm parameters from decay.

> **Say it.** With L2 you add lambda times theta into the gradient, and Adam then divides by the square root of the second moment. So the actual decay each weight receives is scaled by its own gradient history, and the weights with the largest gradients get decayed the least, which is exactly backwards. AdamW subtracts eta times lambda times theta from the weights directly, outside the adaptive scaling, so every parameter decays at the same relative rate and lambda is independently tunable. That closed most of the generalisation gap with SGD and is now the default. Exclude biases and norm parameters from decay.

### Q3. Batch norm versus layer norm, and why do transformers use layer norm?

Batch norm normalises each feature across the examples in the batch. Layer norm normalises each example across its own features. Three reasons transformers use layer norm. First, sequence length varies, and batch statistics over padded, variable-length sequences are unstable and leak information across positions. Second, batch norm's statistics depend on batch size, and it degrades badly at the small per-device batch sizes used in large-scale distributed training; layer norm has no batch dependence at all. Third, batch norm keeps running averages and so behaves differently at train time and inference time, which breaks autoregressive decoding where you process one token at a time with an effective batch of one. Layer norm computes the same function in training and inference. A fourth practical reason is that batch norm requires a cross-device sync to compute correct statistics, which is a communication cost layer norm does not have. RMSNorm drops the mean subtraction and is now common.

> **Say it.** Batch norm normalises each feature across the batch; layer norm normalises each example across its features. Transformers use layer norm because sequence lengths vary, so batch statistics over padded sequences are unstable; because batch norm degrades at the small per-device batch sizes distributed training uses; and because batch norm behaves differently at train and inference time, which breaks autoregressive decoding one token at a time. Layer norm is identical in training and inference and needs no cross-device sync. RMSNorm, which drops the mean subtraction, is the current common variant.

### Q4. Pre-norm versus post-norm: why does pre-norm train more stably at depth?

Post-norm is $x_{l+1} = \text{LN}(x_l + F(x_l))$, the original transformer. Pre-norm is $x_{l+1} = x_l + F(\text{LN}(x_l))$. In pre-norm the residual stream is never normalised, so there is a clean identity path from the input to the output. The gradient flowing back to layer $l$ therefore contains an unmodified copy of the gradient from the top, and it cannot be scaled away by $L$ successive layer norms. In post-norm every residual addition passes through a layer norm, so the backward pass multiplies by $L$ Jacobians of the normalisation, and the gradient magnitude at the bottom layers shrinks or grows as depth increases. Practically: post-norm needs careful warmup and usually fails above roughly a few dozen layers without tricks; pre-norm trains at hundreds of layers with little or no warmup. The cost is that the residual stream variance grows with depth, so pre-norm models need a final layer norm before the output head, and post-norm sometimes reaches slightly better final quality when it does converge.

> **Say it.** Post-norm normalises after the residual add; pre-norm normalises the branch input and leaves the residual stream untouched. That gives pre-norm a clean identity path, so the gradient reaching the bottom layer includes an unscaled copy from the top. Post-norm multiplies the backward signal by a layer-norm Jacobian at every one of the $L$ layers, so gradients at the bottom shrink or blow up with depth. Pre-norm therefore trains at hundreds of layers with little warmup. The cost is that the residual variance grows with depth, so you need a final layer norm before the head.

### Q5. Vanishing and exploding gradients: what causes them and what are the modern fixes?

Backpropagation through $L$ layers multiplies $L$ Jacobians. If their typical singular value is below one the product shrinks geometrically and lower layers get no signal; if above one it grows geometrically and you get overflow. Sigmoid and tanh made this worse, because the sigmoid derivative peaks at 0.25, so ten layers already gives a factor near $10^{-6}$. The modern fixes, in the order I would list them: ReLU and GELU activations, whose derivative is about one in the active region; residual connections, which add an identity path so the Jacobian is $I + \frac{\partial F}{\partial x}$ and the product does not decay; normalisation layers, which keep activation scale constant across depth; careful initialisation, He for ReLU with variance $2/n_{\text{in}}$ and Xavier for tanh, so the forward variance is preserved at the start; gradient clipping by global norm, which is a hard cap against explosion; and pre-norm placement. For recurrent networks specifically, gating in LSTMs and GRUs provides the additive path.

> **Say it.** Backprop multiplies one Jacobian per layer, so if the typical singular value is under one the product vanishes geometrically and if it is over one it explodes. Sigmoid made it worse because its derivative maxes at 0.25. The fixes are ReLU or GELU with derivative near one, residual connections that make the Jacobian identity plus a small term, normalisation layers holding the scale constant, He or Xavier initialisation preserving forward variance, gradient clipping by global norm as a hard cap against explosion, and pre-norm placement. In recurrent nets the gating in LSTMs gives you the same additive path.

### Q6. Why does warmup help?

Two reasons, and they are different. The first is Adam's second moment. Early in training $\hat{v}$ is estimated from very few gradients, so its variance is high. When $\hat{v}$ happens to be small on some coordinate the update $\frac{\hat{m}}{\sqrt{\hat{v}}}$ is huge. Warmup keeps $\eta$ small over the same window in which $\hat{v}$ becomes reliable. This is exactly the argument behind rectified Adam. The second reason is curvature. At initialisation the loss surface is poorly conditioned and the largest Hessian eigenvalue can be large, so the stable learning rate $\eta < 2/\lambda_{\max}$ is small; as training progresses the sharpest directions flatten and a larger $\eta$ becomes stable. Warmup tracks that. Warmup matters most with large batches, where you use a large peak learning rate, and with post-norm, where early gradients are badly scaled. Typical settings are a few hundred to a few thousand linear steps, or one to five percent of total steps.

> **Say it.** Two reasons. Adam's second-moment estimate is built from very few gradients early on, so it is noisy; if it comes out small on some coordinate you get an enormous step. Warmup keeps the learning rate low over exactly the window where that estimate stabilises. Separately, the loss surface is sharpest at initialisation, so the largest Hessian eigenvalue caps the stable learning rate below what you want to use later, and the sharp directions flatten as training proceeds. Warmup matters most with big batches and high peak rates. Typically one to five percent of total steps, ramped linearly.

### Q7. What does a learning-rate finder tell you?

You run a short training pass while increasing the learning rate exponentially each step, from something tiny like $10^{-7}$ up to something clearly too large like $10$, and you plot loss against the log of the learning rate. The curve has three regions. On the left the loss is flat, because the steps are too small to change anything. In the middle the loss falls steeply, and this is the usable band. On the right the loss turns up and then diverges. The rule is to pick a rate roughly one order of magnitude below the minimum of the curve, or equivalently the point of steepest downward slope, not the point of lowest loss, because the loss minimum is already close to the unstable edge. What it actually measures is the largest step size at which the local linear model of the loss still holds, which is set by the curvature. Caveats: it is a short-horizon test, so it can overestimate what is safe for a long run, and you should rerun it after changing batch size, architecture, or optimizer.

> **Say it.** You ramp the learning rate exponentially over a few hundred steps and plot loss against log learning rate. You get a flat region where nothing happens, a steep descent which is the usable band, and a blow-up. You pick the point of steepest descent, or about a factor of ten below the loss minimum, because the minimum itself is already near the unstable edge. What it measures is the largest step where the local linear approximation still holds, which is set by curvature. It is short-horizon, so it can be optimistic, and you rerun it whenever batch size or architecture changes.

### Q8. How does batch size affect the gradient noise, and what is the linear-scaling heuristic?

The minibatch gradient is an average of $B$ independent per-example gradients, so its variance scales as $\frac{\sigma^2}{B}$ and the noise standard deviation as $\frac{\sigma}{\sqrt{B}}$. Doubling the batch therefore cuts gradient noise by only $\sqrt{2}$, not 2. The SGD update's noise contribution scales like $\frac{\eta^2}{B}$ per step, so to keep the same noise level per unit of data when you multiply $B$ by $k$, you multiply $\eta$ by $k$. That is the linear-scaling heuristic: scale the learning rate linearly with batch size, with a warmup because the large rate is unstable at initialisation. It holds up to a critical batch size, above which the gradient is already nearly noiseless and further increases buy almost no reduction in steps to convergence, so you are just wasting compute. For Adam the empirical rule is often closer to square-root scaling, because the update is already normalised by the gradient magnitude.

> **Say it.** The minibatch gradient averages $B$ samples, so its noise falls as one over root $B$; doubling the batch only cuts noise by about 1.4. To keep the same amount of noise per unit of data when you multiply the batch by $k$, you multiply the learning rate by $k$, with warmup because that rate is unstable at initialisation. That is linear scaling. It holds until the critical batch size, above which the gradient is already nearly exact and more data per step buys no fewer steps. For Adam, square-root scaling often fits better, because Adam already normalises by gradient magnitude.

### Q9. How do you diagnose a loss that plateaus, versus one that diverges, versus one that goes NaN?

Three different failures. A plateau at a non-trivial value usually means the learning rate is too low, the schedule has decayed too early, or the model lacks capacity; check first whether the loss equals the entropy of the label prior, because that means the model predicts the base rate and has learned nothing, which points at a broken data pipeline or a label misalignment. Verify by overfitting a single batch to near-zero loss: if that fails, the bug is in the model or the data, not the optimizer. Divergence, loss rising steadily, means the learning rate is above the stability limit $2/\lambda_{\max}$; lower it by an order of magnitude, add warmup, and clip gradients. NaN is different and is not usually a learning-rate problem alone: look for a log of zero or a division by zero, an fp16 overflow, a softmax over unclipped large logits, a zero denominator in a normalisation, or an exploding gradient. Print the global gradient norm every step; a spike immediately before the NaN identifies the batch and the layer.

> **Say it.** Three different bugs. A plateau means too low a learning rate, a schedule that decayed early, or no capacity, and I first check whether the loss equals the entropy of the label prior, which would mean the model learned nothing and the data pipeline is broken. I confirm by overfitting one batch to near zero. A steadily rising loss means the rate is past the stability limit, so I drop it tenfold, add warmup and clip. NaN is separate: log of zero, division by zero, fp16 overflow, or an exploding gradient. I log the global gradient norm every step to find the batch.

### Q10. What does gradient accumulation buy you?

It simulates a large batch on limited memory. You run $k$ forward and backward passes on microbatches of size $B$, summing the gradients, and call the optimizer once. The result is mathematically equivalent to one step on a batch of $kB$, provided you average the loss correctly: divide each microbatch loss by $k$, or sum the losses and divide once, so the gradient magnitude matches a true $kB$ batch instead of being $k$ times too large. Activation memory stays at the cost of one microbatch, because each backward pass frees its own activations. What it buys: the noise reduction and stability of a large batch on a GPU that cannot hold one. What it costs: wall-clock time, because you do $k$ times as many forward and backward passes for one optimizer step, and it does not speed anything up. Two details: with batch norm the statistics are still computed per microbatch, so the equivalence breaks, and in distributed training you should skip the gradient all-reduce on all but the last microbatch.

> **Say it.** It lets you take a large-batch step on a GPU that cannot hold a large batch. You run $k$ microbatches, sum the gradients, and step once, which is mathematically the same as one step at $k$ times the batch size as long as you divide the loss by $k$ so the magnitude is right. Activation memory stays at one microbatch. It costs wall-clock time and speeds nothing up. Two catches: batch norm still computes statistics per microbatch so the equivalence breaks there, and in distributed training you skip the all-reduce until the last microbatch.

### Q11. Explain mixed precision, and why fp16 needs a loss scaler but bf16 usually does not.

In mixed precision you keep a master copy of the weights in fp32 and run the forward and backward passes in 16-bit, which halves activation memory and uses tensor cores. The difference between the formats is exponent bits. fp16 has 5 exponent bits, so its smallest normal value is about $6\times10^{-5}$ and its maximum is about 65504. Gradients in a trained network are routinely smaller than $10^{-5}$, so they flush to zero in fp16 and those weights stop updating. The loss scaler fixes this: multiply the loss by a large constant $S$, typically $2^{15}$ or higher, before the backward pass, which shifts every gradient up by $S$ into representable range, then divide the gradients by $S$ before the optimizer step. Dynamic scaling raises $S$ when no overflow occurs and halves it when an infinity appears, skipping that step. bf16 has 8 exponent bits, the same range as fp32, so underflow is not a problem and no scaler is needed. bf16 pays with only 7 mantissa bits against fp16's 10, so it is less precise but far more robust.

> **Say it.** You keep fp32 master weights and run the forward and backward in 16-bit for memory and tensor cores. fp16 has only 5 exponent bits, so anything below about six times ten to the minus five flushes to zero, and real gradients are routinely smaller than that. The loss scaler multiplies the loss by something like two to the fifteenth before backward, shifting gradients into range, then divides them out before the step, halving the scale on overflow. bf16 has 8 exponent bits, the same range as fp32, so nothing underflows and no scaler is needed. It trades mantissa bits for that range.

### Q12. What does a "stiff" direction of the Hessian mean for the usable learning rate?

The Hessian $H$ describes local curvature. In the eigenbasis, each direction $i$ has eigenvalue $\lambda_i$, and gradient descent on a quadratic multiplies the error in that direction by $(1 - \eta\lambda_i)$ each step. Convergence requires $|1 - \eta\lambda_i| < 1$, so $\eta < 2/\lambda_{\max}$. That is a single global constraint set by the stiffest, highest-curvature direction. But the slowest direction converges at rate $(1 - \eta\lambda_{\min})$, so with $\eta$ pinned just under $2/\lambda_{\max}$ the number of steps to converge scales with the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$. This is the whole problem: one stiff direction caps the step size for every direction, and the flat directions then crawl. Momentum improves the dependence from $\kappa$ to roughly $\sqrt{\kappa}$. Adam's per-parameter scaling by $\sqrt{\hat{v}}$ is a diagonal approximation to preconditioning, which is why Adam tolerates poorly conditioned problems. Normalisation layers help by reducing $\kappa$ directly.

> **Say it.** Curvature along a Hessian eigendirection with eigenvalue lambda means gradient descent multiplies the error there by one minus eta lambda, so stability requires eta below two over lambda-max. One stiff direction therefore caps the step size for the whole model, and the flattest direction then converges at a rate set by the condition number, lambda-max over lambda-min. That is why ill-conditioned problems are slow. Momentum improves the dependence to roughly the square root of the condition number, Adam's per-parameter scaling is a diagonal preconditioner, and normalisation layers cut the condition number directly.

## Done when

- You can write Adam with bias correction from memory in NumPy in under five minutes and it matches `torch.optim.Adam` to machine precision on a quadratic.
- You can state the stability bound $\eta < 2/\lambda_{\max}$ and explain in one sentence why the condition number, not the learning rate, sets the number of steps.
- Given a loss curve that plateaus, diverges, or NaNs, you name the top two causes and the first diagnostic you would run for each, without hesitating.
- You can say why fp16 needs a loss scaler and bf16 does not, quoting the exponent-bit count of each.
