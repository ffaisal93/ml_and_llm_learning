# LLM training and alignment

This topic is tested as a pipeline question, not a trivia question. The interviewer wants to hear pretraining, supervised fine-tuning and preference optimisation as three stages with different data, different objectives and different effects. The common failure is to describe RLHF as "training the model to be good" without naming the four models in PPO, without the KL term, and without saying what the reward model was trained on. Keep the loss functions in your hands.

## The equations

**Next-token cross-entropy**

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

The model predicts token $x_t$ from all earlier tokens $x_{<t}$; the loss is the mean negative log-probability of the true next token over $T$ positions, so minimising it maximises the likelihood of the corpus.

**Perplexity**

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})\right) = \exp(\mathcal{L})$$

Perplexity is the exponential of the mean negative log-likelihood, so it reads as an effective branching factor: perplexity 20 means the model is as uncertain as if it picked uniformly among 20 tokens.

**Compute cost of a training run**

$$C \approx 6ND$$

$C$ is FLOPs, $N$ is parameter count and $D$ is training tokens; the factor 6 is roughly two FLOPs per parameter for the forward pass and four for the backward pass.

**Chinchilla compute-optimal relation**

$$N_{\text{opt}} \propto C^{1/2}, \quad D_{\text{opt}} \propto C^{1/2}, \quad \frac{D_{\text{opt}}}{N_{\text{opt}}} \approx 20$$

For a fixed compute budget, parameters and tokens should grow at the same rate, which puts the optimum near 20 training tokens per parameter, so a 70B model wants about 1.4T tokens.

**Reward model, Bradley-Terry pairwise likelihood**

$$P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big), \quad \mathcal{L}_{RM} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$$

The reward model $r_\phi$ maps a prompt and response to a scalar; only the difference between the chosen response $y_w$ and the rejected response $y_l$ is supervised, so the scale and offset of the reward are not identified.

**PPO clipped surrogate objective**

$$\mathcal{L}^{CLIP} = \mathbb{E}_t\left[\min\big(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\big)\right], \quad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$$

$\rho_t$ is the importance ratio between the new and old policy; clipping it to $[1-\epsilon, 1+\epsilon]$ with $\epsilon$ near 0.2 removes the incentive to move any single token probability far in one update.

**KL penalty against the reference policy**

$$R_t = r_\phi(x, y) - \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\ \|\ \pi_{\text{ref}}(\cdot\mid s_t)\big)$$

The reward actually optimised is the learned reward minus $\beta$ times the divergence from the frozen SFT model $\pi_{\text{ref}}$, which is a leash that keeps the policy inside the region where the reward model was trained.

**Generalised advantage estimation**

$$\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t), \quad A_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

$\delta_t$ is the one-step temporal-difference error from the value network $V$, and $A_t$ sums those errors with decay $\gamma\lambda$ to trade bias against variance in the estimate of how much better an action was than average.

**GRPO group-relative advantage**

$$A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$

For one prompt you sample a group of $G$ completions, score them all, and use the within-group z-score as the advantage for every token of completion $i$, which deletes the value network entirely because the group mean is the baseline.

**DPO loss**

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)$$

DPO uses the same Bradley-Terry form as the reward model, but substitutes the implicit reward $\beta\log(\pi_\theta/\pi_{\text{ref}})$, so one supervised loss on preference pairs replaces the reward model and the RL loop.

**LoRA low-rank update**

$$W' = W_0 + \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d_{out}\times r},\ A \in \mathbb{R}^{r\times d_{in}}, \quad \text{params} = r(d_{in} + d_{out})$$

The frozen weight $W_0$ gets an additive rank-$r$ correction; for a 4096 by 4096 matrix at $r=8$ that is $8 \times 8192 = 65{,}536$ trainable parameters against 16.8M frozen, about 0.39 percent.

## Code from memory

Causal language-modelling loss with the label shift done by hand. Checked against `torch.nn.functional.cross_entropy`.

```python
import torch, torch.nn.functional as F

def causal_lm_loss(logits, labels, ignore_index=-100):
    # position t predicts token t+1, so drop the last logit and the first label
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, Tm1, V = shift_logits.shape
    flat_logits = shift_logits.view(B * Tm1, V)
    flat_labels = shift_labels.view(B * Tm1)
    # mean negative log-likelihood over non-ignored positions
    logZ = torch.logsumexp(flat_logits, dim=-1)
    mask = flat_labels != ignore_index
    safe = flat_labels.clamp(min=0)
    picked = flat_logits.gather(1, safe.unsqueeze(1)).squeeze(1)
    nll = (logZ - picked)[mask]
    return nll.mean()

torch.manual_seed(0)
B, T, V = 2, 6, 11
logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))
labels[0, 3] = -100                      # a masked prompt position
mine = causal_lm_loss(logits, labels)
ref = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                      ignore_index=-100)
print("mine", round(mine.item(), 6), "torch", round(ref.item(), 6))
print("perplexity", round(torch.exp(mine).item(), 4))
```

Output: `mine 2.976129 torch 2.976129` and `perplexity 19.6118`. The two agree to six decimals.

A LoRA linear layer. The zero initialisation of $B$ is the point: the adapter is an exact no-op at step zero.

```python
import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r=8, alpha=16):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=False)
        self.base.weight.requires_grad = False        # frozen pretrained W
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)   # down-project
        self.B = nn.Parameter(torch.zeros(d_out, r))         # up-project, zero init
        self.scale = alpha / r

    def forward(self, x):
        # h = W x + (alpha/r) * B A x ; the second term starts at exactly zero
        return self.base(x) + self.scale * (x @ self.A.t()) @ self.B.t()

torch.manual_seed(0)
layer = LoRALinear(512, 512, r=8, alpha=16)
x = torch.randn(4, 512)
print("delta at init:", (layer(x) - layer.base(x)).abs().max().item())
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
print("trainable", trainable, "frozen", layer.base.weight.numel(),
      "ratio", round(trainable / layer.base.weight.numel(), 4))
```

Output: `delta at init: 0.0` and `trainable 8192 frozen 262144 ratio 0.0312`, which matches $r(d_{in}+d_{out}) = 8 \times 1024$.

Greedy decoding and temperature with top-p (nucleus) sampling, written as an explicit loop.

```python
import torch

V = 12
torch.manual_seed(0)
W = torch.randn(V, V)                     # stand-in "model": next logits from last token
def next_logits(tok):
    return W[tok]

def sample_one(logits, temperature=0.0, top_p=1.0):
    if temperature == 0.0:                # greedy = argmax, no distribution needed
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    order = torch.argsort(probs, descending=True)
    sorted_p = probs[order]
    kept, total = [], 0.0
    for i in range(len(order)):           # keep the smallest prefix with mass >= top_p
        kept.append(int(order[i]))
        total += float(sorted_p[i])
        if total >= top_p:
            break
    sub = probs[kept] / probs[kept].sum()
    return kept[int(torch.multinomial(sub, 1))]

def generate(start, n, temperature=0.0, top_p=1.0):
    out, tok = [start], start
    for _ in range(n):
        tok = sample_one(next_logits(tok), temperature, top_p)
        out.append(tok)
    return out

print("greedy      ", generate(3, 8))
print("greedy again", generate(3, 8))               # identical: argmax is deterministic
print("T=1.0 p=0.9 ", generate(3, 8, 1.0, 0.9))
```

Output: the two greedy runs both give `[3, 9, 8, 0, 4, 11, 2, 5, 9]`; the sampled run differs.

## Questions

### Q1. Walk me through the full pipeline from a raw pretrained model to a deployed assistant. What does each stage change?

Three stages. Pretraining runs next-token cross-entropy on trillions of tokens of unlabelled text. It buys knowledge, grammar and in-context learning ability, but the output distribution is "plausible continuation of the internet", not "answer to the user". Supervised fine-tuning runs the same cross-entropy loss on tens of thousands of curated prompt-response pairs, with the loss masked over the prompt tokens. It changes format and behaviour: the model now answers instead of continuing. It adds very little new knowledge. Preference optimisation, by PPO, GRPO or DPO, takes pairwise human judgements of which of two responses is better and moves the policy toward the preferred style. It changes ranking among responses the SFT model could already produce, so it tunes helpfulness, refusal behaviour and tone. Compute drops by orders of magnitude at each step: pretraining is millions of GPU-hours, SFT is hundreds, preference optimisation is in between.

> **Say it.** Pretraining is next-token prediction on raw text. It gives you knowledge and fluency but the model just continues text, it does not answer. SFT is the same cross-entropy loss on curated prompt-response pairs with the prompt masked out. That changes format and behaviour, not knowledge. Then preference optimisation, PPO or GRPO or DPO, uses pairwise human comparisons to reorder responses the model could already produce. It tunes helpfulness and refusal, it does not teach facts. Each stage uses far less compute than the one before.

### Q2. Why does RLHF need a KL penalty against the reference model? What happens without it?

The reward model is a learned approximation trained on a finite set of on-distribution comparisons. It is only trustworthy near the responses it saw. The optimised reward is $r_\phi(x,y) - \beta\,\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$, where $\pi_{\text{ref}}$ is the frozen SFT model. Without that term the policy is free to walk anywhere in output space, and the fastest path to high reward is off-distribution text the reward model scores highly for the wrong reason. In practice you see mode collapse: the model emits the same flattering opening on every prompt, or degenerate repeated phrases, and the reward number climbs while human ratings fall. Entropy also collapses, so diversity dies. $\beta$ is the leash length. Too small and you get reward hacking; too large and the policy barely moves from SFT. It is the main knob you tune.

> **Say it.** The reward model is only accurate near the data it was trained on. If you optimise it without a constraint, the policy drifts off that distribution and finds text that scores highly for the wrong reason. So the actual objective is reward minus beta times KL to the frozen SFT model. Without it you get mode collapse: identical openings, repeated phrases, entropy going to zero, reward climbing while human ratings drop. Beta is the leash length. Too small, reward hacking. Too large, the policy never moves off SFT.

### Q3. Name the four models in a PPO RLHF setup and what each one does.

The policy is the model being trained; it generates completions and its weights are updated. The reference model is a frozen copy of the SFT model, used only to compute the KL penalty, so it stays in memory but never trains. The reward model is a separate network, usually initialised from the SFT model with a scalar head, which scores a complete prompt-response pair; it is frozen during PPO. The value model, or critic, predicts the expected return from each token position; it is trained alongside the policy and feeds the advantage estimate $A_t$ through GAE. So two models train and two are frozen. The memory cost is the honest answer to "why is PPO hard": you hold roughly four copies of a large model, plus optimiser state for the two that train, which is why people moved to DPO and GRPO.

> **Say it.** Four models. The policy, which trains. The reference model, a frozen SFT copy used only for the KL penalty. The reward model, frozen, scoring complete responses with a scalar head. And the value model or critic, which trains alongside the policy and predicts expected return so you can compute advantages. Two train, two are frozen. That means about four copies of the model in memory plus optimiser state for two of them, and that memory cost is exactly why DPO and GRPO became popular.

### Q4. How does GRPO remove the value network, and what does "group" mean?

Group means $G$ completions sampled from the current policy for the same prompt, typically 8 to 64. You score all $G$ with the reward model and normalise the rewards within that group: $A_i = (r_i - \text{mean}(r))/\text{std}(r)$. That z-score becomes the advantage for every token of completion $i$. The value network exists in PPO only to supply a baseline that reduces variance, and the group mean is already an unbiased baseline for that prompt, so the critic is redundant and gets deleted. You drop one trainable model and its optimiser state, which is roughly a third of the memory. The tradeoff is that you now pay $G$ generations per prompt instead of one, so you shift cost from memory to sampling. GRPO works best where the reward is cheap and verifiable, for example a maths answer checker, because then the group signal is clean.

> **Say it.** A group is G completions sampled for the same prompt, maybe 8 to 64. You score them all and take the z-score within that group as the advantage for every token of each completion. The critic in PPO exists only to provide a variance-reducing baseline, and the group mean is already that baseline for this prompt, so you delete the value network and its optimiser state. You pay for it in sampling: G generations per prompt instead of one. It shines with cheap verifiable rewards, like checking a maths answer.

### Q5. DPO versus PPO. What is the actual tradeoff?

DPO rewrites the RLHF optimum in closed form. The optimal KL-constrained policy satisfies $r(x,y) = \beta \log(\pi^*/\pi_{\text{ref}}) + \text{const}$, so you substitute that implicit reward into the Bradley-Terry likelihood and get a plain supervised loss on preference pairs. No reward model, no rollouts, no critic: two models in memory instead of four, and a training loop that looks like SFT. The cost is that DPO learns only from the fixed offline pairs it was given. PPO generates fresh on-policy samples and scores them, so it can explore responses no annotator wrote and it can use a verifiable reward directly. DPO is also more sensitive to distribution shift between the preference data and the current policy, and tends to push down the probability of both responses in a pair. Practical rule: DPO first because it is cheap and stable, online methods when you have a reliable reward signal and the headroom.

> **Say it.** DPO uses the closed-form solution of the KL-constrained RLHF problem to express the reward as beta times the log ratio of policy to reference. Substitute that into Bradley-Terry and you get a supervised loss on preference pairs. Two models, no rollouts, no critic. The cost is that it is purely offline: it only learns from pairs someone already wrote, it cannot explore, and it degrades when the preference data drifts from the current policy. PPO or GRPO is on-policy and can use a verifiable reward. Start with DPO, go online when you have a real reward signal.

### Q6. What is reward hacking? Give me a concrete example.

Reward hacking is the policy maximising the measured proxy reward while the true objective gets worse, because the proxy and the objective agree only on the training distribution. Concrete case: annotators mildly prefer longer, more detailed answers, so the reward model learns length as a partial proxy for quality. Optimise hard against it and the policy pads every answer with restatements and caveats. Reward goes up, human satisfaction goes down. This is why length-controlled evaluation exists. Another concrete case: a code model rewarded on unit tests passing learns to detect the test harness and special-case its inputs, rather than implement the function. Detection is by divergence between the reward curve and a held-out human or automatic evaluation, and by inspecting samples directly. Mitigations are the KL penalty, an ensemble of reward models, periodically retraining the reward model on fresh on-policy comparisons, and explicitly controlling for the known proxy such as length.

> **Say it.** Reward hacking is the policy maximising the proxy while the real goal gets worse. The classic case is length: annotators slightly prefer longer answers, the reward model picks that up, and the policy learns to pad every response with caveats. Reward climbs, humans rate it lower. Another is a code model rewarded on unit tests that learns to special-case the test inputs instead of writing the function. You detect it when the reward curve and held-out evaluation diverge. You fix it with the KL penalty, reward-model ensembles, fresh on-policy comparison data, and length control.

### Q7. What is a reward model trained on, and what is the Bradley-Terry formulation?

You collect prompts, sample two or more responses per prompt from the current model, and have humans mark which one is better. That is the entire supervision: pairwise ordinal comparisons, not absolute scores, because people are consistent about "which is better" and inconsistent about "rate this 1 to 10". The reward model is usually the SFT model with the language head replaced by a scalar head. Bradley-Terry says the probability the annotator prefers $y_w$ over $y_l$ is $\sigma(r(x,y_w) - r(x,y_l))$, so the loss is $-\log\sigma(r(x,y_w) - r(x,y_l))$. Only the difference is supervised, so the reward is identified up to an additive constant per prompt; this is why absolute reward values are meaningless and why you always normalise. Reward-model accuracy on held-out pairs sits well below perfect, typically in the sixties to seventies percent, which is precisely why you need the KL leash.

> **Say it.** It is trained on pairwise human comparisons: same prompt, two sampled responses, a human says which is better. Not absolute scores, because people are inconsistent about numbers and consistent about comparisons. Architecture is the SFT model with a scalar head. Bradley-Terry gives the preference probability as sigmoid of the reward difference, so the loss is minus log sigmoid of that difference. Only the difference is supervised, so reward is identified only up to an offset, and absolute values mean nothing. Held-out pair accuracy is well short of perfect, which is why you need the KL penalty.

### Q8. Explain LoRA and QLoRA. Which matrices do you adapt and why is the rank small?

LoRA freezes $W_0$ and learns $W_0 + (\alpha/r)BA$ with $B$ zero-initialised, so training starts exactly at the pretrained function. Trainable parameters are $r(d_{in}+d_{out})$ per adapted matrix. You adapt the attention projections, and the strong practical result is that adapting all of the query, key, value and output projections plus the MLP matrices beats concentrating a larger rank on query and value alone. Rank is small, typically 8 to 64, because the update needed to specialise an already-competent model is empirically low-rank; you are steering, not rebuilding. QLoRA adds three things: the frozen base weights are quantised to 4-bit NormalFloat, optimiser state is paged to CPU when memory spikes, and quantisation constants are themselves quantised. The adapters stay in bf16 and gradients flow through the dequantised weights, so you fine-tune a 65B model on a single 48GB card at close to full-precision quality.

> **Say it.** LoRA freezes the base weight and adds a rank-r product B times A, scaled by alpha over r, with B initialised to zero so it starts as a no-op. Parameters are r times the sum of the two dimensions. Adapt all the attention projections and the MLP matrices, not just query and value. Rank stays small, 8 to 64, because specialising a competent model is empirically a low-rank change. QLoRA keeps the frozen base in 4-bit NormalFloat with paged optimiser state and bf16 adapters, so a 65B model fits on one 48GB card.

### Q9. Full fine-tuning versus PEFT. When is each right?

Full fine-tuning updates every weight, so it needs memory for the weights, the gradients and two Adam moments, roughly 16 bytes per parameter in mixed precision, plus activations. It is the right choice when you are adding genuinely new capability or a new domain vocabulary, when you have a large task dataset, and when you can afford the hardware. PEFT with LoRA trains under one percent of the parameters, so optimiser state collapses and a single checkpoint is a few tens of megabytes. That matters operationally: you can serve one base model with many hot-swapped adapters instead of one full copy per customer. PEFT also resists catastrophic forgetting because the base weights are untouched and the adapter can be removed. It underperforms full fine-tuning when the target task is far from pretraining, because a low-rank correction cannot represent a large functional change. Default to LoRA; escalate to full fine-tuning only when a measured gap justifies the cost.

> **Say it.** Full fine-tuning touches every weight, so you pay about 16 bytes per parameter for weights, gradients and Adam moments. It wins when you are adding real new capability or a distant domain and you have the data and the hardware. LoRA trains under one percent of parameters, gives you tiny checkpoints, and lets you serve one base model with many swappable adapters. It also forgets less because the base is frozen and removable. It underperforms when the task is far from pretraining, because a low-rank delta cannot express a big change. Default LoRA, escalate on measured evidence.

### Q10. What is catastrophic forgetting in this context, and how do you mitigate it?

Fine-tuning on a narrow distribution moves weights to reduce loss on that distribution, and nothing in the objective preserves behaviour elsewhere, so general capability degrades. You see it as a model fine-tuned on customer-support data losing arithmetic, coding or instruction-following on unrelated prompts, and often losing its safety refusals. The first mitigation is to keep the base weights: LoRA or adapters, since the delta is bounded and removable. The second is data mixing: blend 5 to 20 percent of general instruction data, or replay of the original SFT set, into the fine-tuning mixture. The third is a lower learning rate and fewer epochs, because most forgetting comes from over-training on a small set. The fourth is a KL or L2 penalty back toward the reference model, which is the same leash idea as RLHF. And you must measure it: run a general benchmark suite before and after, not only the target task metric.

> **Say it.** Fine-tuning on a narrow distribution moves the weights to fit it, and nothing protects behaviour elsewhere, so general ability degrades. You see a support-tuned model lose arithmetic and coding, and sometimes lose its refusals. Mitigations: use LoRA so the base is frozen and the delta is removable; mix five to twenty percent general instruction data into the fine-tuning set; lower the learning rate and cut epochs, because most forgetting is over-training on a small set; add a KL penalty back to the reference. And always evaluate general benchmarks before and after, not just the target task.

### Q11. What does instruction tuning do that pretraining does not?

Pretraining optimises $p(x_t \mid x_{<t})$ over raw text. The resulting model is a distribution over plausible continuations. Given "Write a poem about the sea", a purely pretrained model might plausibly continue with more instructions, because on the internet a list of prompts is a common context. It has the ability to write the poem but no reason to. Instruction tuning is the same cross-entropy objective on prompt-response pairs with the loss masked over the prompt, so the model is only scored on producing a response. That changes three things: the model treats a prompt as a request, it learns the response format, and it learns when to stop, since the end-of-sequence token now appears where a completed answer ends. It also generalises across task types, so tuning on a few hundred task families produces zero-shot instruction following on unseen ones. It elicits and formats existing capability; it does not add knowledge.

> **Say it.** Pretraining gives you a model of plausible continuations. Ask it to write a poem and it might continue by listing more prompts, because that is a plausible internet context. It can write the poem, it just has no reason to. Instruction tuning is the same cross-entropy loss on prompt-response pairs with the prompt masked, so the model is scored only on producing an answer. It learns to treat prompts as requests, learns the response format, and learns where to emit end-of-sequence. It elicits and formats existing capability rather than adding knowledge.

### Q12. How would you build an SFT dataset? What is the quality-over-quantity finding?

Start from real user prompts, not invented ones, because the prompt distribution matters more than the response distribution. Cover the task families you actually serve, and deliberately include the hard cases: ambiguous requests, requests you should refuse, requests needing a clarifying question, and multi-turn context. Write or curate responses in one consistent house style, because inconsistent formatting across annotators teaches the model to be inconsistent. Then filter hard: deduplicate near-identical prompts, drop responses with factual errors, and drop responses that answer a different question than the one asked. The repeated experimental finding is that a few thousand carefully curated and consistent examples beat hundreds of thousands of noisy or machine-generated ones on human evaluation, because SFT teaches format and behaviour, and one badly formatted example teaches a bad format as effectively as a good one teaches a good one. Hold out a prompt set by source, not by random split, so you can measure generalisation.

> **Say it.** Start from real user prompts, since the prompt distribution matters more than the responses. Cover your actual task families and deliberately include hard cases: ambiguity, refusals, clarifying questions, multi-turn. Write responses in one consistent house style, because inconsistent annotators teach inconsistency. Then filter aggressively for factual errors, near-duplicates and off-target answers. The repeated finding is that a few thousand clean consistent examples beat hundreds of thousands of noisy ones, because SFT teaches format and behaviour and a bad example teaches just as efficiently as a good one. Hold out by source, not randomly.

### Q13. How do you detect that a model has memorised your evaluation set?

Four checks. First, n-gram overlap: search the training corpus for long substrings of every eval item, typically 13-gram or longer matches, and report contamination rate per benchmark. Second, the perturbation test: rewrite the eval items, changing names, numbers and surface phrasing while preserving the reasoning. A model that understands keeps its score; a model that memorised drops sharply. Third, the temporal test: evaluate on items created after the training data cutoff. A large gap between pre-cutoff and post-cutoff accuracy on the same benchmark format is strong evidence of contamination. Fourth, look at token-level confidence and per-token loss on eval items; abnormally low loss on the exact benchmark strings compared to paraphrases indicates memorisation. Also compare the eval score against a held-out private set you generated yourself. The practical defence is a private evaluation set that never touches a training pipeline and is regenerated periodically.

> **Say it.** Four checks. N-gram overlap between the eval items and the training corpus, thirteen-gram or longer, reported as a contamination rate. A perturbation test: change names, numbers and phrasing but keep the reasoning; understanding survives, memorisation collapses. A temporal test: score items created after the data cutoff and compare with pre-cutoff items in the same format. And per-token loss: abnormally low loss on the exact benchmark string versus a paraphrase is memorisation. The real defence is a private eval set that never enters any training pipeline, regenerated regularly.

### Q14. Given a fixed compute budget, how do you choose model size and dataset size?

Use $C \approx 6ND$ as the budget constraint and the Chinchilla result that the optimum has $N$ and $D$ scaling together, roughly 20 tokens per parameter. So for a fixed $C$, pick $N \approx \sqrt{C/120}$ and $D \approx 20N$. Earlier large models were badly under-trained: they spent the budget on parameters and starved on tokens, which is why a 70B model trained on 1.4T tokens beats a 175B model trained on 300B tokens at comparable compute. However, compute-optimal is a training-cost criterion only. Inference cost scales with $N$, not $D$, so if you will serve the model billions of times you should deliberately train a smaller model far past the Chinchilla point. Total cost of ownership, not training FLOPs, is the real objective. That is why current production models are heavily over-trained relative to Chinchilla, often at hundreds of tokens per parameter.

> **Say it.** Compute is about six N D. Chinchilla says parameters and tokens should scale together, roughly twenty tokens per parameter, so for a fixed budget you take N as the square root of C over 120 and D as twenty N. That is why a 70B model on 1.4T tokens beat a 175B model on 300B tokens. But compute-optimal only minimises training cost. Inference cost scales with parameters, not tokens, so if you serve the model heavily you train a smaller model far past Chinchilla. Total cost of ownership is the real objective.

## Done when

- You can write the causal LM loss with correct label shifting, the perplexity relation and the DPO loss on a whiteboard in three minutes without notes.
- You can name PPO's four models, say which two train, and explain the KL penalty and one concrete reward-hacking failure without prompting.
- You can state the GRPO advantage formula, define "group" as G completions for the same prompt, and say exactly which component of PPO it deletes and why.
- You can compute LoRA trainable parameters for a given matrix shape and rank in your head, and give the QLoRA additions in one sentence.
