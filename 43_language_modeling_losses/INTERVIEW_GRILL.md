# Language Modeling Losses — Interview Grill

> 35 questions on LM pretraining objectives. Drill until you can answer 25+ cold.

---

## A. CLM (Causal Language Modeling)

**1. What's the CLM loss?**
$\mathcal{L} = -\sum_t \log P(\text{token}_t \mid \text{tokens}_{<t}; \theta)$. Cross-entropy on next-token prediction at every position. Equivalent to maximizing autoregressively factored joint probability of the sequence.

> **Saying it out loud.** It's just cross-entropy on next-token prediction, applied at every position at once. For each position the model produces a probability distribution over the vocabulary, and you take the negative log of the probability it assigned to the token that actually came next. Sum that over all positions and you've got the loss. The reason it looks like a big scary product in the papers is that predicting the whole sequence factorizes into predicting each token given everything before it, and taking logs turns that product into a sum. The practical consequence is that one forward pass over a 4,000-token sequence gives you 4,000 training signals, which is why this objective is so compute-efficient.

**2. Why is CLM the dominant LLM objective?**
(a) Direct generation — training task matches inference task. (b) Computationally efficient — $N$ losses per forward pass. (c) Scales as a power law in compute/parameters/data. (d) Natural in-context learning. (e) Single architecture for everything.

> **Saying it out loud.** Mainly because the training task is literally the inference task — you train the model to continue text, and at deploy time you ask it to continue text, so there's no gap to bridge. On top of that it's the most compute-efficient objective anyone has found: every single token in the batch produces a gradient, unlike masked language modeling where you throw away 85 percent of the positions. It also happens to give you in-context learning for free, because continuing a pattern is exactly what few-shot prompting asks for. And empirically it scales as a clean power law in compute, which is what lets labs predict the loss of a run before they launch it.

**3. Walk me through CLM loss computation in code.**
```
logits[:, :-1, :]   # predictions at positions 0..N-2
targets[:, 1:]       # tokens at positions 1..N-1
loss = cross_entropy(logits.view(-1, V), targets.view(-1))
```
Shift-by-one: at position t, predict token at t+1.

> **Saying it out loud.** The whole thing is a shift-by-one. The model outputs a prediction at every position, but the prediction at position $t$ is about the token at position $t+1$, so you line them up by dropping the last logit and the first target. Then it's one flat cross-entropy call over batch-times-sequence rows. The classic bug is forgetting the shift — the loss drops suspiciously fast because the model learns to copy the current token, and you don't notice until generation comes out as gibberish. If you see a training loss near zero within a few hundred steps, check your shift.

**4. How do you handle padding in CLM loss?**
Mask out padding positions: `mask = tokens != pad_id; loss = (per_token_loss * mask).sum() / mask.sum()`. Or use PyTorch's `ignore_index = -100` for padding.

> **Saying it out loud.** You mask them out, and the part people get wrong is the denominator. Padding tokens are filler with no information, so if you let them into the loss the model spends capacity learning to predict pad, and worse, your reported loss depends on how much padding happened to be in the batch. In PyTorch the clean way is setting padded targets to $-100$, which cross-entropy ignores by default. If you roll your own mask, remember to divide by the number of real tokens, not the total tensor size — dividing by the wrong denominator silently scales your gradients with batch composition and makes runs irreproducible.

**5. How do you mask the prompt during SFT?**
Set the loss for prompt tokens to 0 (or replace target with `-100`). Only response tokens contribute to the loss. The prompt is "given context"; only the response is "what we want to predict."

> **Saying it out loud.** You zero out the loss on the prompt and only train on the response. The reasoning is that the prompt is given — at inference the user supplies it, so there's no reason to spend gradient learning to generate it. Concretely you set the prompt targets to $-100$ so cross-entropy skips them. If you skip this and train on the whole thing, the model gets better at imitating user messages, which shows up as the model writing its own follow-up questions or hallucinating a user turn. On short responses with long prompts the effect is big, because most of your gradient is going into the wrong half.

**6. Why can't CLM directly use bidirectional context?**
Bidirectional attention would let the model see the answer when predicting it (trivial). The causal mask enforces that position $t$ can only see positions $\leq t$, making the next-token prediction meaningful.

> **Saying it out loud.** Because then the answer is in the input. If position $t$ could attend to position $t+1$, predicting token $t+1$ becomes a copy operation, the loss goes to zero, and the model learns nothing. The causal mask is what makes the task non-trivial — each position only sees what came before it. This is also why a subtle mask bug is so dangerous: a single off-by-one that lets a token peek at itself gives you a beautiful training curve and a model that generates garbage.

**7. How does CLM enable in-context learning?**
Trained to continue patterns from any prefix. With scale, the model develops induction heads (Olsson et al. 2022) that copy tokens after prefix matches. Few-shot prompts work because the model continues the demonstrated pattern.

> **Saying it out loud.** Because next-token prediction on a huge, messy corpus is secretly training for pattern continuation. Text on the internet is full of lists, translations, question-answer pairs — so a model that gets good at continuing arbitrary prefixes has necessarily learned 'figure out the pattern and keep going.' A few-shot prompt is just a prefix with an obvious pattern in it. Mechanistically, Olsson and colleagues traced this to induction heads, circuits that find where a token appeared before and copy what followed it. Those heads appear abruptly during training, and you can see the moment on the loss curve as a small bump.

---

## B. MLM (Masked Language Modeling)

**8. What's MLM?**
BERT's objective. Mask 15% of tokens; predict them from bidirectional context. Loss is cross-entropy on masked positions only.

> **Saying it out loud.** Masked language modeling is fill-in-the-blank. You blank out about 15 percent of the tokens and ask the model to recover them using context from both sides. That bidirectional view is the whole point — when you're building a representation of a word, knowing what comes after it helps as much as knowing what came before. It's BERT's objective and it's still the right choice for encoders. The cost is that only the masked positions produce a loss, so you're paying for a full forward pass and harvesting a signal from 15 percent of it.

**9. Walk me through BERT's masking strategy.**
For 15% of selected tokens: 80% replaced with `[MASK]`, 10% replaced with random token, 10% kept unchanged. Mitigates train-test mismatch (model can't assume `[MASK]` always means "predict me").

> **Saying it out loud.** They pick 15 percent of tokens, and then within those they do a three-way split: 80 percent get replaced with the MASK token, 10 percent get replaced with a random word, and 10 percent are left exactly as they are. The reason for the last two is that MASK never appears at fine-tuning time, so if masked positions were the only ones scored, the model would learn to build good representations only where it sees MASK. The random-token and unchanged cases force it to keep a useful representation at every position, because it can't tell which ones it will be graded on. It's a hack, but it measurably helps — and dropping it costs about a point on downstream GLUE tasks.

**10. Why is MLM less sample-efficient than CLM?**
Only 15% of tokens contribute to the loss. The other 85% is "wasted" forward-pass compute. CLM has $N$ losses per forward pass.

> **Saying it out loud.** It's a signal-per-FLOP argument. Both objectives run the same expensive forward pass over the sequence, but CLM gets a loss term at every one of the $N$ positions while MLM gets one at roughly $0.15N$. So you're paying full price and collecting about a seventh of the training signal. You could mask more, but push much past 15 percent and there isn't enough context left to make the prediction learnable, so quality falls off. That tension — more signal versus enough context — is exactly what ELECTRA was invented to escape.

**11. Why does MLM not directly support generation?**
Trained to fill the middle, not extend the end. To generate, you'd need to autoregressively mask one position and fill it — slow and unnatural. Modern LLMs need generation; they use CLM.

> **Saying it out loud.** Because it was trained to fill holes, not to extend the end of a sequence. To generate you'd have to append a MASK, predict it, append another, and re-run the whole encoder each time, which is both slow and off-distribution — the model never saw a sequence with all the future missing during training. There's no KV cache to exploit either, since bidirectional attention means every new token changes every earlier representation. So generation costs you a full quadratic forward pass per token instead of an incremental one, which is a non-starter.

**12. Where does MLM still win?**
Encoder models for embeddings, classification, NER. Sentence-BERT, BGE, E5 — all use MLM-style pretraining. Bidirectional context produces richer per-token representations.

> **Saying it out loud.** Anywhere you want to understand text rather than produce it. If your job is classification, named entity recognition, or turning a sentence into an embedding for retrieval, bidirectional context gives you a strictly better representation of each token, because the meaning of a word genuinely depends on what follows it. That's why the good retrieval encoders — Sentence-BERT, E5, BGE — are all MLM-pretrained. And they're small: a 110M-parameter encoder can beat a 7B decoder on embedding quality while being fifty times cheaper to run, which is why nobody has replaced them with LLMs.

**13. MLM's train-test mismatch?**
`[MASK]` appears in training but not at inference. The 80/10/10 mix mitigates but doesn't fully eliminate this. ELECTRA's replaced-token detection sidesteps the issue.

> **Saying it out loud.** The mismatch is that MASK is a token the model sees constantly in pretraining and never once in the real world. So the model's job at fine-tuning time is subtly different from the job it was trained on. The 80/10/10 mix softens this but doesn't remove it — 80 percent of scored positions still show a MASK. ELECTRA sidesteps it entirely by never asking the model to fill a blank: instead it shows a corrupted sentence and asks which tokens are fake, so the input distribution at pretraining actually looks like real text.

---

## C. NSP and ELECTRA

**14. What was NSP?**
Next Sentence Prediction. BERT trained with an auxiliary task: predict whether sentence B follows sentence A. 50% positive pairs (consecutive sentences); 50% negative (random pairs).

> **Saying it out loud.** Next sentence prediction was BERT's second objective: show it two segments and have it predict whether B genuinely followed A in the corpus, with half the examples being random pairs. The idea was to teach discourse-level relationships that fill-in-the-blank alone might miss, which mattered because BERT targeted tasks like question answering and entailment that involve two pieces of text. It's a binary classification head on the CLS token, so it costs almost nothing. Good motivation, and it turned out not to work.

**15. Why was NSP removed?**
RoBERTa (Liu et al. 2019) showed empirically: NSP doesn't help, removing it improves downstream performance. The task is too easy (random sentences are trivially distinguishable from continuation) — minimal useful signal.

> **Saying it out loud.** Because RoBERTa showed it doesn't help and removing it makes things better. The problem is the task is too easy for the wrong reason — a randomly sampled negative sentence usually comes from a totally different document, so the model can call it just by noticing the topic changed. It never has to learn anything about discourse coherence. Once you can solve a task with a shortcut, it stops producing useful gradient, and the capacity is better spent on masked prediction. ALBERT later replaced it with sentence-order prediction, which uses two real consecutive sentences and just swaps them, so topic overlap can't give the answer away.

**16. What's ELECTRA?**
Clark et al. 2020. Replaced-token detection. A small generator fills in masked tokens; a larger discriminator predicts which tokens were replaced. Loss is binary classification per token.

> **Saying it out loud.** ELECTRA turns pretraining into spot-the-fake. A small generator model fills in the masked positions with plausible guesses, and then the main model — the discriminator — looks at every token in the resulting sentence and says real or replaced. It's a binary classification at every position rather than a vocabulary-sized softmax at 15 percent of positions. You throw the generator away afterwards and keep the discriminator. It's GAN-flavored but not actually adversarial: the generator is trained on plain masked language modeling, not to fool the discriminator, because backpropagating through discrete token sampling doesn't work.

**17. Why is ELECTRA more sample-efficient than MLM?**
Every token contributes to the loss (binary classification at every position), not just 15%. ~4x more sample-efficient than MLM. Matches BERT performance with ~25% the compute.

> **Saying it out loud.** Every token contributes to the loss instead of just 15 percent, so you extract roughly seven times more signal per forward pass. And the negatives aren't random junk — they come from a trained generator, so they're plausible-but-wrong words, which makes the discrimination task genuinely hard and informative. The published number is that ELECTRA matches BERT's downstream performance with about a quarter of the compute, or beats it at equal compute. The tradeoff is that the pretraining setup is more complicated: you're training two models and you have to tune their relative sizes, with the generator usually around a quarter to a half the discriminator's size.

**18. Where is ELECTRA-style pretraining used?**
Some efficient encoder models. Doesn't directly support generation, so not relevant for LLMs. ELECTRA-style ideas are sometimes incorporated into hybrid objectives.

> **Saying it out loud.** Mostly in efficient encoder models, and it never crossed over to LLMs for one structural reason: a discriminator that outputs real-or-fake per token has no generative head at all, so you can't sample from it. That was fine when the field's center of gravity was encoders and fine-tuning, and it stopped mattering the moment everyone wanted generation. The ideas do survive as auxiliary objectives in some hybrid setups. It's a good example of a genuinely better objective that lost because the field's requirements changed underneath it.

---

## D. Span corruption and PrefixLM

**19. What's span corruption?**
T5/BART objective. Mask contiguous spans of tokens (~3 tokens average); replace each span with a sentinel; encoder-decoder predicts the masked spans as output. Encoder is bidirectional; decoder is causal.

> **Saying it out loud.** It's masked language modeling where you blank out contiguous chunks instead of scattered individual words. T5 drops spans averaging around three tokens, replaces each with a single sentinel marker, and has the decoder generate the missing spans in order. The motivation is that predicting one word from both sides is often trivial — the surrounding words nearly give it away — whereas recovering a whole phrase requires actual understanding. It also compresses the target, since each span collapses to one sentinel in the input. T5's recipe was 15 percent corruption with mean span length three.

**20. Walk me through span corruption with an example.**
Input: "The <X> jumped over the <Y> dog". Target: "<X> quick brown fox <Y> lazy `<eos>`". Each `<X>` etc. is a sentinel token marking a masked span. Decoder generates the spans in order.

> **Saying it out loud.** Take 'The quick brown fox jumped over the lazy dog.' You blank two spans, so the encoder sees 'The <X> jumped over the <Y> dog', where <X> and <Y> are sentinel tokens. The decoder's target is just the missing pieces tagged by sentinel: '<X> quick brown fox <Y> lazy'. Notice the target is much shorter than the input — you never regenerate the parts that weren't corrupted, which is what makes it efficient. The sentinels are what tell the decoder which hole it's currently filling, and T5 reserves a hundred of them in the vocabulary.

**21. Pros and cons of span corruption?**
Pros: efficient (whole spans contribute), captures phrase-level semantics, encoder-decoder architecture flexible. Cons: encoder-decoder is heavier than decoder-only; doesn't directly enable free-form generation as cleanly as CLM. Modern LLMs prefer CLM.

> **Saying it out loud.** On the plus side it forces phrase-level understanding rather than local word guessing, the targets are short so decoding is cheap, and the encoder-decoder split gives you bidirectional encoding of the input, which is genuinely better for tasks like summarization and translation. The downsides are structural: an encoder-decoder is roughly twice the parameters for the same depth, you have cross-attention to serve, and free-form open-ended generation isn't what it was trained for. Modern LLMs went with plain next-token prediction because one causal stack does everything with a simpler serving story — and the compute savings compound at scale.

**22. What's PrefixLM?**
Hybrid attention pattern: bidirectional over the prefix; causal over the target. Used in T5.1.1, GLM. Theoretically combines benefits of bidirectional encoding and autoregressive generation.

> **Saying it out loud.** PrefixLM is the compromise: attention is bidirectional over the input prefix and causal over the part you're generating. So the model gets to read the prompt with full two-way context, like an encoder, then continue it autoregressively, like a decoder — but in a single stack rather than two. T5.1.1 and GLM use it. On paper it's the best of both, since there's no obvious reason your prompt should be read left-to-right when you already have all of it.

**23. Why didn't PrefixLM dominate?**
Implementation complexity (different attention in different parts). Decoder-only with in-context demonstrations achieves similar effects with less complexity. PrefixLM never took off at frontier scale.

> **Saying it out loud.** Mostly because the complexity doesn't pay for itself. You need a per-example attention mask that changes shape depending on where the prefix ends, which fights with every optimized attention kernel and with KV caching — you can't reuse cached prefix states the same way when the prefix attends to itself bidirectionally. Meanwhile plain causal models turned out to be shockingly good at using their prompts anyway, once they were big enough. So you're taking on real infrastructure pain for a benefit that shrinks with scale, and nobody has shown a frontier-scale win.

---

## E. Modern variants

**24. What's Mixture-of-Denoisers (UL2)?**
Tay et al. 2022. Combines multiple denoising objectives: R-denoising (regular spans), S-denoising (sequential prefix-LM), X-denoising (extreme corruption). Model learns multiple skills. Research-prominent; not adopted at frontier scale.

> **Saying it out loud.** UL2's idea is to stop picking one corruption objective and train on several, with a special token telling the model which mode it's in. R-denoising is normal span corruption, S-denoising is prefix continuation, and X-denoising is extreme — very long spans or very high corruption rates. The model learns both understanding-style and generation-style skills, and you can switch modes at inference. It was well-received in research and never adopted at frontier scale, which is the recurring pattern here: clever multi-objective schemes keep losing to plain next-token prediction plus more data.

**25. What's multi-token prediction?**
Gloeckle et al. 2024, used in DeepSeek-V3. Predict the next $k$ tokens at each position using $k$ separate prediction heads. Denser signal per token. Auxiliary heads enable speculative decoding without a separate draft model.

> **Saying it out loud.** Instead of predicting just the next token, you predict the next $k$ tokens from each position using $k$ small output heads. The argument is that next-token prediction is myopic — the model can get low loss by tracking local grammar without planning ahead — and forcing it to commit to several tokens at once pushes it to represent more of the future. DeepSeek-V3 uses this and reports better quality at the same compute. The bonus is genuinely practical: those extra heads are a built-in draft model, so you get speculative decoding without training a separate one, which is roughly a two times speedup at inference.

**26. What's the role of contrastive losses?**
Used for embedding models (Sentence-BERT, BGE, E5): pull similar sentences together in embedding space, push different ones apart. Different paradigm from generative LM losses; supports retrieval and semantic search.

> **Saying it out loud.** Contrastive loss is what you use when the thing you want is a good embedding rather than good text. You take pairs that should be similar, like a question and its answer, and pull them together in vector space while pushing everything else in the batch away. That's a completely different target from next-token prediction — language modeling never asks that similar sentences land near each other, so LM-trained representations are surprisingly bad at retrieval out of the box. This is why every serious embedding model, from Sentence-BERT to E5 to BGE, has a contrastive stage. And the practical lever is batch size: more in-batch negatives means a better signal, which is why these get trained with batches in the thousands.

---

## F. Cross-entropy specifics

**27. Why cross-entropy as the LM loss?**
MLE under categorical distribution. The log-likelihood of the data given the model factorizes as $\sum_t \log P(\text{token}_t \mid \text{context})$. Negative gives cross-entropy. It's not a design choice; it's what likelihood mandates.

> **Saying it out loud.** It isn't really a choice — it falls out of maximum likelihood. If you say your model defines a probability distribution over next tokens and you want to maximize the probability of the data you observed, then taking the log and flipping the sign gives you exactly cross-entropy. There's no separate design decision about which loss to use. That framing is worth having ready, because interviewers like to ask why not mean squared error, and the answer is that MSE corresponds to assuming Gaussian noise, which makes no sense for a categorical variable over 100,000 discrete tokens.

**28. What's perplexity?**
$\text{PPL} = \exp(\text{cross-entropy})$. Geometric inverse of average per-token probability. Bounded below by $\exp(\text{true entropy})$ (perfect LM $\approx 1$); bounded above by $|V|$ (uniform random model = vocab size).

> **Saying it out loud.** Perplexity is just cross-entropy exponentiated, so it lives in units of tokens instead of nats, and the intuition is that it's the effective number of choices the model is deciding between at each step. A perplexity of 10 means the model is about as confused as if it were picking uniformly from 10 options. Perfect prediction gives you 1, a uniform model over the vocabulary gives you the vocab size — so 50,000 or so. It's the same information as the loss, but it's the number people quote because 'perplexity 12' is more intuitive than 'loss 2.48'. The floor isn't 1 in practice, it's the true entropy of the text, which nobody knows.

**29. Why can't you compare PPL across tokenizers?**
PPL is per-token. Different tokenizers split text into different numbers of tokens. A tokenizer with finer splits gets lower PPL on the same text purely from having more easy predictions. Compare per-byte/per-character likelihood for fair comparison.

> **Saying it out loud.** Because perplexity is per token, and different tokenizers cut the same text into different numbers of tokens. A tokenizer with a bigger vocabulary produces fewer, chunkier tokens that are individually harder to predict, so its perplexity looks worse — even on an identical model and identical text. Go the other way, split into characters, and perplexity plummets because predicting the next character is easy. Neither number tells you which model is better. The fix is to normalize by something tokenizer-independent, like bits per byte or per character, and that's what careful papers report.

**30. Cross-entropy gradient w.r.t. logits?**
$\nabla \mathcal{L} / \nabla z = \mathrm{softmax}(z) - \text{one-hot}(\text{target})$. Same form as logistic regression. Clean because softmax is the canonical link function for the categorical distribution (GLM theory).

> **Saying it out loud.** It's beautifully simple: softmax of the logits minus the one-hot target. Predicted probability minus actual, exactly like logistic regression. So if the model put 0.7 on the right token, the gradient on that logit is $-0.3$ and every wrong token gets pushed down by its own probability. The reason it comes out this clean is that softmax is the canonical link function for the categorical distribution — the messy softmax Jacobian and the messy log derivative cancel each other exactly. That cancellation is also why frameworks fuse log-softmax and NLL into one op instead of computing them separately.

---

## G. Implementation gotchas

**31. What's the log-sum-exp trick and why?**
For numerical stability when computing softmax: $\log \sum_v \exp(z_v) = \max(z) + \log \sum_v \exp(z_v - \max(z))$. Without it, large logits would overflow `exp`. Standard in all softmax/cross-entropy implementations.

> **Saying it out loud.** It's the thing that stops softmax from overflowing. Exponentiating a logit of a few hundred gives you infinity in floating point, and then everything downstream becomes NaN. The trick is to subtract the maximum logit before exponentiating — mathematically it changes nothing, since the shift cancels between numerator and denominator, but now the largest exponent is exactly zero so nothing overflows, and anything that underflows to zero was negligible anyway. Every framework does this inside its softmax. It matters concretely in mixed-precision training, where fp16 tops out around 65,504 and you'd blow past that with logits above about 11.

**32. Why is `F.cross_entropy` better than `softmax + log + nll_loss`?**
PyTorch's `F.cross_entropy` combines log-softmax with negative log-likelihood in one numerically stable operation. Computing softmax first then taking log can lose precision via overflow/underflow.

> **Saying it out loud.** Because the fused version never materializes the probabilities. If you compute softmax first, a very confident wrong prediction gives a probability that rounds to zero in floating point, and then taking the log gives you negative infinity and your loss becomes NaN. Computing log-softmax directly keeps everything in log space where the numbers stay in a sane range. It's also faster and uses less memory, since it avoids a full vocabulary-sized intermediate tensor. The failure mode is sneaky: it doesn't happen on step one, it happens after the model gets confident, so your run dies at hour six for no visible reason.

**33. How do you handle very large vocabularies efficiently?**
Sampled softmax (during training): only compute softmax over a sampled subset of the vocab. Hierarchical softmax: tree-structured factorization. Adaptive softmax: cluster vocab by frequency. For modern LLMs, full softmax is feasible and standard.

> **Saying it out loud.** The honest modern answer is that you usually don't do anything special — full softmax over 100,000 tokens is fine on current hardware, and all the classic tricks are largely historical. The reason they existed is that the output layer used to dominate cost; with a 512-dimensional hidden state and a million-word vocabulary, the softmax was most of the model. Sampled softmax computes the denominator over a random subset, hierarchical softmax factors the vocabulary into a tree, adaptive softmax gives rare words fewer parameters. Where it does still bite is memory, not compute — the logits tensor is batch times sequence times vocab, which at 4K context can be several gigabytes, so people chunk the loss computation over the sequence.

---

## H. Advanced and frontier

**34. What's z-loss / output normalization regularization?**
Add $\alpha \cdot (\log Z(x))^2$ to the loss, where $Z$ is the partition function. Prevents the model from learning extremely large logits (which can cause instability). Used in some LLM pretraining recipes (PaLM, GPT-3 likely).

> **Saying it out loud.** Z-loss is a small penalty on the log of the softmax denominator, squared. The problem it solves is that softmax only cares about differences between logits, so the model can drift all its logits upward without changing predictions at all — and then you're doing exponentials on huge numbers, which is where fp16 and bf16 training goes unstable. Adding roughly $10^{-4}$ times $(\log Z)^2$ anchors the absolute scale without constraining the differences. PaLM used it explicitly, and it's cheap insurance: it costs almost nothing and prevents a class of loss spikes that would otherwise cost you a restart from a checkpoint.

**35. What's auxiliary loss in MoE training?**
For Mixture-of-Experts models: an auxiliary loss to encourage balanced expert utilization (so all experts get used roughly equally). Without it, the router collapses to using a few experts. See `41_mixture_of_experts/`.

> **Saying it out loud.** In a mixture-of-experts model the router picks which experts handle each token, and left alone it collapses — a few experts get good early, so the router sends them more tokens, so they get better still, and the rest of your parameters sit idle. The auxiliary load-balancing loss penalizes uneven routing to break that feedback loop. The tradeoff is that it fights the main objective: push the balancing weight too high and you're forcing tokens to experts that are wrong for them, which hurts quality. Typical weights are around 0.01, and newer work like DeepSeek-V3 tries to get balance from a bias-adjustment trick instead, to avoid the interference entirely.

---

## Quick fire

**36.** *Default mask ratio in BERT?* 15%.
**37.** *NSP removed in?* RoBERTa.
**38.** *T5 objective?* Span corruption.
**39.** *ELECTRA paper?* Clark et al. 2020.
**40.** *Modern LLM default?* CLM (next-token prediction).

---

## Self-grading

If you can't answer 1-10, you don't know LM losses. If you can't answer 11-25, you'll struggle on architecture interviews. If you can't answer 26-40, frontier-lab interviews will go past you.

Aim for 25+/40 cold.
