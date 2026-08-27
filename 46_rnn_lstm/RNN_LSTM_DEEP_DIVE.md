# RNNs and LSTMs — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

RNNs lost the architecture race to transformers, but they're still asked in interviews because: (1) the failure modes (vanishing gradients) motivate every modern architectural choice, (2) LSTM gating is the conceptual ancestor of attention, and (3) modern SSMs (Mamba) are essentially "RNNs done right" — knowing the lineage matters.

---

## 1. The vanilla RNN

A recurrent network maintains a hidden state $h_t$ that summarizes everything seen so far:

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

$$
y_t = W_{hy} h_t + b_y
$$

Same parameters $W_{hh}, W_{xh}, W_{hy}$ at every time step. The network unrolls over time but reuses weights.

> **Saying it out loud.** An RNN keeps one hidden vector and rewrites it at every step — mix in what you remembered, mix in the new input, squash through a tanh, read the output off with a linear layer. Same weights every step, which is why it can handle sequences longer than anything it trained on. The whole story of the field is contained in that squash: it keeps the state bounded, and it's also what strangles the gradient over long sequences. Everything from the LSTM to Mamba is an attempt to keep the memory while getting rid of the squash.

### Why parameter sharing?

The model assumes the dynamics are time-invariant — what works for predicting from history at $t = 5$ also works at $t = 50$. Far fewer parameters than feed-forward over the full sequence.

> **Saying it out loud.** You use the same weights at step 5 and step 500 because the rule for updating memory shouldn't depend on where you are in the sequence. It keeps the parameter count independent of length and lets the model generalize to sequences longer than it saw in training. The sting is that sharing means the same matrix gets applied over and over, and repeatedly multiplying by one matrix is precisely what makes gradients explode or vanish. The property that buys generalization is the property that causes the pathology.

### Capacity

Universal approximator for sequence-to-sequence functions in principle. But practical training is hard.

> **Saying it out loud.** Yes, an RNN is a universal approximator for sequence functions, and with unbounded precision it's even Turing complete. That tells you almost nothing useful, because the question was never whether a good solution exists in weight space — it's whether gradient descent can reach it. Long-range solutions exist and are effectively unreachable by BPTT. It's a nice case study in a representational theorem with no practical content.

---

## 2. Backpropagation through time (BPTT)

To compute gradients, "unroll" the RNN across $T$ time steps and backprop through the resulting deep network.

> **Saying it out loud.** BPTT just means unrolling the recurrence into a deep feed-forward network — one layer per timestep, all sharing weights — and running ordinary backprop. So a 100-step sequence is a 100-layer network whose gradients all pile into the same weight matrix. Once you hold that picture, everything follows: the vanishing gradient is the depth problem, the memory cost is storing 100 layers of activations, and truncation is just cutting the network short.

### Gradient form

> **In plain language.** The formula below says the gradient at any step has to travel back through every intermediate step, and travelling through each one means multiplying by that step's Jacobian. That chain of multiplications is the entire source of RNN training difficulty.

For loss $\mathcal{L} = \sum_t \ell_t$:

$$
\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^T \sum_{k=1}^t \frac{\partial \ell_t}{\partial h_t} \left(\prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}\right) \frac{\partial h_k}{\partial W_{hh}}
$$

The product of Jacobians is the source of all RNN training pain.

> **Saying it out loud.** The double sum looks awful, but the only part that matters is the product in the middle. To get gradient from a loss at step 100 back to what happened at step 1, you multiply by the same Jacobian 99 times. Multiply any number by itself 99 times and it either goes to zero or to infinity — landing near one is a knife's edge. So the RNN's difficulty isn't a tuning issue, it's arithmetic, and no learning rate schedule fixes it.

### Vanishing / exploding gradients

$\partial h_j / \partial h_{j-1} = W_{hh}^\top \cdot \mathrm{diag}(\tanh'(\cdot))$. The product over many steps:

- If spectral radius of $W_{hh} < 1$: gradient vanishes geometrically. Long-range dependencies untrainable.
- If spectral radius > 1: gradient explodes. NaN.

This was *the* central problem of pre-2015 sequence modeling. Solutions:

- **Gradient clipping**: $\|\nabla\| \leq \tau$. Standard fix for explosion.
- **Better activations** ($\tanh$ → ReLU): partially helps but still vanishes.
- **Better init**: orthogonal $W_{hh}$ to keep eigenvalues near 1.
- **LSTM / GRU**: structured architectural fix (next section).

> **Saying it out loud.** The Jacobian is the recurrent weight matrix times the tanh derivative, and both of those tend to shrink things — tanh's derivative is at most 1 and much less once it saturates. If the effective factor is 0.9, then after 50 steps you're down 200-fold and after 100 steps there's nothing left, so distant dependencies are simply untrainable. Above 1 you get the opposite, an explosion to NaN. The asymmetry worth noticing is that explosion announces itself immediately and clipping fixes it, while vanishing is silent — the model trains fine and just quietly never learns anything long-range.

### Truncated BPTT

For very long sequences, unroll only $K$ steps backward (forget gradients beyond). Trade longer-range learning for memory feasibility.

> **Saying it out loud.** You cap the backward pass at some window, typically 20 to 50 steps, and treat everything older as constant. The forward state still crosses the boundary, so the model can in principle remember longer than it can learn to remember. That's the trade: bounded memory and compute, in exchange for making dependencies longer than your window structurally unlearnable. If your task needs 200-step credit assignment and you truncate at 35, you'll never learn it and nothing in the loss curve will tell you why.

---

## 3. LSTM — long short-term memory

> **In plain language.** The LSTM adds a second vector — the cell state — that gets updated by addition instead of being rewritten from scratch, plus three sigmoid-valued dials controlling what gets erased, what gets written, and what gets read out. The equations below are those three dials and the additive update.

Hochreiter & Schmidhuber (1997) introduced LSTMs to fix vanishing gradients.

> **Saying it out loud.** The LSTM's whole idea is to give memory a route through time that doesn't get multiplied by a weight matrix. Picture the cell state as a whiteboard: the forget gate decides what to erase, the input gate decides what to write, and the output gate decides how much of it to read out this step. Because the update is add-and-scale rather than transform-and-squash, the gradient has a nearly clean path backward. That's the same trick residual connections would rediscover for depth 18 years later.

### Cell state and gates

LSTMs maintain two hidden vectors: cell state $c_t$ (long-term memory) and hidden state $h_t$ (short-term/output).

Three gates control information flow:

**Forget gate** $f_t$: what to drop from previous cell state.

$$
f_t = \sigma(W_f [h_{t-1}; x_t] + b_f)
$$

**Input gate** $i_t$ + candidate cell content $\tilde{c}_t$: what to add.

$$
i_t = \sigma(W_i [h_{t-1}; x_t] + b_i), \quad \tilde{c}_t = \tanh(W_c [h_{t-1}; x_t] + b_c)
$$

**Output gate** $o_t$: what to read from cell state.

$$
o_t = \sigma(W_o [h_{t-1}; x_t] + b_o)
$$

> **Saying it out loud.** There are two state vectors because they do different jobs. The cell state is protected long-term memory with an additive path; the hidden state is the working output you actually expose to the next layer. Each gate is a sigmoid, so it emits a number between zero and one per dimension — a soft, differentiable switch. Per-dimension is the detail people gloss over: the model can hold the subject of a sentence in some dimensions while clearing a finished clause in others, all in the same step.

### Update

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

> **Saying it out loud.** The one symbol that matters here is the plus. Old memory times the forget gate, plus new content times the input gate — nothing gets pushed through a matrix multiply and a nonlinearity on its way forward. With the forget gate near one, the cell state essentially passes through unchanged, and so does the gradient going the other way. The hidden state is then a gated, squashed view of that memory, which is what lets the model hold something for a hundred steps without acting on it and then surface it exactly when relevant.

### Why does this fix vanishing gradients?

The key is the cell state update $c_t = f_t \odot c_{t-1} + (\ldots)$. If forget gate $f_t \approx 1$, then $c_t \approx c_{t-1}$ — there's an *additive* identity-like path through time. Gradient flows backward without multiplicative decay (much like residual connections in deep networks).

This is the same principle that residual connections later used for spatial depth: provide an additive shortcut so gradient never has to be multiplied through every layer.

> **Saying it out loud.** Because backward through the cell state, the multiplication is by the forget gate, and a forget gate near one is multiplication by one. Compare that to the vanilla RNN, where every step multiplies by a weight matrix and a tanh derivative, both shrinking. Be honest about the limit though: it postpones the problem rather than removing it, since forget gates below one still compound, so LSTMs reliably manage hundreds of steps rather than thousands. Better constant, same asymptotics — which is exactly why attention's $O(1)$ path was such a big deal.

### Bias trick

Initialize forget gate bias $b_f$ to a positive value (e.g., 1.0) so forget gates start near 1 → cell state propagates by default.

> **Saying it out loud.** Start the forget gate bias at about 1 so the sigmoid opens near 0.8 and the default behavior is to keep memory. Leave it at zero and the gate sits near 0.5, meaning the cell state halves every step — a one-step half-life, and the gradient dies long before the model learns to open the gate. It's a single line of code worth a large improvement on long-sequence tasks, and it's a perfect example of initialization deciding whether a capability is learnable at all rather than just how fast.

---

## 4. GRU — gated recurrent unit

Cho et al. (2014). Simpler variant: merge forget and input gates into a single update gate; eliminate separate cell state.

$$
z_t = \sigma(W_z [h_{t-1}; x_t]) \quad (\text{update gate})
$$

$$
r_t = \sigma(W_r [h_{t-1}; x_t]) \quad (\text{reset gate})
$$

$$
\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}; x_t])
$$

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

Fewer parameters. Comparable to LSTM in practice; sometimes slightly worse, sometimes equivalent.

> **Saying it out loud.** The GRU compresses the LSTM into two gates and one state vector. The update gate does the forget and input jobs together, forcing what you keep and what you write to sum to one — you can't keep everything and also add more, the way an LSTM can. The reset gate separately controls how much of the past feeds into the new candidate, which is effectively a start-fresh switch for sentence or topic boundaries. Empirically it's a coin flip against the LSTM, roughly 25 percent fewer parameters and a bit faster, so pick GRU if you're compute-bound and don't expect the choice to decide anything.

---

## 5. Bidirectional RNN

For tasks where future context matters (NER, POS tagging), use two RNNs: forward (left-to-right) and backward (right-to-left). Concatenate their hidden states:

$$
h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]
$$

Cannot be used for autoregressive generation (you don't have the future).

> **Saying it out loud.** Run two independent RNNs, one each direction, and glue their states together so every position sees the whole sequence. It's a genuine quality win for tagging, entity recognition and classification, because the meaning of a word often depends on what comes after it. What you give up is streaming and generation — nothing can be emitted until the entire input has arrived, and during generation the future doesn't exist to encode. Same reason BERT can't generate and GPT can't look right: bidirectional whenever you have the full sequence, unidirectional whenever you're producing it.

---

## 6. Seq2seq with attention

The architecture that powered neural machine translation pre-transformer.

> **Saying it out loud.** Seq2seq was the 2014 breakthrough — one RNN reads the source, hands its final state to a second RNN that writes the translation, end to end with no alignment tables. Its flaw is sitting right in the middle: the entire meaning of the sentence has to fit through one fixed vector, so quality falls off badly past about 30 words. Attention fixes that by letting the decoder look back at every encoder state and take a weighted average, recomputed at every output step. That fix is the direct ancestor of the transformer — the 2017 paper's move was to keep the attention and throw the RNN away.

### Encoder-decoder
- Encoder RNN reads source sequence, produces final hidden state.
- Decoder RNN starts from encoder's final state, generates target sequence autoregressively.

### Bottleneck problem
The single fixed-size encoder vector struggles to capture all source information for long sentences.

> **Saying it out loud.** Everything the decoder can ever know about the source has to squeeze through one vector of maybe 1,000 numbers, whether the sentence is five words or fifty. That's not something more data fixes; it's an information bottleneck built into the architecture, and you can see it as BLEU degrading sharply with source length in the original papers. Naming that specific curve is a good way to show you know the empirical result and not just the story.

### Bahdanau attention (2014) / Luong attention (2015)

At each decoder step, attend to *all* encoder hidden states:

$$
\alpha_{t, s} = \frac{\exp(\mathrm{score}(h_t^{\mathrm{dec}}, h_s^{\mathrm{enc}}))}{\sum_{s'} \exp(\mathrm{score}(h_t^{\mathrm{dec}}, h_{s'}^{\mathrm{enc}}))}
$$

$$
c_t = \sum_s \alpha_{t, s} h_s^{\mathrm{enc}}
$$

Decoder receives context vector $c_t$ alongside its hidden state. Lets it attend dynamically to the relevant part of the source.

This was the seed that grew into the transformer's self-attention. The transformer (Vaswani et al. 2017) realized you can drop the RNN entirely and just stack attention.

> **Saying it out loud.** Instead of one summary vector, the decoder scores every encoder state against its current state, softmaxes those scores into weights, and reads a weighted average. So while generating the third word it can focus on whatever part of the source is relevant right now, and the length-degradation curve flattens out. Bahdanau's scoring uses a small tanh network; Luong's uses a plain dot product, which won because it's a single matmul that maps cleanly onto GPUs. Add a $\sqrt{d}$ scaling to Luong's version and you have transformer attention exactly.

---

## 7. Why transformers won

LSTMs were dominant 2014–2017. Why did transformers replace them?

> **Saying it out loud.** The honest answer is parallelism, not intelligence. An RNN's step 500 needs step 499, so training a long sequence is a chain of dependent operations no matter how many GPUs you own; attention does the whole sequence in a few big matrix multiplies. That means transformers can absorb a datacenter and RNNs can't, and once scaling laws showed that more compute reliably buys more quality, the architecture that consumes compute best wins by default. The direct-path advantage for long-range dependencies is real too, but it's second — the economics is what ended the era.

### Parallelism
LSTMs process tokens sequentially: $h_t$ depends on $h_{t-1}$. Can't parallelize across the time dimension. Transformers compute attention for all positions simultaneously.

> **Saying it out loud.** This is the whole ballgame. A 1,000-token sequence in an LSTM is 1,000 sequential steps, and adding hardware doesn't shorten that chain. In a transformer it's a handful of kernel launches, with every position computed at once. The consequence is that transformer training time scales with how much hardware you can buy, and RNN training time doesn't — which is why you can spend ten million dollars usefully on one and not on the other.

### Long-range dependencies
LSTMs *better* than RNNs but still struggle with sequences > a few hundred tokens. Self-attention has direct $O(1)$-step paths between any two positions.

> **Saying it out loud.** In an RNN, information from token one reaches token five hundred by surviving 499 sequential updates, each a chance to be overwritten. In a transformer it's one hop — any position can look at any other directly. That's $O(T)$ path length versus $O(1)$, and it's why transformers handle coreference and long-distance agreement so much better. LSTMs stretched the usable range from tens of steps to hundreds; they didn't stop it being a path.

### Scaling
Transformers scale: more compute → consistently better performance (Kaplan et al. 2020). LSTMs plateau earlier.

> **Saying it out loud.** Transformers follow clean power laws over many orders of magnitude, so you can predict the final loss of a huge run from small pilots — that's what makes a multi-million-dollar training run an investment rather than a gamble. LSTMs flatten out earlier, partly because the sequential bottleneck limits how much data you can push through and partly because a fixed-size state limits what extra parameters can do with more context. Predictability, more than any single benchmark, is what made labs commit.

### Architecture stability
Transformers benefit from pre-LN, residual connections, normalization in ways that turned out to be more stable at scale.

> **Saying it out loud.** Transformers turned out to be unusually cooperative with the tricks that make deep networks trainable — residual streams, pre-LN normalization, careful initialization — so you can stack a hundred layers and have it just work. LSTMs are much fussier to deepen; stacking more than a few layers rarely helped and often hurt. Part of why is that the transformer's residual stream is a clean identity path through depth, which is the same structural idea the LSTM had for time but never got for depth.

### What LSTMs still do
- Small / fast tasks where transformer overhead isn't worth it.
- Streaming / online tasks where causal sequential processing is natural.
- Specialized domains (some signal processing, speech with low latency).
- Modern SSMs (Mamba) revive RNN-like sequential processing with better trainability.

> **Saying it out loud.** They survive where the sequence never ends and latency matters — streaming speech, real-time control, sensors on an embedded device — because an LSTM is constant time and constant memory per step while a transformer's cost grows with how much context it's holding. They also still win on genuinely small datasets, where the stronger inductive bias beats a transformer that has nothing to learn structure from. So: streaming, tiny hardware, small data. Everything else has moved on.

---

## 8. The connection to modern SSMs

Mamba and S4 are conceptually "RNNs that work":

- **Linear recurrence**: $h_t = A h_{t-1} + B x_t$ instead of nonlinear $\tanh$.
- **Carefully chosen $A$**: HiPPO matrices (S4) or input-dependent (Mamba) ensure long-range memory without vanishing.
- **Parallel scan**: linear recurrences can be computed in parallel via the parallel scan algorithm — fixes the sequential training problem.
- **Selectivity (Mamba)**: $A, B, C$ depend on input, mimicking attention's content-based mixing.

In some sense, modern SSMs are "what RNNs would have been if we'd known about HiPPO and parallel scans in 1997."

> **Saying it out loud.** Mamba and S4 are RNNs that finally work, and the differences are specific. The recurrence is linear — no tanh between steps — which means consecutive updates compose into another update of the same form, so a parallel scan computes the whole sequence in logarithmic depth instead of walking it. The state matrix gets a principled HiPPO-style initialization rather than a random one, so memory is stable over thousands of steps by construction. And Mamba adds selectivity, making the update input-dependent so each token controls how much it writes. The lesson to say out loud is that the equation was never the problem — initialization and hardware mapping were.

---

## 9. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| What's the vanishing gradient cause? | Bad activation | Multiplicative decay through repeated $\partial h/\partial h_{prev}$ products |
| LSTM vs GRU — major architectural diff? | None | LSTM has separate cell + hidden states + 3 gates; GRU merges into 2 gates and one state |
| Why does LSTM cell state help? | Mystery | Additive identity-like path; gradient flows back without decay (like residual connections) |
| Why aren't LSTMs the dominant architecture today? | They are | Transformers parallelize over time, scale better |
| Bidirectional RNN for autoregressive generation? | Sure | No — needs future context that doesn't exist at generation time |
| Truncated BPTT — what does it sacrifice? | Nothing | Gradients beyond $K$ steps are lost; can't learn dependencies > $K$ |
| Seq2seq + attention pre-transformer? | Same thing | Attention was added on top of RNN seq2seq before transformer dropped the RNN entirely |

---

## 10. Eight most-asked interview questions

1. **Walk through vanilla RNN forward and backward pass.** (BPTT; vanishing/exploding gradients explained.)
2. **Why does LSTM solve vanishing gradient?** (Cell state additive update; identity gradient path with $f_t \approx 1$.)
3. **LSTM vs GRU — when use each?** (LSTM more expressive; GRU simpler/faster; both comparable in practice.)
4. **Why bidirectional RNN?** (Future context; can't be used for autoregressive generation.)
5. **What's BPTT and why is it expensive?** (Unroll over $T$ steps; memory $O(T \cdot \text{hidden size})$.)
6. **Why did transformers replace LSTMs?** (Parallelism, long-range, scaling.)
7. **What's gradient clipping and why is it needed for RNNs?** (Cap $\|\nabla\|$; prevents explosion through long sequences.)
8. **How does seq2seq + attention work?** (Encoder hidden states; decoder attends weighted sum at each step.)

---

## 11. Drill plan

- Hand-derive vanilla RNN forward and one BPTT step.
- Recite LSTM gates and updates from memory. 5 minutes.
- Recite why LSTM cell-state additive update fixes vanishing gradients.
- Sketch encoder-decoder with attention diagram.
- For each "transformer beats LSTM" reason, recite + counter-example where LSTM is still chosen.

---

## 12. Further reading

- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory.*
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder-Decoder* (GRU + seq2seq).
- Sutskever, Vinyals, Le (2014), *Sequence to Sequence Learning with Neural Networks.*
- Bahdanau, Cho, Bengio (2014), *Neural Machine Translation by Jointly Learning to Align and Translate* (attention).
- Pascanu, Mikolov, Bengio (2013), *On the difficulty of training recurrent neural networks* (gradient analysis).
- Karpathy (2015 blog), *The Unreasonable Effectiveness of Recurrent Neural Networks*.
- Olah (2015 blog), *Understanding LSTM Networks* — clearest visualization of LSTM gates.
