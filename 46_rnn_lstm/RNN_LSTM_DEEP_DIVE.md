# RNNs and LSTMs — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

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

### Why parameter sharing?

The model assumes the dynamics are time-invariant — what works for predicting from history at $t = 5$ also works at $t = 50$. Far fewer parameters than feed-forward over the full sequence.

### Capacity

Universal approximator for sequence-to-sequence functions in principle. But practical training is hard.

---

## 2. Backpropagation through time (BPTT)

To compute gradients, "unroll" the RNN across $T$ time steps and backprop through the resulting deep network.

### Gradient form

For loss $\mathcal{L} = \sum_t \ell_t$:

$$
\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^T \sum_{k=1}^t \frac{\partial \ell_t}{\partial h_t} \left(\prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}\right) \frac{\partial h_k}{\partial W_{hh}}
$$

The product of Jacobians is the source of all RNN training pain.

### Vanishing / exploding gradients

$\partial h_j / \partial h_{j-1} = W_{hh}^\top \cdot \mathrm{diag}(\tanh'(\cdot))$. The product over many steps:

- If spectral radius of $W_{hh} < 1$: gradient vanishes geometrically. Long-range dependencies untrainable.
- If spectral radius > 1: gradient explodes. NaN.

This was *the* central problem of pre-2015 sequence modeling. Solutions:

- **Gradient clipping**: $\|\nabla\| \leq \tau$. Standard fix for explosion.
- **Better activations** ($\tanh$ → ReLU): partially helps but still vanishes.
- **Better init**: orthogonal $W_{hh}$ to keep eigenvalues near 1.
- **LSTM / GRU**: structured architectural fix (next section).

### Truncated BPTT

For very long sequences, unroll only $K$ steps backward (forget gradients beyond). Trade longer-range learning for memory feasibility.

---

## 3. LSTM — long short-term memory

Hochreiter & Schmidhuber (1997) introduced LSTMs to fix vanishing gradients.

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

### Update

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

### Why does this fix vanishing gradients?

The key is the cell state update $c_t = f_t \odot c_{t-1} + (\ldots)$. If forget gate $f_t \approx 1$, then $c_t \approx c_{t-1}$ — there's an *additive* identity-like path through time. Gradient flows backward without multiplicative decay (much like residual connections in deep networks).

This is the same principle that residual connections later used for spatial depth: provide an additive shortcut so gradient never has to be multiplied through every layer.

### Bias trick

Initialize forget gate bias $b_f$ to a positive value (e.g., 1.0) so forget gates start near 1 → cell state propagates by default.

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

---

## 5. Bidirectional RNN

For tasks where future context matters (NER, POS tagging), use two RNNs: forward (left-to-right) and backward (right-to-left). Concatenate their hidden states:

$$
h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]
$$

Cannot be used for autoregressive generation (you don't have the future).

---

## 6. Seq2seq with attention

The architecture that powered neural machine translation pre-transformer.

### Encoder-decoder
- Encoder RNN reads source sequence, produces final hidden state.
- Decoder RNN starts from encoder's final state, generates target sequence autoregressively.

### Bottleneck problem
The single fixed-size encoder vector struggles to capture all source information for long sentences.

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

---

## 7. Why transformers won

LSTMs were dominant 2014–2017. Why did transformers replace them?

### Parallelism
LSTMs process tokens sequentially: $h_t$ depends on $h_{t-1}$. Can't parallelize across the time dimension. Transformers compute attention for all positions simultaneously.

### Long-range dependencies
LSTMs *better* than RNNs but still struggle with sequences > a few hundred tokens. Self-attention has direct $O(1)$-step paths between any two positions.

### Scaling
Transformers scale: more compute → consistently better performance (Kaplan et al. 2020). LSTMs plateau earlier.

### Architecture stability
Transformers benefit from pre-LN, residual connections, normalization in ways that turned out to be more stable at scale.

### What LSTMs still do
- Small / fast tasks where transformer overhead isn't worth it.
- Streaming / online tasks where causal sequential processing is natural.
- Specialized domains (some signal processing, speech with low latency).
- Modern SSMs (Mamba) revive RNN-like sequential processing with better trainability.

---

## 8. The connection to modern SSMs

Mamba and S4 are conceptually "RNNs that work":

- **Linear recurrence**: $h_t = A h_{t-1} + B x_t$ instead of nonlinear $\tanh$.
- **Carefully chosen $A$**: HiPPO matrices (S4) or input-dependent (Mamba) ensure long-range memory without vanishing.
- **Parallel scan**: linear recurrences can be computed in parallel via the parallel scan algorithm — fixes the sequential training problem.
- **Selectivity (Mamba)**: $A, B, C$ depend on input, mimicking attention's content-based mixing.

In some sense, modern SSMs are "what RNNs would have been if we'd known about HiPPO and parallel scans in 1997."

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
