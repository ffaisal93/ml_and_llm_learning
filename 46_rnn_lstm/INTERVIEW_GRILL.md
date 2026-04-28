# RNNs & LSTMs — Interview Grill

> 40 questions on RNN/LSTM/GRU mechanics, BPTT, attention, transformer transition. Drill until you can answer 28+ cold.

---

## A. Vanilla RNN

**1. Vanilla RNN update?**
$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$.

**2. Why parameter sharing across time?**
Time-invariance assumption; same dynamics at every step. Drastically fewer params than feed-forward over full sequence.

**3. RNN universal approximator?**
For sequence-to-sequence functions, in principle. Practical training is hard.

**4. RNN output formula?**
$y_t = W_{hy} h_t + b_y$ (or fed through softmax for classification).

---

## B. BPTT

**5. What is BPTT?**
Backpropagation through time. Unroll the RNN over $T$ steps; backprop through resulting deep computation graph.

**6. Memory cost of BPTT for $T$-step sequence?**
$O(T \cdot \text{hidden size})$ per layer — store all activations.

**7. What's truncated BPTT?**
Only backprop $K$ steps; treat earlier as fixed. Saves memory; loses long-range gradient info.

**8. Vanishing gradient cause?**
Repeated multiplication by $\partial h/\partial h_{\text{prev}}$ — when spectral radius < 1, product → 0.

**9. Exploding gradient cause?**
Spectral radius > 1 → product blows up.

**10. Standard fix for explosion?**
Gradient clipping by global norm (typically 1.0).

**11. Why is $\tanh$ specifically problematic?**
Saturates at $\pm 1$; derivative is at most 1, often much smaller. Multiplied through $T$ steps → vanishes.

**12. Orthogonal initialization — why?**
Initialize $W_{hh}$ orthogonal so eigenvalues are exactly 1 — gradient neither vanishes nor explodes initially.

---

## C. LSTM

**13. LSTM has how many gates?**
Three: forget, input, output.

**14. Forget gate formula?**
$f_t = \sigma(W_f [h_{t-1}; x_t] + b_f)$.

**15. Cell state update?**
$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$.

**16. Hidden state output?**
$h_t = o_t \odot \tanh(c_t)$.

**17. Why does cell state fix vanishing gradients?**
Additive update $c_t = f_t \odot c_{t-1} + (\ldots)$. With $f_t \approx 1$, identity-like gradient path. No multiplicative decay.

**18. Standard forget-bias initialization?**
$b_f \approx 1$ (positive). Sigmoid evaluates near 1 → cell state propagates by default.

**19. Why have separate cell and hidden state?**
Cell state: pure long-term memory, additive updates. Hidden state: passed to next layer / output, gated read.

**20. Connection between LSTM and residual networks?**
Both: additive identity path keeps gradient stable across many "depths" (time steps for LSTM, layers for ResNet).

---

## D. GRU

**21. GRU has how many gates?**
Two: update gate $z$ and reset gate $r$.

**22. GRU vs LSTM — what's combined?**
Forget and input gates merged into single update gate. No separate cell state.

**23. GRU update formula?**
$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$.

**24. Reset gate role?**
$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}; x_t])$. Resets memory before computing candidate.

**25. GRU vs LSTM in practice?**
Comparable. GRU faster (fewer params); LSTM slightly more expressive. Empirical results mixed.

---

## E. Bidirectional + seq2seq

**26. Bidirectional RNN?**
Forward + backward RNN; concatenate hidden states. Captures both past and future context.

**27. Why not bidirectional for generation?**
Future tokens don't exist at generation time. BiRNN only for tasks with full sequence available (NER, POS, classification).

**28. Seq2seq architecture?**
Encoder RNN reads source; passes final hidden state to decoder RNN that generates target autoregressively.

**29. Bottleneck problem in seq2seq?**
Encoder compresses entire source into one fixed vector. Hard for long sentences.

**30. Bahdanau attention idea?**
At each decoder step, compute weighted average of *all* encoder hidden states. Decoder reads from source dynamically.

**31. Attention scoring functions?**
Bahdanau: $v^\top \tanh(W_1 h^{\mathrm{dec}} + W_2 h^{\mathrm{enc}})$. Luong: $h^{\mathrm{dec} \top} h^{\mathrm{enc}}$ (dot product) or $h^{\mathrm{dec} \top} W h^{\mathrm{enc}}$ (general).

---

## F. Transformer transition

**32. Why are transformers parallelizable but RNNs aren't?**
RNN: $h_t$ depends on $h_{t-1}$ — sequential. Transformer: attention over all positions independent of order — parallel matmul.

**33. Long-range dependency comparison?**
RNN: signal must traverse $O(T)$ steps. LSTM helps but still degrades over long range. Transformer: any pair of positions $O(1)$ steps apart.

**34. Scaling behavior of LSTM vs transformer?**
Transformer scales better. LSTMs plateau in performance with more compute; transformers keep improving (scaling laws).

**35. When still use LSTM today?**
Streaming/online tasks where causal sequential is natural. Tiny tasks where transformer overhead isn't worth it. Some signal processing / low-latency speech.

---

## G. Modern context

**36. Mamba vs LSTM — what's similar?**
Both are recurrent: state evolves with each input. Both have linear complexity in sequence length.

**37. Mamba vs LSTM — what's different?**
Mamba: linear recurrence with carefully chosen $A$ (HiPPO-inspired or selective); parallel scan for training; no $\tanh$.

**38. Why couldn't RNNs do what Mamba does in 1997?**
Parallel scan algorithm wasn't connected to RNN training; HiPPO theory wasn't developed. Modern SSMs are "what RNNs should have been."

**39. Catastrophic forgetting in RNNs?**
Adding capacity for new task overwrites old. RNNs especially vulnerable due to shared parameter across all positions/tasks.

**40. Why did Karpathy's "Unreasonable Effectiveness of RNNs" 2015 hold but not in 2024?**
Transformer + scale destroyed the RNN advantage. RNNs are still effective but not state of the art for any flagship NLP task.

---

## Quick fire

**41.** *RNN gradient problem source?* Repeated multiplication of Jacobians.
**42.** *LSTM cell state pathway?* Additive (residual-like).
**43.** *LSTM gates count?* 3.
**44.** *GRU gates count?* 2.
**45.** *Forget bias init?* Positive (~1.0).
**46.** *Standard gradient clip?* 1.0 by global norm.
**47.** *Bidirectional for generation?* No.
**48.** *Bahdanau attention introduced?* 2014.
**49.** *Transformer year?* 2017.
**50.** *RNN vs Transformer parallel?* Transformer.

---

## Self-grading

If you can't answer 1-15, you don't know RNNs. If you can't answer 16-30, you'll struggle on LSTM/seq2seq questions. If you can't answer 31-45, you can't connect RNN history to modern architectures.

Aim for 30+/50 cold.
