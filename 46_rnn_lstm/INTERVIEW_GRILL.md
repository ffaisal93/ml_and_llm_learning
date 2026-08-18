# RNNs & LSTMs — Interview Grill

> 40 questions on RNN/LSTM/GRU mechanics, BPTT, attention, transformer transition. Drill until you can answer 28+ cold.

---

## A. Vanilla RNN

**1. Vanilla RNN update?**
$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$.

> **Saying it out loud.** An RNN keeps one hidden vector and rewrites it at every step: mix in what you remembered, mix in the new input, squash it through a tanh. That's it — the same three lines run at every position. The squashing is what keeps the state bounded, and it's also the thing that kills the gradient over long sequences, so the whole history of this field is people trying to keep the memory without the squash.

**2. Why parameter sharing across time?**
Time-invariance assumption; same dynamics at every step. Drastically fewer params than feed-forward over full sequence.

> **Saying it out loud.** You use the same weights at every timestep because the rule for updating memory shouldn't depend on whether you're at word five or word five hundred. Practically, it means the parameter count doesn't grow with sequence length, and the model can handle sequences longer than anything it saw in training. The cost is that the same matrix gets applied over and over, and repeatedly multiplying by one matrix is exactly what makes gradients vanish or explode — the sharing that buys generalization is the same thing that causes the pathology.

**3. RNN universal approximator?**
For sequence-to-sequence functions, in principle. Practical training is hard.

> **Saying it out loud.** In theory yes, an RNN with enough hidden units can approximate any sequence-to-sequence mapping, and there's a proof that they're Turing complete with unbounded precision. In practice that tells you almost nothing, because the question was never whether a solution exists in the weight space — it's whether gradient descent can find it. Long-range dependencies exist in parameter space and are essentially unreachable by BPTT. It's a good example of a representational result with no practical content.

**4. RNN output formula?**
$y_t = W_{hy} h_t + b_y$ (or fed through softmax for classification).

> **Saying it out loud.** You read the output off the hidden state with a linear layer, then softmax if you're classifying. The thing worth noticing is that everything the model knows about the entire past has to fit through that one hidden vector — typically a few hundred numbers. That's the fixed-size-bottleneck problem, and it's the same limitation that later motivated attention and, much later, the criticism of state space models.

---

## B. BPTT

**5. What is BPTT?**
Backpropagation through time. Unroll the RNN over $T$ steps; backprop through resulting deep computation graph.

> **Saying it out loud.** You unroll the recurrence into a deep feed-forward network — one layer per timestep, all sharing weights — and then run ordinary backpropagation through it. So a 100-step sequence is effectively a 100-layer network, and the gradients from every step get summed into the same shared weights. Once you see it that way, everything about RNN training makes sense: the vanishing gradient is just the depth problem, and the memory cost is just storing 100 layers of activations.

**6. Memory cost of BPTT for $T$-step sequence?**
$O(T \cdot \text{hidden size})$ per layer — store all activations.

> **Saying it out loud.** You have to keep every intermediate hidden state around until the backward pass, so memory grows linearly with sequence length times hidden size times layers. For a 1,000-step sequence that's real memory, and it's the practical reason nobody trained RNNs on very long sequences — you ran out of GPU before you ran out of patience. It's the same activation-memory problem transformers have, which is why gradient checkpointing exists in both worlds.

**7. What's truncated BPTT?**
Only backprop $K$ steps; treat earlier as fixed. Saves memory; loses long-range gradient info.

> **Saying it out loud.** You cap how far back you propagate — say 20 or 50 steps — and treat anything older as a constant. The forward pass still carries the hidden state across the boundary, so the model can in principle remember longer than it can learn to remember. That's the exact tradeoff: bounded memory and compute, in exchange for the model being structurally unable to learn a dependency longer than your truncation window. If your task needs 200-step credit assignment and you truncate at 35, you will never learn it, and the loss curve won't tell you why.

**8. Vanishing gradient cause?**
Repeated multiplication by $\partial h/\partial h_{\text{prev}}$ — when spectral radius < 1, product → 0.

> **Saying it out loud.** Going backward, the gradient gets multiplied by the same recurrent Jacobian at every step, so after $T$ steps you've got that matrix to the power $T$. If its largest singular value is even slightly under one — say 0.9 — then after 50 steps you're down by a factor of 200, and after 100 steps there's nothing left. So gradients from distant timesteps are numerically zero and the model simply cannot learn long-range dependencies. Note it doesn't crash or look wrong; it just silently only learns short-range patterns.

**9. Exploding gradient cause?**
Spectral radius > 1 → product blows up.

> **Saying it out loud.** The same compounding, in the other direction. If the recurrent matrix's largest singular value is above one, repeated multiplication grows exponentially, and by step 50 you have a gradient of $10^{20}$ that turns your weights into NaN in a single update. The difference from vanishing is that this one announces itself — loss goes to NaN and you notice immediately — which is precisely why it's the easier problem and why a one-line fix was enough.

**10. Standard fix for explosion?**
Gradient clipping by global norm (typically 1.0).

> **Saying it out loud.** Clip the global gradient norm, usually at 1.0 — if the whole gradient vector's norm exceeds the threshold, rescale it down but keep its direction. Direction is the important word: clipping each component separately would distort where you're heading, while norm clipping just shortens the step. It works because exploding gradients are rare spikes rather than a persistent condition, so you're capping the occasional catastrophe without changing normal training. Everyone still does it, including in transformer pretraining.

**11. Why is $\tanh$ specifically problematic?**
Saturates at $\pm 1$; derivative is at most 1, often much smaller. Multiplied through $T$ steps → vanishes.

> **Saying it out loud.** Because its derivative is at most 1 and is well below that almost everywhere. Once the pre-activation drifts past about 2 in magnitude, tanh saturates and the derivative is near zero, so you're multiplying by tiny numbers at every step going backward. Even in the best case, with a derivative of exactly 1, you're only just breaking even. So tanh guarantees gradient decay unless every unit sits near zero — that's why the LSTM's fix isn't a better activation but an additive path that skips the nonlinearity entirely.

**12. Orthogonal initialization — why?**
Initialize $W_{hh}$ orthogonal so its singular values are exactly 1 (eigenvalues have modulus 1) — gradient neither vanishes nor explodes initially.

> **Saying it out loud.** An orthogonal matrix preserves vector lengths — all its singular values are exactly 1 — so at initialization the gradient neither shrinks nor grows as it passes back through time. You start in the one regime where the compounding is neutral. It genuinely helps, and it's only an initialization: training moves the weights and the property degrades. That's why orthogonal init alone was never enough and gated architectures were needed.

---

## C. LSTM

> **In plain language.** The LSTM adds a second memory vector — the cell state — that gets updated by addition rather than by being rewritten, plus three little sigmoid-valued dials that decide what gets erased, what gets written, and what gets read out. The equations below are those three dials plus the additive update.

**13. LSTM has how many gates?**
Three: forget, input, output.

> **Saying it out loud.** Three: forget, input, output. The way I'd frame it is that the cell state is a whiteboard, and the gates decide what gets erased, what gets written, and what gets read out this step. Each gate is just a sigmoid, so it outputs numbers between zero and one that multiply the thing it's controlling — a soft, differentiable switch. Everything else about the LSTM follows from that picture.

**14. Forget gate formula?**
$f_t = \sigma(W_f [h_{t-1}; x_t] + b_f)$.

> **Saying it out loud.** The forget gate looks at the previous hidden state and the current input, runs them through a sigmoid, and produces a number between zero and one for every dimension of the cell state. One means keep this memory intact, zero means wipe it. It's per-dimension, which is the part people gloss over — the model can hold onto the subject of a sentence in some dimensions while clearing out a finished clause in others. This gate turned out to be the most important of the three; ablations that remove it hurt far more than removing the output gate.

**15. Cell state update?**
$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$.

> **Saying it out loud.** The cell state is the old cell state times the forget gate, plus new candidate content times the input gate. The critical word is plus. Nothing here rewrites the memory through a matrix multiply and a nonlinearity — old memory travels forward by addition, scaled only by a gate. That's what creates a nearly-uninterrupted path across hundreds of timesteps, and it's the same trick a residual connection uses across layers.

**16. Hidden state output?**
$h_t = o_t \odot \tanh(c_t)$.

> **Saying it out loud.** The hidden state is a filtered view of the cell state: squash the cell through tanh, then multiply by the output gate to decide how much to expose. So the LSTM can hold something in memory for a hundred steps without acting on it, and then reveal it exactly when it's relevant. That separation between what you remember and what you currently say is precisely why there are two state vectors instead of one.

**17. Why does cell state fix vanishing gradients?**
Additive update $c_t = f_t \odot c_{t-1} + (\ldots)$. With $f_t \approx 1$, identity-like gradient path. No multiplicative decay.

> **Saying it out loud.** Because the gradient's path backward through the cell state is multiplication by the forget gate, and when the forget gate is near one that's multiplication by one — an identity path. Compare that to the vanilla RNN, where every step multiplies by a weight matrix and a tanh derivative, both of which shrink things. The honest caveat is that it doesn't eliminate the problem, it postpones it: forget gates below one still compound, so LSTMs reliably handle hundreds of steps rather than thousands. It's a much better constant, not a change in the asymptotics.

**18. Standard forget-bias initialization?**
$b_f \approx 1$ (positive). Sigmoid evaluates near 1 → cell state propagates by default.

> **Saying it out loud.** You initialize the forget gate's bias to about 1, so the sigmoid starts near 0.8 and the default behavior is to keep memory rather than erase it. With a zero bias the gate starts near 0.5, meaning the cell state halves every timestep — memory has a half-life of one step and the gradient dies before the model ever learns to open the gate. It's a one-line change worth a large improvement on long-sequence tasks, and it's a lovely example of initialization determining whether a capability is learnable at all.

**19. Why have separate cell and hidden state?**
Cell state: pure long-term memory, additive updates. Hidden state: passed to next layer / output, gated read.

> **Saying it out loud.** They do different jobs. The cell state is protected long-term memory with an additive, largely linear path — you want the gradient to flow through it undisturbed. The hidden state is the working output, gated and squashed, that feeds the next layer and the prediction. If you merged them you'd have to distort your memory every time you wanted to produce an output, which is exactly the compromise the GRU makes — and the GRU gets away with it because the update gate keeps the additive structure.

**20. Connection between LSTM and residual networks?**
Both: additive identity path keeps gradient stable across many "depths" (time steps for LSTM, layers for ResNet).

> **Saying it out loud.** They're the same idea discovered in two different places 18 years apart. An LSTM's cell state carries information forward across time by addition; a ResNet's skip connection carries it forward across depth by addition. In both cases the point is to give the gradient a route that isn't multiplied by a weight matrix at every step. It's a nice thing to say in an interview because it shows you see the pattern rather than memorizing two architectures — and the same pattern shows up again in the transformer's residual stream.

---

## D. GRU

**21. GRU has how many gates?**
Two: update gate $z$ and reset gate $r$.

> **Saying it out loud.** Two — update and reset — versus the LSTM's three, and there's only one state vector instead of two. The update gate does the job of the forget and input gates together, and the reset gate controls how much of the past feeds into the new candidate. Fewer gates means roughly 25 percent fewer parameters and a correspondingly faster step.

**22. GRU vs LSTM — what's combined?**
Forget and input gates merged into single update gate. No separate cell state.

> **Saying it out loud.** The key merge is that GRU ties forgetting and writing together: whatever fraction you keep of the old state, you fill the rest with new content, so the two are forced to sum to one. An LSTM can do both independently — keep everything and also add more, or erase everything and write nothing. That's strictly more expressive, and in practice the constraint rarely costs you anything. The GRU also drops the separate cell state, so there's one vector doing both memory and output.

**23. GRU update formula?**
$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$.

> **Saying it out loud.** It's a convex blend: the new state is $(1-z)$ times the old state plus $z$ times the new candidate. So $z$ near zero means carry the old memory forward untouched, and $z$ near one means overwrite it. That $(1-z)$ path is the GRU's equivalent of the LSTM cell state — a mostly-additive route for the gradient. Same fix, expressed as an interpolation instead of two independent gates.

**24. Reset gate role?**
$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}; x_t])$. Resets memory before computing candidate.

> **Saying it out loud.** The reset gate decides how much of the previous state gets to influence the new candidate content. Set it near zero and the candidate is computed almost entirely from the current input, which is effectively a way to say 'start fresh here' — useful at sentence boundaries or topic shifts. It's a different job from the update gate, which decides how much of the old state survives into the output. One controls what you compute from, the other controls what you keep.

**25. GRU vs LSTM in practice?**
Comparable. GRU faster (fewer params); LSTM slightly more expressive. Empirical results mixed.

> **Saying it out loud.** Honestly, it's a coin flip, and there's a well-known Greff and Schmidhuber study that ran thousands of variants and found no consistent winner. GRUs train a bit faster and use fewer parameters; LSTMs occasionally edge ahead on tasks needing very long memory, where independent forget and input control helps. The practical advice is to pick GRU if you're compute-constrained, LSTM if you're not, and spend your effort on data instead — the gap between them is far smaller than the gap either has to a transformer.

---

## E. Bidirectional + seq2seq

**26. Bidirectional RNN?**
Forward + backward RNN; concatenate hidden states. Captures both past and future context.

> **Saying it out loud.** You run two independent RNNs, one left to right and one right to left, and concatenate their hidden states at each position. So every position gets a representation informed by the entire sequence, not just its past. It's a real quality win for tagging and classification, and it costs you twice the compute plus the ability to stream — you can't emit anything until the whole input has arrived.

**27. Why not bidirectional for generation?**
Future tokens don't exist at generation time. BiRNN only for tasks with full sequence available (NER, POS, classification).

> **Saying it out loud.** Because the future doesn't exist yet. When you're generating token by token, there's nothing to the right to encode, so the backward pass has no input. It's the same reason transformers use a causal mask for language modeling and BERT can't generate. The rule of thumb is simple: bidirectional whenever you have the full sequence up front, unidirectional whenever you're producing it.

**28. Seq2seq architecture?**
Encoder RNN reads source; passes final hidden state to decoder RNN that generates target autoregressively.

> **Saying it out loud.** One RNN reads the source sentence and compresses it into its final hidden state; a second RNN starts from that state and generates the translation. It was a genuinely big idea in 2014 — end-to-end translation with no alignment machinery or phrase tables. And it has an obvious flaw sitting right in the middle: the entire meaning of the source has to fit into one fixed-size vector, which is what attention was invented to fix.

**29. Bottleneck problem in seq2seq?**
Encoder compresses entire source into one fixed vector. Hard for long sentences.

> **Saying it out loud.** Everything the decoder knows about the source has to squeeze through one vector, typically 500 or 1,000 numbers, no matter whether the sentence is five words or fifty. So translation quality falls off sharply with source length — the original papers show BLEU degrading badly past about 30 words. That's not a training problem you can fix with more data; it's an information bottleneck in the architecture, which is why the fix had to be architectural.

**30. Bahdanau attention idea?**
At each decoder step, compute weighted average of *all* encoder hidden states. Decoder reads from source dynamically.

> **Saying it out loud.** Instead of forcing the decoder to work from one summary vector, let it look back at every encoder state and take a weighted average, with the weights recomputed at each output step. So when generating the third word of a translation, the decoder can focus on whatever part of the source is relevant right now. That removes the fixed-size bottleneck entirely, and the length-degradation curve flattens out. It's also the direct ancestor of transformer attention — the 2017 paper's contribution was throwing away the RNN and keeping only this.

**31. Attention scoring functions?**
Bahdanau: $v^\top \tanh(W_1 h^{\mathrm{dec}} + W_2 h^{\mathrm{enc}})$. Luong: $h^{\mathrm{dec} \top} h^{\mathrm{enc}}$ (dot product) or $h^{\mathrm{dec} \top} W h^{\mathrm{enc}}$ (general).

> **Saying it out loud.** There are two families. Bahdanau's additive version feeds the decoder and encoder states through a small neural net with a tanh, which is more expressive and slower. Luong's multiplicative version just takes a dot product, optionally with a learned matrix in the middle. The dot product won because it's a single matrix multiply that maps perfectly onto GPU hardware — same quality, much better throughput. That's exactly the scaled dot-product attention transformers use, with the scaling by $\sqrt{d}$ added to stop the logits growing with dimension.

---

## F. Transformer transition

**32. Why are transformers parallelizable but RNNs aren't?**
RNN: $h_t$ depends on $h_{t-1}$ — sequential. Transformer: attention over all positions independent of order — parallel matmul.

> **Saying it out loud.** Because an RNN's state at step $t$ literally requires the state at step $t-1$, so training a 1,000-token sequence means 1,000 dependent steps no matter how many GPUs you have. Attention computes all positions at once as one big matrix multiply, so the whole sequence is a handful of kernel launches. That's the difference between an architecture that can absorb a datacenter and one that can't, and it's the real reason transformers won — not that they're smarter per parameter, but that you can actually train them at scale.

**33. Long-range dependency comparison?**
RNN: signal must traverse $O(T)$ steps. LSTM helps but still degrades over long range. Transformer: any pair of positions $O(1)$ steps apart.

> **Saying it out loud.** In an RNN, information from token one reaches token five hundred by passing through 499 sequential updates, each one an opportunity to be overwritten or attenuated. In a transformer it's one attention hop — every position can look at every other position directly. That's the difference between path length $O(T)$ and $O(1)$, and it's why transformers handle long-range agreement and coreference so much better. LSTMs stretched the usable range from tens of steps to hundreds, but they didn't change the fact that it's a path.

**34. Scaling behavior of LSTM vs transformer?**
Transformer scales better. LSTMs plateau in performance with more compute; transformers keep improving (scaling laws).

> **Saying it out loud.** Both improve with scale; the transformer's curve is steeper and doesn't flatten. LSTMs hit a point where more parameters and more data stop buying much, partly because the sequential bottleneck limits how much data you can push through and partly because the fixed-size state limits what a bigger model can use. Transformers follow clean power laws over many orders of magnitude, which is what makes multi-million-dollar training runs a predictable investment rather than a gamble. The economics of scaling, more than any single benchmark, is what ended the RNN era.

**35. When still use LSTM today?**
Streaming/online tasks where causal sequential is natural. Tiny tasks where transformer overhead isn't worth it. Some signal processing / low-latency speech.

> **Saying it out loud.** When the sequence is unbounded and latency matters — streaming speech, real-time control, sensor data on an embedded device. An LSTM processes one step in constant time and constant memory, while a transformer's cost grows with how much context it's holding. There's also the small-data case: with a few thousand training examples an LSTM's stronger inductive bias can beat a transformer that has nothing to learn its structure from. So the honest answer is streaming, tiny devices, and small datasets — everything else has moved on.

---

## G. Modern context

**36. Mamba vs LSTM — what's similar?**
Both are recurrent: state evolves with each input. Both have linear complexity in sequence length.

> **Saying it out loud.** Both keep a fixed-size state that they update as they walk the sequence, and both are therefore linear in sequence length with constant memory at inference. If you squint, Mamba is an RNN. That's the setup for the more interesting question, which is why one of them scales and the other doesn't.

**37. Mamba vs LSTM — what's different?**
Mamba: linear recurrence with carefully chosen $A$ (HiPPO-inspired or selective); parallel scan for training; no $\tanh$.

> **Saying it out loud.** The recurrence is linear — no tanh, no nonlinearity between steps — and that's what changes everything. Because the update is linear, composing two of them gives another one of the same form, which means you can compute the whole sequence with a parallel scan in logarithmic depth instead of walking it step by step. Mamba also uses a structured, principled initialization for the state matrix rather than a random one, so memory is stable over thousands of steps. Same shape of equation, completely different training story.

**38. Why couldn't RNNs do what Mamba does in 1997?**
Parallel scan algorithm wasn't connected to RNN training; HiPPO theory wasn't developed. Modern SSMs are "what RNNs should have been."

> **Saying it out loud.** Partly theory, partly hardware. Nobody had connected parallel scan to sequence-model training, and the HiPPO theory for how to initialize a recurrence so it genuinely remembers didn't exist until 2020. But mostly there were no GPUs where making the recurrence parallelizable would have been worth anything — in 1997 you were training on a CPU and the sequential bottleneck wasn't the binding constraint. The lesson worth stating is that architectures win when they match the hardware of their era, which is the same reason attention beat convolution for sequences.

**39. Catastrophic forgetting in RNNs?**
Adding capacity for new task overwrites old. RNNs especially vulnerable due to shared parameter across all positions/tasks.

> **Saying it out loud.** Train on a new task and the weights that encoded the old one get overwritten, because gradient descent has no reason to preserve them. RNNs are especially exposed since one shared weight matrix handles every position and every task, so there's nowhere for the old skill to hide. It's not unique to RNNs — every neural network does it — but the parameter sharing concentrates the damage. The mitigations are the usual ones: replay old data, penalize movement in important weights as in EWC, or give new tasks their own parameters.

**40. Why did Karpathy's "Unreasonable Effectiveness of RNNs" 2015 hold but not in 2024?**
Transformer + scale destroyed the RNN advantage. RNNs are still effective but not state of the art for any flagship NLP task.

> **Saying it out loud.** Because in 2015 an RNN really was the best way to turn a pile of text into a model that generated plausible text, and that was genuinely surprising at the time. What changed isn't that RNNs got worse — it's that transformers plus scale opened a gap that no amount of RNN tuning closes, mostly because you can throw a thousand GPUs at attention and you can't at a recurrence. The post is still right about the core claim, that sequence prediction on raw characters produces structure nobody programmed in. It just turned out the ceiling was set by how much compute you could pour in, and that's the axis RNNs lose on.

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
