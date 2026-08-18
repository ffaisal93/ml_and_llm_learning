# Positional Embeddings — Interview Grill

> 40 questions on positional encoding. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. Why do transformers need positional encoding?**
Pure attention is permutation-equivariant: shuffle tokens and the output shuffles the same way. Attention has no innate notion of order. Positional encoding is the only mechanism by which transformers know what comes first.

> **Saying it out loud.** Attention is a weighted average, and averages don't care about order. So if you feed the model a shuffled sentence, every token still sees the same bag of other tokens and you get the same answers back, just rearranged. Positional encoding is the one place order gets injected into the whole architecture. The failure mode if you skip it is total for language: the model can't tell subject from object, can't count, can't copy in order.

**2. State permutation equivariance formally.**
For any permutation matrix $P$: $\mathrm{Attention}(P \cdot X) = P \cdot \mathrm{Attention}(X)$. This means the attention output depends only on the multiset of input tokens, not their order. Adding positional information breaks this.

> **Saying it out loud.** Formally you say: attention of $P X$ equals $P$ times attention of $X$ for any permutation matrix $P$. In plain English, permuting the inputs just permutes the outputs — the network's function only depends on the multiset of tokens, not the sequence. That's the property positional encoding is designed to break. Worth naming precisely, because interviewers distinguish equivariance from invariance and the sloppy answer is to say "invariant."

**3. What are the main families of positional encoding?**
Absolute (sinusoidal, learned), relative (T5 bias, Transformer-XL), rotary (RoPE), bias-based (ALiBi), and none (NoPE).

> **Saying it out loud.** There are basically five buckets. Absolute, where every position gets its own code — that's sinusoidal and learned embeddings. Relative, where only the gap between two tokens matters — T5 bias, Transformer-XL. Rotary, which is RoPE, and which is clever because it encodes absolute positions but the dot product comes out relative. Bias-based, which is ALiBi. And none at all, NoPE, which only works for causal models. Modern production is RoPE with ALiBi as the main alternative.

---

## B. Sinusoidal

**4. Walk me through sinusoidal positional encoding.**
For each position $t$ and dimension $2i$ (even) or $2i+1$ (odd):

$$
\mathrm{PE}(t, 2i) = \sin(t / 10000^{2i/d}), \qquad \mathrm{PE}(t, 2i+1) = \cos(t / 10000^{2i/d})
$$

Add $\mathrm{PE}(t)$ to the token embedding. Different dimensions oscillate at exponentially different frequencies, giving each position a unique signature.

> **Saying it out loud.** You build a fixed table of sine and cosine values — no learning involved — and add it to the token embedding. Even dimensions get a sine, odd dimensions get a cosine, and the frequency drops off geometrically as you move up the dimensions. Think of a rack of clocks ticking at speeds from once every few tokens to once every ten thousand, so each position has a unique combination of hand angles. The point is that it's deterministic, so it's defined at any position, even ones you never trained on.

**5. Why exponentially-spaced frequencies?**
Spans many orders of magnitude (low frequencies for global structure, high frequencies for local). The base 10000 is empirical; not deeply principled. Could be 1000 or 100000 with similar results.

> **Saying it out loud.** Because you want to cover both local and global structure at once. A high frequency lets the model tell position 5 from position 6; a very low frequency lets it tell the beginning of the document from the end. If all frequencies were similar you'd only get one scale of resolution. The base 10000 is an empirical pick, not derived — 1000 or 100000 works about as well, which is a good thing to say if someone asks why that number.

**6. Why does sinusoidal in theory enable extrapolation?**
The encoding is defined for any position $t$, including beyond training length. Plus the theoretical property: for any $\Delta t$, there exists a fixed linear transform $M_{\Delta t}$ such that $\mathrm{PE}(t + \Delta t) = M_{\Delta t} \cdot \mathrm{PE}(t)$, so relative positions can be computed by linear operations on absolute encodings.

> **Saying it out loud.** Because it's a formula rather than a lookup table, so position 100000 has a perfectly well-defined encoding even if you trained at 512. On top of that there's the nice property that shifting by a fixed offset is a fixed linear map: $\mathrm{PE}(t + \Delta t) = M_{\Delta t} \cdot \mathrm{PE}(t)$. So relative offsets are in principle recoverable by a linear layer. The word to stress is *in principle* — nothing forces the model to actually learn that.

**7. Why does sinusoidal extrapolation fail in practice?**
The encoding is well-defined at long range, but the **learned weights** that work with it are trained only on positions seen in training. The model's attention patterns at position 5000 (when trained at 1024) are unreliable.

> **Saying it out loud.** Because being defined isn't the same as being trained. The encoding at position 5000 is a valid vector, but every weight that reads positional information was only ever fit on positions up to the training length, so at 5000 the attention patterns are just out of distribution. It's the classic gap between the input being well-defined and the function being reliable there. In practice sinusoidal degrades quickly past training length, which is exactly what motivated ALiBi and RoPE extension methods.

**8. What replaced sinusoidal?**
Learned positional embeddings (BERT, GPT-2/3) for simplicity, then RoPE for relative-position handling and better extrapolation.

> **Saying it out loud.** First learned embeddings — BERT, GPT-2, GPT-3 all just learned a vector per position because it was simpler and slightly better in range. Then RoPE took over from 2021 onward because it gives relative position inside attention and extends much further. Sinusoidal is now essentially historical. If you're asked what a modern model uses, the answer is RoPE, and the reason is length flexibility.

---

## C. Learned positional embeddings

**9. What's a learned positional embedding?**
A $\text{max-position} \times d$ matrix; the $t$-th row is the position embedding for position $t$. Added to token embeddings: $\text{input}_t = \text{embedding}(\text{token}_t) + \text{position-embedding}[t]$. Used in BERT, GPT-2, GPT-3.

> **Saying it out loud.** It's literally an embedding table indexed by position instead of by token — a max-position by $d$ matrix, and you add row $t$ to the token embedding at slot $t$. No formula, no structure, the model figures out what each position should mean. BERT, GPT-2 and GPT-3 all did this. It's the simplest possible answer to "how do I tell the model where a token is."

**10. Pros of learned positional embeddings?**
Simple. Empirically strong within training range. No hand-designed function.

> **Saying it out loud.** It's simple and it works. You don't have to design a function or justify a frequency schedule; the model learns whatever positional structure helps. And empirically, inside the training range, it matched or beat sinusoidal, which is why the big early models used it. The catch is entirely about what happens outside that range.

**11. Cons of learned positional embeddings?**
Hard cap at `max_position`. No extrapolation possible. Position embeddings near `max_position` are noisier than near 0 (less training data for those positions).

> **Saying it out loud.** The hard cap is the big one — the table has a fixed number of rows, so there is literally no vector for position max-position plus one. You also get a data-imbalance problem: almost every training document reaches position 10, but very few reach position 1023, so the late rows are undertrained and noisier. And it's absolute, not relative, so the model has to learn distance relationships from scratch rather than getting them for free.

**12. Why did learned positional embeddings lose to RoPE?**
The hard cap on context length. Modern users want flexible context lengths, often longer than training. Learned PE cannot extend beyond training length without retraining.

> **Saying it out loud.** Because of the cap. Everybody wants longer context now, and with a learned table the only way to go from 2K to 32K is to retrain, which for a 70B model is not a thing you do casually. RoPE lets you train at 4K and stretch at inference with YaRN. That's the tradeoff sentence: learned embeddings buy you a little in-range quality and cost you all your length flexibility.

---

## D. RoPE

**13. Walk me through RoPE.**
For each pair of dimensions $(d_{2k}, d_{2k+1})$, treat as a 2D vector and rotate by angle $t \cdot \theta_k$ where $t$ is position and $\theta_k = 10000^{-2k/d}$. Apply this rotation to $Q$ and $K$ (not $V$) before computing attention scores.

> **Saying it out loud.** Take a query vector, chop it into pairs of adjacent dimensions, and treat each pair as a point in a 2D plane. Then rotate each pair by an angle equal to the token's position times a per-pair frequency $\theta_k = 10000^{-2k/d}$ — fast rotation for the early pairs, very slow for the late ones. Do the same to the keys. Leave the values alone. That's it, and the magic is what happens when you take the dot product.

**14. Why does the dot product of rotated Q and K depend only on relative position?**
For $Q$ at position $m$ and $K$ at position $n$:

$$
[R(m\theta)\, q]^\top [R(n\theta)\, k] = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R((n-m)\theta)\, k
$$

The rotation matrices' product simplifies to $R((n-m)\theta)$ — a rotation by the difference. So the dot product depends only on $n - m$, not absolute $m$ or $n$.

> **Saying it out loud.** Because rotations compose by adding angles. When you dot a query rotated by $m\theta$ against a key rotated by $n\theta$, the transpose of one rotation times the other collapses to a single rotation by $(n-m)\theta$. So the absolute positions cancel and only the gap survives. That's the whole beauty of RoPE — you apply absolute position but attention only ever feels relative position, with no extra parameters.

**15. Why isn't V rotated in RoPE?**
$V$ carries content (the actual information being mixed via attention weights). Rotating $V$ would entangle position with content. Rotating only $Q$ and $K$ cleanly separates position (in attention scores) from content (in value mixing).

> **Saying it out loud.** Because $V$ is the payload, not the address. $Q$ and $K$ only exist to decide how much attention flows where, so it's fine, even desirable, for position to live there. $V$ is the actual content being mixed into the residual stream, and rotating it would make the retrieved information depend on where the token sat. Keeping $V$ clean is what separates "where" from "what" — that's the phrase to use.

**16. What's the complex-number interpretation of RoPE?**
View each pair $(q_{2k}, q_{2k+1})$ as a complex number. Multiplication by $e^{i \cdot t \cdot \theta_k}$ rotates by $t \cdot \theta_k$. The attention dot product becomes the real part of $q^* \cdot k$, which depends on the relative angle.

> **Saying it out loud.** Take each pair of dimensions and call it a complex number, real part and imaginary part. Then rotating by an angle is just multiplying by $e^{i t \theta_k}$, and the dot product you care about is the real part of $\tilde q^* \tilde k$, which only depends on the difference of the phases. It's the same math in fewer symbols, and it's how the efficient kernels are written. Bring it up as a second framing if the interviewer wants to see depth.

**17. Why does RoPE outperform sinusoidal in practice?**
(a) Applied at every layer's attention, not just to inputs — stronger positional signal throughout. (b) Relative position by construction — the right inductive bias. (c) Better empirical extrapolation, especially with NTK/YaRN.

> **Saying it out loud.** Three reasons, and I'd say them in this order. One, RoPE lives inside the attention computation at every layer, whereas sinusoidal gets added once at the input and dilutes as it propagates. Two, RoPE is relative by construction rather than by hope — the dot product mathematically depends only on $n - m$. Three, it extrapolates far better, especially once you add NTK scaling or YaRN, which buys you around 16x training length.

**18. Where is RoPE used in production?**
LLaMA, LLaMA-2, LLaMA-3, Mistral, Mixtral, Qwen, Gemma, Gemma 2, Falcon (some variants), GPT-J, GPT-NeoX. Effectively the modern standard for decoder-only LLMs.

> **Saying it out loud.** Essentially the whole modern decoder-only ecosystem: LLaMA 1, 2 and 3, Mistral and Mixtral, Qwen, Gemma, GPT-J, GPT-NeoX, and some Falcon variants. If someone hands you an open-weights model today, RoPE is the safe default guess. The main exceptions are the ALiBi models — BLOOM and MPT.

---

## E. RoPE extension (NTK, YaRN)

**19. Why doesn't RoPE extrapolate naively?**
High-frequency components ($\theta_k$ for small $k$) cycle quickly, so positions beyond training have "phase configurations" the model never saw. The model can't generalize to those configurations.

> **Saying it out loud.** Because of phase wraparound in the fast frequencies. The early dimension pairs rotate quickly, so past the training length they've gone around the circle many times, and the model sees angle combinations it has never encountered. The slow frequencies are fine — they haven't even completed one turn — but the fast ones are effectively garbage. That's why every extension technique treats high and low frequencies differently.

**20. What's NTK-aware scaling?**
Scale RoPE's base frequency to compress frequencies into a wider range. Effectively interpolates between trained frequencies, allowing longer context. Free at inference time. Up to ~4× extension with mild quality loss.

> **Saying it out loud.** You change the RoPE base so all the rotation frequencies get compressed, which squeezes a longer sequence back into the angle range the model already knows. It's a one-line change at inference time — no retraining, no extra parameters, just a different base constant. It buys you roughly 4x context with mild quality loss. The name comes from neural-tangent-kernel intuition about which frequencies you can safely stretch.

**21. What's YaRN?**
Combines per-frequency interpolation (high frequencies fully interpolated, low frequencies untouched) with attention scaling adjustment. Extends context up to ~16× training length with minimal quality loss. Used in several recent open models.

> **Saying it out loud.** YaRN is the refined version. Instead of scaling all frequencies uniformly, it fully interpolates the high frequencies, which are the ones that wrap around, leaves the low frequencies alone, since they extrapolate fine, and then rescales the attention temperature to compensate for the softened score distribution. That combination gets you to about 16x training length with very little degradation, which is why several recent open models ship with it.

**22. What's linear positional interpolation (Chen et al. 2023)?**
Rescale positions: divide by $L_{\text{test}} / L_{\text{train}}$ so the effective range matches training. Simple. Loses precision at high frequencies. Good for ~4× extension.

> **Saying it out loud.** It's the blunt version: divide every position index by the extension factor, so if you trained at 4K and want 16K, position 12000 gets treated as position 3000. Everything now sits inside the trained angle range. It works and it's trivially simple, but you lose resolution — nearby tokens become harder to distinguish because they're squeezed together at high frequencies. Good to about 4x; beyond that YaRN is the better call.

**23. Why does context extension matter for production?**
Training a 70B model from scratch at 128K context is infeasibly expensive. Extension methods let you train at 4K–32K and serve at 128K+ with mild quality degradation. Critical for cost-effective long-context serving.

> **Saying it out loud.** Because context length is a headline product feature and pretraining is where the money goes. Training a 70B model at 128K attention from scratch is quadratically painful and not something you'd do just to advertise a bigger number. Extension methods let you pretrain at a cheap length and then stretch at serving time, usually with a short fine-tune to clean things up. The tradeoff you name is cost versus a few points of long-context quality.

**24. What's LongRoPE?**
Microsoft's search-based approach to RoPE frequency scaling for very long context (millions of tokens). More expensive to set up than YaRN but reportedly better quality at extreme lengths.

> **Saying it out loud.** Microsoft's approach where, instead of picking a scaling formula, you run an evolutionary search over per-dimension rescaling factors to find the best one for your model. It's more setup work than YaRN because you have to actually run the search, but it reportedly holds quality out to extreme lengths — the million-token-context claims lean on it. Mention it as the state of the art in extreme extension.

---

## F. ALiBi

**25. Walk me through ALiBi.**
Add a linear bias to attention scores penalizing distant positions:

$$
\text{scores}[i, j] \mathrel{+}= -m_h \cdot |i - j|
$$

where $m_h$ is a head-specific slope. No positional embeddings needed; the bias provides position information.

> **Saying it out loud.** ALiBi is the no-embedding approach: you just subtract something proportional to distance from every attention score, so far-apart tokens get penalised. The slope is per-head and fixed by formula, not learned, so it costs zero parameters. Because a straight line is defined at every distance, it extrapolates for free — the paper's title is literally "train short, test long." The cost is that a single linear penalty is a blunter tool than RoPE's per-frequency structure.

**26. How are ALiBi slopes chosen?**
Press et al. propose $m_h = 2^{-8h/H}$ for head $h$ of $H$. Geometric range from $2^{-8/H}$ (small slope, attends far) to $2^{-8}$ (large slope, attends close). Different heads naturally specialize for different ranges.

> **Saying it out loud.** Geometrically. Head $h$ out of $H$ gets slope $2^{-8h/H}$, so across the heads you sweep from a very gentle penalty to a very steep one. The gentle heads can look across the whole sequence, the steep ones effectively become local windows. That's the design intent — you get a built-in mixture of attention ranges rather than having to learn one.

**27. ALiBi pros/cons vs RoPE?**
Pros: simpler (no rotations), extrapolates trivially (bias is well-defined at any distance), no need for context extension techniques. Cons: empirically slightly weaker than RoPE at large scales, less expressive (a single bias per relative offset vs RoPE's frequency decomposition).

> **Saying it out loud.** ALiBi's win is simplicity and free extrapolation: no rotations, no NTK, no YaRN, the bias is defined at any distance so you can just run longer. RoPE's win is expressiveness — it decomposes position across many frequencies rather than collapsing it into one scalar per distance — and at large scale that shows up as measurably better quality. The industry picked expressiveness plus extension tricks over free extrapolation, which is why RoPE dominates.

**28. Where is ALiBi used?**
BLOOM, MPT, some Falcon variants. Its popularity declined as RoPE became dominant.

> **Saying it out loud.** BLOOM and MPT are the two you should name, plus some Falcon variants. Both are 2022-era models. It hasn't been picked up much since, mostly because RoPE plus YaRN solved the long-context problem that ALiBi was invented to solve.

---

## G. T5 relative bias

**29. What's T5-style relative position bias?**
Add a learned bias to attention scores based on bucketed relative offset:

$$
\text{scores}[i, j] \mathrel{+}= b(\text{bucket}(i - j))
$$

The buckets are typically log-spaced: small offsets get individual buckets; large offsets get coarser bins.

> **Saying it out loud.** You add a learned scalar to the attention score depending on how far apart the two tokens are, but you bucket the distances — small offsets get their own bucket, large ones get log-spaced bins. So you learn maybe 32 numbers per head instead of one per possible distance. It's fully relative, no absolute positions anywhere, and the position information goes straight into attention rather than into the embeddings.

**30. Pros/cons of T5 relative bias?**
Pros: Truly relative. Can extrapolate to longer lengths if bucketing is sensible. Cons: Adds parameters per (head, bucket). Less expressive than RoPE for certain pattern types.

> **Saying it out loud.** The pro is that it's honestly relative and can stretch to longer sequences as long as the bucketing degrades gracefully. The cons are that it adds parameters per head per bucket, the bucketing scheme is hand-designed and arbitrary, and it's less expressive than RoPE because a scalar per bucket can't represent the kind of periodic structure a frequency decomposition can. Net: fine, but not free and not best.

**31. Why isn't it more popular?**
Mostly superseded by RoPE for decoder-only models. Still used in T5, Flan-T5, and some encoder-decoder variants.

> **Saying it out loud.** Because RoPE arrived and gave you the same relative-position property with no extra parameters and better extension behaviour. T5 bias survives in the T5 and Flan-T5 family and some encoder-decoder work, but nobody starts a new decoder-only model with it. That's really the whole story — it was superseded rather than disproven.

---

## H. NoPE and edge cases

**32. What's NoPE?**
No positional encoding at all. Just rely on the causal mask to break permutation invariance.

> **Saying it out loud.** It means you literally add no positional signal at all: no sinusoids, no learned table, no rotations, no bias. You feed the token embeddings in and rely on the causal mask alone to break the symmetry. It sounds like it shouldn't work, and for encoders it doesn't — but for decoder-only models it sometimes does.

**33. Why can NoPE work for causal LMs?**
The causal mask itself breaks permutation invariance: position $i$ can only see positions $\leq i$, so the *role* of each position differs (first token has no context; last has full context). This asymmetry provides some implicit position signal.

> **Saying it out loud.** Because the causal mask already makes positions non-interchangeable. Token 1 can only see itself, token 500 can see five hundred things, so the size and composition of each token's visible context is different — that asymmetry is itself a position signal, and the model can decode it through depth. Kazemnejad et al. actually found NoPE generalised to unseen lengths *better* than RoPE or ALiBi on their benchmarks. The caveat is that it hasn't transferred to flagship-scale pretraining.

**34. Why doesn't NoPE work for encoder LMs?**
Encoder LMs (bidirectional) have no causal mask; tokens see each other in both directions. Without explicit position, true permutation invariance returns. NoPE is specifically a causal-LM phenomenon.

> **Saying it out loud.** Because without a causal mask every token sees every other token, so nothing distinguishes the positions at all — you're back to exact permutation equivariance and the model can only see a bag of words. The causal mask is doing all the work in NoPE, so remove it and the whole thing collapses. That's the crisp way to say it: NoPE isn't "no position," it's "position from the mask."

**35. NoPE vs RoPE in practice?**
NoPE works comparably at moderate scales for causal LMs. At large scales and long contexts, RoPE generally wins. NoPE is more of a research curiosity than a production technique.

> **Saying it out loud.** On length generalisation benchmarks NoPE looks surprisingly good, sometimes better than RoPE. But at production scale and on real long-context quality, RoPE wins, and there's no flagship model shipping without positional encoding. So the safe framing is: theoretically interesting, tells you something real about where position information comes from, not a production choice. Saying that shows you know the result without overclaiming it.

---

## I. Conceptual gotchas

**36. What's the difference between absolute and relative positional encoding?**
Absolute: each position has a unique fixed encoding (sinusoidal, learned). Relative: only position differences matter (T5 bias, RoPE). Modern LLMs prefer relative because it generalizes better.

> **Saying it out loud.** Absolute means each position gets its own code, so the model knows this is token 17. Relative means only the gap matters, so the model knows this token is 3 back from that one. Relative generalises better because language patterns are mostly about local distance, not absolute index, and because a relative scheme naturally handles lengths it hasn't seen. RoPE is the sneaky one — it applies absolute rotations but the attention dot product only ever depends on the difference.

**37. Can you mix two types of positional encodings?**
You can, but rarely useful. Adding both sinusoidal and learned doubles the position information; mostly redundant. Some research mixes RoPE with global tokens that don't get rotated, but these are special cases.

> **Saying it out loud.** You can, and people occasionally do, but it's usually redundant — two encodings of the same information just means the model learns to ignore one. The cases where mixing genuinely helps are structural, like giving global or special tokens a position-free treatment so they can be attended from anywhere. If asked, say you'd default to one scheme and only add a second if you have a specific structural reason.

**38. What's xPos?**
RoPE + exponential decay on long-range attention. Better extrapolation at slight quality cost. Used in some research models; not mainstream.

> **Saying it out loud.** xPos is RoPE plus an exponential decay factor that shrinks the contribution of distant positions, so long-range attention fades out smoothly instead of oscillating. That damping is what makes it extrapolate better than plain RoPE. The price is a small quality hit at the length you actually trained on, which is why it stayed in research models rather than going mainstream.

**39. How does positional encoding interact with sparse attention?**
For sliding window: position information must work within the window. RoPE works fine because relative offsets within a window are small. For global tokens (Longformer), you may need special position handling (no positional encoding for `[CLS]`, etc.).

> **Saying it out loud.** For sliding-window attention it's basically fine — every relative offset inside the window is small, well inside the trained range, so RoPE just works. The trouble is global tokens: a token that everyone attends to from any distance doesn't have a meaningful relative position, so models like Longformer treat those specially, often by exempting them from positional treatment. The failure mode to name is applying a big relative offset to a global token and getting it penalised out of the picture.

---

## J. Quick fire

**40.** *Original positional encoding paper?* Vaswani et al. 2017.
**41.** *RoPE paper?* Su et al. 2021.
**42.** *ALiBi paper?* Press et al. 2021.
**43.** *YaRN paper?* Peng et al. 2023.
**44.** *RoPE base frequency?* $10000^{-2i/d}$.
**45.** *Default ALiBi slope?* $2^{-8h/H}$ for head $h$ of $H$.

---

## Self-grading

If you can't answer 1-10, you don't know positional encodings. If you can't answer 11-25, you'll struggle on architecture deep-dives. If you can't answer 26-40, frontier-lab interviews will go past you.

Aim for 30+/40 cold.
