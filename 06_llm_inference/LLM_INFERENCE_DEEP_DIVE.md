# LLM Inference: From First Principles to the Frontier

> **What this is.** A teaching document about how large language models actually run in production, written so that you can read it once and understand it, and read it twice and *say* it. Every major section is built the same way: an intuition, then a concrete number, then the formula, then what the number means for a system you might have to design. Each section also ends with two short blocks — one telling you what an interviewer is really probing when they ask, and one giving you the words to say out loud.
>
> **Who it's for.** Someone with a solid ML background — you know what a transformer is, you know what attention computes — who now has to answer questions like "why is decode memory-bound," "how many GPUs do I need for 20 requests per second," and "explain PagedAttention" in a room with a whiteboard and a skeptical staff engineer.
>
> **A promise about numbers.** Every arithmetic result in this document was computed, not remembered. Where a number comes from a vendor datasheet, I say so. Where it comes from a paper, I cite the paper. Where it depends on your hardware, your framework version, or your workload, I mark it as something to verify rather than assert. Section 17 collects all of this in one place.

---

## 0. How to read this

Read sections 1 through 4 in order. They are the scaffold — a walkthrough of a single request, a plain-language account of what the GPU is doing, the prefill/decode split, and the roofline argument. Almost every optimization in the rest of the document is a response to something in those four sections, and if you have them cold, the rest becomes obvious rather than memorized.

After that you can jump around. But I'd suggest KV cache (5) before batching (7), and batching before PagedAttention (8), because each one is the reason the next one exists.

Section 15 is a full capacity-planning exercise worked end to end with real arithmetic. If you can reproduce that on a whiteboard, you are in the top decile of candidates for an inference-adjacent role. It is the single highest-leverage thing in here.

---

## 1. One request, end to end

Before any optimization makes sense, you need a picture of what happens when a request arrives. Let's narrate one, concretely, on a system that looks like vLLM or TensorRT-LLM or SGLang — they differ in the details but the shape is the same.

Say you POST to `/v1/chat/completions` with a 6,000-token prompt and ask for up to 500 tokens of output, streaming.

**Step 1 — HTTP arrives, text becomes integers.** The server's web layer (typically FastAPI or a Rust equivalent) parses your JSON. Your messages get rendered into a single flat string by a *chat template* — a little Jinja-ish program that inserts the special tokens the model was trained with, things like `<|start_header_id|>user<|end_header_id|>`. That string goes through the tokenizer, a byte-pair-encoding model that turns text into a list of integers. Your 6,000-token prompt is now a Python list of 6,000 `int32` values. Nothing has touched the GPU yet, and this step costs single-digit milliseconds.

**Step 2 — The request joins a queue.** The server does *not* immediately run your prompt. It hands the token list to a **scheduler**, which is the real brain of an inference server. The scheduler holds a waiting queue and a running set, and once per iteration it decides who gets GPU time. Before it admits you, it asks a question that will dominate the rest of this document: *do I have enough free KV-cache memory to hold this sequence as it grows?* If not, you wait. This is why an overloaded LLM server shows up as queueing delay rather than slow tokens.

**Step 3 — Prefill.** You get admitted. The server copies your 6,000 token IDs to GPU memory and runs one forward pass over all 6,000 positions at once. This is a genuinely large computation — we'll compute it exactly in section 15, but it is roughly $8.5 \times 10^{14}$ floating-point operations, and on four H100s at realistic efficiency it takes about **450 ms**. Two things come out of it. First, a probability distribution over the vocabulary for position 6,000, which is where your first output token comes from. Second, and more importantly for everything that follows, the **KV cache**: for each of the model's 80 layers, the key and value vectors for all 6,000 positions, written into a big preallocated GPU buffer. That buffer is 1.83 GiB for this one request. Hold that number.

**Step 4 — First token, and the stream opens.** The logits for the last position get turned into an actual token ID by the sampler (section 6). That ID goes back through the tokenizer's decoder into a text fragment, gets wrapped in a server-sent-event frame, and lands in your terminal. The clock time from step 1 to here is your **TTFT**, time to first token. For this request it's about half a second, and essentially all of it was step 3.

**Step 5 — Decode, 499 more times.** Now the loop. Take the token you just produced. Embed it — one row of the embedding table. Run it through all 80 layers. In each layer's attention, this single token produces one query vector, and that query attends against *every key in the cache*, all 6,001 of them, then combines the corresponding values. Its own new key and value get appended to the cache, which is now 6,001 tokens long. The feed-forward network runs on one token's worth of activations. At the top, logits, sampler, token, stream it out, repeat.

Each pass through this loop takes about **30 ms** in the configuration we'll build in section 15. That number — the average gap between consecutive output tokens — is your **TPOT** (time per output token, also called inter-token latency, ITL). Notice how strange the economics are: the prefill pass processed 6,000 tokens in 450 ms, which is 0.075 ms per token. Decode processes one token in 30 ms. Per token, decode is *400 times less efficient*. Understanding why is the whole game.

**Step 6 — Termination and cleanup.** The loop stops when the model emits an end-of-sequence token, or you hit `max_tokens`, or a stop string matches, or the client disconnects. The scheduler frees your KV blocks back to the pool, and — this is the part people forget — that freed memory is what lets the *next* queued request get admitted. End-to-end you waited $450 + 500 \times 30 = 15{,}450$ ms, about 15.5 seconds, of which 97% was decode.

**The one thing to notice.** You were never alone on that GPU. At every decode step, your single token was being computed alongside one token from each of roughly 80 other users, in a single fused batch, because that is the only way to make decode economical. The scheduler was rebuilding that batch every 30 ms as people finished and new people arrived. Everything in sections 7 and 8 is about making that work.

> **Why the interviewer asks this.** They want to know whether you've actually operated a serving system or only read about it — a candidate who has will naturally mention the scheduler and the memory-admission check, and a candidate who hasn't will describe the model and skip the server.

> **Saying it out loud.** "A request comes in, gets tokenized, and then it sits in the scheduler's queue until there's enough free KV-cache memory to admit it. Once it's admitted you do one big prefill pass over the whole prompt in parallel — that's your time-to-first-token, and it's compute-bound. After that you're in the decode loop, one token at a time, where each step reads the entire model's weights out of HBM to produce a single token, so it's memory-bandwidth-bound. The trick that makes it economical is that your token is batched with tokens from every other active user, and the scheduler rebuilds that batch on every single step."

---

## 2. What the GPU is actually doing

Everything about LLM inference performance comes from one fact: **on modern accelerators, moving a number is far more expensive than doing arithmetic on it.** If you internalize this, most of the field's design decisions stop looking clever and start looking inevitable.

### The memory hierarchy, from fast to slow

An H100 SXM has, roughly, four tiers of storage, and they differ by orders of magnitude in both size and speed.

**Registers.** Each of the GPU's 132 streaming multiprocessors (SMs) has a register file of about 256 KB. Access is effectively free — a fused multiply-add reads its operands from registers in a single instruction. This is where arithmetic actually happens.

**Shared memory / L1 (the "SRAM" you hear about).** 228 KB per SM on Hopper, of which a single thread block can use up to 227 KB ([NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)). This is software-managed scratchpad: a kernel explicitly copies a tile of data in, works on it, and copies results out. Aggregate across all SMs it's about 30 MB and its aggregate bandwidth is in the hundreds of TB/s. When you hear "FlashAttention keeps things in SRAM," this is the SRAM.

**L2 cache.** 50 MB, shared across the whole chip, hardware-managed, bandwidth above 10 TB/s. Useful, but far too small to hold a 70B model's weights.

**HBM (high-bandwidth memory).** 80 GB on the H100 SXM, at **3.35 TB/s** — that's the number on [NVIDIA's H100 page](https://www.nvidia.com/en-us/data-center/h100/), and it is the single most important number in this document. It's stacked DRAM sitting next to the die on the same package. Enormous by GPU standards, glacial by register standards.

Two more channels matter once you have multiple GPUs: **NVLink 4** gives about 900 GB/s per GPU of GPU-to-GPU bandwidth inside a node, and **PCIe Gen5** gives about 128 GB/s to the host. Notice NVLink is roughly 3.7× *slower* than local HBM, which is why tensor parallelism has a real cost.

### Why moving data costs more than computing on it

Here is the arithmetic that makes the point. The H100 SXM's dense bf16 tensor-core throughput is **989.5 TFLOP/s**. (NVIDIA's marketing number is 1,979 TFLOPS, but that figure is *with structured sparsity* — a footnote that catches people out. Dense is half of it.) A bf16 multiply-add is 2 FLOPs, so the chip performs about $4.9 \times 10^{14}$ multiply-adds per second.

Meanwhile HBM delivers $3.35 \times 10^{12}$ bytes per second. A bf16 number is 2 bytes, so HBM supplies about $1.68 \times 10^{12}$ numbers per second.

$$\frac{4.9 \times 10^{14}\ \text{MACs/s}}{1.68 \times 10^{12}\ \text{numbers/s}} \approx 295$$

**The chip can do roughly 295 floating-point operations in the time it takes to fetch one byte from HBM.** If your algorithm doesn't do at least ~295 FLOPs per byte it touches, you are not limited by the tensor cores at all — you are limited by the memory bus, and the tensor cores are waiting.

The analogy I like: imagine a kitchen with a brigade of 132 line cooks who can chop at superhuman speed, but the pantry is a warehouse a five-minute walk away and you can only carry one ingredient at a time. The chopping is not the problem. The walking is the problem. Every optimization in this document is either "walk less" or "carry more per trip."

### What "tensor cores sit idle" actually means

This phrase gets used loosely, so let's be precise. Tensor cores are dedicated matrix-multiply units — each one consumes small tiles of two matrices and accumulates a product. They are what makes 989 TFLOP/s possible; ordinary CUDA cores on the same chip do roughly 67 TFLOP/s of fp32.

Tensor cores need matrices. If you hand them a matrix-*vector* product — one row of activations times a big weight matrix, which is exactly what a decode step does — there is no second dimension to fill the tile with. The hardware will do it, but most of the multiply-accumulate lanes are multiplying by structural zeros or simply unused, and the time is set entirely by how fast the weight matrix streams in from HBM. When a profiler reports "tensor core utilization 4%" during decode, that is not a bug. That is the geometry of the problem.

Batching fixes it by turning the matrix-vector product into a matrix-matrix product. With 80 concurrent users, that one row of activations becomes 80 rows, the weight matrix still gets read exactly once, and now the tensor cores have something to chew on. Same warehouse trip, 80 orders cooked.

### One more thing: kernel launch overhead

At small scale there's a fourth cost besides compute, memory, and communication: the CPU-side cost of telling the GPU what to do. A 70B model's forward pass involves hundreds of kernel launches, each with a few microseconds of overhead. At batch size 1 with a small model, this can be a genuine fraction of step time — which is why frameworks use **CUDA graphs** (record the whole launch sequence once, replay it as a single submission) for decode. It's an unglamorous optimization that buys real milliseconds. Worth mentioning if the interviewer asks what else you'd check.

> **Why the interviewer asks this.** They're testing whether your performance intuitions bottom out in hardware or in vibes. "The GPU is fast" is not an answer; "the GPU has a 295:1 compute-to-bandwidth ratio and decode has a 1:1 arithmetic intensity" is.

> **Saying it out loud.** "The thing to know about a modern GPU is that compute is nearly free relative to data movement. An H100 does about 990 dense bf16 teraflops but only has 3.35 terabytes a second of HBM bandwidth, so it can do about 295 floating-point ops in the time it takes to read one byte. Any kernel that doesn't hit that ratio is bandwidth-bound and the tensor cores are just idling. Decode is the extreme case — you read every weight in the model to compute one token, so your intensity is about one op per byte. You're off the roofline by more than two orders of magnitude."

---

## 3. The two phases: prefill and decode

### The intuition

Reading and writing are different activities. When you read a page of text, your eyes take in whole lines at a time and you process the page in parallel — you don't read letter by letter with a pause between each. When you *write* a sentence, you produce one word, then the next, and each word depends on the ones before it. You cannot write word seven before you've decided on word six.

An LLM has exactly this asymmetry, and it is entirely a consequence of causal attention plus autoregressive generation. The prompt already exists, so every position in it can be processed simultaneously. The output doesn't exist yet, so it has to be produced strictly one token at a time.

These two activities are called **prefill** and **decode**, and they have opposite performance characteristics. This is the most important structural fact in LLM inference.

### Prefill, precisely

You have a prompt of length $P$. You run a single forward pass in which all $P$ positions go through the network together. Every matrix multiply in the model becomes a $P \times d$ activation matrix times a $d \times d$ weight matrix — a big, fat, dense matrix-matrix product, exactly the shape tensor cores were designed for.

The FLOP count has two parts. The dominant one is the weight matrices: every parameter in the model gets used once per token, and a multiply-add is 2 FLOPs, so

$$\text{FLOPs}_{\text{weights}} \approx 2 \cdot N_{\text{params}} \cdot P$$

The second part is attention itself, which has no parameters but does have a quadratic term because every query attends to every key:

$$\text{FLOPs}_{\text{attn}} \approx 2 \cdot n_{\text{layers}} \cdot P^2 \cdot d_{\text{model}}$$

(That's $4 \cdot n_{\text{layers}} \cdot P^2 \cdot d_{\text{model}}$ for the two matmuls $QK^\top$ and $\text{AV}$, halved because causal masking means you only need the lower triangle.)

| Symbol | Meaning | Llama 3.1 70B value |
|---|---|---|
| $N_{\text{params}}$ | total model parameters | $70.6 \times 10^9$ |
| $P$ | prompt length in tokens | (workload) |
| $n_{\text{layers}}$ | transformer blocks | 80 |
| $d_{\text{model}}$ | hidden size | 8192 |

**Worked example, $P = 6000$:**

$$\text{FLOPs}_{\text{weights}} = 2 \times 70.6 \times 10^9 \times 6000 = 8.472 \times 10^{14}$$

$$\text{FLOPs}_{\text{attn}} = 2 \times 80 \times 6000^2 \times 8192 = 4.719 \times 10^{13}$$

Total $\approx 8.94 \times 10^{14}$ FLOPs, with attention contributing **5.6%**. That ratio is worth remembering because it tells you attention is *not* the dominant cost at 6K context — the FFN and projection weights are. Attention only takes over when $P$ gets large: the crossover where the quadratic term equals the linear term is at roughly $P \approx N_{\text{params}} / (n_{\text{layers}} \cdot d_{\text{model}}) = 70.6\times10^9 / (80 \times 8192) \approx 108{,}000$ tokens. So for a 70B model, attention becomes the majority of prefill cost somewhere around 100K context. This is exactly why long-context work is where FlashAttention and Ring Attention live.

Now the arithmetic intensity. The bytes you must move are dominated by reading the weights once: $70.6 \times 10^9 \times 2 = 1.41 \times 10^{11}$ bytes. So

$$\text{intensity} = \frac{8.472 \times 10^{14}}{1.41 \times 10^{11}} = 6000\ \text{FLOP/byte}$$

which is not a coincidence — the intensity of prefill in the weight-bound part is just $P$. At $P = 6000$ you are at 6000 FLOP/byte against a machine balance point of 295. **Prefill is compute-bound by a factor of about 20.** Tensor cores are genuinely busy. Your only lever is doing fewer FLOPs or doing them in lower precision.

### Decode, precisely

Now the other side. One token enters the network. Every matrix multiply is a $1 \times d$ vector times a $d \times d$ matrix. You still read every single weight in the model out of HBM, and you get one token out of it.

$$\text{FLOPs per decode step} \approx 2 \cdot N_{\text{params}} \cdot B \quad\text{(for batch size } B)$$
$$\text{Bytes per decode step} \approx N_{\text{params}} \cdot \text{bytes-per-weight} + \text{KV bytes read}$$

At batch size 1 with bf16 weights, ignoring KV for the moment:

$$\text{intensity} = \frac{2 \cdot N_{\text{params}} \cdot 1}{N_{\text{params}} \cdot 2} = 1.0\ \text{FLOP/byte}$$

**Exactly one FLOP per byte.** Not "about two," not "roughly one" — it is exactly $2/\text{bytes-per-weight}$, so 1.0 for bf16 and 4.0 for int4. Against a balance point of 295, you are running at roughly $1/295 = 0.34\%$ of the chip's arithmetic capability. That figure is not hyperbole; it is what a profiler will actually show you.

The time per step is therefore set purely by bandwidth:

$$t_{\text{step}} \ge \frac{N_{\text{params}} \cdot \text{bytes-per-weight}}{\text{HBM bandwidth}}$$

For a 70B model in bf16 that's $1.41 \times 10^{11}$ bytes. Divided by one H100's 3.35 TB/s gives **42.1 ms**, or 23.7 tokens/second — and note this is a *hypothetical*, because 141 GB of weights does not fit in an 80 GB H100 at all. On two H100s with tensor parallelism, aggregate bandwidth is 6.7 TB/s and the floor is **21.1 ms**, or 47.5 tok/s. On four, 13.4 TB/s, **10.5 ms**, 94.9 tok/s. Those are hard floors that no software can beat without changing the bytes-read term: quantize the weights, or produce more than one token per weight read.

### The per-token efficiency gap

Put the two side by side for our example request. Prefill: 6,000 tokens in 450 ms, so 0.075 ms/token. Decode: 1 token in 30 ms. Decode is 400× less efficient per token. Both phases run the identical model on the identical hardware. The only difference is the shape of the matrices, and therefore whether you get to reuse each weight you fetched.

This is why the field's optimizations cleave so cleanly into two families. **For prefill you reduce work**: chunking it to smooth out latency, FlashAttention to cut the memory traffic of the quadratic part, FP8 to double the effective FLOPs, prefix caching to skip it entirely when the prompt is a repeat. **For decode you increase reuse**: batching so one weight read serves many sequences, quantization so there are fewer bytes to read, speculative decoding so one weight read yields several tokens, GQA and MLA so the KV read shrinks.

### A caveat worth having ready

"Prefill is compute-bound, decode is memory-bound" is the right first-order model, and you should lead with it. But an interviewer who knows the area may push, and the honest refinements are: (a) at very long context the *attention* part of decode also becomes bandwidth-heavy, because you read the whole KV cache every step, and at 128K context that read can exceed the weight read; (b) at very large batch sizes decode does cross into compute-bound territory, which we'll compute exactly in the next section; (c) with a tiny model and a small batch, kernel-launch overhead rather than either bandwidth or FLOPs can dominate.

> **Why the interviewer asks this.** This is the load-bearing dichotomy of the whole field. If you have it, every follow-up question has an obvious frame; if you don't, you'll answer each optimization question from scratch and it will show.

> **Saying it out loud.** "Every request has two phases with opposite bottlenecks. Prefill runs the whole prompt through in one parallel pass — big dense matmuls, tensor cores saturated, compute-bound. For a 6K prompt on a 70B model that's about 850 teraflops of work and an arithmetic intensity around 6,000 ops per byte, way above the H100's balance point of 295. Then decode goes one token at a time, and each step reads all 141 gigabytes of weights to produce a single token, so the intensity is exactly one op per byte. You're running at a third of a percent of the chip's compute. Per token, decode is a couple hundred times less efficient than prefill, and basically every optimization in serving is aimed at one phase or the other."

---

## 4. The roofline model: how to know which regime you're in

### The intuition

The roofline model is a way of drawing a ceiling over what any kernel can achieve on a given machine, using only two hardware numbers. Think of it as a road with two speed limits: one imposed by your engine, one imposed by how fast fuel can be delivered. Below a certain fuel-efficiency, the pump is the limit and a bigger engine buys you nothing.

### The construction

Define **arithmetic intensity** $I$ as FLOPs performed per byte moved from HBM. Then achievable throughput is

$$\text{FLOP/s}_{\text{achievable}} = \min\big(\text{peak FLOP/s},\ \ I \times \text{bandwidth}\big)$$

Plotted with $I$ on the x-axis, that's a rising diagonal (bandwidth-limited) that hits a horizontal ceiling (compute-limited). The corner is the **balance point** or **ridge point**:

$$I^* = \frac{\text{peak FLOP/s}}{\text{bandwidth (bytes/s)}}$$

| Accelerator | Dense bf16 | HBM bandwidth | Balance point |
|---|---|---|---|
| A100 80GB SXM | 312 TFLOP/s | 2.039 TB/s | **153 FLOP/byte** |
| H100 SXM | 989.5 TFLOP/s | 3.35 TB/s | **295 FLOP/byte** |
| H200 SXM | 989.5 TFLOP/s | 4.8 TB/s | **206 FLOP/byte** |
| B200 SXM | ~2,250 TFLOP/s | 8.0 TB/s | **~281 FLOP/byte** |

All FLOP figures are dense, i.e. half the sparsity-inclusive numbers on the datasheets. H100 and H200 memory bandwidth are from NVIDIA's product pages; the B200 per-GPU figures are derived by dividing the [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) system specs (1,440 GB, 64 TB/s, 144 PFLOPS FP4, 72 PFLOPS FP8, all sparsity-inclusive) by eight — treat them as approximate and verify against a current datasheet, since Blackwell SKUs vary (180 GB in DGX B200, 192 GB in some GB200 configurations).

**Two things worth noticing in that table.** First, H200 is the *same compute die* as H100 with faster, larger memory. Its balance point drops from 295 to 206, which means memory-bound workloads — decode — get a straight 43% speedup while compute-bound workloads get nothing. That is exactly why H200 was marketed on inference. Second, generation over generation NVIDIA has grown compute faster than bandwidth, so balance points trend upward, which means the *decode problem gets structurally worse over time*, not better. This is a good thing to say out loud; it shows you're reasoning about trends rather than reciting a table.

### Where decode sits, and where batching puts it

At batch size $B$, decode does $2 N_{\text{params}} B$ FLOPs while reading $N_{\text{params}} \times 2$ bytes of bf16 weights. The weight read is *independent of $B$* — that's the entire point. So intensity is:

$$I_{\text{decode}} = \frac{2 N_{\text{params}} B}{2 N_{\text{params}}} = B$$

**The arithmetic intensity of bf16 decode is numerically equal to the batch size.** It's the cleanest result in this document and it's worth stating exactly that way in an interview.

| Batch size | Intensity (FLOP/byte) | vs H100 ridge (295) |
|---|---|---|
| 1 | 1 | memory-bound, 295× below |
| 8 | 8 | memory-bound |
| 32 | 32 | memory-bound |
| 128 | 128 | memory-bound |
| 256 | 256 | memory-bound, still |
| 512 | 512 | compute-bound |

So the crossover on an H100 with bf16 weights is at $B \approx 295$. Two important caveats. First, this ignores the KV-cache read, which *does* scale with batch size and with context length — include it and the crossover moves higher, because you're adding bytes proportional to $B$. Second, if you quantize weights to int4, the byte count drops 4× and intensity becomes $4B$, so the crossover drops to $B \approx 74$ — quantization doesn't just make decode faster, it makes it reach the compute-bound regime sooner.

In practice you will rarely run batch 295 on a 70B model with long contexts, because KV-cache memory runs out first (section 15 shows this concretely: we get to batch 82). So the honest summary is: **realistic production decode is memory-bound, and batching is the lever that gets you as close to the ridge as your KV memory allows.**

### Using the roofline as a diagnostic

The practical value of this model in an interview is that it lets you *predict* whether an optimization will help before you try it. Someone proposes switching from bf16 to FP8 compute for decode: you ask what the batch size is, and if it's 30 you say it won't help much, because at intensity 30 the tensor cores aren't the constraint — FP8 *weights* would help (fewer bytes) but FP8 *math* wouldn't. Someone proposes a fancier attention kernel to speed up TTFT on 32K prompts: you say yes, because prefill at 32K is compute- and attention-heavy. That's the move interviewers are looking for.

> **Why the interviewer asks this.** The roofline is the shibboleth for "can this person reason about performance quantitatively rather than by folklore." A candidate who can compute a balance point and place a workload on it can be trusted to evaluate an optimization proposal.

> **Saying it out loud.** "Roofline says your achievable throughput is the min of peak flops and intensity times bandwidth, and the corner is peak-flops-over-bandwidth. On an H100 that's 989 teraflops over 3.35 terabytes a second, so about 295 flops per byte. And here's the neat part — for bf16 decode, the arithmetic intensity is just the batch size, because you read the weights once no matter how many sequences you're serving. So batch one is intensity one, and you'd need batch around 295 to be compute-bound. That's the whole justification for continuous batching: it's how you climb the diagonal. In practice you run out of KV-cache memory long before you get to 295, which is why decode stays memory-bound in production."

---

## 5. The KV cache

### The problem it solves

Start by imagining you *don't* have one. You've generated 500 tokens after a 6,000-token prompt and you want token 501. Attention at each layer needs the key and value vectors for all 6,500 preceding positions. Without a cache, you'd feed the entire 6,500-token sequence through the network again, computing all 6,500 keys and values from scratch — even though 6,499 of them are bit-for-bit identical to what you computed on the previous step.

That is not a small waste. Let's price it. With a cache, total FLOPs for our request are the prefill pass plus 500 single-token passes:

$$\underbrace{2 \times 70.6\times10^9 \times 6000}_{8.472\times10^{14}\ \text{(prefill)}} + \underbrace{2 \times 70.6\times10^9 \times 500}_{7.060\times10^{13}\ \text{(decode)}} = 9.178 \times 10^{14}\ \text{FLOPs}$$

Without a cache, step $i$ reruns a full forward pass over $6000 + i$ tokens:

$$\sum_{i=0}^{499} 2 \times 70.6\times10^9 \times (6000+i) \;+\; \text{prefill} = 4.421 \times 10^{17}\ \text{FLOPs}$$

That's **482× more compute**. The generic statement is that caching turns the total cost of generating $n$ tokens from $O(n^2 d^2)$ into $O(n d^2)$ — a factor-of-$n$ saving — and this is why no one has ever seriously shipped autoregressive generation without it. When the README in this folder says "10–100× speedup," that's an understatement for realistic prompt lengths; the true factor is proportional to sequence length.

The insight that makes it work is one line: **in causal attention, the key and value vectors for position $j$ depend only on tokens up to $j$.** They can never change when you append token $j+1$. So compute them once and keep them. The query vector is the only thing that's genuinely new each step, which is why we cache K and V and not Q.

### What's actually stored

For each layer, for each key/value head, for each position, two vectors of length $d_{\text{head}}$. The size formula:

$$\text{KV bytes per token} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv heads}} \cdot d_{\text{head}} \cdot b$$

| Symbol | Meaning | Note |
|---|---|---|
| 2 | one tensor for $K$, one for $V$ | not negotiable |
| $n_{\text{layers}}$ | number of transformer blocks | every layer caches separately |
| $n_{\text{kv heads}}$ | number of *key/value* heads | **not** query heads — this is where GQA saves you |
| $d_{\text{head}}$ | dimension per head | typically 128 in Llama-family models |
| $b$ | bytes per element | 2 for fp16/bf16, 1 for fp8/int8, 0.5 for int4 |

Multiply by sequence length for one request, and by batch size for the whole server.

> **A note on the more common form of this formula.** You will often see $2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot \text{seqlen} \cdot b$. That is only correct for plain multi-head attention, where $n_{\text{heads}} \cdot d_{\text{head}} = d_{\text{model}}$. Every current production model uses GQA or MLA, so **use the $n_{\text{kv heads}}$ form** or you will overestimate by 8× and an interviewer will catch it.

### Worked example: Llama 3.1 70B

Architecture from the published config: `num_hidden_layers` 80, `num_attention_heads` 64, `num_key_value_heads` 8, `hidden_size` 8192, `intermediate_size` 28672, `vocab_size` 128256, `max_position_embeddings` 131072 ([config.json](https://huggingface.co/HiTZ/Latxa-Llama-3.1-70B-Instruct/blob/main/config.json)). Note $d_{\text{head}} = 8192/64 = 128$, and GQA with 8 KV heads means each KV head is shared by 8 query heads.

**Per token, fp16:**

$$2 \times 80 \times 8 \times 128 \times 2 = 327{,}680\ \text{bytes} = 320\ \text{KiB per token}$$

That is a number worth memorizing, because it makes everything else a one-line multiplication. 320 KiB per token.

**One 8,192-token sequence:** $327{,}680 \times 8192 = 2{,}684{,}354{,}560$ bytes $= 2.50$ GiB.

**Batch of 32 such sequences:** $\times 32 = 85{,}899{,}345{,}920$ bytes $= 80.0$ GiB $= 85.9$ GB.

So the KV cache for 32 users at 8K context is **80 GiB — one entire H100's worth of memory**, on top of the 131.5 GiB (141.2 GB) the weights already occupy. This is the number that should reframe how you think about serving: memory is not mostly the model.

**The GQA counterfactual.** If Llama 3.1 70B used plain MHA with all 64 heads caching K and V:

$$2 \times 80 \times 64 \times 128 \times 2 = 2{,}621{,}440\ \text{bytes} = 2{,}560\ \text{KiB} = 2.5\ \text{MiB per token}$$

Batch 32 at 8K: $640$ GiB $= 687$ GB. Exactly 8× more, which is the ratio $64/8$. Grouped-query attention is not a minor efficiency tweak; it is the difference between eight GPUs of KV cache and one.

**A smaller model for contrast.** Llama 3.1 8B has 32 layers, also 8 KV heads and $d_{\text{head}}=128$, so $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes $= 128$ KiB per token. A *single* sequence at its full 128K context needs $131072 \times 131072 = 16.0$ GiB of KV cache — while the weights are only 16 GB. At maximum context, one user's cache is as big as the entire model. That's the cleanest illustration of why long context is a memory problem, not a compute problem.

### What it implies for serving

Rearrange the formula to answer the question that actually matters to an SRE: how many users fit?

$$B_{\max} = \frac{\text{HBM total} - \text{weights} - \text{overhead}}{\text{KV bytes per token} \times \text{average sequence length}}$$

Every term in that numerator is a lever, and every term in the denominator is a lever. Quantize weights → bigger numerator. Quantize the KV cache → smaller denominator. Switch MHA to GQA → smaller denominator. Evict old tokens → smaller denominator. Eliminate allocator waste → the numerator gets closer to its theoretical value, which is PagedAttention's contribution.

And the punchline: **on a modern serving system, the maximum batch size is set by KV-cache memory, not by compute.** The GPU would happily process 300 concurrent sequences; it runs out of room to remember them at 82. Every serving-system paper of the last three years is, at bottom, about that constraint.

### Ways to shrink it

| Technique | Mechanism | Saving | Lossy? |
|---|---|---|---|
| MQA (Shazeer 2019) | one K/V head shared by all query heads | $n_{\text{heads}}\times$ | slight quality cost |
| GQA (Ainslie et al. 2023) | groups of query heads share a K/V head | $n_{\text{heads}}/n_{\text{groups}}\times$ | near-MHA quality |
| MLA (DeepSeek-V2/V3) | cache a low-rank latent, decompress at use | ≈ GQA with 2.25 groups | reported *better* than MHA |
| KV quantization | store K/V in fp8/int8/int4 | 2–4× | mild, method-dependent |
| PagedAttention | eliminate allocator fragmentation | recovers 2–5× of *wasted* memory | lossless |
| Prefix sharing | one copy of a shared prompt prefix | workload-dependent, can be large | lossless |
| Sliding-window attention | only attend within a window | context/window | lossy beyond window |
| Eviction (StreamingLLM, H2O) | drop low-importance tokens | policy-dependent | lossy |
| Cross-layer KV sharing (YOCO, CLA) | layers reuse one cache | up to ~$n_{\text{layers}}\times$ | architecture change, needs training |

The first three are architectural — chosen at pretraining time, you can't retrofit them. The middle three are serving-system choices you can make today on an existing checkpoint. The last three trade quality or require retraining. Knowing which bucket each one falls in is a good way to sound like you've shipped something.

> **Why the interviewer asks this.** The KV cache formula is the single most common piece of arithmetic in ML-systems interviews, and the follow-up — "so what limits your batch size?" — separates people who memorized the formula from people who understand what it's for.

> **Saying it out loud.** "The KV cache stores the keys and values for every position you've already processed, so each decode step only computes the new token's K and V instead of redoing the whole prefix. Without it you'd be doing quadratic work — for a 6K prompt and 500 output tokens on a 70B model it's about 480 times more compute. The size is two, for K and V, times layers, times *KV* heads, times head dim, times bytes, times sequence length, times batch. For Llama 3.1 70B in fp16 that's 320 kibibytes per token, so 32 users at 8K context is 80 gibibytes — a whole H100 just for the cache, on top of 141 gigabytes of weights. And that's the real point: in production your max batch size is set by KV memory, not by compute. GQA is what makes it survivable; with full 64-head MHA that same batch would be 687 gigabytes."

---

## 6. Sampling: from logits to a token

This section exists because it's a gap in most treatments, and because interviewers sometimes ask "where does temperature actually apply?" to check that you know the pipeline end to end rather than as a black box.

### What comes out of the network

The last transformer layer produces a hidden vector of size $d_{\text{model}}$ for the current position. That gets multiplied by the language-model head — a $d_{\text{model}} \times |V|$ matrix, where $|V|$ is vocabulary size — producing a vector of $|V|$ real numbers called **logits**. For Llama 3.1, $|V| = 128{,}256$. Logits are unnormalized scores; higher means the model prefers that token. They are not probabilities and can be any real value.

### The four things you can do with them

**Greedy decoding.** Take $\arg\max$. Deterministic given the same input and the same numerics. Good for tasks with a single right answer — extraction, classification, code that must compile. Its failure mode is repetition and blandness: greedy text loops.

**Temperature.** Divide logits by $T$ before softmax:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

$T = 1$ leaves the model's distribution alone. $T < 1$ sharpens it — differences between logits get magnified, the top token gets more probability mass, output becomes more deterministic. As $T \to 0$ this converges to greedy. $T > 1$ flattens the distribution and makes rare tokens more likely; above about 1.5 most models produce noticeable incoherence. The concrete intuition: with logits $[4, 2, 1]$, at $T=1$ the softmax is roughly $[0.84, 0.11, 0.04]$; at $T=0.5$ it's about $[0.98, 0.018, 0.002]$; at $T=2$ it's about $[0.62, 0.23, 0.14]$.

**Top-$k$.** Keep the $k$ highest-probability tokens, zero the rest, renormalize. Simple and cheap. Its weakness is that $k$ is fixed while the *shape* of the distribution isn't: when the model is confident (one token at 0.95), $k=50$ lets 49 bad tokens in; when it's genuinely uncertain over 200 plausible continuations, $k=50$ cuts off good options.

**Top-$p$ (nucleus sampling, Holtzman et al. 2019).** Sort descending, walk down accumulating probability, stop when the cumulative sum first exceeds $p$, keep that set, renormalize. This adapts to the distribution's shape: a confident step keeps one or two tokens, an uncertain step keeps hundreds. $p = 0.9$ to $0.95$ is the common range and is the default in most APIs. Often combined with a temperature and a generous top-$k$ as a safety net.

There are related knobs — **min-$p$** (keep tokens above a fraction of the top token's probability), **repetition** and **frequency penalties** (subtract from logits of tokens already emitted), and **logit bias** (hand-specified additive nudges). All are applied to logits before the softmax, in a well-defined order that your framework documents.

### Why sampling is effectively free

Here's the number that makes the point. One decode step of Llama 3.1 70B at batch 1 is $2 \times 70.6\times10^9 = 1.41 \times 10^{11}$ FLOPs. The pieces of sampling:

| Operation | Rough cost | As a fraction of the forward pass |
|---|---|---|
| LM head matmul ($2 \cdot d \cdot \lvert V \rvert$) | $2.10 \times 10^{9}$ FLOPs | 1.49% |
| argmax over vocab | 128,256 ops | 0.000091% |
| softmax with temperature | ~513,000 ops | 0.00036% |
| full sort for top-$p$ ($\lvert V\rvert \log_2 \lvert V\rvert$) | ~2.18 M ops | 0.0015% |

So even the most expensive sampling policy costs about one-thousandth of one percent of the step. **You should never choose greedy over top-$p$ for performance reasons.** Choose based on the task.

Two honest caveats. The LM head itself is not free — at 1.5% of the forward pass and $128256 \times 8192 \times 2 = 2.1$ GB of weights to read, it's a real tensor, and for very large vocabularies with tensor parallelism it needs its own attention (frameworks often shard it and do a distributed argmax). And sampling *kernels* can show up in profiles at large batch sizes not because of FLOPs but because of launch overhead and poor memory-access patterns on the $B \times |V|$ logits tensor — at batch 256, that tensor is $256 \times 128256 \times 4$ bytes $= 131$ MB, and touching it several times costs real bandwidth. If someone tells you their sampler is slow, that's the reason, and the fix is a fused kernel, not a simpler policy.

> **Why the interviewer asks this.** Usually as a warm-up or a sanity check — but the good version of the question is "how much does top-p cost you," and the right answer is "essentially nothing, and here's the arithmetic," because it shows you cost things out rather than guessing.

> **Saying it out loud.** "The network gives you logits over the vocabulary, and sampling is what turns those into a token. Temperature divides the logits before the softmax — lower sharpens toward greedy, higher flattens. Top-k keeps the k best, top-p keeps the smallest set whose probability mass exceeds p, which is nicer because it adapts to how confident the model is. And the cost is negligible: a 70B forward pass is about 141 gigaflops per token, and even a full sort of a 128K vocabulary is a couple million operations — call it a thousandth of a percent. So you pick your sampling strategy for output quality, never for speed. The only time it shows up in a profile is at big batch sizes where the logits tensor itself is hundreds of megabytes and you're bandwidth-bound on it."

---

## 7. Batching: the mechanics, and why the naive version is so bad

### Why you batch at all

Section 4 gave the answer: decode's arithmetic intensity equals the batch size. Batch 1 wastes 99.7% of an H100's arithmetic capability. Batching is not an optimization; it is the difference between a viable business and a demo.

The mental image: your chef has to walk to the warehouse for every ingredient. If she's cooking one dish, the trip dominates. If she's cooking eighty dishes that all need flour, she makes one trip and cooks eighty. The weight matrices are the flour. Batching is cooking many orders per warehouse trip.

### Static batching, and its two failure modes

The naive implementation — the one every research codebase ships — is: collect $B$ requests, pad them to a common length, run them together, return all results when all are done.

**Failure mode one: padding waste in prefill.** Requests have different prompt lengths. If you batch prompts of length 100, 400, and 6,000 into a rectangular tensor, you pad the first two out to 6,000. You've just done $3 \times 6000 = 18{,}000$ token-positions of work to accomplish $6{,}500$ tokens of useful work — 64% waste. The masks make the padded positions not *affect* the answer, but you paid for them. Real systems avoid this entirely with **ragged batching**: concatenate all sequences into one flat 1-D tensor of total length $\sum P_i$, carry a `cu_seqlens` array of cumulative offsets, and use attention kernels (FlashAttention's varlen entry points) that read those offsets to know where each sequence begins and ends. No padding at all. If an interviewer asks how you'd batch variable-length prompts, "flatten and pass cumulative sequence lengths to a varlen attention kernel" is the answer that signals you've read the code.

**Failure mode two: straggler waste in decode.** This one is worse and it can't be fixed by better tensor layout. Output lengths in real traffic are wildly variable — someone asks "what's 2+2" and someone else asks for an essay. Take a batch of 8 with output lengths 30, 50, 80, 120, 200, 350, 600, 900. Static batching runs until the longest finishes: 900 steps. Each step occupies all 8 slots. So you consume $900 \times 8 = 7{,}200$ slot-steps to produce $30+50+\cdots+900 = 2{,}330$ tokens.

$$\text{utilization} = \frac{2330}{7200} = 32.4\%$$

**Two-thirds of your GPU time is spent computing padding for requests that already finished.** And it's worse than the number suggests, because the short requests also sat there for 900 steps of wall clock before their results were released, so their latency is set by the essay-writer's length, not their own. This is the concrete meaning of "GPU utilization for serving is often under 30% without continuous batching."

### Continuous batching

The fix (Yu et al., **Orca**, OSDI 2022) is to schedule at the granularity of a *decode iteration* rather than a request. Hence the other names for it: iteration-level scheduling, or dynamic batching.

The analogy is a bus versus a taxi rank. Static batching is a bus: it fills up, drives the whole route, and nobody gets off until the end. Continuous batching is a taxi rank with a fixed number of cars — the instant one comes free, the next person in line takes it. No seat sits empty waiting for the slowest passenger.

Concretely, once per step the scheduler does something like:

1. **Reap.** For each running sequence, check whether it just emitted EOS, hit `max_tokens`, or matched a stop string. If so, finalize it and free its KV blocks back to the pool.
2. **Admit.** Look at the waiting queue. For each candidate, in priority order, estimate its KV-memory need and check it against the free pool. Admit as many as fit.
3. **Prefill the new arrivals.** Run their prompts — possibly as chunks interleaved with decode (section 16), possibly as a separate step.
4. **Build the step.** Assemble one flat batch containing exactly one token from each running sequence, gather their KV block tables, and launch the forward pass.
5. **Sample and emit.** One token per sequence, streamed out.
6. **Handle pressure.** If free memory has fallen below a threshold — because sequences grew longer than expected — preempt: pick victims, either swap their KV cache out to CPU memory or discard it and mark them for recompute later, and put them back in the queue.

Because a decode step is the same amount of work per sequence regardless of *which* sequences they are, swapping membership between steps is nearly free — you're changing pointers into a block table, not moving tensors.

The result is that utilization goes from that 32% figure toward the 70–90% range, and throughput improvements of roughly 2–4× over static batching are routine. The vLLM paper's own comparison is against Orca variants and shows 1.7×–2.7× higher sustainable request rates on ShareGPT traffic and up to 22× over FasterTransformer for basic sampling ([Kwon et al. 2023](https://arxiv.org/pdf/2309.06180)).

### The hard parts

Continuous batching sounds simple and is not, for four reasons worth being able to name.

**Prefill and decode don't mix cleanly.** A newly admitted request needs a 450 ms prefill. If you run that as a monolithic step, every one of your 80 already-running decodes stalls for 450 ms, which blows their TPOT by a factor of fifteen. This is called a **generation stall** and it's the central problem chunked prefill solves.

**You can't predict output length.** Admission is a bet: you admit a request assuming it'll generate ~200 tokens, and it generates 4,000. Now you're over-committed on memory and have to preempt someone. Some systems train length predictors; most use conservative watermarks and accept occasional preemption.

**Scheduling policy is a real design space.** FCFS is fair and leaves throughput on the table. Shortest-job-first is great for mean latency and starves long requests. Memory-aware admission avoids preemption but underutilizes. Priority classes let you sell an SLA tier. There is no free answer; the right one depends on what you're promising customers.

**Preemption is expensive when you get it wrong.** Swapping a sequence's KV to host memory moves gigabytes over PCIe at 128 GB/s; recomputing it instead burns prefill FLOPs. Both are much worse than not having to do it.

Every serious framework — vLLM, TensorRT-LLM, TGI, SGLang, LMDeploy — implements continuous batching, and their scheduler policies are where they actually differ.

> **Why the interviewer asks this.** It's the highest-leverage systems idea in serving, and it's a clean test of whether you think about utilization. The great follow-up, which you should be ready for, is "what happens when a new request needs prefill in the middle of your decode batch?"

> **Saying it out loud.** "Static batching is where you collect a batch, run it to completion, and return everything at the end — and it's terrible, because output lengths vary a lot. If you batch eight requests and the longest generates 900 tokens while the median generates 150, you burn 7,200 slot-steps to produce 2,300 tokens. That's 32% utilization, and the short requests wait for the long one. Continuous batching schedules per decode iteration instead: every step you reap the sequences that just finished, free their KV blocks, admit whatever fits from the queue, and rebuild the batch. It's cheap because decode work per sequence is constant, so you're just editing block-table pointers. That takes you from about 30% utilization to 70-plus, and it's typically 2 to 4x throughput. The hard part is that a newly admitted request needs a prefill, and a big monolithic prefill stalls everyone else's decode — which is exactly why chunked prefill exists."

---

## 8. PagedAttention: virtual memory for the KV cache

### The problem

You have continuous batching now, so the scheduler wants to pack as many sequences as possible into the KV memory you have. How do you allocate that memory?

The obvious approach, and the one every pre-2023 system used, is to give each request a contiguous buffer sized for the worst case: `max_seq_len` tokens. For Llama 3.1 70B at 128K max context that's $327{,}680 \times 131{,}072 = 40$ GiB *per request*. Even at a modest 8K cap it's 2.5 GiB per request. If the request actually generates 100 tokens, you used 32 MiB of a 2.5 GiB reservation and wasted 98.7%.

This is textbook **internal fragmentation** — space reserved inside an allocation that never gets used. And you get **external fragmentation** too: as requests of different sizes come and go, the free memory ends up as a scattering of holes, none individually large enough for the next contiguous request, even though their sum is plenty.

The vLLM paper measured this on real systems and found that **only 20.4% to 38.2% of KV-cache memory in existing systems was holding actual token state.** The rest was reservation and fragmentation. Meaning: roughly two-thirds to four-fifths of the most precious resource in your cluster was being wasted by the allocator.

### The fix

Kwon et al. (SOSP 2023) noticed this is precisely the problem operating systems solved in the 1960s with paged virtual memory, and applied the same solution. That's not an analogy — it's the same mechanism.

Carve GPU KV memory into fixed-size **blocks** (vLLM's default is 16 tokens' worth per block per layer). Give each sequence a **block table**: an array mapping logical token positions to physical block IDs. When a sequence needs room for token 17, allocate one more block and append its ID to the table; the block can be anywhere in physical memory. The attention kernel is modified to consult the block table and gather K/V from scattered physical locations rather than assuming contiguity.

The correspondence is exact: logical token position ↔ virtual address; block ↔ page; block table ↔ page table; the modified attention kernel ↔ the MMU; swapping a sequence's blocks to host RAM under memory pressure ↔ swapping to disk. If you've taught an OS course, you can explain PagedAttention in thirty seconds.

### What it buys you

**Internal fragmentation collapses to at most one partial block per sequence per layer.** With 16-token blocks, worst case you waste 15 token-slots on a sequence that might be thousands of tokens long. The paper's framing is that block size 16 is "large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation."

**External fragmentation disappears entirely.** All blocks are the same size, so any free block satisfies any request. There is no such thing as a hole too small.

**Copy-on-write sharing becomes trivial.** Two sequences with a shared prefix point their block tables at the *same physical blocks*, with a reference count. This is what makes parallel sampling ($n > 1$ from one prompt), beam search, and cross-request prefix caching cheap instead of memory-multiplying. When a shared block needs to diverge, copy just that block.

**Preemption becomes clean.** Under memory pressure you can swap out a specific sequence's blocks without touching anyone else's, because nothing is contiguous with anything.

Net effect: you recover most of that 60–80% wasted memory, which translates directly into a larger batch — and since throughput at fixed hardware scales with batch size in the memory-bound regime, directly into throughput. Reported end-to-end gains are 2–4× versus contemporaneous systems, though the exact multiple depends entirely on how wasteful the baseline was.

### Costs and tuning

The attention kernel gets more complicated and slightly slower per FLOP, since it does an indirection through the block table and its memory accesses are less regular. In practice this is a small percentage loss that buys a multiple in batch size, so it's an easy trade — but it's the right thing to say when asked about downsides.

Block size is a real hyperparameter. Too small (say 1 or 2 tokens) and you get per-block bookkeeping overhead and poor memory coalescing in the kernel. Too large (say 512) and you're back to meaningful internal fragmentation. 16 is the common default; some systems use 32 for long-context workloads where the bookkeeping-to-data ratio matters more.

vLLM, TensorRT-LLM (as "paged KV cache"), SGLang, and TGI all implement a version of this. It is now table stakes rather than a differentiator, but interviewers ask about it by name because it's a beautifully clean idea and because knowing it signals you've read serving-systems papers.

> **Why the interviewer asks this.** Two things at once: do you know the systems literature, and can you recognize a classic OS problem in a new domain. The best answer explicitly names the virtual-memory correspondence.

> **Saying it out loud.** "PagedAttention is virtual memory applied to the KV cache — literally the same idea as paging. The problem it fixes is that the old approach reserved a contiguous buffer sized for max sequence length per request, so if a request generated a hundred tokens out of a possible eight thousand you wasted 99% of the reservation. The vLLM paper measured real systems and found only 20 to 38% of KV memory held actual tokens. So instead you split memory into fixed 16-token blocks and give each sequence a block table mapping logical positions to physical blocks, allocated on demand. Internal fragmentation drops to at most fifteen wasted slots per sequence, external fragmentation vanishes because all blocks are interchangeable, and you get prefix sharing for free by reference-counting blocks. The cost is a slightly more complex attention kernel that has to gather through the block table, which is a few percent — and you get two to four times the concurrent requests for it."

---

## 9. FlashAttention: making attention I/O-aware

### The problem, stated correctly

This is the section where the popular explanation is most often wrong, so let's be careful.

The textbook attention implementation does this: compute $S = QK^\top$, a full $N \times N$ matrix. Write it to HBM. Read it back, compute $P = \text{softmax}(S)$ row-wise. Write $P$ to HBM. Read it back, compute $O = PV$. Write $O$.

For $N = 8192$ with fp16, that $N \times N$ matrix is $8192^2 \times 2 = 134$ MB **per head per layer**. With 64 heads and 80 layers you're not storing them all at once, but you are writing and reading a 134 MB tensor several times for each of $64 \times 80 = 5{,}120$ head-layer pairs. The L2 cache is 50 MB, so none of this stays on chip. That is an enormous amount of HBM traffic, and — critically — the FLOPs involved in the softmax are trivial. You are moving hundreds of megabytes to do almost no arithmetic. Arithmetic intensity in the toilet, tensor cores idle.

The formal statement from the paper: standard attention performs $\Theta(Nd + N^2)$ HBM accesses.

### What FlashAttention actually does

**It does not reduce FLOPs.** Say this explicitly, because a lot of people get it wrong. FlashAttention computes $O(N^2 d)$ FLOPs, the same as standard attention, and returns the *exact same result* — the paper's title is "Fast and Memory-Efficient **Exact** Attention with IO-Awareness." It is not an approximation, unlike Performer or Linformer or the various low-rank and sparse schemes it competes with. Identical output, bit-comparable up to floating-point reassociation.

What it reduces is **HBM traffic**. Two ingredients:

**Tiling.** Split $Q$, $K$, $V$ into blocks that fit in the SM's 228 KB of shared memory. For a block of queries, loop over blocks of keys and values, computing that tile of the attention output incrementally. The $N \times N$ score matrix is never materialized in HBM — only one tile of it exists at a time, on chip.

**Online softmax** (the trick from Milakov & Gimelshein 2018 that makes tiling possible). Softmax needs a normalizer that depends on all the scores in a row, which naively means you can't emit any output until you've seen every key. The fix is to maintain, per query row, a running maximum $m$ and a running sum $\ell$, and rescale the accumulated output whenever a new tile reveals a larger maximum. Given the state after tile $j$ and the statistics of tile $j+1$:

$$m^{\text{new}} = \max(m, \tilde m), \qquad \ell^{\text{new}} = e^{m - m^{\text{new}}}\ell + e^{\tilde m - m^{\text{new}}}\tilde\ell$$

$$O^{\text{new}} = \frac{e^{m-m^{\text{new}}}\ell}{\ell^{\text{new}}} O + \frac{e^{\tilde m - m^{\text{new}}}}{\ell^{\text{new}}} \tilde P \tilde V$$

The running max is there for numerical stability — it's the standard "subtract the max before exponentiating" trick, made incremental. Everything else is bookkeeping to keep the partial sums consistent as the normalizer changes.

### The complexity result, precisely

The paper's Theorem 2 says FlashAttention requires

$$\Theta\!\left(\frac{N^2 d^2}{M}\right) \text{ HBM accesses, versus } \Theta(Nd + N^2) \text{ for standard attention}$$

where $M$ is SRAM size in elements. Since typically $M \gg d^2$ (for $d = 128$, $d^2 = 16{,}384$ elements, while $M$ is on the order of $10^5$), this is a substantial reduction — a factor of roughly $M/d^2$ in the quadratic term. Separately, Theorem 1 says the *memory footprint* is $O(N)$ beyond inputs and outputs, versus $O(N^2)$ for standard attention, which is what lets you train and serve long contexts at all.

**Note the correction here:** it is common to see "FlashAttention reduces memory access from $O(N^2)$ to $O(N)$." That conflates two different results. The $O(N)$ is the *memory footprint*; the HBM *access* count is $\Theta(N^2 d^2 / M)$, still quadratic in $N$, just with a much better constant. Getting this right is a cheap way to sound like you read the paper.

### Reported speedups, and the version history

The original paper reports 15% end-to-end speedup on BERT-large at sequence length 512, **3× on GPT-2** at 1K context, and 2.4× on Long Range Arena at 1K–4K. So "2–4× on long sequences" is fair for the original, with the important qualifier that the benefit grows with $N$ — at short sequence lengths there's little to save.

**FlashAttention-2** (Dao, 2023) restructured the parallelization — better work partitioning across warps and thread blocks, fewer non-matmul operations (which matter because non-matmul FLOPs run on the much slower non-tensor-core path), and parallelization over the sequence dimension. Roughly 2× over FA1, reaching about 50–73% of A100 peak.

**FlashAttention-3** (2024) targets Hopper specifically, using three techniques: warp-specialization with async tensor cores and TMA to overlap data movement with compute; interleaved block-wise matmul and softmax (pingpong scheduling) so the fast matmul units aren't waiting on the ~256×-slower special-function units; and, for FP8, *incoherent processing* — a random-signed Hadamard transform applied to Q and K before quantization to spread out outliers, reducing quantization error by 2.6×. Reported results: **1.5–2.0× faster than FA2 in FP16, up to 740 TFLOP/s (75% of H100 theoretical max, versus 35% for FA2), and close to 1.2 PFLOP/s in FP8** ([Dao, 2024](https://tridao.me/blog/2024/flash3/)).

### What it means for inference specifically

**For prefill: directly and substantially.** Prefill is exactly the regime FlashAttention was designed for — a large $N \times N$ attention computation. Long-prompt TTFT improves meaningfully, and the improvement grows with prompt length.

**For decode: not directly, and the reason is instructive.** At a decode step you have a single query attending to $N$ keys. There is no $N \times N$ matrix to avoid materializing; the score vector is length $N$. The bottleneck is different — you're streaming the whole KV cache through the SMs to do a tiny amount of arithmetic, which is bandwidth-bound but for a different reason. The specialized answer is **FlashDecoding**, which splits the KV cache along the *sequence* dimension across many thread blocks so that a single long-context decode can use the whole GPU rather than a handful of SMs, then combines the partial results with the same online-softmax rescaling. FlashDecoding++ adds further refinements. If someone asks "does FlashAttention help decode," the sophisticated answer is "not the original formulation; you want a decode-specialized kernel that parallelizes over KV length."

> **Why the interviewer asks this.** It's the cleanest available test of whether you understand that memory movement, not FLOPs, is the modern bottleneck. Candidates who say "it reduces the quadratic complexity" have failed the question.

> **Saying it out loud.** "FlashAttention is exact — same output, same FLOP count as standard attention. What it changes is memory traffic. Naive attention materializes the full N-by-N score matrix in HBM and reads it back twice, and for 8K context that's 134 megabytes per head per layer of pure data movement for almost no arithmetic. FlashAttention tiles the computation so the blocks fit in the SM's 228 kilobytes of shared memory, and uses online softmax — a running max and running sum that get rescaled as new tiles arrive — so you never need the whole row at once. HBM accesses go from theta of N-squared plus N-d to theta of N-squared-d-squared over SRAM size, and the memory footprint goes from quadratic to linear in N. That's 2 to 4x wall-clock on long sequences, and FlashAttention-3 on Hopper hits 740 teraflops, about 75% of peak. For inference it's mostly a prefill win — for decode there's no N-by-N matrix to avoid, so you want FlashDecoding instead, which splits the KV cache across thread blocks."

---

## 10. Speculative decoding

### The intuition

A junior writer drafts a sentence quickly. A senior editor reads the whole draft in one pass and marks where it first goes wrong. Everything up to that point is kept; the editor fixes that one word themselves; the junior starts again from there. The editor's time is the expensive resource, and the editor spent it *once* to validate several words. If the junior is decent, you get several words per editorial pass instead of one.

Now the reason this maps onto LLM decode. Section 4 established that a decode step at small batch is memory-bound: you spend 21 ms streaming 141 GB of weights and get one token. But a forward pass over $k$ tokens at once costs *almost exactly the same 21 ms*, because it reads the same weights and the extra FLOPs are free in the memory-bound regime. So if you had $k$ candidate tokens in hand, you could check all of them for the price of generating one.

The problem is you don't have candidates — that's what generation is for. Speculative decoding's answer: get them from a cheap model, then use the expensive model's single pass to check them, and — this is the clever part — use a sampling rule that makes the check *provably* preserve the expensive model's output distribution.

### The algorithm

Let $p$ be the target (large) model's distribution and $q$ the draft (small) model's. One iteration:

1. Run the draft model autoregressively for $\gamma$ steps to get candidate tokens $x_1, \ldots, x_\gamma$, recording $q(x_i \mid \text{prefix}, x_{<i})$ for each.
2. Run the target model **once** on the prefix plus all $\gamma$ candidates. Because attention is causal, one pass gives you $p(\cdot \mid \text{prefix})$, $p(\cdot \mid \text{prefix}, x_1)$, ..., $p(\cdot \mid \text{prefix}, x_{<\gamma})$ — all $\gamma+1$ distributions you need.
3. Walk $i = 1 \ldots \gamma$. Draw $r_i \sim U(0,1)$. Accept $x_i$ if
   $$r_i \le \min\!\left(1, \frac{p(x_i)}{q(x_i)}\right)$$
   Stop at the first rejection. Let $n$ be the number accepted.
4. If $n = \gamma$ (all accepted), sample one bonus token from $p(\cdot \mid \text{prefix}, x_{1..\gamma})$ — this is why an iteration can produce $\gamma+1$ tokens. If instead $x_{n+1}$ was rejected, sample its replacement from the **residual distribution**
   $$p'(x) = \frac{\max\big(0,\ p(x) - q(x)\big)}{\sum_{x'} \max\big(0,\ p(x') - q(x')\big)}$$

Leviathan et al.'s Algorithm 1 states the rejection index as $n \leftarrow \min\big(\{i-1 \mid 1 \le i \le \gamma,\ r_i > p_i(x)/q_i(x)\} \cup \{\gamma\}\big)$ and the resample as $p'(x) \leftarrow \text{norm}(\max(0, p_{n+1}(x) - q_{n+1}(x)))$, which is the same thing written compactly.

### Why it's exact — the intuition behind the proof

The theorem ([Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)) is: *for any distributions $p$ and $q$, tokens produced by speculative sampling are distributed identically to tokens sampled from $p$ alone.* Not approximately. Identically.

The reason is a two-case decomposition. The probability that a given token $x$ comes out of one round is (probability the draft proposes it) × (probability we accept it) + (probability we reject something) × (probability the residual gives us $x$):

$$q(x)\cdot\min\!\left(1,\frac{p(x)}{q(x)}\right) \;+\; \Pr[\text{reject}]\cdot p'(x) \;=\; \min\big(p(x), q(x)\big) + \big(p(x) - \min(p(x),q(x))\big) = p(x)$$

The first term captures $\min(p, q)$ — the mass the two models agree on — and the residual is constructed to be exactly the shortfall $(p - q)_+$ normalized, so the rejection branch contributes precisely the mass the acceptance branch was missing. Two pieces, they sum to $p$, done. If you can sketch that on a whiteboard you're ahead of almost everyone.

**Why this matters so much:** it means speculative decoding is not a quality/speed trade-off. It is *free latency*, up to compute overhead. You can enable it in production without an eval regression, which is a completely different risk profile from quantization or eviction. That framing is the thing to lead with.

*One practical caveat to have ready:* the guarantee is over the distribution, so with a fixed random seed you will not get bit-identical output to non-speculative decoding, and under greedy decoding the acceptance rule degenerates to "accept iff the draft's argmax matches the target's argmax," which is exact in a stronger, deterministic sense. Also, in real implementations FP16 non-determinism in how $p$ is computed with different batch shapes can cause tiny divergences. Say "distributionally exact" rather than "bit-identical" and you're safe.

### What determines the speedup

Let $\alpha$ be the expected per-token acceptance rate and $c$ the ratio of draft-model cost to target-model cost per token. Theorem 3.8 of the paper gives the expected wall-clock improvement factor:

$$\text{speedup} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)\,(\gamma c + 1)}$$

| Symbol | Meaning |
|---|---|
| $\alpha$ | probability a proposed token is accepted (empirical, workload-dependent) |
| $\gamma$ | number of tokens the draft proposes per iteration |
| $c$ | draft cost / target cost per token — e.g. a 1B draft for a 70B target is roughly $c \approx 0.015$, but overheads push the effective value higher |

The numerator $\frac{1-\alpha^{\gamma+1}}{1-\alpha} = 1 + \alpha + \alpha^2 + \cdots + \alpha^\gamma$ is the **expected number of tokens produced per iteration** — a geometric sum, because you need $i$ consecutive acceptances to reach token $i$. The denominator is the cost of an iteration in units of target-forward-passes: one target pass plus $\gamma$ draft passes.

**Worked table** (computed, not estimated):

| $\alpha$ | $\gamma$ | $c$ | Expected tokens/iter | Speedup |
|---|---|---|---|---|
| 0.5 | 3 | 0.05 | 1.88 | 1.63× |
| 0.5 | 7 | 0.15 | 1.99 | **0.97× (a loss)** |
| 0.7 | 3 | 0.05 | 2.53 | 2.20× |
| 0.7 | 5 | 0.05 | 2.94 | 2.35× |
| 0.7 | 5 | 0.15 | 2.94 | 1.68× |
| 0.8 | 5 | 0.05 | 3.69 | 2.95× |
| 0.8 | 7 | 0.05 | 4.16 | 3.08× |
| 0.9 | 7 | 0.05 | 5.70 | 4.22× |
| 0.9 | 7 | 0.15 | 5.70 | 2.78× |

Three lessons fall straight out of that table. **First, $\alpha$ dominates everything** — going from 0.7 to 0.9 nearly doubles the speedup at fixed $\gamma$ and $c$. That's why the whole research direction has been about raising acceptance rates rather than raising $\gamma$. **Second, there's an optimal $\gamma$ and going past it hurts**, because the geometric sum saturates at $1/(1-\alpha)$ while the cost term $\gamma c$ keeps growing linearly. At $\alpha = 0.5$ and $c = 0.15$, $\gamma = 7$ is actually *slower than not speculating at all*. **Third, $c$ matters more than people expect** — an insufficiently cheap draft eats the win, which is the argument against using a 7B draft for a 13B target.

### The variants, and why each exists

**Vanilla two-model.** A separate small model from the same family — Llama 3.2 1B drafting for Llama 3.1 70B. Needs no training. Downsides: you have to fit and serve a second model, and $\alpha$ is limited because the two models have genuinely different representations. Typical $\alpha$ in the 0.6–0.8 range for in-distribution text.

**Self-speculative / layer skipping.** Use a subset of the target's own layers as the draft (e.g. every other layer, with early exit). No extra parameters, no extra memory. Acceptance is usually lower.

**Medusa** (Cai et al. 2024). Bolt several extra decoding heads onto the target's final hidden state, each trained to predict the token at offset $+2$, $+3$, and so on. Combine their proposals into a tree of candidates and verify the whole tree in one target pass with a specialized attention mask. No separate model, and roughly 2× speedup with a small amount of head training.

**EAGLE / EAGLE-2 / EAGLE-3.** The strongest line of work. The key idea is that drafting at the *feature* level — predicting the target's penultimate-layer hidden states rather than tokens — is a better-conditioned problem, and it lets the drafter share the target's representation, which pushes $\alpha$ much higher. EAGLE-2 adds context-aware dynamic draft trees. [EAGLE-3](https://arxiv.org/abs/2503.01840) drops the feature-prediction constraint in favor of direct token prediction with multi-layer feature fusion plus "training-time test" so that draft quality scales with training data; it reports **up to 6.5× speedup, about 1.4× over EAGLE-2**, and a 1.38× throughput gain at batch 64 in SGLang. Note that "up to 6.5×" is a best-case benchmark figure on favorable tasks; treat single-digit multiples on your own workload as needing measurement.

**Lookahead decoding / n-gram (a.k.a. prompt-lookahead) speculation.** Propose candidates by copying n-grams from the prompt or from earlier output, or by Jacobi iteration. Zero extra parameters, zero training. Surprisingly effective for summarization, code editing, and RAG, where output heavily quotes input — and useless for open-ended generation. Cheap enough to always enable for the right workload.

**Tree / multi-candidate speculation.** Instead of one linear chain of $\gamma$ tokens, propose a tree of alternatives and verify all branches in one pass using a block-diagonal-ish attention mask. Raises the expected accepted length for the same target-pass budget. Used by Medusa, EAGLE-2/3, and SpecInfer.

### The trade-off everyone forgets

Speculative decoding improves **per-request latency** and can *reduce* aggregate throughput. Here's why: it does strictly more total compute — every rejected token's target-model FLOPs are thrown away, plus all the draft-model work. In the memory-bound regime those extra FLOPs are free, so you win. But on a server already running batch 200, you are *not* memory-bound; the tensor cores are busy, and the wasted FLOPs come directly out of someone else's tokens.

So the correct production answer is: **speculative decoding is a latency optimization for low-to-medium batch sizes**, and good schedulers turn it on adaptively — speculate aggressively when the batch is small, reduce $\gamma$ as the batch grows, disable it under heavy load. If you say this unprompted, you've demonstrated you understand the regime dependence rather than the technique in isolation.

> **Why the interviewer asks this.** It was *the* algorithm question of 2024–2025 and it tests three things at once: do you understand that decode is memory-bound (the motivation), can you state a rejection-sampling argument (the correctness), and do you know that latency and throughput can move in opposite directions (the systems judgment).

> **Saying it out loud.** "Decode is memory-bound, so a forward pass over one token and a forward pass over five tokens cost nearly the same — you read the same weights either way. Speculative decoding exploits that. A small draft model proposes gamma tokens autoregressively, then the big model verifies all of them in a single pass. You accept token i with probability min of one and p over q, and on the first rejection you resample from the normalized positive part of p minus q. That rule is what makes it exact — the accepted-mass term gives you min of p and q, the residual term gives you exactly the shortfall, and they sum to p. So it's not a quality trade-off; you get the target model's distribution, just faster. Speedup is roughly the expected accepted length over the cost of an iteration, so with 70% acceptance and a cheap draft you're around 2 to 2.5x. The thing people miss is that it costs extra total compute on rejected tokens, so it helps latency at small batch and can actually hurt throughput on a heavily loaded server. Good schedulers make it adaptive."

---

## 11. Quantization

### The intuition

A weight stored in bf16 uses 2 bytes. Store it in 4 bits instead and it uses 0.5 bytes. Section 3 established that decode time is bytes-read divided by bandwidth, so cutting bytes by 4× cuts decode time by up to 4×. That is the entire motivation, and it's a bigger deal than the memory saving, though the memory saving matters too — 4-bit weights let a 70B model fit on one 80 GB GPU instead of needing two, which removes tensor-parallel communication from the critical path entirely.

The analogy: you're not summarizing the book, you're printing it in a smaller font. The information is mostly still there; the question is whether the reader can still make out the fine print, and where exactly it gets illegible.

### Four different things you can quantize

This is the taxonomy that keeps discussions from getting muddled, and it *is* a genuine list, so bullets are appropriate:

- **Weights.** Static, known offline, quantized once. Easiest, biggest win for memory-bound decode. This is where nearly all production value lives.
- **Activations.** Dynamic, differ per input, must be quantized on the fly. Needed if you want to use low-precision *tensor cores* (int8/fp8/fp4 matmul) rather than just saving memory. Harder because of outliers.
- **KV cache.** Static once written, but written continuously. Halving KV precision roughly doubles your batch size, which is often worth more than anything you do to the weights.
- **Gradients / optimizer state.** Training-only, irrelevant to inference. Mentioned to close the taxonomy.

The shorthand `W{x}A{y}` means $x$-bit weights and $y$-bit activations. **W4A16** — 4-bit weights, 16-bit activations — is the workhorse of open-source LLM inference: you dequantize the weights to bf16 in registers just before the matmul, so you save the HBM read (which is what mattered) but do the arithmetic in bf16. **W8A8** quantizes both and can therefore run on int8 tensor cores for roughly 2× the FLOPs. **W4A4** on Blackwell FP4 tensor cores is the frontier.

### The number formats

| Format | Bits | Notes |
|---|---|---|
| FP32 | 32 | reference only; essentially never used for LLM inference |
| BF16 | 16 | the default baseline; 8 exponent bits, so same dynamic range as FP32 |
| FP16 | 16 | 5 exponent bits — narrower range, occasional overflow issues; still common |
| FP8 E4M3 | 8 | native on Hopper and later; the usual choice for weights and activations |
| FP8 E5M2 | 8 | more exponent range, less mantissa; used where dynamic range matters |
| INT8 | 8 | symmetric or asymmetric integer with a scale; int8 tensor cores are ~2× bf16 |
| INT4 / NF4 | 4 | 4× memory saving; needs group-wise scales to work well |
| NVFP4 | 4 | Blackwell; 16-element blocks with an FP8 (E4M3) per-block scale |
| MXFP4 | 4 | OCP open standard; 32-element blocks with an E8M0 (exponent-only) scale |

**FP8 versus INT8** is a question that comes up. Floating point spends bits on an exponent, so it has non-uniform resolution — fine near zero, coarse far away — which happens to match how neural network weights and activations are distributed. Integer formats have uniform resolution and therefore need a well-chosen scale per group and are much more sensitive to outliers. In practice FP8 needs far less calibration care than INT8 for equivalent quality, which is why it became the default on Hopper. INT8 remains relevant on hardware without FP8 units.

**NVFP4 versus MXFP4** is the current version of that debate one level down. NVFP4 uses 16-element blocks with an FP8 scale; MXFP4 uses 32-element blocks with an exponent-only E8M0 scale. NVFP4 therefore carries twice the scale overhead (0.5 bits per weight versus 0.25) and gets finer-grained, higher-resolution scales, and generally lands closer to bf16 quality. MXFP4 is the open Open Compute Project standard and runs on AMD MI355X as well as Blackwell, so it's the portable choice. Both execute on the same FP4 tensor cores on Blackwell. *Both require Blackwell or MI355-class hardware — neither runs natively on H100/H200/A100* — so treat vendor throughput claims here as needing verification against your actual SKU.

### The activation-outlier problem, and SmoothQuant

Here is the single most important quality issue in quantization, and it's worth explaining properly because it's a favorite follow-up.

Weight distributions in trained transformers are well-behaved — roughly Gaussian, no wild tails. Activations are not. In LLMs above a few billion parameters, a small number of *specific hidden dimensions* consistently carry values 20× to 100× larger than the typical channel. These are systematic, not random: the same channels light up across different inputs, and they appear to serve a functional role related to attention sinks.

Why that breaks naive INT8: quantization picks a scale from the tensor's dynamic range. One channel at 100× forces a scale 100× too coarse for every other channel, so nearly all your activations collapse into a handful of integer levels. Quality falls off a cliff.

Three families of fix, all worth naming:

**Keep the outliers in high precision.** LLM.int8() (Dettmers et al. 2022) detects outlier channels at runtime and does those columns in fp16 while the rest go int8. Works, but the mixed-precision path costs speed. SpQR extends the idea to weights, keeping the worst outliers in fp16 alongside a sparse structure.

**Move the problem.** **SmoothQuant** (Xiao et al. 2022) is the elegant one. Since $Y = XW$, you can insert a diagonal rescaling: $Y = (X \operatorname{diag}(s)^{-1})(\operatorname{diag}(s) W)$. Choose $s$ per channel to divide the outlier magnitude out of $X$ and multiply it into $W$ — and because $W$ was well-behaved to begin with, it can absorb the extra range without trouble. Both tensors end up quantization-friendly, and the rescaling folds into the preceding LayerNorm so it's free at runtime. This is what makes W8A8 viable.

**Rotate the problem away.** Apply a random orthogonal (typically Hadamard) transform before quantizing so that outlier energy is spread across all channels; the transform's inverse folds into adjacent weights. This is the idea in QuaRot and SpinQuant, and FlashAttention-3's "incoherent processing" for FP8 attention is the same trick applied to Q and K.

### Calibration-based weight quantization methods

| Method | Core idea | Typical use |
|---|---|---|
| **Round-to-nearest (RTN)** | just round, with per-group scales | baseline; fine at 8 bits, poor at 4 |
| **GPTQ** (Frantar et al. 2022) | layer-wise second-order (Hessian-based) error compensation — quantize weights one column at a time and update the remaining ones to absorb the error | W4A16 with a few hundred calibration samples |
| **AWQ** (Lin et al. 2023) | *activation-aware*: identify the ~1% of weight channels that multiply large activations and protect them by per-channel scaling before RTN | W4A16; faster to apply than GPTQ, often better generalization |
| **SmoothQuant** (Xiao et al. 2022) | migrate activation outliers into weights via diagonal rescaling | W8A8, to enable int8 tensor cores |
| **SpQR** (Dettmers et al. 2023) | sparse-quantized: bulk in ~3–4 bits, outlier weights kept in fp16 | best-quality sub-4-bit |
| **NF4** (Dettmers et al. 2023, QLoRA) | 4-bit format with quantiles of a normal distribution as levels — information-theoretically optimal for Gaussian weights | quantized fine-tuning, and inference where NF4 kernels exist |
| **QuaRot / SpinQuant** | rotate with Hadamard transforms to eliminate outliers, then quantize | W4A4 |

The unifying insight behind GPTQ and AWQ is that **not all weights matter equally**, and the two methods operationalize that differently: GPTQ uses curvature of the loss (a Hessian approximation) to decide how to compensate; AWQ uses the magnitude of the activations a weight will multiply. AWQ is cheaper to run and tends to be more robust out of distribution; GPTQ has been around longer and has broader kernel support. Both target W4A16 and in practice land within a point or two of each other on standard benchmarks.

### KV cache quantization, briefly

Because KV memory sets batch size (section 5), this is often the highest-value quantization you can do. Worked for our Llama 3.1 70B configuration at 6,500-token sequences, in a 164.5 GiB aggregate KV budget:

| KV precision | Per token | Per 6,500-token sequence | Concurrent sequences |
|---|---|---|---|
| fp16/bf16 | 320 KiB | 1.98 GiB | **82** |
| fp8 / int8 | 160 KiB | 0.99 GiB | **165** |
| int4 | 80 KiB | 0.50 GiB | **331** |

Doubling batch size in the memory-bound regime roughly doubles throughput, so **FP8 KV cache is close to a free 2× on tokens per dollar**, which is why it went from research to default so fast. FP8 KV on Hopper is the easy version (native format, minimal quality loss reported). INT4 KV needs care: K and V have different distributional shapes, which is why **KIVI** quantizes K per-channel and V per-token, and **KVQuant** adds outlier handling. NVIDIA has published work on NVFP4 KV cache for long-context, large-batch serving on Blackwell — vendor-published, so verify the claims on your workload.

### What to actually say about quality

Be precise and be honest, because interviewers can smell overclaiming here. Reasonable, widely-reproduced statements as of 2026:

- FP8 weights and activations: quality loss usually within measurement noise on standard benchmarks. Safe default on Hopper and later.
- W4A16 via GPTQ or AWQ with group size 128: typically small perplexity increase and near-parity on most benchmarks for models above ~7B. Larger models tolerate it better.
- W4A4: still needs rotation-based methods to be respectable, and quality varies by task.
- **Aggregate benchmarks hide task-specific damage.** Multi-step reasoning, long-chain arithmetic, code generation, and low-resource languages degrade more than MMLU suggests. Always evaluate on the workload you actually serve, not the leaderboard.
- Calibration-set mismatch is a real failure mode: calibrate on data resembling production traffic.

> **Why the interviewer asks this.** Quantization is the most commonly deployed inference optimization, so knowing the vocabulary is table stakes. The discriminating question is "why does INT8 sometimes hurt," and the answer they want is activation outliers plus the SmoothQuant mechanism.

> **Saying it out loud.** "There are four separable things you can quantize — weights, activations, the KV cache, and optimizer state, which doesn't apply at inference. Weights are the easy win because decode is bandwidth-bound, so a 4x smaller weight read is up to a 4x faster step. W4A16 with GPTQ or AWQ is the standard open-source setup: 4-bit weights, dequantized in registers, bf16 math. If you want low-precision *math* you have to quantize activations too, and that's harder because LLM activations have systematic outlier channels 50 to 100 times larger than typical, which wrecks the scale for everything else. SmoothQuant fixes that by inserting a diagonal rescaling that moves the outlier magnitude from the activations into the weights, which can absorb it — and it folds into the preceding LayerNorm so it's free. And honestly the highest-value quantization is often the KV cache, not the weights: going from fp16 to fp8 KV doubles how many sequences fit, which roughly doubles throughput in the memory-bound regime."

---

## 12. Multi-GPU inference

A 70B model in bf16 is 141 GB of weights and does not fit on an 80 GB H100. Even when a model does fit, you may want more aggregate bandwidth. So you shard — and there are four ways, which are orthogonal and routinely combined.

### Tensor parallelism (intra-layer)

Split each weight matrix across GPUs. The canonical Megatron-style scheme: for the FFN, shard the first matrix column-wise and the second row-wise, so each GPU computes a partial output and you finish with one **all-reduce** per block. For attention, shard by head — each GPU owns a subset of heads and its own slice of the KV cache.

The cost is communication on the critical path. Every transformer block needs an all-reduce, so a 80-layer model does on the order of 160 collectives per forward pass. Inside a node over NVLink at 900 GB/s this is tolerable; across nodes over InfiniBand it usually is not, which is the practical rule: **keep tensor parallelism within a node**, TP≤8 on an 8-GPU box.

The benefit beyond capacity is that aggregate bandwidth scales with TP degree, so the decode floor improves nearly linearly. From the calculation in section 3: 70B bf16 has a floor of 42.1 ms/token on one (hypothetical) H100, 21.1 ms at TP=2, 10.5 ms at TP=4. That's why you sometimes run higher TP than memory requires — you're buying bandwidth, not space.

One wrinkle worth knowing: TP must divide the number of KV heads, or you replicate them. Llama 3.1 70B has 8 KV heads, so TP=8 gives one KV head per GPU and TP=16 would force replication.

### Pipeline parallelism (inter-layer)

GPU 0 holds layers 1–20, GPU 1 holds 21–40, and so on; activations flow forward through the pipeline. Communication is tiny — just the activations at the boundary, not an all-reduce over the hidden state — so it works across slow interconnects, which is its main appeal.

The problem for inference is **pipeline bubbles**. In decode, a single token's forward pass must traverse all stages sequentially, so with $S$ stages your per-token latency includes $S$ hops and each GPU is idle for $(S-1)/S$ of the time unless you have enough independent microbatches in flight to fill the pipe. You can fill it with concurrent requests, but you've now added latency to every one. So: **pipeline parallelism is a training technique that shows up in inference mainly when you're out of options** — a model too big for one node's TP group, or a cheap multi-node setup with no fast interconnect. Sarathi-Serve's largest reported gain (5.6× for Falcon-180B) was in exactly this setting, because chunked prefill helps fill pipeline bubbles.

### Expert parallelism (for MoE)

A mixture-of-experts layer has many FFN "experts" and a router that sends each token to a few of them. Expert parallelism puts different experts on different GPUs and routes tokens across the network. DeepSeek-V3 has 256 routed experts plus 1 shared expert per layer, activating 8 routed experts per token — 37B of 671B parameters active.

The characteristic problems are **load imbalance** (a popular expert becomes a hotspot, and the step waits for the slowest GPU) and **all-to-all communication cost** (tokens have to physically travel to their experts and their results travel back). Mitigations include capacity factors with token dropping, auxiliary load-balancing losses at training time, expert replication for hot experts, and — DeepSeek's approach — restricting how many nodes a token can route to. MoE serving is its own specialization and is increasingly what "inference infra" interviews at frontier labs are actually about.

### Data parallelism (across replicas)

Each GPU or each TP group holds a complete copy of the model and serves independent requests. Embarrassingly parallel, no communication, linear scaling. This is how you scale *throughput* once you've chosen a per-replica configuration, and it's what the capacity plan in section 15 multiplies out.

### How they compose in practice

The standard production shape is: **tensor parallelism within a node to fit the model and buy bandwidth, data parallelism across nodes to scale throughput, expert parallelism if the model is MoE, and pipeline parallelism only if you're forced into it.** So a 70B service might be TP=4, DP=13, which is 52 GPUs — exactly what section 15 arrives at.

A subtlety worth mentioning if pressed: TP degree is a latency-versus-efficiency dial. Higher TP lowers TPOT (more aggregate bandwidth per request) but wastes capacity (more communication overhead, smaller per-GPU matmuls, and the same KV cache split more ways). If you're latency-critical, run higher TP than you need; if you're throughput-critical, run the minimum TP that fits and add replicas.

> **Why the interviewer asks this.** To check that you know parallelism strategies are not interchangeable and that the inference answer differs from the training answer. Saying "pipeline parallelism, because that's what we used in training" is a classic miss.

> **Saying it out loud.** "Four axes. Tensor parallelism splits each weight matrix across GPUs with an all-reduce per block — it's what you use for inference, because it also multiplies your aggregate HBM bandwidth, which is what sets decode latency. A 70B in bf16 has a 42-millisecond-per-token floor on one GPU's bandwidth and about 10.5 at TP=4. But the all-reduce is on the critical path, so you keep TP inside a node over NVLink. Pipeline parallelism splits by layer, communicates almost nothing, and is great for training but bad for low-latency inference because of bubbles — a token has to walk through every stage. Expert parallelism is for MoE, where the hard parts are load imbalance and all-to-all. And data parallelism across replicas is how you actually scale throughput. Typical production shape is TP inside the node, DP across nodes."

---

## 13. The metrics: TTFT, TPOT, and the latency–throughput frontier

### The four numbers you must be able to define

**TTFT — time to first token.** Wall clock from request arrival to the first token reaching the client. It is queueing delay plus tokenization plus prefill. It scales with prompt length (because prefill does) and with load (because queueing does). This is the number a user experiences as "did it hear me."

**TPOT — time per output token**, also called ITL, inter-token latency. The average gap between consecutive output tokens during decode. Set by bytes-read-per-step over bandwidth, so it degrades as batch size grows and as contexts get longer. This is the number a user experiences as "is it typing fast enough."

**End-to-end latency.** $\text{TTFT} + (\text{output tokens} - 1) \times \text{TPOT}$. For our example: $452 + 499 \times 30 \approx 15.4$ s. Notice how completely TPOT dominates for long outputs — a 10 ms improvement in TPOT is worth 5 seconds here, while a 100 ms improvement in TTFT is worth 0.1 s. Know which one your workload actually cares about.

**Throughput**, and specifically **goodput**. Raw throughput is total output tokens per second across all users. Goodput is the subset of that produced by requests which *met their SLO*. This distinction matters enormously: a server can report high throughput while every individual user has an unusable 300 ms TPOT. Goodput is the metric the disaggregated-serving papers optimize and the one you should name if asked "what would you put on the dashboard."

### Targets worth memorizing

| Application | TTFT | TPOT | Why |
|---|---|---|---|
| Streaming chat | < 500 ms | < 50 ms | 50 ms is 20 tok/s ≈ 15 words/s, comfortably above reading speed |
| Voice / real-time conversation | < 200 ms | < 30 ms | turn-taking in human conversation breaks down past ~250 ms |
| Coding assistant (inline) | < 200 ms | < 20 ms | competing with the user's own typing |
| Batch / offline | don't care | don't care | maximize throughput; use huge batches and offline schedulers |
| Agentic / tool-calling | < 1 s | < 50 ms | many sequential LLM calls, so per-call latency compounds |

These are widely-cited industry rules of thumb rather than measured constants; treat them as design targets, not facts.

### The tension, explained mechanically

Increase batch size $B$. What happens?

Bytes read per decode step goes up a little — the weight term is constant, the KV term grows with $B$. Tokens produced per step goes up by exactly $B$. So **throughput rises nearly linearly in $B$** until you hit either the compute roofline or KV memory.

But each step now takes longer (more KV to read, more compute), and each individual user gets exactly one token per step. So **TPOT rises with $B$**. And because admission is capped by memory, larger in-flight batches also mean deeper queues for new arrivals, so **TTFT rises with $B$** too.

That's the frontier. You don't get to pick a point on a line; you pick a point on a curve, and the operating discipline is:

1. Fix the SLO — say p99 TPOT < 50 ms and p99 TTFT < 800 ms.
2. Load-test with *realistic* traffic — real prompt-length and output-length distributions, not fixed-length synthetic requests, because the variance is what breaks schedulers.
3. Find the largest sustained request rate at which the SLO still holds. That rate is your per-replica capacity.
4. Divide target QPS by it, add headroom for traffic spikes and for the fact that p99 degrades sharply near saturation, and that's your replica count.

Everything in this document — PagedAttention, continuous batching, chunked prefill, quantization, speculation, disaggregation — is an attempt to push that curve outward so that the largest SLO-compliant rate is higher.

### The classic interview question, worked

*"Your service has 200 ms TTFT and 30 ms TPOT. Reduce TTFT without hurting TPOT."*

The trap is to suggest lowering batch size. That does reduce queueing and therefore TTFT, but it also reduces throughput, and if the interviewer's premise is fixed hardware it makes the problem worse under load. The good answers:

- **Chunked prefill.** Splits long prefills into ~512-token chunks interleaved with decode. It reduces the *variance* in TTFT and eliminates decode stalls; it doesn't reduce the total prefill FLOPs, so the mean TTFT for a very long prompt may not drop much, but tail TTFT does.
- **Prefix caching.** If your prompts share a system prompt or RAG context, cache hits eliminate most of the prefill entirely. This is by far the biggest available TTFT win when the workload allows it, and it costs nothing in TPOT.
- **Better prefill kernels.** FlashAttention-3, FP8 prefill. Directly reduces prefill time.
- **Disaggregate.** Move prefill to dedicated compute-optimized workers so prefill never contends with decode. Best structural fix, most operational complexity.
- **Reduce the prompt.** Retrieve less, compress the system prompt, trim conversation history. Unglamorous and frequently the highest-ROI change.
- **Raise TP degree for the prefill workers.** More aggregate FLOPs on the prefill path.

*"And now reduce TPOT without hurting TTFT?"* Quantize weights (fewer bytes per step), quantize the KV cache (fewer bytes and more batch headroom), enable speculative decoding (more tokens per weight read), increase TP degree (more aggregate bandwidth), use GQA/MLA if you get to pick the architecture.

> **Why the interviewer asks this.** These are the two numbers on every serving dashboard, and the question tests whether you understand they're controlled by different mechanisms. Conflating them is the tell of someone who hasn't operated a service.

> **Saying it out loud.** "TTFT is queueing plus prefill, so it scales with prompt length and load. TPOT is bytes read per decode step over bandwidth, so it scales with batch size and context length. They're controlled by completely different things, which is why you can't optimize 'latency' as one number. For a 500-token response, TPOT dominates end-to-end — at 30 milliseconds that's 15 seconds of decode versus half a second of prefill. And the real serving discipline is that you don't optimize throughput, you optimize goodput: fix an SLO, load-test with realistic length distributions, find the highest request rate that still meets p99, and that's your per-replica capacity."

---

## 14. Prompt caching and prefix sharing

### The observation

Real traffic is enormously repetitive at the front of the prompt. A production chat app sends the same 2,000-token system prompt to every request. An agent sends the same 8,000 tokens of tool schemas on every step of every trajectory. A multi-turn conversation resends the entire history each turn — turn 10 of a conversation re-prefills everything from turns 1 through 9. A RAG system serving a popular query retrieves the same documents for many users.

And prefill is the expensive, compute-bound part. If the KV cache for a prefix has already been computed, and the prefix is *bit-identical*, then those K and V vectors are identical too — causality guarantees it. You can skip that work entirely.

### Two scales of the same idea

**Within a batch / within a server: prefix sharing.** PagedAttention makes this nearly trivial. Maintain a hash of block contents (or of the token sequence up to each block boundary) in a radix tree or hash map. On a new request, walk the tree to find the longest matching prefix, point the new sequence's block table at those existing physical blocks, bump their reference counts, and prefill only the suffix. SGLang calls its version **RadixAttention** and made it a headline feature; vLLM ships `enable_prefix_caching`. The saving is exact and lossless.

The concrete arithmetic: a request with a 2,000-token cached system prompt and 200 new tokens does prefill on 200 tokens instead of 2,200 — a 91% reduction in prefill FLOPs and therefore in TTFT contribution. For an agentic workload where the tool definitions are 8,000 tokens and the new content is 300, the saving is 96%.

**Across requests and across time: provider prompt caching.** API providers extend this to a persistent cache with an explicit TTL, and price it. Anthropic's published pricing: **cache writes cost 1.25× the base input token price for a 5-minute TTL or 2× for a 1-hour TTL, and cache reads cost 0.1× the base price** — a 90% discount on cached input. Minimum cacheable prefix length varies by model (512 tokens for the Claude Opus 5 generation, higher for some earlier models) ([Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). OpenAI and Google offer analogous mechanisms with their own multipliers; verify current numbers before quoting them, since pricing changes.

The break-even arithmetic is worth being able to do out loud. If you write once at 1.25× and read $n$ times at 0.1×, versus paying 1.0× every time:

$$1.25 + 0.1n \;<\; 1 + n \quad\Longrightarrow\quad n > \frac{0.25}{0.9} \approx 0.28$$

So caching pays off from the *second* use, essentially always — the only case where it loses is a prefix used exactly once. For an agent doing 40 steps with an 8,000-token preamble, the input cost of the preamble drops from $40 \times 1.0 = 40$ units to $1.25 + 39 \times 0.1 = 5.15$ units, an 87% saving on that portion.

### The gotchas

**The prefix must match exactly, from the very first token.** A timestamp, a session ID, or a randomized greeting at the top of your system prompt invalidates everything after it. The practical rule is: **order your prompt from most static to most dynamic.** System instructions, then tool definitions, then retrieved documents, then conversation history, then the current user message. This one piece of prompt-engineering advice is worth more in production cost than most model choices, and stating it unprompted is a strong signal.

**Cache eviction is a memory-management problem.** On a single server, cached prefix blocks compete for the same HBM as active sequences' KV. LRU eviction is standard, and a large cache can reduce your effective batch size. Some systems tier the cache to CPU RAM or NVMe, which trades a PCIe transfer (fast, 128 GB/s) against a prefill recomputation (slow, but pure GPU compute) — a genuine engineering trade-off with no universal answer.

**Cross-user sharing has security implications.** If the cache is keyed only on content, cache-hit *timing* can leak whether another user has recently sent a particular prefix. Providers scope caches per organization for this reason. Mentioning this is a good way to show product judgment.

**It doesn't help decode at all.** Prompt caching removes prefill work. If your workload is short prompts and long outputs, it's nearly worthless. Match the optimization to the shape of your traffic.

> **Why the interviewer asks this.** It's the cheapest big win in production LLM serving, and knowing it signals you've thought about cost, not just latency. The "order static content first" answer is the specific thing that marks experience.

> **Saying it out loud.** "Prefill is the expensive compute-bound phase, and real traffic repeats the front of the prompt constantly — same system prompt, same tool definitions, same conversation history each turn. Since K and V for a position depend only on tokens up to that position, an identical prefix has an identical KV cache, so you can reuse it. Inside a server that's prefix sharing: PagedAttention lets you hash block contents into a radix tree and just point a new request's block table at existing blocks with a refcount. Across requests, providers expose it as prompt caching — Anthropic charges 1.25x to write and 0.1x to read, so it breaks even on the second use and saves close to 90% on a repeated preamble. The main practical rule is to order your prompt from most static to most dynamic, because a timestamp at the top invalidates everything after it."

---

## 15. Capacity planning, worked end to end

This is the exercise. Someone hands you a model, a GPU, a workload, and a target, and asks how many GPUs. Here is the full derivation with every number computed. If you can walk a whiteboard through this, you can hold your own in a systems interview.

### The problem statement

- **Model:** Llama 3.1 70B, bf16. 80 layers, 8 KV heads, $d_{\text{head}} = 128$, $d_{\text{model}} = 8192$, 70.6B parameters.
- **Hardware:** H100 SXM 80 GB, 3.35 TB/s HBM, 989.5 TFLOP/s dense bf16, NVLink within an 8-GPU node.
- **Workload:** average prompt 6,000 tokens, average output 500 tokens.
- **SLO:** p99 TPOT < 50 ms, TTFT < 1 s.
- **Target:** 20 requests/second sustained.

### Step 1 — Choose a parallelism configuration

Weights are $70.6 \times 10^9 \times 2 = 1.412 \times 10^{11}$ bytes $= 141.2$ GB $= 131.5$ GiB. That exceeds 80 GB, so TP ≥ 2 is forced. TP=2 leaves $80 - 65.8 = 14.2$ GiB per GPU for KV, which is very tight. **Choose TP=4**: it fits comfortably, gives 13.4 TB/s aggregate bandwidth, and stays inside a node.

Weights per GPU: $131.5 / 4 = 32.9$ GiB.

### Step 2 — Compute the KV budget

Reserve overhead for activations, workspace, the CUDA context, NCCL buffers, and fragmentation headroom. **6 GiB per GPU** is a reasonable engineering allowance (verify empirically for your framework — this is the softest number in the whole calculation).

$$\text{KV per GPU} = 80 - 32.9 - 6 = 41.1\ \text{GiB}$$
$$\text{Aggregate KV budget} = 41.1 \times 4 = 164.5\ \text{GiB}$$

(Under tensor parallelism each GPU holds 2 of the 8 KV heads, so summing across the group is the right way to get the total cache capacity.)

### Step 3 — KV cost per sequence, and max batch

Per token: $2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ bytes $= 320$ KiB.

Average sequence length $= 6000 + 500 = 6500$ tokens.

$$\text{KV per sequence} = 327{,}680 \times 6500 = 2{,}129{,}920{,}000\ \text{bytes} = 1.984\ \text{GiB}$$

$$B_{\max} = \left\lfloor \frac{164.5}{1.984} \right\rfloor = \mathbf{82\ concurrent\ sequences}$$

**This is the number that governs everything downstream.** Compute could support ~300 concurrent sequences (section 4's crossover); memory allows 82. Memory wins, as it almost always does.

### Step 4 — Decode step time and TPOT

Bytes read per GPU per step, assuming the cache runs about 90% full on average:

$$32.9\ \text{GiB (weights)} + 0.9 \times 41.1\ \text{GiB (KV)} = 69.9\ \text{GiB} = 7.50 \times 10^{10}\ \text{bytes}$$

Real kernels don't achieve peak bandwidth. **75% of theoretical** is a reasonable assumption for well-optimized attention and GEMM kernels — measure this on your own stack; it's an assumption, not a fact.

$$t_{\text{step}} = \frac{7.50 \times 10^{10}}{3.35 \times 10^{12} \times 0.75} = 0.0299\ \text{s} = \mathbf{29.9\ ms}$$

**TPOT ≈ 30 ms.** That meets the 50 ms SLO with headroom, which is what you want — if it came out at 48 ms you'd have no room for tail latency.

Note how much of that step time is KV, not weights: 37 GiB of the 69.9 GiB read is KV cache. **At batch 82 and 6.5K context, more than half of your decode bandwidth goes to reading the cache, not the model.** That's a genuinely important observation and it's the direct argument for GQA, MLA, and KV quantization — and a reason the naive "decode time = weights / bandwidth" formula understates reality at scale.

### Step 5 — Decode throughput

Each step produces one token for each of 82 sequences:

$$\text{throughput} = \frac{82\ \text{tokens}}{0.0299\ \text{s}} = 2{,}745\ \text{output tokens/s per 4-GPU group}$$

At 500 output tokens per request, the decode side can retire

$$\frac{2745}{500} = 5.49\ \text{requests/s}$$

### Step 6 — Prefill cost and TTFT

$$\text{FLOPs} = \underbrace{2 \times 70.6\times10^9 \times 6000}_{8.472\times10^{14}} + \underbrace{2 \times 80 \times 6000^2 \times 8192}_{4.719\times10^{13}} = 8.944 \times 10^{14}$$

Aggregate compute: $4 \times 989.5 = 3{,}958$ TFLOP/s dense bf16. Assume **50% MFU** for prefill (again: assumption; well-tuned prefill on H100 with FlashAttention-3 can exceed this, and poorly-tuned setups fall well below).

$$t_{\text{prefill}} = \frac{8.944 \times 10^{14}}{3.958 \times 10^{15} \times 0.5} = 0.452\ \text{s} = \mathbf{452\ ms}$$

TTFT floor is 452 ms plus queueing. Comfortably under the 1 s SLO if queueing stays modest — which it will only if you're not running at saturation.

If prefill were all you did: $1 / 0.452 = 2.21$ requests/s per group.

### Step 7 — Combine, because they share the GPUs

The two phases contend for the same silicon. A useful first-order model: each request consumes 452 ms of exclusive prefill time plus 500 decode-slot-steps, and a decode slot-step costs $t_{\text{step}}/B_{\max}$ of group time.

$$\text{group-seconds per request} = 0.452 + 500 \times \frac{0.0299}{82} = 0.452 + 0.182 = 0.634\ \text{s}$$

$$\text{QPS per group} = \frac{1}{0.634} = \mathbf{1.58\ requests/s}$$

Look at where the time goes: **71% of GPU time is prefill**, even though 97% of wall-clock latency is decode. That inversion is the entire motivation for disaggregated serving (section 16) — the two phases want completely different hardware and completely different batching, and forcing them to share both is why neither is efficient.

Note this model is pessimistic in treating prefill as fully serialized against decode; chunked prefill overlaps them and recovers part of it. It's optimistic in ignoring queueing dynamics and scheduling overhead. Real measured numbers will land in the same order of magnitude, and you should say that out loud rather than pretending to three-digit precision.

### Step 8 — Scale out

$$\text{groups needed} = \left\lceil \frac{20}{1.58} \right\rceil = 13$$

$$\text{GPUs} = 13 \times 4 = \mathbf{52\ H100s}$$

Plus headroom. Since p99 latency degrades sharply as you approach saturation, provision for 60–70% average utilization: call it **16 groups, 64 GPUs**, i.e. eight 8-GPU nodes. For a 10 QPS target the same arithmetic gives 7 groups, 28 GPUs, plus headroom.

### Step 9 — Cost per token

At 2,745 output tokens/s, a group produces $2745 \times 3600 = 9.88$ million output tokens per hour.

| H100 price | Group cost/hr | Cost per 1M output tokens |
|---|---|---|
| \$2.50/GPU-hr | \$10.00 | **\$1.01** |
| \$3.50/GPU-hr | \$14.00 | **\$1.42** |

H100 on-demand rates in 2026 span roughly \$1.50 to \$7.00 per GPU-hour depending on provider, commitment, and region — this is a fast-moving, vendor-specific number, so treat the \$2.50–\$3.50 range as illustrative and price your own contracts. Note also that this attributes all cost to output tokens; a proper model splits it, since prefill is 71% of the GPU time and is what you'd charge input tokens for.

### Step 10 — Now improve it, in priority order

The value of doing the arithmetic is that it tells you where the leverage is.

1. **FP8 KV cache.** KV per token halves to 160 KiB, so KV per sequence halves to 0.99 GiB and $B_{\max}$ goes 82 → **165**. The bytes read per step do *not* change, because you're filling the same 41.1 GiB of physical memory — just with twice as many sequences in it. So step time stays at about 29.9 ms while tokens per step doubles: throughput goes from 2,745 to roughly 5,500 output tokens/s per group, and QPS per group roughly doubles. **A 2× throughput improvement at unchanged TPOT.** This is the single best move available and it's why it's first on the list.
2. **FP8 weights.** Weights per GPU drop to 16.4 GiB, freeing another 16.5 GiB per GPU for KV, and halving the weight term in the step read. Larger batch *and* faster steps. Requires a quality eval, but FP8 is low-risk on Hopper.
3. **Chunked prefill.** Doesn't reduce the 452 ms of prefill FLOPs, but overlaps it with decode instead of serializing, which recovers a large share of step 7's pessimism and dramatically improves TPOT stability. Roughly free.
4. **Prefix caching.** If prompts share a system prompt or RAG context, this attacks the 71% of GPU time that is prefill — potentially the largest win of all, but entirely workload-dependent. Check your traffic first.
5. **Disaggregated prefill/decode.** Structurally correct, operationally expensive. Worth it at scale.
6. **Speculative decoding.** Improves per-user TPOT at low batch; at batch 82 it's marginal and may hurt. Do it if you're latency-bound, not if you're throughput-bound.
7. **W4A16 weights.** 8.2 GiB/GPU of weights, huge KV headroom, potentially TP=1 with data parallelism instead. Biggest raw win, biggest quality risk. Evaluate carefully.

> **Why the interviewer asks this.** This *is* the job. Everything else in this document is background for being able to do this on demand. A candidate who produces a defensible number with stated assumptions, and who knows which assumptions are the shaky ones, is immediately credible.

> **Saying it out loud.** "I'd work it in five steps. First, does the model fit — 70B in bf16 is 141 gigabytes, so I need at least TP=2, and I'd pick TP=4 for bandwidth headroom, which leaves about 33 gibibytes of weights per GPU. Second, KV budget: 80 minus 33 minus about 6 for overhead is 41 per GPU, times four is 164 gibibytes. Third, KV per sequence: Llama 70B is 320 kibibytes per token, times 6,500 tokens is about 2 gibibytes, so I get 82 concurrent sequences — and notice compute would allow around 300, so memory is the binding constraint. Fourth, step time: I read about 70 gibibytes per GPU per step, weights plus cache, at maybe 75% of 3.35 terabytes a second, so 30 milliseconds. That's 82 tokens per 30 milliseconds, so about 2,700 output tokens a second per group. Fifth, prefill is 895 teraflops per request, about 450 milliseconds at 50% MFU, and since the phases share GPUs each request costs roughly 0.63 group-seconds — so about 1.6 QPS per group, 13 groups for 20 QPS, 52 H100s, and I'd provision 64 for headroom. The interesting thing that falls out is that prefill is 71% of the GPU time even though decode is 97% of the latency, which is exactly the argument for disaggregating them."

---

## 16. The frontier: what's live in 2024–2026

Everything above is settled practice. This section is what interviewers at frontier labs use to find the ceiling of your knowledge. Each subsection is short by design — you need to be able to explain the mechanism and the motivation, not reproduce the paper.

### 16.1 MLA — Multi-head Latent Attention (DeepSeek-V2/V3)

**The motivation.** GQA reduces the KV cache by sharing K/V heads across query heads, which costs you representational capacity — eight query heads looking at the same key vector genuinely can't differentiate as much as eight distinct keys. MLA asks whether you can get the memory saving *without* the capacity loss.

**The mechanism.** Instead of caching $K$ and $V$ directly, project the hidden state down to a low-rank latent $c_t^{KV}$ and cache only that:

$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \qquad d_c \ll n_h \cdot d_h$$

At attention time, project back up with learned matrices $W^{UK}$ and $W^{UV}$ to recover per-head $K$ and $V$. Each head gets its *own* up-projection, so heads are not forced to share the same keys the way GQA forces them to — you recover per-head diversity from a shared compressed representation. The paper reports MLA performing *better* than MHA on benchmarks, not merely comparably, which is unusual for a compression technique.

**The RoPE subtlety, which is the thing to know.** Rotary position embedding applies a position-dependent rotation to $K$. That rotation does not commute with the up-projection matrix, so you cannot absorb $W^{UK}$ into the query projection (which is the trick that makes MLA fast) if RoPE is applied after decompression. DeepSeek's fix is a **decoupled** design: each head's key is split into a compressed no-RoPE part (recovered from the latent) and a small full-dimensional RoPE part ($d_h^R = 64$) that is cached separately and uncompressed. This is why every clear explanation of MLA gets complicated at exactly this point, and being able to name the reason is a strong signal.

**The arithmetic.** [DeepSeek-V2's Table 1](https://arxiv.org/pdf/2405.04434) gives KV cache per token in elements:

| Attention | Elements cached per token |
|---|---|
| MHA | $2 n_h d_h l$ |
| GQA | $2 n_g d_h l$ |
| MQA | $2 d_h l$ |
| MLA | $(d_c + d_h^R)\, l \approx \tfrac{9}{2} d_h l$ |

The paper's own framing: MLA needs the same KV cache as **GQA with only 2.25 groups**, while performing better than MHA. So versus GQA-8 the saving is $8/2.25 = 3.6\times$.

For DeepSeek-V3 concretely — 61 layers, $d_c = 512$, $d_h^R = 64$, 128 heads of dimension 128 ([DeepSeek-V3 technical report](https://arxiv.org/pdf/2412.19437)):

$$(512 + 64) \times 61 \times 2\ \text{bytes} = 70{,}272\ \text{bytes} = \mathbf{68.6\ KiB\ per\ token}$$

The MHA counterfactual for that same architecture would be $2 \times 61 \times 128 \times 128 \times 2 = 3{,}997{,}696$ bytes $= 3.9$ MiB per token — a **56.9×** ratio. And per layer, MLA caches 1,152 bytes where Llama 3.1 70B's GQA-8 caches 4,096 bytes.

The consequence: at 128K context, one DeepSeek-V3 sequence's KV cache is 8.58 GiB, versus 40 GiB for Llama 3.1 70B at the same length despite DeepSeek-V3 being nearly ten times larger in total parameters. Combined with MoE sparsity (37B of 671B active), that's how a 671B model ends up cheaper to serve per token than a dense 70B. DeepSeek-V2 reported a **93.3% KV cache reduction** versus DeepSeek-67B, though that figure reflects layer-count and other architectural changes as well as MLA itself.

> **Saying it out loud.** "MLA is the biggest KV-cache win in years. Instead of caching K and V per head, you project the hidden state down to a 512-dimensional latent and cache only that, then use per-head learned up-projections to recover keys and values at attention time. Because each head has its own up-projection you keep per-head diversity, unlike GQA where heads are literally forced to share a key vector — DeepSeek reports it beating MHA on quality. The messy part is RoPE: the rotation doesn't commute with the up-projection, so they split each head into a compressed no-RoPE component and a small 64-dimensional RoPE component that's cached uncompressed. Numerically it's about 68 kibibytes per token for V3 versus 320 for Llama 70B, which is why a 671B model can be cheaper to serve than a dense 70B."

### 16.2 Chunked prefill

**The problem.** A 6,000-token prefill takes 452 ms as one monolithic GPU operation. Run it as a step and every decode request in your batch stalls for 452 ms, blowing their 30 ms TPOT by 15×. This is a **generation stall**, and it's what makes naive continuous batching produce terrible tail latency.

**The mechanism** (Agrawal et al., [Sarathi-Serve, OSDI 2024](https://arxiv.org/abs/2403.02310)). Split the prefill into fixed-size chunks — typically 512 to 2,048 tokens — and process one chunk per iteration, alongside decode tokens from other requests in the same fused batch. Sarathi-Serve calls this **stall-free batching**: you construct each batch with a token budget, fill it first with the decode tokens of running requests, then top it up with as much prefill work as fits.

**The arithmetic.** 6,000 tokens at chunk size 512 is $\lceil 6000/512 \rceil = 12$ chunks, roughly 38 ms of prefill work each. A co-batched decode request now waits at most one chunk — about 38 ms — instead of 452 ms. The prefill's total FLOPs are unchanged (slightly worse, actually, since later chunks must attend to all earlier ones and you lose a little kernel efficiency at smaller shapes), but the *interference* drops by an order of magnitude.

There's a second, subtler benefit: prefill chunks are compute-bound and decode tokens are memory-bound, so a batch containing both has better balanced resource usage than either alone. Sarathi-Serve reports serving-capacity improvements under SLO of **2.6× for Mistral-7B on one A100, 3.7× for Yi-34B on two A100s, and 5.6× for Falcon-180B with pipeline parallelism**, all versus vLLM. Chunked prefill is now standard in vLLM (0.6+), SGLang, and TensorRT-LLM, and is often on by default.

### 16.3 Disaggregated prefill/decode serving

**The observation**, which section 15 derived numerically: prefill is compute-bound and consumed 71% of GPU time; decode is memory-bandwidth-bound and consumed 29%. They want different hardware, different batch sizes, different parallelism degrees, and different kernels. Running both on the same homogeneous pool means neither gets what it wants, and — worse — they interfere with each other's latency.

**The mechanism** (DistServe, [Mooncake](https://arxiv.org/abs/2407.00079) from Moonshot AI's Kimi, and now SGLang's and vLLM's disaggregated modes). Split the cluster:

- **Prefill workers** run large prefill batches, tuned for MFU, possibly at higher TP for FLOPs, possibly with FP8 compute.
- **Decode workers** run large decode batches, tuned for bandwidth and KV capacity, possibly at different TP.
- After prefill completes, the KV cache is **transferred** from prefill worker to decode worker over NVLink or RDMA.

You can then scale the two pools independently to match your workload's prompt-to-output ratio, and tune each pool's SLO separately: prefill workers own TTFT, decode workers own TPOT.

**The costs.** The KV transfer is real — for our 6,000-token request that's 1.83 GiB moving between machines. Over 900 GB/s NVLink it's ~2 ms and irrelevant; over a 100 Gb/s network it's ~150 ms and ruinous. So disaggregation demands a fast fabric, which is exactly why Mooncake is built around an RDMA-based transfer engine and a distributed KV store. Operationally it's also two pools to size, monitor, and autoscale instead of one. Reported gains are commonly quoted as 2–4× higher goodput at fixed SLO, with DistServe reporting substantially more in favorable configurations — treat specific multiples as configuration-dependent and verify.

### 16.4 Advanced speculation: Medusa, EAGLE, lookahead

Covered mechanically in section 10. The frontier framing: the field has converged on the view that **acceptance rate $\alpha$ is the only variable worth optimizing**, because the speedup formula saturates in $\gamma$ but is nearly linear in the accepted length. That's why the progression went from separate draft models ($\alpha \approx 0.6$–$0.8$, limited by representational mismatch) to extra heads on the target (Medusa) to feature-level drafting that shares the target's representation (EAGLE) to trained-in multi-token prediction. DeepSeek-V3 ships an MTP (multi-token prediction) module trained jointly with the model specifically for use as a speculator at inference time — the logical endpoint of this line: make the model's own architecture speculation-friendly.

### 16.5 KV cache eviction

At 128K+ context, even MLA leaves you with a large cache. Eviction drops tokens you judge unimportant. Unlike everything else in this document, **this is lossy** — the evicted KV is gone and cannot be recovered.

**StreamingLLM** (Xiao et al., ICLR 2024) rests on a surprising empirical finding: transformers dump enormous attention mass onto the *first few tokens* of a sequence regardless of their content, apparently because softmax must allocate its mass somewhere and early tokens are visible to every position. These are **attention sinks**. Keep the first ~4 tokens plus a sliding window of recent tokens, evict the middle, and a model trained with a finite context can generate over millions of tokens without the perplexity blowup that naive window attention causes. Note carefully what this does and doesn't give you: fluent unbounded streaming, *not* the ability to recall information from the evicted middle.

**H2O (Heavy Hitter Oracle)** tracks accumulated attention scores per token and evicts the low scorers, adapting to the actual prompt rather than using a fixed positional policy. **SnapKV** compresses at prefill time using observed attention patterns from the end of the prompt. **Quest** takes a different tack — rather than evicting, it keeps everything but retrieves only the query-relevant pages each step, making it lossless-ish at the cost of a selection step.

Rule of thumb: eviction is for very long contexts where you have no alternative. At 8K–32K, GQA plus KV quantization plus PagedAttention is enough, and you shouldn't accept a lossy method for a problem you can solve losslessly.

### 16.6 KV cache quantization

Quantify the win from section 15's table: fp16 → fp8 doubles your batch, fp16 → int4 quadruples it. Method notes:

- **FP8 KV** on Hopper and later is the easy path: native format, minimal quality loss reported, supported in vLLM and TensorRT-LLM. Do this first.
- **KIVI** treats K and V differently, because they have different distributional shapes — K is quantized per-channel, V per-token. This asymmetry is the interesting technical detail and worth naming.
- **KVQuant** adds outlier-aware handling and non-uniform levels to reach usable 3-bit and even 2-bit KV.
- **NVFP4 KV cache** on Blackwell is NVIDIA's current push for long-context, large-batch serving. Vendor-published; verify.

### 16.7 FP4 and FP6 on Blackwell

Blackwell tensor cores support FP4 natively, and the marketing throughput is roughly 2× FP8 and 4× BF16. The formats and their trade-off (NVFP4's 16-element FP8-scaled blocks versus MXFP4's 32-element E8M0-scaled blocks) are in section 11. The practical picture as of 2026: **FP4 weights with FP8 activations** is the emerging production sweet spot, since weights are the bandwidth problem and activations are where outliers live. **W4A4** delivers maximum throughput and needs rotation-based calibration (QuaRot/SpinQuant-style) to hold quality. FP6 exists as a middle option with software support but less hardware acceleration.

Two cautions. First, all of this requires Blackwell or MI355-class hardware; on H100/H200 you get FP8 and no lower. Second, published throughput multiples are vendor benchmarks on favorable shapes — mark them as needing verification, and remember that a 4× FLOP increase only helps in the compute-bound regime, so for decode the real win is the 4× reduction in bytes read, not the FLOPs.

### 16.8 Scheduler internals

The scheduler is where production serving systems actually differ, and "how does the scheduler decide what runs this step" is a great senior-level question. The dimensions:

**Admission policy.** FCFS is fair and predictable and leaves throughput on the table. Shortest-remaining-first minimizes mean latency and starves long requests. Priority tiers let you sell an SLA. Some systems predict output length to admit smarter.

**Memory watermarks.** Admit only if projected KV need fits under a high-water mark, leaving slack so growing sequences don't force preemption. Set it too conservatively and you underutilize; too aggressively and you thrash.

**Preemption and recovery.** Two options when you run out: **swap** the victim's KV blocks to host memory (PCIe at 128 GB/s, so a 2 GiB sequence takes ~16 ms each way) or **discard and recompute** on resume (burns prefill FLOPs but no PCIe). vLLM supports both; recompute often wins for short sequences, swap for long ones. Victim selection is usually the newest or lowest-priority request, to avoid the convoy effect of preempting something that's nearly done.

**Batch composition.** With chunked prefill you get a token budget per step and must decide the prefill/decode mix. Decode-first (prioritize running requests' TPOT) versus prefill-first (prioritize new requests' TTFT) is a direct SLO trade, and some systems make it dynamic based on which SLO is currently at risk.

**Prefix-cache-aware routing.** In a multi-replica deployment, route a request to the replica that already has its prefix cached. SGLang's router does this and it can be worth more than any single-node optimization for agentic workloads, because it converts a cross-replica cache miss into a hit.

### 16.9 Ring Attention and long-context serving

When a sequence's activations or KV cache exceed a single GPU's memory — million-token contexts — you must shard along the *sequence* dimension. **Ring Attention** (Liu et al. 2023) arranges GPUs in a logical ring, gives each a contiguous slice of the sequence, and rotates K/V blocks around the ring while each GPU computes its local queries' attention against whatever K/V block currently sits with it. After a full rotation every query has seen every key. Because it uses the same online-softmax accumulation as FlashAttention, the result is exact, and the K/V communication for step $i+1$ overlaps with the computation for step $i$, so with a fast enough interconnect the communication is hidden entirely.

The cost is that inter-GPU bandwidth becomes a first-class constraint and the ring's latency is $O(\text{ring size})$ hops. It pairs naturally with FlashAttention (same math, one level of the hierarchy up) and is understood to be part of how million-token-context models are trained and served, though specific production details at commercial labs are not public — treat claims about which product uses it as unverified.

### 16.10 A few things not in the original that are worth knowing exist

**Structured output / constrained decoding.** Masking logits to enforce a JSON schema or grammar. Nearly free at the sampling step but the mask *computation* can be expensive; systems precompile grammars into FSMs (Outlines, XGrammar) so the per-step cost is a lookup. Comes up constantly in product-facing roles.

**Multi-LoRA serving.** Serve many fine-tuned adapters over one base model by batching requests with different adapters and applying the low-rank deltas as a grouped GEMM (S-LoRA, Punica). Turns "one deployment per customer" into "one deployment."

**Hybrid and linear-attention architectures.** Mamba/SSM and hybrid models (Jamba, and the Mamba-Transformer hybrids) have constant-size recurrent state instead of a growing KV cache, which changes the memory story fundamentally. If someone asks how to serve 1M context cheaply, "use an architecture whose state doesn't grow" is a legitimate answer.

**Speculative *retrieval* and cascades.** Route easy requests to a small model and hard ones to a large one, with a router or a confidence check. Not the same as speculative decoding — it's approximate — but often the largest available cost win in a product.

---

## 17. Where the numbers come from, and which ones move

Interviewers respect calibration more than confidence. This section is the honest accounting.

### Vendor-published hardware specs (stable, but check the SKU)

| Number | Value | Source | Caution |
|---|---|---|---|
| H100 SXM HBM | 80 GB @ 3.35 TB/s | [NVIDIA H100 page](https://www.nvidia.com/en-us/data-center/h100/) | H100 PCIe and NVL variants differ; PCIe is ~2 TB/s |
| H100 bf16 dense | 989.5 TFLOP/s | NVIDIA datasheet ÷ 2 | datasheets advertise 1,979 **with sparsity** |
| H200 SXM | 141 GB @ 4.8 TB/s, 989.5 TFLOP/s bf16 dense | [NVIDIA H200 page](https://www.nvidia.com/en-us/data-center/h200/) | same compute die as H100 |
| B200 (per GPU) | ~180 GB, ~8 TB/s, ~2.25 PFLOP/s bf16 dense | derived from [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) system specs ÷ 8 | **verify** — SKUs vary (180 vs 192 GB), system FLOPS are sparsity-inclusive |
| H100 SRAM / L2 / SMs | 228 KB per SM (227 KB usable per block), 50 MB L2, 132 SMs | [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) | SM count varies by SKU |
| NVLink 4 / PCIe 5 | ~900 GB/s per GPU / ~128 GB/s | NVIDIA | topology-dependent |

Balance points (295 for H100, 206 for H200, ~281 for B200, 153 for A100) are **computed** from those two columns, not quoted.

### Model architecture (published configs — stable)

Llama 3.1 70B: 80 layers, 64 query heads, 8 KV heads, $d_{\text{head}}=128$, $d_{\text{model}}=8192$, FFN 28672, vocab 128256, max position 131072 ([config.json](https://huggingface.co/HiTZ/Latxa-Llama-3.1-70B-Instruct/blob/main/config.json)). Llama 3.1 8B: 32 layers, same head geometry. DeepSeek-V3: 61 layers, $d_{\text{model}}=7168$, 128 heads × 128 dim, $d_c = 512$, $d_c' = 1536$, $d_h^R = 64$, 671B total / 37B active, 256 routed + 1 shared expert, 8 activated ([technical report](https://arxiv.org/pdf/2412.19437)).

### Paper-reported results (stable claims, but benchmark-specific)

FlashAttention HBM complexity and exactness (Theorems 1–2, [arXiv 2205.14135](https://arxiv.org/abs/2205.14135)). FlashAttention-3's 740 TFLOP/s / 75% utilization / 1.5–2.0× over FA2 ([Dao 2024](https://tridao.me/blog/2024/flash3/)). Speculative sampling's acceptance rule, residual distribution, exactness theorem, and speedup formula ([Leviathan et al.](https://arxiv.org/pdf/2211.17192)). vLLM's 20.4–38.2% memory-utilization finding, block size 16, and throughput comparisons ([Kwon et al.](https://arxiv.org/pdf/2309.06180)). Sarathi-Serve's 2.6×/3.7×/5.6× capacity gains ([Agrawal et al.](https://arxiv.org/abs/2403.02310)). MLA's per-token cache table and "GQA with 2.25 groups" framing ([DeepSeek-V2](https://arxiv.org/pdf/2405.04434)). EAGLE-3's up-to-6.5× ([arXiv 2503.01840](https://arxiv.org/abs/2503.01840)). All of these are best-case or benchmark-specific figures; **quote them as "the paper reports," not as what you'll get.**

### Computed here (reproducible arithmetic)

KV cache: 320 KiB/token for Llama 3.1 70B, 2.50 GiB per 8K sequence, 80.0 GiB at batch 32 (= 85.9 GB decimal), 640 GiB / 687 GB for the MHA-64 counterfactual, 128 KiB/token and 16.0 GiB at full context for the 8B, 68.6 KiB/token for DeepSeek-V3 MLA. Decode floors: 42.1 / 21.1 / 10.5 ms for TP=1/2/4. No-cache waste factor 482×. Static-batching utilization 32.4%. The entire section 15 capacity plan. Speculative speedup table. Sampling cost fractions.

### Assumptions — the soft numbers that will move on you

These are the ones to flag explicitly in an interview, because flagging them is the difference between a plausible answer and a credible one.

| Assumption | Value used | Reality |
|---|---|---|
| Achievable fraction of peak HBM bandwidth | 75% | 60–90% depending on kernel, access pattern, and framework version. **Measure it.** |
| Prefill MFU | 50% | 35–70%. FA3 + FP8 on tuned stacks goes higher; unoptimized stacks much lower. |
| Non-KV, non-weight GPU overhead | 6 GiB/GPU | framework- and version-specific. Measure with `nvidia-smi` on a real deployment. |
| Average KV occupancy | 90% of budget | scheduler- and watermark-dependent. |
| KV bytes/element for bf16 | 2 | correct, but frameworks sometimes pad head dims or store extra metadata |

### Things I could not verify, and which you should mark as such

- **Cloud GPU pricing.** H100 on-demand spans roughly \$1.50–\$7.00/GPU-hour across providers in 2026 with enormous variation by commitment and region. The \$2.50–\$3.50 range in section 15 is illustrative. Never state a price as fact in an interview; state a range and say it depends on commitment.
- **Latency SLO targets** (TTFT < 500 ms, TPOT < 50 ms, and the voice variants). These are widely-repeated industry rules of thumb, not measured constants. Present them as design targets.
- **"2–4× more concurrent users from PagedAttention"** and **"2–4× goodput from disaggregation."** Directionally right and widely reported, but the multiple depends entirely on how bad the baseline was. Prefer citing the mechanism plus the paper's own measured comparison.
- **Which production systems use Ring Attention.** Long-context serving details at commercial labs are not public. Don't assert it.
- **Blackwell per-GPU specs and FP4 throughput multiples.** Derived from system-level figures and vendor benchmarks. Verify against a current datasheet for the exact SKU you're discussing.
- **Typical acceptance rates for specific draft/target pairs.** Highly workload-dependent (code and summarization accept much better than open-ended chat). The 0.6–0.8 range for vanilla two-model speculation is folklore-grade; measure yours.

### Corrections to the previous version of this document

If you studied the earlier draft, these are the things that changed and why. Each one is a fact-check result, and a couple of them are things an interviewer could catch you on.

1. **H100 bandwidth: "~3 TB/s" → 3.35 TB/s** (SXM, vendor-published).
2. **H100 balance point: "around 330 ops/byte" → 295 FLOP/byte.** The 330 figure came from dividing by 3 TB/s instead of 3.35.
3. **Decode intensity at batch 1: "around 2 ops/byte" → exactly 1.0** for bf16 weights. It's $2/\text{bytes-per-weight}$, so 1.0 at bf16 and 4.0 at int4. The old text said both "≈1" and "around 2" in adjacent paragraphs.
4. **Batch size at which decode becomes compute-bound: "~256+" → ~295** for bf16 on H100, and higher once you count the KV read, which scales with batch. Lower (~74) for int4 weights.
5. **Prefill FFN cost: "$O(P \cdot d_{\text{model}})$" → $O(P \cdot d_{\text{model}}^2)$.** The old exponent was wrong; the FFN is a matmul against a $d \times d$-ish weight matrix, not a vector.
6. **FlashAttention: "memory access drops from $O(N^2)$ to $O(N)$" → HBM accesses go from $\Theta(Nd + N^2)$ to $\Theta(N^2 d^2/M)$**, while the *memory footprint* goes from $O(N^2)$ to $O(N)$. Two different results that the old text merged into one wrong claim. The old file was right that it's exact and doesn't reduce FLOPs — that part was good.
7. **FlashAttention-3: "again ~2x" → 1.5–2.0× over FA2 in FP16, 740 TFLOP/s, 75% of H100 peak vs FA2's 35%.** Now sourced, with the three mechanisms named.
8. **Single-GPU 70B decode speed: "~21 tok/s in fp16" on one H100 → incoherent as stated.** 141 GB of weights doesn't fit in 80 GB. The correct figures: a hypothetical single H100 gives 42.1 ms/token (23.7 tok/s); TP=2 gives 21.1 ms (47.5 tok/s); TP=4 gives 10.5 ms (94.9 tok/s). The old "21" was the TP=2 *millisecond* figure mislabeled as tokens per second.
9. **The "cost per output token" formula was actually a *time* per token formula**, with units of seconds, not dollars. Relabeled, and a real cost-per-token calculation added in section 15 step 9.
10. **TP=2 aggregate bandwidth: "~6 TB/s" → 6.7 TB/s** (2 × 3.35).
11. **KV cache formula in the quick-answers list** was given in the MHA form $2 n_L d_{\text{model}} \cdot \text{seqlen} \cdot b$, which overestimates any GQA model by 8×. Corrected to the $n_{\text{kv heads}} \cdot d_{\text{head}}$ form everywhere.
12. **Example model switched from LLaMA-2 70B to Llama 3.1 70B.** The attention geometry is identical (80 layers, 64 query heads, 8 KV heads, $d_h=128$), but the old file used an 8K context example with LLaMA-2, whose native context is only 4K. Llama 3.1 supports 128K, so the example is now coherent.
13. **MLA reduction versus MHA: "~10×" → depends on the comparison, and for DeepSeek-V3's own 128-head geometry it's 56.9×.** The "~3–4× vs GQA-8" claim checks out — the paper's framing is "equivalent to GQA with 2.25 groups," so $8/2.25 = 3.6\times$.
14. **"most frontier serving will use these by mid-2025"** (FP4) — stale forward-looking claim, replaced with the current state and an explicit hardware requirement.
15. **The 86 GB and 687 GB KV figures were correct** and I verified them by recomputation: 85.9 GB decimal (80.0 GiB) and 687.2 GB decimal (640 GiB). The 8× GQA ratio is exact. Worth saying because these are the numbers most likely to be challenged.
16. **The speculative-decoding speedup formula was correct**; $1 + \alpha + \cdots + \alpha^\gamma$ over $\gamma c + 1$ matches Leviathan Theorem 3.8. Verified and now accompanied by a computed table and an explicit statement that the numerator is the expected accepted length.
17. **Added a "prefill is 71% of GPU time while decode is 97% of latency" result**, which the old file implied but never computed, and which is the sharpest available argument for disaggregation.

---

## 18. The most-asked questions, with answers you can say

These are compressed. The full explanation for each is in the section referenced.

**1. Why is decode memory-bandwidth-bound?** (§3, §4) "Each decode step reads every weight in the model out of HBM and produces one token per sequence. At batch 1 in bf16 that's exactly one FLOP per byte, against an H100 balance point of 295 — so you're using about 0.34% of the arithmetic. Time per step is bytes read over bandwidth: 141 gigabytes over 3.35 terabytes a second is 42 milliseconds for a 70B model."

**2. Why is prefill compute-bound?** (§3) "Prefill processes all P prompt tokens in one pass, so every matmul is a big matrix-matrix product and you reuse each weight P times. Arithmetic intensity in the weight-bound part is just P — at 6,000 tokens that's 6,000 ops per byte, twenty times above the ridge point. Tensor cores are genuinely saturated."

**3. What does the KV cache cost in memory?** (§5) "Two, for K and V, times layers, times **KV** heads — not query heads — times head dim, times bytes per element, times sequence length, times batch. Llama 3.1 70B in fp16 is 2 × 80 × 8 × 128 × 2 = 320 kibibytes per token. So 32 users at 8K context is 80 gibibytes, on top of 141 gigabytes of weights."

**4. What limits your batch size in production?** (§5, §15) "KV-cache memory, essentially always. Compute would support a batch around 300 on an H100 before you hit the roofline, but a 70B model at 6.5K average context runs out of memory around 82. That's why every serving-systems paper is really about KV memory."

**5. What's PagedAttention?** (§8) "Virtual memory for the KV cache. Instead of a contiguous max-length reservation per request, you allocate fixed 16-token blocks on demand and keep a per-sequence block table mapping logical positions to physical blocks. The vLLM paper found existing systems only used 20 to 38% of KV memory for actual tokens; paging removes both internal and external fragmentation and gives you refcounted prefix sharing for free."

**6. What's continuous batching?** (§7) "Scheduling at the granularity of a decode iteration instead of a request. Every step you reap finished sequences, free their blocks, admit whatever fits from the queue, and rebuild the batch. Static batching with realistic output-length variance runs at about 32% slot utilization because everyone waits for the longest generation; continuous batching gets you to 70-plus and is typically 2 to 4x throughput."

**7. Walk me through speculative decoding.** (§10) "A small draft model proposes gamma tokens; the big model verifies all of them in one forward pass, which costs about the same as generating one token because decode is memory-bound. You accept token i with probability min of one and p over q, and on the first rejection you resample from the normalized positive part of p minus q. That makes it distributionally exact — the accept branch contributes min of p and q, the residual contributes the shortfall, they sum to p. Speedup is expected accepted length over iteration cost, so 2 to 2.5x at 70% acceptance."

**8. What's FlashAttention, and does it reduce FLOPs?** (§9) "No — same FLOPs, exact same output. It reduces HBM traffic. Naive attention materializes the N-by-N score matrix in HBM; FlashAttention tiles the computation to fit in the SM's 228 kilobytes of shared memory and uses online softmax with a running max and sum so it never needs a whole row at once. HBM accesses go from theta of N-d plus N-squared to theta of N-squared d-squared over SRAM size, and the footprint goes from quadratic to linear."

**9. What's W4A16, and what's W8A8?** (§11) "W4A16 is 4-bit weights, 16-bit activations — you dequantize in registers before the matmul, so you save the HBM read without needing low-precision tensor cores. GPTQ and AWQ target this and it's the open-source default. W8A8 quantizes activations too so you can use int8 tensor cores for roughly 2x the FLOPs, and it needs SmoothQuant to handle activation outliers."

**10. Why does INT8 sometimes hurt quality?** (§11) "Activation outliers. LLMs above a few billion parameters have specific hidden dimensions that consistently carry values 50 to 100 times larger than typical, so the quantization scale gets set by those channels and everything else collapses into a few levels. SmoothQuant fixes it by inserting a diagonal rescaling that divides the outlier magnitude out of the activations and multiplies it into the weights, which were well-behaved to begin with — and it folds into the preceding LayerNorm so it's free at runtime."

**11. GPTQ vs AWQ?** (§11) "Both do 4-bit weight-only quantization. GPTQ quantizes column by column using a Hessian approximation and updates the remaining weights to compensate for the error it just introduced. AWQ is activation-aware: it finds the roughly 1% of channels that multiply large activations and scales them to protect them before rounding. AWQ is cheaper to run and tends to generalize better out of distribution; GPTQ has been around longer with broader kernel support. They land within a point or two of each other."

**12. MQA vs GQA vs MLA?** (§5, §16.1) "MQA has one K/V head shared by all query heads — maximum saving, some quality loss. GQA has groups of query heads sharing a K/V head, which is the compromise everyone ships; Llama 3.1 70B uses 8 KV heads for 64 query heads, an 8x saving. MLA caches a low-rank latent instead of K and V and decompresses per-head at attention time, so you get GQA-with-2.25-groups memory while reportedly beating MHA on quality — the catch is RoPE doesn't commute with the up-projection, so they split each head into a compressed part and a small uncompressed RoPE part."

**13. TTFT vs TPOT?** (§13) "TTFT is queueing plus prefill, so it scales with prompt length and load. TPOT is bytes read per decode step over bandwidth, so it scales with batch size and context length. Different mechanisms, different fixes. For long outputs TPOT dominates end-to-end — at 500 output tokens, 30 milliseconds of TPOT is 15 seconds versus half a second of prefill."

**14. Why does throughput rise while latency worsens as batch grows?** (§4, §13) "Because the weight read is amortized over the whole batch — decode's arithmetic intensity literally equals the batch size — so tokens per second across all users goes up nearly linearly. But each step reads more KV and does more compute, so it takes longer, and each individual user only gets one token per step. Throughput is a sum, latency is a per-request quantity, and batching helps the sum at the cost of the individual."

**15. What's prefix caching, and when does it not help?** (§14) "Reusing the KV cache for an identical prompt prefix — valid because K and V at a position depend only on tokens up to that position. Inside a server it's hash-keyed block sharing with refcounts; across requests, providers price it at about 0.1x for a read. It attacks prefill, so it's huge for agents and multi-turn chat with big static preambles, and worthless if your prompts are short and your outputs long. And the prefix must match from the very first token, so put a timestamp at the top of your system prompt and you've broken it."

**16. How would you size a deployment?** (§15) Walk the eight steps. Model fit → parallelism choice → KV budget → max batch → step time → throughput → prefill cost → combine and scale out. State your assumptions about achievable bandwidth and MFU as assumptions.

**17. Reduce TTFT without hurting TPOT.** (§13) "Chunked prefill, prefix caching, better prefill kernels like FA3 or FP8 prefill, disaggregate prefill onto its own workers, or shorten the prompt. What I would *not* do is lower the batch size — that reduces queueing but costs throughput, so under load it makes things worse."

**18. What happens when a new request needs prefill mid-decode?** (§7, §16.2) "That's a generation stall. A 6,000-token prefill is 450 milliseconds, and if you run it as one step every decoding request's TPOT jumps from 30 milliseconds to 450. Chunked prefill fixes it: split into 512-token chunks and fold one chunk into each step alongside the decode tokens, so the worst-case interference is one chunk, about 38 milliseconds. It also balances the batch nicely, since prefill chunks are compute-bound and decode tokens are memory-bound."

**19. Why disaggregate prefill and decode?** (§16.3) "Because they're opposite workloads sharing a GPU. In a 70B, 6K-prompt, 500-output configuration, prefill is about 71% of GPU time while decode is 97% of the latency — they want different batch sizes, different parallelism, different kernels. Split them into separate pools, transfer the KV cache after prefill, and you can scale and tune each independently. The requirement is a fast fabric, because you're moving nearly 2 gibibytes per request."

**20. How much does sampling cost?** (§6) "Essentially nothing. A 70B forward pass is 141 gigaflops per token; a full sort of a 128K vocabulary for top-p is about 2 million operations, so a thousandth of a percent. Pick your sampling strategy for output quality. The only time it appears in a profile is at large batch, where the batch-by-vocab logits tensor is hundreds of megabytes and you're bandwidth-bound on it — and the fix is a fused kernel, not a simpler policy."

---

## 19. A drill plan

**Tier 1 — do not walk into an interview without these.**

1. State the prefill/decode dichotomy and the arithmetic-intensity argument for each, from scratch, in under a minute.
2. Write the KV cache formula from memory with the $n_{\text{kv heads}}$ form, and compute 320 KiB/token for Llama 3.1 70B.
3. Compute the H100 balance point (989.5 / 3.35 = 295) and state that bf16 decode intensity equals batch size.
4. Compute the decode time floor for a 70B model at TP=2 and TP=4.
5. Explain why max batch size is set by memory, not compute.

**Tier 2 — the mechanisms, each in 60 seconds with the mechanism named.**

6. PagedAttention as virtual memory, with the 20–38% utilization finding.
7. Continuous batching, with the 32% static-utilization arithmetic.
8. FlashAttention as tiling plus online softmax, exact, same FLOPs.
9. Speculative decoding, including sketching the two-term proof that it's exact.
10. SmoothQuant's diagonal rescaling and why activation outliers break INT8.
11. GQA vs MQA vs MLA, with MLA's RoPE subtlety.
12. Chunked prefill and the generation-stall problem it solves.

**Tier 3 — the whiteboard exercise.**

13. Do the section 15 capacity plan end to end, out loud, for a model and GPU the interviewer names. Practice with different inputs: Llama 3.1 8B on one L40S; DeepSeek-V3 on a node of H200s; a 70B at 32K context. The arithmetic is the same shape every time.
14. Then answer the follow-up: "which assumption in that calculation are you least confident about?" (Achievable bandwidth fraction and the overhead reservation. Say so.)

**Tier 4 — frontier depth, for frontier labs.**

15. MLA's per-token cache expression and why it's equivalent to GQA-2.25.
16. Disaggregated serving, with the prefill-is-71%-of-GPU-time argument.
17. EAGLE's feature-level drafting and why acceptance rate is the only variable that matters.
18. StreamingLLM's attention sinks, and what eviction does and does not buy you.
19. NVFP4 vs MXFP4, and the fact that neither runs on Hopper.
20. Scheduler design: admission policy, watermarks, preemption via swap vs recompute, prefill/decode batch composition, prefix-cache-aware routing.

**How to practice.** For each item, say the answer out loud, timed, without notes. If it takes more than 60 seconds you don't have it compressed enough; if you can't produce a number, you don't have it at all. The "Saying it out loud" blocks in this document are written to be memorized nearly verbatim — they're deliberately in spoken register, with contractions, because reciting written prose sounds rehearsed and reciting spoken prose sounds like you know it.

---

## 20. Further reading

**Foundational serving math**
- Pope et al., "Efficiently Scaling Transformer Inference" (2022). The original careful treatment of inference arithmetic, batching, and parallelism trade-offs. Read this one properly.
- Williams, Waterman & Patterson, "Roofline: An Insightful Visual Performance Model" (2009). The performance model everything rests on.
- The JAX/DeepMind [Scaling Book](https://jax-ml.github.io/scaling-book/roofline/) — the clearest modern treatment of rooflines and critical batch size.

**Serving systems**
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022). Origin of continuous batching.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023). vLLM.
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (OSDI 2024). Chunked prefill and stall-free batching.
- Zhong et al., "DistServe" (OSDI 2024) and Qin et al., "Mooncake" (2024). Prefill/decode disaggregation.
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (2023). RadixAttention and prefix-cache-aware serving.

**Kernels**
- Dao et al., "FlashAttention" (2022); Dao, "FlashAttention-2" (2023); Shah et al., "FlashAttention-3" (2024).
- Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018). The trick that makes tiling possible.
- Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023).

**Attention architecture**
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019). MQA.
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023).
- DeepSeek-AI, "DeepSeek-V2" (2024) and "DeepSeek-V3 Technical Report" (2024). MLA, and MoE serving at scale.

**Speculative decoding**
- Leviathan, Kalman & Matias, "Fast Inference from Transformers via Speculative Decoding" (2023).
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (DeepMind, 2023). Independent, concurrent derivation.
- Cai et al., "Medusa" (2024); Li et al., "EAGLE" (2024), "EAGLE-2" (2024), "EAGLE-3" (2025).

**Quantization**
- Dettmers et al., "LLM.int8()" (2022); "QLoRA" / NF4 (2023); "SpQR" (2023).
- Xiao et al., "SmoothQuant" (2022).
- Frantar et al., "GPTQ" (2022).
- Lin et al., "AWQ" (2023).
- Ashkboos et al., "QuaRot" (2024) and Liu et al., "SpinQuant" (2024). Rotation-based W4A4.
- Micikevicius et al., "FP8 Formats for Deep Learning" (2022).

**KV cache compression**
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM, ICLR 2024).
- Zhang et al., "H2O: Heavy-Hitter Oracle" (2023).
- Liu et al., "KIVI" (2024); Hooper et al., "KVQuant" (2024).
- Tang et al., "Quest" (2024).

---

## Closing

The reason LLM inference feels like a grab bag of tricks on first encounter, and like a coherent field on second, is that almost all of it descends from a single fact: **a modern GPU can perform roughly three hundred arithmetic operations in the time it takes to fetch one byte from memory.** Prefill happens to sit far above that ratio, so it's a compute problem. Decode sits far below it, so it's a bandwidth problem. Once you know which side of the line you're on, the right optimization is usually obvious — batch harder, read fewer bytes, get more tokens per read, or stop doing work you've already done.

The other thing worth carrying out of this document is the habit of arithmetic. Nobody expects you to remember that PagedAttention gives 2.4× or that EAGLE-3 gives 6.5×; those are benchmark numbers and they will change. What people *do* expect, in an interview and much more so on the job, is that when someone says "can we serve this model at twenty requests a second," you reach for a whiteboard instead of an opinion. Section 15 is the whole job in one page. Learn to do it cold, know which of its assumptions are soft, and say so before you're asked.
