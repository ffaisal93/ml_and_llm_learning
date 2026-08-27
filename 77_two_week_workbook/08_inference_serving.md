# LLM inference and serving

This topic is tested with numbers. The interviewer wants to hear that prefill and decode are two different workloads on the same weights, and that decode is limited by memory bandwidth, not by arithmetic. The common failure is to answer "we batch it and add more GPUs" without the arithmetic-intensity argument, without a KV-cache size, and without separating time-to-first-token from inter-token latency. Be able to produce a cache figure for a 7B model from memory.

## The equations

**KV-cache size**

$$\text{bytes} = 2 \times L \times H_{kv} \times d_{head} \times b \times S \times B$$

The 2 is for K and V; $L$ is layers, $H_{kv}$ is key-value heads, $d_{head}$ is head dimension, $b$ is bytes per element, $S$ is sequence length and $B$ is batch size, so this is the memory that grows with every generated token.

**Weight memory budget**

$$\text{bytes} = N \times \frac{\text{bits}}{8}$$

A 7B model needs 14GB at FP16, 7GB at INT8 and 3.5GB at INT4, and you must add the KV cache and activation workspace on top before you claim a model fits.

**Arithmetic intensity**

$$I = \frac{\text{FLOPs}}{\text{bytes moved}}$$

Intensity says how much arithmetic you do per byte you drag out of HBM; it is the single number that decides whether a kernel is compute-bound or memory-bound.

**Roofline balance point**

$$I^* = \frac{\text{peak FLOP/s}}{\text{peak bandwidth}}, \qquad \text{perf} = \min(\text{peak FLOP/s},\ I \times \text{bandwidth})$$

An accelerator at roughly 312 TFLOP/s of bf16 and about 2 TB/s of HBM has $I^* \approx 156$ FLOPs per byte, so any kernel below that intensity is memory-bound and cannot reach peak arithmetic.

**Prefill versus decode FLOPs**

$$\text{FLOPs}_{\text{prefill}} \approx 2 N S_{\text{prompt}}, \qquad \text{FLOPs}_{\text{decode step}} \approx 2 N$$

Both read the same $2N$ bytes of FP16 weights, but prefill does $S_{\text{prompt}}$ times more arithmetic per read, so for a 7B model and a 2000-token prompt prefill is 28 TFLOPs against 14 GFLOPs per decode step.

**Decode throughput ceiling from bandwidth**

$$\text{tokens/s} \le \frac{\text{bandwidth}}{N \times \text{bytes per weight}}$$

At batch 1 a 7B FP16 model must stream 14GB per token, so 2 TB/s gives a hard ceiling near 143 tokens per second, and no amount of extra arithmetic changes that.

**Latency decomposition**

$$T_{\text{e2e}} = \text{TTFT} + \text{ITL} \times (n_{\text{out}} - 1), \qquad \text{TTFT} = t_{\text{queue}} + t_{\text{prefill}}$$

Time-to-first-token covers queueing plus the whole prompt forward pass; inter-token latency is the per-step decode time, and only the second one scales with output length.

**Throughput under continuous batching**

$$\text{tokens/s}_{\text{system}} = \frac{B_{\text{eff}}}{\text{ITL}(B_{\text{eff}})}$$

$B_{\text{eff}}$ is the average number of sequences decoding together; throughput rises with batch until the KV cache exhausts memory or ITL starts growing linearly, and past that point you only add latency.

**Little's law**

$$L = \lambda W$$

The average number of requests in the system equals arrival rate times average time in system, so 20 requests per second at 2 seconds each means 40 concurrent requests, which is the number you must size KV-cache memory for.

**Speculative decoding expected speedup**

$$\mathbb{E}[\text{tokens per verify}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}, \qquad \text{speedup} \approx \frac{\mathbb{E}[\text{tokens}]}{1 + K c}$$

$\alpha$ is the per-token acceptance rate, $K$ is the draft length and $c$ is the draft-to-target cost ratio, so the method pays only when $\alpha$ is high and the draft model is genuinely cheap.

## Code from memory

KV-cache calculator, run for two realistic 7B configurations and for the weight budget.

```python
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, dtype_bytes, seq_len, batch):
    # 2 for K and V; every layer stores one K and one V vector per token per KV head
    per_token_per_layer = 2 * n_kv_heads * head_dim * dtype_bytes
    return per_token_per_layer * n_layers * seq_len * batch

GiB = 1024 ** 3
mha = dict(n_layers=32, n_kv_heads=32, head_dim=128, dtype_bytes=2)  # Llama-2-7B style, no GQA
gqa = dict(n_layers=32, n_kv_heads=8, head_dim=128, dtype_bytes=2)   # Mistral-7B style, GQA 4:1

for name, cfg in [("MHA 32 kv-heads", mha), ("GQA  8 kv-heads", gqa)]:
    per_tok = kv_cache_bytes(seq_len=1, batch=1, **cfg)
    print(name, "| per token", per_tok / 1024, "KiB",
          "| 4096 tok x 1", round(kv_cache_bytes(seq_len=4096, batch=1, **cfg) / GiB, 3), "GiB",
          "| 4096 tok x 32", round(kv_cache_bytes(seq_len=4096, batch=32, **cfg) / GiB, 2), "GiB")

for bits in (16, 8, 4):   # weight budget for 7e9 params
    print("weights 7B at", bits, "bit:", round(7e9 * bits / 8 / GiB, 2), "GiB")
```

Output: MHA is 512 KiB per token, 2.0 GiB at 4096 tokens, 64 GiB at batch 32. GQA with 8 KV heads is 128 KiB per token, 0.5 GiB, and 16 GiB. Weights are 13.04, 6.52 and 3.26 GiB. The batch-32 MHA number is the whole point: the cache is larger than the weights.

Naive incremental decoding with a single-head attention, showing exactly where the cache is appended.

```python
import torch, math

d, V = 16, 20
torch.manual_seed(0)
Wq, Wk, Wv = (torch.randn(d, d) / math.sqrt(d) for _ in range(3))
Wo = torch.randn(d, V) / math.sqrt(d)
emb = torch.randn(V, d)

def step(tok, cache):
    x = emb[tok]                          # one token, shape (d,)
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    cache["K"].append(k)                  # <-- the cache append: K and V grow by one row
    cache["V"].append(v)
    K = torch.stack(cache["K"])           # (t, d) -- everything seen so far
    V_ = torch.stack(cache["V"])
    attn = torch.softmax(K @ q / math.sqrt(d), dim=0)   # (t,)
    ctx = attn @ V_                       # (d,)
    return ctx @ Wo                       # logits over vocab

def generate(prompt, n_new):
    cache = {"K": [], "V": []}
    for t in prompt:                      # prefill: all prompt tokens, cache filled
        logits = step(t, cache)
    out = []
    for _ in range(n_new):                # decode: one token in, one token out
        tok = int(logits.argmax())
        out.append(tok)
        logits = step(tok, cache)
    return out, len(cache["K"])

toks, cache_len = generate([1, 5, 9, 2], 6)
print("generated", toks)
print("cache rows", cache_len, "= prompt 4 + decoded 6")
```

Output: `cache rows 10 = prompt 4 + decoded 6`. Only Q is recomputed each step; K and V for old tokens are read, never recomputed.

Speculative decoding with the rejection-sampling acceptance rule, run 200000 times to show it is lossless.

```python
import torch

V, K = 8, 4
torch.manual_seed(0)

def speculative_step(p_list, q_list, draft):
    # p_list[i], q_list[i]: target and draft distributions at position i
    accepted = []
    for i in range(len(draft)):
        x = draft[i]
        p, q = p_list[i][x], q_list[i][x]
        if torch.rand(1).item() < min(1.0, (p / q).item()):
            accepted.append(x)            # accept: keep the draft token
        else:
            resid = torch.clamp(p_list[i] - q_list[i], min=0)   # reject: resample
            resid = resid / resid.sum()                          # from (p - q)+
            accepted.append(int(torch.multinomial(resid, 1)))
            return accepted, i            # everything after a rejection is discarded
    # all K accepted: the target's own next-token distribution gives one bonus token
    accepted.append(int(torch.multinomial(p_list[K], 1)))
    return accepted, K

p_list = [torch.softmax(torch.randn(V), 0) for _ in range(K + 1)]
q_list = [torch.softmax(torch.randn(V), 0) for _ in range(K)]

counts, n_acc, trials = torch.zeros(V), 0, 200000
for _ in range(trials):
    draft = [int(torch.multinomial(q_list[i], 1)) for i in range(K)]
    out, k = speculative_step(p_list, q_list, draft)
    counts[out[0]] += 1
    n_acc += k
print("empirical first-token dist", [round(c, 3) for c in (counts / trials).tolist()])
print("target p_0               ", [round(c, 3) for c in p_list[0].tolist()])
print("mean accepted per step", round(n_acc / trials, 3), "of", K)
```

Output: the empirical distribution `[0.399, 0.064, 0.010, 0.152, ...]` matches the target `[0.400, 0.064, 0.010, 0.151, ...]` to three decimals. Here the draft is random noise, so acceptance is poor and 1.174 tokens land per step, which illustrates the failure mode as well as the correctness.

## Questions

### Q1. Why is prefill compute-bound and decode memory-bound?

Compare arithmetic to bytes moved. Both phases read the whole weight matrix, $2N$ bytes at FP16. Prefill processes $S$ prompt tokens against those weights, so it does about $2NS$ FLOPs, and its arithmetic intensity is roughly $S$ FLOPs per byte. For a 2000-token prompt that is far above the balance point near 156, so the GPU runs near peak arithmetic. Decode processes one token per step, so it does about $2N$ FLOPs for the same $2N$ bytes: intensity is about 1 FLOP per byte. That is two orders of magnitude below the balance point, so the arithmetic units sit idle while HBM streams weights. Concretely a 7B FP16 model must move 14GB per decode step, and at 2 TB/s that is 7 milliseconds, giving a ceiling near 143 tokens per second at batch 1. Batching raises decode intensity because one weight read serves many sequences, which is why batching is the whole game in serving.

> **Say it.** Look at FLOPs per byte. Both phases read the same weights, about two N bytes. Prefill does two N S FLOPs across S prompt tokens, so intensity is about S, well above a balance point near 156 FLOPs per byte, so it saturates the arithmetic units. Decode does two N FLOPs for the same read, so intensity is about one. It is memory-bound by two orders of magnitude. A 7B FP16 model streams 14GB per token, so 2 TB/s caps you near 143 tokens a second at batch one. Batching is what raises decode intensity.

### Q2. What actually limits decode throughput on a real server?

Two things in sequence. First memory bandwidth, because every decode step re-reads the weights and the whole KV cache. Second KV-cache capacity, because that is what caps the batch size that would fix the bandwidth problem. Take an 80GB card with a 7B FP16 model: 14GB of weights leaves about 64GB, and at 512 KiB per token for a multi-head 7B, a 4096-token context costs 2GB per sequence, so you fit about 32 concurrent sequences. GQA with 8 KV heads cuts that to 0.5GB and lets you fit roughly 128. So the practical limiter is usually cache memory, and every technique that matters attacks it: GQA and MQA shrink the per-token cost, PagedAttention removes the fragmentation waste, quantised caches halve the bytes, and prefix sharing deduplicates common system prompts. Compute is almost never the binding constraint during decode.

> **Say it.** First memory bandwidth, because every step re-reads the weights and the whole cache. Then KV-cache capacity, because that caps the batch size that would have fixed the bandwidth problem. On an 80GB card a 7B FP16 model leaves about 64GB, and a 4096-token multi-head context costs 2GB, so roughly 32 sequences. With GQA at eight KV heads it is 0.5GB and about 128. So cache memory is usually the real limit, and that is why GQA, PagedAttention, cache quantisation and prefix sharing all exist. Compute is rarely binding in decode.

### Q3. What is PagedAttention and what problem does it solve?

Classic serving allocates one contiguous KV buffer per sequence, sized for the maximum possible output length. Two kinds of waste follow. Internal fragmentation: a request that generates 100 tokens in a buffer reserved for 2048 wastes 95 percent of it. External fragmentation: the free memory is in the wrong-shaped holes for the next request. Reported waste in that regime is large, with only a minority of cache memory doing useful work. PagedAttention borrows virtual memory. The cache is cut into fixed-size blocks, typically 16 tokens, and each sequence keeps a block table mapping logical positions to physical blocks. Blocks are allocated on demand, so waste is bounded by one partial block per sequence. It also makes sharing trivial: two requests with the same system prompt point at the same physical blocks, with copy-on-write when they diverge, and parallel samples from one prompt share the prompt blocks. The result is a much larger effective batch and higher throughput at the same memory.

> **Say it.** The old way gives each sequence one contiguous buffer sized for the worst-case output length. If it generates a hundred tokens in a two-thousand-token reservation, almost all of that is wasted, and the leftover holes are the wrong shape for the next request. PagedAttention applies virtual memory: fixed blocks of about sixteen tokens plus a per-sequence block table, allocated on demand. Waste drops to one partial block per sequence. It also makes prefix sharing free, because two requests with the same system prompt point at the same physical blocks with copy-on-write. Bigger effective batch, more throughput.

### Q4. Continuous batching versus static batching. What does each do to latency and throughput?

Static batching collects $B$ requests, runs them together, and returns when the last one finishes. Because generation lengths differ wildly, short requests sit padded and idle while the longest one decodes, and no new request can join until the whole batch retires. Utilisation is poor and tail latency is set by the longest member. Continuous batching, sometimes called in-flight batching, schedules at the granularity of one decode step. When a sequence emits its end-of-sequence token its slot is freed immediately and a queued request takes it on the next iteration. Throughput improvement is large in practice, often several-fold on mixed-length workloads. The effect on latency is two-sided: queue wait falls sharply because you no longer wait for a batch to form or drain, but per-token latency for an individual sequence rises, because it now shares bandwidth with more concurrent sequences. So continuous batching improves throughput and mean latency, and you control the per-token cost by capping the maximum running batch.

> **Say it.** Static batching gathers B requests, runs them together, and everyone waits for the longest one, so short requests idle and no new request can join. Continuous batching schedules per decode step: when a sequence emits end-of-sequence, its slot is freed immediately and a queued request enters on the next iteration. Throughput often improves several-fold on mixed-length traffic. Latency is two-sided. Queue wait drops a lot. Per-token latency rises, because you are sharing bandwidth with more sequences. You control that by capping the maximum running batch.

### Q5. Walk me through quantisation from FP16 to INT8 to INT4. What actually breaks?

Memory scales directly: a 7B model is 13.04 GiB at FP16, 6.52 at INT8 and 3.26 at INT4, and because decode is bandwidth-bound, fewer bytes per weight also means faster decode. INT8 with per-channel scales and outlier handling is close to lossless on most tasks. INT4 with a good method such as GPTQ or AWQ, using group-wise scales at group size 64 or 128, is usually a small quality loss on general text. What breaks is specific, not uniform. Transformer activations contain systematic outlier channels whose magnitude is far above the rest; a single per-tensor scale to represent them crushes everything else to a few levels. That is why per-channel or per-group scales and outlier-aware methods exist. Degradation also concentrates in long-chain reasoning, arithmetic and code, where small logit errors compound over many steps, while short factual answers look fine. Therefore evaluate on your hardest task, not on perplexity, because perplexity moves much less than task accuracy.

> **Say it.** Memory scales directly: 13, 6.5, 3.3 GiB for a 7B model, and since decode is bandwidth-bound, fewer bytes also means faster tokens. INT8 with per-channel scales is near lossless. INT4 with GPTQ or AWQ and group size 64 or 128 costs a little quality. What breaks is activation outliers: a few channels have huge magnitude, and a single per-tensor scale crushes everything else, which is why per-group scales exist. Damage concentrates in multi-step reasoning, maths and code, where small errors compound. Evaluate on your hardest task, not perplexity.

### Q6. Explain speculative decoding. Why is it lossless, and what acceptance rate do you need?

A small draft model proposes $K$ tokens autoregressively. The target model verifies all $K$ in one forward pass, which costs almost the same as generating one token because decode is memory-bound and the extra positions are nearly free arithmetic. Acceptance uses rejection sampling: accept draft token $x$ with probability $\min(1, p(x)/q(x))$, and on the first rejection sample from the normalised residual $(p-q)^+$ and discard the rest of the draft. That rule is exactly the standard rejection-sampling construction, so the output distribution is provably the target's own distribution. It is lossless in distribution, not token-for-token identical to a given greedy run. Expected tokens per verify is $(1-\alpha^{K+1})/(1-\alpha)$, and the cost is one target pass plus $K$ draft passes, so the speedup is roughly that divided by $1 + Kc$. With a draft costing 5 percent of the target and $K=4$, you need acceptance above roughly 0.6 to see a useful win; below that the draft passes are pure overhead.

> **Say it.** A small draft model proposes K tokens, the target verifies all of them in one pass, which is nearly free because decode is memory-bound. Accept each draft token with probability min of one and p over q; on the first rejection sample from the normalised positive part of p minus q and throw away the rest. That is textbook rejection sampling, so the output distribution is exactly the target's. Lossless in distribution, not token-identical. Expected tokens is one minus alpha to the K plus one, over one minus alpha. You generally need acceptance above about 0.6 to pay.

### Q7. What are MQA and GQA, and why do they matter for serving?

Multi-head attention gives every query head its own key and value head, so the cache holds $H$ K vectors and $H$ V vectors per token per layer. Multi-query attention keeps all query heads but shares one single K/V head, cutting the cache by a factor of $H$. Grouped-query attention is the middle ground: $H_{kv}$ groups, each shared by $H/H_{kv}$ query heads, typically an 8:1 or 4:1 ratio. From the serving side this is purely a KV-cache and bandwidth argument. For a 32-layer, 32-head, 128-dimension model at FP16 the cache is 512 KiB per token with MHA and 128 KiB with 8 KV heads, so at 4096 tokens it drops from 2 GiB to 0.5 GiB per sequence. You get four times the concurrent sequences in the same memory, and each decode step reads a quarter of the cache bytes. MQA loses measurable quality; GQA at 8 groups is close to MHA quality, which is why nearly every recent model ships GQA.

> **Say it.** MHA gives every query head its own K and V. MQA shares one K and V across all query heads. GQA sits in between, with a small number of KV groups, usually four-to-one or eight-to-one. For serving it is a cache-size argument. Thirty-two layers, head dimension 128, FP16: 512 KiB per token with MHA, 128 KiB with eight KV heads. At four thousand tokens that is two gigabytes against half a gigabyte per sequence. Four times the batch and a quarter of the cache bytes read per step. MQA costs quality, GQA barely does.

### Q8. Tensor parallel versus pipeline parallel. What are the communication patterns?

Tensor parallelism splits each weight matrix across GPUs, so every layer runs on all devices at once. In a transformer block you column-split the first MLP matrix and row-split the second, which needs one all-reduce per block, and the same for attention. That is an all-reduce of the full activation tensor twice per layer, on every forward pass, so it demands very high bandwidth between devices; you use it inside one node over NVLink, typically up to 8 GPUs. Pipeline parallelism assigns whole contiguous layer groups to different devices, so communication is a point-to-point send of one activation tensor at each stage boundary, which is small and tolerates slower interconnect. Its cost is the pipeline bubble: with $P$ stages and $M$ microbatches the idle fraction is about $(P-1)/(M+P-1)$, and interactive decode has very few microbatches, so bubbles hurt. Practical layout: tensor parallel within a node, pipeline parallel across nodes.

> **Say it.** Tensor parallel splits every weight matrix across devices, so all GPUs work on every layer. You column-split then row-split the MLP, which costs an all-reduce of the full activation tensor about twice per block, every forward pass. That needs NVLink-class bandwidth, so you keep it inside one node, usually up to eight GPUs. Pipeline parallel gives whole layer ranges to different devices, so you only send one activation tensor at each stage boundary, which is cheap over Ethernet. Its cost is the bubble, about P minus one over M plus P minus one. Tensor within a node, pipeline across nodes.

### Q9. You have a p99 latency target of two seconds end to end. How do you budget it across stages?

Write the stages down: network in, queue wait, prefill, decode of $n$ tokens, network out. End-to-end is TTFT plus ITL times $n-1$, and TTFT is queue plus prefill. Then the key point: quantiles do not add. The p99 of a sum is not the sum of the per-stage p99s, because the stages rarely have their bad moments together, so summing p99s over-estimates and summing means under-estimates. So you must measure the end-to-end distribution directly, and use per-stage quantiles only to find which stage owns the tail. Practically, you budget on means with headroom, then verify the true p99 by load test. Queue wait is usually the tail owner, and it is controlled by admission control and by capping the running batch, not by making the model faster. Also bound the output length, because $n$ multiplies ITL directly; an unbounded generation makes any latency target meaningless.

> **Say it.** List the stages: network, queue, prefill, decode times output length, network back. End-to-end is TTFT plus ITL times n minus one. The important caveat is that quantiles do not add. The p99 of the sum is not the sum of the p99s, because stages do not have their bad moments together, so that sum over-estimates and the sum of means under-estimates. Measure the end-to-end distribution and use per-stage quantiles only to find which stage owns the tail. Usually it is queue wait, fixed by admission control and batch caps. And cap output length.

### Q10. What does streaming fix, and what does it not fix?

Streaming sends each token as it is produced instead of waiting for the full response. What it fixes is perceived latency: the user sees output after TTFT rather than after the whole generation, so a request that takes eight seconds to complete feels responsive if the first token lands in 300 milliseconds. It also lets the client cancel early, which returns capacity to the server. What it does not fix: total time to a complete answer is unchanged, throughput is unchanged, and the time to the last token is unchanged. It cannot help anything that needs the full response before acting, such as JSON validation, a moderation pass over the whole output, or a tool call that must be parsed. It also makes some things harder, because you cannot retract a token you already sent, so post-hoc filtering has to become incremental. And streaming does nothing for TTFT itself; a slow prefill is still a slow start.

> **Say it.** Streaming sends tokens as they are produced, so the user sees output after time-to-first-token instead of after the whole generation. That fixes perceived latency and enables early cancellation, which gives capacity back. It does not change total generation time, time to last token, or throughput. It does not help anything that needs the complete response first: schema validation, a moderation pass, a tool call you must parse. And once a token is sent you cannot take it back, so filtering has to become incremental. It also does nothing for a slow prefill.

### Q11. Why does batching help throughput but hurt interactive latency?

Because decode is memory-bound. One decode step reads the full weight set regardless of batch size, so at batch 1 those $2N$ bytes serve one token and at batch 32 they serve 32 tokens. Arithmetic intensity rises linearly with batch until it crosses the roofline balance point, and system throughput rises with it, roughly $B/\text{ITL}(B)$. The cost is that each individual sequence now waits behind more work per step: attention over the KV cache does scale with batch, memory bandwidth is shared, and the step time grows. So a user who saw 60 milliseconds per token alone may see 100 at batch 64. There is also a queueing component, since a request may wait for a scheduling slot. The operating decision is which metric you are paid on. Interactive chat caps the batch to hold ITL down; offline bulk generation runs the batch as large as the KV cache allows.

> **Say it.** Decode is memory-bound, so one step reads the whole weight set no matter the batch size. At batch one those bytes produce one token; at batch thirty-two they produce thirty-two. Intensity rises with batch, and so does system throughput. But each sequence now shares bandwidth and waits behind more attention work, so per-token latency grows: sixty milliseconds alone might become a hundred at batch sixty-four. Plus queueing for a slot. So it is a choice of metric. Cap the batch for interactive chat, run it as large as the cache allows for offline jobs.

### Q12. A user says temperature zero should make output deterministic, but they see different results. Explain.

Temperature zero only means argmax over the logits; it removes sampling noise, not numerical noise. Floating-point addition is not associative, so the logits themselves change with execution conditions. Batch composition changes them, because a different batch size selects a different GEMM kernel and a different reduction order. Continuous batching therefore makes your logits depend on who else is being served at that moment. Tensor parallelism changes the all-reduce order. Different GPU models, driver versions, kernel autotuning results and mixed-precision accumulation all shift the last bits. Normally this does not matter, but when the top two logits are nearly tied, a difference in the last bit flips the argmax, and after that the whole continuation diverges. There is also ordinary MoE routing sensitivity in mixture models. To get reproducibility you must pin the model version, the batch size, the parallel layout, the kernels and the seed, which usually costs throughput.

> **Say it.** Temperature zero is argmax; it removes sampling noise, not numerical noise. Floating-point addition is not associative, so the logits themselves move with batch size, kernel selection, reduction order, tensor-parallel all-reduce order, GPU model and driver. With continuous batching your logits depend on who else is in the batch at that instant. Most of the time it does not matter, but when the top two logits are nearly tied, a last-bit difference flips the argmax and the whole continuation diverges. Real reproducibility means pinning version, batch size, parallel layout, kernels and seed, and paying in throughput.

## Done when

- You can write the KV-cache formula and produce the 512 KiB per token and 2 GiB at 4096 tokens numbers for a multi-head 7B model in under two minutes without notes.
- You can give the arithmetic-intensity argument for prefill versus decode with the roofline balance point and the 143 tokens per second bandwidth ceiling.
- You can explain PagedAttention, continuous batching and speculative decoding, including the rejection-sampling rule and why it is lossless, without looking anything up.
- You can state that quantiles do not add and describe how you would actually validate a p99 target by load test.
