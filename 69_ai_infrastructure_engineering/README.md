# Topic 69: AI Infrastructure Engineering — Production Playbook

> Production-flavored counterpart to the research-scientist chapters. Built around the 8-area AI Infrastructure Engineer skill checklist:
>
> 1. GPU / VRAM fundamentals, quantization & batching
> 2. vLLM / TensorRT-LLM / inference optimization
> 3. KV caching, speculative decoding & token throughput
> 4. Distributed training basics (DDP / FSDP / DeepSpeed)
> 5. Model serving & autoscaling
> 6. Vector DB retrieval pipelines
> 7. Prompt caching & cost optimization
> 8. Observability for LLM apps

> 🔥 **Read these first:**
> - **`AI_INFRA_ENGINEER_PLAYBOOK.md`** — 21 sections covering the full production stack: GPU hardware mental model with frontier specs (A100/H100/H200/B200); quantization in production (FP8, INT8, INT4 with GPTQ/AWQ/SmoothQuant); batching strategies (static/dynamic/continuous/chunked-prefill/disaggregated); inference engines (vLLM/TensorRT-LLM/TGI/SGLang/LMDeploy with decision matrix); KV caching production (PagedAttention, prefix cache, eviction, KV quant math); speculative decoding production gotchas; SLO metrics (TTFT/TPOT/TPS/ITL); distributed training infra (DDP/FSDP/DeepSpeed/Megatron + slurm/k8s/Ray); serving platforms (Triton/KServe/BentoML/Ray Serve/Modal/Baseten) with autoscaling and multi-tenant LoRA; vector DB pipelines (Pinecone/Weaviate/Qdrant/Milvus/pgvector + HNSW/IVF-PQ + hybrid retrieval); prompt caching and cost optimization (10-point checklist + worked GPU-cost example); observability for LLM apps (LangSmith/Langfuse/Helicone/Arize + OpenTelemetry GenAI); capacity planning math; reliability patterns (blue-green, canary, shadow, multi-AZ); infra-layer security; full production architecture diagram. Plus 100 interview-grill questions across A–M and a 7-day drill plan.

## Why this exists

The other chapters (06_llm_inference, 61_large_scale_llm_systems, 62_frontier_training_playbook, 63_paged_attention_and_llm_serving) cover the *algorithms and internals*. They lean research-scientist. This folder covers the **production operations layer** an AI Infrastructure Engineer is expected to know:

- Which inference engine and version (vLLM 0.6, TRT-LLM 0.13, etc.)
- Which serving platform (Triton, KServe, Ray Serve)
- Which observability stack (LangSmith, Langfuse, OpenTelemetry GenAI)
- How to compute GPU costs for a planned product
- How to design SLOs (TTFT p95 < 1s, TPOT p95 < 50ms)
- How to do canary / blue-green / shadow deploys
- How to manage multi-tenant LoRA, prompt caching, model routing for cost
- Which vector DB at which scale, with which index type

These are the topics that come up in **AI Infrastructure Engineer** interviews at OpenAI, Anthropic, Cohere, Together, Fireworks, Anyscale, Modal, Baseten, Replicate, etc.

## Single sentence to remember

> AI infra engineering = pick engine + quantization + batching for the SLO budget; KV math determines hardware; PagedAttention + continuous batching + prompt caching are the throughput trinity; multi-replica + canary + observability are the reliability trinity.

## Cross-references

- **`06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`** — algorithm internals.
- **`61_large_scale_llm_systems/EFFICIENT_TRAINING_INFERENCE_PLAYBOOK.md`** — research-flavored training/inference depth.
- **`63_paged_attention_and_llm_serving/`** — paged attention deep dive.
- **`41_mixture_of_experts/MOE_DEEP_DIVE.md`** — MoE serving considerations.
- **`39_rag_retrieval_augmented_generation/RAG_DEEP_DIVE.md`** — RAG algorithm side; this folder covers the infra side.
- **`65_llm_security/LLM_SECURITY_DEEP_DIVE.md`** — security at the model layer; this folder covers infra-layer security.

## How to use

1. Read `AI_INFRA_ENGINEER_PLAYBOOK.md` cover-to-cover once.
2. Memorize the GPU spec table (§2.3), inference engine decision matrix (§5.7), and the full stack diagram (§17).
3. Be able to derive KV-cache size for any model in seconds.
4. Be able to do the GPU cost calculation (§14.2) on a whiteboard for any DAU/QPS scenario.
5. Drill the 100-question grill in §20.
6. **If you've shipped this stack before, lean on specifics in interviews** ("at \$LASTROLE we ran vLLM 0.6 with FP8 KV cache on H100 TP=2 with chunked prefill enabled, hit p95 TTFT of 800ms at 50 concurrent users"). That kind of concrete answer beats theory every time.
