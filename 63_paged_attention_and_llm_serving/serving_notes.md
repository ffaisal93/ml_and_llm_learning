# Serving Notes

## Fast Interview Story

If you need a short answer:

1. KV cache makes autoregressive decoding faster by reusing past keys and values.
2. But long-running requests make KV memory huge, and naive contiguous allocation wastes space.
3. Paged attention breaks KV cache into fixed-size blocks and maps logical positions to physical blocks.
4. That reduces fragmentation and enables efficient dynamic serving features like prefix sharing and continuous batching.

## One-Minute Version

Serving LLMs is often memory-bound because every active request accumulates KV cache over time. The raw attention math has not changed, but the engine has to keep old keys and values resident and quickly accessible. If you reserve one large contiguous buffer per request, you waste space and create fragmentation. Paged attention fixes this by allocating fixed-size KV blocks and tracking them through a block table. That gives the server more flexibility, improves memory efficiency, and works especially well with dynamic workloads and prefix reuse.

## Follow-Up Questions

### Why is this different from training?

Training pays more compute for forward and backward passes and stores activations for backpropagation. Serving pays a long-lived memory cost for active requests and often becomes limited by KV-cache movement and capacity.

### Why does GQA help serving?

Because KV-cache size scales with the number of KV heads. Reducing KV heads can cut memory and bandwidth costs significantly.

### Why does continuous batching matter?

Because requests have different lengths. Static batches waste capacity when some sequences finish early, while continuous batching keeps the device busy with a changing set of requests.
