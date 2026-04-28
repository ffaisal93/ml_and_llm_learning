# Topic 7: LLM Problem Solving

> 🔥 **For interviews, read these first:**
> - **`LLM_PROBLEMS_DEEP_DIVE.md`** — frontier-lab deep dive: long-context challenges (lost-in-the-middle), hallucination types and mitigations (overview), prompting techniques (CoT, self-consistency, ToT, ReAct), jailbreaks + defenses, indirect prompt injection, agent architectures and failure modes, multi-turn memory, latency/cost, evaluation.
> - **`HALLUCINATION_DETECTION_DEEP_DIVE.md`** — dedicated comprehensive chapter on detecting hallucinations in LLM outputs: full taxonomy, why models hallucinate, reference-based / reference-free / internal-states-based detection (NLI, SelfCheckGPT, semantic entropy, CoVe, truth probes, RAGAS), benchmarks, mitigation, production design. The interview-grade reference.
> - **`HALLUCINATION_INTERVIEW_GRILL.md`** — 90 active-recall questions on hallucination detection (taxonomy → causes → reference-based → reference-free → internal-states → RAG → benchmarks → mitigation → production → eval methodology).
> - **`LLM_EVALUATION_DEEP_DIVE.md`** — frontier-lab deep dive on LLM evaluation: why eval is hard, capability benchmarks (knowledge/reasoning/code/long-context/agent/multimodal), instruction following, LLM-as-judge (biases + calibration), pairwise / ELO / Chatbot Arena, open-ended evaluation, factuality measurement (FactScore/SAFE/RAGAS), contamination detection, robustness, statistical methodology, harnesses (lm-eval-harness/HELM/Inspect), online telemetry, A/B testing, full product eval case study.
> - **`LLM_EVALUATION_INTERVIEW_GRILL.md`** — 115 active-recall questions on LLM eval, organized A–M.
> - **`AGENT_IN_30_MIN.md`** — codable-from-memory agent: 70-line working loop with tool calls + parser + complete walkthrough. Drill until you can write it cold in 25 minutes.
> - **`INTERVIEW_GRILL.md`** — 55 active-recall questions on broader LLM problems.

## What You'll Learn

This topic teaches you to solve common LLM problems:
- Long context length solutions
- Memory efficiency
- Speed optimization
- Detailed explanations with code

## Why We Need This

### Interview Importance
- **Common question**: "How do you handle long context?"
- **Problem-solving**: Show you understand challenges
- **Optimization**: Critical for production

### Real-World Application
- **Production constraints**: Memory, speed limits
- **User requirements**: Need long context
- **Cost optimization**: Efficient solutions save money

## Industry Use Cases

### 1. **Long Context Processing**
**Use Case**: Document analysis, code review
- Process entire codebases
- Analyze long documents
- Multi-document reasoning

### 2. **Memory Efficiency**
**Use Case**: Resource-constrained environments
- Edge devices
- Cost optimization
- Multiple models

### 3. **Speed Optimization**
**Use Case**: Real-time applications
- Chatbots
- Code completion
- Interactive applications

## Core Intuition

LLM problems are usually not about one isolated trick. They are about conflicting constraints.

In practice, you often want all of these at once:
- longer context
- lower latency
- lower memory
- lower cost
- better quality

But those goals fight each other.

That is why interview questions in this area are really trade-off questions.

### Long Context

Long context is useful because many tasks need information that does not fit in a short prompt:
- codebases
- long documents
- multi-turn conversations
- retrieval-heavy tasks

But vanilla attention becomes expensive as sequence length grows.

### Memory Efficiency

Memory pressure appears in both:
- training
- inference

At inference time, memory is often dominated by:
- model weights
- KV cache
- batching overhead

At training time, memory is also heavily affected by:
- activations
- gradients
- optimizer states

### Speed Optimization

Speed is not one thing.

You should separate:
- latency: how long one request takes
- throughput: how many requests you can process overall

Many interview mistakes come from improving one while ignoring the other.

## Technical Details Interviewers Often Want

### Chunking Trade-Off

Chunking reduces memory pressure by avoiding a single giant attention computation.

But it can hurt because:
- information across chunk boundaries becomes weaker
- global reasoning can degrade

That is why chunking often needs overlap or retrieval support.

### Sliding Window Trade-Off

Sliding window attention works well when dependencies are mostly local.

It struggles when:
- critical dependencies are far away
- the task needs global document structure

### Long Context Does Not Automatically Mean Better Quality

This is a very common misconception.

If the extra context is noisy, redundant, or poorly selected, performance can stay flat or even get worse.

### Speed Techniques Have Costs

- KV cache improves speed but increases memory
- batching improves throughput but can hurt latency
- quantization saves memory but can degrade quality
- speculative decoding can help only if draft generation and verification balance out well

## Common Failure Modes

- assuming longer context always helps
- chunking without overlap and losing important dependencies
- optimizing throughput while making per-user latency unacceptable
- solving compute cost while creating a KV-cache memory bottleneck
- mixing up training memory solutions with inference memory solutions

## Edge Cases and Follow-Up Questions

1. Why can longer context hurt quality?
2. Why is chunking not a free solution to long context?
3. What is the difference between latency and throughput in LLM serving?
4. Why can a memory optimization at training time be irrelevant at inference time?
5. Why does the KV cache become a bottleneck for long-context serving?

## What to Practice Saying Out Loud

1. Why long-context handling is a trade-off problem
2. Why more context is only useful if relevant context is selected well
3. Why speed and memory optimizations usually help one bottleneck by stressing another

## Industry-Standard Boilerplate Code

### Long Context Solutions

```python
"""
Long Context Solutions
Problem: Standard attention is O(n²), too expensive for long sequences
Solutions: Chunking, sliding window, sparse attention
"""
import numpy as np

def chunked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                     chunk_size: int, d_k: int) -> np.ndarray:
    """
    Chunked Attention: Process in chunks to reduce memory
    
    Problem: Full attention O(n²) memory
    Solution: Process sequence in chunks
    """
    seq_len = Q.shape[0]
    outputs = []
    
    for i in range(0, seq_len, chunk_size):
        chunk_end = min(i + chunk_size, seq_len)
        Q_chunk = Q[i:chunk_end]
        
        # Attend to all K, V (or can limit to window)
        scores = Q_chunk @ K.T / np.sqrt(d_k)
        attention_weights = softmax(scores)
        output_chunk = attention_weights @ V
        
        outputs.append(output_chunk)
    
    return np.concatenate(outputs, axis=0)

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                            window_size: int, d_k: int) -> np.ndarray:
    """
    Sliding Window Attention: Each position only attends to local window
    
    Problem: Full attention too expensive
    Solution: Local attention + some global positions
    """
    seq_len = Q.shape[0]
    outputs = []
    
    for i in range(seq_len):
        # Define window
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)
        
        Q_i = Q[i:i+1]
        K_window = K[start:end]
        V_window = V[start:end]
        
        # Local attention
        scores = Q_i @ K_window.T / np.sqrt(d_k)
        attention_weights = softmax(scores)
        output = attention_weights @ V_window
        
        outputs.append(output)
    
    return np.concatenate(outputs, axis=0)
```

## Detailed Problem Explanations

### Problem 1: Long Context Length

**The Challenge:**
- Standard attention: O(n²) complexity
- 10K tokens = 100M attention computations
- Memory: O(n²) for attention matrix
- Time: O(n²) for computation

**Solutions:**

**1. Chunking**
- Split sequence into chunks
- Process chunks separately
- Combine results
- **Trade-off**: May lose long-range dependencies

**2. Sliding Window**
- Each position attends to local window
- Add few global positions
- **Trade-off**: Limited context, but efficient

**3. Sparse Attention**
- Only attend to important positions
- Learned or heuristic patterns
- **Trade-off**: Complexity vs accuracy

**4. Hierarchical Attention**
- Attend at multiple levels
- Coarse then fine
- **Trade-off**: More complex implementation

### Problem 2: Memory Efficiency

**The Challenge:**
- Large models (7B+ parameters)
- KV cache grows with sequence length
- Multiple concurrent requests

**Solutions:**

**1. Model Quantization**
- FP32 → INT8 → INT4
- 2-8x memory reduction
- Minimal accuracy loss

**2. Gradient Checkpointing**
- Trade compute for memory
- Recompute activations
- Useful for training

**3. Model Sharding**
- Split model across GPUs
- Distribute memory load
- Requires communication

### Problem 3: Speed Optimization

**The Challenge:**
- Autoregressive generation is slow
- Each token requires full forward pass
- User wants fast responses

**Solutions:**

**1. KV Caching**
- Cache attention K/V
- Avoid recomputation
- 10-100x speedup

**2. Speculative Decoding**
- Draft model generates multiple tokens
- Main model verifies
- Faster if draft is good

**3. Continuous Batching**
- Process multiple requests together
- Better GPU utilization
- Higher throughput

## Exercises

1. Implement chunked attention
2. Compare memory usage
3. Measure speed improvements
4. Test on long sequences

## Next Steps

- **Topic 8**: Training techniques
- **Topic 9**: Sampling techniques
