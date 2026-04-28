# Topic 16: Training Behaviors & Single GPU Optimization

> 🔥 **For interviews, read these first:**
> - **`TRAINING_BEHAVIORS_DEEP_DIVE.md`** — frontier-lab deep dive: healthy loss curves and pathologies, LR (warmup, decay, finder), batch size effects (linear scaling, critical batch, generalization gap), gradient norm tracking, mixed precision (FP16/BF16/FP8), loss spike recovery, catastrophic forgetting + mitigations.
> - **`INTERVIEW_GRILL.md`** — 45 active-recall questions.

## What You'll Learn

This topic teaches you:
- How to train on single GPU efficiently
- Parameter changes for memory optimization
- Why loss spikes happen
- Gradient accumulation
- Mixed precision training
- Memory optimization techniques

## Why We Need This

### Interview Importance
- **Common question**: "How to fit large model in single GPU?"
- **Practical knowledge**: Essential for real training
- **Problem-solving**: Shows you understand training

### Real-World Application
- **Resource constraints**: Not everyone has multi-GPU
- **Cost optimization**: Single GPU training saves money
- **Debugging**: Understand training issues

## Industry Use Cases

### 1. **Single GPU Training**
**Use Case**: Personal projects, startups
- Fit large models in limited memory
- Use gradient accumulation
- Mixed precision training

### 2. **Memory Optimization**
**Use Case**: All training scenarios
- Gradient checkpointing
- Parameter sharding
- Efficient attention

### 3. **Loss Spike Debugging**
**Use Case**: Training stability
- Identify causes
- Fix training issues
- Improve stability

## Core Intuition

Training instability is usually not one mysterious bug.

It is usually one of a small number of things:
- the step size is wrong
- gradients are too large or too noisy
- the batch is problematic
- precision or normalization is unstable
- memory pressure forces a bad training configuration

Good interview answers in this area are procedural.

You should sound like you know how to isolate causes, not just list buzzwords.

### Why Single-GPU Optimization Matters

A lot of real experiments start with resource constraints.

If a model does not fit in memory, you need to decide which trade-off to make:
- smaller batch
- more accumulation
- lower precision
- shorter context
- checkpointing

Each one changes a different part of the training system.

## Technical Details Interviewers Often Want

### Gradient Accumulation

Gradient accumulation does **not** change instantaneous memory for activations of one microbatch very much.

What it does is:
- keep microbatches small enough to fit
- accumulate their gradients
- update less often

That gives a larger effective batch size while respecting memory limits.

### Mixed Precision

Mixed precision helps because many tensors can safely use lower precision.

But it introduces risks:
- underflow
- overflow
- gradient-scaling issues

So the correct explanation is:
- saves memory
- often improves throughput
- can require careful stability handling

### Gradient Checkpointing

Checkpointing saves memory by recomputing some activations during backward pass.

The trade-off is simple:
- lower memory
- more compute
- slower wall-clock in exchange for larger trainable configuration

## Common Failure Modes

- accumulation used incorrectly so effective learning-rate assumptions break
- mixed precision producing NaNs without gradient scaling or stable ops
- checkpointing expected to reduce all memory, when activations were not the main bottleneck
- loss spikes caused by a few bad batches rather than the whole run
- blaming the optimizer when the real issue is data or targets

## Edge Cases and Follow-Up Questions

1. Why can loss spike even if average training looks stable?
2. Why does reducing batch size not always fix OOM?
3. What if the model fits but optimizer states do not?
4. Why can mixed precision help throughput but hurt stability?
5. What is the difference between effective batch size and microbatch size?

## What to Practice Saying Out Loud

1. How you would debug a sudden loss spike
2. How you would fit a slightly larger model on the same GPU
3. Which memory lever you would try first, and why

## Industry-Standard Boilerplate Code

### Gradient Accumulation

```python
"""
Gradient Accumulation: Simulate larger batch size
"""
import torch
import torch.nn as nn

def train_with_gradient_accumulation(model, dataloader, optimizer, 
                                     accumulation_steps: int = 4):
    """
    Train with gradient accumulation
    
    Accumulates gradients over multiple batches
    Before updating weights
    Effectively increases batch size
    """
    model.train()
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass (accumulates gradients)
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### Mixed Precision Training

```python
"""
Mixed Precision: Use FP16 to save memory
"""
from torch.cuda.amp import autocast, GradScaler

def train_mixed_precision(model, dataloader, optimizer):
    """
    Mixed precision training
    
    Forward pass: FP16 (half precision)
    Backward pass: FP32 (full precision)
    Saves ~50% memory
    """
    scaler = GradScaler()
    model.train()
    
    for data, target in dataloader:
        optimizer.zero_grad()
        
        # Forward pass in FP16
        with autocast():
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
        
        # Backward pass (scaler handles FP16/FP32)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Memory Optimization Checklist

```python
"""
Memory Optimization Techniques
"""
def optimize_memory_usage():
    """
    Checklist for single GPU training:
    
    1. Reduce batch size
    2. Use gradient accumulation (simulate larger batch)
    3. Use mixed precision (FP16)
    4. Use gradient checkpointing
    5. Reduce sequence length
    6. Use efficient attention (flash attention)
    7. Free unused variables
    8. Use CPU offloading for optimizer states
    """
    pass

# Parameter changes:
# - batch_size: 32 → 8 (4x less memory)
# - gradient_accumulation_steps: 1 → 4 (same effective batch size)
# - precision: fp32 → fp16 (2x less memory)
# - max_seq_len: 2048 → 1024 (2x less memory)
# - gradient_checkpointing: False → True (trade compute for memory)
```

## Why Loss Spikes Happen

### Common Causes

1. **Learning Rate Too High**
   - Solution: Reduce learning rate
   - Check: LR schedule, warmup

2. **Gradient Explosion**
   - Solution: Gradient clipping
   - Check: Gradient norms

3. **Bad Batch**
   - Solution: Skip or downweight
   - Check: Batch statistics

4. **Numerical Instability**
   - Solution: Mixed precision, better initialization
   - Check: NaN/Inf values

5. **Scheduler Issues**
   - Solution: Fix LR schedule
   - Check: LR at spike time

### Detection Code

```python
"""
Detect and handle loss spikes
"""
def detect_loss_spike(losses: list, threshold: float = 2.0) -> bool:
    """
    Detect if current loss is spike
    
    Spike = loss > threshold * recent_average
    """
    if len(losses) < 10:
        return False
    
    recent_avg = np.mean(losses[-10:-1])
    current = losses[-1]
    
    if current > threshold * recent_avg:
        return True
    return False

def handle_loss_spike(model, optimizer, losses):
    """Handle loss spike"""
    if detect_loss_spike(losses):
        # Option 1: Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
        
        # Option 2: Skip this update
        # optimizer.zero_grad()
        
        # Option 3: Restore previous checkpoint
        # load_checkpoint(model, optimizer)
```

## Single GPU Training Strategy

### Memory Budget Breakdown

```
Total GPU Memory (e.g., 24GB):
- Model weights: ~7GB (7B params × 4 bytes)
- Optimizer states: ~14GB (Adam: 2x model size)
- Activations: ~2GB (batch × seq_len)
- Gradients: ~7GB (same as weights)
- Overhead: ~1GB
```

### Optimization Steps

1. **Reduce Batch Size**: 32 → 8 (saves ~6GB)
2. **Gradient Accumulation**: Accumulate 4 batches (same effective batch)
3. **Mixed Precision**: FP32 → FP16 (saves ~7GB)
4. **Gradient Checkpointing**: Trade compute for memory (saves ~2GB)
5. **Reduce Sequence Length**: 2048 → 1024 (saves ~1GB)

## Exercises

1. Implement gradient accumulation
2. Add mixed precision
3. Detect loss spikes
4. Optimize memory usage

## Next Steps

- **Topic 17**: Probability math Q&A
- **Topic 18**: Distribution classification
