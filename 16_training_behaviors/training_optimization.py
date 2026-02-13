"""
Training Behaviors: Single GPU Optimization
Interview question: "How to train large model on single GPU?"
"""
import numpy as np

# ==================== Gradient Accumulation ====================

def gradient_accumulation_example():
    """
    Gradient Accumulation: Simulate larger batch size
    
    Instead of:
        batch_size = 32 (might not fit in GPU)
    
    Use:
        batch_size = 8
        accumulation_steps = 4
        Effective batch size = 8 × 4 = 32
    """
    batch_size = 8
    accumulation_steps = 4
    effective_batch_size = batch_size * accumulation_steps
    
    print(f"Actual batch size: {batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Pseudocode
    """
    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        loss = model(batch) / accumulation_steps
        loss.backward()  # Accumulates gradients
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Clear gradients
    """

# ==================== Memory Optimization Checklist ====================

def memory_optimization_checklist():
    """
    How to fit large model in single GPU:
    
    1. Reduce batch_size: 32 → 8 (saves ~6GB)
    2. Use gradient_accumulation: 4 steps (same effective batch)
    3. Use mixed_precision: fp32 → fp16 (saves ~7GB)
    4. Reduce max_seq_len: 2048 → 1024 (saves ~1GB)
    5. Use gradient_checkpointing: True (saves ~2GB, slower)
    6. Use efficient attention: Flash attention
    7. Offload optimizer states to CPU (if needed)
    """
    optimizations = {
        'batch_size': '32 → 8',
        'gradient_accumulation_steps': '1 → 4',
        'precision': 'fp32 → fp16',
        'max_seq_len': '2048 → 1024',
        'gradient_checkpointing': 'False → True',
    }
    
    print("Memory Optimization Checklist:")
    for key, value in optimizations.items():
        print(f"  {key}: {value}")

# ==================== Loss Spike Detection ====================

def detect_loss_spike(losses: list, threshold: float = 2.0) -> bool:
    """
    Detect if current loss is a spike
    
    Spike = current_loss > threshold × recent_average
    """
    if len(losses) < 10:
        return False
    
    recent_avg = np.mean(losses[-10:-1])
    current = losses[-1]
    
    is_spike = current > threshold * recent_avg
    
    if is_spike:
        print(f"Loss spike detected!")
        print(f"  Recent average: {recent_avg:.4f}")
        print(f"  Current loss: {current:.4f}")
        print(f"  Ratio: {current/recent_avg:.2f}x")
    
    return is_spike

def why_loss_spikes_happen():
    """
    Common causes of loss spikes:
    
    1. Learning rate too high
       - Solution: Reduce LR, use warmup
       
    2. Gradient explosion
       - Solution: Gradient clipping (max_norm=1.0)
       
    3. Bad batch (outliers)
       - Solution: Skip batch, use gradient clipping
       
    4. Numerical instability
       - Solution: Mixed precision, better initialization
       
    5. LR scheduler issue
       - Solution: Check LR schedule, reduce max LR
       
    6. Model architecture issue
       - Solution: Check for NaN/Inf, add normalization
    """
    causes = [
        "Learning rate too high → Reduce LR",
        "Gradient explosion → Gradient clipping",
        "Bad batch → Skip or clip gradients",
        "Numerical instability → Mixed precision",
        "LR scheduler issue → Fix schedule",
        "Model architecture → Add normalization"
    ]
    
    print("Why Loss Spikes Happen:")
    for cause in causes:
        print(f"  - {cause}")

# ==================== Parameter Changes for Single GPU ====================

def single_gpu_parameters():
    """
    Parameter changes to fit model in single GPU
    """
    print("Parameter Changes for Single GPU Training:")
    print()
    
    # Original (multi-GPU)
    original = {
        'batch_size': 32,
        'gradient_accumulation_steps': 1,
        'precision': 'fp32',
        'max_seq_len': 2048,
        'gradient_checkpointing': False,
    }
    
    # Optimized (single GPU)
    optimized = {
        'batch_size': 8,  # 4x smaller
        'gradient_accumulation_steps': 4,  # Compensate
        'precision': 'fp16',  # 2x smaller
        'max_seq_len': 1024,  # 2x smaller
        'gradient_checkpointing': True,  # Trade compute for memory
    }
    
    print("Original (Multi-GPU):")
    for key, value in original.items():
        print(f"  {key}: {value}")
    
    print("\nOptimized (Single GPU):")
    for key, value in optimized.items():
        print(f"  {key}: {value}")
    
    print("\nMemory Savings:")
    print("  Batch size: ~6GB saved")
    print("  Precision: ~7GB saved")
    print("  Seq length: ~1GB saved")
    print("  Checkpointing: ~2GB saved")
    print("  Total: ~16GB saved (fits in 24GB GPU)")

# ==================== Usage ====================

if __name__ == "__main__":
    print("Training Behaviors & Single GPU Optimization")
    print("=" * 60)
    print()
    
    gradient_accumulation_example()
    print()
    
    memory_optimization_checklist()
    print()
    
    # Loss spike example
    losses = [0.5, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 1.2]
    detect_loss_spike(losses)
    print()
    
    why_loss_spikes_happen()
    print()
    
    single_gpu_parameters()

