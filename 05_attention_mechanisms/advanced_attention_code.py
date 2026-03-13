"""
Advanced Attention Mechanisms: GQA, MQA, Paged Attention
Complete implementations with detailed explanations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict

# ==================== MULTI-QUERY ATTENTION (MQA) ====================

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention: Shares K and V across all heads
    
    KEY DIFFERENCE FROM MHA:
    - MHA: Each head has separate Q, K, V
    - MQA: Each head has separate Q, but shares K and V
    
    MEMORY REDUCTION:
    - KV Cache: seq_len × (d_k + d_v) instead of num_heads × seq_len × (d_k + d_v)
    - Reduction: num_heads× (e.g., 32× for 32 heads)
    
    WHY IT WORKS:
    - Queries need to be different (capture different aspects)
    - Keys and values can be shared (same information, different queries)
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q: Separate per head (like MHA)
        self.W_q = nn.Linear(d_model, d_model)
        
        # K, V: Shared across all heads (KEY DIFFERENCE)
        self.W_k = nn.Linear(d_model, self.d_k)  # Single projection, not num_heads
        self.W_v = nn.Linear(d_model, self.d_k)  # Single projection, not num_heads
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, past_key_values: Optional[Tuple] = None):
        """
        Forward pass with MQA
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
            past_key_values: Optional cached K, V
        Returns:
            output, (K, V) for caching
        """
        batch_size, seq_len, _ = x.shape
        
        # Q: Separate per head
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, d_k)
        
        # K, V: Shared (single projection, then expand for all heads)
        K = self.W_k(x)  # (batch, seq_len, d_k) ← Single, not per head!
        K = K.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # Shape: (batch, num_heads, seq_len, d_k) ← Expanded to match Q
        
        V = self.W_v(x)  # (batch, seq_len, d_k) ← Single, not per head!
        V = V.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # Shape: (batch, num_heads, seq_len, d_k) ← Expanded to match Q
        
        # Use cached K, V if provided
        if past_key_values is not None:
            K_past, V_past = past_key_values
            # Concatenate: cached + new
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)
        
        # Attention computation (same as MHA)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        # Cache K, V (shared, so only store once, not per head)
        # But we expand it for attention computation
        K_cache = K[:, 0, :, :].unsqueeze(1)  # Take first head (all same)
        V_cache = V[:, 0, :, :].unsqueeze(1)  # Take first head (all same)
        
        return output, (K_cache, V_cache)
    
    def get_kv_cache_size(self, seq_len: int) -> int:
        """
        Get KV cache memory size
        
        MQA: Only stores K, V once (shared), not per head
        """
        return seq_len * (self.d_k + self.d_k)  # K + V, single copy


# ==================== GROUP QUERY ATTENTION (GQA) ====================

class GroupQueryAttention(nn.Module):
    """
    Group Query Attention: Shares K and V within groups of heads
    
    KEY DIFFERENCE:
    - MHA: Each head has separate Q, K, V
    - MQA: All heads share K, V
    - GQA: Heads grouped, K, V shared within each group
    
    EXAMPLE:
    - 32 heads, 8 groups → 4 heads per group
    - Group 1 (heads 0-3): Q_0-3 separate, K_group1 shared, V_group1 shared
    - Group 2 (heads 4-7): Q_4-7 separate, K_group2 shared, V_group2 shared
    - etc.
    
    MEMORY REDUCTION:
    - KV Cache: num_groups × seq_len × (d_k + d_v)
    - Reduction: (num_heads / num_groups)× compared to MHA
    - Example: 32 heads, 8 groups → 4× reduction
    """
    def __init__(self, d_model: int, num_heads: int, num_groups: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.d_k = d_model // num_heads
        
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        
        # Q: Separate per head (like MHA)
        self.W_q = nn.Linear(d_model, d_model)
        
        # K, V: Shared per group (KEY DIFFERENCE)
        self.W_k = nn.Linear(d_model, num_groups * self.d_k)
        self.W_v = nn.Linear(d_model, num_groups * self.d_k)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, past_key_values: Optional[Tuple] = None):
        """
        Forward pass with GQA
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
            past_key_values: Optional cached K, V
        Returns:
            output, (K, V) for caching
        """
        batch_size, seq_len, _ = x.shape
        
        # Q: Separate per head
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, d_k)
        
        # K, V: Shared per group
        K = self.W_k(x)  # (batch, seq_len, num_groups * d_k)
        K = K.view(batch_size, seq_len, self.num_groups, self.d_k).transpose(1, 2)
        # Shape: (batch, num_groups, seq_len, d_k)
        
        V = self.W_v(x)  # (batch, seq_len, num_groups * d_k)
        V = V.view(batch_size, seq_len, self.num_groups, self.d_k).transpose(1, 2)
        # Shape: (batch, num_groups, seq_len, d_k)
        
        # Use cached K, V if provided
        if past_key_values is not None:
            K_past, V_past = past_key_values
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)
        
        # Expand K, V for each head in group
        # Each group has heads_per_group heads that share the same K, V
        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)
        # Shape: (batch, num_heads, seq_len, d_k)
        
        # Attention computation (same as MHA)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        # Cache K, V (per group, not per head)
        return output, (K[:, ::self.heads_per_group, :, :], V[:, ::self.heads_per_group, :, :])
    
    def get_kv_cache_size(self, seq_len: int) -> int:
        """
        Get KV cache memory size
        
        GQA: Stores K, V per group, not per head
        """
        return self.num_groups * seq_len * (self.d_k + self.d_k)


# ==================== PAGED ATTENTION (CONCEPTUAL) ====================

class PagedKVCache:
    """
    Paged KV Cache: Memory-efficient cache management
    
    Manages KV cache in non-contiguous pages (blocks)
    Similar to virtual memory in operating systems
    
    KEY BENEFITS:
    1. No memory fragmentation
    2. Efficient memory reuse
    3. Can handle variable-length sequences
    4. Better GPU memory utilization (~95% vs ~70%)
    """
    def __init__(self, block_size: int = 16, d_k: int = 128, d_v: int = 128):
        """
        Args:
            block_size: Number of tokens per page
            d_k: Key dimension
            d_v: Value dimension
        """
        self.block_size = block_size
        self.d_k = d_k
        self.d_v = d_v
        
        # Page storage: page_id -> (K_page, V_page)
        self.pages: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # Free page pool: pages available for allocation
        self.free_pages: List[int] = []
        
        # Active pages: sequence_id -> [page_ids]
        self.sequence_pages: Dict[int, List[int]] = {}
        
        # Next page ID
        self.next_page_id = 0
    
    def _create_new_page(self) -> int:
        """Create a new page"""
        page_id = self.next_page_id
        self.next_page_id += 1
        
        # Initialize empty page
        K_page = torch.zeros(1, self.block_size, self.d_k)  # (batch=1, tokens, d_k)
        V_page = torch.zeros(1, self.block_size, self.d_v)  # (batch=1, tokens, d_v)
        
        self.pages[page_id] = (K_page, V_page)
        return page_id
    
    def allocate_pages(self, sequence_id: int, num_tokens: int) -> List[int]:
        """
        Allocate pages for a sequence
        
        Args:
            sequence_id: Unique ID for sequence
            num_tokens: Number of tokens to store
        Returns:
            List of page IDs allocated
        """
        num_pages = (num_tokens + self.block_size - 1) // self.block_size
        
        page_ids = []
        for _ in range(num_pages):
            if self.free_pages:
                # Reuse free page
                page_id = self.free_pages.pop()
            else:
                # Allocate new page
                page_id = self._create_new_page()
            page_ids.append(page_id)
        
        self.sequence_pages[sequence_id] = page_ids
        return page_ids
    
    def free_sequence(self, sequence_id: int):
        """
        Free pages when sequence finishes
        
        Returns pages to free pool for reuse
        """
        if sequence_id in self.sequence_pages:
            page_ids = self.sequence_pages.pop(sequence_id)
            # Clear pages and return to pool
            for page_id in page_ids:
                K_page, V_page = self.pages[page_id]
                K_page.zero_()
                V_page.zero_()
                self.free_pages.append(page_id)
    
    def get_kv_for_sequence(self, sequence_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get K, V for a sequence (across multiple pages)
        
        Collects pages and concatenates them
        """
        if sequence_id not in self.sequence_pages:
            return None, None
        
        page_ids = self.sequence_pages[sequence_id]
        
        # Collect K, V from all pages
        K_pages = []
        V_pages = []
        
        for page_id in page_ids:
            K_page, V_page = self.pages[page_id]
            K_pages.append(K_page)
            V_pages.append(V_page)
        
        # Concatenate (non-contiguous in memory, but logically contiguous)
        K = torch.cat(K_pages, dim=1)  # (batch, total_tokens, d_k)
        V = torch.cat(V_pages, dim=1)  # (batch, total_tokens, d_v)
        
        return K, V
    
    def update_page(self, sequence_id: int, page_idx: int, token_idx: int,
                   K_new: torch.Tensor, V_new: torch.Tensor):
        """
        Update a specific token in a page
        
        Args:
            sequence_id: Sequence ID
            page_idx: Which page (0-indexed within sequence)
            token_idx: Which token within page (0 to block_size-1)
            K_new: New key, shape (1, 1, d_k)
            V_new: New value, shape (1, 1, d_v)
        """
        page_ids = self.sequence_pages[sequence_id]
        page_id = page_ids[page_idx]
        
        K_page, V_page = self.pages[page_id]
        K_page[:, token_idx, :] = K_new.squeeze(1)
        V_page[:, token_idx, :] = V_new.squeeze(1)


# ==================== COMPARISON ====================

def compare_attention_mechanisms(d_model: int = 768, num_heads: int = 12,
                                 seq_len: int = 2048, num_groups: int = 4):
    """
    Compare memory usage of different attention mechanisms
    """
    d_k = d_model // num_heads
    
    print("=" * 80)
    print("ATTENTION MECHANISM COMPARISON")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_k: {d_k}")
    
    # MHA
    mha_kv = num_heads * seq_len * (d_k + d_k)
    mha_params = 3 * num_heads * d_model * d_k
    
    # GQA
    gqa_kv = num_groups * seq_len * (d_k + d_k)
    gqa_params = num_heads * d_model * d_k + 2 * num_groups * d_model * d_k
    
    # MQA
    mqa_kv = 1 * seq_len * (d_k + d_k)
    mqa_params = num_heads * d_model * d_k + 2 * d_model * d_k
    
    print("\n" + "-" * 80)
    print("KV CACHE MEMORY (for one sequence)")
    print("-" * 80)
    print(f"MHA:  {mha_kv:>12,} values ({mha_kv * 2 / 1e6:.2f} MB)")
    print(f"GQA:  {gqa_kv:>12,} values ({gqa_kv * 2 / 1e6:.2f} MB) - {num_heads/num_groups:.1f}× reduction")
    print(f"MQA:  {mqa_kv:>12,} values ({mqa_kv * 2 / 1e6:.2f} MB) - {num_heads:.1f}× reduction")
    
    print("\n" + "-" * 80)
    print("PARAMETERS (Q, K, V projections)")
    print("-" * 80)
    print(f"MHA:  {mha_params:>12,} parameters")
    print(f"GQA:  {gqa_params:>12,} parameters ({mha_params/gqa_params:.2f}× reduction)")
    print(f"MQA:  {mqa_params:>12,} parameters ({mha_params/mqa_params:.2f}× reduction)")
    
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)
    print("MHA:  Use for training, maximum quality")
    print("GQA:  Use for production inference (best balance)")
    print("MQA:  Use when maximum efficiency needed")
    print("Paged: Use with any of above for better memory utilization")


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("Advanced Attention Mechanisms")
    print("=" * 80)
    
    # Comparison
    compare_attention_mechanisms(d_model=768, num_heads=32, seq_len=2048, num_groups=8)
    
    print("\n\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
    1. MQA: Shares K, V across all heads → num_heads× memory reduction
    2. GQA: Shares K, V within groups → (num_heads/num_groups)× reduction
    3. Paged: Better memory utilization (95%+ vs 70%)
    4. GQA is recommended for production (best balance)
    5. Paged Attention enables efficient serving (vLLM)
    
    Note: "Multi-head latent attention" is not a standard term.
    Related concepts: latent variables in attention, low-rank attention.
    Production systems use GQA, MQA, or standard MHA.
    """)

