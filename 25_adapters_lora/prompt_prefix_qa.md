# Prompt Tuning and Prefix Tuning: Interview Q&A

## Q1: What is prompt tuning? How does it differ from fine-tuning?

**Answer:**

**Prompt Tuning:**
- Parameter-efficient fine-tuning method
- Adds trainable "soft prompts" (continuous embeddings) to input
- Keeps entire pre-trained model frozen
- Only trains prompt embeddings (typically 20-100 tokens)
- Extremely parameter-efficient: <0.01% of model parameters

**Key Differences from Fine-tuning:**

**1. Parameters Updated:**
- **Fine-tuning**: Updates all model parameters (billions)
- **Prompt tuning**: Updates only prompt embeddings (thousands)
- **Efficiency**: Prompt tuning uses 100-1000x fewer parameters

**2. Storage:**
- **Fine-tuning**: Need full model copy per task (GBs)
- **Prompt tuning**: Only store small prompt embeddings (MBs)
- **Benefit**: Can deploy many tasks with same base model

**3. Training:**
- **Fine-tuning**: Updates all parameters, risk of catastrophic forgetting
- **Prompt tuning**: Model stays frozen, preserves pre-trained knowledge
- **Benefit**: Model can still do original tasks

**4. Multi-task:**
- **Fine-tuning**: Separate model per task
- **Prompt tuning**: Same model, different prompts per task
- **Benefit**: Efficient multi-task serving

**How It Works:**
1. Prepend trainable prompt embeddings to input
2. Pass [prompt; input] through frozen model
3. Only update prompt embeddings during training
4. Prompt learns to encode task-specific information

**Example:**
- Model: GPT-2 (125M parameters)
- Prompt tuning: 20 tokens × 768 dim = 15,360 parameters
- Efficiency: 15,360 / 125,000,000 = 0.012% of parameters

---

## Q2: What is prefix tuning? How does it differ from prompt tuning?

**Answer:**

**Prefix Tuning:**
- Similar to prompt tuning but adds parameters at every layer
- Adds trainable "prefix" key-value pairs at each transformer layer
- More expressive than prompt tuning
- Still parameter-efficient compared to full fine-tuning

**Key Differences:**

**1. Where Parameters Are Added:**
- **Prompt tuning**: Only at input layer (embeddings)
- **Prefix tuning**: At every transformer layer (attention)
- **Impact**: Prefix can influence model at multiple levels

**2. What's Added:**
- **Prompt tuning**: Prompt embeddings (added to input)
- **Prefix tuning**: Prefix keys and values (added to attention)
- **Impact**: Prefix directly modifies attention computation

**3. Parameters:**
- **Prompt tuning**: p × d_model (p = prompt length)
- **Prefix tuning**: L × p × 2d_model (L = layers, for K and V)
- **Example**: 12 layers, 20 tokens, 768 dim
  - Prompt: 20 × 768 = 15,360
  - Prefix: 12 × 20 × 2 × 768 = 368,640
  - Still much less than full model

**4. Expressiveness:**
- **Prompt tuning**: Simpler, may be less expressive
- **Prefix tuning**: More complex, often better performance
- **Trade-off**: More parameters for better performance

**5. Implementation:**
- **Prompt tuning**: Simple concatenation at input
- **Prefix tuning**: Modify attention at each layer
- **Complexity**: Prefix tuning more complex to implement

**When to Use:**
- **Prompt tuning**: Simple tasks, maximum efficiency
- **Prefix tuning**: Complex tasks, need better performance

---

## Q3: Explain the mathematical formulation of prompt tuning.

**Answer:**

**Mathematical Setup:**

Given:
- Pre-trained model with frozen parameters θ
- Input tokens: x = [x₁, x₂, ..., xₙ]
- Prompt length: p
- Model dimension: d_model

**Step 1: Token Embedding**
```
E_input = Embedding_θ(x)
Shape: (n, d_model)
```

**Step 2: Prompt Embedding (Trainable)**
```
P = [p₁, p₂, ..., pₚ]  # Learnable parameters
Shape: (p, d_model)
```

**Step 3: Concatenate**
```
E_combined = [P; E_input]
Shape: (p + n, d_model)
```

**Step 4: Forward Pass (Model Frozen)**
```
output = Model_θ(E_combined)
# All parameters θ are frozen, no gradients
```

**Step 5: Loss and Update**
```
loss = CrossEntropy(output, target)
∇P = ∂loss/∂P  # Only gradients for prompt P
P ← P - α∇P  # Update only prompt embeddings
# θ remains unchanged
```

**Key Equations:**

**Attention with Prompt:**
```
Q = E_combined W_q
K = E_combined W_k
V = E_combined W_v

Attention = softmax(QK^T / √d_k) V
# Prompt tokens influence attention to input tokens
```

**Parameter Count:**
```
Trainable parameters = p × d_model
# Example: 20 × 768 = 15,360 parameters
```

**Why It Works:**
- Prompt tokens attend to and influence input processing
- Model learns to interpret prompt as task instruction
- Prompt encodes task-specific information efficiently

---

## Q4: Explain the mathematical formulation of prefix tuning.

**Answer:**

**Mathematical Setup:**

Given:
- Pre-trained model with L layers, frozen parameters θ
- Input tokens: x = [x₁, x₂, ..., xₙ]
- Prefix length: p
- Model dimension: d_model

**At Each Layer l (l = 1, ..., L):**

**Step 1: Standard Q, K, V Computation**
```
Q_l = X_l W_q^l  # Queries from sequence
K_l = X_l W_k^l  # Keys from sequence
V_l = X_l W_v^l  # Values from sequence
Shape: (n, d_model)
```

**Step 2: Add Prefix (Trainable)**
```
P_l^K = PrefixKey_l  # Learnable prefix keys
P_l^V = PrefixValue_l  # Learnable prefix values
Shape: (p, d_model) each

K_l = [P_l^K; K_l]  # Concatenate prefix keys
V_l = [P_l^V; V_l]  # Concatenate prefix values
Shape: (p + n, d_model)
# Q_l remains unchanged
```

**Step 3: Attention with Prefix**
```
Attention_l = softmax(Q_l K_l^T / √d_k) V_l
# Q_l attends to both prefix and sequence tokens
Shape: (n, d_model)
```

**Total Parameters:**
```
For each layer:
- Prefix embeddings: p × d_model (reparameterized: p × d_model/2)
- K projection: d_model × d_model
- V projection: d_model × d_model

Total = L × (p × d_model/2 + 2 × d_model²)
# With reparameterization
```

**Example:**
- L = 12 layers
- p = 20 tokens
- d_model = 768
- Total ≈ 12 × (20 × 384 + 2 × 768²) ≈ 14M parameters
- Still much less than full model (125M+)

**Why It Works:**
- Prefix influences attention at every layer
- Can guide model behavior at multiple abstraction levels
- More flexible than prompt tuning (only input layer)

---

## Q5: What are the advantages and disadvantages of prompt tuning vs prefix tuning?

**Answer:**

**Prompt Tuning Advantages:**
- **Maximum efficiency**: Fewest parameters (p × d_model)
- **Simple implementation**: Just concatenate at input
- **Fast training**: Fewer parameters to update
- **Easy to deploy**: Very small storage per task
- **Good for simple tasks**: Sufficient for many applications

**Prompt Tuning Disadvantages:**
- **Less expressive**: Only influences input layer
- **May underperform**: On complex tasks compared to prefix/full fine-tuning
- **Limited capacity**: Small number of parameters may not capture complex patterns

**Prefix Tuning Advantages:**
- **More expressive**: Influences every layer
- **Better performance**: Often matches full fine-tuning
- **Multi-level influence**: Can guide model at different abstraction levels
- **Still efficient**: Much fewer parameters than full fine-tuning

**Prefix Tuning Disadvantages:**
- **More parameters**: L × p × 2d_model vs p × d_model
- **More complex**: Need to modify attention at each layer
- **Slower training**: More parameters to update
- **More storage**: Larger than prompt tuning (but still small)

**Comparison Table:**

| Aspect | Prompt Tuning | Prefix Tuning | Full Fine-tuning |
|--------|---------------|---------------|------------------|
| **Parameters** | p × d_model | L × p × 2d_model | All parameters |
| **Efficiency** | Highest | High | Low |
| **Performance** | Good | Very Good | Best |
| **Complexity** | Simple | Moderate | Simple |
| **Storage** | Smallest | Small | Large |
| **Use Case** | Simple tasks | Complex tasks | Maximum performance |

**Recommendation:**
- Start with prompt tuning (simpler, more efficient)
- If performance insufficient, try prefix tuning
- Use full fine-tuning only if needed and resources available

---

## Q6: How do you initialize prompt/prefix embeddings? What strategies work best?

**Answer:**

**Initialization Strategies:**

**1. Random Initialization:**
```
P ~ N(0, 0.02²)  # Small random values
```
- **Pros**: Simple, unbiased
- **Cons**: May require more training, slower convergence
- **Use**: Default, works for most cases

**2. Vocabulary-Based Initialization:**
```
Sample random tokens from vocabulary
Use their embeddings as initial prompt
```
- **Pros**: Starts with semantic information
- **Cons**: May bias towards specific tokens
- **Use**: Often works better than random

**3. Task-Specific Initialization:**
```
Use embeddings from task-related tokens
E.g., for sentiment: "sentiment", "positive", "negative"
```
- **Pros**: Better starting point, faster convergence
- **Cons**: Requires domain knowledge
- **Use**: When you know relevant tokens

**4. Learned Initialization (Transfer):**
```
Train on related task first
Use learned prompts as initialization
```
- **Pros**: Transfers knowledge from related tasks
- **Cons**: Requires related task data
- **Use**: Multi-task scenarios

**5. Reparameterization (Prefix Tuning):**
```
Learn in smaller space (d_model/2)
Project up to full dimension
```
- **Pros**: More stable training
- **Cons**: Slightly more parameters
- **Use**: Prefix tuning, improves stability

**Best Practices:**
- **Prompt tuning**: Start with vocabulary-based
- **Prefix tuning**: Use reparameterization + random init
- **Experiment**: Try different strategies, use validation performance
- **Task-specific**: Use domain knowledge when available

---

## Q7: What is the optimal prompt/prefix length? How do you choose it?

**Answer:**

**Prompt/Prefix Length Selection:**

**Typical Ranges:**
- **Prompt tuning**: 20-100 tokens (commonly 20-50)
- **Prefix tuning**: 10-50 tokens (commonly 10-20 per layer)

**Factors to Consider:**

**1. Task Complexity:**
- **Simple tasks** (binary classification): 20 tokens often sufficient
- **Complex tasks** (QA, generation): 50-100 tokens may be needed
- **Rule of thumb**: More complex task → longer prompt/prefix

**2. Dataset Size:**
- **Large datasets**: Can support longer prompts (less overfitting risk)
- **Small datasets**: Shorter prompts (avoid overfitting)
- **Balance**: Enough capacity but not too much

**3. Model Size:**
- **Larger models**: Can utilize longer prompts effectively
- **Smaller models**: Shorter prompts may be sufficient
- **Match capacity**: Prompt capacity should match model capacity

**Selection Process:**

**1. Start with Moderate Length:**
- Prompt: 20-30 tokens
- Prefix: 10-20 tokens per layer

**2. Validation Experiment:**
- Try different lengths: [10, 20, 50, 100]
- Train on each, evaluate on validation set
- Choose length with best validation performance

**3. Consider Trade-offs:**
- Longer: More capacity, more parameters, risk of overfitting
- Shorter: Less capacity, fewer parameters, less overfitting risk

**4. Practical Guidelines:**
- **Minimum**: 10 tokens (may not have enough capacity)
- **Common**: 20-50 tokens (good balance)
- **Maximum**: 100+ tokens (diminishing returns, overfitting risk)

**Empirical Finding:**
- Performance improves with length up to a point
- Then plateaus or degrades (overfitting)
- Sweet spot: 20-50 tokens for most tasks

---

## Q8: Compare prompt tuning, prefix tuning, LoRA, and full fine-tuning.

**Answer:**

**Parameter Efficiency:**

**Full Fine-tuning:**
- Parameters: 100% of model
- Example: 125M parameters for GPT-2
- Storage: Full model per task

**LoRA:**
- Parameters: Low-rank matrices (r × d_model)
- Example: r=8, d_model=768 → ~6K per layer
- Total: ~0.1-1% of model parameters
- Storage: Small adapter weights

**Prefix Tuning:**
- Parameters: L × p × 2d_model
- Example: 12 × 20 × 2 × 768 ≈ 368K
- Total: ~0.3% of model parameters
- Storage: Prefix embeddings per task

**Prompt Tuning:**
- Parameters: p × d_model
- Example: 20 × 768 = 15K
- Total: ~0.01% of model parameters
- Storage: Smallest (just prompt embeddings)

**Performance:**

**Full Fine-tuning:**
- Best performance (all parameters optimized)
- Risk of catastrophic forgetting
- Requires most resources

**LoRA:**
- Near full fine-tuning performance
- Good balance of efficiency and performance
- Most popular method currently

**Prefix Tuning:**
- Very good performance (often matches full fine-tuning)
- More expressive than prompt tuning
- Good for complex tasks

**Prompt Tuning:**
- Good performance (may be slightly lower)
- Sufficient for many tasks
- Maximum efficiency

**Use Cases:**

**Full Fine-tuning:**
- Maximum performance needed
- Have resources and data
- Single task deployment

**LoRA:**
- Best balance of efficiency and performance
- Most common in practice
- Multi-task scenarios

**Prefix Tuning:**
- Complex tasks, need good performance
- Can afford slightly more parameters
- Multi-task scenarios

**Prompt Tuning:**
- Simple tasks
- Maximum efficiency needed
- Many tasks, limited resources

**Comparison Table:**

| Method | Parameters | Performance | Complexity | Storage |
|--------|-----------|-------------|------------|---------|
| **Full Fine-tuning** | 100% | Best | Simple | Large |
| **LoRA** | 0.1-1% | Excellent | Moderate | Small |
| **Prefix Tuning** | 0.3% | Very Good | Moderate | Small |
| **Prompt Tuning** | 0.01% | Good | Simple | Smallest |

---

## Q9: How do you implement prompt tuning? Show the key code.

**Answer:**

**Key Implementation Steps:**

**1. Freeze Model:**
```python
for param in model.parameters():
    param.requires_grad = False
```

**2. Create Prompt Embeddings:**
```python
prompt_length = 20
d_model = 768
prompt_embeddings = nn.Parameter(
    torch.randn(prompt_length, d_model) * 0.02
)
```

**3. Concatenate with Input:**
```python
input_embeddings = model.transformer.wte(input_ids)
# Shape: (batch, seq_len, d_model)

prompt = prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
# Shape: (batch, prompt_length, d_model)

combined = torch.cat([prompt, input_embeddings], dim=1)
# Shape: (batch, prompt_length + seq_len, d_model)
```

**4. Forward Pass:**
```python
outputs = model.transformer(inputs_embeds=combined)
logits = model.lm_head(outputs.last_hidden_state)
```

**5. Training:**
```python
optimizer = torch.optim.Adam([prompt_embeddings], lr=0.3)
# Only prompt_embeddings are updated
```

**Complete Code:**
See `prompt_prefix_code.py` for full implementation!

**Key Points:**
- Only prompt_embeddings requires gradients
- Model parameters stay frozen
- Very simple implementation
- Extremely parameter-efficient

---

## Q10: How do you implement prefix tuning? What's the complexity?

**Answer:**

**Key Implementation Steps:**

**1. Freeze Model:**
```python
for param in model.parameters():
    param.requires_grad = False
```

**2. Create Prefix for Each Layer:**
```python
# Reparameterized prefix
prefix_emb = nn.Parameter(
    torch.randn(prefix_length, d_model // 2) * 0.02
)
prefix_proj = nn.Linear(d_model // 2, d_model)

# Project to K and V for each layer
prefix_k_proj = nn.ModuleList([
    nn.Linear(d_model, d_model) for _ in range(num_layers)
])
prefix_v_proj = nn.ModuleList([
    nn.Linear(d_model, d_model) for _ in range(num_layers)
])
```

**3. Modify Attention at Each Layer:**
```python
for layer_idx, layer in enumerate(model.transformer.h):
    # Get prefix for this layer
    prefix_k, prefix_v = get_prefix_kv(layer_idx)
    
    # Standard Q, K, V
    Q = compute_queries(hidden_states)
    K = compute_keys(hidden_states)
    V = compute_values(hidden_states)
    
    # Add prefix
    K = torch.cat([prefix_k, K], dim=1)
    V = torch.cat([prefix_v, V], dim=1)
    
    # Attention
    attention = softmax(Q @ K.T / sqrt(d_k)) @ V
```

**Complexity:**

**Parameters:**
- Prefix embeddings: p × d_model/2 (reparameterized)
- Projection: (d_model/2) × d_model
- K/V projections per layer: 2 × d_model²
- Total: L × (p × d_model/2 + 2 × d_model²)

**Time Complexity:**
- Same as standard attention: O(n²d)
- Prefix adds p tokens, so O((n+p)²d)
- Typically p << n, so similar to standard

**Space Complexity:**
- Store prefix embeddings: O(p × d_model)
- Per-layer projections: O(L × d_model²)
- Much less than full model

**Implementation Complexity:**
- More complex than prompt tuning
- Need to modify attention at each layer
- Requires understanding of transformer internals

**See `prompt_prefix_code.py` for complete implementation!**

---

## Summary

Prompt tuning and prefix tuning are powerful parameter-efficient fine-tuning methods. Prompt tuning adds trainable embeddings at the input layer, making it extremely efficient (0.01% parameters) and simple to implement. Prefix tuning adds trainable key-value pairs at every layer, providing more expressiveness (0.3% parameters) and often matching full fine-tuning performance. Both methods keep the pre-trained model frozen, enabling efficient multi-task deployment and preserving pre-trained knowledge. The choice between them depends on the task complexity, available resources, and performance requirements.

