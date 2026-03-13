# Prompt Tuning and Prefix Tuning: Complete Guide

## Overview

Prompt tuning and prefix tuning are parameter-efficient fine-tuning techniques that add trainable parameters to the input while keeping the pre-trained model frozen. These methods are much more efficient than full fine-tuning, requiring only a small fraction of parameters to be trained while achieving competitive performance.

---

## Part 1: Prompt Tuning

### What is Prompt Tuning?

Prompt tuning is a parameter-efficient fine-tuning method where a small number of trainable "soft prompts" (continuous embeddings) are prepended to the input embeddings, while the entire pre-trained model remains frozen. Unlike traditional fine-tuning which updates all model parameters, prompt tuning only trains the prompt embeddings, making it extremely parameter-efficient.

The key idea is that instead of using hard-coded text prompts (like "Translate to French:"), we learn continuous prompt embeddings that are optimized for the task. These soft prompts are learned through backpropagation during training, allowing them to capture task-specific information in a compact form.

### Why Prompt Tuning?

Prompt tuning offers several advantages over full fine-tuning. First, it is extremely parameter-efficient: for a model with billions of parameters, prompt tuning might only require training a few thousand to tens of thousands of parameters (the prompt embeddings), representing less than 0.01% of the model's parameters. This makes it feasible to fine-tune large models on limited hardware.

Second, prompt tuning is storage-efficient: instead of storing a full copy of the model for each task (which could be hundreds of gigabytes), you only need to store the small prompt embeddings (a few kilobytes to megabytes). This allows deploying many task-specific models without the storage overhead.

Third, prompt tuning enables multi-task learning: you can train different prompts for different tasks while sharing the same base model, making it easy to serve multiple tasks from a single model instance. This is particularly valuable in production systems where you want to support many tasks efficiently.

Fourth, prompt tuning reduces the risk of catastrophic forgetting: since the base model is frozen, the knowledge learned during pre-training is preserved, and the model can still perform well on its original tasks. This is in contrast to full fine-tuning, which can cause the model to forget pre-trained knowledge.

### Architecture

Prompt tuning adds trainable prompt embeddings at the input layer. The architecture works as follows: the input text is tokenized and converted to token embeddings using the pre-trained embedding layer. Then, a number of trainable prompt tokens (typically 20-100) are prepended to these input embeddings. These prompt embeddings are randomly initialized and learned during training.

The combined sequence (prompt embeddings + input embeddings) is then passed through the frozen pre-trained model. During the forward pass, the model processes the entire sequence, with the prompt tokens influencing how the model processes the actual input tokens. The prompt embeddings learn to encode task-specific information that guides the model's behavior.

The model's output is used to compute the loss, and gradients are backpropagated only through the prompt embeddings (the model parameters remain frozen). This allows the prompt embeddings to learn optimal representations for the task without modifying the underlying model.

### Mathematical Formulation

Given a pre-trained language model with frozen parameters θ, input tokens x = [x₁, x₂, ..., xₙ], and a prompt of length p, prompt tuning works as follows:

**Step 1: Token Embedding**
```
E_input = Embedding(x)  # Shape: (n, d_model)
```

**Step 2: Prompt Embedding**
```
P = [p₁, p₂, ..., pₚ]  # Trainable prompt embeddings
# Shape: (p, d_model)
```

**Step 3: Concatenate**
```
E_combined = [P; E_input]  # Shape: (p + n, d_model)
```

**Step 4: Forward Pass**
```
output = Model_θ(E_combined)  # Model parameters θ are frozen
```

**Step 5: Loss and Update**
```
loss = Loss(output, target)
∇P = ∂loss/∂P  # Only gradients for prompt P
P ← P - α∇P  # Update only prompt embeddings
```

The key insight is that only the prompt embeddings P are trainable, while all model parameters θ remain frozen. This makes the number of trainable parameters equal to p × d_model, which is typically much smaller than the total model parameters.

### Implementation

```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class PromptTuning(nn.Module):
    """
    Prompt Tuning: Learn continuous prompt embeddings
    
    Only trains prompt embeddings, keeps model frozen
    """
    def __init__(self, model, prompt_length=20, prompt_init="random"):
        super().__init__()
        self.model = model
        self.prompt_length = prompt_length
        self.d_model = model.config.n_embd  # Model dimension
        
        # Freeze the entire model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Initialize prompt embeddings
        if prompt_init == "random":
            # Random initialization
            self.prompt_embeddings = nn.Parameter(
                torch.randn(prompt_length, self.d_model) * 0.02
            )
        elif prompt_init == "vocab":
            # Initialize from vocabulary embeddings
            vocab_embeddings = model.transformer.wte.weight
            # Sample random tokens and use their embeddings
            random_indices = torch.randint(0, vocab_embeddings.size(0), 
                                         (prompt_length,))
            self.prompt_embeddings = nn.Parameter(
                vocab_embeddings[random_indices].clone()
            )
        else:
            raise ValueError(f"Unknown prompt_init: {prompt_init}")
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with prompt tuning
        
        Args:
            input_ids: Token indices, shape (batch_size, seq_len)
            attention_mask: Optional attention mask
        Returns:
            Model outputs
        """
        batch_size = input_ids.size(0)
        
        # Get input embeddings
        input_embeddings = self.model.transformer.wte(input_ids)
        # Shape: (batch_size, seq_len, d_model)
        
        # Expand prompt embeddings for batch
        prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        # Shape: (batch_size, prompt_length, d_model)
        
        # Concatenate: [prompt; input]
        combined_embeddings = torch.cat(
            [prompt_embeddings, input_embeddings], dim=1
        )
        # Shape: (batch_size, prompt_length + seq_len, d_model)
        
        # Adjust attention mask if provided
        if attention_mask is not None:
            # Add ones for prompt tokens
            prompt_mask = torch.ones(
                batch_size, self.prompt_length, 
                device=attention_mask.device, 
                dtype=attention_mask.dtype
            )
            combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # Forward through frozen model
        outputs = self.model.transformer(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask
        )
        
        # Get logits from language model head
        logits = self.model.lm_head(outputs.last_hidden_state)
        
        return logits
    
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Training Example
def train_prompt_tuning(model, dataloader, num_epochs=5, lr=0.3):
    """
    Train prompt tuning
    
    Only prompt embeddings are updated, model stays frozen
    """
    # Create prompt tuning wrapper
    prompt_model = PromptTuning(model, prompt_length=20)
    
    # Only optimize prompt embeddings
    optimizer = torch.optim.Adam(
        prompt_model.prompt_embeddings, 
        lr=lr
    )
    
    prompt_model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # Forward pass
            logits = prompt_model(input_ids)
            
            # Compute loss (shift labels for next token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Backward pass (only updates prompt embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        print(f"Trainable parameters: {prompt_model.get_trainable_parameters()}")
    
    return prompt_model
```

### Prompt Initialization Strategies

The initialization of prompt embeddings can significantly affect performance. Random initialization starts with random values sampled from a normal distribution, which is simple but may require more training to converge. Vocabulary-based initialization samples embeddings from actual token embeddings in the vocabulary, which can provide a better starting point since these embeddings already encode semantic information.

Task-specific initialization uses embeddings from tokens that are semantically related to the task. For example, for a sentiment analysis task, you might initialize prompts with embeddings from words like "sentiment", "positive", "negative", etc. This can help the prompts learn faster by starting closer to useful representations.

Recent research has also explored using learned initialization, where prompts are first trained on a related task or dataset, then fine-tuned on the target task. This transfer learning approach can improve performance, especially when the target task has limited training data.

### Prompt Length Selection

The length of the prompt (number of prompt tokens) is an important hyperparameter. Longer prompts provide more capacity to encode task-specific information but require more parameters and may overfit on small datasets. Shorter prompts are more parameter-efficient but may not have enough capacity for complex tasks.

Typical prompt lengths range from 20 to 100 tokens, with 20-50 being common for many tasks. The optimal length depends on the task complexity: simple tasks like binary classification might work well with 20 tokens, while complex tasks like question answering might benefit from 50-100 tokens.

Empirical studies have shown that prompt tuning performance improves with prompt length up to a point, then plateaus. This suggests that there's a sweet spot where the prompt has enough capacity without being wasteful. Cross-validation or validation set performance can be used to select the optimal prompt length.

---

## Part 2: Prefix Tuning

### What is Prefix Tuning?

Prefix tuning is similar to prompt tuning but adds trainable parameters at every layer of the transformer, not just the input layer. Instead of prepending prompts only to the input embeddings, prefix tuning prepends trainable "prefix" activations at each transformer layer. This gives the method more flexibility to influence the model's behavior at different levels of abstraction.

The key difference from prompt tuning is that prefix tuning modifies the activations at every layer, allowing it to have a more pervasive influence on the model's computations. This can be more powerful than prompt tuning, especially for complex tasks, but it also requires more parameters (though still much fewer than full fine-tuning).

### Why Prefix Tuning?

Prefix tuning offers advantages similar to prompt tuning but with potentially better performance. By adding trainable parameters at every layer, prefix tuning can influence the model's behavior at multiple levels of abstraction, from low-level token representations to high-level semantic understanding. This multi-level influence can be more effective than only modifying the input layer.

Prefix tuning is still parameter-efficient: even though it adds parameters at every layer, the prefix length is typically small (e.g., 10-20 tokens per layer), so the total number of parameters is still much smaller than the model size. For a model with L layers, prefix tuning adds approximately L × prefix_length × d_model parameters, which is still a small fraction of the total model parameters.

Prefix tuning has been shown to achieve performance closer to full fine-tuning than prompt tuning, especially for complex tasks. This makes it a good middle ground between prompt tuning (most efficient) and full fine-tuning (best performance but expensive).

### Architecture

Prefix tuning adds trainable prefix activations at each transformer layer. At each layer, the prefix consists of key and value vectors (but not query vectors, to maintain efficiency). These prefix key-value pairs are prepended to the attention computation at each layer.

The architecture works as follows: at each transformer layer, the standard attention computation uses queries from the current sequence and keys/values from both the prefix and the current sequence. The prefix key-value pairs are learned parameters that influence how attention is computed at that layer.

Specifically, for each layer l, prefix tuning maintains trainable parameters Pₗ^K and Pₗ^V (prefix keys and values). During attention computation, these are concatenated with the layer's computed keys and values: K = [Pₗ^K; K_layer] and V = [Pₗ^V; V_layer]. The queries Q remain unchanged, but they attend to both the prefix and the sequence tokens.

### Mathematical Formulation

For a transformer with L layers, prefix tuning adds trainable parameters at each layer:

**At each layer l:**

**Step 1: Compute standard Q, K, V**
```
Q_l = X_l W_q^l  # Queries from current sequence
K_l = X_l W_k^l  # Keys from current sequence
V_l = X_l W_v^l  # Values from current sequence
```

**Step 2: Add prefix**
```
K_l = [P_l^K; K_l]  # Prepend prefix keys
V_l = [P_l^V; V_l]  # Prepend prefix values
# Q_l remains unchanged
```

**Step 3: Attention**
```
Attention_l = softmax(Q_l K_l^T / √d_k) V_l
```

**Total Parameters:**
```
Total = L × prefix_length × (d_k + d_v)
```

For L=12 layers, prefix_length=20, d_k=d_v=768:
- Total = 12 × 20 × (768 + 768) = 368,640 parameters
- Much less than full model (e.g., 125M parameters for GPT-2)

### Implementation

```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class PrefixTuning(nn.Module):
    """
    Prefix Tuning: Add trainable prefixes at each layer
    
    Adds prefix key-value pairs at every transformer layer
    """
    def __init__(self, model, prefix_length=20, reparam=False):
        super().__init__()
        self.model = model
        self.prefix_length = prefix_length
        self.num_layers = model.config.n_layer
        self.d_model = model.config.n_embd
        self.num_heads = model.config.n_head
        self.d_k = self.d_model // self.num_heads
        
        # Freeze the entire model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Prefix parameters for each layer
        # We'll store them as a single parameter and reshape per layer
        if reparam:
            # Reparameterization trick (more stable training)
            # Store in smaller dimension, project up
            self.prefix_emb = nn.Parameter(
                torch.randn(prefix_length, self.d_model // 2) * 0.02
            )
            self.prefix_proj = nn.Linear(self.d_model // 2, self.d_model)
        else:
            # Direct parameterization
            self.prefix_emb = nn.Parameter(
                torch.randn(prefix_length, self.d_model) * 0.02
            )
            self.prefix_proj = None
        
        # Project to key and value for each layer
        self.prefix_k_proj = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model) 
            for _ in range(self.num_layers)
        ])
        self.prefix_v_proj = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model) 
            for _ in range(self.num_layers)
        ])
    
    def get_prefix_kv(self, layer_idx):
        """
        Get prefix key and value for a specific layer
        
        Args:
            layer_idx: Which transformer layer
        Returns:
            prefix_k, prefix_v: Prefix keys and values
        """
        batch_size = 1  # Will be expanded in forward
        
        # Get prefix embeddings
        if self.prefix_proj is not None:
            prefix = self.prefix_proj(self.prefix_emb)
        else:
            prefix = self.prefix_emb
        # Shape: (prefix_length, d_model)
        
        # Project to key and value for this layer
        prefix_k = self.prefix_k_proj[layer_idx](prefix)
        prefix_v = self.prefix_v_proj[layer_idx](prefix)
        # Shape: (prefix_length, d_model)
        
        return prefix_k, prefix_v
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with prefix tuning
        
        Modifies attention at each layer to include prefix
        """
        # Standard embedding
        hidden_states = self.model.transformer.wte(input_ids)
        
        # Store original forward function
        original_forward = self.model.transformer.h[0].attn.forward
        
        # Modify attention to include prefix
        def modified_attention(self_attn, hidden_states, layer_idx):
            """Modified attention with prefix"""
            batch_size, seq_len, d_model = hidden_states.shape
            
            # Get prefix for this layer
            prefix_k, prefix_v = self.get_prefix_kv(layer_idx)
            # Expand for batch
            prefix_k = prefix_k.unsqueeze(0).expand(batch_size, -1, -1)
            prefix_v = prefix_v.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Standard Q, K, V computation
            Q = self_attn.c_attn(hidden_states)[:, :, :self.d_model]
            K = self_attn.c_attn(hidden_states)[:, :, self.d_model:2*self.d_model]
            V = self_attn.c_attn(hidden_states)[:, :, 2*self.d_model:]
            
            # Reshape for multi-head
            Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            # Project prefix to key/value space
            prefix_k = prefix_k.view(batch_size, self.prefix_length, 
                                    self.num_heads, self.d_k).transpose(1, 2)
            prefix_v = prefix_v.view(batch_size, self.prefix_length, 
                                    self.num_heads, self.d_k).transpose(1, 2)
            
            # Concatenate prefix with sequence
            K = torch.cat([prefix_k, K], dim=2)  # (batch, heads, prefix+seq, d_k)
            V = torch.cat([prefix_v, V], dim=2)
            
            # Compute attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # Adjust attention mask
            if attention_mask is not None:
                # Add ones for prefix
                prefix_mask = torch.ones(
                    batch_size, 1, 1, self.prefix_length,
                    device=attention_mask.device
                )
                seq_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                combined_mask = torch.cat([prefix_mask, seq_mask], dim=-1)
                scores = scores.masked_fill(combined_mask == 0, -1e9)
            
            # Apply causal mask if needed
            seq_len_with_prefix = self.prefix_length + seq_len
            causal_mask = torch.triu(
                torch.ones(seq_len_with_prefix, seq_len_with_prefix),
                diagonal=1
            ).bool().to(scores.device)
            scores = scores.masked_fill(causal_mask, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attention_weights, V)
            
            # Reshape and project
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch_size, seq_len_with_prefix, self.d_model)
            # Remove prefix from output (only keep sequence part)
            output = output[:, self.prefix_length:, :]
            
            return output
        
        # Apply modified attention to each layer
        outputs = []
        hidden = hidden_states
        
        for i, layer in enumerate(self.model.transformer.h):
            # Modified attention
            attn_output = modified_attention(layer.attn, hidden, i)
            hidden = hidden + attn_output  # Residual
            
            # Feed-forward (standard)
            ff_output = layer.mlp(layer.ln_2(hidden))
            hidden = hidden + ff_output  # Residual
            
            outputs.append(hidden)
        
        # Final layer norm
        hidden = self.model.transformer.ln_f(hidden)
        
        # Language model head
        logits = self.model.lm_head(hidden)
        
        return logits
    
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Training Example
def train_prefix_tuning(model, dataloader, num_epochs=5, lr=2e-5):
    """
    Train prefix tuning
    
    Only prefix parameters are updated
    """
    prefix_model = PrefixTuning(model, prefix_length=20, reparam=True)
    
    # Only optimize prefix parameters
    optimizer = torch.optim.Adam(
        prefix_model.parameters(), 
        lr=lr
    )
    
    prefix_model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # Forward pass
            logits = prefix_model(input_ids)
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        print(f"Trainable parameters: {prefix_model.get_trainable_parameters()}")
    
    return prefix_model
```

### Reparameterization Trick

Prefix tuning can use a reparameterization trick to improve training stability. Instead of directly learning the prefix embeddings, the method learns a smaller representation and projects it up. This reparameterization helps because the optimization landscape is smoother: learning in a lower-dimensional space and then projecting can be more stable than learning directly in the high-dimensional space.

The reparameterization works as follows: instead of learning prefix embeddings of size (prefix_length, d_model), we learn embeddings of size (prefix_length, d_model/2) and use a learned linear projection to map them to (prefix_length, d_model). This reduces the number of parameters while maintaining the same effective capacity, and often leads to more stable training.

### Comparison: Prompt Tuning vs Prefix Tuning

**Prompt Tuning:**
- Adds parameters only at input layer
- Simpler implementation
- Fewer parameters (p × d_model)
- May be less expressive for complex tasks
- Faster training (fewer parameters to update)

**Prefix Tuning:**
- Adds parameters at every layer
- More complex implementation
- More parameters (L × p × 2d_model, for K and V)
- More expressive, better for complex tasks
- Slower training (more parameters, but still efficient)

**When to Use:**
- **Prompt Tuning**: Simple tasks, maximum efficiency, limited compute
- **Prefix Tuning**: Complex tasks, need better performance, can afford slightly more parameters

---

## Part 3: Theory and Intuition

### Why Do These Methods Work?

Prompt tuning and prefix tuning work because they leverage the pre-trained model's ability to adapt its behavior based on context. Pre-trained language models are trained to be context-sensitive: they adjust their processing based on the input they receive. By adding trainable context (prompts or prefixes), we can guide the model's behavior without modifying its core parameters.

The key insight is that the model's attention mechanism naturally attends to the prompt/prefix tokens, and the representations learned in these tokens influence how the model processes the actual input. The prompt/prefix acts like a "task instruction" that the model learns to interpret and follow.

Another perspective is that prompts/prefixes act as a form of task-specific conditioning. The pre-trained model has learned general language understanding, and the prompts/prefixes provide task-specific signals that tell the model how to use this general knowledge for the specific task. This is similar to how few-shot learning works, but instead of using example text, we learn optimal continuous representations.

### Information-Theoretic Perspective

From an information-theoretic perspective, prompt tuning and prefix tuning are learning to encode task-specific information in a compact form. The prompt/prefix needs to convey enough information to guide the model's behavior for the task, but it's constrained to a small number of tokens. This forces the method to learn efficient, compressed representations of task knowledge.

The fact that these methods work with so few parameters suggests that much of the task-specific knowledge can be encoded in a small number of continuous tokens. This is consistent with the observation that language models can perform many tasks with just a few examples (few-shot learning), suggesting that the task knowledge is already largely present in the model, and we just need to provide the right "instructions" to activate it.

### Connection to In-Context Learning

Prompt tuning and prefix tuning are closely related to in-context learning, where models learn from examples provided in the prompt. In in-context learning, the model uses the examples to adapt its behavior, but the adaptation happens only during inference (no training). Prompt tuning and prefix tuning can be seen as "learning to in-context learn": they learn optimal prompt/prefix representations that would be equivalent to providing good examples, but in a more efficient, learned form.

This connection suggests that the learned prompts/prefixes might encode similar information to what would be in good few-shot examples, but in a compressed, continuous form. This explains why prompt tuning can sometimes match or exceed few-shot performance: it has learned the optimal "instructions" rather than relying on example selection.

---

## Part 4: Best Practices and Tips

### Initialization

Good initialization is crucial for prompt tuning and prefix tuning. Random initialization with small variance (e.g., std=0.02) is a common starting point, but vocabulary-based initialization (sampling from actual token embeddings) often works better. For prefix tuning, the reparameterization trick can help with initialization by starting in a lower-dimensional space.

### Learning Rate

Prompt tuning and prefix tuning typically require different learning rates than full fine-tuning. Since only a small number of parameters are being trained, they can often use higher learning rates (e.g., 0.3 for prompt tuning, 2e-5 for prefix tuning) without causing instability. The optimal learning rate depends on the task and should be tuned on a validation set.

### Prompt/Prefix Length

The length of prompts/prefixes is an important hyperparameter. Start with a moderate length (20-50 tokens) and adjust based on validation performance. Longer prompts/prefixes provide more capacity but may overfit on small datasets. For complex tasks, you might need longer prompts/prefixes, while simple tasks can work with shorter ones.

### Regularization

Since prompt tuning and prefix tuning have relatively few parameters, overfitting is less of a concern than with full fine-tuning. However, dropout can still be applied to the prompt/prefix embeddings during training. Additionally, early stopping based on validation performance can help prevent overfitting.

### Multi-Task Learning

One advantage of prompt/prefix tuning is the ability to train multiple task-specific prompts/prefixes while sharing the same base model. This can be done by training separate prompts/prefixes for each task, or by training a shared prompt/prefix that works for multiple related tasks. The latter approach can improve generalization and reduce the total number of parameters needed.

---

## Summary

Prompt tuning and prefix tuning are powerful parameter-efficient fine-tuning methods that add small numbers of trainable parameters while keeping the pre-trained model frozen. Prompt tuning adds parameters only at the input layer, making it extremely efficient, while prefix tuning adds parameters at every layer, providing more expressiveness. Both methods achieve competitive performance with full fine-tuning while requiring orders of magnitude fewer parameters, making them practical for deploying many task-specific models efficiently.

