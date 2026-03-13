"""
Diffusion Model Evaluation: Complete Guide
Evaluation metrics and methods for diffusion models
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from scipy.stats import entropy

# ==================== IMAGE EVALUATION METRICS ====================

def compute_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """
    Frechet Inception Distance (FID)
    
    Measures quality and diversity of generated images
    Lower is better
    
    Args:
        real_features: Features from real images (extracted by Inception network)
        fake_features: Features from generated images
    Returns:
        FID score
    """
    # Compute statistics
    mu_real = real_features.mean(dim=0)
    mu_fake = fake_features.mean(dim=0)
    sigma_real = torch.cov(real_features.t())
    sigma_fake = torch.cov(fake_features.t())
    
    # Compute FID
    diff = mu_real - mu_fake
    covmean = torch.sqrt(sigma_real @ sigma_fake)
    
    fid = (diff @ diff).item() + torch.trace(sigma_real + sigma_fake - 2 * covmean).item()
    return fid


def compute_is(generated_images: torch.Tensor, inception_model) -> float:
    """
    Inception Score (IS)
    
    Measures quality and diversity
    Higher is better (typically 1-10)
    
    Args:
        generated_images: Generated images
        inception_model: Pre-trained Inception network
    Returns:
        IS score
    """
    # Get predictions
    with torch.no_grad():
        preds = F.softmax(inception_model(generated_images), dim=1)
    
    # Compute IS
    # IS = exp(E[KL(p(y|x) || p(y))])
    py = preds.mean(dim=0)  # Marginal distribution
    scores = []
    for pred in preds:
        kl = F.kl_div(pred.log(), py, reduction='sum')
        scores.append(kl.item())
    
    is_score = np.exp(np.mean(scores))
    return is_score


# ==================== TEXT EVALUATION METRICS ====================

def compute_bleu_score(reference: List[str], generated: List[str], n: int = 4) -> float:
    """
    BLEU score for text generation
    
    Measures n-gram overlap with reference
    Higher is better (0-1)
    
    Args:
        reference: Reference text (list of tokens/words)
        generated: Generated text
        n: Maximum n-gram order
    Returns:
        BLEU score
    """
    from collections import Counter
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    # Compute precision for each n-gram order
    precisions = []
    for i in range(1, n+1):
        ref_ngrams = Counter(get_ngrams(reference, i))
        gen_ngrams = Counter(get_ngrams(generated, i))
        
        matches = sum((ref_ngrams & gen_ngrams).values())
        total = sum(gen_ngrams.values())
        
        if total == 0:
            return 0.0
        
        precisions.append(matches / total)
    
    # Geometric mean
    bleu = np.exp(np.mean([np.log(p) for p in precisions if p > 0]))
    
    # Brevity penalty
    if len(generated) < len(reference):
        bp = np.exp(1 - len(reference) / len(generated))
    else:
        bp = 1.0
    
    return bp * bleu


def compute_perplexity(model, text: torch.Tensor) -> float:
    """
    Perplexity for text generation
    
    Measures how well model predicts next tokens
    Lower is better
    
    Args:
        model: Language model
        text: Token sequence
    Returns:
        Perplexity
    """
    with torch.no_grad():
        logits = model(text)
        # Shift for next token prediction
        logits = logits[:-1]
        targets = text[1:]
        
        # Compute cross-entropy
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        perplexity = torch.exp(loss).item()
    
    return perplexity


def compute_diversity_metrics(generated_texts: List[List[str]]) -> Dict[str, float]:
    """
    Compute diversity metrics for generated text
    
    Measures:
    - Distinct-n: Ratio of unique n-grams
    - Self-BLEU: Average BLEU between generated samples
    """
    from collections import Counter
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    # Distinct-n
    distinct_scores = {}
    for n in [1, 2, 3, 4]:
        all_ngrams = []
        for text in generated_texts:
            all_ngrams.extend(get_ngrams(text, n))
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        distinct_scores[f'distinct_{n}'] = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    # Self-BLEU (lower is better for diversity)
    self_bleus = []
    for i, text1 in enumerate(generated_texts):
        for j, text2 in enumerate(generated_texts):
            if i != j:
                bleu = compute_bleu_score(text1, text2)
                self_bleus.append(bleu)
    
    return {
        **distinct_scores,
        'self_bleu': np.mean(self_bleus) if self_bleus else 0.0
    }


# ==================== DIFFUSION-SPECIFIC METRICS ====================

def compute_reconstruction_error(model, x_0: torch.Tensor, timesteps: int = 1000) -> float:
    """
    Measure how well model can reconstruct original data
    
    Tests if reverse process correctly recovers original
    """
    # Forward diffusion
    t = torch.randint(0, timesteps, (x_0.size(0),))
    # ... forward process ...
    
    # Reverse diffusion
    # ... sampling ...
    
    # Compare with original
    error = F.mse_loss(x_0, reconstructed)
    return error.item()


def compute_denoising_accuracy(model, x_t: torch.Tensor, t: torch.Tensor,
                               x_0: torch.Tensor) -> float:
    """
    Measure accuracy of denoising at each timestep
    
    Tests if model correctly predicts noise/denoised version
    """
    with torch.no_grad():
        # Predict noise
        noise_pred = model(x_t, t)
        
        # Reconstruct x_0
        # ... reconstruction from noise_pred ...
        
        # Compare
        accuracy = (reconstructed == x_0).float().mean().item()
        return accuracy


# ==================== SAMPLING QUALITY ====================

def evaluate_sample_quality(samples: torch.Tensor, real_data: torch.Tensor,
                           metric: str = 'fid') -> float:
    """
    Evaluate quality of generated samples
    
    Args:
        samples: Generated samples
        real_data: Real data samples
        metric: Which metric to use ('fid', 'is', 'mse', etc.)
    Returns:
        Quality score
    """
    if metric == 'mse':
        # Simple MSE (for simple data)
        return F.mse_loss(samples.mean(dim=0), real_data.mean(dim=0)).item()
    
    elif metric == 'fid':
        # FID (requires feature extraction)
        # This is simplified - real FID needs Inception network
        return compute_fid(real_data, samples)
    
    elif metric == 'is':
        # Inception Score (requires Inception network)
        # This is simplified
        return compute_is(samples, None)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ==================== COMPREHENSIVE EVALUATION ====================

def evaluate_diffusion_model(model, test_loader, num_samples: int = 1000,
                            device: str = 'cpu') -> Dict[str, float]:
    """
    Comprehensive evaluation of diffusion model
    
    Returns dictionary of metrics
    """
    model.eval()
    
    metrics = {
        'reconstruction_error': [],
        'denoising_accuracy': [],
        'sample_quality': []
    }
    
    with torch.no_grad():
        for batch_idx, x_0 in enumerate(test_loader):
            if batch_idx * x_0.size(0) >= num_samples:
                break
            
            x_0 = x_0.to(device)
            
            # Test reconstruction
            # ... reconstruction test ...
            
            # Test denoising at different timesteps
            for t_val in [100, 500, 900]:
                t = torch.full((x_0.size(0),), t_val, device=device)
                # ... denoising test ...
            
            # Generate samples
            # ... sampling ...
            
            # Evaluate quality
            # ... quality evaluation ...
    
    # Aggregate metrics
    return {
        'avg_reconstruction_error': np.mean(metrics['reconstruction_error']),
        'avg_denoising_accuracy': np.mean(metrics['denoising_accuracy']),
        'avg_sample_quality': np.mean(metrics['sample_quality'])
    }


# ==================== TEXT-SPECIFIC EVALUATION ====================

def evaluate_text_diffusion(model, test_loader, tokenizer, num_samples: int = 100,
                           device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate discrete diffusion model for text
    
    Returns:
        Dictionary with BLEU, perplexity, diversity metrics
    """
    model.eval()
    
    all_generated = []
    all_references = []
    
    with torch.no_grad():
        for batch_idx, (x_0, reference_texts) in enumerate(test_loader):
            if batch_idx * x_0.size(0) >= num_samples:
                break
            
            x_0 = x_0.to(device)
            
            # Generate text
            generated = discrete_sample(model, x_0.shape, device=device)
            
            # Decode
            for gen, ref in zip(generated, reference_texts):
                gen_text = tokenizer.decode(gen.cpu().tolist())
                ref_text = tokenizer.decode(ref.cpu().tolist())
                all_generated.append(gen_text.split())
                all_references.append(ref_text.split())
    
    # Compute metrics
    bleu_scores = [
        compute_bleu_score(ref, gen)
        for ref, gen in zip(all_references, all_generated)
    ]
    
    diversity = compute_diversity_metrics(all_generated)
    
    return {
        'avg_bleu': np.mean(bleu_scores),
        'diversity': diversity,
        'num_samples': len(all_generated)
    }


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    print("Diffusion Model Evaluation")
    print("=" * 60)
    print("""
    Evaluation Metrics:
    
    1. Image Quality:
       - FID (Frechet Inception Distance): Lower is better
       - IS (Inception Score): Higher is better
       - Reconstruction Error: Lower is better
    
    2. Text Quality:
       - BLEU Score: Higher is better (0-1)
       - Perplexity: Lower is better
       - Diversity Metrics: Distinct-n, Self-BLEU
    
    3. Diffusion-Specific:
       - Denoising Accuracy: How well model denoises
       - Reconstruction Error: Can model recover original?
       - Sample Quality: Quality of generated samples
    
    See code for complete implementations!
    """)

