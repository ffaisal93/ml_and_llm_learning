"""
CLIP: Contrastive Language-Image Pre-training
Simplified implementation for understanding
"""
import numpy as np
from typing import List, Tuple

class SimpleCLIP:
    """
    Simplified CLIP model for educational purposes
    
    Architecture:
    - Image Encoder: Encodes images to embeddings
    - Text Encoder: Encodes text to embeddings
    - Contrastive Loss: Aligns matching text-image pairs
    """
    
    def __init__(self, image_dim: int = 512, text_dim: int = 512, embedding_dim: int = 512):
        """
        Args:
            image_dim: Dimension of image features
            text_dim: Dimension of text features
            embedding_dim: Dimension of shared embedding space
        """
        self.embedding_dim = embedding_dim
        
        # Image encoder (simplified: linear projection)
        self.image_proj = np.random.randn(image_dim, embedding_dim) * 0.01
        
        # Text encoder (simplified: linear projection)
        self.text_proj = np.random.randn(text_dim, embedding_dim) * 0.01
    
    def encode_image(self, image_features: np.ndarray) -> np.ndarray:
        """
        Encode image to embedding
        
        Args:
            image_features: Image features (batch_size, image_dim)
        
        Returns:
            Image embeddings (batch_size, embedding_dim)
        """
        # Project to embedding space
        embeddings = image_features @ self.image_proj
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def encode_text(self, text_features: np.ndarray) -> np.ndarray:
        """
        Encode text to embedding
        
        Args:
            text_features: Text features (batch_size, text_dim)
        
        Returns:
            Text embeddings (batch_size, embedding_dim)
        """
        # Project to embedding space
        embeddings = text_features @ self.text_proj
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def compute_similarity(self, image_embeddings: np.ndarray, 
                          text_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix
        
        Args:
            image_embeddings: (batch_size, embedding_dim)
            text_embeddings: (batch_size, embedding_dim)
        
        Returns:
            Similarity matrix (batch_size, batch_size)
            similarity[i, j] = similarity between image i and text j
        """
        # Dot product (cosine similarity since embeddings are normalized)
        similarity = image_embeddings @ text_embeddings.T
        
        return similarity
    
    def contrastive_loss(self, image_embeddings: np.ndarray, 
                        text_embeddings: np.ndarray,
                        temperature: float = 0.07) -> Tuple[float, float, float]:
        """
        Compute contrastive loss
        
        Loss = -log(exp(sim_pos / τ) / Σ exp(sim / τ))
        
        Where:
        - sim_pos: Similarity of matching pairs (diagonal)
        - sim: Similarity of all pairs
        - τ: Temperature parameter
        
        Returns:
            (total_loss, image_to_text_loss, text_to_image_loss)
        """
        batch_size = image_embeddings.shape[0]
        
        # Compute similarity matrix
        logits = self.compute_similarity(image_embeddings, text_embeddings)
        logits = logits / temperature  # Scale by temperature
        
        # Labels: diagonal is positive (image i matches text i)
        labels = np.arange(batch_size)
        
        # Image-to-text loss
        # For each image, find matching text
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
        probs_i2t = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        loss_i2t = -np.mean(np.log(probs_i2t[np.arange(batch_size), labels] + 1e-8))
        
        # Text-to-image loss
        # For each text, find matching image
        exp_logits_t = np.exp(logits.T - np.max(logits.T, axis=1, keepdims=True))
        probs_t2i = exp_logits_t / np.sum(exp_logits_t, axis=1, keepdims=True)
        loss_t2i = -np.mean(np.log(probs_t2i[np.arange(batch_size), labels] + 1e-8))
        
        # Total loss (symmetric)
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return total_loss, loss_i2t, loss_t2i
    
    def zero_shot_classification(self, image_features: np.ndarray,
                                class_prompts: List[str],
                                text_encoder_fn) -> np.ndarray:
        """
        Zero-shot image classification
        
        Args:
            image_features: Image features (1, image_dim)
            class_prompts: List of text prompts for each class
            text_encoder_fn: Function to encode text prompts
        
        Returns:
            Class probabilities (num_classes,)
        """
        # Encode image
        image_emb = self.encode_image(image_features)
        
        # Encode text prompts
        text_features = text_encoder_fn(class_prompts)  # (num_classes, text_dim)
        text_embs = self.encode_text(text_features)
        
        # Compute similarities
        similarities = self.compute_similarity(image_emb, text_embs)[0]  # (num_classes,)
        
        # Convert to probabilities (softmax)
        exp_sim = np.exp(similarities - np.max(similarities))
        probs = exp_sim / np.sum(exp_sim)
        
        return probs

def clip_training_example():
    """
    Example of CLIP training
    """
    print("CLIP Training Example")
    print("=" * 60)
    
    # Simulate data
    batch_size = 4
    image_dim = 512
    text_dim = 512
    embedding_dim = 512
    
    # Random image and text features (in practice, from encoders)
    image_features = np.random.randn(batch_size, image_dim)
    text_features = np.random.randn(batch_size, text_dim)
    
    # Create model
    model = SimpleCLIP(image_dim, text_dim, embedding_dim)
    
    # Forward pass
    image_embeddings = model.encode_image(image_features)
    text_embeddings = model.encode_text(text_features)
    
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print()
    
    # Compute similarity matrix
    similarity = model.compute_similarity(image_embeddings, text_embeddings)
    print("Similarity matrix (image i, text j):")
    print(similarity)
    print()
    print("Diagonal = matching pairs (should be high after training)")
    print()
    
    # Compute loss
    total_loss, loss_i2t, loss_t2i = model.contrastive_loss(image_embeddings, text_embeddings)
    print(f"Image-to-text loss: {loss_i2t:.4f}")
    print(f"Text-to-image loss: {loss_t2i:.4f}")
    print(f"Total loss: {total_loss:.4f}")
    print()
    
    print("Training objective:")
    print("  - Maximize similarity of matching pairs (diagonal)")
    print("  - Minimize similarity of non-matching pairs")
    print("  - After training, diagonal should be high, off-diagonal low")

def zero_shot_example():
    """
    Example of zero-shot classification
    """
    print("\nZero-Shot Classification Example")
    print("=" * 60)
    
    # Simulate image
    image_dim = 512
    image_features = np.random.randn(1, image_dim)
    
    # Class prompts
    class_prompts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a car"
    ]
    
    # Simple text encoder (in practice, use transformer)
    def simple_text_encoder(prompts):
        # Just return random features (in practice, use actual encoder)
        return np.random.randn(len(prompts), 512)
    
    # Create model
    model = SimpleCLIP(image_dim, 512, 512)
    
    # Classify
    probs = model.zero_shot_classification(image_features, class_prompts, simple_text_encoder)
    
    print("Class probabilities:")
    for prompt, prob in zip(class_prompts, probs):
        print(f"  {prompt}: {prob:.4f}")
    
    predicted_class = np.argmax(probs)
    print(f"\nPredicted class: {class_prompts[predicted_class]}")

if __name__ == "__main__":
    print("CLIP: Contrastive Language-Image Pre-training")
    print("=" * 60)
    
    # Training example
    clip_training_example()
    
    # Zero-shot example
    zero_shot_example()
    
    print("\n" + "=" * 60)
    print("Key Points:")
    print("  1. CLIP uses contrastive learning to align text and images")
    print("  2. Training: Maximize similarity of matching pairs")
    print("  3. Zero-shot: Can classify images without training on task")
    print("  4. In practice: Train on 400M text-image pairs for days/weeks")

