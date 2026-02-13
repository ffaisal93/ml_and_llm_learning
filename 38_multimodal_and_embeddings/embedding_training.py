"""
How to Train Embedding Models
Complete implementations and explanations
"""
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import math

# ==================== WORD2VEC SKIP-GRAM ====================

class Word2VecSkipGram:
    """
    Word2Vec Skip-gram Model
    
    Architecture:
    - Input: One-hot vector for center word
    - Hidden: Embedding layer (V × d)
    - Output: Softmax over vocabulary (predict context words)
    
    Training:
    - Given center word, predict context words
    - Use negative sampling for efficiency
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings (input and output)
        # In practice, these are separate matrices
        self.W_input = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_output = np.random.randn(embedding_dim, vocab_size) * 0.01
    
    def create_training_pairs(self, text: List[int], window_size: int = 5) -> List[Tuple[int, int]]:
        """
        Create (center_word, context_word) pairs
        
        Args:
            text: List of word indices
            window_size: Context window size
        
        Returns:
            List of (center, context) pairs
        """
        pairs = []
        for i, center_word in enumerate(text):
            # Context: words in window [i-window, i+window]
            start = max(0, i - window_size)
            end = min(len(text), i + window_size + 1)
            
            for j in range(start, end):
                if j != i:  # Don't include center word itself
                    context_word = text[j]
                    pairs.append((center_word, context_word))
        
        return pairs
    
    def negative_sampling_loss(self, center_word: int, context_word: int, 
                               negative_samples: List[int], 
                               learning_rate: float = 0.01) -> float:
        """
        Compute loss with negative sampling
        
        Loss = -log σ(v_context · v_center) - Σ log σ(-v_neg · v_center)
        
        Where σ = sigmoid function
        """
        # Get embeddings
        v_center = self.W_input[center_word]  # (embedding_dim,)
        v_context = self.W_output[:, context_word]  # (embedding_dim,)
        
        # Positive example (context word)
        pos_score = np.dot(v_center, v_context)
        pos_loss = -np.log(self._sigmoid(pos_score))
        
        # Negative examples
        neg_loss = 0.0
        for neg_word in negative_samples:
            v_neg = self.W_output[:, neg_word]
            neg_score = np.dot(v_center, v_neg)
            neg_loss += -np.log(self._sigmoid(-neg_score))
        
        total_loss = pos_loss + neg_loss
        
        # Update embeddings (simplified gradient)
        # In practice, use proper backpropagation
        grad_center = (self._sigmoid(pos_score) - 1) * v_context
        for neg_word in negative_samples:
            v_neg = self.W_output[:, neg_word]
            neg_score = np.dot(v_center, v_neg)
            grad_center += self._sigmoid(neg_score) * v_neg
        
        # Update (simplified)
        self.W_input[center_word] -= learning_rate * grad_center
        
        return total_loss
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Get embedding for a word"""
        return self.W_input[word_idx]

# ==================== GLOVE ====================

class GloVe:
    """
    GloVe: Global Vectors for Word Representation
    
    Objective: w_i · w_j + b_i + b_j ≈ log(X_ij)
    
    Where:
    - w_i, w_j: Word embeddings
    - X_ij: Co-occurrence count
    - b_i, b_j: Bias terms
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Word embeddings
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Bias terms
        self.b = np.zeros(vocab_size)
        
        # Co-occurrence matrix (sparse, stored as dict)
        self.cooccurrence = defaultdict(float)
    
    def build_cooccurrence_matrix(self, text: List[int], window_size: int = 10):
        """
        Build co-occurrence matrix
        
        X_ij = count of times word j appears in context of word i
        """
        for i, word_i in enumerate(text):
            start = max(0, i - window_size)
            end = min(len(text), i + window_size + 1)
            
            for j in range(start, end):
                if j != i:
                    word_j = text[j]
                    # Distance weighting (closer words count more)
                    distance = abs(i - j)
                    weight = 1.0 / distance
                    self.cooccurrence[(word_i, word_j)] += weight
    
    def weighting_function(self, x: float, x_max: float = 100.0, alpha: float = 0.75) -> float:
        """
        Weighting function for GloVe
        
        f(x) = (x/x_max)^α if x < x_max else 1
        """
        if x < x_max:
            return (x / x_max) ** alpha
        else:
            return 1.0
    
    def compute_loss(self, word_i: int, word_j: int, X_ij: float) -> float:
        """
        Compute GloVe loss for word pair
        
        Loss = f(X_ij) (w_i · w_j + b_i + b_j - log X_ij)²
        """
        # Get embeddings
        w_i = self.W[word_i]
        w_j = self.W[word_j]
        
        # Compute dot product
        dot_product = np.dot(w_i, w_j)
        
        # Compute target
        target = math.log(X_ij + 1)  # +1 to avoid log(0)
        
        # Compute error
        error = dot_product + self.b[word_i] + self.b[word_j] - target
        
        # Weighting
        weight = self.weighting_function(X_ij)
        
        # Loss
        loss = weight * (error ** 2)
        
        return loss
    
    def train_step(self, word_i: int, word_j: int, X_ij: float, learning_rate: float = 0.01):
        """
        Single training step for GloVe
        """
        # Compute gradients (simplified)
        w_i = self.W[word_i]
        w_j = self.W[word_j]
        
        target = math.log(X_ij + 1)
        error = np.dot(w_i, w_j) + self.b[word_i] + self.b[word_j] - target
        weight = self.weighting_function(X_ij)
        
        # Gradients
        grad_w_i = 2 * weight * error * w_j
        grad_w_j = 2 * weight * error * w_i
        grad_b_i = 2 * weight * error
        grad_b_j = 2 * weight * error
        
        # Update
        self.W[word_i] -= learning_rate * grad_w_i
        self.W[word_j] -= learning_rate * grad_w_j
        self.b[word_i] -= learning_rate * grad_b_i
        self.b[word_j] -= learning_rate * grad_b_j
    
    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Get embedding for a word"""
        return self.W[word_idx]

# ==================== TRAINING EXAMPLES ====================

def word2vec_training_example():
    """
    Example of training Word2Vec Skip-gram
    """
    print("Word2Vec Skip-gram Training Example")
    print("=" * 60)
    
    # Simple vocabulary
    vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
    vocab_size = len(vocab)
    
    # Simple text
    text = ["the", "cat", "sat", "on", "the", "mat"]
    text_indices = [vocab[word] for word in text]
    
    # Create model
    model = Word2VecSkipGram(vocab_size, embedding_dim=10)
    
    # Create training pairs
    pairs = model.create_training_pairs(text_indices, window_size=2)
    print(f"Created {len(pairs)} training pairs")
    print(f"Example pairs: {pairs[:5]}")
    print()
    
    # Training (simplified)
    print("Training (simplified example):")
    for epoch in range(10):
        total_loss = 0.0
        for center, context in pairs[:5]:  # Just first 5 for demo
            # Negative sampling (simplified: random words)
            negative_samples = [i for i in range(vocab_size) if i != context][:3]
            loss = model.negative_sampling_loss(center, context, negative_samples)
            total_loss += loss
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    
    print("\nLearned embeddings:")
    for word, idx in vocab.items():
        embedding = model.get_embedding(idx)
        print(f"  {word}: {embedding[:5]}...")  # Show first 5 dims

def glove_training_example():
    """
    Example of training GloVe
    """
    print("\nGloVe Training Example")
    print("=" * 60)
    
    # Simple vocabulary
    vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
    vocab_size = len(vocab)
    
    # Simple text
    text = ["the", "cat", "sat", "on", "the", "mat"] * 10  # Repeat for more co-occurrences
    text_indices = [vocab[word] for word in text]
    
    # Create model
    model = GloVe(vocab_size, embedding_dim=10)
    
    # Build co-occurrence matrix
    model.build_cooccurrence_matrix(text_indices, window_size=5)
    print(f"Co-occurrence pairs: {len(model.cooccurrence)}")
    print(f"Example co-occurrences:")
    for (i, j), count in list(model.cooccurrence.items())[:5]:
        word_i = [w for w, idx in vocab.items() if idx == i][0]
        word_j = [w for w, idx in vocab.items() if idx == j][0]
        print(f"  ({word_i}, {word_j}): {count:.2f}")
    print()
    
    # Training
    print("Training:")
    for epoch in range(20):
        total_loss = 0.0
        for (word_i, word_j), X_ij in list(model.cooccurrence.items())[:10]:  # First 10 for demo
            loss = model.compute_loss(word_i, word_j, X_ij)
            model.train_step(word_i, word_j, X_ij, learning_rate=0.01)
            total_loss += loss
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    
    print("\nLearned embeddings:")
    for word, idx in vocab.items():
        embedding = model.get_embedding(idx)
        print(f"  {word}: {embedding[:5]}...")  # Show first 5 dims

# ==================== USAGE ====================

if __name__ == "__main__":
    print("Embedding Model Training: Word2Vec and GloVe")
    print("=" * 60)
    
    # Word2Vec example
    word2vec_training_example()
    
    # GloVe example
    glove_training_example()
    
    print("\n" + "=" * 60)
    print("Key Points:")
    print("  - Word2Vec: Predict context from center word (Skip-gram)")
    print("  - GloVe: Preserve co-occurrence statistics")
    print("  - Both learn dense, low-dimensional embeddings")
    print("  - In practice, train on billions of words for days")

