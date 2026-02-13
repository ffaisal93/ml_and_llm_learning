"""
N-gram Language Model with Laplace Smoothing
Complete implementation
"""
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class NGramLanguageModel:
    """
    N-gram Language Model
    
    Models probability of next word given previous n-1 words:
    P(wᵢ|wᵢ₋ₙ₊₁, ..., wᵢ₋₁)
    
    For bigram (n=2): P(wᵢ|wᵢ₋₁)
    For trigram (n=3): P(wᵢ|wᵢ₋₂, wᵢ₋₁)
    """
    
    def __init__(self, n: int = 2, smoothing: bool = True, k: float = 1.0):
        """
        Args:
            n: N-gram size (2=bigram, 3=trigram, etc.)
            smoothing: Use Laplace smoothing
            k: Smoothing parameter (k=1 is Laplace, k=0.5 is less aggressive)
        """
        self.n = n
        self.smoothing = smoothing
        self.k = k
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocabulary = set()
        self.total_words = 0
    
    def train(self, documents: List[List[str]]):
        """
        Train n-gram model on documents
        
        Steps:
        1. Add start/end markers
        2. Create n-grams
        3. Count n-grams and contexts
        4. Build vocabulary
        """
        for doc in documents:
            # Add start markers (n-1 markers for n-gram)
            if self.n > 1:
                doc = ['<START>'] * (self.n - 1) + doc + ['<END>']
            
            # Create n-grams
            for i in range(len(doc) - self.n + 1):
                ngram = tuple(doc[i:i+self.n])
                context = ngram[:-1] if self.n > 1 else tuple()
                
                self.ngram_counts[ngram] += 1
                if self.n > 1:
                    self.context_counts[context] += 1
                else:
                    self.total_words += 1
                
                # Add to vocabulary
                self.vocabulary.update(ngram)
    
    def probability(self, word: str, context: Tuple[str, ...] = None) -> float:
        """
        Compute probability P(word|context)
        
        Without smoothing:
        P(w|context) = count(context, w) / count(context)
        
        With smoothing (add-k):
        P(w|context) = (count(context, w) + k) / (count(context) + k*V)
        """
        if self.n == 1:
            # Unigram: P(w) = count(w) / total
            count = sum(1 for ngram in self.ngram_counts.keys() if ngram[0] == word)
            
            if self.smoothing:
                V = len(self.vocabulary)
                return (count + self.k) / (self.total_words + self.k * V)
            else:
                return count / self.total_words if self.total_words > 0 else 0.0
        
        else:
            # N-gram: P(w|context)
            if context is None:
                context = tuple()
            
            ngram = context + (word,)
            count = self.ngram_counts[ngram]
            context_count = self.context_counts[context]
            
            if self.smoothing:
                V = len(self.vocabulary)
                return (count + self.k) / (context_count + self.k * V)
            else:
                return count / context_count if context_count > 0 else 0.0
    
    def predict_next_word(self, context: Tuple[str, ...], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict most likely next words given context
        
        Returns top-k words with their probabilities
        """
        probs = []
        for word in self.vocabulary:
            if word not in ['<START>', '<END>']:
                prob = self.probability(word, context)
                probs.append((word, prob))
        
        # Sort by probability (descending)
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]
    
    def generate_text(self, start_words: List[str], max_length: int = 10) -> List[str]:
        """
        Generate text using n-gram model
        
        Steps:
        1. Start with given words
        2. Predict next word using context
        3. Add to sequence
        4. Repeat until max_length or <END>
        """
        if len(start_words) < self.n - 1:
            start_words = ['<START>'] * (self.n - 1 - len(start_words)) + start_words
        
        generated = start_words.copy()
        
        for _ in range(max_length):
            # Get context (last n-1 words)
            context = tuple(generated[-(self.n-1):]) if self.n > 1 else tuple()
            
            # Predict next word
            next_words = self.predict_next_word(context, top_k=1)
            if not next_words:
                break
            
            next_word = next_words[0][0]
            if next_word == '<END>':
                break
            
            generated.append(next_word)
        
        return generated


# Usage Example
if __name__ == "__main__":
    print("N-gram Language Model")
    print("=" * 60)
    
    # Training data
    documents = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "mat"],
        ["the", "cat", "and", "the", "dog"],
        ["the", "cat", "chased", "the", "dog"]
    ]
    
    print("Training data:")
    for doc in documents:
        print(f"  {' '.join(doc)}")
    print()
    
    # Train bigram model
    model = NGramLanguageModel(n=2, smoothing=True, k=1.0)
    model.train(documents)
    
    # Test predictions
    print("Predictions:")
    print("  P(cat|the) =", model.probability("cat", ("the",)))
    print("  P(dog|the) =", model.probability("dog", ("the",)))
    print("  P(bird|the) =", model.probability("bird", ("the",)), "(unseen, but > 0 with smoothing)")
    print()
    
    # Predict next words
    print("Most likely next words after 'the':")
    next_words = model.predict_next_word(("the",), top_k=3)
    for word, prob in next_words:
        print(f"  {word}: {prob:.4f}")
    print()
    
    # Generate text
    print("Generated text (starting with 'the'):")
    generated = model.generate_text(["the"], max_length=5)
    print(f"  {' '.join(generated)}")

