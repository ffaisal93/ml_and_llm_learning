"""
BPE: Byte Pair Encoding
Interview question: "Implement BPE"
Simple implementation
"""
from collections import Counter, defaultdict

class SimpleBPE:
    """
    Simple BPE implementation
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
    
    def get_word_freqs(self, corpus: list) -> dict:
        """Get word frequencies from corpus"""
        word_freqs = Counter()
        for text in corpus:
            words = text.split()
            word_freqs.update(words)
        return dict(word_freqs)
    
    def get_splits(self, word: str) -> list:
        """Split word into characters with end token"""
        return list(word) + ['</w>']
    
    def get_pairs(self, word: list) -> set:
        """Get all pairs of consecutive symbols"""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i+1]))
        return pairs
    
    def merge_vocab(self, pair: tuple, vocab: dict) -> dict:
        """Merge most frequent pair in vocabulary"""
        new_vocab = {}
        bigram = ''.join(pair)
        
        for word in vocab:
            # Replace pair with merged bigram
            new_word = word.replace(' '.join(pair), bigram)
            new_vocab[new_word] = vocab[word]
        
        return new_vocab
    
    def train(self, corpus: list):
        """Train BPE on corpus"""
        # Initialize vocabulary with character-level splits
        word_freqs = self.get_word_freqs(corpus)
        vocab = {}
        for word, freq in word_freqs.items():
            vocab[' '.join(self.get_splits(word))] = freq
        
        # Build vocabulary through merges
        num_merges = self.vocab_size - 256  # Start with 256 base chars
        
        for i in range(num_merges):
            # Count all pairs
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                word_pairs = self.get_pairs(word.split())
                for pair in word_pairs:
                    pairs[pair] += freq
            
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            # Merge the pair
            vocab = self.merge_vocab(best_pair, vocab)
        
        self.vocab = vocab
    
    def tokenize(self, text: str) -> list:
        """Tokenize text using learned BPE"""
        words = text.split()
        tokens = []
        
        for word in words:
            word_split = self.get_splits(word)
            
            # Apply all merges
            for pair in self.merges:
                bigram = ' '.join(pair)
                if bigram in ' '.join(word_split):
                    word_split = ' '.join(word_split).replace(bigram, ''.join(pair)).split()
            
            tokens.extend(word_split)
        
        return tokens


# Usage Example
if __name__ == "__main__":
    print("BPE Tokenization Example")
    print("=" * 60)
    
    # Sample corpus
    corpus = [
        "hello world",
        "hello there",
        "world peace",
        "hello hello"
    ]
    
    # Train BPE
    bpe = SimpleBPE(vocab_size=50)
    bpe.train(corpus)
    
    print(f"Number of merges: {len(bpe.merges)}")
    print(f"First few merges: {bpe.merges[:5]}")
    
    # Tokenize
    text = "hello world"
    tokens = bpe.tokenize(text)
    print(f"\nTokenizing '{text}':")
    print(f"Tokens: {tokens}")

