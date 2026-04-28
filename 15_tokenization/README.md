# Topic 15: Tokenization Methods

> 🔥 **For interviews, read these first:**
> - **`TOKENIZATION_DEEP_DIVE.md`** — frontier-lab interview deep dive: BPE/WordPiece/Unigram/SentencePiece, byte-level BPE, vocabulary trade-offs, arithmetic and multilingual quirks, glitch tokens, special tokens, multimodal extensions.
> - **`INTERVIEW_GRILL.md`** — 45 active-recall questions.

## What You'll Learn

This topic teaches you different tokenization methods:
- BPE (Byte Pair Encoding)
- WordPiece
- SentencePiece
- Character-level
- Subword tokenization
- Simple implementations

## Why We Need This

### Interview Importance
- **Common question**: "How does BPE work?"
- **Understanding**: Tokenization is fundamental
- **Implementation**: May ask to implement BPE

### Real-World Application
- **All LLMs**: Use tokenization
- **GPT**: Uses BPE
- **BERT**: Uses WordPiece
- **T5**: Uses SentencePiece

## Industry Use Cases

### 1. **BPE (Byte Pair Encoding)**
**Use Case**: GPT-2, GPT-3, GPT-4
- Most common in modern LLMs
- Learns subword units
- Handles unknown words

### 2. **WordPiece**
**Use Case**: BERT
- Similar to BPE
- Different merging strategy
- Used in BERT family

### 3. **SentencePiece**
**Use Case**: T5, mT5
- Language-agnostic
- Handles multiple languages
- Used in multilingual models

## Core Intuition

Tokenization decides how raw text becomes model input units.

That sounds like a preprocessing detail, but it affects:
- vocabulary size
- sequence length
- memory cost
- softmax cost
- how well rare words and multilingual text are handled

This is why tokenization shows up in serious LLM interviews.

### Why Not Word-Level Tokens?

Word-level vocabularies have a major problem:
- too many rare or unseen words

Character-level vocabularies avoid that problem, but sequences become much longer.

Subword tokenization is the compromise:
- shorter than characters
- more flexible than words

### BPE

BPE repeatedly merges the most useful symbol pairs.

Intuition:
- common fragments become single units
- rare words can still be decomposed into smaller pieces

### WordPiece

WordPiece is similar in spirit to BPE, but the scoring of merges differs.

The interview-safe answer is:
- both build subwords
- they differ in how they choose which merges to keep

### SentencePiece

SentencePiece is useful because it can operate without assuming whitespace-separated words in the same way many older tokenizers do.

That makes it strong for:
- multilingual data
- languages with different segmentation properties

## Technical Details Interviewers Often Want

### Vocabulary Size Trade-Off

Large vocabulary:
- shorter sequences
- larger embedding and output layers

Small vocabulary:
- longer sequences
- more compositional coverage

This is one of the most important tokenization trade-offs to explain clearly.

### Tokenization Affects Everything Downstream

Bad tokenization can hurt:
- efficiency
- context length usage
- handling of code or multilingual text
- alignment between pretraining and downstream tasks

### "Unknown Word" Handling

Subword tokenization is powerful because it avoids a hard unknown-word failure mode for many tokens by decomposing unfamiliar words into smaller known pieces.

## Common Failure Modes

- treating tokenization as an unimportant preprocessing detail
- confusing BPE and WordPiece as identical
- ignoring the cost trade-off between vocab size and sequence length
- choosing tokenization poorly for multilingual or code data
- assuming more tokens always means more information rather than just more fragmentation

## Edge Cases and Follow-Up Questions

1. Why not use pure word-level tokens?
2. Why not use pure character-level tokens?
3. How does vocabulary size trade off against sequence length?
4. Why can tokenization matter a lot for multilingual models?
5. Why can tokenization affect both quality and cost?

## What to Practice Saying Out Loud

1. Why subword tokenization became dominant
2. The practical trade-off between vocabulary size and sequence length
3. Why tokenization is a modeling choice, not just a preprocessing choice

## Industry-Standard Boilerplate Code

### BPE (Simplified)

```python
"""
BPE: Byte Pair Encoding
Interview question: "Implement BPE"
"""
from collections import Counter, defaultdict

class BPE:
    """
    Simple BPE implementation
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
    
    def get_word_freqs(self, corpus: list) -> dict:
        """Get word frequencies"""
        word_freqs = Counter()
        for text in corpus:
            words = text.split()
            word_freqs.update(words)
        return dict(word_freqs)
    
    def get_splits(self, word: str) -> list:
        """Split word into characters"""
        return list(word) + ['</w>']  # End token
    
    def get_pairs(self, word: list) -> set:
        """Get all pairs of consecutive symbols"""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i+1]))
        return pairs
    
    def merge_vocab(self, pair: tuple, vocab: dict) -> dict:
        """Merge most frequent pair"""
        new_vocab = {}
        bigram = ''.join(pair)
        
        for word in vocab:
            new_word = word.replace(' '.join(pair), bigram)
            new_vocab[new_word] = vocab[word]
        
        return new_vocab
    
    def train(self, corpus: list):
        """Train BPE"""
        # Initialize vocabulary
        word_freqs = self.get_word_freqs(corpus)
        vocab = {}
        for word, freq in word_freqs.items():
            vocab[' '.join(self.get_splits(word))] = freq
        
        # Build vocabulary
        num_merges = self.vocab_size - 256  # Start with 256 base chars
        
        for i in range(num_merges):
            # Count pairs
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
            
            # Merge
            vocab = self.merge_vocab(best_pair, vocab)
        
        self.vocab = vocab
    
    def tokenize(self, text: str) -> list:
        """Tokenize text"""
        words = text.split()
        tokens = []
        
        for word in words:
            word_split = self.get_splits(word)
            
            # Apply merges
            for pair in self.merges:
                bigram = ' '.join(pair)
                if bigram in ' '.join(word_split):
                    word_split = ' '.join(word_split).replace(bigram, ''.join(pair)).split()
            
            tokens.extend(word_split)
        
        return tokens
```

### WordPiece (Simplified)

```python
"""
WordPiece: Similar to BPE but different scoring
"""
class WordPiece:
    """
    WordPiece tokenization
    Uses likelihood instead of frequency
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
    
    def get_score(self, pair: tuple, vocab: dict) -> float:
        """
        Score pair using likelihood
        score = count(pair) / (count(pair[0]) * count(pair[1]))
        """
        pair_count = sum(vocab.get(''.join(pair), 0) for vocab in [vocab])
        first_count = sum(1 for word in vocab if pair[0] in word)
        second_count = sum(1 for word in vocab if pair[1] in word)
        
        if first_count * second_count == 0:
            return 0
        
        return pair_count / (first_count * second_count)
    
    # Similar structure to BPE but uses score instead of frequency
```

## Theory

### BPE Algorithm
1. Start with character vocabulary
2. Count all pairs of consecutive symbols
3. Merge most frequent pair
4. Repeat until desired vocab size

### Why Subword Tokenization
- **Handles unknown words**: Can break into subwords
- **Balanced vocabulary**: Not too many (word-level) or too few (char-level)
- **Efficient**: Good compression

## Exercises

1. Implement BPE
2. Compare BPE vs WordPiece
3. Tokenize sample text
4. Measure vocabulary size

## Next Steps

- **Topic 16**: Training behaviors
- **Topic 17**: Probability math
