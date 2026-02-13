"""
Chunking Strategies: Complete Implementations
All chunking methods with detailed code
"""
import numpy as np
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ==================== BASE CLASSES ====================

@dataclass
class Chunk:
    """Text chunk with metadata"""
    content: str
    chunk_id: str
    metadata: Dict = None
    start_index: int = 0
    end_index: int = 0

class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Split text into chunks"""
        raise NotImplementedError

# ==================== 1. FIXED-SIZE CHUNKING ====================

class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-size chunking with overlap
    
    Use Case: Simple documents, fast prototyping, uniform content
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Args:
            chunk_size: Size of each chunk (characters)
            overlap: Overlap between chunks (characters)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Split text into fixed-size chunks"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunk = Chunk(
                content=chunk_text,
                chunk_id=f"chunk_{chunk_index}",
                metadata=metadata or {},
                start_index=start,
                end_index=end
            )
            chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.overlap
            chunk_index += 1
        
        return chunks

# ==================== 2. SENTENCE-BASED CHUNKING ====================

class SentenceBasedChunker(ChunkingStrategy):
    """
    Sentence-based chunking
    
    Use Case: Narrative text, general documents, production systems
    """
    
    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 100):
        """
        Args:
            max_chunk_size: Maximum chunk size (characters)
            min_chunk_size: Minimum chunk size (characters)
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simplified)"""
        # In practice, use NLTK or spaCy
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Split text into sentence-based chunks"""
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_index = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds max size, create new chunk
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = Chunk(
                        content=chunk_text,
                        chunk_id=f"chunk_{chunk_index}",
                        metadata=metadata or {},
                        start_index=start_index,
                        end_index=start_index + len(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    start_index += len(chunk_text) + 1
                
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for space
        
        # Add remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = Chunk(
                    content=chunk_text,
                    chunk_id=f"chunk_{chunk_index}",
                    metadata=metadata or {},
                    start_index=start_index,
                    end_index=start_index + len(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks

# ==================== 3. PARAGRAPH-BASED CHUNKING ====================

class ParagraphBasedChunker(ChunkingStrategy):
    """
    Paragraph-based chunking
    
    Use Case: Structured documents, academic papers, long-form content
    """
    
    def __init__(self, max_paragraphs_per_chunk: int = 3, min_chunk_size: int = 100):
        """
        Args:
            max_paragraphs_per_chunk: Maximum paragraphs per chunk
            min_chunk_size: Minimum chunk size
        """
        self.max_paragraphs = max_paragraphs_per_chunk
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Split text into paragraph-based chunks"""
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        chunk_index = 0
        start_index = 0
        
        for para in paragraphs:
            if len(current_chunk) >= self.max_paragraphs:
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = Chunk(
                        content=chunk_text,
                        chunk_id=f"chunk_{chunk_index}",
                        metadata=metadata or {},
                        start_index=start_index,
                        end_index=start_index + len(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    start_index += len(chunk_text) + 2
                
                current_chunk = [para]
            else:
                current_chunk.append(para)
        
        # Add remaining
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = Chunk(
                    content=chunk_text,
                    chunk_id=f"chunk_{chunk_index}",
                    metadata=metadata or {},
                    start_index=start_index,
                    end_index=start_index + len(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks

# ==================== 4. SEMANTIC CHUNKING ====================

class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking using embeddings
    
    Use Case: High accuracy needs, topic-based documents, complex documents
    """
    
    def __init__(self, similarity_threshold: float = 0.7, 
                 max_chunk_size: int = 512, min_chunk_size: int = 100,
                 embedding_fn=None):
        """
        Args:
            similarity_threshold: Threshold for semantic similarity
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
            embedding_fn: Function to generate embeddings
        """
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.embedding_fn = embedding_fn or self._dummy_embedding
    
    def _dummy_embedding(self, text: str) -> np.ndarray:
        """Dummy embedding (in practice, use actual model)"""
        # Simplified: hash-based embedding
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(384)
        return embedding / np.linalg.norm(embedding)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Split text using semantic similarity"""
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # Generate embeddings
        embeddings = [self.embedding_fn(s) for s in sentences]
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        current_size = len(sentences[0])
        chunk_index = 0
        start_index = 0
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i]
            sentence_size = len(sentence)
            
            # Compute similarity
            similarity = self._cosine_similarity(current_embedding, sentence_embedding)
            
            # Check if should split
            should_split = (
                similarity < self.similarity_threshold or
                current_size + sentence_size > self.max_chunk_size
            )
            
            if should_split and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = Chunk(
                        content=chunk_text,
                        chunk_id=f"chunk_{chunk_index}",
                        metadata=metadata or {},
                        start_index=start_index,
                        end_index=start_index + len(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    start_index += len(chunk_text) + 1
                
                current_chunk = [sentence]
                current_embedding = sentence_embedding
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                # Update embedding (weighted average)
                current_embedding = (current_embedding * current_size + 
                                   sentence_embedding * sentence_size) / (current_size + sentence_size)
                current_size += sentence_size + 1
        
        # Add remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = Chunk(
                    content=chunk_text,
                    chunk_id=f"chunk_{chunk_index}",
                    metadata=metadata or {},
                    start_index=start_index,
                    end_index=start_index + len(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks

# ==================== 5. RECURSIVE CHUNKING ====================

class RecursiveChunker(ChunkingStrategy):
    """
    Recursive chunking with hierarchical separators
    
    Use Case: General-purpose, variable structure, production systems
    """
    
    def __init__(self, chunk_size: int = 512, 
                 separators: List[str] = None):
        """
        Args:
            chunk_size: Target chunk size
            separators: Hierarchical separators (largest to smallest)
        """
        self.chunk_size = chunk_size
        self.separators = separators or ['\n\n', '\n', '. ', ' ']
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Recursively split text"""
        return self._recursive_chunk(text, self.separators, metadata or {}, 0)
    
    def _recursive_chunk(self, text: str, separators: List[str], 
                        metadata: Dict, start_index: int) -> List[Chunk]:
        """Recursive chunking helper"""
        if not separators:
            # No separators left, use fixed-size
            fixed_chunker = FixedSizeChunker(self.chunk_size, self.chunk_size // 10)
            chunks = fixed_chunker.chunk(text, metadata)
            # Update start indices
            for i, chunk in enumerate(chunks):
                chunk.start_index = start_index + i * self.chunk_size
                chunk.end_index = chunk.start_index + len(chunk.content)
            return chunks
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator not in text:
            # Separator not found, try next
            return self._recursive_chunk(text, remaining_separators, metadata, start_index)
        
        splits = text.split(separator)
        
        # Check if all splits fit
        if all(len(s) <= self.chunk_size for s in splits):
            # All fit, return them
            chunks = []
            current_index = start_index
            for i, split in enumerate(splits):
                if split.strip():
                    chunk = Chunk(
                        content=split.strip(),
                        chunk_id=f"chunk_{i}",
                        metadata=metadata,
                        start_index=current_index,
                        end_index=current_index + len(split.strip())
                    )
                    chunks.append(chunk)
                    current_index += len(split) + len(separator)
            return chunks
        else:
            # Some too large, recurse
            chunks = []
            current_index = start_index
            for split in splits:
                if len(split) <= self.chunk_size:
                    if split.strip():
                        chunk = Chunk(
                            content=split.strip(),
                            chunk_id=f"chunk_{len(chunks)}",
                            metadata=metadata,
                            start_index=current_index,
                            end_index=current_index + len(split.strip())
                        )
                        chunks.append(chunk)
                    current_index += len(split) + len(separator)
                else:
                    # Recurse with next separator
                    sub_chunks = self._recursive_chunk(split, remaining_separators, metadata, current_index)
                    chunks.extend(sub_chunks)
                    current_index += len(split) + len(separator)
            return chunks

# ==================== 6. SLIDING WINDOW CHUNKING ====================

class SlidingWindowChunker(ChunkingStrategy):
    """
    Sliding window chunking with stride
    
    Use Case: Sequential information, code, long documents
    """
    
    def __init__(self, chunk_size: int = 512, stride: int = 256):
        """
        Args:
            chunk_size: Size of each chunk
            stride: Step size (overlap = chunk_size - stride)
        """
        self.chunk_size = chunk_size
        self.stride = stride
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Split text with sliding window"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunk = Chunk(
                content=chunk_text,
                chunk_id=f"chunk_{chunk_index}",
                metadata=metadata or {},
                start_index=start,
                end_index=end
            )
            chunks.append(chunk)
            
            start += self.stride
            chunk_index += 1
        
        return chunks

# ==================== 7. TOKEN-BASED CHUNKING ====================

class TokenBasedChunker(ChunkingStrategy):
    """
    Token-based chunking
    
    Use Case: LLM systems, accurate token counting, cost optimization
    """
    
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50,
                 tokenizer_fn=None):
        """
        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap in tokens
            tokenizer_fn: Function to tokenize text
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer_fn = tokenizer_fn or self._simple_tokenizer
    
    def _simple_tokenizer(self, text: str) -> List[str]:
        """Simple tokenizer (in practice, use actual tokenizer)"""
        # Simplified: split by whitespace
        return text.split()
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Split text by token count"""
        tokens = self.tokenizer_fn(text)
        
        chunks = []
        start = 0
        chunk_index = 0
        char_start = 0
        
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Reconstruct text (simplified)
            chunk_text = ' '.join(chunk_tokens)
            
            chunk = Chunk(
                content=chunk_text,
                chunk_id=f"chunk_{chunk_index}",
                metadata=metadata or {},
                start_index=char_start,
                end_index=char_start + len(chunk_text)
            )
            chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.overlap_tokens
            char_start += len(chunk_text) + 1  # Approximate
            chunk_index += 1
        
        return chunks

# ==================== COMPARISON AND USAGE ====================

def compare_chunking_strategies():
    """Compare different chunking strategies"""
    print("Chunking Strategies Comparison")
    print("=" * 60)
    
    # Sample text
    text = """
    Machine learning is a subset of artificial intelligence. 
    It enables computers to learn from data without being explicitly programmed.
    
    Deep learning uses neural networks with multiple layers. 
    These networks can learn complex patterns in data.
    
    Natural language processing is another important field. 
    It focuses on understanding and generating human language.
    """
    
    strategies = [
        ("Fixed-Size", FixedSizeChunker(100, 20)),
        ("Sentence-Based", SentenceBasedChunker(200, 50)),
        ("Paragraph-Based", ParagraphBasedChunker(2, 50)),
        ("Recursive", RecursiveChunker(200)),
        ("Sliding Window", SlidingWindowChunker(100, 50)),
    ]
    
    print(f"Original text length: {len(text)} characters\n")
    
    for name, chunker in strategies:
        chunks = chunker.chunk(text)
        print(f"{name}:")
        print(f"  Number of chunks: {len(chunks)}")
        print(f"  Average chunk size: {np.mean([len(c.content) for c in chunks]):.1f} chars")
        print(f"  Chunk sizes: {[len(c.content) for c in chunks]}")
        print()

if __name__ == "__main__":
    print("Chunking Strategies: Complete Implementations")
    print("=" * 60)
    
    # Example text
    text = """
    Machine learning is a subset of artificial intelligence. 
    It enables computers to learn from data without being explicitly programmed.
    
    Deep learning uses neural networks with multiple layers. 
    These networks can learn complex patterns in data.
    
    Natural language processing is another important field. 
    It focuses on understanding and generating human language.
    """
    
    # Test different strategies
    print("1. Fixed-Size Chunking:")
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk(text)
    for i, chunk in enumerate(chunks, 1):
        print(f"   Chunk {i}: {chunk.content[:50]}...")
    print()
    
    print("2. Sentence-Based Chunking:")
    chunker = SentenceBasedChunker(max_chunk_size=200, min_chunk_size=50)
    chunks = chunker.chunk(text)
    for i, chunk in enumerate(chunks, 1):
        print(f"   Chunk {i}: {chunk.content[:50]}...")
    print()
    
    print("3. Paragraph-Based Chunking:")
    chunker = ParagraphBasedChunker(max_paragraphs_per_chunk=2)
    chunks = chunker.chunk(text)
    for i, chunk in enumerate(chunks, 1):
        print(f"   Chunk {i}: {chunk.content[:50]}...")
    print()
    
    # Comparison
    compare_chunking_strategies()
    
    print("\nUse Cases:")
    print("  - Fixed-Size: Simple, fast, prototyping")
    print("  - Sentence-Based: General documents, production")
    print("  - Paragraph-Based: Structured documents, papers")
    print("  - Semantic: High accuracy, topic-based")
    print("  - Recursive: General-purpose, robust")
    print("  - Sliding Window: Sequential data, code")
    print("  - Token-Based: LLM systems, accurate sizing")

