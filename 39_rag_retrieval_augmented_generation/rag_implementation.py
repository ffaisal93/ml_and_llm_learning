"""
RAG Implementation: Industry-Standard Code
Production-ready RAG system with all components
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import re

# ==================== DATA STRUCTURES ====================

@dataclass
class Document:
    """Document with metadata"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

@dataclass
class Chunk:
    """Text chunk with metadata"""
    id: str
    content: str
    document_id: str
    chunk_index: int
    metadata: Dict
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    """Retrieval result with score"""
    chunk: Chunk
    score: float
    rank: int

# ==================== CHUNKING STRATEGIES ====================
# Note: For complete chunking implementations with all strategies,
# see chunking_implementations.py

class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into chunks"""
        raise NotImplementedError

class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-size chunking with overlap
    
    Industry standard: 512-1024 tokens with 50-100 token overlap
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Args:
            chunk_size: Size of each chunk (in characters or tokens)
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into fixed-size chunks"""
        chunks = []
        text = document.content
        start = 0
        
        chunk_index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            # Create chunk
            chunk = Chunk(
                id=f"{document.id}_chunk_{chunk_index}",
                content=chunk_text,
                document_id=document.id,
                chunk_index=chunk_index,
                metadata={
                    **document.metadata,
                    "start_char": start,
                    "end_char": end
                }
            )
            chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.overlap
            chunk_index += 1
        
        return chunks

class SentenceBasedChunker(ChunkingStrategy):
    """
    Sentence-based chunking
    
    Splits on sentence boundaries, respects semantic units
    """
    
    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into sentence-based chunks"""
        # Simple sentence splitting (in practice, use NLTK, spaCy)
        sentences = re.split(r'[.!?]+\s+', document.content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds max size, create new chunk
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = Chunk(
                        id=f"{document.id}_chunk_{chunk_index}",
                        content=chunk_text,
                        document_id=document.id,
                        chunk_index=chunk_index,
                        metadata={**document.metadata}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
        
        # Add remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = Chunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    content=chunk_text,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata={**document.metadata}
                )
                chunks.append(chunk)
        
        return chunks

class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking
    
    Uses embeddings to find semantic boundaries
    Groups similar sentences together
    """
    
    def __init__(self, max_chunk_size: int = 512, similarity_threshold: float = 0.7):
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
    
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document using semantic similarity
        
        Note: In practice, would use actual embeddings
        This is a simplified version
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', document.content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # Check if should start new chunk
            # (In practice, compare embeddings with previous sentence)
            should_split = (
                current_size + sentence_size > self.max_chunk_size or
                (current_chunk and i > 0)  # Simplified: split every few sentences
            )
            
            if should_split and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = Chunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    content=chunk_text,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata={**document.metadata}
                )
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1
        
        # Add remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = Chunk(
                id=f"{document.id}_chunk_{chunk_index}",
                content=chunk_text,
                document_id=document.id,
                chunk_index=chunk_index,
                metadata={**document.metadata}
            )
            chunks.append(chunk)
        
        return chunks

# ==================== VECTOR DATABASE (SIMPLIFIED) ====================

class VectorDatabase:
    """
    Simplified vector database
    
    In production: Use Pinecone, Weaviate, Qdrant, etc.
    """
    
    def __init__(self):
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata_index: Dict[str, List[str]] = defaultdict(list)
    
    def add_chunk(self, chunk: Chunk, embedding: np.ndarray):
        """Add chunk with embedding"""
        self.chunks[chunk.id] = chunk
        self.embeddings[chunk.id] = embedding
        
        # Index metadata
        for key, value in chunk.metadata.items():
            self.metadata_index[f"{key}:{value}"].append(chunk.id)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               filters: Optional[Dict] = None) -> List[RetrievalResult]:
        """
        Vector similarity search
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters
        
        Returns:
            List of retrieval results sorted by score
        """
        results = []
        
        for chunk_id, chunk_embedding in self.embeddings.items():
            chunk = self.chunks[chunk_id]
            
            # Apply filters
            if filters:
                if not self._matches_filters(chunk, filters):
                    continue
            
            # Compute similarity (cosine)
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)
            
            results.append(RetrievalResult(
                chunk=chunk,
                score=similarity,
                rank=0  # Will be set after sorting
            ))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Set ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2 + 1e-8)
    
    def _matches_filters(self, chunk: Chunk, filters: Dict) -> bool:
        """Check if chunk matches metadata filters"""
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                return False
        return True

# ==================== RAG SYSTEM ====================

class RAGSystem:
    """
    Complete RAG system
    
    Industry-standard implementation with all components
    """
    
    def __init__(self, 
                 chunking_strategy: ChunkingStrategy,
                 embedding_model=None,  # In practice, use actual model
                 vector_db: Optional[VectorDatabase] = None):
        """
        Args:
            chunking_strategy: Strategy for chunking documents
            embedding_model: Model for generating embeddings
            vector_db: Vector database (creates new if None)
        """
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        self.vector_db = vector_db or VectorDatabase()
        self.documents: Dict[str, Document] = {}
    
    def add_document(self, document: Document):
        """
        Add document to RAG system
        
        Steps:
        1. Chunk document
        2. Generate embeddings
        3. Store in vector database
        """
        self.documents[document.id] = document
        
        # Chunk document
        chunks = self.chunking_strategy.chunk(document)
        
        # Generate embeddings and store
        for chunk in chunks:
            # In practice, use actual embedding model
            # For demo, use random embeddings
            embedding = self._generate_embedding(chunk.content)
            chunk.embedding = embedding
            self.vector_db.add_chunk(chunk, embedding)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text
        
        In practice: Use sentence-transformers, OpenAI, etc.
        """
        # Simplified: random embedding (in practice, use actual model)
        # This is just for demonstration
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(384)  # 384-dim embedding
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        return embedding
    
    def retrieve(self, query: str, top_k: int = 5,
                filters: Optional[Dict] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for query
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filters: Metadata filters
        
        Returns:
            List of retrieval results
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Search vector database
        results = self.vector_db.search(query_embedding, top_k=top_k, filters=filters)
        
        return results
    
    def generate_answer(self, query: str, top_k: int = 5,
                       max_context_length: int = 2000) -> Dict:
        """
        Generate answer using RAG
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            max_context_length: Maximum context length
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        retrieval_results = self.retrieve(query, top_k=top_k)
        
        # Assemble context
        context_chunks = []
        total_length = 0
        
        for result in retrieval_results:
            chunk_text = result.chunk.content
            chunk_length = len(chunk_text)
            
            if total_length + chunk_length > max_context_length:
                break
            
            context_chunks.append({
                "content": chunk_text,
                "source": result.chunk.document_id,
                "score": result.score,
                "metadata": result.chunk.metadata
            })
            total_length += chunk_length
        
        # Format context
        context = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        # Generate answer (simplified - in practice, use LLM API)
        answer = self._generate_with_llm(query, context)
        
        return {
            "answer": answer,
            "sources": [chunk["source"] for chunk in context_chunks],
            "chunks": context_chunks,
            "retrieval_scores": [chunk["score"] for chunk in context_chunks]
        }
    
    def _generate_with_llm(self, query: str, context: str) -> str:
        """
        Generate answer using LLM
        
        In practice: Use OpenAI, Anthropic, self-hosted LLM, etc.
        """
        # Simplified prompt (in practice, use proper prompt engineering)
        prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""
        
        # Simplified generation (in practice, call LLM API)
        # This is just a placeholder
        return f"Based on the provided context, the answer to '{query}' is: [Generated answer would appear here]"

# ==================== ADVANCED: RE-RANKING ====================

class Reranker:
    """
    Re-ranker for improving retrieval accuracy
    
    Uses cross-encoder for better accuracy than bi-encoder
    """
    
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Re-rank retrieval results
        
        Args:
            query: User query
            results: Initial retrieval results
        
        Returns:
            Re-ranked results
        """
        # In practice, use cross-encoder model (e.g., sentence-transformers)
        # This is a simplified version
        
        # Compute re-ranking scores
        for result in results:
            # In practice: score = cross_encoder.predict([query, result.chunk.content])
            # Simplified: use combination of original score and length
            length_score = 1.0 / (1.0 + len(result.chunk.content) / 1000)  # Prefer medium-length
            result.score = 0.7 * result.score + 0.3 * length_score
        
        # Re-sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results

# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    print("RAG System: Industry-Standard Implementation")
    print("=" * 60)
    
    # Create documents
    doc1 = Document(
        id="doc1",
        content="Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed. Deep learning uses neural networks with multiple layers.",
        metadata={"source": "ml_textbook", "page": 1}
    )
    
    doc2 = Document(
        id="doc2",
        content="Natural language processing (NLP) is a field of AI that focuses on understanding human language. Transformers are a type of neural network architecture used in modern NLP models like BERT and GPT.",
        metadata={"source": "nlp_textbook", "page": 5}
    )
    
    # Create RAG system with sentence-based chunking
    chunker = SentenceBasedChunker(max_chunk_size=200, min_chunk_size=50)
    rag = RAGSystem(chunking_strategy=chunker)
    
    # Add documents
    print("Adding documents...")
    rag.add_document(doc1)
    rag.add_document(doc2)
    print(f"Added {len(rag.documents)} documents")
    print()
    
    # Query
    query = "What is machine learning?"
    print(f"Query: {query}")
    print()
    
    # Retrieve
    print("Retrieving relevant chunks...")
    results = rag.retrieve(query, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result.score:.4f}):")
        print(f"  Content: {result.chunk.content[:100]}...")
        print(f"  Source: {result.chunk.metadata.get('source', 'unknown')}")
    print()
    
    # Generate answer
    print("Generating answer...")
    answer = rag.generate_answer(query, top_k=3)
    print(f"\nAnswer: {answer['answer']}")
    print(f"\nSources: {answer['sources']}")
    print(f"Retrieval scores: {[f'{s:.4f}' for s in answer['retrieval_scores']]}")

