"""
Retrieval Methods: BM25, TF-IDF, Dense, Hybrid
Complete implementations with industry-standard code
"""
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import math
from dataclasses import dataclass

# ==================== DATA STRUCTURES ====================

@dataclass
class RetrievalResult:
    """Retrieval result with score"""
    doc_id: str
    score: float
    rank: int

# ==================== TF-IDF RETRIEVAL ====================

class TFIDFRetriever:
    """
    TF-IDF based retrieval
    
    Use Case: Keyword-based queries, exact term matching
    """
    
    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.vocabulary: set = set()
        self.tf_idf_matrix: Dict[str, Dict[str, float]] = {}
        self.idf_scores: Dict[str, float] = {}
    
    def add_document(self, doc_id: str, text: str):
        """Add document to index"""
        self.documents[doc_id] = text
        words = self._tokenize(text)
        self.vocabulary.update(words)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (in practice, use proper tokenizer)"""
        return text.lower().split()
    
    def _compute_tf(self, term: str, doc_id: str) -> float:
        """Term frequency"""
        text = self.documents[doc_id]
        words = self._tokenize(text)
        if len(words) == 0:
            return 0.0
        return words.count(term.lower()) / len(words)
    
    def _compute_idf(self, term: str) -> float:
        """Inverse document frequency"""
        N = len(self.documents)
        if N == 0:
            return 0.0
        
        docs_with_term = sum(1 for doc_id, text in self.documents.items()
                           if term.lower() in self._tokenize(text))
        
        if docs_with_term == 0:
            return 0.0
        
        return math.log(N / docs_with_term)
    
    def build_index(self):
        """Build TF-IDF index"""
        # Compute IDF for all terms
        for term in self.vocabulary:
            self.idf_scores[term] = self._compute_idf(term)
        
        # Compute TF-IDF for all documents
        for doc_id in self.documents:
            self.tf_idf_matrix[doc_id] = {}
            words = self._tokenize(self.documents[doc_id])
            word_counts = Counter(words)
            total_words = len(words)
            
            for term in self.vocabulary:
                tf = word_counts.get(term.lower(), 0) / total_words if total_words > 0 else 0
                idf = self.idf_scores[term]
                self.tf_idf_matrix[doc_id][term] = tf * idf
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Search using TF-IDF"""
        query_terms = self._tokenize(query)
        
        scores = {}
        for doc_id in self.documents:
            score = 0.0
            for term in query_terms:
                if term in self.tf_idf_matrix[doc_id]:
                    score += self.tf_idf_matrix[doc_id][term]
            scores[doc_id] = score
        
        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], 1):
            results.append(RetrievalResult(doc_id=doc_id, score=score, rank=rank))
        
        return results

# ==================== BM25 RETRIEVAL ====================

class BM25Retriever:
    """
    BM25 (Best Matching 25) retrieval
    
    Industry standard for sparse retrieval
    Better than TF-IDF with term frequency saturation and length normalization
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter (usually 1.2-2.0)
            b: Length normalization parameter (usually 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.term_freqs: Dict[str, Dict[str, int]] = {}  # doc_id -> term -> count
        self.df: Dict[str, int] = {}  # term -> document frequency
        self.N: int = 0  # Total documents
    
    def add_document(self, doc_id: str, text: str):
        """Add document to index"""
        self.documents[doc_id] = text
        words = self._tokenize(text)
        self.doc_lengths[doc_id] = len(words)
        
        # Count term frequencies
        self.term_freqs[doc_id] = Counter(words)
        
        # Update document frequency
        for term in set(words):
            self.df[term] = self.df.get(term, 0) + 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()
    
    def build_index(self):
        """Build BM25 index"""
        self.N = len(self.documents)
        
        if self.N == 0:
            return
        
        # Compute average document length
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.N
    
    def _compute_idf(self, term: str) -> float:
        """
        BM25 IDF formula
        
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
        """
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        
        return math.log((self.N - df + 0.5) / (df + 0.5))
    
    def _compute_bm25_term(self, term: str, doc_id: str) -> float:
        """
        BM25 score for a single term in a document
        
        BM25(t, d) = IDF(t) × (f(t, d) × (k₁ + 1)) / (f(t, d) + k₁ × (1 - b + b × |d|/avgdl))
        """
        # Term frequency in document
        f_td = self.term_freqs[doc_id].get(term, 0)
        
        if f_td == 0:
            return 0.0
        
        # Document length
        doc_length = self.doc_lengths[doc_id]
        
        # IDF
        idf = self._compute_idf(term)
        
        # BM25 formula
        numerator = f_td * (self.k1 + 1)
        denominator = f_td + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
        
        bm25_score = idf * (numerator / denominator)
        
        return bm25_score
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Search using BM25
        
        Score(query, doc) = Σ BM25(term, doc) for term in query
        """
        query_terms = self._tokenize(query)
        
        scores = {}
        for doc_id in self.documents:
            score = 0.0
            for term in query_terms:
                score += self._compute_bm25_term(term, doc_id)
            scores[doc_id] = score
        
        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], 1):
            results.append(RetrievalResult(doc_id=doc_id, score=score, rank=rank))
        
        return results

# ==================== DENSE RETRIEVAL ====================

class DenseRetriever:
    """
    Dense retrieval using embeddings
    
    Use Case: Semantic similarity, related concepts
    """
    
    def __init__(self, embedding_fn=None):
        """
        Args:
            embedding_fn: Function to generate embeddings (in practice, use model)
        """
        self.embedding_fn = embedding_fn or self._dummy_embedding
        self.documents: Dict[str, str] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def _dummy_embedding(self, text: str) -> np.ndarray:
        """Dummy embedding (in practice, use actual model)"""
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(384)
        return embedding / np.linalg.norm(embedding)
    
    def add_document(self, doc_id: str, text: str):
        """Add document and generate embedding"""
        self.documents[doc_id] = text
        self.embeddings[doc_id] = self.embedding_fn(text)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Search using dense embeddings"""
        query_embedding = self.embedding_fn(query)
        
        scores = {}
        for doc_id, doc_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            scores[doc_id] = similarity
        
        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], 1):
            results.append(RetrievalResult(doc_id=doc_id, score=score, rank=rank))
        
        return results

# ==================== HYBRID RETRIEVAL ====================

class HybridRetriever:
    """
    Hybrid retrieval: BM25 + Dense
    
    Combines sparse (BM25) and dense (embeddings) retrieval
    Industry standard for production RAG systems
    """
    
    def __init__(self, bm25_retriever: BM25Retriever, 
                 dense_retriever: DenseRetriever,
                 alpha: float = 0.5):
        """
        Args:
            bm25_retriever: BM25 retriever instance
            dense_retriever: Dense retriever instance
            alpha: Weight for BM25 (1-alpha for dense)
                  alpha=0.5: Equal weight
                  alpha>0.5: More weight to BM25
                  alpha<0.5: More weight to dense
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.alpha = alpha
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Normalize scores to [0, 1] using min-max"""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores same, return as is
            return results
        
        normalized = []
        for result in results:
            normalized_score = (result.score - min_score) / (max_score - min_score)
            normalized.append(RetrievalResult(
                doc_id=result.doc_id,
                score=normalized_score,
                rank=result.rank
            ))
        
        return normalized
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Hybrid search: Combine BM25 and dense retrieval
        
        Steps:
        1. Retrieve from both methods
        2. Normalize scores
        3. Combine with weighted sum
        4. Re-rank and return top-K
        """
        # Retrieve from both methods
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)  # Get more for combination
        dense_results = self.dense_retriever.search(query, top_k=top_k * 2)
        
        # Normalize scores
        bm25_results = self._normalize_scores(bm25_results)
        dense_results = self._normalize_scores(dense_results)
        
        # Combine scores
        combined_scores = {}
        
        # Add BM25 scores
        for result in bm25_results:
            combined_scores[result.doc_id] = self.alpha * result.score
        
        # Add dense scores
        for result in dense_results:
            if result.doc_id in combined_scores:
                combined_scores[result.doc_id] += (1 - self.alpha) * result.score
            else:
                combined_scores[result.doc_id] = (1 - self.alpha) * result.score
        
        # Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-K
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], 1):
            results.append(RetrievalResult(doc_id=doc_id, score=score, rank=rank))
        
        return results

# ==================== USAGE EXAMPLES ====================

def compare_retrieval_methods():
    """Compare different retrieval methods"""
    print("Retrieval Methods Comparison")
    print("=" * 60)
    
    # Sample documents
    documents = {
        "doc1": "Machine learning is a subset of artificial intelligence",
        "doc2": "Deep learning uses neural networks with multiple layers",
        "doc3": "Natural language processing focuses on understanding human language",
        "doc4": "Computer vision enables machines to interpret visual information",
    }
    
    query = "artificial intelligence and neural networks"
    
    print(f"Query: {query}\n")
    
    # TF-IDF
    print("1. TF-IDF Retrieval:")
    tfidf = TFIDFRetriever()
    for doc_id, text in documents.items():
        tfidf.add_document(doc_id, text)
    tfidf.build_index()
    tfidf_results = tfidf.search(query, top_k=3)
    for result in tfidf_results:
        print(f"   {result.doc_id}: {result.score:.4f}")
    print()
    
    # BM25
    print("2. BM25 Retrieval:")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    for doc_id, text in documents.items():
        bm25.add_document(doc_id, text)
    bm25.build_index()
    bm25_results = bm25.search(query, top_k=3)
    for result in bm25_results:
        print(f"   {result.doc_id}: {result.score:.4f}")
    print()
    
    # Dense
    print("3. Dense Retrieval:")
    dense = DenseRetriever()
    for doc_id, text in documents.items():
        dense.add_document(doc_id, text)
    dense_results = dense.search(query, top_k=3)
    for result in dense_results:
        print(f"   {result.doc_id}: {result.score:.4f}")
    print()
    
    # Hybrid
    print("4. Hybrid Retrieval (BM25 + Dense, α=0.5):")
    hybrid = HybridRetriever(bm25, dense, alpha=0.5)
    hybrid_results = hybrid.search(query, top_k=3)
    for result in hybrid_results:
        print(f"   {result.doc_id}: {result.score:.4f}")
    print()
    
    print("Key Observations:")
    print("  - TF-IDF: Simple, keyword-based")
    print("  - BM25: Better than TF-IDF (saturation, normalization)")
    print("  - Dense: Semantic similarity")
    print("  - Hybrid: Combines both for best results")

def bm25_parameter_tuning():
    """Demonstrate BM25 parameter effects"""
    print("\nBM25 Parameter Tuning")
    print("=" * 60)
    
    documents = {
        "doc1": "machine learning machine learning machine learning",
        "doc2": "artificial intelligence",
        "doc3": "deep learning neural networks",
    }
    
    query = "machine learning"
    
    print(f"Query: {query}\n")
    
    # Different k1 values
    print("Effect of k1 (term frequency saturation):")
    for k1 in [0.5, 1.5, 3.0]:
        bm25 = BM25Retriever(k1=k1, b=0.75)
        for doc_id, text in documents.items():
            bm25.add_document(doc_id, text)
        bm25.build_index()
        results = bm25.search(query, top_k=1)
        print(f"  k1={k1}: doc1 score = {results[0].score:.4f}")
    print()
    
    # Different b values
    print("Effect of b (length normalization):")
    for b in [0.0, 0.75, 1.0]:
        bm25 = BM25Retriever(k1=1.5, b=b)
        for doc_id, text in documents.items():
            bm25.add_document(doc_id, text)
        bm25.build_index()
        results = bm25.search(query, top_k=1)
        print(f"  b={b}: doc1 score = {results[0].score:.4f}")
    print()
    
    print("Interpretation:")
    print("  - Higher k1: More weight to term frequency")
    print("  - Higher b: More length normalization")

if __name__ == "__main__":
    print("Retrieval Methods: BM25, TF-IDF, Dense, Hybrid")
    print("=" * 60)
    
    # Comparison
    compare_retrieval_methods()
    
    # Parameter tuning
    bm25_parameter_tuning()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - TF-IDF: Simple, interpretable, keyword-based")
    print("  - BM25: Industry standard, better than TF-IDF")
    print("  - Dense: Semantic understanding, embeddings")
    print("  - Hybrid: Best of both (BM25 + Dense)")
    print("\nRecommendation:")
    print("  - Start with BM25")
    print("  - Add dense if semantic understanding needed")
    print("  - Use hybrid for production systems")

