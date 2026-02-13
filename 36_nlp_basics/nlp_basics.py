"""
NLP Basics: TF-IDF, N-grams, Laplace Smoothing
Complete implementations with detailed explanations

Note: For L1/L2 priors (Bayesian interpretation of regularization),
see Topic 37: MLE and MAP Estimation
"""
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import math

# ==================== TF-IDF ====================

def term_frequency(term: str, document: List[str]) -> float:
    """
    Term Frequency: TF(t, d) = count(t, d) / |d|
    
    Measures how often a term appears in a document
    Normalized by document length
    """
    if len(document) == 0:
        return 0.0
    count = document.count(term)
    return count / len(document)

def inverse_document_frequency(term: str, documents: List[List[str]]) -> float:
    """
    Inverse Document Frequency: IDF(t, D) = log(N / |{d : t ∈ d}|)
    
    Measures how rare a term is across documents
    Common words (appear in many docs) → low IDF
    Rare words (appear in few docs) → high IDF
    """
    N = len(documents)
    docs_with_term = sum(1 for doc in documents if term in doc)
    
    if docs_with_term == 0:
        return 0.0  # Term never appears
    
    return math.log(N / docs_with_term)

def tf_idf(term: str, document: List[str], documents: List[List[str]]) -> float:
    """
    TF-IDF: TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
    
    Combines term frequency (importance in document) with
    inverse document frequency (rarity across corpus)
    
    High TF-IDF: Term appears often in this document (high TF)
                 but rarely in other documents (high IDF)
    → Characteristic word for this document
    """
    tf = term_frequency(term, document)
    idf = inverse_document_frequency(term, documents)
    return tf * idf

def compute_tfidf_matrix(documents: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute TF-IDF matrix for all documents
    
    Returns:
    - Matrix: (n_documents, n_terms) TF-IDF scores
    - Vocabulary: List of all unique terms
    """
    # Get vocabulary (all unique terms)
    vocabulary = list(set(term for doc in documents for term in doc))
    vocabulary.sort()
    
    # Compute TF-IDF for each document-term pair
    n_docs = len(documents)
    n_terms = len(vocabulary)
    tfidf_matrix = np.zeros((n_docs, n_terms))
    
    for i, doc in enumerate(documents):
        for j, term in enumerate(vocabulary):
            tfidf_matrix[i, j] = tf_idf(term, doc, documents)
    
    return tfidf_matrix, vocabulary

def tfidf_example():
    """
    TF-IDF example with detailed explanation
    """
    print("TF-IDF Example")
    print("=" * 60)
    
    # Sample documents
    doc1 = ["machine", "learning", "algorithm"]
    doc2 = ["deep", "learning", "model"]
    doc3 = ["machine", "learning", "is", "great"]
    
    documents = [doc1, doc2, doc3]
    
    print("Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  Doc {i}: {doc}")
    print()
    
    # Compute TF-IDF for specific terms
    terms_to_check = ["machine", "learning", "algorithm", "deep"]
    
    print("TF-IDF Scores:")
    print(f"{'Term':<12} {'Doc1':<12} {'Doc2':<12} {'Doc3':<12}")
    print("-" * 50)
    
    for term in terms_to_check:
        scores = [tf_idf(term, doc, documents) for doc in documents]
        print(f"{term:<12} {scores[0]:<12.4f} {scores[1]:<12.4f} {scores[2]:<12.4f}")
    
    print("\nInterpretation:")
    print("  - 'algorithm': High in Doc1 (only appears there) → High TF-IDF")
    print("  - 'learning': Appears in all docs → Lower IDF → Lower TF-IDF")
    print("  - 'machine': Appears in Doc1 and Doc3 → Medium TF-IDF")
    print("  - 'deep': Only in Doc2 → High TF-IDF in Doc2")

# ==================== N-GRAMS ====================

def create_ngrams(text: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Create n-grams from text
    
    Args:
        text: List of words
        n: N-gram size (1=unigram, 2=bigram, 3=trigram)
    
    Returns:
        List of n-gram tuples
    """
    if len(text) < n:
        return []
    
    ngrams = []
    for i in range(len(text) - n + 1):
        ngram = tuple(text[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

def build_ngram_model(documents: List[List[str]], n: int = 2) -> Dict:
    """
    Build n-gram language model
    
    For bigram (n=2):
    P(wᵢ|wᵢ₋₁) = count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)
    
    Returns:
    - ngram_counts: Count of each n-gram
    - context_counts: Count of each context (n-1 gram)
    - vocabulary: All unique words
    """
    ngram_counts = Counter()
    context_counts = Counter()
    vocabulary = set()
    
    for doc in documents:
        # Add start/end markers
        if n > 1:
            doc = ['<START>'] * (n-1) + doc + ['<END>']
        
        # Create n-grams
        ngrams = create_ngrams(doc, n)
        
        for ngram in ngrams:
            ngram_counts[ngram] += 1
            vocabulary.update(ngram)
            
            # Context is first n-1 words
            if n > 1:
                context = ngram[:-1]
                context_counts[context] += 1
    
    return {
        'ngram_counts': ngram_counts,
        'context_counts': context_counts,
        'vocabulary': vocabulary,
        'n': n
    }

def ngram_probability(ngram: Tuple[str, ...], model: Dict, 
                     smoothing: bool = False, k: float = 1.0) -> float:
    """
    Compute n-gram probability
    
    Without smoothing:
    P(wᵢ|wᵢ₋₁) = count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)
    
    With Laplace smoothing (add-k):
    P(wᵢ|wᵢ₋₁) = (count(wᵢ₋₁, wᵢ) + k) / (count(wᵢ₋₁) + k*V)
    """
    ngram_counts = model['ngram_counts']
    context_counts = model['context_counts']
    vocabulary = model['vocabulary']
    n = model['n']
    
    if n == 1:
        # Unigram: P(w) = count(w) / total
        total = sum(ngram_counts.values())
        count = ngram_counts[ngram]
        
        if smoothing:
            V = len(vocabulary)
            return (count + k) / (total + k * V)
        else:
            return count / total if total > 0 else 0.0
    
    else:
        # N-gram: P(wᵢ|context) = count(context, wᵢ) / count(context)
        context = ngram[:-1]
        count = ngram_counts[ngram]
        context_count = context_counts[context]
        
        if smoothing:
            V = len(vocabulary)
            return (count + k) / (context_count + k * V)
        else:
            return count / context_count if context_count > 0 else 0.0

def ngram_example():
    """
    N-gram model example
    """
    print("\nN-gram Model Example")
    print("=" * 60)
    
    # Sample documents
    documents = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "mat"],
        ["the", "cat", "and", "the", "dog"]
    ]
    
    print("Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  Doc {i}: {' '.join(doc)}")
    print()
    
    # Build bigram model
    model = build_ngram_model(documents, n=2)
    
    # Test probabilities
    test_bigrams = [
        ("the", "cat"),
        ("the", "dog"),
        ("cat", "sat"),
        ("the", "bird"),  # Unseen
    ]
    
    print("Bigram Probabilities (without smoothing):")
    for bigram in test_bigrams:
        prob = ngram_probability(bigram, model, smoothing=False)
        print(f"  P({bigram[1]}|{bigram[0]}) = {prob:.4f}")
    
    print("\nBigram Probabilities (with Laplace smoothing, k=1):")
    for bigram in test_bigrams:
        prob = ngram_probability(bigram, model, smoothing=True, k=1.0)
        print(f"  P({bigram[1]}|{bigram[0]}) = {prob:.4f}")
    
    print("\nKey Observation:")
    print("  - Without smoothing: P(bird|the) = 0 (unseen bigram)")
    print("  - With smoothing: P(bird|the) > 0 (can handle unseen)")

# ==================== LAPLACE SMOOTHING ====================

def laplace_smoothing_explanation():
    """
    Detailed explanation of Laplace smoothing
    """
    print("\nLaplace Smoothing: Detailed Explanation")
    print("=" * 60)
    
    print("""
Problem: Zero Probability Problem

Without smoothing:
  - If n-gram never seen: P = 0
  - Product of probabilities becomes 0
  - Model can't handle unseen n-grams

Example:
  Training: "the cat", "the dog"
  Test: "the bird" (unseen)
  
  Without smoothing:
    P(bird|the) = 0/2 = 0  (problem!)
  
  With add-1 smoothing:
    P(bird|the) = (0+1)/(2+3) = 1/5 = 0.2  (fixed!)

How it works:
  - Add k to each count
  - Add k*V to denominator (V = vocabulary size)
  - Redistributes probability from seen to unseen

Effect:
  - Seen n-grams: Slightly lower probability
  - Unseen n-grams: Non-zero probability
  - Prevents zeros, allows generalization
    """)

def compare_with_without_smoothing():
    """
    Compare probabilities with and without smoothing
    """
    print("\nSmoothing Comparison")
    print("=" * 60)
    
    # Simple example
    training = ["the", "cat", "the", "dog", "the", "cat"]
    
    # Count bigrams
    bigrams = []
    for i in range(len(training) - 1):
        bigrams.append((training[i], training[i+1]))
    
    bigram_counts = Counter(bigrams)
    context_counts = Counter(training[:-1])
    
    print("Training data:", training)
    print(f"Bigram counts: {dict(bigram_counts)}")
    print(f"Context 'the' appears: {context_counts['the']} times")
    print()
    
    # Test bigrams
    test = [("the", "cat"), ("the", "dog"), ("the", "bird")]
    V = len(set(training))  # Vocabulary size
    
    print("Probabilities:")
    print(f"{'Bigram':<15} {'Without Smoothing':<20} {'With Smoothing (k=1)':<20}")
    print("-" * 60)
    
    for bigram in test:
        count = bigram_counts[bigram]
        context_count = context_counts[bigram[0]]
        
        # Without smoothing
        prob_no_smooth = count / context_count if context_count > 0 else 0.0
        
        # With smoothing
        prob_smooth = (count + 1) / (context_count + V)
        
        print(f"{str(bigram):<15} {prob_no_smooth:<20.4f} {prob_smooth:<20.4f}")
    
    print("\nKey Insight:")
    print("  - 'bird' has 0 probability without smoothing (can't handle unseen)")
    print("  - 'bird' has positive probability with smoothing (can generalize)")

# Note: L1/L2 priors (Bayesian interpretation of regularization) 
# are covered in Topic 37: MLE and MAP Estimation

# ==================== USAGE ====================

if __name__ == "__main__":
    print("NLP Basics: TF-IDF, N-grams, Smoothing, Regularization")
    print("=" * 60)
    
    # TF-IDF
    tfidf_example()
    
    # N-grams
    ngram_example()
    
    # Smoothing
    laplace_smoothing_explanation()
    compare_with_without_smoothing()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - TF-IDF: Measures word importance (TF × IDF)")
    print("  - N-grams: Model language with n-word sequences")
    print("  - Laplace Smoothing: Handles unseen n-grams (add-k)")
    print("\nNote: For L1/L2 priors (Bayesian interpretation of regularization),")
    print("      see Topic 37: MLE and MAP Estimation")

