"""
RAG Evaluation: Industry-Standard Metrics
Complete evaluation framework for RAG systems
"""
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ==================== EVALUATION METRICS ====================

@dataclass
class EvaluationResult:
    """Evaluation result with metrics"""
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    end_to_end_metrics: Dict[str, float]

class RAGEvaluator:
    """
    RAG System Evaluator
    
    Evaluates retrieval, generation, and end-to-end performance
    """
    
    def evaluate_retrieval(self, 
                          retrieved_chunks: List[str],
                          relevant_chunks: List[str],
                          k: int = 10) -> Dict[str, float]:
        """
        Evaluate retrieval performance
        
        Args:
            retrieved_chunks: List of retrieved chunk IDs
            relevant_chunks: List of relevant chunk IDs (ground truth)
            k: Top-K for metrics
        
        Returns:
            Dictionary of retrieval metrics
        """
        # Convert to sets for easier computation
        retrieved_set = set(retrieved_chunks[:k])
        relevant_set = set(relevant_chunks)
        
        # Intersection (correctly retrieved)
        intersection = retrieved_set & relevant_set
        
        # Precision@K
        precision_at_k = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
        
        # Recall@K
        recall_at_k = len(intersection) / len(relevant_set) if relevant_set else 0.0
        
        # F1@K
        f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0.0
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, chunk_id in enumerate(retrieved_chunks[:k], 1):
            if chunk_id in relevant_set:
                mrr = 1.0 / i
                break
        
        # Mean Average Precision (MAP)
        map_score = 0.0
        relevant_retrieved = []
        for i, chunk_id in enumerate(retrieved_chunks[:k], 1):
            if chunk_id in relevant_set:
                relevant_retrieved.append(i)
                precision_at_i = len(relevant_retrieved) / i
                map_score += precision_at_i
        
        if relevant_retrieved:
            map_score /= len(relevant_set)
        
        # NDCG@K (simplified - assumes binary relevance)
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved_chunks[:k], 1):
            if chunk_id in relevant_set:
                dcg += 1.0 / np.log2(i + 1)
        
        # Ideal DCG (all relevant at top)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant_set), k) + 1))
        ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
        
        return {
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "f1@k": f1_at_k,
            "mrr": mrr,
            "map": map_score,
            "ndcg@k": ndcg_at_k
        }
    
    def evaluate_generation(self,
                           generated_answer: str,
                           reference_answer: str) -> Dict[str, float]:
        """
        Evaluate generation quality
        
        Args:
            generated_answer: Generated answer
            reference_answer: Reference answer (ground truth)
        
        Returns:
            Dictionary of generation metrics
        """
        # BLEU score (simplified - character n-grams)
        bleu = self._bleu_score(generated_answer, reference_answer)
        
        # ROUGE-L (simplified - longest common subsequence)
        rouge_l = self._rouge_l(generated_answer, reference_answer)
        
        # Semantic similarity (simplified - would use BERTScore in practice)
        semantic_sim = self._semantic_similarity(generated_answer, reference_answer)
        
        # Answer length ratio
        length_ratio = len(generated_answer) / len(reference_answer) if len(reference_answer) > 0 else 0.0
        
        return {
            "bleu": bleu,
            "rouge_l": rouge_l,
            "semantic_similarity": semantic_sim,
            "length_ratio": length_ratio
        }
    
    def evaluate_end_to_end(self,
                            generated_answer: str,
                            reference_answer: str,
                            retrieved_chunks: List[str],
                            relevant_chunks: List[str],
                            context_used: str) -> Dict[str, float]:
        """
        Evaluate end-to-end RAG performance
        
        Args:
            generated_answer: Generated answer
            reference_answer: Reference answer
            retrieved_chunks: Retrieved chunk IDs
            relevant_chunks: Relevant chunk IDs
            context_used: Context used for generation
        
        Returns:
            Dictionary of end-to-end metrics
        """
        # Answer relevance (simplified - would use model in practice)
        answer_relevance = self._answer_relevance(generated_answer, reference_answer)
        
        # Answer correctness (simplified)
        answer_correctness = self._answer_correctness(generated_answer, reference_answer)
        
        # Answer completeness
        answer_completeness = self._answer_completeness(generated_answer, reference_answer)
        
        # Context utilization
        context_utilization = self._context_utilization(generated_answer, context_used)
        
        # Citation quality (simplified)
        citation_quality = self._citation_quality(retrieved_chunks, relevant_chunks)
        
        return {
            "answer_relevance": answer_relevance,
            "answer_correctness": answer_correctness,
            "answer_completeness": answer_completeness,
            "context_utilization": context_utilization,
            "citation_quality": citation_quality
        }
    
    def _bleu_score(self, generated: str, reference: str) -> float:
        """Simplified BLEU score"""
        # In practice, use nltk.translate.bleu_score
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()
        
        # Unigram precision
        gen_counts = {}
        for word in gen_words:
            gen_counts[word] = gen_counts.get(word, 0) + 1
        
        ref_counts = {}
        for word in ref_words:
            ref_counts[word] = ref_counts.get(word, 0) + 1
        
        matches = sum(min(gen_counts.get(w, 0), ref_counts.get(w, 0)) for w in gen_counts)
        precision = matches / len(gen_words) if gen_words else 0.0
        
        return precision  # Simplified - full BLEU uses n-grams and brevity penalty
    
    def _rouge_l(self, generated: str, reference: str) -> float:
        """Simplified ROUGE-L (LCS-based)"""
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()
        
        # LCS length
        lcs_length = self._lcs_length(gen_words, ref_words)
        
        if len(ref_words) == 0:
            return 0.0
        
        recall = lcs_length / len(ref_words)
        precision = lcs_length / len(gen_words) if gen_words else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Simplified semantic similarity"""
        # In practice, use BERTScore, sentence-transformers, etc.
        # This is a placeholder
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _answer_relevance(self, generated: str, reference: str) -> float:
        """Answer relevance score"""
        # Simplified - in practice, use model to judge relevance
        return self._semantic_similarity(generated, reference)
    
    def _answer_correctness(self, generated: str, reference: str) -> float:
        """Answer correctness score"""
        # Simplified - in practice, use model or human evaluation
        return self._semantic_similarity(generated, reference)
    
    def _answer_completeness(self, generated: str, reference: str) -> float:
        """Answer completeness score"""
        # Simplified - check if key information present
        ref_words = set(reference.lower().split())
        gen_words = set(generated.lower().split())
        
        if not ref_words:
            return 0.0
        
        coverage = len(gen_words & ref_words) / len(ref_words)
        return coverage
    
    def _context_utilization(self, answer: str, context: str) -> float:
        """How well answer uses context"""
        # Simplified - check word overlap
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if not answer_words:
            return 0.0
        
        utilization = len(answer_words & context_words) / len(answer_words)
        return utilization
    
    def _citation_quality(self, retrieved: List[str], relevant: List[str]) -> float:
        """Citation quality score"""
        # Simplified - precision of citations
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        if not retrieved_set:
            return 0.0
        
        precision = len(retrieved_set & relevant_set) / len(retrieved_set)
        return precision

# ==================== EVALUATION DATASET ====================

@dataclass
class QAExample:
    """Question-Answer example for evaluation"""
    question: str
    answer: str
    relevant_chunks: List[str]  # Chunk IDs that contain answer
    context: str  # Full context

class RAGEvaluationDataset:
    """Dataset for RAG evaluation"""
    
    def __init__(self):
        self.examples: List[QAExample] = []
    
    def add_example(self, example: QAExample):
        """Add evaluation example"""
        self.examples.append(example)
    
    def evaluate_rag_system(self, rag_system, evaluator: RAGEvaluator) -> EvaluationResult:
        """
        Evaluate RAG system on dataset
        
        Args:
            rag_system: RAG system to evaluate
            evaluator: Evaluator instance
        
        Returns:
            Evaluation result with all metrics
        """
        all_retrieval_metrics = []
        all_generation_metrics = []
        all_e2e_metrics = []
        
        for example in self.examples:
            # Retrieve
            retrieval_results = rag_system.retrieve(example.question, top_k=10)
            retrieved_chunk_ids = [r.chunk.id for r in retrieval_results]
            
            # Generate answer
            answer_result = rag_system.generate_answer(example.question, top_k=5)
            generated_answer = answer_result["answer"]
            context_used = "\n".join([c["content"] for c in answer_result["chunks"]])
            
            # Evaluate retrieval
            retrieval_metrics = evaluator.evaluate_retrieval(
                retrieved_chunk_ids,
                example.relevant_chunks,
                k=10
            )
            all_retrieval_metrics.append(retrieval_metrics)
            
            # Evaluate generation
            generation_metrics = evaluator.evaluate_generation(
                generated_answer,
                example.answer
            )
            all_generation_metrics.append(generation_metrics)
            
            # Evaluate end-to-end
            e2e_metrics = evaluator.evaluate_end_to_end(
                generated_answer,
                example.answer,
                retrieved_chunk_ids,
                example.relevant_chunks,
                context_used
            )
            all_e2e_metrics.append(e2e_metrics)
        
        # Average metrics
        avg_retrieval = {
            key: np.mean([m[key] for m in all_retrieval_metrics])
            for key in all_retrieval_metrics[0].keys()
        }
        
        avg_generation = {
            key: np.mean([m[key] for m in all_generation_metrics])
            for key in all_generation_metrics[0].keys()
        }
        
        avg_e2e = {
            key: np.mean([m[key] for m in all_e2e_metrics])
            for key in all_e2e_metrics[0].keys()
        }
        
        return EvaluationResult(
            retrieval_metrics=avg_retrieval,
            generation_metrics=avg_generation,
            end_to_end_metrics=avg_e2e
        )

# ==================== USAGE ====================

if __name__ == "__main__":
    print("RAG Evaluation: Industry-Standard Metrics")
    print("=" * 60)
    
    # Create evaluator
    evaluator = RAGEvaluator()
    
    # Example retrieval evaluation
    retrieved = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    relevant = ["chunk2", "chunk4", "chunk6"]
    
    retrieval_metrics = evaluator.evaluate_retrieval(retrieved, relevant, k=5)
    print("Retrieval Metrics:")
    for metric, value in retrieval_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Example generation evaluation
    generated = "Machine learning is a subset of AI that enables computers to learn from data."
    reference = "Machine learning is a method of data analysis that automates analytical model building."
    
    generation_metrics = evaluator.evaluate_generation(generated, reference)
    print("Generation Metrics:")
    for metric, value in generation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    print("Key Metrics:")
    print("  - Retrieval: Precision@K, Recall@K, MRR, MAP, NDCG@K")
    print("  - Generation: BLEU, ROUGE-L, Semantic Similarity")
    print("  - End-to-End: Answer Relevance, Correctness, Completeness")

