# ML & LLM Learning: Coding Interview Preparation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A comprehensive repository for ML and LLM coding interview preparation with implementations, theory, and interview Q&A.

## 🎯 What You'll Learn

This repository covers everything you need for ML/LLM coding interviews:

- **Classical ML Algorithms** - Simple implementations (pure Python/NumPy + PyTorch)
- **Evaluation Metrics** - All common metrics with simple code
- **Transformers & Attention** - Core concepts with simple implementations
- **LLM Inference Techniques** - KV cache, quantization (simple code)
- **Attention Mechanisms** - Different types with clear code
- **LLM Problem Solving** - Long context, efficiency solutions
- **Training Techniques** - RLHF, DPO (simplified implementations)
- **Sampling Techniques** - Top-p, nucleus, temperature (pure Python)
- **Optimizers** - SGD, Adam, etc. (from scratch)
- **Regularization** - L1, L2, dropout (simple implementations)
- **Theory & Interview Q&A** - Comprehensive coverage

## 💡 Code Philosophy

**All code is kept simple:**
- **Pure Python/NumPy versions** - No heavy dependencies, easy to understand
- **PyTorch versions** - Simple PyTorch implementations for comparison
- **From scratch** - Understand how things work internally
- **Interview-ready** - Code you can write in interviews

## 📁 Repository Structure

```
ml_and_llm_learning/
├── 00_pytorch_fundamentals/      # PyTorch basics (START HERE if new to PyTorch)
├── 01_classical_ml/              # Linear/logistic regression, KNN, K-means
├── 02_gradient_descent/          # Different GD variants
├── 03_evaluation_metrics/        # All evaluation metrics
├── 04_transformers/              # Transformer architecture
├── 05_attention_mechanisms/      # Different attention types
├── 06_llm_inference/             # KV cache, optimization
├── 07_llm_problems/              # Long context, efficiency
├── 08_training_techniques/       # RLHF, DPO, PPO, GRPO
├── 09_sampling_techniques/       # Top-p, nucleus, temperature
├── 10_optimizers/                # Optimizer implementations
├── 11_regularization/            # Regularization techniques
├── 12_theory/                    # Comprehensive theory
└── 13_interview_qa/              # Interview questions & answers
```

## 🚀 Quick Start

### 1. Set Up Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Learn PyTorch Basics (If Needed)
```bash
# If you're new to PyTorch, start here
cd 00_pytorch_fundamentals
python pytorch_basics.py
```

### 3. Start Learning
```bash
# Classical ML
cd 01_classical_ml
python linear_regression.py

# Transformers
cd 04_transformers
python attention.py

# LLM Inference
cd 06_llm_inference
python kv_cache.py
```

## 📚 Learning Path

See `LEARNING_PATH.md` for the complete learning journey.

## 🎓 Prerequisites

- Python 3.9+
- Basic Python knowledge
- Understanding of linear algebra
- (Optional) PyTorch/TensorFlow experience

## 🔧 Technologies

- **NumPy**: Numerical computations (pure Python/NumPy implementations)
- **PyTorch**: Simple PyTorch versions (optional, for comparison)
- **Matplotlib**: Visualization (optional)
- **Pure Python**: Most code uses only NumPy (minimal dependencies)

## 📖 Topics Covered

1. **Classical ML** - Linear/logistic regression, KNN, K-means
2. **Gradient Descent** - Batch, SGD, Mini-batch, Adam
3. **Evaluation Metrics** - Accuracy, precision, recall, F1, etc.
4. **Transformers** - Architecture, attention, decoding
5. **Attention Mechanisms** - Self-attention, cross-attention, etc.
6. **LLM Inference** - KV cache, quantization, optimization
7. **LLM Problems** - Long context, efficiency solutions
8. **Training Techniques** - RLHF, DPO, PPO, GRPO
9. **Sampling Techniques** - Top-p, nucleus, temperature
10. **Optimizers** - SGD, Adam, AdamW, etc.
11. **Regularization** - L1, L2, dropout, etc.
12. **Theory** - Comprehensive ML/LLM theory
13. **Interview Q&A** - 100+ interview questions
14. **Advanced Positional Embeddings** - RoPE, ALiBi
15. **Tokenization** - BPE, WordPiece, SentencePiece
16. **Training Behaviors** - Single GPU, loss spikes
17. **Probability Math** - Common probability Q&A
18. **Distribution Classification** - Which distribution?
19. **Advanced Clustering** - Hierarchical, DBSCAN, GMM
20. **Multi-Turn Conversations** - Design & long context
21. **Dimensionality Reduction** - PCA, theory & math
22. **Recommendation Systems** - Matrix factorization, evaluation
23. **Clustering Evaluation** - Silhouette, ARI, NMI
24. **Linear Algebra Q&A** - Eigenvalues, SVD, rank
25. **Adapters & LoRA** - Parameter-efficient fine-tuning
26. **Tree-Based Methods** - Decision Tree, Random Forest, Gradient Boosting, XGBoost
27. **Advanced Theory** - Bias-variance, cross-validation, learning curves
28. **Business Use Cases** - Churn, recommendations, fraud, pricing (detailed solutions)
29. **System Design for ML** - Scalable pipelines, serving, monitoring
30. **A/B Testing** - Statistical testing, sample size, interpretation
31. **Neural Networks** - Forward pass, backpropagation from scratch (detailed)
32. **Anomaly Detection** - Isolation Forest (detailed explanation, when to use)
33. **Information Theory** - Entropy, KL divergence, cross-entropy, mutual information, Gini
34. **Discriminative vs Generative** - Model types, assumptions (Linear, Logistic, SVM), Bayes' theorem
35. **Kernel Functions** - Linear, Polynomial, RBF, Sigmoid (detailed explanations, when to use)
36. **NLP Basics** - TF-IDF, N-grams, Laplace smoothing, L1/L2 priors (detailed explanations)
37. **MLE and MAP Estimation** - Maximum Likelihood, Maximum A Posteriori (detailed derivations)
38. **Multimodal Models & Embedding History** - CLIP, embedding training, NLP evolution (TF-IDF → Word2Vec → GloVe → BERT)
39. **RAG (Retrieval-Augmented Generation)** - Industry-standard architecture, challenges, solutions, evaluation (production-ready)

## 🎯 Prerequisites

**New to PyTorch?** Start with `00_pytorch_fundamentals/` to learn all PyTorch concepts you'll need for this repository.

## 🎯 Learning Goals

By completing this repository, you'll be able to:

- ✅ Implement ML algorithms from scratch
- ✅ Understand transformer architecture deeply
- ✅ Optimize LLM inference
- ✅ Answer interview questions confidently
- ✅ Understand training techniques (RLHF, DPO, etc.)
- ✅ Implement evaluation metrics
- ✅ Understand theory behind ML/LLM

---

**Ready to start?** Open `LEARNING_PATH.md` and begin your journey! 🚀

