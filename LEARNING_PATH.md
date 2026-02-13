# ML & LLM Learning: Complete Learning Path

## 🎯 How to Use This Guide

This guide is organized by **learning topics** for coding interview preparation. Work through each topic at your own pace. Each topic includes implementations, theory, and interview questions.

## 📚 Learning Topics (In Order)

### 0. **PyTorch Fundamentals** (START HERE if new to PyTorch)
**What you'll learn**: Essential PyTorch concepts
- Tensors and operations
- Autograd (automatic differentiation)
- Neural network layers
- Loss functions and optimizers
- Training loops
- Device management (CPU/GPU)
- Data loading

**Practice**: `00_pytorch_fundamentals/`

**When to use**: Reference this whenever you need PyTorch syntax or concepts.

### 1. **Classical ML Algorithms**
**What you'll learn**: Implement ML algorithms from scratch
- Linear regression
- Logistic regression
- K-Nearest Neighbors (KNN)
- K-Means clustering
- Theory and intuition

**Practice**: `01_classical_ml/`

### 2. **Gradient Descent Variants**
**What you'll learn**: Different optimization algorithms
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Momentum
- Adam optimizer
- Theory and trade-offs

**Practice**: `02_gradient_descent/`

### 3. **Evaluation Metrics**
**What you'll learn**: All common evaluation metrics
- Classification: Accuracy, Precision, Recall, F1
- Regression: MSE, MAE, R²
- Ranking: NDCG, MAP
- Implementation from scratch

**Practice**: `03_evaluation_metrics/`

### 4. **Transformers**
**What you'll learn**: Transformer architecture
- Self-attention mechanism
- Multi-head attention
- Position encoding
- Encoder-decoder architecture
- Decoding strategies

**Practice**: `04_transformers/`

### 5. **Attention Mechanisms**
**What you'll learn**: Different attention types
- Self-attention
- Cross-attention
- Scaled dot-product attention
- Sparse attention
- Longformer, BigBird attention
- What problems they solve

**Practice**: `05_attention_mechanisms/`

### 6. **LLM Inference Techniques**
**What you'll learn**: Optimizing LLM inference
- KV caching
- Quantization (INT8, INT4)
- Speculative decoding
- Continuous batching
- Memory optimization

**Practice**: `06_llm_inference/`

### 7. **LLM Problem Solving**
**What you'll learn**: Solving LLM challenges
- Long context length solutions
- Efficiency optimization
- Memory management
- Speed optimization
- Detailed explanations

**Practice**: `07_llm_problems/`

### 8. **Training Techniques**
**What you'll learn**: Advanced training methods
- RLHF (Reinforcement Learning from Human Feedback)
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Policy Optimization)
- Theory and implementations

**Practice**: `08_training_techniques/`

### 9. **Sampling Techniques**
**What you'll learn**: Text generation sampling
- Greedy decoding
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature sampling
- Beam search
- Implementations

**Practice**: `09_sampling_techniques/`

### 10. **Optimizers**
**What you'll learn**: Optimization algorithms
- SGD
- Adam
- AdamW
- RMSprop
- Theory and implementations

**Practice**: `10_optimizers/`

### 11. **Regularization**
**What you'll learn**: Preventing overfitting
- L1 regularization (Lasso)
- L2 regularization (Ridge)
- Dropout
- Early stopping
- Theory and implementations

**Practice**: `11_regularization/`

### 12. **Theory**
**What you'll learn**: Comprehensive theory
- Classical ML theory
- LLM theory
- LLM inference theory
- Bias-variance tradeoff
- Regularization theory

**Practice**: `12_theory/`

### 13. **Interview Q&A**
**What you'll learn**: Interview preparation
- 100+ interview questions
- Detailed answers
- Code examples
- Theory explanations
- Common patterns

**Practice**: `13_interview_qa/`

### 25. **Adapters & LoRA**
**What you'll learn**: Parameter-efficient fine-tuning
- Adapter layers
- LoRA (Low-Rank Adaptation)
- How they work mathematically
- When to use them
- Simple implementations

**Practice**: `25_adapters_lora/`

### 26. **Tree-Based Methods**
**What you'll learn**: Tree algorithms
- Decision Trees (how they're learned)
- Random Forest
- Gradient Boosting
- XGBoost
- Pruning techniques
- When to use what

**Practice**: `26_tree_based_methods/`

### 27. **Advanced Theory**
**What you'll learn**: Deep ML theory
- Bias-variance tradeoff (detailed)
- Cross-validation
- Learning curves
- Feature engineering theory
- Data leakage

**Practice**: `27_advanced_theory/`

### 28. **Business Use Cases**
**What you'll learn**: Real-world problems
- Customer churn prediction (complete solution)
- Recommendation systems
- Fraud detection
- Price optimization
- Demand forecasting
- Step-by-step solutions

**Practice**: `28_business_usecases/`

### 29. **System Design for ML**
**What you'll learn**: Production ML systems
- Scalable ML pipelines
- Model serving architecture
- Feature stores
- Monitoring and alerting
- Cost optimization

**Practice**: `29_system_design_ml/`

### 30. **A/B Testing**
**What you'll learn**: Experimentation
- Statistical foundations
- Hypothesis testing
- Sample size calculation
- Interpreting results
- Multiple testing correction

**Practice**: `30_ab_testing/`

### 31. **Neural Networks from Scratch**
**What you'll learn**: Build neural networks
- Forward pass (detailed)
- Backpropagation (mathematical explanation)
- Activation functions
- Training loop
- Complete implementation

**Practice**: `31_neural_networks/`

### 32. **Anomaly Detection & Isolation Forest**
**What you'll learn**: Anomaly detection
- Isolation Forest (detailed explanation)
- How it works mathematically
- When to use it
- Comparison with other methods
- Simple implementation

**Practice**: `32_anomaly_detection/`

### 33. **Information Theory & Probability Metrics**
**What you'll learn**: Essential information theory
- Entropy (uncertainty measure)
- Cross-Entropy (loss function)
- KL Divergence (distribution distance)
- Mutual Information (feature selection)
- Gini Impurity (decision trees)
- Jensen-Shannon Divergence
- Detailed mathematical explanations
- Simple implementations

**Practice**: `33_information_theory/`

### 34. **Discriminative vs Generative Models**
**What you'll learn**: Model types and assumptions
- Discriminative models (what they are, how they work)
- Generative models (what they are, how they work)
- Key differences and when to use each
- Model assumptions (Linear Regression, Logistic Regression, SVM)
- Bayes' Theorem (detailed paragraph-style explanation)
- Simple implementations

**Practice**: `34_discriminative_generative/`

### 35. **Kernel Functions**
**What you'll learn**: Kernel methods in detail
- What kernels are and why we need them
- Linear kernel (when to use)
- Polynomial kernel (degree, parameters)
- RBF kernel (gamma parameter, most common)
- Sigmoid kernel (rarely used)
- Kernel trick explained
- When to use each kernel
- Parameter tuning
- Simple implementations with examples

**Practice**: `35_kernel_functions/`

### 36. **NLP Basics**
**What you'll learn**: Fundamental NLP concepts
- TF-IDF (Term Frequency-Inverse Document Frequency)
- N-gram models (unigram, bigram, trigram)
- Laplace smoothing (add-k smoothing)
- Language modeling
- Bayesian interpretation of L1/L2 regularization
- Detailed explanations with examples
- Simple implementations

**Practice**: `36_nlp_basics/`

### 37. **MLE and MAP Estimation**
**What you'll learn**: Maximum likelihood and Bayesian estimation
- Maximum Likelihood Estimation (MLE) - detailed derivations
- Maximum A Posteriori (MAP) - detailed derivations
- Connection between MLE and MAP
- MLE for Bernoulli, Normal, Linear Regression
- MAP with different priors (Beta, Gaussian, Laplace)
- Regularization as Bayesian prior
- Intuitive explanations with examples
- When to use each approach

**Practice**: `37_mle_map_estimation/`

### 38. **Multimodal Models & Embedding History**
**What you'll learn**: Multimodal AI and NLP evolution
- Multimodal models (CLIP, etc.) - detailed backgrounds
- CLIP architecture, training, and evaluation
- Evaluation of multimodal models
- How to train embedding models (Word2Vec, GloVe)
- History of NLP embeddings: TF-IDF → N-grams → Word2Vec → GloVe → BERT
- Training procedures for each embedding method
- Evolution of NLP from statistical to neural methods

**Practice**: `38_multimodal_and_embeddings/`

### 39. **RAG (Retrieval-Augmented Generation)**
**What you'll learn**: Industry-standard RAG systems
- RAG architecture and components
- Document ingestion and chunking strategies
- Embedding generation and vector databases
- Retrieval strategies (dense, sparse, hybrid)
- Re-ranking techniques
- Context assembly and generation
- Real-world challenges and solutions
- Production considerations (scalability, latency, cost)
- Evaluation metrics and frameworks
- Industry-standard code implementations

**Practice**: `39_rag_retrieval_augmented_generation/`

## 🚀 Quick Start Guide

### Step 1: Start with Classical ML
```bash
cd 01_classical_ml
python linear_regression.py
```

### Step 2: Progress Through Topics
Work through each numbered topic. Each includes:
- **Code implementations**: From scratch
- **Theory explanations**: Why it works
- **Interview questions**: Related Q&A

### Step 3: Practice Interview Questions
```bash
cd 13_interview_qa
# Review questions by topic
```

## 🎓 Learning Approach

### For Each Topic:
1. **Read the theory** - Understand concepts
2. **Study the code** - See implementations
3. **Run the code** - Execute examples
4. **Modify and experiment** - Change parameters
5. **Review interview Q&A** - Test understanding

## 📖 Prerequisites

### Required:
- Python 3.9+
- Basic Python knowledge
- Understanding of linear algebra
- Basic calculus

### Helpful but not required:
- PyTorch/TensorFlow experience
- Deep learning background

## 🔧 Technology Stack

- **NumPy**: Numerical computations
- **PyTorch**: Deep learning
- **Matplotlib**: Visualization
- **Scikit-learn**: Reference (for comparison)

## ❓ Common Questions

**Q: Do I need to know ML already?**
A: Basic understanding helps, but topics start from fundamentals.

**Q: How long will this take?**
A: Depends on your pace. Each topic can take a few hours to a few days.

**Q: Can I skip topics?**
A: Topics build on each other, but you can focus on areas you need.

**Q: What if I get stuck?**
A: Check the theory section, read code comments, experiment with simpler examples.

## 🎯 Learning Goals

By the end, you'll be able to:
- ✅ Implement ML algorithms from scratch
- ✅ Understand transformer architecture
- ✅ Optimize LLM inference
- ✅ Answer interview questions confidently
- ✅ Understand advanced training techniques

Let's start learning! 🚀

