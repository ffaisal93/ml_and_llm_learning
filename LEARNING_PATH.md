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

**What you'll learn**: Complete RAG system
- Architecture and components
- Chunking strategies
- Retrieval methods (TF-IDF, BM25, Dense, Hybrid)
- Evaluation metrics
- Real-world challenges and solutions

**Practice**: `39_rag_retrieval_augmented_generation/`

### 40. **Diffusion Models**

**What you'll learn**: Diffusion models for generation
- What are diffusion models and how they work
- Mathematical foundations (forward, reverse process)
- Training procedures
- Evaluation methods
- NLP applications (discrete diffusion)
- Text generation, inpainting, editing

**Practice**: `40_diffusion_models/`

### 41. **Mixture of Experts (MoE)**

**What you'll learn**: MoE architecture for efficient scaling
- What is MoE and how it works
- Routing mechanisms (top-k, switch)
- Load balancing
- Training procedures
- Memory and computation efficiency
- Real-world applications (GPT-4, Mixtral)

**Practice**: `41_mixture_of_experts/`

### 42. **State Space Models (SSM)**

**What you'll learn**: SSMs for long sequence modeling
- What are State Space Models
- Linear State Space Models (S4)
- Mamba architecture (selective SSM)
- Linear complexity vs quadratic
- Long-range dependencies
- Comparison with transformers

**Practice**: `42_state_space_models/`

### 43. **Language Modeling Training Losses**

**What you'll learn**: Pre-training objectives for language models
- MLM (Masked Language Modeling) - BERT-style
- CLM (Causal Language Modeling) - GPT-style
- NSP (Next Sentence Prediction) - BERT-style
- Mathematical formulations
- Implementation details
- When to use each

**Practice**: `43_language_modeling_losses/`

### 44. **Normalization Techniques**

**What you'll learn**: Normalization in neural networks
- Batch Normalization (BatchNorm)
- Layer Normalization (LayerNorm)
- Mathematical formulations
- Differences and when to use each
- Why transformers use LayerNorm
- Implementation from scratch

**Practice**: `44_normalization/`

### 45. **Reinforcement Learning Fundamentals**

**What you'll learn**: RL basics in easy language
- Markov Decision Process (MDP)
- Multi-Armed Bandit (exploration vs exploitation)
- Q-Learning (value-based RL)
- Monte Carlo methods (model-free learning)
- Policy Gradients (policy-based RL)
- Value Iteration and Policy Iteration
- Temporal Difference Learning
- Easy-to-understand explanations

**Practice**: `45_reinforcement_learning_fundamentals/`

### 46. **RNN and LSTM**

**What you'll learn**: Sequence models before transformers
- RNN (Recurrent Neural Network) from scratch
- LSTM (Long Short-Term Memory) from scratch
- Simple, short, precise implementations
- Key concepts and differences
- Vanishing gradient problem

**Practice**: `46_rnn_lstm/`

### 47. **Statistical Inference**
**What you'll learn**: How to reason about estimators and uncertainty
- Population vs sample
- Bias, variance, and MSE
- Maximum likelihood estimation
- Confidence intervals
- Hypothesis testing and p-values
- Bootstrap confidence intervals
- Beta-Bernoulli Bayesian updating

**Practice**: `47_statistical_inference/`

### 48. **Optimization and Matrix Calculus**
**What you'll learn**: Optimization intuition for ML interviews
- Gradients, Jacobians, and Hessians
- Chain rule for models
- Common closed-form gradients
- Convexity and conditioning
- SGD, momentum, and Adam trade-offs
- Numerical gradient checking
- Constraint intuition with Lagrange multipliers

**Practice**: `48_optimization_and_matrix_calculus/`

### 49. **Generalization and Evaluation**
**What you'll learn**: How to trust or challenge model results
- Train / validation / test roles
- Overfitting and underfitting
- Data leakage
- Class imbalance and metric choice
- Calibration
- Slice-based error analysis
- Distribution shift
- Ablations and metric confidence intervals

**Practice**: `49_generalization_and_evaluation/`

### 50. **ML Coding Interview Patterns**
**What you'll learn**: Fast reusable patterns for coding rounds
- Stable softmax
- Masking and causal masks
- Vectorized distance computation
- Padding and batching
- Top-k and top-p filtering
- k-means update steps
- Pressure-friendly implementation habits

**Practice**: `50_ml_coding_interview_patterns/`

### 51. **LLM Research Interview Prep**
**What you'll learn**: How to reason like an LLM researcher in interviews
- Pretraining objective discussion
- Tokenization and scaling trade-offs
- Perplexity, exact match, F1, pass@k, retrieval metrics
- Pairwise preference evaluation
- Ablation design and interpretation
- RAG failure diagnosis
- Paper discussion structure

**Practice**: `51_llm_research_interview_prep/`

### 52. **Statistical Learning Theory**
**What you'll learn**: Why models generalize or fail to generalize
- Empirical risk vs population risk
- Generalization gap
- Capacity and complexity intuition
- VC/PAC-style interview intuition
- Regularization as inductive bias
- Why more data helps
- Double descent intuition

**Practice**: `52_statistical_learning_theory/`

### 53. **ML Debugging and Mock Coding**
**What you'll learn**: How to debug and code under interview pressure
- Flat loss debugging
- NaN and numerical stability checks
- Leakage detection
- Shape and mask debugging
- Timed coding prompts
- Training-step inspection habits

**Practice**: `53_ml_debugging_and_mock_coding/`

### 54. **Data Manipulation for ML**
**What you'll learn**: Practical feature-table and preprocessing patterns
- Filling missing values correctly
- Z-scoring with training statistics only
- Groupby aggregations
- Joins and merge logic
- One-hot encoding
- Leakage-aware preprocessing

**Practice**: `54_data_manipulation_for_ml/`

### 55. **Research Papers and Mock Interviews**
**What you'll learn**: Final-layer research interview reasoning
- Paper discussion structure
- Experiment criticism
- Distribution-membership questions
- Research judgment prompts
- Oral mock interview questions

**Practice**: `55_research_papers_and_mock_interviews/`

### 56. **Spoken Interview Question Bank**
**What you'll learn**: How to answer clearly out loud under interview pressure
- Probability and statistics answers
- Optimization and generalization answers
- Coding and debugging verbal reasoning
- LLM systems answers
- Research judgment and paper discussion answers
- Short quick-drill responses

**Practice**: `56_spoken_interview_question_bank/`

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
