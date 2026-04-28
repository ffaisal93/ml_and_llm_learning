# ML & LLM Learning: Coding Interview Preparation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A comprehensive repository for ML and LLM coding interview preparation with implementations, theory, and interview Q&A.

---

## 🔥 Frontier-Lab Interview Prep — Start Here

These deep-dive and grill files are the highest-yield content in the repo for applied scientist / ML engineer interviews at frontier labs and big tech. Each topic has a `*_DEEP_DIVE.md` (interview-grade theory) plus an `INTERVIEW_GRILL.md` (50–60 brutal questions with strong answers). Drill the grill files until you can answer 40+ cold.

| Topic | Why it matters | Files |
|---|---|---|
| **Learning rate / Gradient descent** | The single hyperparameter most likely to make training succeed or fail. Interviewers probe this to test if you actually understand optimization. | [`02_gradient_descent/LEARNING_RATE_DEEP_DIVE.md`](02_gradient_descent/LEARNING_RATE_DEEP_DIVE.md) · [`02_gradient_descent/INTERVIEW_GRILL.md`](02_gradient_descent/INTERVIEW_GRILL.md) |
| **Optimizers** | SGD, Adam, AdamW, Lion, Sophia, Shampoo — what each fixes, when each wins. AdamW vs Adam+L2 is a high-frequency interview question. | [`10_optimizers/README.md`](10_optimizers/README.md) · [`10_optimizers/INTERVIEW_GRILL.md`](10_optimizers/INTERVIEW_GRILL.md) |
| **Logistic regression** | Simplest model with the richest theoretical structure. Many senior offers turn on five hard logistic-regression questions. | [`01_classical_ml/LOGISTIC_REGRESSION_DEEP_DIVE.md`](01_classical_ml/LOGISTIC_REGRESSION_DEEP_DIVE.md) · [`01_classical_ml/LOGISTIC_REGRESSION_INTERVIEW_GRILL.md`](01_classical_ml/LOGISTIC_REGRESSION_INTERVIEW_GRILL.md) |
| **LLM inference** | Prefill vs decode, KV cache memory math, PagedAttention, FlashAttention, speculative decoding (with rejection-sampling proof), quantization. Critical for serving / infra roles. | [`06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`](06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md) · [`06_llm_inference/INTERVIEW_GRILL.md`](06_llm_inference/INTERVIEW_GRILL.md) |
| **Post-training & alignment** | RLHF math, full DPO derivation (whiteboard-ready), the alphabet soup (IPO/KTO/ORPO/SimPO/GRPO), Constitutional AI, reward hacking, KL blowup, alignment tax. | [`08_training_techniques/ALIGNMENT_DEEP_DIVE.md`](08_training_techniques/ALIGNMENT_DEEP_DIVE.md) · [`08_training_techniques/INTERVIEW_GRILL.md`](08_training_techniques/INTERVIEW_GRILL.md) |
| **Transformers** | Scaled-dot-product derivation, multi-head reasoning, FFN role, residual stream view, pre-LN vs post-LN, encoder/decoder/cross-attention, scaling laws. | [`04_transformers/TRANSFORMERS_DEEP_DIVE.md`](04_transformers/TRANSFORMERS_DEEP_DIVE.md) · [`04_transformers/INTERVIEW_GRILL.md`](04_transformers/INTERVIEW_GRILL.md) |
| **Attention mechanisms** | MHA → MQA → GQA → MLA hierarchy, sliding window receptive-field math, sparse and linear attention, induction heads, attention sinks. | [`05_attention_mechanisms/ATTENTION_DEEP_DIVE.md`](05_attention_mechanisms/ATTENTION_DEEP_DIVE.md) · [`05_attention_mechanisms/INTERVIEW_GRILL.md`](05_attention_mechanisms/INTERVIEW_GRILL.md) |
| **Normalization** | BN/LN/RMSNorm/GroupNorm, why BN fails for transformers, pre-LN vs post-LN stability, the affine transform, the loss-landscape-smoothing argument. | [`44_normalization/NORMALIZATION_DEEP_DIVE.md`](44_normalization/NORMALIZATION_DEEP_DIVE.md) · [`44_normalization/INTERVIEW_GRILL.md`](44_normalization/INTERVIEW_GRILL.md) |
| **Positional embeddings** | Sinusoidal/learned/T5-bias/RoPE/ALiBi/NoPE, full RoPE derivation showing relative-position from rotated dot products, NTK scaling, YaRN. | [`14_advanced_positional_embeddings/POSITIONAL_DEEP_DIVE.md`](14_advanced_positional_embeddings/POSITIONAL_DEEP_DIVE.md) · [`14_advanced_positional_embeddings/INTERVIEW_GRILL.md`](14_advanced_positional_embeddings/INTERVIEW_GRILL.md) |
| **Tokenization** | BPE/WordPiece/Unigram/SentencePiece, byte-level BPE, vocabulary trade-offs, arithmetic and multilingual quirks, glitch tokens, multimodal extensions. | [`15_tokenization/TOKENIZATION_DEEP_DIVE.md`](15_tokenization/TOKENIZATION_DEEP_DIVE.md) · [`15_tokenization/INTERVIEW_GRILL.md`](15_tokenization/INTERVIEW_GRILL.md) |
| **Evaluation metrics** | Classification (precision/recall/F1/AUROC/PR-AUC), regression (MSE/MAE/R²/quantile), ranking (MAP/NDCG/MRR), LLM-specific (PPL/pass@k/BLEU/LLM-as-judge), calibration, Goodhart's Law. | [`03_evaluation_metrics/EVALUATION_METRICS_DEEP_DIVE.md`](03_evaluation_metrics/EVALUATION_METRICS_DEEP_DIVE.md) · [`03_evaluation_metrics/INTERVIEW_GRILL.md`](03_evaluation_metrics/INTERVIEW_GRILL.md) |
| **Regularization** | Bias-variance, L1/L2 geometry and Bayesian priors, dropout (3 stories), early stopping ≈ L2, MixUp/CutMix, label smoothing, SAM, implicit regularization of SGD. | [`11_regularization/REGULARIZATION_DEEP_DIVE.md`](11_regularization/REGULARIZATION_DEEP_DIVE.md) · [`11_regularization/INTERVIEW_GRILL.md`](11_regularization/INTERVIEW_GRILL.md) |
| **Sampling techniques** | Greedy/beam/temperature/top-k/top-p/min-p/typical/Mirostat/penalties, why beam search fails for LLMs, speculative decoding, best-of-N for test-time scaling. | [`09_sampling_techniques/SAMPLING_DEEP_DIVE.md`](09_sampling_techniques/SAMPLING_DEEP_DIVE.md) · [`09_sampling_techniques/INTERVIEW_GRILL.md`](09_sampling_techniques/INTERVIEW_GRILL.md) |
| **Language modeling losses** | CLM/MLM/Span-corruption/PrefixLM/MoD/ELECTRA, why CLM dominates, why NSP died, how ICL emerges from CLM, multi-token prediction, prompt masking for SFT. | [`43_language_modeling_losses/LM_LOSSES_DEEP_DIVE.md`](43_language_modeling_losses/LM_LOSSES_DEEP_DIVE.md) · [`43_language_modeling_losses/INTERVIEW_GRILL.md`](43_language_modeling_losses/INTERVIEW_GRILL.md) |
| **Information theory** | Entropy/cross-entropy/KL/MI, forward vs reverse KL, why MLE = forward KL, MI for contrastive (InfoNCE/CLIP), KL in VAE/RLHF/distillation, source coding theorem. | [`33_information_theory/INFORMATION_THEORY_DEEP_DIVE.md`](33_information_theory/INFORMATION_THEORY_DEEP_DIVE.md) · [`33_information_theory/INTERVIEW_GRILL.md`](33_information_theory/INTERVIEW_GRILL.md) |
| **RAG** | Indexing/retrieve/rerank/generate pipeline, chunking strategies, BM25 vs dense vs hybrid, HNSW/IVF/PQ, embedding models, HyDE, lost-in-the-middle, RAGAS, Self-RAG/GraphRAG. | [`39_rag_retrieval_augmented_generation/RAG_DEEP_DIVE.md`](39_rag_retrieval_augmented_generation/RAG_DEEP_DIVE.md) · [`39_rag_retrieval_augmented_generation/INTERVIEW_GRILL.md`](39_rag_retrieval_augmented_generation/INTERVIEW_GRILL.md) |
| **Mixture of Experts** | Top-k routing, load balancing loss derivation, capacity factor / token dropping, expert parallelism + all-to-all, Switch/Mixtral/DeepSeek-V3, auxiliary-loss-free balancing. | [`41_mixture_of_experts/MOE_DEEP_DIVE.md`](41_mixture_of_experts/MOE_DEEP_DIVE.md) · [`41_mixture_of_experts/INTERVIEW_GRILL.md`](41_mixture_of_experts/INTERVIEW_GRILL.md) |
| **State Space Models** | Continuous SSM ODE, discretization, recurrent vs convolutional view, HiPPO, S4 (DPLR), Mamba (selectivity + parallel scan), hybrid models (Jamba). | [`42_state_space_models/SSM_DEEP_DIVE.md`](42_state_space_models/SSM_DEEP_DIVE.md) · [`42_state_space_models/INTERVIEW_GRILL.md`](42_state_space_models/INTERVIEW_GRILL.md) |
| **Diffusion models** | Forward/reverse processes, why predict noise, score-matching connection, DDIM/DPM-Solver, classifier-free guidance, latent diffusion, DiT, flow matching. | [`40_diffusion_models/DIFFUSION_DEEP_DIVE.md`](40_diffusion_models/DIFFUSION_DEEP_DIVE.md) · [`40_diffusion_models/INTERVIEW_GRILL.md`](40_diffusion_models/INTERVIEW_GRILL.md) |
| **LoRA & PEFT** | LoRA math (ΔW = B·A), intrinsic-dimension hypothesis, α/r scaling, QLoRA's NF4 + double quantization + paged optimizer, adapters/prefix/IA³/DoRA/GaLore, multi-LoRA serving. | [`25_adapters_lora/LORA_DEEP_DIVE.md`](25_adapters_lora/LORA_DEEP_DIVE.md) · [`25_adapters_lora/INTERVIEW_GRILL.md`](25_adapters_lora/INTERVIEW_GRILL.md) |
| **Tree-based methods** | Gini/entropy splits, RF (bagging + feature subsampling), GBDT (functional gradient descent), XGBoost (second-order + regularization), LightGBM (histogram + leaf-wise), CatBoost (ordered boosting). | [`26_tree_based_methods/TREES_DEEP_DIVE.md`](26_tree_based_methods/TREES_DEEP_DIVE.md) · [`26_tree_based_methods/INTERVIEW_GRILL.md`](26_tree_based_methods/INTERVIEW_GRILL.md) |
| **Kernel functions** | Kernel trick, Mercer's theorem, RBF/polynomial/string kernels, SVM dual, RKHS, kernel ridge, NTK, attention-as-kernel-smoothing. | [`35_kernel_functions/KERNELS_DEEP_DIVE.md`](35_kernel_functions/KERNELS_DEEP_DIVE.md) · [`35_kernel_functions/INTERVIEW_GRILL.md`](35_kernel_functions/INTERVIEW_GRILL.md) |
| **Clustering (advanced)** | K-means as coordinate descent, GMM with full EM derivation, DBSCAN core/border/noise, hierarchical linkage, spectral clustering, evaluation metrics. | [`19_advanced_clustering/CLUSTERING_DEEP_DIVE.md`](19_advanced_clustering/CLUSTERING_DEEP_DIVE.md) · [`19_advanced_clustering/INTERVIEW_GRILL.md`](19_advanced_clustering/INTERVIEW_GRILL.md) |
| **Dimensionality reduction** | PCA (variance-max derivation, SVD, Eckart-Young), kernel PCA, t-SNE (KL with Student-t), UMAP, autoencoders/VAE, ICA, NMF, method-selection guide. | [`21_dimensionality_reduction/DIMENSIONALITY_REDUCTION_DEEP_DIVE.md`](21_dimensionality_reduction/DIMENSIONALITY_REDUCTION_DEEP_DIVE.md) · [`21_dimensionality_reduction/INTERVIEW_GRILL.md`](21_dimensionality_reduction/INTERVIEW_GRILL.md) |
| **Neural networks fundamentals** | MLP, universal approximation, activations (ReLU/GELU/SwiGLU), He/Xavier init derivations, full backprop, vanishing/exploding gradients, residual connections, modern training tricks. | [`31_neural_networks/NEURAL_NETWORKS_DEEP_DIVE.md`](31_neural_networks/NEURAL_NETWORKS_DEEP_DIVE.md) · [`31_neural_networks/INTERVIEW_GRILL.md`](31_neural_networks/INTERVIEW_GRILL.md) |
| **Statistical inference** | Estimators (unbiased/consistent/efficient + CRLB), MLE asymptotics, Wald/bootstrap/credible intervals, hypothesis testing, multiple testing (Bonferroni/BH), Bayesian updates with conjugate priors. | [`47_statistical_inference/STATISTICAL_INFERENCE_DEEP_DIVE.md`](47_statistical_inference/STATISTICAL_INFERENCE_DEEP_DIVE.md) · [`47_statistical_inference/INTERVIEW_GRILL.md`](47_statistical_inference/INTERVIEW_GRILL.md) |
| **MLE & MAP** | Full MLE derivations (Bernoulli/Gaussian/Poisson/multinomial/linreg/logreg), asymptotic theory, MAP-as-regularization (ridge from Gaussian prior, lasso from Laplace), conjugate priors, MLE = forward KL, RLHF/DPO connections. | [`37_mle_map_estimation/MLE_MAP_DEEP_DIVE.md`](37_mle_map_estimation/MLE_MAP_DEEP_DIVE.md) · [`37_mle_map_estimation/INTERVIEW_GRILL.md`](37_mle_map_estimation/INTERVIEW_GRILL.md) |
| **Linear algebra for ML** | Rank, eigendecomposition (spectral theorem), SVD (Eckart-Young), positive (semi)definiteness, matrix calculus (OLS gradient + Hessian), conditioning, projections. | [`24_linear_algebra_qa/LINEAR_ALGEBRA_DEEP_DIVE.md`](24_linear_algebra_qa/LINEAR_ALGEBRA_DEEP_DIVE.md) · [`24_linear_algebra_qa/INTERVIEW_GRILL.md`](24_linear_algebra_qa/INTERVIEW_GRILL.md) |
| **Probability for ML** | Axioms, Bayes' theorem (with base-rate fallacy), expectations and variance (linearity, total expectation, total variance), common distributions, multivariate Gaussian (marginals/conditionals), LLN/CLT. | [`17_probability_math/PROBABILITY_DEEP_DIVE.md`](17_probability_math/PROBABILITY_DEEP_DIVE.md) · [`17_probability_math/INTERVIEW_GRILL.md`](17_probability_math/INTERVIEW_GRILL.md) |
| **Picking distributions / GLMs** | Which distribution for which data type, exponential family unification, GLMs and canonical links (linreg/logreg/Poisson), heavy-tailed distributions, common pitfalls. | [`18_distribution_classification/DISTRIBUTIONS_DEEP_DIVE.md`](18_distribution_classification/DISTRIBUTIONS_DEEP_DIVE.md) · [`18_distribution_classification/INTERVIEW_GRILL.md`](18_distribution_classification/INTERVIEW_GRILL.md) |
| **Generalization & evaluation** | Data leakage (4 types), calibration (ECE, Platt/isotonic/temperature), distribution shift (covariate/label/concept), class imbalance, double descent, cross-validation done right, ablations, metric uncertainty. | [`49_generalization_and_evaluation/GENERALIZATION_DEEP_DIVE.md`](49_generalization_and_evaluation/GENERALIZATION_DEEP_DIVE.md) · [`49_generalization_and_evaluation/INTERVIEW_GRILL.md`](49_generalization_and_evaluation/INTERVIEW_GRILL.md) |
| **A/B testing** | Hypothesis tests, sample-size formulas, CUPED, peeking and sequential testing, SUTVA / network effects, SRM check, novelty effects, multiple testing, Bayesian A/B, ML-specific (interleaving, holdback, off-policy / IPS). | [`30_ab_testing/AB_TESTING_DEEP_DIVE.md`](30_ab_testing/AB_TESTING_DEEP_DIVE.md) · [`30_ab_testing/INTERVIEW_GRILL.md`](30_ab_testing/INTERVIEW_GRILL.md) |
| **Large-scale LLM systems** | Training memory math ($16P$ rule), activation checkpointing, BF16/FP8, ZeRO-1/2/3 / FSDP, Megatron tensor parallelism, pipeline parallelism + bubble formula, 3D parallelism, expert parallelism for MoE, sequence/context parallelism, MFU, training failure modes. | [`61_large_scale_llm_systems/LARGE_SCALE_LLM_DEEP_DIVE.md`](61_large_scale_llm_systems/LARGE_SCALE_LLM_DEEP_DIVE.md) · [`61_large_scale_llm_systems/INTERVIEW_GRILL.md`](61_large_scale_llm_systems/INTERVIEW_GRILL.md) |
| **RL fundamentals** | MDPs, Bellman equations, value/policy iteration, Q-learning vs SARSA (on vs off-policy), DQN tricks, policy gradient theorem with derivation, REINFORCE + baselines, actor-critic, TRPO/PPO with clipped surrogate, GAE, RLHF connection, GRPO. | [`45_rl_fundamentals/RL_DEEP_DIVE.md`](45_rl_fundamentals/RL_DEEP_DIVE.md) · [`45_rl_fundamentals/INTERVIEW_GRILL.md`](45_rl_fundamentals/INTERVIEW_GRILL.md) |
| **ML system design** | 6-step framework (clarify → frame → data → features+model → serving → monitoring), two-stage retrieval, cold start, cost asymmetry, drift detection, shadow/canary deployment, worked examples (recommender, fraud). | [`29_system_design_for_ml/ML_SYSTEM_DESIGN_DEEP_DIVE.md`](29_system_design_for_ml/ML_SYSTEM_DESIGN_DEEP_DIVE.md) · [`29_system_design_for_ml/INTERVIEW_GRILL.md`](29_system_design_for_ml/INTERVIEW_GRILL.md) |
| **Optimization (deeper)** | Convex/strongly-convex/smooth definitions, GD convergence rates, Nesterov acceleration, Newton/BFGS/Gauss-Newton, SGD scaling, Lagrangian + KKT (with SVM dual), deep-learning loss landscape (saddles dominate, flat minima, edge of stability). | [`48_optimization_and_matrix_calculus/OPTIMIZATION_DEEP_DIVE.md`](48_optimization_and_matrix_calculus/OPTIMIZATION_DEEP_DIVE.md) · [`48_optimization_and_matrix_calculus/INTERVIEW_GRILL.md`](48_optimization_and_matrix_calculus/INTERVIEW_GRILL.md) |
| **Multimodal & embedding history** | BoW/TF-IDF → Word2Vec/GloVe → BERT → Sentence-BERT → CLIP → multimodal LLMs (Flamingo, LLaVA), full CLIP loss derivation, InfoNCE as MI bound, SigLIP, vector search (HNSW/IVF-PQ), hybrid retrieval. | [`38_multimodal_models_and_embedding_history/MULTIMODAL_EMBEDDING_DEEP_DIVE.md`](38_multimodal_models_and_embedding_history/MULTIMODAL_EMBEDDING_DEEP_DIVE.md) · [`38_multimodal_models_and_embedding_history/INTERVIEW_GRILL.md`](38_multimodal_models_and_embedding_history/INTERVIEW_GRILL.md) |
| **Statistical learning theory** | ERM, PAC learning, VC dimension, Rademacher complexity, bias-variance, double descent, no-free-lunch theorem, regularization-as-inductive-bias, modern bounds (PAC-Bayes, stability, compression). | [`52_statistical_learning_theory/STATISTICAL_LEARNING_THEORY_DEEP_DIVE.md`](52_statistical_learning_theory/STATISTICAL_LEARNING_THEORY_DEEP_DIVE.md) · [`52_statistical_learning_theory/INTERVIEW_GRILL.md`](52_statistical_learning_theory/INTERVIEW_GRILL.md) |
| **RNNs & LSTMs** | Vanilla RNN forward/BPTT, vanishing/exploding gradients (Jacobian product analysis), LSTM gates and cell-state additive update, GRU, bidirectional, seq2seq + attention (Bahdanau/Luong), transformer transition, connection to modern SSMs. | [`46_rnn_lstm/RNN_LSTM_DEEP_DIVE.md`](46_rnn_lstm/RNN_LSTM_DEEP_DIVE.md) · [`46_rnn_lstm/INTERVIEW_GRILL.md`](46_rnn_lstm/INTERVIEW_GRILL.md) |
| **Discriminative vs generative** | $p(y\|x)$ vs $p(x,y)$, Naive Bayes, LDA/QDA decision boundaries, LDA = linear boundary same as logistic regression, Ng & Jordan sample-complexity result, HMM, modern generative models (VAE/GAN/diffusion/LLM). | [`34_discriminative_vs_generative/DISCRIMINATIVE_VS_GENERATIVE_DEEP_DIVE.md`](34_discriminative_vs_generative/DISCRIMINATIVE_VS_GENERATIVE_DEEP_DIVE.md) · [`34_discriminative_vs_generative/INTERVIEW_GRILL.md`](34_discriminative_vs_generative/INTERVIEW_GRILL.md) |
| **Frontier training playbook** | Methodology over architecture, scaling laws (Kaplan/Chinchilla), past-Chinchilla for inference cost, MoE/GQA/MLA trade-offs, data dedup + filtering, stability tricks (z-loss, softcapping, QK-norm), staged training, ablation methodology. | [`62_frontier_training_playbook/frontier_training_deep_dive.md`](62_frontier_training_playbook/frontier_training_deep_dive.md) · [`62_frontier_training_playbook/INTERVIEW_GRILL.md`](62_frontier_training_playbook/INTERVIEW_GRILL.md) |
| **Paged attention & LLM serving** | KV-cache math (GQA/MQA/MLA savings), PagedAttention internals (block tables, paging analogy), continuous batching, prefix caching / RadixAttention, speculative decoding, INT8/INT4/FP8 quantization, vLLM/SGLang/TensorRT-LLM. | [`63_paged_attention_and_llm_serving/paged_attention_deep_dive.md`](63_paged_attention_and_llm_serving/paged_attention_deep_dive.md) · [`63_paged_attention_and_llm_serving/INTERVIEW_GRILL.md`](63_paged_attention_and_llm_serving/INTERVIEW_GRILL.md) |
| **Recommendation systems** | Collaborative filtering, matrix factorization (BPR), two-tower retrieval (in-batch negatives, ANN serving), sequential models (GRU4Rec/SASRec/BERT4Rec), two-stage retrieval+ranking, GBDT/DeepFM/DLRM, NDCG/MAP/MRR, cold start, echo chamber + exploration. | [`22_recommendation_systems/RECOMMENDATION_SYSTEMS_DEEP_DIVE.md`](22_recommendation_systems/RECOMMENDATION_SYSTEMS_DEEP_DIVE.md) · [`22_recommendation_systems/INTERVIEW_GRILL.md`](22_recommendation_systems/INTERVIEW_GRILL.md) |
| **Anomaly detection** | Statistical (z-score, Mahalanobis), density-based (KDE, LOF), Isolation Forest score derivation, One-Class SVM, autoencoder reconstruction, embedding-based AD, time-series anomalies (point/contextual/collective), AUPRC over AUC. | [`32_anomaly_detection/ANOMALY_DETECTION_DEEP_DIVE.md`](32_anomaly_detection/ANOMALY_DETECTION_DEEP_DIVE.md) · [`32_anomaly_detection/INTERVIEW_GRILL.md`](32_anomaly_detection/INTERVIEW_GRILL.md) |
| **Business case studies** | 9-step case-study framework + canonical templates (churn, fraud, recs, forecasting, pricing, lead scoring, content moderation, search) — end-to-end answers covering data, leakage, model, evaluation, deployment, iteration. | [`28_business_use_cases/BUSINESS_CASE_STUDIES_DEEP_DIVE.md`](28_business_use_cases/BUSINESS_CASE_STUDIES_DEEP_DIVE.md) · [`28_business_use_cases/INTERVIEW_GRILL.md`](28_business_use_cases/INTERVIEW_GRILL.md) |
| **NLP basics** | TF-IDF, n-gram language models, smoothing (Laplace, Good-Turing, Kneser-Ney with continuation count), perplexity, Zipf's law, Heaps' law, edit distance DP, BM25 with hyperparameter intuition. | [`36_nlp_basics/NLP_BASICS_DEEP_DIVE.md`](36_nlp_basics/NLP_BASICS_DEEP_DIVE.md) · [`36_nlp_basics/INTERVIEW_GRILL.md`](36_nlp_basics/INTERVIEW_GRILL.md) |
| **Advanced ML theory** | Bias-variance decomposition with proof, cross-validation theory (k-fold/stratified/group/time-series/nested with LOO closed form), learning curves, AIC vs BIC, ROC/PR curves with cost-aware operating points, F-beta scores. | [`27_advanced_theory/ADVANCED_THEORY_DEEP_DIVE.md`](27_advanced_theory/ADVANCED_THEORY_DEEP_DIVE.md) · [`27_advanced_theory/INTERVIEW_GRILL.md`](27_advanced_theory/INTERVIEW_GRILL.md) |
| **LLM problems & mitigations** | Long-context (lost-in-the-middle), hallucination overview, prompting (CoT, self-consistency, ToT, ReAct), jailbreaks + defenses, indirect prompt injection, agent architectures and failure modes, multi-turn memory, latency/cost, evaluation. | [`07_llm_problems/LLM_PROBLEMS_DEEP_DIVE.md`](07_llm_problems/LLM_PROBLEMS_DEEP_DIVE.md) · [`07_llm_problems/INTERVIEW_GRILL.md`](07_llm_problems/INTERVIEW_GRILL.md) |
| **Hallucination detection (LLM)** | Full taxonomy (factual / faithfulness / source / logical / self-contradictory; intrinsic vs extrinsic), causes (RLHF-honesty paradox, lost-in-the-middle, citation hallucination), detection methods across 3 families (reference-based: NLI/QA/citation/KG; reference-free: SelfCheckGPT, semantic entropy, CoVe; internal-states: truth probes, EigenScore, SAPLMA), RAG-specific (RAGAS, citation faithfulness, AIS), benchmarks (TruthfulQA, SimpleQA, HaluEval, FactScore, RAGTruth), production cascade design, 90 active-recall questions. | [`07_llm_problems/HALLUCINATION_DETECTION_DEEP_DIVE.md`](07_llm_problems/HALLUCINATION_DETECTION_DEEP_DIVE.md) · [`07_llm_problems/HALLUCINATION_INTERVIEW_GRILL.md`](07_llm_problems/HALLUCINATION_INTERVIEW_GRILL.md) |
| **LLM evaluation** | Why LLM eval is hard, capability benchmarks (MMLU-Pro, GPQA, MATH/AIME, HumanEval+/SWE-Bench-Verified/LiveCodeBench, RULER long-context, MMMU, GAIA, TAU-bench), instruction following (IFEval, MT-Bench, AlpacaEval-2 length-controlled, Arena-Hard-Auto), LLM-as-judge methodology (5 biases + calibration), pairwise / ELO / Bradley-Terry / Chatbot Arena, factuality measurement (FactScore, SAFE, RAGAS, FACTS Grounding), contamination detection (Min-K%-prob, time-shifted benchmarks), robustness, statistical methodology (CIs, pass@k, multiple comparisons), harnesses (lm-eval-harness, HELM, OpenCompass, Inspect), online telemetry, A/B testing for LLM products, full product eval suite case study, 115 active-recall questions. | [`07_llm_problems/LLM_EVALUATION_DEEP_DIVE.md`](07_llm_problems/LLM_EVALUATION_DEEP_DIVE.md) · [`07_llm_problems/LLM_EVALUATION_INTERVIEW_GRILL.md`](07_llm_problems/LLM_EVALUATION_INTERVIEW_GRILL.md) |
| **Build an agent in 30 min** | Codable-from-memory agent: 70-line working loop with tool calls + parser; production extensions (memory, parallel tools, planner, observability, streaming); 8 failure modes with mitigations; 5-min interview narrative. | [`07_llm_problems/AGENT_IN_30_MIN.md`](07_llm_problems/AGENT_IN_30_MIN.md) |
| **Clustering evaluation** | Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn), external metrics (ARI, NMI, V-measure, purity), choosing $K$ (elbow / silhouette / gap statistic / stability), bootstrap stability validation, common pitfalls. | [`23_clustering_evaluation/CLUSTERING_EVALUATION_DEEP_DIVE.md`](23_clustering_evaluation/CLUSTERING_EVALUATION_DEEP_DIVE.md) · [`23_clustering_evaluation/INTERVIEW_GRILL.md`](23_clustering_evaluation/INTERVIEW_GRILL.md) |
| **Cross-topic synthesis** | Meta-document: 5 archetype questions (design/train/why-works/debug/tradeoff), bridge topics (cross-entropy, embeddings, attention, bias-variance, data curation), first-principles answer pattern, common mistakes, synthesis cheatsheet. | [`64_integrated_ai_ml_interview_synthesis/INTERVIEW_SYNTHESIS_DEEP_DIVE.md`](64_integrated_ai_ml_interview_synthesis/INTERVIEW_SYNTHESIS_DEEP_DIVE.md) · [`64_integrated_ai_ml_interview_synthesis/INTERVIEW_GRILL.md`](64_integrated_ai_ml_interview_synthesis/INTERVIEW_GRILL.md) |
| **ML coding patterns** | Stable softmax + log-sum-exp, scaled dot-product attention with masking and multi-head, top-k/top-p sampling, beam search with length normalization, K-means, padding/masking, vectorized cosine similarity, logistic regression, backprop from scratch. | [`50_ml_coding_interview_patterns/CODING_PATTERNS_DEEP_DIVE.md`](50_ml_coding_interview_patterns/CODING_PATTERNS_DEEP_DIVE.md) · [`50_ml_coding_interview_patterns/INTERVIEW_GRILL.md`](50_ml_coding_interview_patterns/INTERVIEW_GRILL.md) |
| **ML debugging** | 8-layer debugging tree, loss-curve interpretation (flat / explode / val gap / spike), sanity checks (overfit one batch, tiny dataset), NaN debugging (FP16/log-of-zero/anomaly detection), leakage detection, gradient checking, distribution-shift investigation. | [`53_ml_debugging_and_mock_coding/ML_DEBUGGING_DEEP_DIVE.md`](53_ml_debugging_and_mock_coding/ML_DEBUGGING_DEEP_DIVE.md) · [`53_ml_debugging_and_mock_coding/INTERVIEW_GRILL.md`](53_ml_debugging_and_mock_coding/INTERVIEW_GRILL.md) |
| **Training behaviors** | Healthy loss curves and pathologies, LR (warmup, decay, finder), batch size effects (linear scaling, critical batch, generalization gap), gradient norm tracking, mixed precision (FP16/BF16/FP8), loss spike recovery, catastrophic forgetting + replay/EWC. | [`16_training_behaviors/TRAINING_BEHAVIORS_DEEP_DIVE.md`](16_training_behaviors/TRAINING_BEHAVIORS_DEEP_DIVE.md) · [`16_training_behaviors/INTERVIEW_GRILL.md`](16_training_behaviors/INTERVIEW_GRILL.md) |
| **Whiteboard derivations** | Meta-collection of 13 must-master derivations: backprop, attention, OLS, logistic gradient, KL, EM, PCA via SVD, SVM dual, RoPE rotation, DPO, ELBO, bias-variance, info gain — each with step-by-step proof + cross-reference. | [`58_whiteboard_derivations/WHITEBOARD_DERIVATIONS_DEEP_DIVE.md`](58_whiteboard_derivations/WHITEBOARD_DERIVATIONS_DEEP_DIVE.md) · [`58_whiteboard_derivations/INTERVIEW_GRILL.md`](58_whiteboard_derivations/INTERVIEW_GRILL.md) |
| **Multi-turn conversation design** | Memory strategies (sliding window / summarization / retrieval / hybrid), persona consistency + sycophancy, multi-turn evaluation (simulated users, trajectory metrics), state management, tool integration, prompt template formats, prompt caching, personalization. | [`20_multi_turn_conversations/MULTI_TURN_DEEP_DIVE.md`](20_multi_turn_conversations/MULTI_TURN_DEEP_DIVE.md) · [`20_multi_turn_conversations/INTERVIEW_GRILL.md`](20_multi_turn_conversations/INTERVIEW_GRILL.md) |

**Recommended drill sequence:** (1) Read each `*_DEEP_DIVE.md` start to finish. (2) Drill `INTERVIEW_GRILL.md` until 40+/60 cold. (3) Cycle back through the misses the next day. (4) Ask a friend to randomly pick 10 questions per topic and grill you out loud.

The remaining content of this repo (60+ topic folders) is supporting material. The five interview-grade pairs above are the highest-leverage files.

---


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
- **Diffusion Models** - Complete theory, training, evaluation, NLP applications
- **Mixture of Experts (MoE)** - Architecture, routing, load balancing, efficiency
- **State Space Models (SSM)** - Mamba, linear complexity, long sequence modeling
- **Language Modeling Losses** - MLM, CLM, NSP implementations and explanations
- **Normalization Techniques** - BatchNorm and LayerNorm with detailed theory and implementations
- **Reinforcement Learning Fundamentals** - MDP, Q-Learning, Multi-Armed Bandit, Monte Carlo in easy language
- **RNN and LSTM** - Simple, short, precise implementations from scratch

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
├── 13_interview_qa/              # Interview questions & answers
├── ...
├── 47_statistical_inference/     # Estimators, CIs, tests, Bayesian updates
├── 48_optimization_and_matrix_calculus/  # Gradients, Hessians, conditioning
├── 49_generalization_and_evaluation/     # Leakage, calibration, ablations
├── 50_ml_coding_interview_patterns/      # Pressure-friendly coding templates
├── 51_llm_research_interview_prep/       # LLM eval, ablations, research judgment
├── ...
├── 62_frontier_training_playbook/        # Architecture, stability, data, ablations
├── 63_paged_attention_and_llm_serving/   # KV cache, fragmentation, paging, batching
└── 64_integrated_ai_ml_interview_synthesis/  # Cross-topic interview answer patterns
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
40. **Diffusion Models** - Complete theory, training, evaluation, NLP applications
41. **Mixture of Experts (MoE)** - Architecture, routing, load balancing, efficiency
42. **State Space Models (SSM)** - Mamba, linear complexity, long sequence modeling
43. **Language Modeling Losses** - MLM, CLM, NSP implementations and explanations
44. **Normalization Techniques** - BatchNorm and LayerNorm with detailed theory and implementations
45. **Reinforcement Learning Fundamentals** - MDP, Q-Learning, Multi-Armed Bandit, Monte Carlo in easy language
46. **RNN and LSTM** - Simple, short, precise implementations from scratch
47. **Statistical Inference** - Estimators, MLE, confidence intervals, hypothesis tests, bootstrap, Bayesian updates
48. **Optimization and Matrix Calculus** - Gradients, Jacobians, Hessians, convexity, conditioning, optimizer intuition
49. **Generalization and Evaluation** - Leakage, calibration, class imbalance, distribution shift, ablations, metric uncertainty
50. **ML Coding Interview Patterns** - Stable softmax, masking, vectorization, top-k/top-p, padding, k-means update templates
51. **LLM Research Interview Prep** - Perplexity, pass@k, retrieval metrics, ablation reasoning, paper discussion structure
52. **Statistical Learning Theory** - Empirical vs population risk, capacity, generalization gap, regularization as inductive bias
53. **ML Debugging and Mock Coding** - Timed coding prompts, NaN/debugging patterns, leakage checks, training failure diagnosis
54. **Data Manipulation for ML** - Pandas feature-table work, joins, groupby, normalization, preprocessing without leakage
55. **Research Papers and Mock Interviews** - Paper discussion prompts, research judgment, and probability questions like distribution membership
56. **Spoken Interview Question Bank** - Live-interview style model answers for ML theory, coding, probability, and LLM research questions
57. **Meta-Style Mock Interviews** - Full simulated technical loops, follow-ups, and scoring rubric
58. **Whiteboard Derivations** - Stepwise must-master derivations and memory skeletons
59. **Blind Coding Drills** - No-search, timed implementation drills from memory
60. **Research Judgment Rounds** - Scenario-based evidence review, ablations, and claim evaluation
61. **Large-Scale LLM Systems** - Training-memory, sharding, parallelism, serving, and scale trade-offs
62. **Frontier Training Playbook** - Methodology for architecture choices, stability, data mixture, and believable ablations
63. **Paged Attention and LLM Serving Internals** - KV-cache fragmentation, block tables, prefix sharing, continuous batching, and serving bottlenecks
64. **Integrated AI and ML Interview Synthesis** - Bridges across theory, coding, systems, and research judgment with answer frameworks

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
