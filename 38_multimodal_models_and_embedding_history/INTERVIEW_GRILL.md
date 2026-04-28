# Multimodal & Embedding History — Interview Grill

> 45 questions on the embedding lineage, contrastive learning, CLIP, multimodal LLMs. Drill until you can answer 30+ cold.

---

## A. Bag of words and TF-IDF

**1. TF-IDF formula?**
$\mathrm{TF}(t, d) \cdot \log(N/\mathrm{DF}(t))$. Term frequency times inverse document frequency.

**2. Why IDF?**
Downweights common terms ("the", "is") so that rare informative terms dominate similarity.

**3. BoW limitation?**
"Dog" and "puppy" are orthogonal — no semantic generalization. No word order.

**4. Is TF-IDF still used?**
Yes — BM25 (a TF-IDF refinement) is a strong sparse retrieval baseline, often combined with dense embeddings in hybrid search.

---

## B. Word2Vec / GloVe

**5. Word2Vec — two architectures?**
CBOW (predict center from context) and Skip-gram (predict context from center).

**6. Skip-gram with negative sampling — loss?**
$\mathcal{L} = -\sum_{(w,c) \in D^+} \log \sigma(v_w^\top v_c) - \sum_{(w,c) \in D^-} \log \sigma(-v_w^\top v_c)$. Minimize NLL of binary "true vs noise" classification.

**7. Why negative sampling vs full softmax?**
Full softmax over vocabulary is expensive ($O(V)$ per step). Negative sampling: a few negatives per positive.

**8. Famous Word2Vec arithmetic?**
$v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$. Linear analogy works.

**9. GloVe — what does it factorize?**
Co-occurrence matrix log values. Weighted least-squares on $v_i^\top v_j + b_i + b_j \approx \log X_{ij}$.

**10. Word2Vec / GloVe limitation?**
Static embeddings. One vector per word — can't handle polysemy.

---

## C. Contextual embeddings

**11. ELMo architecture?**
Bidirectional LSTM language model. Token rep = weighted sum of forward + backward hidden states.

**12. BERT pre-training objectives?**
Masked Language Modeling (MLM) + Next Sentence Prediction (NSP).

**13. BERT's MLM masking ratio?**
15% of tokens masked. Of those: 80% [MASK], 10% random token, 10% unchanged.

**14. Why was NSP later dropped (RoBERTa)?**
Found to not help much; harder to train; extra complexity not worth it.

**15. Encoder-only vs decoder-only vs encoder-decoder?**
Encoder-only (BERT): bidirectional, for understanding. Decoder-only (GPT): causal, for generation. Encoder-decoder (T5): seq-to-seq.

---

## D. Sentence embeddings

**16. Why does averaging BERT token embeddings give bad similarity?**
Token embeddings aren't trained for sentence-level similarity. Variance is in different directions.

**17. Sentence-BERT idea?**
Fine-tune BERT siamese with similarity loss (e.g., NLI) so output is similarity-ready.

**18. Two-tower retriever training?**
(Query, positive passage, negative passages). Contrastive loss on cosine similarity. MS MARCO is the canonical dataset.

**19. Modern sentence embedders?**
E5, BGE, GTE, OpenAI text-embedding-3, Cohere embed. All trained on web-scale weakly-supervised pairs.

---

## E. Contrastive learning

**20. InfoNCE formula?**
$-\log \frac{\exp(s(x, y^+)/\tau)}{\sum_y \exp(s(x, y)/\tau)}$ over $K+1$ candidates ($1$ positive, $K$ negatives).

**21. What does InfoNCE optimize?**
Lower bound on mutual information $I(x; y^+)$.

**22. Role of temperature $\tau$?**
Sharpens or smooths the softmax. Lower $\tau$ → harder negatives matter more. Empirically tuned.

**23. SimCLR — what are the positive pairs?**
Two augmentations of the same image. Self-supervised.

**24. MoCo — what's the trick?**
Maintain a queue of negative samples encoded by a momentum-updated encoder. Memory-efficient large negative pool.

---

## F. CLIP

**25. CLIP architecture?**
Image encoder + text encoder, both projecting to a shared $d$-dim space. Trained contrastively on (image, caption) pairs.

**26. CLIP loss?**
Symmetric InfoNCE on $N \times N$ batch similarity matrix. Diagonals are positives.

**27. Why symmetric (image-to-text + text-to-image)?**
Both directions of retrieval matter. Without symmetry, training is biased toward one direction.

**28. CLIP scale?**
400M (image, text) pairs from the web. Cheap supervision.

**29. Zero-shot classification with CLIP?**
Compute text embeddings of "a photo of a {class}" for each class. Pick class whose embedding is most similar to the image embedding.

**30. CLIP weaknesses?**
Weak at OCR/text in images, fine-grained categories, compositional reasoning. Bias inherited from web data.

**31. SigLIP improvement?**
Replace softmax with sigmoid loss. Each pair labeled independently positive/negative. Faster, scales better at small batch.

---

## G. Multimodal LLMs

**32. Flamingo's key idea?**
Frozen LLM + vision encoder + new gated cross-attention layers. Don't retrain the LLM; augment it.

**33. LLaVA architecture in 1 sentence?**
CLIP ViT image encoder → linear projection → concatenated with text tokens → fed to LLM.

**34. LLaVA training stages?**
Stage 1: train projection only on caption data (alignment). Stage 2: instruction tune on image-text instruction data.

**35. Native multimodal vs bolted-on?**
Native (Gemini 1.5+, GPT-4o): trained from scratch on multiple modalities. Bolted-on (LLaVA, early Flamingo): vision adapter on top of pre-trained LLM. Native generalizes better but expensive.

**36. How are images "tokenized" for LLMs?**
ViT-style patch embeddings, optionally compressed via Q-former or perceiver to fewer tokens. Then treated as a sequence of "image tokens" in the LLM's input.

---

## H. Vector search / retrieval

**37. ANN vs exact KNN?**
ANN trades small recall loss for huge speedup. Exact $O(N)$, ANN can be sub-linear.

**38. HNSW — what is it?**
Hierarchical Navigable Small World graph. Multi-layer graph; greedy search at each layer. State-of-the-art for many ANN workloads.

**39. IVF-PQ?**
Inverted File Index + Product Quantization. Cluster vectors (IVF), then quantize each cluster's residuals (PQ). Memory- and speed-efficient.

**40. Hybrid search components?**
Sparse (BM25) + dense (embedding) retrieval. Combined via score blending or rank fusion (RRF).

**41. Why hybrid?**
Dense catches paraphrases; sparse catches rare exact terms (proper nouns, IDs). Together more robust.

---

## I. Subtleties

**42. Cosine similarity vs dot product?**
Cosine: normalized; magnitude doesn't matter. Dot product: unnormalized; can be useful when magnitude carries information.

**43. Why $\ell_2$-normalize embeddings before retrieval?**
Cosine similarity = dot product after normalization. Makes search uniform; avoids dominant high-magnitude vectors.

**44. Embedding space anisotropy?**
Pre-trained embeddings often cluster in a narrow cone. Reduces effective dimensionality. Whitening / contrastive fine-tuning can fix.

**45. Catastrophic forgetting in vision-language fine-tuning?**
Fine-tuning a multimodal model on a narrow task can wipe out general capabilities. LoRA / adapter methods mitigate.

---

## Quick fire

**46.** *Word2Vec architecture?* Shallow NN, not transformer.
**47.** *BERT pretraining?* MLM (+ NSP, later dropped).
**48.** *CLIP loss?* Symmetric InfoNCE.
**49.** *CLIP zero-shot?* Compare to text class prompts.
**50.** *InfoNCE bound?* Lower bound on MI.
**51.** *LLaVA = ?* Vision encoder + projection + LLM.
**52.** *HNSW = ?* Graph-based ANN.
**53.** *Hybrid search = ?* Sparse + dense.
**54.** *SigLIP vs CLIP?* Sigmoid replaces softmax.
**55.** *Sentence-BERT improvement?* Siamese fine-tune for similarity.

---

## Self-grading

If you can't answer 1-15, you don't know embedding history. If you can't answer 16-30, you'll struggle on contrastive / CLIP questions. If you can't answer 31-45, frontier-lab multimodal interviews will go past you.

Aim for 35+/55 cold.
