# Multimodal Models & Embedding History — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

The history of representation learning is the spine of modern ML. From bag-of-words to CLIP to multimodal LLMs, each era solved a specific limitation of the previous one. Frontier interviews probe this evolution — partly because contrastive learning, image-text alignment, and multimodal models are *the* hot topics, and partly because understanding the lineage shows you can reason about *why* design choices win.

---

## 1. The lineage in one paragraph

Bag-of-words / TF-IDF (1970s–) → distributed word vectors (Word2Vec 2013, GloVe 2014) → contextual embeddings (ELMo 2018, BERT 2018) → sentence embeddings (Sentence-BERT 2019) → image-text contrastive (CLIP 2021) → multimodal LLMs (Flamingo 2022, LLaVA 2023, GPT-4V 2023, GPT-4o, Claude 3.5/Sonnet vision, Gemini 1.5/2.0/2.5, Llama 3.2 vision 2024+).

The driver: each stage handles *more context* and *more modalities* with the same fundamental idea — produce a vector that captures meaning.

---

## 2. Bag-of-words and TF-IDF

**Bag of words**: count word occurrences, ignore order. Vector dim = vocab size; sparse, mostly zeros.

**TF-IDF**: weight terms by inverse document frequency:

$$
\mathrm{TF\text{-}IDF}(t, d) = \mathrm{TF}(t, d) \cdot \log\frac{N}{\mathrm{DF}(t)}
$$

Common terms get downweighted; rare informative terms upweighted. Standard for information retrieval pre-2010s.

**Strengths**: cheap, interpretable, surprisingly strong baseline for retrieval (still used in BM25 / hybrid search).

**Weaknesses**: doesn't capture word similarity ("dog" and "puppy" are orthogonal vectors); doesn't model word order; no semantic generalization.

---

## 3. Word2Vec and GloVe — distributed word vectors

The shift: words become dense vectors in $\mathbb{R}^d$ ($d \sim 100$–$300$). Similar words get similar vectors.

### Word2Vec (Mikolov et al. 2013)

Two architectures:
- **CBOW** (Continuous Bag of Words): predict center word from surrounding context.
- **Skip-gram**: predict surrounding context from center word.

Trained with **negative sampling**: distinguish true (word, context) pairs from random negative pairs. Loss (NLL of binary classification with sigmoid):

$$
\mathcal{L} = -\sum_{(w, c) \in D^+} \log \sigma(v_w^\top v_c) - \sum_{(w, c) \in D^-} \log \sigma(-v_w^\top v_c)
$$

The famous demo: $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$. Linear arithmetic in semantic space.

### GloVe (Pennington et al. 2014)

Different motivation: directly factorize the word-context co-occurrence matrix $X$. Loss:

$$
\mathcal{L} = \sum_{i,j} f(X_{ij}) (v_i^\top v_j + b_i + b_j - \log X_{ij})^2
$$

with $f$ a weighting function. Works similarly to Word2Vec; sometimes a bit better on analogy tasks.

### Limitations
- One vector per word — can't handle polysemy ("bank" = river vs financial).
- Static — same embedding regardless of context.
- No phrase/sentence representation (averaging is a bad baseline).

---

## 4. Contextual embeddings — ELMo, BERT, GPT

The next leap: each token's embedding depends on its context.

### ELMo (Peters et al. 2018)

Bidirectional LSTM language model. Token embedding = concatenation of forward + backward hidden states from each layer, weighted-summed. Vastly better than Word2Vec on downstream tasks.

### BERT (Devlin et al. 2018)

Bidirectional transformer encoder. Pre-trained with:
- **Masked LM**: predict 15% randomly-masked tokens.
- **Next Sentence Prediction (NSP)**: binary classify if sentence B follows A. (Later research showed NSP doesn't help much; RoBERTa drops it.)

Fine-tune on downstream task. Dominated NLP from 2018–2020.

### GPT family (Radford et al. 2018+)

Causal (left-to-right) transformer decoder. Trained as autoregressive language model. Now dominant via in-context learning and instruction tuning (see LM losses deep dive).

### Why contextual matters

"The bank by the river" vs "The bank gave me a loan" → BERT gives different vectors for "bank" in each. Static embeddings (Word2Vec) cannot.

---

## 5. Sentence embeddings — making BERT useful for retrieval

Vanilla BERT outputs a vector per token. For retrieval / similarity, you need a single vector per sentence.

### Naive approach
Average BERT token embeddings. Surprisingly poor (worse than averaging GloVe!). BERT's token embeddings aren't trained to be similarity-friendly.

### Sentence-BERT (Reimers & Gurevych 2019)

Fine-tune BERT with **siamese architecture** + similarity loss on labeled pairs (e.g., NLI: entailment/contradiction/neutral). Output: sentence embedding suitable for cosine similarity.

### MS MARCO / dual-encoder retrievers

Train two-tower (query encoder + passage encoder) on (query, relevant passage) pairs from MS MARCO. Standard for dense retrieval.

### Modern sentence embedders
- E5 (2022): trained on weakly-supervised text pairs at scale.
- BGE (2023): high-performing open embedder.
- OpenAI text-embedding-3 (2024): commercial.

These power RAG retrieval, semantic search, classification.

---

## 6. CLIP — text-image contrastive learning

**CLIP** (Contrastive Language-Image Pretraining, Radford et al. 2021) was the watershed moment for multimodal.

### Architecture
- Image encoder: ViT or ResNet → embedding.
- Text encoder: transformer → embedding.
- Both project to shared $d$-dim space.

### Training

Given a batch of $N$ (image, caption) pairs:
1. Compute image embeddings $\{v_i\}$, text embeddings $\{u_i\}$.
2. Form $N \times N$ similarity matrix $S_{ij} = v_i^\top u_j / \tau$.
3. Loss: cross-entropy treating diagonals (true pairs) as positives:

$$
\mathcal{L} = -\frac{1}{2N}\sum_i \log \frac{\exp(S_{ii})}{\sum_j \exp(S_{ij})} - \frac{1}{2N}\sum_i \log \frac{\exp(S_{ii})}{\sum_j \exp(S_{ji})}
$$

(Symmetric: rank text against images, rank images against text.)

### Why CLIP was a big deal
- **Zero-shot classification**: classify image by computing similarity to text prompts ("a photo of a {class}"). No training on the classes.
- **Massive scale**: 400M (image, caption) pairs from the web. Cheap supervision.
- **Image-text aligned space**: enables search, generation conditioning, multi-modal reasoning.

### Variants
- **ALIGN** (Google 2021): same idea, even larger scale, noisier data.
- **OpenCLIP**: open replication.
- **EVA-CLIP**: scaling laws + better training recipes.
- **SigLIP** (Google 2023): replace softmax with sigmoid loss → faster, better at small batch.

---

## 7. InfoNCE — the contrastive loss in general

The CLIP loss is an instance of InfoNCE (van den Oord et al. 2018):

$$
\mathcal{L}_{\mathrm{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f(x, y^+)/\tau)}{\sum_{y \in \{y^+, y^-_1, \ldots, y^-_K\}} \exp(f(x, y)/\tau)}\right]
$$

One positive pair, $K$ negatives.

**InfoNCE is a lower bound on mutual information** between $x$ and $y^+$. Optimizing InfoNCE maximizes (a lower bound on) the dependency between paired views.

### Other contrastive setups
- **SimCLR** (Chen et al. 2020): two augmentations of the same image as positive pair. Self-supervised image rep learning.
- **MoCo** (He et al. 2020): negative bank with momentum encoder. Memory-efficient.
- **DINO / DINOv2**: self-distillation on vision transformers; non-contrastive but related.
- **Sentence-BERT**: NLI-pair contrastive in NLP.
- **CodeContrast / CodeSage**: contrastive on code.

The recipe is the same: define positive pairs (semantically equivalent / paired across modalities), pull them together, push everything else apart.

---

## 8. Multimodal LLMs — Flamingo, LLaVA, modern frontier

### Flamingo (DeepMind 2022)

Frozen LLM + vision encoder. New gated cross-attention layers interleaved into the LLM, attending to image features. Trained on web-scraped image-text data.

Key idea: don't retrain the LLM; *augment* it with vision capability.

### LLaVA (Liu et al. 2023+)

Simpler recipe:
1. Vision encoder (CLIP ViT-L) outputs image patch embeddings.
2. Linear projection (or MLP) maps them into LLM's input embedding space.
3. Concatenate with text tokens; feed to LLM.
4. Train projection + LLM (lightly) on instruction-tuned image-text data.

LLaVA showed you don't need exotic architectures — a projection layer plus standard CLIP + LLM is enough.

### Frontier multimodal LLMs (2023+)

GPT-4V (2023) → GPT-4o (2024, native multimodal). Claude 3 Opus / 3.5 Sonnet vision. Gemini 1.5 Pro → 2.0 / 2.5. Llama 3.2 vision (2024, open). Architectures vary but follow the general pattern: vision encoder produces tokens that go into an LLM-style decoder. The 2024+ frontier shifted toward natively multimodal pre-training rather than vision-bolted-on.

### Native multimodal training

Newer models (Gemini 1.5+, GPT-4o) train on multiple modalities from scratch, not bolting vision onto a text LLM. Tokenize images directly into the same space as text. Better cross-modal reasoning.

### Audio and beyond

- Whisper (OpenAI 2022): speech-to-text, encoder-decoder transformer.
- AudioLM, MusicLM: tokenize audio for generative LLMs.
- Multimodal foundation models increasingly handle text + image + audio + video natively.

---

## 9. Embeddings in production — vector search

Embeddings live or die by retrieval quality and speed.

### Approximate nearest neighbor (ANN)
- **Flat**: brute force, exact, slow for large $N$.
- **IVF (inverted file index)**: cluster vectors; search nearest clusters. Trade recall for speed.
- **HNSW** (Hierarchical Navigable Small World): graph-based ANN. State of the art for many use cases.
- **Product Quantization (PQ)**: compress vectors via per-subspace quantization. Memory-efficient.

### Vector databases
Pinecone, Weaviate, Qdrant, Milvus, FAISS. All implement ANN + metadata filters + persistence.

### Hybrid search (BM25 + dense)
Combine sparse (lexical) and dense (semantic) retrieval. Dense catches paraphrases; sparse catches rare entities/terms.

---

## 10. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Word2Vec architecture? | Transformer | Shallow neural net (1 hidden layer); skip-gram or CBOW |
| Why average BERT to get sentence embedding? | Standard | Often poor; use Sentence-BERT or train with siamese setup |
| CLIP loss? | Cross-entropy | Symmetric InfoNCE on $N \times N$ batch similarity matrix |
| Zero-shot classification with CLIP? | Magic | Score image vs text prompts ("a photo of a {class}"); pick max |
| Multimodal LLM = LLM with vision encoder? | Yes | Almost — need projection / cross-attention to align modalities |
| InfoNCE optimizes? | Cosine similarity | Lower bound on mutual information |
| Hybrid retrieval — what fuses? | Random | Rank fusion (RRF) or score combination |

---

## 11. Eight most-asked interview questions

1. **Walk me through the history from BoW to CLIP.** (Use the lineage; emphasize what each era fixed.)
2. **Word2Vec skip-gram — what's the loss?** (Negative sampling logistic.)
3. **Why does averaging BERT tokens fail for similarity?** (Token embeddings aren't trained for similarity; need siamese-style fine-tuning.)
4. **CLIP training loss — derive it.** (Symmetric softmax over $N \times N$ similarity matrix.)
5. **What's InfoNCE and why does it work?** (Cross-entropy with one positive vs $K$ negatives; lower bound on MI.)
6. **How does CLIP do zero-shot classification?** (Compute similarity to text prompts of class names.)
7. **LLaVA architecture in 1 minute.** (Vision encoder → projection → LLM. Minimal new params.)
8. **You need to retrieve from 100M docs — design.** (Embed with strong sentence encoder; HNSW/IVF index; possibly hybrid with BM25; reranker on top-K.)

---

## 12. Drill plan

- For each era (BoW, Word2Vec, BERT, Sentence-BERT, CLIP, multimodal LLM), recite: idea, what it fixed.
- Write CLIP loss on paper from scratch.
- Recite InfoNCE formula + interpretation.
- Sketch LLaVA architecture in 30 seconds.
- For "design a search system," use embeddings + ANN + hybrid + reranker; recite in 3 minutes.

---

## 13. Further reading

- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space.*
- Pennington, Socher, Manning (2014), *GloVe: Global Vectors for Word Representation.*
- Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers.*
- Reimers & Gurevych (2019), *Sentence-BERT.*
- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).
- van den Oord et al. (2018), *Representation Learning with Contrastive Predictive Coding* (InfoNCE).
- Alayrac et al. (2022), *Flamingo: a Visual Language Model for Few-Shot Learning.*
- Liu et al. (2023), *Visual Instruction Tuning* (LLaVA).
- Zhai et al. (2023), *Sigmoid Loss for Language Image Pre-Training* (SigLIP).
