# Multimodal Models & Embedding History — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

The history of representation learning is the spine of modern ML. From bag-of-words to CLIP to multimodal LLMs, each era solved a specific limitation of the previous one. Frontier interviews probe this evolution — partly because contrastive learning, image-text alignment, and multimodal models are *the* hot topics, and partly because understanding the lineage shows you can reason about *why* design choices win.

---

## 1. The lineage in one paragraph

Bag-of-words / TF-IDF (1970s–) → distributed word vectors (Word2Vec 2013, GloVe 2014) → contextual embeddings (ELMo 2018, BERT 2018) → sentence embeddings (Sentence-BERT 2019) → image-text contrastive (CLIP 2021) → multimodal LLMs (Flamingo 2022, LLaVA 2023, GPT-4V 2023, GPT-4o, Claude 3.5/Sonnet vision, Gemini 1.5/2.0/2.5, Llama 3.2 vision 2024+).

The driver: each stage handles *more context* and *more modalities* with the same fundamental idea — produce a vector that captures meaning.

> **Saying it out loud.** The whole history is one idea getting progressively less crude: turn meaning into a vector. Bag of words gave every word its own dimension, so nothing was similar to anything. Word2Vec made the vectors dense so similar words got similar vectors, but each word got exactly one vector forever. BERT made the vector depend on the sentence, so "bank" finally had two meanings. Sentence-BERT made it work at the sentence level, CLIP put images and text in the same space, and multimodal LLMs put that space inside a generative model. Each step fixed the specific thing the previous one couldn't do, and the through-line is that the amount of context folded into a single vector kept growing.

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

> **Saying it out loud.** Bag of words counts words and throws away everything else. TF-IDF makes that useful by weighting each count by how rare the word is across the corpus, so "the" contributes nothing and "photosynthesis" contributes a lot, with a log to keep the rare-word weights from exploding. It's cheap, it's interpretable, and BM25 — the refined version — is still a hard baseline in retrieval today. The two named weaknesses are that "dog" and "puppy" are perfectly orthogonal, so there's zero semantic generalization, and word order is gone, so "dog bites man" and "man bites dog" are literally the same vector.

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

> **Saying it out loud.** Word2Vec's move was to stop treating words as IDs and start treating them as points in a few hundred dimensions, learned from the company they keep. Skip-gram with negative sampling is the workhorse: given a word, push its vector toward the words that actually appeared near it and away from randomly sampled words, which turns an expensive softmax over the vocabulary into cheap binary classification with maybe five negatives. GloVe gets to a similar place by directly fitting dot products to log co-occurrence counts over the whole corpus at once. Both produce the king-minus-man-plus-woman geometry. And both share the fatal limitation: one vector per word, permanently, so "bank" has a single blurry embedding that averages the river and the money.

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

> **Saying it out loud.** Contextual embeddings mean the representation is computed from the sentence rather than looked up in a table. ELMo did it with two LSTMs, one running each direction, and combined their hidden states. BERT did it properly with a transformer encoder and masked language modeling — blank out 15% of tokens and predict them using both sides at once, which is what makes the representation deeply bidirectional. GPT went the other way with causal masking, which is worse for understanding tasks and is the only thing that makes generation possible. The concrete payoff: "the bank by the river" and "the bank gave me a loan" now produce genuinely different vectors for the same word, which no static embedding can do.

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

> **Saying it out loud.** BERT gives you a vector per token, but retrieval needs one vector per sentence, and the obvious fix — averaging the token vectors — works badly. It's actually worse than averaging GloVe vectors, which is the embarrassing result that started this line of work. The reason is that nothing in masked language modeling ever asked for the geometry to encode sentence similarity. Sentence-BERT fixes it by training for the thing you want: run two copies of BERT with shared weights over a sentence pair and use a loss that pulls similar pairs together. The huge practical win is precomputation — you embed each document once and compare with a dot product, so scoring ten thousand candidates is ten thousand dot products instead of ten thousand transformer forward passes.

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

> **Saying it out loud.** CLIP is two encoders, one for images and one for text, trained so that a picture and its own caption land near each other in a shared space and far from every other caption in the batch. You build an N-by-N similarity matrix, the diagonal is correct, and you run cross-entropy across rows and columns. What made it a watershed wasn't the loss, it was the data — four hundred million image-caption pairs scraped off the web, which is free supervision over an open-ended vocabulary instead of a thousand fixed ImageNet labels. That's what buys zero-shot classification: embed "a photo of a {class}" for whatever classes you invent today and take the nearest. The tradeoff is batch size — with softmax, difficulty scales with batch, and CLIP needed 32,768, which is why SigLIP's sigmoid variant matters for anyone without a warehouse of GPUs.

---

## 7. InfoNCE — the contrastive loss in general

*In plain language:* InfoNCE is the general form of the loss CLIP uses. It's a multiple-choice question: here's an anchor, here's one correct partner and $K$ wrong ones, pick the right one. The formula below is ordinary softmax cross-entropy where the "classes" are candidate partners and the logits are similarities divided by a temperature.

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

> **Saying it out loud.** InfoNCE is the general recipe underneath all of this, and it's just multiple choice. Take an anchor, take one true partner and a pile of distractors, score all of them, and use cross-entropy to make the true one win. What changes between methods is only where the positive pair comes from: two augmentations of the same photo gives you SimCLR, an image and its caption gives you CLIP, a sentence and its entailment gives you Sentence-BERT. Formally it maximizes a lower bound on the mutual information between the two views. The number worth naming is that the bound is capped at $\log K$ for $K$ negatives, so with 256 negatives you can't certify more than about 5.5 nats — which is exactly why everyone fights for bigger batches and negative queues.

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

> **Saying it out loud.** There are really two designs. Flamingo's is "don't touch the language model" — freeze the LLM, freeze the vision encoder, and insert new gated cross-attention layers between them, with gates initialized at zero so the model starts out behaving exactly like the original LLM. LLaVA's is even simpler and is what most people copy: run CLIP over the image, push the patch features through one projection matrix so they look like word embeddings, and paste them into the token sequence. The LLM doesn't know some of its tokens came from pixels. Frontier models have since moved toward native multimodal pretraining, where images and audio are in the mix from step one, which reasons across modalities much better. The tradeoff is blunt: an adapter costs a day on a few GPUs, native pretraining is a full frontier run you can't redo.

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

> **Saying it out loud.** In production, an embedding is only as good as the index around it. Brute force is exact but linear, so past a few million vectors you switch to approximate search — HNSW if you want the best recall-latency curve and can afford the memory, IVF-PQ if you need to fit a billion vectors in RAM, since product quantization takes a 3-kilobyte float vector down to about 64 bytes. Then you almost always run BM25 alongside and fuse the rankings, because dense retrieval handles paraphrase and sparse retrieval handles exact strings like part numbers and error codes, and those failure modes barely overlap. Finish with a cross-encoder reranker over the top 50 or 100. The tradeoff to name at every stage is recall versus latency — ANN recall is a tunable knob, typically set around 95 to 99 percent.

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

> **Saying it out loud.** The traps here are mostly about vagueness. Word2Vec is a shallow network, not a transformer — it predates transformers by four years. Averaging BERT tokens isn't the standard way to get a sentence embedding, it's the naive way, and it underperforms averaged GloVe. CLIP's loss isn't just "cross-entropy," it's *symmetric* InfoNCE over the full batch similarity matrix, both directions. InfoNCE isn't optimizing cosine similarity, it's bounding mutual information. And hybrid search doesn't fuse by averaging raw scores, since BM25 and cosine live on incomparable scales — it's reciprocal rank fusion, which uses ranks and ignores the scores entirely.

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
