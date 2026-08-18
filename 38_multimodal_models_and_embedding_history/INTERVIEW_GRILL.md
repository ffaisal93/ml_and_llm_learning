# Multimodal & Embedding History — Interview Grill

> 45 questions on the embedding lineage, contrastive learning, CLIP, multimodal LLMs. Drill until you can answer 30+ cold.

---

## A. Bag of words and TF-IDF

**1. TF-IDF formula?**
$\mathrm{TF}(t, d) \cdot \log(N/\mathrm{DF}(t))$. Term frequency times inverse document frequency.

> **Saying it out loud.** TF-IDF scores a word in a document by how often it shows up there, discounted by how common it is everywhere else. So "the" appears constantly and gets crushed to nearly nothing, while "photosynthesis" appearing three times is a strong signal about what the document is about. The log on the inverse document frequency is there to compress the range so a single ultra-rare word doesn't swamp the score. It's a purely lexical measure, though — it has no idea "car" and "automobile" are related, which is exactly the gap embeddings filled.

**2. Why IDF?**
Downweights common terms ("the", "is") so that rare informative terms dominate similarity.

> **Saying it out loud.** IDF exists because raw counts are dominated by words that carry no information. Every document contains "the" and "of," so matching on them tells you nothing about relevance; what distinguishes documents is the rare stuff. So you weight each term by the log of how few documents contain it, and the boring words fall out of the score almost entirely. The failure mode to name is the opposite extreme: a typo or a one-off token appearing in exactly one document gets an enormous IDF, which is why real systems floor it or smooth it.

**3. BoW limitation?**
"Dog" and "puppy" are orthogonal — no semantic generalization. No word order.

> **Saying it out loud.** Bag of words has two holes and both are fatal for meaning. First, every word is its own dimension, so "dog" and "puppy" are geometrically as unrelated as "dog" and "spreadsheet" — there's no notion that words can be similar. Second, order is thrown away entirely, so "dog bites man" and "man bites dog" produce identical vectors. It's still surprisingly strong for topic-level tasks, but anything requiring paraphrase understanding needs a dense representation.

**4. Is TF-IDF still used?**
Yes — BM25 (a TF-IDF refinement) is a strong sparse retrieval baseline, often combined with dense embeddings in hybrid search.

> **Saying it out loud.** Absolutely, and anyone who says otherwise hasn't shipped a search system. BM25, which is TF-IDF with saturation and length normalization added, is still a brutally hard baseline on keyword queries, exact product codes, names, and error messages. Dense embeddings beat it on paraphrase and conceptual queries and lose to it on rare literal strings. That's why production RAG runs both and fuses the rankings — the two failure modes barely overlap.

---

## B. Word2Vec / GloVe

**5. Word2Vec — two architectures?**
CBOW (predict center from context) and Skip-gram (predict context from center).

> **Saying it out loud.** Word2Vec has two flavors that are mirror images of each other. CBOW takes the surrounding words and tries to predict the missing middle word; skip-gram takes the middle word and tries to predict its neighbors. CBOW is faster because it averages the context into one prediction, but skip-gram works better on rare words since every occurrence generates several training pairs. In practice skip-gram with negative sampling is what people actually used.

**6. Skip-gram with negative sampling — loss?**
$\mathcal{L} = -\sum_{(w,c) \in D^+} \log \sigma(v_w^\top v_c) - \sum_{(w,c) \in D^-} \log \sigma(-v_w^\top v_c)$. Minimize NLL of binary "true vs noise" classification.

> **Saying it out loud.** The clever move in skip-gram with negative sampling is turning a giant prediction problem into a tiny yes-or-no problem. Instead of "which of fifty thousand words comes next," you ask "did this word-context pair really occur, or did I make it up by sampling randomly?" Real pairs get pushed toward high dot product, fake pairs toward low, and it's just logistic regression on dot products. The number that matters is the negative count — typically five to twenty negatives per positive, which is what makes it cheap.

**7. Why negative sampling vs full softmax?**
Full softmax over vocabulary is expensive ($O(V)$ per step). Negative sampling: a few negatives per positive.

> **Saying it out loud.** A full softmax has to normalize over the entire vocabulary at every single training step, so that's fifty thousand dot products and exponentials per word you look at. Negative sampling replaces that with a handful of sampled words — maybe five — and turns the normalization into independent binary decisions. You lose the exact probabilistic interpretation, and you gain roughly a thousandfold speedup, which is the trade that made training on billions of tokens possible in the first place.

**8. Famous Word2Vec arithmetic?**
$v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$. Linear analogy works.

> **Saying it out loud.** The famous one is king minus man plus woman lands near queen, and it's genuinely striking because nobody trained for it. It happens because the objective is built on dot products, so consistent co-occurrence differences — the gender difference, the plural difference, the country-capital difference — end up as roughly consistent directions in the space. Worth being a little skeptical out loud, though: the evaluation usually excludes the input words from the nearest-neighbor search, and analogies outside a few clean relation types are much shakier than the demo suggests.

**9. GloVe — what does it factorize?**
Co-occurrence matrix log values. Weighted least-squares on $v_i^\top v_j + b_i + b_j \approx \log X_{ij}$.

> **Saying it out loud.** GloVe attacks the same problem from the global side instead of the streaming side. You build the full word-by-word co-occurrence count matrix over the corpus once, then fit vectors so that the dot product of two word vectors, plus a couple of bias terms, reproduces the log of how often they co-occurred. The weighting function is the practical detail — it damps the influence of extremely frequent pairs so "of the" doesn't dominate the fit. Same quality ballpark as Word2Vec; different route.

**10. Word2Vec / GloVe limitation?**
Static embeddings. One vector per word — can't handle polysemy.

> **Saying it out loud.** Both give you exactly one vector per word for all time, which means "bank" gets a single embedding that's some blurry average of riverbank and financial institution. There's no mechanism to disambiguate, because the model never sees the sentence at inference — it just looks the word up in a table. That one limitation is the entire motivation for ELMo and BERT: make the representation a function of the context rather than a lookup.

---

## C. Contextual embeddings

**11. ELMo architecture?**
Bidirectional LSTM language model. Token rep = weighted sum of forward + backward hidden states.

> **Saying it out loud.** ELMo was the first big "your embedding should depend on the sentence" model. It trains two LSTM language models, one reading left to right and one right to left, and a word's representation is a learned weighted combination of the hidden states from every layer of both. So "bank" in a river sentence and "bank" in a finance sentence get genuinely different vectors. The limitation that BERT fixed is that the two directions are trained separately and only concatenated at the end, so it isn't truly jointly bidirectional.

**12. BERT pre-training objectives?**
Masked Language Modeling (MLM) + Next Sentence Prediction (NSP).

> **Saying it out loud.** BERT trains on two tasks at once. Masked language modeling blanks out some tokens and asks the model to fill them in using both left and right context, which is what makes the representation deeply bidirectional. Next sentence prediction shows the model two segments and asks whether the second actually followed the first, meant to teach discourse-level relationships. The first objective is the one that carried the weight — NSP turned out to be nearly useless and later models dropped it.

**13. BERT's MLM masking ratio?**
15% of tokens masked. Of those: 80% [MASK], 10% random token, 10% unchanged.

> **Saying it out loud.** Fifteen percent of tokens get selected, and then there's a wrinkle people forget: of those, only 80% actually become the mask token, 10% are swapped for a random word, and 10% are left alone. The reason for that split is a train-test mismatch. The `[MASK]` token never appears at fine-tuning time, so if the model only ever predicted at mask positions it would learn to ignore everything else. Corrupting and leaving some untouched forces it to build a good representation of every token.

**14. Why was NSP later dropped (RoBERTa)?**
Found to not help much; harder to train; extra complexity not worth it.

> **Saying it out loud.** RoBERTa's ablations showed NSP just wasn't earning its slot. The task was too easy — the negative examples came from entirely different documents, so the model could solve it by topic matching alone without learning anything about discourse coherence. Dropping it and training longer on more data with dynamic masking gave a clearly better model. It's a nice example of the general lesson that a pretraining objective is only useful if it's hard in the right way.

**15. Encoder-only vs decoder-only vs encoder-decoder?**
Encoder-only (BERT): bidirectional, for understanding. Decoder-only (GPT): causal, for generation. Encoder-decoder (T5): seq-to-seq.

> **Saying it out loud.** It comes down to what each token is allowed to see. An encoder lets every token attend to every other token in both directions, which is great for understanding but means you can't generate, since predicting the next word would let you peek at it. A decoder uses causal masking so each token only sees the past, which makes generation natural. Encoder-decoder does both, encoding the input bidirectionally and decoding causally. The industry ended up on decoder-only mostly because one stack that does everything scales better and lets you frame every task as text generation.

---

## D. Sentence embeddings

**16. Why does averaging BERT token embeddings give bad similarity?**
Token embeddings aren't trained for sentence-level similarity. Variance is in different directions.

> **Saying it out loud.** Because nothing in BERT's training ever asked for it. Masked language modeling optimizes token-level prediction, so the geometry of the space is organized around predicting words, not around whether two sentences mean the same thing. Averaging those vectors gives you something that's dominated by frequency effects and sits in a narrow cone where almost everything looks similar to everything else. Concretely, averaged BERT embeddings score worse on sentence similarity benchmarks than plain averaged GloVe vectors, which is the embarrassing result that motivated Sentence-BERT.

**17. Sentence-BERT idea?**
Fine-tune BERT siamese with similarity loss (e.g., NLI) so output is similarity-ready.

> **Saying it out loud.** Sentence-BERT's fix is simple: if you want cosine similarity to mean something, train for cosine similarity. You run two copies of BERT with shared weights over a sentence pair, pool to one vector each, and train with a loss that pulls entailment pairs together and pushes contradictions apart, using NLI data. The huge practical payoff is that you can now embed every sentence once and compare with a dot product. Cross-encoder BERT is more accurate but requires a forward pass per pair, so comparing ten thousand sentences goes from about fifty million forward passes to ten thousand.

**18. Two-tower retriever training?**
(Query, positive passage, negative passages). Contrastive loss on cosine similarity. MS MARCO is the canonical dataset.

> **Saying it out loud.** Two-tower means one encoder for queries and one for passages, trained so a query's vector lands near its correct passage and far from everything else. The training signal is contrastive: one positive, many negatives, softmax over similarities. The thing that actually determines quality is where the negatives come from — random negatives are too easy and the model plateaus, so you mine hard negatives, passages that BM25 or an earlier checkpoint ranked highly but that are actually wrong. The architectural tradeoff is that the two towers never see each other, so all the interaction is a single dot product, which is why you re-rank the top 100 with a cross-encoder.

**19. Modern sentence embedders?**
E5, BGE, GTE, OpenAI text-embedding-3, Cohere embed. All trained on web-scale weakly-supervised pairs.

> **Saying it out loud.** The current crop — E5, BGE, GTE on the open side, plus the OpenAI and Cohere hosted models — all follow the same recipe: pretrain contrastively on enormous quantities of weakly paired web text like title-body or question-answer pairs, then fine-tune on curated retrieval data. Two things distinguish them practically. Most use instruction prefixes, so you literally prepend "query:" or "passage:" and forgetting that quietly wrecks your recall. And dimensionality is a real cost decision — some support Matryoshka truncation so you can drop from 1536 down to 256 dimensions and keep most of the quality at a fraction of the index size.

---

## E. Contrastive learning

**20. InfoNCE formula?**
$-\log \frac{\exp(s(x, y^+)/\tau)}{\sum_y \exp(s(x, y)/\tau)}$ over $K+1$ candidates ($1$ positive, $K$ negatives).

> **Saying it out loud.** InfoNCE is just cross-entropy over similarities. You take one anchor, one correct match, and a pile of wrong ones, score them all, divide by a temperature, and softmax. The correct one should get the probability mass. That's it — it's a classification problem where the classes are "which of these candidates is the real partner." The reason it dominates representation learning is that the labels are free: you construct positives from structure in the data rather than annotation.

**21. What does InfoNCE optimize?**
Lower bound on mutual information $I(x; y^+)$.

> **Saying it out loud.** Formally, minimizing InfoNCE maximizes a lower bound on the mutual information between the two views. Intuitively, that means the representation is being pushed to keep whatever information the two views share and throw away whatever's private to each. The catch worth naming is that the bound is capped by the log of the number of negatives, so with 256 negatives you can't certify more than about 5.5 nats of mutual information no matter how good the model is — which is one reason people push batch sizes so high.

**22. Role of temperature $\tau$?**
Sharpens or smooths the softmax. Lower $\tau$ → harder negatives matter more. Empirically tuned.

> **Saying it out loud.** Temperature controls how peaky the softmax over similarities is, and it matters much more than people expect. A low temperature sharpens the distribution, so the loss is dominated by the hardest negatives — the ones already close to the anchor — which gives you tight, well-separated clusters but can be unstable and can punish false negatives brutally. A high temperature spreads the gradient across all negatives and gives a smoother, more uniform space. CLIP made it a learned parameter rather than a hyperparameter, and it converges to around 0.01, which is very sharp.

**23. SimCLR — what are the positive pairs?**
Two augmentations of the same image. Self-supervised.

> **Saying it out loud.** SimCLR builds positives out of thin air: take one image, apply two different random augmentations — crop, color jitter, blur — and declare those two views a matching pair. Everything else in the batch is a negative. No labels anywhere. The finding that surprised people is how much the augmentation choice matters: random cropping combined with color distortion is essential, and if you drop the color jitter the model cheats by matching color histograms instead of learning content. It also needs enormous batches, on the order of 4096, because the batch is your entire negative pool.

**24. MoCo — what's the trick?**
Maintain a queue of negative samples encoded by a momentum-updated encoder. Memory-efficient large negative pool.

> **Saying it out loud.** MoCo's insight is that you shouldn't need a giant batch just to get giant numbers of negatives. So it keeps a queue of previously computed embeddings — tens of thousands of them — and uses those as negatives, which decouples the negative pool from the batch size. The problem that creates is staleness: those queued vectors came from an older version of the encoder. The fix is the momentum encoder, a slowly moving average of the main encoder, typically with momentum 0.999, so representations drift slowly enough that old queue entries stay consistent.

---

## F. CLIP

**25. CLIP architecture?**
Image encoder + text encoder, both projecting to a shared $d$-dim space. Trained contrastively on (image, caption) pairs.

> **Saying it out loud.** CLIP is two separate encoders, one for pixels and one for text, that both dump into the same vector space. Nothing connects them architecturally — the only thing tying them together is the training objective, which says an image and its own caption should land near each other and far from every other caption in the batch. Once that's true, you can compare an image to a sentence with a dot product. That shared space is the whole product; everything CLIP does downstream falls out of it.

**26. CLIP loss?**
Symmetric InfoNCE on $N \times N$ batch similarity matrix. Diagonals are positives.

> **Saying it out loud.** You take a batch of image-caption pairs, embed everything, and build an N-by-N matrix of similarities. The diagonal is the correct pairings and everything off-diagonal is wrong. Then you run cross-entropy across each row and across each column and average the two. So it's a classification problem with the batch as the label set, which is why batch size directly determines difficulty — CLIP trained at 32,768, meaning each image had over thirty thousand distractors.

**27. Why symmetric (image-to-text + text-to-image)?**
Both directions of retrieval matter. Without symmetry, training is biased toward one direction.

> **Saying it out loud.** Because retrieval goes both ways and one-directional training gives you a lopsided space. If you only train image-to-text, the model learns to find the right caption for an image but the reverse search degrades. Averaging both directions forces the embedding to be a genuinely shared space rather than one modality mapped onto the other. It costs nothing — you already have the full similarity matrix, so it's a second softmax over the transpose.

**28. CLIP scale?**
400M (image, text) pairs from the web. Cheap supervision.

> **Saying it out loud.** Four hundred million image-text pairs scraped from the web, which is the actual innovation. The point isn't the architecture — contrastive learning existed. The point is that alt-text and captions are free supervision sitting on the open internet, so you get a supervision signal covering a vast, open-ended set of concepts instead of ImageNet's thousand fixed classes. That's why CLIP can zero-shot categories nobody labeled. The cost is that you inherit all of the web's biases and noise, unfiltered.

**29. Zero-shot classification with CLIP?**
Compute text embeddings of "a photo of a {class}" for each class. Pick class whose embedding is most similar to the image embedding.

> **Saying it out loud.** Zero-shot classification is a trick that follows straight from the shared space. You write out your class names as sentences — "a photo of a golden retriever," "a photo of a tabby cat" — embed those with the text encoder, embed the image, and pick whichever text vector is closest. No training, no labeled examples, and you can redefine the label set at inference time by typing different words. The detail that matters more than it should is the prompt template: "a photo of a {}" beats the bare class name by several accuracy points, because captions in the training data are sentences, not single words.

**30. CLIP weaknesses?**
Weak at OCR/text in images, fine-grained categories, compositional reasoning. Bias inherited from web data.

> **Saying it out loud.** CLIP has three named weaknesses. It's bad at reading text in images, because alt-text rarely transcribes what's written. It's bad at fine-grained distinctions — it knows "bird" much better than it knows which species of warbler. And it's bad at composition: it treats a caption more like a bag of concepts than a structured statement, so "the red cube on the blue sphere" and "the blue cube on the red sphere" get nearly identical embeddings. That last one is the deepest problem, and it's a direct consequence of a single-vector representation trained with a contrastive loss that never had to distinguish those two.

**31. SigLIP improvement?**
Replace softmax with sigmoid loss. Each pair labeled independently positive/negative. Faster, scales better at small batch.

> **Saying it out loud.** SigLIP swaps the softmax for a sigmoid, so instead of asking "which of these N captions is right," you ask each pair independently "do these two go together, yes or no?" That sounds minor and it isn't, because the softmax requires normalizing across the whole batch, which means an all-gather across every GPU and a memory cost quadratic in batch size. The sigmoid loss is fully local, so it trains fine at modest batch sizes and scales cleanly. SigLIP matches or beats CLIP at far smaller batches — that's the practical headline.

---

## G. Multimodal LLMs

**32. Flamingo's key idea?**
Frozen LLM + vision encoder + new gated cross-attention layers. Don't retrain the LLM; augment it.

> **Saying it out loud.** Flamingo's idea was don't touch the language model. You freeze a strong pretrained LLM, freeze a vision encoder, and insert new gated cross-attention layers in between so text tokens can look at image features. The gates start at zero, which means at initialization the model is exactly the original LLM and vision contributions fade in gradually during training — that's what keeps it from destroying the language ability. It also handles interleaved image-text sequences, which is what gave it few-shot multimodal prompting.

**33. LLaVA architecture in 1 sentence?**
CLIP ViT image encoder → linear projection → concatenated with text tokens → fed to LLM.

> **Saying it out loud.** LLaVA is almost embarrassingly simple, which is why it took over. Run the image through a frozen CLIP vision encoder, push the patch features through a single projection layer to match the LLM's embedding dimension, and then just paste those vectors into the token sequence alongside the text embeddings. The LLM has no idea some of its tokens came from pixels. No cross-attention, no gating, no architecture surgery — the entire multimodal interface is one matrix.

**34. LLaVA training stages?**
Stage 1: train projection only on caption data (alignment). Stage 2: instruction tune on image-text instruction data.

> **Saying it out loud.** Two stages, and the split is deliberate. Stage one freezes everything except that projection layer and trains it on plain image-caption data, whose only job is to teach the projection to emit vectors that look like word embeddings to the LLM. Stage two unfreezes the LLM and fine-tunes on multimodal instruction data — conversations about images, generated with GPT-4 from captions and bounding boxes. Doing stage two without stage one is where it goes wrong: the LLM gets flooded with garbage vectors and you damage the language model instead of aligning the vision encoder.

**35. Native multimodal vs bolted-on?**
Native (Gemini 1.5+, GPT-4o): trained from scratch on multiple modalities. Bolted-on (LLaVA, early Flamingo): vision adapter on top of pre-trained LLM. Native generalizes better but expensive.

> **Saying it out loud.** Bolted-on means you take a finished text LLM and graft vision onto it afterward; native means images, audio, and text are in the pretraining mix from the start. Native wins on the things that require modalities to genuinely share representations — reasoning across an image and a document, or generating images as naturally as text. Bolted-on is enormously cheaper and gets you most of the way on captioning and visual question answering. The tradeoff to name is cost versus depth: an adapter can be trained on a handful of GPUs in a day, while native multimodal pretraining is a full frontier-scale run you can't redo cheaply.

**36. How are images "tokenized" for LLMs?**
ViT-style patch embeddings, optionally compressed via Q-former or perceiver to fewer tokens. Then treated as a sequence of "image tokens" in the LLM's input.

> **Saying it out loud.** You chop the image into fixed-size patches — 14 or 16 pixels square — linearly embed each one, and hand the LLM that sequence as if the patches were words. The problem is count: a 336-pixel image at patch size 14 gives you 576 tokens for a single picture, and attention is quadratic, so a handful of images eats your entire context window. That's why Q-Former and Perceiver resamplers exist — they use a small set of learned queries to compress hundreds of patch tokens down to something like 32 or 64. The tradeoff is exactly what you'd expect: fewer tokens means cheaper and longer conversations, and worse performance on fine detail like reading small text.

---

## H. Vector search / retrieval

**37. ANN vs exact KNN?**
ANN trades small recall loss for huge speedup. Exact $O(N)$, ANN can be sub-linear.

> **Saying it out loud.** Exact nearest neighbor means comparing your query against every vector in the index, which is linear in corpus size and completely fine until it isn't. Approximate nearest neighbor gives up the guarantee that you'll find the true top-k in exchange for sub-linear search. In practice you tune it to something like 95 to 99 percent recall and get one to two orders of magnitude speedup. The tradeoff worth naming is that recall is a knob, not a constant — every ANN index has a parameter that trades latency against how many true neighbors you actually retrieve.

**38. HNSW — what is it?**
Hierarchical Navigable Small World graph. Multi-layer graph; greedy search at each layer. State-of-the-art for many ANN workloads.

> **Saying it out loud.** HNSW builds a layered graph where each vector is a node connected to its near neighbors, and the upper layers are sparse long-range shortcuts. You search by starting at the top, greedily walking toward your query, then dropping a layer and refining, like zooming in on a map. It's the default in most vector databases because the recall-latency curve is excellent. The costs to name are memory — you're storing the full graph plus full-precision vectors, often several times the raw data size — and that deletions are awkward, since removing a node damages connectivity and usually requires periodic rebuilds.

**39. IVF-PQ?**
Inverted File Index + Product Quantization. Cluster vectors (IVF), then quantize each cluster's residuals (PQ). Memory- and speed-efficient.

> **Saying it out loud.** IVF-PQ is the memory-conscious alternative. The IVF part clusters all your vectors with k-means and, at query time, only searches the few nearest clusters instead of everything. The PQ part splits each vector into chunks and replaces each chunk with a byte-sized codebook ID, so a 768-dimensional float vector at 3 kilobytes collapses to something like 64 bytes. That's roughly a fiftyfold memory reduction, which is how you fit a billion vectors in RAM. The tradeoff is precision: quantization is lossy, so you typically re-rank the top candidates against the full-precision vectors.

**40. Hybrid search components?**
Sparse (BM25) + dense (embedding) retrieval. Combined via score blending or rank fusion (RRF).

> **Saying it out loud.** Hybrid search runs a lexical retriever and a vector retriever in parallel and merges the results. The merging is the part people get wrong — BM25 scores and cosine similarities live on totally different scales, so blending raw scores requires normalization that's fragile across queries. Reciprocal rank fusion sidesteps that by throwing away the scores entirely and combining based on rank position, one over a constant plus rank. It's dumb, it has one parameter, and it's remarkably hard to beat.

**41. Why hybrid?**
Dense catches paraphrases; sparse catches rare exact terms (proper nouns, IDs). Together more robust.

> **Saying it out loud.** Because the two methods fail in opposite places. Dense retrieval handles "how do I fix my car overheating" matching a document about engine cooling, where no words overlap. But it's bad at exact strings — a part number, a person's name, an error code — because those get tokenized into fragments and their meaning isn't compositional. BM25 nails exactly those and is helpless on paraphrase. Since the failure modes barely overlap, the union recovers substantially more than either alone, typically several points of recall, which is why every serious retrieval stack is hybrid.

---

## I. Subtleties

**42. Cosine similarity vs dot product?**
Cosine: normalized; magnitude doesn't matter. Dot product: unnormalized; can be useful when magnitude carries information.

> **Saying it out loud.** Cosine only looks at direction; dot product looks at direction and length. Which one you want depends on whether magnitude means something in your space. For most sentence embedders, vector length correlates with things like token count and word frequency rather than relevance, so you normalize and use cosine, otherwise long documents win by default. But some models are deliberately trained so magnitude encodes confidence or importance, and normalizing those throws away signal. The rule is simple: match whatever the model was trained with, because a retriever trained with cosine and queried with dot product will quietly underperform.

**43. Why $\ell_2$-normalize embeddings before retrieval?**
Cosine similarity = dot product after normalization. Makes search uniform; avoids dominant high-magnitude vectors.

> **Saying it out loud.** Two reasons, one mathematical and one about efficiency. Mathematically, once every vector has unit length, the dot product *is* the cosine, so you get scale-invariant similarity without a division per comparison. Practically, it stops a few high-magnitude vectors from being everyone's nearest neighbor regardless of direction — a real pathology in unnormalized indexes. And it lets you use fast inner-product search hardware paths, and makes Euclidean distance monotonic with cosine, so an L2 index and a cosine index return the same ranking.

**44. Embedding space anisotropy?**
Pre-trained embeddings often cluster in a narrow cone. Reduces effective dimensionality. Whitening / contrastive fine-tuning can fix.

> **Saying it out loud.** Anisotropy is the finding that embeddings out of a pretrained language model don't fill the space — they occupy a narrow cone, so random pairs of unrelated sentences already have cosine similarity around 0.6 or higher. That squashes your usable dynamic range: everything looks similar to everything, and the differences that matter get compressed into a sliver. It's driven largely by token frequency, with rare tokens pushed far out. You fix it either cheaply with whitening or a post-processing transform, or properly with contrastive fine-tuning, which explicitly spreads negatives apart and is why Sentence-BERT-style models don't have the problem.

**45. Catastrophic forgetting in vision-language fine-tuning?**
Fine-tuning a multimodal model on a narrow task can wipe out general capabilities. LoRA / adapter methods mitigate.

> **Saying it out loud.** Catastrophic forgetting is when fine-tuning on your narrow task overwrites the general capability you were paying for. Fine-tune a vision-language model on chart question answering and it can come back genuinely worse at ordinary captioning, because gradient descent has no obligation to preserve behavior on data it isn't shown. The mitigations all work by limiting how far the weights can move: LoRA and adapters restrict updates to a small number of added parameters, you keep the learning rate low, and you mix a slice of general instruction data back into the fine-tuning set. That last one is the cheapest fix and the one people forget.

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
