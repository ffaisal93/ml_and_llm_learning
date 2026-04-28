# Topic 38: Multimodal Models & Embedding History

> 🔥 **For interviews, read these first:**
> - **`MULTIMODAL_EMBEDDING_DEEP_DIVE.md`** — frontier-lab deep dive: BoW/TF-IDF → Word2Vec/GloVe → BERT → Sentence-BERT → CLIP → multimodal LLMs (Flamingo, LLaVA), full CLIP loss derivation, InfoNCE as MI bound, SigLIP, vector search (HNSW/IVF-PQ), hybrid retrieval.
> - **`INTERVIEW_GRILL.md`** — 55 active-recall questions.

## What You'll Learn

The lineage of representation learning:
- Bag of words and TF-IDF
- Distributed word embeddings (Word2Vec, GloVe)
- Contextual embeddings (ELMo, BERT, GPT)
- Sentence embeddings (Sentence-BERT, modern retrievers)
- Image-text contrastive learning (CLIP, ALIGN, SigLIP)
- Multimodal LLMs (Flamingo, LLaVA, GPT-4V, Gemini)
- Vector search infrastructure (HNSW, IVF-PQ, hybrid)

## Why This Matters

Modern systems (RAG, search, recommendation, multimodal agents) all rest on embeddings. Frontier interviews probe whether you understand *why* CLIP works, *what* InfoNCE optimizes, and *how* multimodal LLMs are actually built. The historical lineage is the cleanest way to convey design judgment.

## Next Steps

- **Topic 33**: Information theory — MI bounds underlying contrastive learning.
- **Topic 39**: RAG — production retrieval systems built on these embeddings.
- **Topic 15**: Tokenization — multi-modal tokenization (image patches, audio frames).
