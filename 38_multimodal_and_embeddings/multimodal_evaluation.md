# Multimodal Model Evaluation

## Overview

Evaluating multimodal models is challenging because they need to be tested on multiple modalities and their interactions. This document covers evaluation methods for multimodal models, especially CLIP and similar models.

## Evaluation Categories

### 1. Zero-Shot Image Classification

**What it tests:**
- Can model classify images without task-specific training?
- Does it understand natural language descriptions?

**Method:**
1. Create text prompts for each class: "a photo of a {class}"
2. Encode prompts using text encoder
3. Encode test images using image encoder
4. Find most similar prompt for each image
5. Compute accuracy

**Datasets:**
- **ImageNet**: 1,000 classes, standard benchmark
- **CIFAR-10/100**: 10/100 classes
- **Oxford-IIIT Pet**: 37 pet breeds
- **Food-101**: 101 food categories

**Metrics:**
- **Accuracy**: Percentage of correct predictions
- **Top-5 Accuracy**: Correct class in top 5 predictions

**Example:**
```python
# Create prompts
prompts = ["a photo of a cat", "a photo of a dog", ...]

# Encode
text_features = clip.encode_text(prompts)
image_features = clip.encode_image(images)

# Classify
similarities = image_features @ text_features.T
predictions = np.argmax(similarities, axis=1)
accuracy = (predictions == labels).mean()
```

**Results:**
- CLIP matches or exceeds supervised models on many datasets
- Especially good on natural images
- Struggles on fine-grained tasks (e.g., counting)

---

### 2. Image-Text Retrieval

**What it tests:**
- Can model find relevant images given text query?
- Can model find relevant text given image query?

**Tasks:**

**a) Image Retrieval (Text → Image):**
- Given text query, find matching images
- Example: "a red car" → find images of red cars

**b) Text Retrieval (Image → Text):**
- Given image, find matching captions
- Example: Image of cat → find "a photo of a cat"

**Datasets:**
- **Flickr30K**: 31,000 images, 5 captions each
- **MS-COCO**: 123,000 images, 5 captions each
- **Conceptual Captions**: 3.3M images with captions

**Metrics:**
- **Recall@K**: Percentage of queries where correct result in top K
  - Recall@1: Correct result is #1
  - Recall@5: Correct result in top 5
  - Recall@10: Correct result in top 10
- **Median Rank**: Median rank of correct result (lower is better)

**Example:**
```python
# Image retrieval
query = "a red car"
query_features = clip.encode_text([query])
image_features = clip.encode_image(images)

similarities = query_features @ image_features.T
top_k_images = np.argsort(similarities[0])[-k:][::-1]

# Compute Recall@K
recall_at_k = compute_recall_at_k(correct_images, top_k_images, k)
```

**Results:**
- CLIP achieves strong retrieval performance
- Better than previous methods without task-specific training
- Scales well with model size

---

### 3. Visual Question Answering (VQA)

**What it tests:**
- Can model answer questions about images?
- Does it understand both visual and textual information?

**Method:**
1. Combine CLIP with language model
2. Encode image and question
3. Generate answer

**Datasets:**
- **VQA v2**: 1.1M questions, 11M answers
- **GQA**: Scene graph-based questions
- **TextVQA**: Questions about text in images

**Metrics:**
- **Accuracy**: Percentage of correct answers
- **Per-question-type accuracy**: Accuracy by question type

**Challenges:**
- Requires reasoning about image content
- Need to combine visual and textual understanding
- Some questions require counting, spatial reasoning

**Results:**
- CLIP alone not sufficient (needs additional components)
- CLIP + language model achieves good performance
- Better on visual questions than text-heavy questions

---

### 4. Image Captioning

**What it tests:**
- Can model describe images in natural language?
- Quality of generated captions

**Method:**
- Fine-tune CLIP + language model on captioning data
- Generate captions autoregressively

**Datasets:**
- **MS-COCO Captions**: 123,000 images with captions
- **Flickr30K**: 31,000 images with captions

**Metrics:**
- **BLEU**: N-gram precision
- **METEOR**: Considers synonyms, paraphrases
- **CIDEr**: Consensus-based image description evaluation
- **SPICE**: Semantic propositional image caption evaluation
- **Human evaluation**: Best but expensive

**Results:**
- CLIP features help with captioning
- Better than training from scratch
- Still not as good as task-specific models

---

### 5. Robustness Evaluation

**What it tests:**
- How robust is model to distribution shifts?
- Performance on out-of-distribution data

**Tests:**

**a) Distribution Shift:**
- Train on one domain, test on another
- Example: Train on photos, test on sketches

**b) Adversarial Examples:**
- Small perturbations to images
- Test if model is fooled

**c) Natural Variations:**
- Different lighting, angles, backgrounds
- Test generalization

**Datasets:**
- **ImageNet-A**: Adversarial examples
- **ImageNet-R**: Renditions (art, cartoons, etc.)
- **ImageNet-Sketch**: Sketch versions

**Results:**
- CLIP more robust than supervised models
- Better generalization to new domains
- Still vulnerable to adversarial attacks

---

### 6. Bias Evaluation

**What it tests:**
- Does model have biases?
- Fairness across different groups

**Tests:**

**a) Gender Bias:**
- Test if model associates certain occupations with gender
- Example: "doctor" → mostly male images?

**b) Racial Bias:**
- Test if model has racial biases
- Example: Crime-related queries → certain groups?

**c) Cultural Bias:**
- Test if model understands diverse cultures
- Example: Wedding images → only Western weddings?

**Methods:**
- Analyze predictions by demographic groups
- Test on diverse datasets
- Measure fairness metrics

**Results:**
- CLIP inherits biases from training data
- Can produce biased outputs
- Need careful evaluation and mitigation

---

### 7. Few-Shot Learning

**What it tests:**
- Can model learn from few examples?
- Transfer learning capability

**Method:**
1. Provide few examples (1-16) of new task
2. Test if model can generalize
3. Compare to training from scratch

**Tasks:**
- Few-shot image classification
- Few-shot object detection
- Few-shot visual question answering

**Results:**
- CLIP good at few-shot learning
- Better than training from scratch
- Can learn from just a few examples

---

## Evaluation Best Practices

### 1. Multiple Metrics

**Don't rely on single metric:**
- Accuracy can be misleading
- Use multiple metrics (accuracy, recall, precision)
- Consider per-category performance

### 2. Diverse Datasets

**Test on diverse data:**
- Different domains (photos, art, sketches)
- Different languages
- Different cultures

### 3. Human Evaluation

**When possible, use human evaluation:**
- Automatic metrics have limitations
- Human judgment is gold standard
- Especially for generation tasks

### 4. Error Analysis

**Understand failure cases:**
- What types of errors does model make?
- Are errors systematic?
- How to improve?

### 5. Fairness Testing

**Test for biases:**
- Evaluate across demographic groups
- Measure fairness metrics
- Mitigate biases if found

---

## Summary

**Key Evaluation Methods:**
1. **Zero-shot classification**: Test generalization
2. **Image-text retrieval**: Test alignment
3. **VQA**: Test reasoning
4. **Captioning**: Test generation
5. **Robustness**: Test generalization
6. **Bias**: Test fairness
7. **Few-shot**: Test transfer learning

**Best Practices:**
- Use multiple metrics
- Test on diverse datasets
- Consider human evaluation
- Analyze errors
- Test for biases

**Challenges:**
- No single metric captures everything
- Need diverse evaluation
- Human evaluation expensive
- Bias evaluation complex

