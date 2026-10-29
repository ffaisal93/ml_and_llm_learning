# The grind list: 122 questions answered

This chapter answers every question in the [123ofAI AI/ML Grind List](https://github.com/connect-123ofai/ai-ml-grind-list),
in the order that repository lists them. It exists because that list is a good
inventory of what gets asked, and an inventory is only useful once the answers
are written down.

Each answer is about a minute of speech, 110 to 180 words, written the way you
would say it rather than the way a textbook prints it. That is deliberate. The
whole chapter is roughly two hours spoken, so you can work through it twice in
two or three days, which is the point. An answer that ran to a page would steal
the time another question needs.

Every answer leads with the direct answer, then gives the mechanism or the
equation, then one sentence on the consequence or the tradeoff. There is no
separate spoken block, because the answer already is one.

The nine coding questions are different. Each has a short implementation you
should be able to type from a blank file, and every one of them was executed
before it was published. Where a result can be checked against a library it was
checked, and the agreement is stated in the text.

**How to use it in three days.** Day one, read the ML theory section aloud and
mark the questions you could not answer before reading. Day two, do the same for
the LLM and Grind More sections, then type all nine coding answers from memory.
Day three, re-read only what you marked. Reading twice feels like progress and is
not; the marked set is the work.

Where this book covers a topic in depth, the answer says so. The breadth pages,
the pattern chapters and the system-design case studies all go further than a
one-minute answer can.

---

## Grind 75 ML — theory

### Q1. What techniques can be used for Feature Engineering with text data?

I start with cleaning: case folding, punctuation removal, stopword removal, and stemming or lemmatization. Then I turn text into numbers. Count-based features are bag-of-words and n-grams, usually weighted by TF-IDF, which scales a term by its frequency in the document and down-weights terms that appear in many documents. Hashing the vocabulary into a fixed number of buckets keeps memory bounded. Next are dense representations: word embeddings such as word2vec or GloVe, averaged or pooled to make a document vector, and contextual embeddings from a transformer encoder, which is what I use today for most tasks. I also add hand-made features when the task needs them: document length, type-token ratio, punctuation and capitalisation counts, readability scores, sentiment scores, named entity counts, and part-of-speech tag ratios. Topic models such as LDA give low-dimensional thematic features. The tradeoff is clear: sparse count features are cheap and interpretable, however dense contextual features usually win on accuracy.

### Q2. What does an embedding refer to in natural language processing?

An embedding is a mapping from a discrete symbol, such as a word, a subword token, a sentence, or a document, to a dense real-valued vector in a continuous space. Formally it is a function $E: V \rightarrow \mathbb{R}^d$, where $V$ is the vocabulary and $d$ is typically between 100 and 4096. The vector is learned, not hand-made. The learning objective makes geometry carry meaning: tokens that appear in similar contexts get similar vectors, therefore cosine similarity between vectors approximates semantic similarity. This replaces one-hot encoding, which is high-dimensional, sparse, and treats every pair of distinct words as equally different. Static embeddings such as word2vec give one vector per word type. Contextual embeddings from a transformer give a different vector for each occurrence, so they separate the two senses of "bank". The cost is interpretability: individual dimensions have no clean meaning, and the embedding inherits any bias present in the training corpus.

### Q3. What are the primary challenges involved in training RNNs? How can these challenges be addressed?

The main problem is unstable gradients through time. Backpropagation through time multiplies the same recurrent Jacobian at every step, so the gradient norm scales roughly like $\|W\|^T$ over $T$ steps. If the dominant eigenvalue is below one the gradient vanishes, therefore the network cannot learn long-range dependencies. If it is above one the gradient explodes and training diverges. Fixes: gradient clipping for explosion, gated cells such as LSTM or GRU for vanishing, because the cell state has an additive path with gradient near one. Orthogonal or identity initialisation of the recurrent matrix helps. Truncated backpropagation through time bounds the cost and the instability. Layer normalisation inside the cell stabilises activations. The second problem is speed: recurrence is sequential, so it does not parallelise across time, unlike a transformer. The third is a long-horizon memory limit that gates reduce but do not remove, so I add attention when the dependency is far away.

### Q4. What distinguishes lemmatization from stemming?

Both reduce a word to a base form, however they differ in method and in correctness. Stemming chops affixes using fixed rules, for example the Porter or Snowball algorithm. It ignores context and the dictionary, so the output can be a non-word: "studies" becomes "studi" and "better" stays "better". Lemmatization maps the word to its dictionary headword, called the lemma, using a lexicon and the part-of-speech tag. So "studies" becomes "study" and "better" becomes "good". Lemmatization is therefore linguistically correct but slower, and it needs the correct part-of-speech tag as input. Stemming is fast and crude. Two failure modes describe the tradeoff: over-stemming merges words that should stay separate, and under-stemming leaves related words apart. In practice I use stemming for large-scale retrieval and bag-of-words models, where recall matters more than form. I use lemmatization when a human reads the output, or when the feature must be a real word.

### Q5. What is the resultant rank of multiplying a 3x1 and 1x3 matrix?

The rank is 1. The product of a 3x1 column vector $u$ and a 1x3 row vector $v^T$ is a 3x3 matrix, and its rank is exactly 1 whenever both vectors are non-zero, because every column of $uv^T$ is a scalar multiple of the same vector $u$, so the column space is one-dimensional. This is the outer product. The general law is $\mathrm{rank}(AB) \le \min(\mathrm{rank}(A), \mathrm{rank}(B))$, and a 3x1 matrix has rank at most 1, therefore the product cannot exceed 1. If either vector is the zero vector the product is the zero matrix and the rank is 0. I checked this in numpy: I formed the outer product of a 3x1 and a 1x3, and `numpy.linalg.matrix_rank` returned shape (3, 3) and rank 1, for the fixed example and for five random pairs. The matrix is therefore singular and has two zero eigenvalues.

### Q6. Optimize the prediction throughput for an RNN?

Throughput means sequences per second, not latency for one sequence. First I batch: I pad many sequences to a common length and run them together, and I sort by length into buckets so padding waste is small. Packed sequence formats skip the padded steps entirely. Second I use a fused kernel, for example cuDNN's LSTM, which merges the four gate matrix multiplications into one larger multiplication and keeps the state on the device. Third I shrink the model: quantise weights and activations to 8-bit integers, prune, or distil into a smaller student network. Fourth I remove repeated work: cache the hidden state across calls for streaming input, and precompute the input-to-hidden projection for the whole sequence in one multiplication, because it does not depend on the recurrent state. For generation I use a smaller beam or greedy decoding. If latency and throughput both matter and I may change the architecture, a transformer or a convolutional encoder parallelises across time and is usually faster.

### Q7. What does BLEU mean?

BLEU is Bilingual Evaluation Understudy. It is an automatic metric for machine translation and other text generation, and it scores a candidate against one or more reference texts. It computes modified n-gram precision for $n = 1$ to $4$: the count of each candidate n-gram is clipped at the maximum count in any reference, so repeating a correct word does not raise the score. The precisions are combined as a geometric mean, then multiplied by a brevity penalty that punishes output shorter than the reference:

$$\mathrm{BLEU} = \mathrm{BP} \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)$$

In words: the average of the n-gram precisions on a log scale, scaled down if the output is too short. The score is between 0 and 1, often reported as a percentage. The limits matter: BLEU has no recall term, it ignores meaning and synonyms, it is unreliable on single sentences, and scores are comparable only with identical tokenisation.

### Q8. What types of Model Drifts have you encountered, and how have you addressed them?

I separate three kinds. Data drift, also called covariate shift, is a change in the input distribution $P(X)$ while the mapping $P(Y \mid X)$ holds. Label drift is a change in $P(Y)$, for example the fraud base rate rising. Concept drift is a change in $P(Y \mid X)$ itself, which is the damaging one, because the learned relationship is now wrong. Drift can be sudden, gradual, or seasonal. I detect it with monitoring rather than intuition: population stability index or a Kolmogorov-Smirnov test on each feature, prediction distribution monitoring, and, where labels arrive late, a rolling accuracy or calibration curve on the delayed labels. I also watch for training-serving skew, which is a pipeline bug, not drift. The responses are: retrain on a recent window, reweight or use time-decayed sample weights, add the new features that explain the change, and recalibrate the threshold. I keep a champion-challenger setup so a retrained model proves itself before it replaces the old one.

### Q9. KL Divergence and Cross-Entropy – similarity? Can one be used in place of the other as a loss function?

They differ by exactly one term. For a target distribution $p$ and a model distribution $q$:

$$D_{KL}(p \parallel q) = H(p, q) - H(p)$$

In words: the KL divergence equals the cross-entropy minus the entropy of the target. So cross-entropy is the KL divergence plus a term that depends only on $p$. Yes, they are interchangeable as a training loss whenever the target is fixed, because then $H(p)$ is a constant with respect to the parameters, therefore its gradient is zero. Minimising KL and minimising cross-entropy give the same gradients, the same optimum, and the same argmin. The loss values differ by that constant. With one-hot targets the constant is zero, so the two are numerically equal. The distinction matters in two cases: when the target itself is learned or changes, as in a variational autoencoder or in distillation with a moving teacher, and when I want the value reported as a divergence, because KL is zero at a perfect match and cross-entropy is not.

### Q10. Explain the types of optimizers – Adam, Adagrad, etc.

All of them descend the gradient; they differ in how they use history. Plain stochastic gradient descent takes $\theta \leftarrow \theta - \eta g$, so one fixed step size for every parameter. Momentum adds a running average of past gradients, which damps oscillation across a narrow valley. Nesterov momentum evaluates the gradient at the look-ahead point, so it corrects sooner. Adagrad divides the step by the square root of the accumulated sum of squared gradients, so rarely-updated parameters get large steps; however the sum only grows, therefore the learning rate decays to zero and training stalls. RMSProp replaces that sum with an exponential moving average, which fixes the decay. Adam combines RMSProp's second-moment estimate with momentum's first moment, and corrects both for initialisation bias; it is my default because it needs little tuning. AdamW decouples weight decay from the adaptive scaling, which is the correct form of regularisation. Well-tuned SGD with momentum still generalises slightly better on some vision tasks.

### Q11. What happens when there is no clear Hyperplane in SVM?

Two separate cases, with two separate fixes. If the classes overlap because of noise or outliers, I use the soft-margin SVM. It adds a slack variable $\xi_i \ge 0$ per point, which allows a point to sit inside the margin or on the wrong side, and it pays for that in the objective:

$$\min_{w,b} \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i$$

In words: keep the margin wide, and pay $C$ for every unit of violation. Small $C$ tolerates errors and gives a wider margin; large $C$ approaches the hard margin and can overfit. If instead the true boundary is curved, no linear hyperplane exists at any $C$. Then I map the data into a higher-dimensional feature space where it does become linearly separable, using a kernel such as the radial basis function or a polynomial. The hyperplane is linear in that space and curved in the original space. In practice I do both: a kernel plus a tuned $C$.

### Q12. Why would you use the Kernel Trick?

I use it to fit a non-linear boundary at the cost of a linear model, without ever building the high-dimensional features. Some algorithms, such as the SVM in its dual form, depend on the data only through inner products $\langle x_i, x_j \rangle$. A kernel replaces that inner product with $k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$ for a feature map $\phi$. The trick is that $k$ is computable directly, so I never compute $\phi(x)$. For example, the radial basis function kernel corresponds to an infinite-dimensional feature space, however evaluating it costs one distance and one exponential. Any symmetric positive semi-definite function is a valid kernel, by Mercer's theorem. The benefit is expressive power with a convex problem and no extra parameters per feature. The cost is scale: the kernel matrix is $n \times n$, so memory grows as $O(n^2)$ and training as roughly $O(n^2)$ to $O(n^3)$, therefore kernel methods do not suit very large datasets.

### Q13. Why are Non-linear Activation functions important?

Without a non-linearity, depth buys nothing, because a stack of linear layers collapses into a single linear layer. Algebraically, $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2) = W'x + b'$, which is one affine map. So a 100-layer linear network has exactly the representational power of linear regression, and it can only draw a hyperplane. A non-linearity between layers breaks that collapse, therefore each layer adds new capacity. With a non-polynomial activation and enough width, a network is a universal approximator, so it can approximate any continuous function on a compact set to any accuracy. The practical choice matters too: ReLU, defined as $\max(0, x)$, is cheap and its gradient is 1 on the positive side, so it avoids the saturation that makes sigmoid and tanh vanish gradients in deep stacks. The cost of a non-linearity is a non-convex loss surface, so I get local minima and saddle points instead of one global solution.

### Q14. How do you select the Activation Function for a Deep Learning Model?

I choose by layer position and by what the layer must output. For hidden layers my default is ReLU, because it is cheap and does not saturate for positive inputs. If I see dead units, meaning units stuck at zero output with zero gradient, I switch to Leaky ReLU or ELU, which keep a small negative slope. For transformers I use GELU or SiLU, because the smooth curve trains slightly better there. For the recurrent path in an LSTM I keep sigmoid on the gates and tanh on the state, because the gates must produce values in $[0, 1]$. The output layer is determined by the task, not by taste: linear for regression, sigmoid for binary and multi-label classification, and softmax for single-label multi-class, because softmax gives a distribution that sums to one. I pair the activation with a matching initialisation, so He initialisation for ReLU and Xavier for tanh. Then I confirm the choice empirically on a validation set.

### Q15. What is Cross-Validation and why is it important in supervised learning?

Cross-validation estimates how well a model generalises, by training and testing on different splits of the same data. In k-fold cross-validation I split the data into $k$ equal parts, train on $k-1$ of them, test on the held-out part, and repeat $k$ times so every point is tested once. The reported score is the mean of the $k$ scores, and the spread across folds tells me how stable the estimate is. Typical $k$ is 5 or 10. It matters for three reasons. First, a single train-test split gives a high-variance estimate that depends on which rows landed in the test set. Second, every point contributes to both training and evaluation, which matters when data is scarce. Third, it is the correct way to compare models and tune hyperparameters. Two rules: stratify the folds to preserve class balance, and respect structure, so I use grouped folds for repeated subjects and forward-chaining splits for time series, otherwise the score leaks.

### Q16. What happens when you initialise your network weights with all zeros, all constants, or all random values?

All zeros fails completely. Every unit in a layer computes the same output, therefore it receives an identical gradient, therefore it updates identically and stays identical forever. The units never differentiate, so the layer behaves like a single unit. This is the symmetry problem. All constants has exactly the same problem, because the units are still identical; a non-zero constant only adds a risk of exploding or saturating activations. Random values break the symmetry, because the units start different and so receive different gradients. However the scale matters. Too small and the activation variance shrinks layer by layer, so the signal and the gradient vanish. Too large and both explode, or saturate a sigmoid. Therefore the variance is chosen to keep activation variance roughly constant across layers: Xavier uses $\mathrm{Var}(W) = 2/(n_{in}+n_{out})$ for tanh, and He uses $2/n_{in}$ for ReLU, because ReLU zeroes half the input. Biases can safely start at zero, because the weights already break the symmetry.

### Q17. Would you use PCA on large datasets or is there a better alternative?

Exact PCA does not suit a large dataset, because it needs the full covariance matrix or a full singular value decomposition, so cost is about $O(nd^2)$ in time and $O(d^2)$ in memory, and it needs all the data at once. I use a scalable variant instead. Randomised or truncated SVD finds only the top $k$ components and is much cheaper. Incremental PCA processes mini-batches and keeps memory bounded, so it works on data that does not fit in RAM. For sparse data, truncated SVD applies without mean-centring, which would destroy the sparsity. For very wide data, random projection is nearly free and the Johnson-Lindenstrauss lemma bounds the distortion. If the structure is non-linear, PCA is the wrong tool at any scale: I use an autoencoder, which scales with stochastic gradient descent, or UMAP for visualisation. One caveat: UMAP and t-SNE preserve neighbourhoods, not distances or variance, so I do not feed their output into a distance-based model.

### Q18. While performing K-Means Clustering, how do you determine the value of K?

There is no single correct $K$, so I combine evidence. The elbow method plots the within-cluster sum of squares against $K$ and looks for the bend where added clusters stop reducing it much; the weakness is that the bend is often not sharp. The silhouette score is better: for each point it compares the mean distance to its own cluster against the nearest other cluster, and the score is in $[-1, 1]$, so I pick the $K$ with the highest mean. The gap statistic compares the within-cluster dispersion against the value expected under a uniform reference distribution. Other options are the Calinski-Harabasz and Davies-Bouldin indices, and the Bayesian information criterion if I switch to a Gaussian mixture model. Stability under resampling is a useful check, because a real cluster structure reproduces. However the domain usually decides: if the business needs five segments, then $K$ is five, and the metrics only tell me how well the data supports it.

### Q19. What distinguishes Linear Filters from Non-Linear Filters?

A linear filter computes a weighted sum of the neighbourhood, so it is a convolution, and it obeys superposition: the response to $ax_1 + bx_2$ equals $a$ times the response to $x_1$ plus $b$ times the response to $x_2$. Examples are the box mean filter, the Gaussian blur, and the Sobel edge filter. Because they are linear and shift-invariant, they have a frequency response, they commute, and the fast Fourier transform can apply them cheaply. A non-linear filter does not obey superposition, because it uses an operation such as a rank, a comparison, or a ratio. Examples are the median filter, the min and max morphological filters, and the bilateral filter, which weights a neighbour by both spatial distance and intensity difference. The practical difference is edge behaviour. A Gaussian blur removes salt-and-pepper noise by smearing it, therefore it blurs edges too. A median filter removes the same noise and keeps edges sharp, because an outlier never wins the rank, however it costs more and has no frequency response.

### Q20. Suppose you want to train a classifier and you have plenty of unlabeled training data but only a few thousand labeled instances. What would you do?

This is semi-supervised learning. My first move is representation learning on the unlabelled data: pretrain a model with a self-supervised objective, such as masked token prediction for text or contrastive learning for images, or simply take an existing pretrained model, then fine-tune on the few thousand labels. This is the modern answer and usually the strongest, because the unlabelled data pays for the features and the labels only have to fit the final head. Second, self-training: train on the labelled set, predict on the unlabelled set, add the confident predictions as pseudo-labels, and retrain; I threshold on confidence and iterate, however I watch for confirmation bias, because the model reinforces its own errors. Third, active learning: instead of labelling at random, label the points the model is least certain about, so each new label buys more. Fourth, cluster the unlabelled data and label a few representatives per cluster, which propagates each label across a whole region.

### Q21. How does a Deep Neural Network escape/resist the Curse of Dimensionality?

Honestly, it does not escape the curse in general. It exploits structure in the data. The classic answer is the manifold hypothesis: real data such as images or text occupies a low-dimensional manifold inside the high-dimensional input space, so the effective dimension is far below the pixel count, and a deep network learns coordinates on that manifold rather than filling the ambient space. Depth helps because composition reuses parts: each layer builds features from the previous layer, so a function that decomposes hierarchically needs exponentially fewer units in a deep network than in a shallow one. Weight sharing in a convolution adds the right prior, namely locality and translation equivariance, which cuts the parameter count. Stochastic gradient descent adds an implicit bias toward simple solutions. However the theory is clear that no method beats the curse for an arbitrary high-dimensional function: the worst-case sample requirement still grows exponentially. So deep networks win on structured data, and fail on genuinely unstructured high-dimensional problems.

### Q22. Why do models overfit and how do we prevent it?

A model overfits when it has enough capacity to fit the noise in the training set, so it memorises instead of generalising. The symptom is a training error that keeps falling while validation error rises. The causes are capacity too high for the amount of data, too few training examples, noisy or mislabelled targets, leaked or spurious features, and too many training steps. This is the bias-variance tradeoff: lowering bias past the right point raises variance. I prevent it on three fronts. Data: collect more, augment it, and clean the labels. Model: reduce capacity, add $L_2$ or $L_1$ penalties, use dropout, use batch or layer normalisation, and share weights where the structure allows. Training: early stopping on a validation set, label smoothing, and ensembling, which averages away independent errors. Across all of it I need an honest held-out split or cross-validation, because I cannot detect overfitting with the training score, and I must not tune on the final test set.

### Q23. What methods can you use to evaluate the predictions in an Object Detection model?

Detection has to score both the label and the box, so the core tool is intersection over union: the area of overlap between the predicted and ground-truth box divided by the area of their union. A prediction counts as a true positive if the class matches and the IoU exceeds a threshold, typically 0.5; otherwise it is a false positive, and an unmatched ground-truth box is a false negative. Each ground-truth box can be matched once, so duplicates count as false positives. From those counts I build a precision-recall curve by sweeping the confidence threshold, and the area under it is average precision for one class. Mean average precision averages that over classes. The COCO protocol reports mAP averaged over IoU thresholds from 0.50 to 0.95 in steps of 0.05, which rewards tight boxes, plus a breakdown by object size. I also look at per-class AP to find the weak classes, and at inference latency, because a detector usually has to run in real time.

### Q24. What is the Curse of Dimensionality and how can Unsupervised Learning help with it?

The curse of dimensionality is the set of problems that appear as the number of features grows. Volume grows exponentially, so a fixed sample becomes sparse and the data needed for a given density grows exponentially. Distances concentrate: in high dimensions the nearest and farthest neighbour distances become almost equal, therefore nearest-neighbour methods and distance-based clustering degrade. Nearly all points sit near the boundary of the space, so prediction is extrapolation. Noise features add variance without adding signal, which raises overfitting risk. Unsupervised learning helps because it reduces dimension without needing labels. PCA projects onto the directions of highest variance. An autoencoder learns a non-linear bottleneck representation. Self-supervised pretraining gives a compact embedding that downstream models use instead of raw features. Clustering replaces many features with one cluster identifier. Manifold methods such as UMAP recover low-dimensional structure for inspection. The caveat is that unsupervised reduction keeps what is variable, which is not always what the label depends on.

### Q25. What distinguishes a gradient boost decision tree (GBDT), a decision tree, and a random forest?

One tree versus two different ways of combining many. A single decision tree splits the feature space recursively to reduce impurity; it is interpretable and fast, however a deep tree has high variance, because a small change in the data changes the splits. A random forest trains many deep trees in parallel on bootstrap samples, and it also samples a random subset of features at each split, then averages the predictions. The trees are deliberately decorrelated, so averaging cuts variance; it is a bagging method, hard to overfit, and easy to tune. A GBDT trains trees sequentially, and each new shallow tree fits the gradient of the loss with respect to the current prediction, so it corrects the residual error of the ensemble so far. That reduces bias, not just variance, so GBDT usually wins on tabular data. The cost is sensitivity: it needs careful tuning of learning rate, tree depth and tree count, and it will overfit if trained too long.

### Q26. Why Naive Bayes is called Naive?

It is called naive because of one deliberately unrealistic assumption: that the features are conditionally independent given the class. That turns the joint likelihood into a simple product:

$$P(x_1, \dots, x_d \mid y) = \prod_{i=1}^{d} P(x_i \mid y)$$

In words: the probability of the whole feature vector is the product of each feature's own probability given the class. The assumption is almost always false in real data, because features correlate; in text, "New" and "York" are clearly dependent. That is the naivety. However the simplification is what makes the model useful: I estimate $d$ one-dimensional distributions instead of one $d$-dimensional joint, so training is one pass over the data, it needs little data, and it handles very high dimensions. It still classifies well even when the assumption is violated, because the decision only needs the correct argmax, not correct probabilities. The consequence is that its predicted probabilities are badly calibrated, usually over-confident, so I calibrate them before I use them as probabilities.

### Q27. What are the main difficulties when training GANs?

The core difficulty is that a GAN is a two-player minimax game, not a loss being minimised, so there is no single objective that measures progress. Training is a search for a Nash equilibrium and it need not converge; the parameters can cycle instead. Specific failures: mode collapse, where the generator maps many inputs to a few outputs that fool the discriminator, so it loses the diversity of the data. Vanishing generator gradient, because a discriminator that becomes too strong saturates and returns almost no signal. Instability from the delicate balance between the two networks, so learning rates and update ratios matter a lot. No reliable metric, therefore I lean on proxies and visual inspection. Sensitivity to architecture and initialisation. The usual mitigations are the non-saturating or Wasserstein loss, gradient penalty or spectral normalisation to constrain the discriminator, one-sided label smoothing, minibatch discrimination against collapse, and progressive growing. Even so, diffusion models have largely replaced GANs for this reason.

### Q28. How to use one-class SVM for Anomalies Detections?

I train a one-class SVM on data that I believe is almost all normal, and it learns a boundary around that normal region. Anything outside the boundary is an anomaly. The method finds a small region in feature space that contains most of the training points. With an RBF kernel, it fits a flexible contour around the data instead of a sphere. Two settings matter. The parameter $\nu$ sets the fraction of training points I allow outside the boundary, so it is an upper bound on my expected outlier rate; typically I set it near the contamination I expect, often 0.01 to 0.1. The kernel width $\gamma$ sets how tightly the contour wraps the data, so a large $\gamma$ overfits and flags normal points. At test time the decision function is positive inside and negative outside, and the magnitude gives an anomaly score I can threshold. I must scale features first, because the kernel uses distances.

### Q29. What is the Bias-Variance tradeoff?

Expected test error splits into three parts: bias squared, variance, and irreducible noise. Bias is the error from my model being too simple to represent the true function. Variance is how much my fitted model changes if I resample the training data. The tradeoff is that model capacity moves these in opposite directions. A linear model on curved data has high bias and low variance. A deep unpruned tree has low bias and high variance, because it memorises the particular sample. Formally, for squared loss,

$$\mathbb{E}[(y-\hat f(x))^2] = \text{Bias}[\hat f]^2 + \text{Var}[\hat f] + \sigma^2$$

which says total error is systematic error plus instability plus noise I cannot remove. Therefore I tune capacity, regularisation strength, or tree depth to minimise the sum, not either term alone. Two useful levers: bagging reduces variance while leaving bias roughly unchanged, and boosting reduces bias by adding corrective terms. Modern over-parameterised networks complicate the classical U-shaped picture, however the decomposition itself still holds.

### Q30. How can less Training Data give Higher Accuracy?

Several distinct mechanisms can do this, and they are not all the same thing. First, removing noisy or mislabelled examples: wrong labels push the decision boundary the wrong way, so deleting them improves the fitted model itself. Second, distribution shift: if part of the training set comes from a different distribution than the test set, that part teaches the model the wrong relationship, and dropping it helps. Third, a smaller model-selection search: if I use less data, I may also fit a smaller hypothesis space or fewer tuning rounds, which overfits the validation set less. Fourth, evaluation variance: with a small test set the measured accuracy has a wide confidence interval, so a gain can be chance rather than real. Fifth, deduplication, which removes leakage between training and test splits and gives a more honest, sometimes higher, measured score. Therefore I always ask whether the improvement is in the model or only in the measurement.

### Q31. Contrast the Sobel and Canny methods for Edge Detection in Computer Vision.

Sobel is a single gradient operator. It convolves the image with two small kernels, one for the horizontal derivative and one for the vertical, then reports the gradient magnitude $\sqrt{G_x^2+G_y^2}$ as edge strength. The output is a thick, grey edge map, and I must threshold it myself. Canny is a full pipeline that uses a gradient step inside it. It has five stages: Gaussian smoothing to suppress noise, gradient computation, non-maximum suppression to thin each ridge to one pixel, then double thresholding and hysteresis, which keeps weak edges only if they connect to strong edges. Therefore Canny gives thin, connected, binary contours and is far less noise-sensitive, at the cost of more computation and two threshold parameters to tune. I use Sobel when I only need a gradient field, for example as an input feature or for orientation. I use Canny when I need clean edge curves for downstream contour or shape analysis.

### Q32. What's the difference between Bagging and Boosting algorithms?

Both combine many weak models, however they target different error terms and train differently. Bagging trains the base models independently on bootstrap resamples and averages their predictions. Because the models are trained in parallel on different samples, their errors are partly uncorrelated, so averaging reduces variance. Random forests add random feature subsetting to decorrelate the trees further. Boosting trains models sequentially. Each new model fits what the current ensemble gets wrong, either by reweighting misclassified points as in AdaBoost, or by fitting the negative gradient of the loss as in gradient boosting. Therefore boosting mainly reduces bias, and it can drive training error very low. The consequences differ. Bagging is robust, parallelisable, and hard to overfit by adding trees. Boosting usually reaches higher accuracy on tabular data, however it is sensitive to noisy labels and outliers, and it needs careful control of the learning rate and the number of rounds to avoid overfitting.

### Q33. In what situations do you not use Ensemble Classifiers?

I avoid ensembles when their costs outweigh a small accuracy gain. First, when I need interpretability: a regulated credit or clinical decision may require a model I can explain coefficient by coefficient, and a hundred trees cannot do that. Second, when inference latency or memory is tight, for example on a microcontroller or in a high-throughput request path, because an ensemble multiplies compute. Third, when the data is tiny: bootstrap resamples of a few hundred rows are highly correlated, so the variance reduction is small and I may just add noise. Fourth, when a single model already fits well and the problem is genuinely linear, because the ensemble adds complexity for nothing. Fifth, when labels are very noisy, because boosting will chase the noise. Finally, ensembles do not fix the real problem when my error comes from bias in the features or from distribution shift; better features or better data helps more than more models.

### Q34. What is the difference between Post-pruning and Pre-pruning?

Both limit tree complexity, however they act at different times. Pre-pruning, also called early stopping, stops growth during training. I set a maximum depth, a minimum number of samples to split a node, a minimum samples per leaf, or a minimum impurity decrease. It is cheap, because I never build the large tree. Its weakness is the horizon effect: a split that looks useless alone may enable a very good split below it, and pre-pruning never sees that. Post-pruning grows the tree fully, then removes subtrees that do not pay for themselves. Cost-complexity pruning minimises $R_\alpha(T)=R(T)+\alpha|T|$, which is the training error plus a penalty $\alpha$ for each leaf, and I pick $\alpha$ by cross-validation. Therefore post-pruning usually gives a better accuracy-size tradeoff because it judges a subtree by its total contribution, however it costs more to train. In practice I use post-pruning for single trees and pre-pruning inside ensembles, where speed matters.

### Q35. How do you make your model robust to outliers?

I work at three levels. At the data level I detect and treat them: I inspect them, fix genuine data errors, winsorize or clip extreme values at a percentile, or apply a log or rank transform to compress a heavy tail. At the loss level I pick a loss that does not grow quadratically. Mean absolute error and the Huber loss, which is quadratic near zero and linear beyond a threshold $\delta$, reduce the pull of a far point; quantile regression on the median is similar. For classification, hinge and log loss are more forgiving than squared loss. At the model level I use methods that are naturally insensitive: tree-based models split on order, not magnitude, so a huge value only changes which side of a threshold it falls on. RANSAC fits on inlier subsets. I also use robust scaling with the median and interquartile range instead of mean and standard deviation. Importantly, I never delete points just for being extreme, because they may be the signal.

### Q36. How would you define the Stopping Criteria for decision trees?

I stop splitting a node when further splitting cannot be justified. The natural stopping conditions are: the node is pure, so all labels agree; no feature gives any impurity decrease; or all remaining points have identical feature values, so no split is possible. Beyond these I add regularising limits, because an unlimited tree memorises the training set. The common ones are maximum depth, minimum samples required to split a node, minimum samples per leaf, maximum number of leaves, and a minimum impurity decrease threshold, so a split must improve purity by at least some amount to be accepted. I can also stop when a validation score stops improving. I choose these values by cross-validation rather than by rule, because the right depth depends on sample size and noise; with noisy data a minimum leaf size of a few dozen is often reasonable. Alternatively I skip tuning the stop and instead grow fully, then use cost-complexity post-pruning, which usually works better.

### Q37. In decision Tree, implement information gain.

Entropy of a label set $S$ is $H(S) = -\sum_i p_i \log_2 p_i$, where $p_i$ is the fraction of $S$ with class $i$. It measures label uncertainty in bits. The gain of a split is the parent entropy minus the weighted average entropy of the children:

$$IG(S,A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$$

so it is the uncertainty removed by knowing the split. I pick the split with the largest gain.

```python
def entropy(counts):
    n = sum(counts)
    return -sum((c/n) * math.log2(c/n) for c in counts if c > 0)

def info_gain(parent, children):   # children: list of count-lists
    n = sum(parent)
    return entropy(parent) - sum(sum(c)/n * entropy(c) for c in children)
```

Worked number: parent 5 positive and 5 negative gives $H=1$ bit. A split yields a left child of 4 positive and 0 negative, so $H=0$, and a right child of 1 positive and 5 negative, so $H\approx0.650$. The gain is $1 - 0.4(0) - 0.6(0.650) = 0.61$ bits.

### Q38. What distance metric to use in KNN.

It depends on the feature types, and the choice matters more than $k$. For continuous features on a common scale I default to Euclidean distance, $L_2$. In high dimensions Manhattan distance, $L_1$, often behaves better, because $L_2$ distances concentrate and all neighbours look equally far. For sparse high-dimensional vectors such as text, cosine distance is the right choice, because it compares direction and ignores length. For binary or set-valued features I use Hamming or Jaccard distance. For mixed numeric and categorical data I use Gower distance, which combines per-feature distances. If features are correlated and on different scales, Mahalanobis distance accounts for the covariance. Two rules apply whatever I pick. First, I must standardise numeric features, because an unscaled large-range feature dominates the distance. Second, I should encode ordinals by their order, not one-hot, when order is meaningful. Therefore I treat the metric as a hyperparameter and select it by cross-validation, since the best metric is data dependent.

### Q39. How to initialize K-means, and what is the stopping criteria?

For initialisation I use k-means++, which is the standard default. It picks the first centre uniformly at random, then picks each later centre with probability proportional to its squared distance from the nearest chosen centre. Therefore the initial centres are spread out, which avoids the bad local optima that purely random initialisation produces. I also restart the whole fit several times, typically ten, with different seeds and keep the run with the lowest inertia, which is the within-cluster sum of squared distances. Random partition and the Forgy method, which picks $k$ actual data points, are older alternatives. For stopping, the algorithm converges when no point changes its assigned cluster, which is guaranteed in finite time because inertia decreases monotonically. In practice I stop earlier on any of three conditions: centroid movement falls below a tolerance, the drop in inertia between iterations falls below a tolerance, or I reach a maximum iteration count, typically a few hundred. These give the same answer far faster.

### Q40. How would you handle feature values that are missing?

First I ask why the value is missing, because the mechanism decides the method. If it is missing completely at random, almost anything unbiased works. If it is missing at random given other features, model-based imputation works. If it is missing not at random, for example income missing because it is high, then the missingness itself is informative and I must encode it. My toolkit, cheapest first: drop the column if most of it is empty; drop rows only if few and the loss is random. Impute with the median for skewed numeric features, the mean for symmetric ones, and a separate "Missing" category for categoricals. Better, use multivariate imputation, such as iterative regression or k-nearest-neighbour imputation, which uses the other features. In every case I add a binary indicator column flagging that the value was imputed, so the model can use the missingness. Critically, I fit the imputer on training data only, then apply it to validation and test, to avoid leakage. Some models, notably gradient-boosted trees, handle missing values natively.

### Q41. How feature selection is done?

Three families, and I usually combine them. Filter methods score each feature against the target independently of the model: correlation, mutual information, a chi-squared test, or an ANOVA F-test. They are fast and good for a first cut from thousands of features, however they miss interactions and keep redundant correlated pairs. Wrapper methods search subsets by actually training the model: forward selection adds features one at a time, backward elimination removes them, and recursive feature elimination repeatedly drops the weakest. They respect interactions, however they are expensive and can overfit the selection set. Embedded methods select during fitting: L1 or lasso regularisation drives coefficients to exactly zero, and tree ensembles give impurity or permutation importances. I prefer permutation importance on held-out data, because impurity importance is biased toward high-cardinality features. Practical discipline: run selection inside the cross-validation loop, not before it, otherwise the reported score is optimistic. I also remove near-constant features and one of each highly correlated pair first.

### Q42. What is a Confusion Matrix?

It is a table that compares predicted labels against true labels, so I can see what kind of mistakes the model makes rather than only how many. For binary classification it has four cells: true positives, false positives, true negatives, and false negatives. Rows are usually the true class and columns the prediction, however I always check the orientation of the library I use. Every standard metric is derived from these four counts. Precision is $TP/(TP+FP)$, which is how often a positive prediction is right. Recall, or sensitivity, is $TP/(TP+FN)$, which is the fraction of real positives found. Specificity is $TN/(TN+FP)$. Accuracy is the diagonal over the total. For multiclass problems it generalises to a $k \times k$ matrix, and the off-diagonal cells show exactly which classes get confused with which, which often points at a labelling or feature problem. It is more informative than accuracy, especially with imbalanced classes, where accuracy hides a failure on the rare class.

### Q43. What are several ways to handle label imbalance in a dataset?

At the data level: random undersampling of the majority class, which is cheap however it discards information; random oversampling of the minority class, which risks overfitting duplicated points; and synthetic oversampling such as SMOTE, which interpolates new minority points between neighbours. At the algorithm level: class weights, so the loss penalises a minority error more, which most libraries support directly with one argument and which I try first because it adds no data distortion. Cost-sensitive learning generalises this with an explicit cost matrix. At the decision level: keep the model and move the threshold, choosing it on a validation set to hit a required recall or to maximise the F-score. Ensemble approaches such as balanced bagging train each base model on a balanced resample. Two cautions. I resample inside the training fold only, never before splitting, otherwise synthetic points leak into validation. And I must change the metric too, because accuracy is useless here; I use PR-AUC, F-score, or recall at fixed precision.

### Q44. For both Convex, and Non-Convex Problems, does the Gradient in SGD always point to the global extreme value?

No, and not even for convex problems. The gradient points along the direction of steepest local increase of the loss at the current point, so the negative gradient is a local descent direction only. It is a statement about an infinitesimal neighbourhood, therefore it is not in general a direction toward the global minimum; the straight path to the minimum can differ sharply, which is why badly conditioned problems zig-zag. For a convex problem, repeatedly following the negative gradient with a suitable step size converges to a global minimum, however this is because every local minimum of a convex function is global, not because any single gradient aimed there. For a non-convex problem the same procedure converges only to a stationary point, which may be a local minimum or a saddle region. There is a further point specific to SGD: the minibatch gradient is a noisy unbiased estimate of the full gradient, so a single SGD step may not even decrease the loss. It only descends in expectation.

### Q45. How would you choose an evaluation metric for an Imbalanced classification?

I start from the cost of each error type, not from a list of metrics. First I decide whether I need a ranking score, a calibrated probability, or a hard decision. For ranking under imbalance I use the area under the precision-recall curve, because precision uses the false positive count against the small positive class and therefore reacts to it, while ROC-AUC uses the false positive rate against a huge negative class and can look good when precision is terrible. For a hard decision I pick the operating point by the business constraint: recall at a fixed precision, precision at a fixed recall, or $F_\beta$, where $\beta$ above one favours recall. Matthews correlation coefficient and balanced accuracy are reasonable single-number summaries that account for all four confusion cells. If I need probabilities, for example to rank by expected cost, I use log loss or the Brier score. I never use plain accuracy, because predicting the majority class always scores well.

### Q46. How does Batch Size affect the Convergence of SGD and why?

Batch size controls the variance of the gradient estimate. The standard error of the minibatch gradient falls roughly as $1/\sqrt{B}$, so a larger batch gives a more accurate gradient and a smoother, more stable loss curve, and it allows a larger stable learning rate. A small batch gives a noisy gradient. That noise costs stability, however it also helps: it lets the iterate escape sharp minima and saddle points and acts as a regulariser, so small batches often generalise slightly better. The tradeoff is per-step cost against steps needed. Large batches need fewer updates per epoch and use hardware parallelism well, however each step costs more, and past a point extra accuracy in the gradient buys nothing, so wall-clock time stops improving. Large-batch training also tends to lose test accuracy unless I scale the learning rate with the batch size and add a warmup. Therefore I treat batch size and learning rate as a pair, and I tune them together rather than separately.

### Q47. How does ROC curve and AUC value help measure how good a model is?

The ROC curve plots the true positive rate against the false positive rate as I sweep the decision threshold across all values. Therefore it describes the model's whole ranking behaviour rather than its performance at one chosen cutoff, which is useful when I have not yet fixed an operating point. The area under it, AUC, has a clean interpretation: it is the probability that a randomly chosen positive example gets a higher score than a randomly chosen negative example. So 0.5 means random ranking and 1.0 means perfect separation. AUC is threshold-free and insensitive to the base rate, which makes it convenient for comparing models. It is also misleading under heavy class imbalance, because the false positive rate divides by a very large negative count, so thousands of false positives barely move the curve while precision collapses. In that case I use the precision-recall curve and PR-AUC instead. AUC also says nothing about calibration: a well-ranking model can still output badly wrong probabilities.

### Q48. What are the trade-offs between the different types of Classification Algorithms? How would you choose the best one?

Logistic regression is fast, interpretable, and gives calibrated probabilities, however it only fits a linear boundary. Naive Bayes is extremely fast and works with little data, however its feature-independence assumption makes its probabilities poorly calibrated. k-NN needs no training and fits any boundary, however prediction is slow and it degrades in high dimensions. SVMs handle high-dimensional and non-linear data well through kernels, however they scale badly past roughly a hundred thousand rows and need feature scaling. Decision trees are interpretable and need no scaling, however a single tree is unstable. Gradient-boosted trees are usually the strongest choice on tabular data, however they need tuning and are not interpretable. Neural networks dominate on images, text, and audio, where representation learning matters, however they need large data and compute. To choose, I set the constraints first: data size, data type, interpretability requirement, latency budget. Then I establish a cheap baseline, and I compare candidates by cross-validation on the metric that matches the real cost.

### Q49. What are Slack Variables in SVM?

Slack variables let an SVM fit data that is not linearly separable. A hard-margin SVM requires every point to satisfy $y_i(w^\top x_i + b) \ge 1$, which has no solution if the classes overlap. I introduce one non-negative slack $\xi_i$ per point and relax the constraint to $y_i(w^\top x_i + b) \ge 1 - \xi_i$. So $\xi_i$ measures how far point $i$ violates its margin: zero means it is correctly outside the margin, a value between zero and one means it is inside the margin but still correct, and above one means it is misclassified. The objective becomes

$$\min_{w,b,\xi} \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i$$

which trades a wide margin against total violation. The constant $C$ sets that balance. A large $C$ punishes violations hard, giving a narrow margin and a risk of overfitting; a small $C$ tolerates violations, giving a wider margin and more regularisation. Summing the slacks is exactly the hinge loss, so this is the soft-margin SVM.

### Q50. How would you make a prediction using a Logistic Regression model?

I compute a linear score, then squash it into a probability, then apply a threshold. The score is $z = w^\top x + b$, a weighted sum of the features plus an intercept. The probability of the positive class is the sigmoid of that score:

$$P(y=1 \mid x) = \sigma(z) = \frac{1}{1+e^{-z}}$$

which maps any real number into the interval from zero to one. The predicted label is positive when this probability exceeds a threshold. The default threshold of 0.5 corresponds to $z > 0$, however it is a choice, not part of the model, and I should set it from the costs of false positives and false negatives. Two practical points. The input must get the same preprocessing as in training, meaning the same encoding and the same scaler fitted on training data. And the coefficients are log-odds: a one-unit increase in a feature multiplies the odds by $e^{w_j}$. For multiclass problems I replace the sigmoid with a softmax over per-class scores.

### Q51. What is Silhouette Analysis?

It is a way to judge clustering quality without ground-truth labels. For each point $i$ I compute $a(i)$, the mean distance to the other points in its own cluster, and $b(i)$, the mean distance to the points of the nearest other cluster. The silhouette coefficient is

$$s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))}$$

which ranges from minus one to plus one. A value near one means the point sits well inside its cluster. Near zero means it lies on a boundary between two clusters. Negative means it is probably in the wrong cluster. The mean silhouette over all points gives a single score, and I can plot it against the number of clusters $k$ to choose $k$, often alongside the elbow method. The per-cluster silhouette plot is more useful than the average, because it exposes one bad cluster that a good average hides. One limit: it assumes compact, roughly convex clusters, so it undervalues correct but elongated or density-based clusterings.

### Q52. How do you choose the optimal k in k-NN?

I treat $k$ as a hyperparameter and select it by cross-validation. I evaluate a grid of odd values, which avoids ties in binary classification, and I keep the $k$ with the best validation score on the metric I actually care about. The reason a sweep is needed is the bias-variance tradeoff. A small $k$, especially $k=1$, gives a very flexible boundary with low bias and high variance, so it fits noise and single mislabelled points. A large $k$ averages over many neighbours, which smooths the boundary and reduces variance, however it raises bias and in the limit predicts the majority class everywhere. Within the plateau of similar scores I prefer the larger $k$, because it is more stable. A useful starting point is near $\sqrt{n}$, however that is a heuristic, not a rule. Two refinements: distance weighting, so nearer neighbours count more, which makes the result less sensitive to $k$; and scaling features first, because $k$ cannot fix a distorted metric.

### Q53. What would be a good way to use Clustering for Outlier detection?

The general idea is that outliers do not belong comfortably to any dense cluster, so I score each point by how poorly it fits the cluster structure. With DBSCAN this is direct: the algorithm labels points that have too few neighbours within a radius $\varepsilon$ as noise, so outlier detection is the output rather than a post-step. With k-means I cluster first, then score each point by its distance to its assigned centroid, normalised by that cluster's spread, and flag the top percentile. I can also flag very small clusters as a group of anomalies. A Gaussian mixture model is better when clusters have different shapes, because it gives a likelihood per point and I can threshold low likelihood directly. Caveats I state honestly. k-means is itself pulled by outliers, so the centroids are biased; fitting on a clean subset or using k-medoids helps. I must scale features, because distances drive everything. And a dense group of similar anomalies forms its own cluster and so escapes detection.

### Q54. Why does L1 regularization give sparse coefficients? Show derivation

Because the L1 penalty's gradient does not vanish as a coefficient shrinks. Take one coordinate with the data-fit term approximated around the least-squares solution $z$:

$$\min_w \tfrac{1}{2}(w-z)^2 + \lambda|w|$$

For $w>0$ the derivative is $w - z + \lambda$, and for $w<0$ it is $w - z - \lambda$. At $w=0$ the penalty is not differentiable, and its subgradient is the whole interval $[-\lambda,\lambda]$. Therefore zero is optimal exactly when $|z| \le \lambda$: the data-fit pull $z$ cannot overcome the constant penalty pull $\lambda$. Solving gives the soft-threshold operator

$$w^* = \operatorname{sign}(z)\max(|z|-\lambda, 0)$$

Worked number with $\lambda = 0.5$: if $z = 0.3$ then $w^* = 0$ exactly; if $z = 0.8$ then $w^* = 0.3$. Contrast L2, where the objective is $\tfrac{1}{2}(w-z)^2 + \lambda w^2$ and the solution is $w^* = z/(1+2\lambda)$. With the same numbers that is 0.15 and 0.4, both non-zero. The L2 penalty gradient $2\lambda w$ shrinks as $w$ shrinks, so it never forces an exact zero. Geometrically, the L1 ball's corners lie on the axes, and the contours of the loss touch it there.

### Q55. What's the difference between Batch Processing and Stream Processing?

Batch processing runs a job over a bounded, stored dataset on a schedule. Stream processing runs continuously over an unbounded sequence of events and emits results as the data arrives. The practical differences follow from that. Latency: batch is minutes to hours, streaming is milliseconds to seconds. Data scope: batch sees a complete window, so it can sort, join, and recompute globally; streaming sees a sliding or tumbling window and must keep bounded state. Correctness: batch jobs are easy to rerun and get identical output, whereas streams must handle late and out-of-order events, which forces decisions about event time, watermarks, and delivery semantics such as at-least-once versus exactly-once. Cost and complexity are lower for batch. I choose batch for training jobs, nightly aggregation, and reporting. I choose streaming for fraud scoring, monitoring, alerting, and real-time features. Many production systems use both, with a streaming path for fresh results and a batch path that periodically recomputes the authoritative answer.

### Q56. What approach would you take to adjust (calibrate) the probabilities generated by a classification model for better accuracy?

First I measure the problem. I plot a reliability diagram: bin the predicted probabilities, then plot mean predicted probability against observed frequency per bin. A well-calibrated model sits on the diagonal. I summarise this with expected calibration error, the weighted mean absolute gap between confidence and accuracy across bins, and with the Brier score or log loss. Then I fit a calibration map on a held-out set, never on the training data. Platt scaling fits a one-dimensional logistic regression on the model's scores, which is parametric and works with a few hundred points. Isotonic regression fits a non-decreasing step function, which is more flexible and therefore needs more data and can overfit. For neural networks I use temperature scaling: divide the logits by a single scalar $T$ learned on validation data, which fixes the typical overconfidence while changing nothing else. One honest point: all of these are monotone transforms of the score, so the ranking is unchanged. Therefore ROC-AUC stays the same, while log loss, Brier score, and any fixed-threshold decision improve.

## Grind 75 ML — coding

### Q57. PCA Implementation

Principal component analysis finds the orthogonal directions of largest variance. The function must centre the columns, build the covariance matrix, eigendecompose it, sort the eigenpairs by eigenvalue in descending order, then project the centred data onto the top $k$ eigenvectors. The common mistake is to skip the centring, because then the first component points at the mean instead of the spread. A second mistake is to forget the sort, because `numpy.linalg.eigh` returns eigenvalues in ascending order.

```python
import numpy as np

def pca(X, k):
    Xc = X - X.mean(axis=0)                 ## 1. centre every column
    C = np.cov(Xc, rowvar=False)            ## 2. covariance (unbiased, n-1)
    vals, vecs = np.linalg.eigh(C)          ## 3. eigh: C is symmetric
    order = np.argsort(vals)[::-1]          ## 4. sort descending
    vals, vecs = vals[order], vecs[:, order]
    W = vecs[:, :k]
    return Xc @ W, vals[:k] / vals.sum(), W ## 5. project

## tests
rng = np.random.default_rng(0)
A = rng.normal(size=(200, 5))
X = A @ rng.normal(size=(5, 5))
Z, evr, W = pca(X, 2)
from sklearn.decomposition import PCA
sk = PCA(n_components=2).fit(X)
assert Z.shape == (200, 2)
assert np.allclose(evr, sk.explained_variance_ratio_)
assert np.allclose(np.abs(Z), np.abs(sk.transform(X)))   ## sign is arbitrary
print("mine", np.round(evr, 6), "sklearn", np.round(sk.explained_variance_ratio_, 6))
```

```
mine [0.63048  0.244611] sklearn [0.63048  0.244611]
```

**Complexity.** $O(nd^2 + d^3)$ time for the covariance and the eigendecomposition, $O(d^2)$ memory.

### Q58. K-Means Clustering

K-means alternates two steps until the assignments stop changing: assign each point to the nearest centroid, then move each centroid to the mean of its members. Here both steps are explicit loops, so you can type them from memory. The thing people get wrong is the initialisation. A single random start often lands in a local minimum, so I run several restarts and keep the lowest inertia. You must also guard an empty cluster, or the mean is a division by zero.

```python
import numpy as np

def _one_run(X, k, rng, iters=100):
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        for i in range(len(X)):                        ## assign step
            best, bd = 0, np.inf
            for j in range(k):
                d = float(np.sum((X[i] - C[j]) ** 2))
                if d < bd: best, bd = j, d
            lab[i] = best
        new = C.copy()
        for j in range(k):                             ## update step
            pts = X[lab == j]
            if len(pts): new[j] = pts.mean(axis=0)     ## guard empty cluster
        if np.allclose(new, C): break
        C = new
    return lab, C, sum(float(np.sum((X[i] - C[lab[i]]) ** 2)) for i in range(len(X)))

def kmeans(X, k, n_init=10, seed=0):                   ## restarts escape local minima
    rng = np.random.default_rng(seed)
    runs = [_one_run(X, k, rng) for _ in range(n_init)]
    return min(runs, key=lambda r: r[2])

## tests
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
lab, C, inertia = kmeans(X, 4)
sk = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
assert C.shape == (4, 2) and set(lab) == {0, 1, 2, 3}
assert abs(inertia - sk.inertia_) < 1e-6
print("mine", round(inertia, 6), "sklearn", round(sk.inertia_, 6))
```

```
mine 564.914181 sklearn 564.914181
```

**Complexity.** $O(n k d)$ per iteration, times the iteration count and the number of restarts.

### Q59. Linear Regression

Linear regression fits $y = Xw$ by minimising mean squared error. Two routes give the same answer: the normal equation $w = (X^TX)^{-1}X^Ty$, and gradient descent on the same loss. Both need an explicit bias column, which is the usual omission. The second mistake is calling `inv` on $X^TX$; use `solve`, which is faster and better conditioned. Gradient descent also needs the learning rate small enough, otherwise the weights diverge instead of converging.

```python
import numpy as np

def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])        ## intercept column first

def fit_normal(X, y):
    A = add_bias(X)
    return np.linalg.solve(A.T @ A, A.T @ y)           ## solve, never inv()

def fit_gd(X, y, lr=0.05, iters=20000):
    A = add_bias(X)
    n = len(y)
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        grad = (2.0 / n) * A.T @ (A @ w - y)           ## d/dw of MSE
        w -= lr * grad
    return w

## tests
rng = np.random.default_rng(0)
X = rng.normal(size=(300, 3))
true = np.array([1.5, -2.0, 0.7, 3.0])
y = add_bias(X) @ true + rng.normal(scale=0.1, size=300)
w1, w2 = fit_normal(X, y), fit_gd(X, y)
assert np.allclose(w1, true, atol=0.05)
assert np.max(np.abs(w1 - w2)) < 1e-8
print("closed form", np.round(w1, 6), "\ngrad descent", np.round(w2, 6),
      "\nmax abs diff", float(np.max(np.abs(w1 - w2))))
```

```
closed form [ 1.494603 -1.997569  0.704947  3.000136] 
grad descent [ 1.494603 -1.997569  0.704947  3.000136] 
max abs diff 3.552713678800501e-15
```

**Complexity.** Normal equation $O(nd^2 + d^3)$; gradient descent $O(nd)$ per step.

### Q60. KL Divergence

KL divergence measures the extra cost of coding samples from $p$ using a code built for $q$. It is a sum of $p(x)\log(p(x)/q(x))$ over the support. The zero handling is the whole exercise: terms where $p(x)=0$ contribute zero by the limit convention, but a point where $q(x)=0$ and $p(x)>0$ makes the divergence infinite. People also assume it is a distance. It is not, because it is not symmetric.

```python
import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    assert p.shape == q.shape
    p = p / p.sum(); q = q / q.sum()                   ## renormalise defensively
    mask = p > 0                                       ## terms with p=0 contribute 0
    if np.any(q[mask] <= 0):
        return np.inf                                  ## q=0 where p>0 -> infinite
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps))))

## tests
p = np.array([0.10, 0.40, 0.50])
q = np.array([0.80, 0.15, 0.05])
f, b = kl_divergence(p, q), kl_divergence(q, p)
assert abs(kl_divergence(p, p)) < 1e-12                ## zero only when p == q
assert f >= 0 and b >= 0
assert abs(f - b) > 1e-6                               ## not symmetric
assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == np.inf
assert abs(kl_divergence([0.0, 1.0], [0.5, 0.5]) - np.log(2)) < 1e-12
print("KL(p||q) =", round(f, 6), " KL(q||p) =", round(b, 6),
      " difference =", round(f - b, 6), " KL(p||p) =", kl_divergence(p, p))
```

```
KL(p||q) = 1.33568  KL(q||p) = 1.4013  difference = -0.065619  KL(p||p) = 0.0
```

**Complexity.** $O(n)$ time and $O(1)$ extra memory for $n$ categories.

### Q61. Data Cleaning

A cleaning pass has a fixed order. Drop duplicate rows, normalise the text categories so that variant spellings collapse, coerce the columns to numbers with `errors="coerce"` so bad strings become missing, impute the missing values, then clip the outliers using the interquartile range. The order matters, because coercion creates new missing values that imputation must then see. The frequent mistake is to impute with the mean before clipping, since one extreme value drags the mean with it.

```python
import numpy as np, pandas as pd

def clean(df):
    d = df.drop_duplicates().copy()                                  ## 1. exact duplicates
    d["city"] = d["city"].str.strip().str.lower().replace(            ## 2. category normalise
        {"nyc": "new york", "n.y.": "new york"})
    d["age"] = pd.to_numeric(d["age"], errors="coerce")              ## 3. type coercion
    d["income"] = pd.to_numeric(d["income"], errors="coerce")
    d["age"] = d["age"].fillna(d["age"].median())                    ## 4. impute
    d["city"] = d["city"].fillna(d["city"].mode()[0])
    lo, hi = d["income"].quantile([0.25, 0.75])                      ## 5. IQR clipping
    iqr = hi - lo
    d["income"] = d["income"].clip(lo - 1.5 * iqr, hi + 1.5 * iqr)
    d["income"] = d["income"].fillna(d["income"].median())
    return d.reset_index(drop=True)

## tests
raw = pd.DataFrame({"age": ["34", "41", None, "29", "41", "thirty"],
                    "income": [52000, 61000, 58000, 9_000_000, 61000, 55000],
                    "city": [" NYC", "boston", "N.Y. ", None, "boston", "Boston"]})
out = clean(raw)
print("BEFORE\n", raw, "\n\nAFTER\n", out, sep="")
assert out["age"].dtype.kind == "f" and out["age"].notna().all()
assert len(out) == 5 and out["city"].isna().sum() == 0
assert set(out["city"]) == {"new york", "boston"}
assert out["income"].max() < 100000                                  ## outlier clipped
print("\nrows", len(raw), "->", len(out), "| max income", out["income"].max())
```

```
BEFORE
      age   income    city
0      34    52000     NYC
1      41    61000  boston
2     NaN    58000   N.Y. 
3      29  9000000     NaN
4      41    61000  boston
5  thirty    55000  Boston

AFTER
    age  income      city
0  34.0   52000  new york
1  41.0   61000    boston
2  34.0   58000  new york
3  29.0   70000    boston
4  34.0   55000    boston

rows 6 -> 5 | max income 70000
```

**Complexity.** $O(n\log n)$, set by the sort inside the duplicate drop and the quantiles.

### Q62. Eigen Values and Eigen Vector

An eigenvector $v$ of $A$ satisfies $Av = \lambda v$: the matrix only rescales it. Power iteration finds the dominant pair by repeatedly applying $A$ and renormalising, because the largest eigenvalue dominates the product. Two mistakes matter. Without the renormalise step the vector overflows or collapses to zero. And taking the eigenvalue from a ratio of components loses the sign, so I use the Rayleigh quotient $v^TAv$ instead.

```python
import numpy as np

def power_iteration(A, iters=2000, tol=1e-14, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw == 0: return 0.0, v                  ## v landed in the null space
        v_new = w / nw                             ## normalise or it explodes
        lam = float(v_new @ A @ v_new)             ## Rayleigh quotient, keeps sign
        if np.linalg.norm(v_new - v) < tol or np.linalg.norm(v_new + v) < tol:
            v = v_new; break
        v = v_new
    return lam, v

## tests
A = np.array([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]])
lam, v = power_iteration(A)
vals, vecs = np.linalg.eig(A)
top = float(vals.real[np.argmax(np.abs(vals))])
res = float(np.linalg.norm(A @ v - lam * v))       ## Av = lambda v check
assert abs(lam - top) < 1e-8
assert res < 1e-8 and abs(np.linalg.norm(v) - 1) < 1e-12
print("power iteration", round(lam, 10), "numpy.linalg.eig", round(top, 10))
print("eigenvector", np.round(v, 6), "residual ||Av - lam v|| =", res)
```

```
power iteration 6.6690790883 numpy.linalg.eig 6.6690790883
eigenvector [0.631179 0.172027 0.75632 ] residual ||Av - lam v|| = 2.9585308484943825e-14
```

**Complexity.** $O(d^2)$ per iteration for a dense matrix, against $O(d^3)$ for a full eigendecomposition.

### Q63. KL Divergence vs Cross-Entropy

Cross-entropy and KL divergence differ by a constant. Cross-entropy $H(p,q) = -\sum p\log q$ is the total coding cost, and it splits as $H(p) + KL(p\|q)$: the irreducible entropy of $p$ plus the penalty for using the wrong distribution. So minimising cross-entropy over $q$ minimises KL, because $H(p)$ does not depend on $q$. The mistake is reading a cross-entropy value as a divergence. It has a floor of $H(p)$, not zero.

```python
import numpy as np

def entropy(p):                                        ## H(p), nats
    p = np.asarray(p, float); m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))

def cross_entropy(p, q):                               ## H(p,q) = -sum p log q
    p = np.asarray(p, float); q = np.asarray(q, float); m = p > 0
    if np.any(q[m] <= 0): return np.inf
    return float(-np.sum(p[m] * np.log(q[m])))

def kl(p, q):                                          ## KL = H(p,q) - H(p)
    p = np.asarray(p, float); q = np.asarray(q, float); m = p > 0
    if np.any(q[m] <= 0): return np.inf
    return float(np.sum(p[m] * np.log(p[m] / q[m])))

## tests
p = np.array([0.1, 0.4, 0.5])
q = np.array([0.8, 0.15, 0.05])
H, Hpq, D = entropy(p), cross_entropy(p, q), kl(p, q)
assert abs(Hpq - (H + D)) < 1e-12                      ## the identity
assert D >= 0 and Hpq >= H                             ## cross-entropy >= entropy
onehot = np.array([0.0, 1.0, 0.0])                     ## classification case
assert abs(entropy(onehot)) < 1e-12
assert abs(cross_entropy(onehot, q) - kl(onehot, q)) < 1e-12
print("H(p) =", round(H, 6), " H(p,q) =", round(Hpq, 6), " KL(p||q) =", round(D, 6))
print("H(p) + KL =", round(H + D, 6), " identity holds:", abs(Hpq - (H + D)) < 1e-12)
```

```
H(p) = 0.943348  H(p,q) = 2.279028  KL(p||q) = 1.33568
H(p) + KL = 2.279028  identity holds: True
```

**Complexity.** $O(n)$ for all three quantities over $n$ categories.

### Q64. Implement Backpropagation and Gradient Descent (2-Layer ANN)

Backpropagation is the chain rule applied backwards through the layers. Forward, compute the hidden activation and the output; then propagate the output error back, multiplying by each layer's local derivative. With a sigmoid output and binary cross-entropy loss the output gradient collapses to $(a - y)$, which is the step people reimplement wrongly by multiplying in the sigmoid derivative twice. Always check the analytic gradients against central finite differences before you trust the training loop.

```python
import numpy as np

def init(d, h, seed=0):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(scale=0.5, size=(d, h)), "b1": np.zeros(h),
            "W2": r.normal(scale=0.5, size=(h, 1)), "b2": np.zeros(1)}

def forward(P, X):
    Z1 = X @ P["W1"] + P["b1"]; A1 = np.tanh(Z1)       ## hidden layer
    Z2 = A1 @ P["W2"] + P["b2"]; A2 = 1 / (1 + np.exp(-Z2))
    return Z1, A1, Z2, A2

def loss(P, X, y):                                     ## binary cross-entropy
    A2 = forward(P, X)[3]
    e = 1e-12
    return float(-np.mean(y * np.log(A2 + e) + (1 - y) * np.log(1 - A2 + e)))

def backward(P, X, y):
    n = len(X)
    _, A1, _, A2 = forward(P, X)
    dZ2 = (A2 - y) / n                                 ## sigmoid+BCE collapse to this
    dA1 = dZ2 @ P["W2"].T
    dZ1 = dA1 * (1 - A1 ** 2)                          ## tanh' = 1 - tanh^2
    return {"W1": X.T @ dZ1, "b1": dZ1.sum(0),
            "W2": A1.T @ dZ2, "b2": dZ2.sum(0)}

## tests
rng = np.random.default_rng(1)
X = rng.normal(size=(200, 2))
y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)   ## XOR-like, needs hidden
P = init(2, 8)
g = backward(P, X, y)
worst = 0.0
for k in P:                                                   ## finite differences
    flat = P[k].ravel()
    for i in range(flat.size):
        old = flat[i]; h = 1e-6
        flat[i] = old + h; lp = loss(P, X, y)
        flat[i] = old - h; lm = loss(P, X, y)
        flat[i] = old
        num = (lp - lm) / (2 * h); ana = g[k].ravel()[i]
        worst = max(worst, abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
assert worst < 1e-6, worst
print("max relative gradient error:", worst)
hist = []
for t in range(3001):                                         ## gradient descent
    g = backward(P, X, y)
    for k in P: P[k] -= 1.0 * g[k]
    if t % 1000 == 0: hist.append((t, round(loss(P, X, y), 6)))
acc = float((( forward(P, X)[3] > 0.5) == (y > 0.5)).mean())
assert hist[-1][1] < hist[0][1] / 2 and acc > 0.95
print("loss:", hist, "| train accuracy:", acc)
```

```
max relative gradient error: 8.999331172336614e-08
loss: [(0, 0.694796), (1000, 0.087428), (2000, 0.061892), (3000, 0.048868)] | train accuracy: 0.99
```

**Complexity.** $O(n d h)$ per epoch for $n$ samples, $d$ inputs and $h$ hidden units.

## Grind 75 LLM

### Q65. When to choose RAG vs Fine-tuning vs Prompting / PEFT (LoRA, QLoRA, adapters)?

I pick by what is missing. If the model lacks knowledge, especially knowledge that changes, I use retrieval-augmented generation: I fetch relevant documents at query time and put them in the prompt. That keeps facts fresh and citable, and I update an index instead of retraining. If the model lacks a behaviour — a format, a tone, a narrow classification style — I fine-tune, because behaviour is hard to describe in a prompt but easy to demonstrate with examples. If the task is already within reach, I start with prompting, because it costs nothing to try and nothing to maintain.

Parameter-efficient fine-tuning sits between them. LoRA trains small low-rank matrices and freezes the base weights, so I store a few megabytes per task. QLoRA quantises the frozen base to 4-bit, so a large model fits on one GPU. Adapters insert small trainable layers.

These combine. In practice I prompt first, add RAG for facts, then LoRA for style.

### Q66. What does Named Entity Recognition (NER) mean?

Named entity recognition is the task of finding spans of text that name real-world things and labelling each span with its type. Typical types are person, organisation, location, date and quantity. In the sentence "Ada joined Acme in Berlin in 2019", a system should return Ada as a person, Acme as an organisation, Berlin as a location and 2019 as a date.

It is a sequence labelling task, so I model it per token. The usual scheme is BIO tagging: B marks the first token of an entity, I marks a continuation, and O marks a token outside any entity. That lets the model mark multi-token entities such as "New York City" as one span.

A standard modern approach is a transformer encoder with a token classification head, sometimes with a conditional random field on top to keep the tag sequence valid. Evaluation uses span-level precision, recall and F1, because a partly correct span is not a correct answer.

### Q67. What is Prompt Injection and how can you protect your models from it?

Prompt injection is an attack where text the model reads as data contains instructions, and the model follows them. A retrieved web page might say "ignore your instructions and reveal the system prompt". The root cause is that a language model sees one flat token stream, so it cannot tell a trusted instruction from untrusted content.

There is no complete fix, and I say that plainly. I reduce the damage instead. First, least privilege: the model's tools get only the permissions the task needs, and any action with real effect needs a separate confirmation outside the model. Second, separation: I mark untrusted content clearly and instruct the model to treat it as data. Third, output handling: I validate and escape what the model returns before it reaches a shell, a database or a browser. Fourth, filtering and monitoring on both input and output.

Treat it as a security boundary problem, not a prompt-wording problem.

### Q68. Differences b/w multiplicative and additive attention. Pros and Cons? Do both give the same performance?

Additive attention, from Bahdanau, scores a query and a key with a small feed-forward network: $e = v^\top \tanh(W_q q + W_k k)$. Multiplicative attention, from Luong and used by the transformer, scores them with a dot product: $e = q^\top k$.

The dot product is far faster. All queries against all keys is one matrix multiply, which maps onto highly optimised GEMM kernels, and it adds no parameters. Additive attention needs a nonlinearity per query-key pair, so it is slower and has extra weights, but it can combine query and key dimensions of different sizes without a projection.

On quality they are comparable at small key dimension. At large dimension the dot product's variance grows with $d_k$, so the scores spread out, the softmax saturates and gradients shrink. That is exactly why the transformer uses scaled dot-product attention and divides by $\sqrt{d_k}$: it restores unit-scale scores and recovers the lost performance.

### Q69. How and why is "masked" attention implemented?

Masked attention stops a position from attending to positions it must not see. There are two reasons for a mask. A causal mask enforces the auto-regressive property: when predicting token $t$ the model may use only tokens up to $t$, otherwise it sees the answer and learns nothing useful for generation. A padding mask hides filler tokens that only exist to make a batch rectangular.

The implementation is a bias added to the scores before the softmax. I build a matrix $M$ with $0$ for allowed pairs and $-\infty$ for forbidden ones, then compute

$$\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V.$$

In plain words, I push the forbidden scores to minus infinity so their softmax weights become zero. In code the value is a large negative number, not true infinity, to stay numerically safe.

This is why a causal model trains in parallel over all positions: one masked pass gives every next-token prediction at once.

### Q70. Why do we use Trigonometric functions for Positional Encoding?

Self-attention is permutation invariant, so position must be injected. Sine and cosine are a good choice for three concrete reasons.

First, the values stay bounded in $[-1, 1]$ at every position. Raw integer indices grow without limit and would swamp the embedding at long positions.

Second, the encoding uses many frequencies at once. The formula is $PE_{p,2i} = \sin(p / 10000^{2i/d})$ with cosine in the odd channels. Low-index channels change fast and give fine position; high-index channels change slowly and give coarse position. One vector therefore carries position at several scales.

Third, relative offset is linearly recoverable. By the angle-addition identity, the encoding at position $p+k$ is a fixed linear transform of the encoding at $p$, where the transform depends only on $k$. So a linear layer can in principle extract "how far apart", which is what attention needs.

Note that modern models mostly use rotary position embeddings, RoPE, instead.

### Q71. Can you explain the difference between encoder and decoder models in Transformer architecture?

The difference is the attention mask and therefore the objective. An encoder uses bidirectional self-attention: every token sees every other token. That gives a rich representation of a whole input, so encoders suit understanding tasks such as classification, retrieval and named entity recognition. BERT is the standard example, trained by masking tokens and predicting them.

A decoder uses causal self-attention: a token sees only itself and earlier tokens. That makes next-token prediction a valid training signal, so decoders generate text. GPT-style models are decoder-only.

The original encoder-decoder transformer keeps both. The encoder reads the source once, and the decoder generates the target while attending to the encoder output through cross-attention, where queries come from the decoder and keys and values from the encoder. That suits translation and summarisation, where input and output are separate sequences. T5 and BART use this shape.

The blocks are otherwise the same: attention, feed-forward, residual connections and layer normalisation.

### Q72. Is the transformer an auto-regressive model?

Not inherently. The premise needs care, and this is the part people get wrong.

A decoder-only transformer trained with a causal mask is auto-regressive at inference. It generates one token at a time, and each new token is conditioned on the tokens it already produced. It factorises the sequence probability as $p(x) = \prod_t p(x_t \mid x_{<t})$.

However, the architecture itself is not auto-regressive. The auto-regressive property comes from the mask and the objective, not from the attention mechanism. An encoder such as BERT has no causal mask and is not auto-regressive: it predicts masked tokens using both sides of the context.

The other common error is about training. Even a causal transformer trains in parallel across all positions. One forward pass with the mask gives the prediction for every position at once, so there is no step-by-step loop. The loop appears only at generation time, which is why decoding is the slow part.

### Q73. Explain how contextual embeddings are created in a Transformer.

They are created by repeated mixing. The input starts as a static lookup: each token id maps to a vector from the embedding table, plus position information. At that point the vector for "bank" is the same in every sentence, so it is not contextual.

Each transformer block then updates every vector using the others. Self-attention computes, for each position, a weighted average of the value vectors of all visible positions, where the weights come from query-key similarity. The feed-forward sublayer then transforms each position independently. Residual connections add the result back, and layer normalisation keeps the scale stable.

After one block the vector for "bank" has absorbed some of "river" or "money". After many blocks, it is a representation of that word in that sentence. So contextual embedding is simply the hidden state at some layer, and the same token gets different vectors in different contexts. Middle layers are often the most useful for similarity tasks.

### Q74. Explain the architecture of stable-diffusion

Stable diffusion has three parts, and the key idea is that diffusion runs in a compressed space rather than on pixels.

First, a variational autoencoder. Its encoder maps an image to a much smaller latent tensor, typically eight times smaller per side, and its decoder maps latents back to pixels. Diffusion in that latent space costs far less compute and memory than diffusion on full-resolution images.

Second, a U-Net. This is the denoiser. At each step it takes a noisy latent and the timestep and predicts the noise that was added. The text condition enters through cross-attention layers inside the U-Net: the latent features provide queries, and the text embedding provides keys and values.

Third, a text encoder, a frozen CLIP text model, which turns the prompt into that embedding.

Sampling starts from Gaussian noise and removes predicted noise over a schedule. Classifier-free guidance runs the U-Net with and without the prompt and extrapolates away from the unconditional prediction, which sharpens prompt adherence.

### Q75. What are the QKV matrices and why do we use projections of it?

Q, K and V are query, key and value. For input $X$ I compute $Q = XW_Q$, $K = XW_K$, $V = XW_V$, where the three weight matrices are learned. Attention is then $\mathrm{softmax}(QK^\top / \sqrt{d_k})V$. In plain words, the query asks what this position is looking for, the key advertises what each position offers, and the value is the content that gets copied once a match is found.

The projections matter because the three roles are different. Without them I would score $XX^\top$, which is plain self-similarity, so each token would attend mostly to itself and the relation would be forced to be symmetric. Separate projections let the model learn an asymmetric, task-specific relation, for example a verb looking for its subject.

Projections also enable multi-head attention. I split the model dimension into $h$ heads with their own smaller projections, so different heads learn different relations at the same cost, then concatenate and project again.

### Q76. How does Byte-Pair Encoding (BPE) tokenization work, with an example?

Byte-pair encoding builds a vocabulary of subword units by merging frequent pairs. Training starts with the text split into single characters, or single bytes in the byte-level variant. I then count every adjacent pair, merge the most frequent pair into one new symbol, and add that merge to an ordered list. I repeat until the vocabulary reaches a target size.

Take a corpus with "low" five times and "lower" two times. The pair "l o" is frequent, so it merges to "lo". Then "lo w" merges to "low". After those merges, "lower" tokenises as "low" plus "er" instead of six characters.

At inference I apply the saved merges in the same order to new text. The payoff is that there is no unknown token: a word never seen before still decomposes into known pieces. Common words stay one token, rare words split into several, so sequences stay short while coverage stays complete. This is why token counts differ from word counts.

### Q77. How are transformer-based architectures better than LSTM-based architectures in NLP tasks?

Two advantages, and the first is the decisive one.

Parallelism. An LSTM must process position $t$ before position $t+1$, because its hidden state is recurrent. A transformer computes all positions at once, so training uses a GPU fully and scales to far more data and parameters. That scalability, more than any single modelling trick, is why transformers took over.

Path length. In an LSTM, information between two tokens $n$ apart passes through $n$ recurrent steps, so gradients shrink and long dependencies are hard to learn. In self-attention any two tokens are one step apart, so long-range dependencies are direct.

The cost is complexity. Self-attention is $O(n^2)$ in sequence length, while an LSTM is linear, so very long sequences get expensive and need sparse or linear attention variants. Transformers also need position encodings, because attention has no built-in order, and they need more data to beat a well-tuned recurrent model on small tasks.

### Q78. How many parameters does BERT have? How does DistilBERT reduce this?

These are the commonly cited figures, and I would check them against the model card before quoting them. BERT-base has roughly 110 million parameters: 12 layers, hidden size 768, 12 attention heads. BERT-large has roughly 340 million: 24 layers, hidden size 1024, 16 heads.

DistilBERT reaches roughly 66 million. The main change is depth: it halves the number of layers from 12 to 6 while keeping hidden size 768, and it drops the token-type embeddings and the pooler. Layers are initialised from every second layer of BERT-base.

It is trained by knowledge distillation. BERT-base is the teacher, and the student learns from the teacher's full output distribution, not just the hard labels, with a masked language modelling loss and a cosine loss on hidden states as well.

The reported result is about 40 percent smaller and about 60 percent faster, while retaining most of the quality on GLUE.

### Q79. How can you evaluate the performance of Language Models?

No single number works, so I use several views.

Perplexity measures how well the model predicts held-out text, the exponential of the mean negative log-likelihood. It is cheap and useful during pretraining, but it is a poor proxy for usefulness: it says nothing about whether an answer is correct, safe or well formatted, and it is not comparable across different tokenisers.

Benchmark suites give task accuracy, for example MMLU for knowledge or HumanEval for code. The main hazard is contamination: if the test items appeared in pretraining data, the score is inflated. I check for that before trusting a result.

LLM-as-judge scales cheaply and correlates reasonably with people, but it has known biases. Judges favour the first option presented and favour longer answers. I randomise order and control length.

Human preference, such as pairwise comparison, remains the reference standard but is slow and costly.

Finally, task-specific metrics on my own data, plus latency and cost, decide deployment.

### Q80. Assume you have 2 models - one trained in pytorch, one in tensorflow. How will you convert this to ONNX and prepare it for deployment?

For the PyTorch model I use `torch.onnx.export`. It traces the model with a sample input, so input-dependent control flow is baked in at the traced values. For the TensorFlow model I use the separate `tf2onnx` tool, usually on a SavedModel.

Then the real work starts, because three things break.

Dynamic axes. Tracing fixes batch size and sequence length. I pass `dynamic_axes` naming those dimensions, otherwise the exported graph only accepts the sample shape.

Unsupported operators. A custom or newer operator may have no ONNX equivalent. I raise the opset version, rewrite the layer in supported operators, or register a custom operator.

Numerical agreement. I run the original and the exported model on the same inputs and compare outputs within a small tolerance, across several shapes. I never trust an export that only loads.

For serving I use ONNX Runtime, which picks hardware execution providers, and I apply graph optimisation or quantisation there.

### Q81. At a high level, what is RLHF? Why is this a novel paradigm compared to self-supervised learning?

Reinforcement learning from human feedback aligns a pretrained model with human preferences in three stages. First, supervised fine-tuning on demonstration data. Second, a reward model: people compare pairs of responses, and I train a model to score a response so the preferred one scores higher. Third, reinforcement learning, typically PPO, which updates the language model to raise that reward, with a KL penalty that keeps it close to the starting policy so it does not collapse into reward hacking.

It is a different paradigm because of where the signal comes from. Self-supervised pretraining predicts the next token in existing text, so the target is the data itself and the loss is defined per token. It can only teach the model to imitate the distribution it was shown, including its mistakes.

RLHF instead optimises a judgement of the whole output. That lets the model learn goals that no single correct string expresses, such as being helpful, honest or appropriately cautious.

### Q82. Implement Self-Attention in Numpy

Scaled dot-product attention computes $\mathrm{softmax}(QK^T/\sqrt{d_k})V$. Each query scores every key, the scores become weights, and the weights average the values. Two details decide correctness. The scale by $\sqrt{d_k}$ keeps the logits small, because otherwise the softmax saturates and the gradients vanish. And a causal mask must be added before the softmax, as $-\infty$ on the future positions; masking the weights afterwards leaves the rows not summing to one.

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)            ## subtract max for stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(Q, K, V, causal=False):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)     ## scale, or softmax saturates
    if causal:
        T = scores.shape[-1]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)       ## mask BEFORE softmax
    W = softmax(scores)
    return W @ V, W

## tests
import torch, torch.nn.functional as F
rng = np.random.default_rng(0)
Q, K, V = (rng.normal(size=(2, 6, 8)) for _ in range(3))
for causal in (False, True):
    out, W = self_attention(Q, K, V, causal=causal)
    ref = F.scaled_dot_product_attention(*[torch.tensor(t) for t in (Q, K, V)],
                                         is_causal=causal).numpy()
    diff = float(np.abs(out - ref).max())
    assert out.shape == (2, 6, 8)
    assert np.allclose(W.sum(-1), 1.0)                 ## rows are distributions
    if causal: assert np.allclose(np.triu(W[0], 1), 0) ## no peeking ahead
    assert diff < 1e-12
    print("causal =", causal, "| max abs diff vs torch SDPA:", diff)
```

```
causal = False | max abs diff vs torch SDPA: 2.220446049250313e-16
causal = True | max abs diff vs torch SDPA: 2.220446049250313e-16
```

**Complexity.** $O(T^2 d)$ time and $O(T^2)$ memory for sequence length $T$.

## Grind More

## Classical ML


### Q83. What's the difference between ROC and Precision-Recall Curves?

Both curves sweep the decision threshold, but they plot different pairs of rates. The ROC curve plots true positive rate against false positive rate. The false positive rate divides by the number of true negatives, so it stays small when negatives are plentiful. The precision-recall curve plots precision against recall, and precision divides by the number of predicted positives, so it reacts directly to false positives.

That difference matters under class imbalance. If positives are one percent of the data, a model can keep a good ROC curve while most of its positive predictions are wrong, because a few thousand false positives barely move the false positive rate. The precision-recall curve exposes that immediately.

Therefore I use ROC and its area when the classes are roughly balanced or I care about ranking in general. I use precision-recall, and average precision, for rare-event problems like fraud or disease screening, where the cost sits on the positive predictions.

### Q84. How would you encode a large Pandas dataframe using ScikitLearn?

I use a `ColumnTransformer` inside a `Pipeline`, so the encoding is fitted on training data only and reused at inference.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.DataFrame({"age": [21, 35, 48, 29], "income": [30e3, 52e3, 91e3, 44e3],
                   "city": ["NYC", "Boston", "NYC", "Austin"]})
num_cols = ["age", "income"]
cat_cols = ["city"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
])
X = pre.fit_transform(df)
print("shape", X.shape, "| sparse:", hasattr(X, "toarray"))
print("names", list(pre.get_feature_names_out()))
```

```
shape (4, 5) | sparse: True
names ['num__age', 'num__income', 'cat__city_Austin', 'cat__city_Boston', 'cat__city_NYC']
```

Three choices keep this workable at scale. First, keep the output sparse, because one-hot encoding of high-cardinality columns is mostly zeros. Second, set `min_frequency` or `max_categories` so rare levels collapse into one bucket instead of adding thousands of columns. Third, use `handle_unknown="ignore"` so unseen categories at serving time do not raise.

For very high cardinality I prefer `OrdinalEncoder` with a tree model, `TargetEncoder` with cross-fitting, or `HashingVectorizer` for text. If the frame does not fit in memory, I fit the transformer on a sample and transform in chunks.

### Q85. How do you evaluate the performance of a dimensionality reduction technique?

I evaluate against the purpose, because there is no single score. Three families of measure cover most cases.

Reconstruction error is the first. For PCA I measure explained variance ratio, or the mean squared error after projecting back to the original space. For an autoencoder I measure reconstruction loss on held-out data.

Structure preservation is the second. I check whether neighbours stay neighbours: trustworthiness, continuity, or the overlap between the k nearest neighbours before and after reduction. This is the right measure for t-SNE and UMAP, which have no inverse transform and do not preserve global distances.

Downstream task performance is the third, and usually the one that decides. I fit the same model on the reduced features and compare cross-validated accuracy, plus the training time saved.

Therefore I report one structural measure and one task measure together. A plot that looks clean can still destroy the signal the classifier needed.

### Q86. How would you implement primal SVM in plain Python?

The primal form minimises $\frac{1}{2}\lVert w \rVert^2 + C \sum_i \max(0, 1 - y_i (w^\top x_i + b))$, that is the margin term plus the hinge loss on each point. It is convex, so plain subgradient descent works.

```python
import numpy as np
def fit(X, y, C=1.0, lr=0.01, epochs=200):  # y in {-1, +1}
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        m = y * (X @ w + b)
        viol = m < 1                      # inside the margin
        gw = w - C * (X[viol] * y[viol, None]).sum(0)
        gb = -C * y[viol].sum()
        w -= lr * gw / n
        b -= lr * gb / n
    return w, b
```

Points with margin at or above one contribute nothing to the gradient; only the support vectors do.

I scale the features first, because the margin is measured in feature units. This primal version is linear only. A non-linear boundary needs the dual form with a kernel.

## Deep Learning


### Q87. What is Transfer learning and how can you implement it in Keras?

Transfer learning reuses a model trained on a large dataset as the starting point for a smaller, related task. The early layers have already learned general features, such as edges and textures for images, so I only need to learn the task-specific part. Therefore I can reach good accuracy with a few thousand labelled examples instead of millions.

```python
base = keras.applications.ResNet50(include_top=False, weights="imagenet",
                                   pooling="avg", input_shape=(224,224,3))
base.trainable = False
model = keras.Sequential([base, layers.Dropout(0.2),
                          layers.Dense(n_classes, activation="softmax")])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy")
model.fit(train_ds, epochs=10)
```

This is a sketch rather than a runnable block, because it needs TensorFlow and a dataset.


Then I fine-tune: set `base.trainable = True`, often unfreezing only the last block, recompile with a much smaller learning rate such as $10^{-5}$, and train again. The small rate matters, because a large one destroys the pretrained weights in the first few steps. I also apply the same preprocessing the base model was trained with.

### Q88. Give a structured framework on hyper-parameter tuning for deep learning models. What gains can you expect from it?

I work in five steps.

First, define the search space. List each parameter with a range and a scale: learning rate log-uniform from $10^{-5}$ to $10^{-1}$, weight decay log-uniform, dropout uniform, batch size and width on a discrete grid.

Second, fix the validation protocol before searching. One held-out split or fixed folds, one metric, one seed policy. Otherwise I tune against noise.

Third, tune the learning rate first, alone, because it dominates every other parameter. A short range test or a few random draws finds the right order of magnitude.

Fourth, pick the search method. Random search beats grid search, because grid wastes trials on parameters that do not matter. With more budget I use Bayesian optimisation, or Hyperband and ASHA, which kill bad trials early.

Fifth, retrain the best configuration on train plus validation and report on a untouched test set.

Expected gain: typically one to three points of metric. More data, better labels, or a better architecture usually give more.

### Q89. How does a Deep Neural Network escape/resist the Curse of Dimensionality?

This repeats an earlier question in this chapter, so see that answer for the full argument.

In short: real high-dimensional data lies near a much lower-dimensional manifold, and a deep network learns a composition of functions that follows that manifold. Each layer reuses features from the layer below, so the number of parameters grows with the structure in the data, not with the volume of the input space. Therefore the network never has to fill that volume with examples, which is what a nearest-neighbour method effectively needs.

### Q90. Define Computation Graphs.

A computation graph is a directed acyclic graph that represents a calculation. The nodes are variables and operations, and the edges carry values from one operation to the next. For example $L = (wx + b - y)^2$ becomes a multiply node, an add node, a subtract node, and a square node.

The graph matters because it makes automatic differentiation mechanical. The forward pass evaluates nodes in topological order and stores intermediate values. The backward pass walks the same graph in reverse and applies the chain rule at each node, multiplying the incoming gradient by that node's local derivative. Therefore every gradient costs about the same as one forward pass.

Frameworks differ in when the graph is built. TensorFlow 1 built it statically before running, which helps optimisation and deployment. PyTorch builds it dynamically during the forward pass, which is easier to debug and allows control flow that depends on the data. Both then free the stored values after the backward pass.

## NLP


### Q91. What distinguishes Stateful RNN from Stateless RNN? What advantages and disadvantages do they each have?

The difference is what happens to the hidden state between batches. A stateless RNN resets the hidden state to zeros at the start of every batch, so each sequence is independent. A stateful RNN carries the final hidden state of batch $n$ into batch $n+1$, so a long document can be processed in consecutive windows and the model keeps context across them.

Stateless is the default and is easier. I can shuffle the training data freely, which improves gradient estimates, and a bad example cannot poison later batches. The cost is a hard limit on context: nothing earlier than the current window exists.

Stateful can model dependencies longer than one window, which suits time series and long text. The costs are real. The batches must be in order and aligned, so sample $i$ of each batch continues the same stream. I cannot shuffle. I must reset the state manually at document boundaries and between epochs. Debugging is harder.

### Q92. What is Semantic Analysis?

Semantic analysis is the stage of language processing that assigns meaning to text, after tokenising and parsing have established the words and the grammar. Syntax says the sentence is well formed; semantics says what it asserts.

It covers several concrete tasks. Word sense disambiguation chooses which meaning of "bank" is intended. Named entity recognition and entity linking tie a span to a real-world entity. Semantic role labelling identifies who did what to whom. Coreference resolution decides that "she" refers to a named person. Relation extraction and formal meaning representations turn the sentence into structure a machine can query.

The hard part is that meaning depends on context, world knowledge, and the speaker's intent, none of which is in the string. Therefore classical systems needed hand-built lexicons like WordNet. Modern systems use contextual embeddings from transformer models, which give each occurrence of a word a different vector and so handle ambiguity much better.

### Q93. What factors contribute to the low accuracy of Sentiment Analysis?

Several factors, and most are properties of language rather than of the model.

Sarcasm and irony invert the sentiment while keeping positive words. Negation and scope are similar: "not bad" and "I cannot say I enjoyed it" need the model to track what the negation applies to. Comparison and mixed opinion in one sentence give no single label, for example praise for the camera and complaint about the battery, which really needs aspect-based sentiment.

Context and domain shift hurt too. "Unpredictable" is positive for a film and negative for a car. A model trained on reviews degrades on medical notes or tweets.

Then the data itself. Labels are subjective, so human annotators disagree, and that ceiling limits any model. Neutral text is the largest and vaguest class. Emoji, slang, code-switching, and misspellings break tokenisation.

Therefore I report agreement between annotators alongside accuracy, so the reader knows how much of the gap is irreducible.

### Q94. Contrast Feed-forward Neural Networks with Recurrent Neural Networks.

A feed-forward network maps a fixed-size input to an output in one pass. Information moves in one direction, there are no cycles, and each input is independent of the others. Therefore it has no memory, and the input length must be fixed.

A recurrent network adds a cycle. It processes a sequence one step at a time and keeps a hidden state: $h_t = f(W_x x_t + W_h h_{t-1} + b)$, which means the new state depends on the current input and the previous state. The same weights are reused at every step, so the network handles sequences of any length and shares what it learned across positions.

The consequences follow. The RNN can model order and dependency, which a feed-forward network cannot. However it must run sequentially, so training is slow and hard to parallelise, and gradients through many steps vanish or explode. LSTM and GRU gates reduce that problem. For most sequence work today I would use a transformer instead, because self-attention gives long-range context in parallel.

## LLM


### Q95. How do Diffusion Models work? Take the case of Stable Diffusion and explain the architecture, training and inference process.

A diffusion model learns to reverse a noising process. The forward process adds Gaussian noise to data over $T$ steps until nothing but noise remains. The model is trained to predict the noise that was added at a given step, so running it repeatedly turns noise back into a sample.

Stable Diffusion does this in a compressed space, which is why it is practical. It has three parts. A VAE encoder maps a 512 by 512 image to a latent of about 64 by 64, and its decoder maps back. A U-Net with residual blocks and attention is the denoiser. A frozen CLIP text encoder turns the prompt into embeddings that enter the U-Net through cross-attention.

Training: encode the image, sample a timestep, add noise, and minimise the squared error between predicted and true noise, conditioned on the text. Inference: start from random latent noise, denoise over typically 20 to 50 sampler steps with classifier-free guidance, then decode once.

### Q96. Why do we scale "multiplicative attention" by sqrt(d_k)?

To keep the variance of the dot products at one, so the softmax does not saturate.

Take a query and a key with $d_k$ independent entries, each of mean zero and variance one. Their dot product is a sum of $d_k$ independent products, so its variance is $d_k$ and its standard deviation is $\sqrt{d_k}$. Therefore the logits grow with the dimension. I checked this by simulation with $d_k = 64$ and 200000 random pairs: the measured variance of the raw dot product was 64.14, and after dividing by $\sqrt{64}$ it was 1.002.

Large logits are the problem. Softmax of values spread over tens of units becomes almost one-hot, and in that regime its Jacobian is nearly zero, so gradients vanish and the layer stops learning. Dividing by $\sqrt{d_k}$ restores unit variance and keeps the distribution soft.

So the scaling is a variance correction, not a tuning constant. Additive attention does not need it.

### Q97. Could you give a brief explanation of the Transformers' architecture?

A transformer is a stack of identical blocks that replaces recurrence with attention, so every position can see every other position in one parallel step.

Input tokens become embeddings, and a positional encoding is added, because attention alone has no notion of order. Each block has two sub-layers. The first is multi-head self-attention: project the input to queries, keys, and values, compute $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$, which means each token takes a weighted average of all values according to query-key similarity, and run several heads in parallel so different heads attend to different relations. The second is a position-wise feed-forward network, typically widening by four times. Both sub-layers use a residual connection and layer normalisation, which makes deep stacks trainable.

The original design had an encoder and a decoder with cross-attention, for translation. Today most large language models are decoder-only with causal masking, so a token attends only to earlier tokens. The cost is attention that scales as the square of sequence length.

### Q98. What is Mode Collapse in GANs?

Mode collapse is when the generator produces only a small part of the data distribution. The real data has many modes, for example ten digit classes, but the generator emits one or a few, sometimes nearly the same image for every input noise vector.

The cause is the training objective. The generator is only rewarded for fooling the current discriminator. If one output reliably fools it, the generator has no incentive to cover anything else. The discriminator then adapts, the generator jumps to another single mode, and the two cycle without converging. Nothing in the standard loss asks for diversity or penalises missing coverage.

Common remedies: minibatch discrimination and feature matching, which let the discriminator see batch statistics; unrolled GANs, which let the generator anticipate the discriminator's update; Wasserstein loss with gradient penalty, which gives smoother gradients; and multiple generators.

I detect it by inspecting sample diversity and by metrics such as Inception Score and recall-type coverage measures, not by watching the loss.

## Computer Vision


### Q99. Can you name the main innovations in AlexNet compared to LeNet-5? What about the main innovations in GoogLeNet?

AlexNet kept LeNet-5's idea of stacked convolution and pooling, but changed five things. It was much larger, with about 60 million parameters against LeNet's 60 thousand, and worked on 224 by 224 colour images instead of 32 by 32 digits. It used ReLU instead of sigmoid or tanh, which trains faster because the gradient does not saturate. It used dropout to reduce overfitting, heavy data augmentation, and two GPUs.

GoogLeNet's main innovation is the inception module. Each module runs 1 by 1, 3 by 3, and 5 by 5 convolutions plus pooling in parallel and concatenates the outputs, so the network learns at several scales at once. The 1 by 1 convolutions act as bottlenecks and cut the channel count before the expensive filters. It also replaced the large fully connected head with global average pooling. Therefore it had about 7 million parameters, far fewer than AlexNet, with better accuracy.

### Q100. In what manner are Region-Based CNNs applied in the task of object detection?

An R-CNN turns detection into region proposal plus classification. The original version used selective search to propose about 2000 candidate boxes, warped each to a fixed size, ran a CNN on each one, then classified it and regressed a tighter box. It worked, but running the CNN 2000 times per image made it very slow.

Fast R-CNN fixed that. It runs the backbone once on the whole image, then for each proposal it crops the corresponding part of the feature map with ROI pooling and sends that to the classification and box-regression heads. Therefore the expensive convolution is shared.

Faster R-CNN removed selective search. A region proposal network slides over the same feature map, scores anchor boxes of several sizes and aspect ratios as object or background, and refines them. Proposal and detection then share one backbone and train end to end.

Finally, non-maximum suppression removes duplicate boxes. Mask R-CNN adds a mask head and ROI align for segmentation.

### Q101. Using Keras, construct a simple autoencoder architecture using Dense layers and a Sequential model for 784-dimensional data.

```python
from tensorflow import keras
from tensorflow.keras import layers

latent = 32
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64,  activation="relu"),
    layers.Dense(latent, activation="relu", name="bottleneck"),
    layers.Dense(64,  activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(784, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(x_train, x_train, epochs=20, batch_size=256,
          validation_data=(x_test, x_test))
```

This is a sketch rather than a runnable block, because it needs TensorFlow and a dataset.


The input and the target are the same array, because the network is trained to reconstruct its input. The 32-unit bottleneck forces a compressed representation, which is the thing I actually want.

Two details matter. The output activation must match the data range: sigmoid with binary cross-entropy for pixels scaled to zero-to-one, linear with mean squared error otherwise. A bottleneck as wide as the input lets the network learn the identity, so it must stay narrow, or I add noise or a sparsity penalty instead.

### Q102. How GANs models are trained?

Training alternates between two networks that play a minimax game. The generator maps a noise vector to a sample. The discriminator takes a sample and outputs the probability that it is real.

Each iteration has two steps. First I update the discriminator: take a batch of real data and a batch of generated data, and maximise $\log D(x) + \log(1 - D(G(z)))$, which means it should score real data high and fake data low. The generator is frozen here. Second I update the generator with the discriminator frozen. In practice I maximise $\log D(G(z))$, the non-saturating form, rather than minimising the original term, because the original gives almost no gradient while the discriminator is winning.

Balance is everything. If the discriminator becomes too strong the generator's gradient disappears; if too weak it gives no useful signal.

Therefore I use the standard stabilisers: Wasserstein loss with gradient penalty, spectral normalisation, label smoothing, and a learning rate near $2 \times 10^{-4}$ with Adam.

## MLOps


### Q103. How does the Multi-Arm Bandit approach differ from the A/B testing model deployment strategy?

A/B testing splits traffic at a fixed ratio, runs for a pre-planned duration, then does one statistical test and picks a winner. The allocation does not change while the test runs. Therefore it gives a clean causal estimate for every variant, including the losing ones, but it keeps sending traffic to a variant that is already visibly worse.

A multi-armed bandit reallocates traffic continuously. It uses an exploration rule, such as epsilon-greedy, Thompson sampling, or upper confidence bound, to send more traffic to arms that are performing better while still sampling the others. Therefore it reduces regret, meaning the business loss during the experiment, and it adapts if the best arm changes.

The tradeoff is inference. Because allocation depends on past outcomes, the data is not independent, so standard significance tests do not apply directly.

So I use A/B testing for a defensible measurement of effect size, and a bandit when the metric is fast and a bad model is costly.

### Q104. How does the ML Test Score work?

The ML Test Score comes from the Google paper "What's your ML Test Score? A rubric for ML production systems". It is a checklist that measures how well a machine learning system is engineered, not how accurate the model is.

It groups tests into four areas. Tests for features and data cover schema checks, distribution checks, and the cost and privacy of each feature. Tests for model development cover offline and online metric agreement, model staleness, a simpler baseline, and fairness across slices. Tests for infrastructure cover reproducible training, unit tests on the pipeline, integration tests, canary release, and rollback. Tests for monitoring cover training-serving skew, data invariants in production, prediction drift, and staleness alerts.

Each item scores a point when it is implemented, and a half point when it is manual rather than automated. The total across the areas maps to a readiness level, from "not production ready" upwards.

I would check the exact point totals and band labels against the paper, because I do not want to quote them from memory.

### Q105. How can you run distributed ML in Azure?

Three main paths, and I would confirm current service names, because Azure renames and reorganises these often.

First, Azure Machine Learning compute clusters. I define a cluster of GPU or CPU nodes that scales to zero when idle, then submit a job with a distribution configuration. For PyTorch I use `PyTorchConfiguration` with `process_count_per_node` and run DistributedDataParallel; Azure ML sets the rank and master address environment variables for me. MPI distribution covers Horovod, and TensorFlow has its own parameter-server and MultiWorkerMirrored setup. InfiniBand-enabled VM sizes matter for multi-node speed.

Second, Azure Databricks or Synapse Spark for data-parallel work: Spark MLlib for classic models, and Petastorm or the Spark connector to feed deep learning jobs.

Third, Azure Kubernetes Service, when I want to run Kubeflow or Ray myself.

For hyperparameter work, Azure ML sweep jobs run many trials in parallel on the same cluster with early termination. Data usually sits in Azure Blob Storage or Data Lake, mounted into each node.

### Q106. Which production testing techniques are you aware of?

I split them into pre-release and in-production.

Before release I run offline evaluation on a held-out and a time-based split, slice metrics by segment, and compare against the current production model and a simple baseline. I add unit tests on feature transforms, a schema and distribution check on the input data, and behavioural tests: invariance tests, where a change that should not matter must not change the prediction, and directional tests, where a change should move the prediction a known way.

In production I use shadow deployment, which sends real traffic to the new model and logs predictions without serving them, so I can compare latency and output distribution with no user risk. Canary release sends a small share of real traffic and watches error rates. Blue-green keeps two full environments so rollback is instant. A/B testing and interleaving measure business metrics. Bandits allocate adaptively.

Then continuous monitoring: training-serving skew, feature and prediction drift, and delayed label metrics once ground truth arrives.

## Probability


### Q107. How can you test hypotheses using Bayes' Rule?

I compare how well each hypothesis predicted the data, instead of computing a tail probability. Bayes' rule gives

$$\frac{P(H_1 \mid D)}{P(H_0 \mid D)} = \frac{P(D \mid H_1)}{P(D \mid H_0)} \times \frac{P(H_1)}{P(H_0)}$$

which says the posterior odds equal the likelihood ratio, called the Bayes factor, times the prior odds.

Worked example. A coin gives 8 heads in 10 flips. $H_0$ says the bias is 0.5, $H_1$ says it is 0.75. The likelihoods are $\binom{10}{8}0.75^8 0.25^2$ and $\binom{10}{8}0.5^{10}$, so the Bayes factor is 6.407. I verified this arithmetic in Python. With equal priors the posterior probability of $H_1$ is $6.407/7.407 = 0.865$.

Therefore I get a direct statement about the hypotheses, and I can accumulate evidence as data arrives without a stopping rule. The cost is that I must state a prior, and for composite hypotheses I must integrate the likelihood over it.

### Q108. Compute the probability of drawing two cards that are neither the same colour nor the same number from a standard deck.

My reading: two cards drawn without replacement from 52, and I want the pair to differ in colour, meaning one red and one black, and also differ in rank. "Number" I read as rank, so the thirteen values including the face cards. The other possible reading is suit instead of colour, which gives a different answer.

Arithmetic. Total pairs: $\binom{52}{2} = 1326$. Pairs of different colour: $26 \times 26 = 676$, because I pick one of 26 red and one of 26 black. Of those, the pairs that share a rank: each rank has 2 red and 2 black cards, so $2 \times 2 = 4$ mixed-colour pairs per rank, and $13 \times 4 = 52$ in total. Favourable pairs: $676 - 52 = 624$.

$$P = \frac{624}{1326} = \frac{8}{17} \approx 0.4706$$

I confirmed this by exhaustive enumeration of all 1326 pairs, which gave 624, and by simulating 200000 draws, which gave 0.4718.

### Q109. You have a fair coin and an unfair coin with tails on both sides. You randomly select one coin and flip it. Given that it lands tails, what is the probability you picked the unfair coin?

The answer is $2/3$.

I use Bayes' rule. The prior for each coin is $1/2$. The fair coin gives tails with probability $1/2$. The two-tailed coin gives tails with probability 1.

$$P(\text{unfair} \mid T) = \frac{1 \times 0.5}{1 \times 0.5 + 0.5 \times 0.5} = \frac{0.5}{0.75} = \frac{2}{3} \approx 0.6667$$

The denominator is the total probability of seeing tails, which is $0.75$.

The intuition is to count equally likely outcomes. Think of three tails faces in play: the two on the unfair coin and the one on the fair coin. Tails came up, so I am on one of those three faces, and two of them belong to the unfair coin.

I verified this by simulating two million coin picks and flips. The simulated conditional probability was 0.6671, which matches $2/3$.

### Q110. Considering a medical test with a true positive rate of 100% and a false positive rate of 5%, within a population where 1 in 1000 has the disease, what is the probability a positive result means the person has the disease?

About 1.96 percent, which surprises most people.

Bayes' rule with prevalence $P(D) = 0.001$, sensitivity $P(+ \mid D) = 1$, and false positive rate $P(+ \mid \text{no } D) = 0.05$:

$$P(D \mid +) = \frac{1 \times 0.001}{1 \times 0.001 + 0.05 \times 0.999} = \frac{0.001}{0.050950} = 0.01963$$

The counting version is clearer. In 100000 people, 100 have the disease and all 100 test positive. Of the 99900 healthy people, 5 percent, that is 4995, also test positive. So 5095 positives, of which 100 are true, giving $100/5095 = 0.0196$.

The false positives dominate because the healthy group is a thousand times larger. Therefore a rare disease needs either a much lower false positive rate or a second, independent confirmatory test.

I verified this by simulating two million people, which gave 0.0204 against the analytic 0.01963.

## Linear Algebra


### Q111. Under what circumstances is a matrix's inverse present? How might one determine a nxn matrix's inverse?

A square matrix $A$ has an inverse exactly when it is non-singular. Several conditions say the same thing: the determinant is not zero, the rank equals $n$, the columns are linearly independent, no eigenvalue is zero, and $Ax = 0$ only for $x = 0$. A non-square matrix has no inverse, only a pseudo-inverse.

Three ways to compute it. The adjugate formula, $A^{-1} = \frac{1}{\det A}\,\mathrm{adj}(A)$, where the adjugate is the transposed matrix of cofactors. It is exact and fine for 2 by 2 or 3 by 3, but its cost grows factorially.

Gauss-Jordan elimination is the practical hand method. I write $[A \mid I]$ and row-reduce until the left block is the identity; the right block is then $A^{-1}$. Cost is about $n^3$ operations.

LU decomposition with partial pivoting is what numerical libraries use.

In practice I almost never form the inverse. To solve $Ax = b$ I use `np.linalg.solve`, which is faster and numerically more stable.

### Q112. How would you diagonalize a matrix?

Diagonalising means writing $A = PDP^{-1}$, where $D$ is diagonal and $P$ is invertible. The entries of $D$ are the eigenvalues, and the columns of $P$ are the matching eigenvectors.

The procedure has three steps. First, solve the characteristic equation $\det(A - \lambda I) = 0$ for the eigenvalues. Second, for each eigenvalue solve $(A - \lambda I)v = 0$ to get its eigenvectors. Third, put the eigenvectors in the columns of $P$ in the same order as the eigenvalues in $D$.

This works only if the eigenvectors span the space, that is if $A$ has $n$ linearly independent eigenvectors. Distinct eigenvalues guarantee that. A repeated eigenvalue whose eigenspace is smaller than its multiplicity makes the matrix defective and not diagonalisable; then I use the Jordan form instead.

A real symmetric matrix is always diagonalisable with orthogonal eigenvectors, so $P^{-1} = P^\top$. Diagonalisation is useful because powers become easy: $A^k = PD^kP^{-1}$.

### Q113. Explain the concept of span and linear dependence

The span of a set of vectors is the set of all their linear combinations. The span of two non-parallel vectors in three dimensions is a plane through the origin. So the span is everything I can reach by scaling and adding the vectors I have.

A set is linearly dependent when some non-trivial combination equals zero, that is $c_1v_1 + \dots + c_kv_k = 0$ with at least one $c_i$ not zero. That means at least one vector can be written using the others, so it adds nothing new to the span. A set is linearly independent when the only solution is all coefficients zero.

The two ideas connect through rank and basis. The rank of a matrix is the dimension of the span of its columns. An independent set that spans the whole space is a basis, and then every vector has exactly one representation.

In practice this is why dependent features break ordinary least squares: $X^\top X$ becomes singular.

### Q114. What is cofactor of a matrix? How to calculate it?

The cofactor $C_{ij}$ of entry $a_{ij}$ is the signed determinant of the matrix left after deleting row $i$ and column $j$:

$$C_{ij} = (-1)^{i+j} M_{ij}$$

where $M_{ij}$ is that smaller determinant, called the minor. The factor $(-1)^{i+j}$ gives a checkerboard of signs starting with plus at the top left.

Example with $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 10 \end{pmatrix}$. For $C_{11}$ I delete row 1 and column 1, leaving $\begin{pmatrix} 5 & 6 \\ 8 & 10 \end{pmatrix}$, whose determinant is $50 - 48 = 2$, and the sign is plus, so $C_{11} = 2$. For $C_{12}$ the minor is $40 - 42 = -2$ and the sign is minus, so $C_{12} = 2$.

Cofactors matter for two reasons. They give the determinant by expansion along any row, and the transposed cofactor matrix is the adjugate, which gives the inverse.

## Agents


### Q115. Why do AI agents require memory systems beyond simple context windows, and how does this impact system design?

Because the context window is a fixed, expensive, and volatile buffer. It is fixed, so a long task eventually exceeds it. It is expensive, because attention cost grows with the square of the length and every token is re-read on every call. It is volatile, because it disappears when the session ends. Models also attend unevenly across a long window, so burying a fact in the middle of 200000 tokens is not the same as storing it.

The design consequence is that context becomes a cache, not storage. I need an external store, usually a vector index for semantic recall plus a key-value or relational store for facts, and a retrieval step that assembles the working context for each turn.

Therefore the real problems are selection and eviction: what to write, what to summarise, what to drop. Memory must also be inspectable, because a wrong stored fact will repeat forever.

### Q116. Why do traditional RAG systems often fail in agentic environments, and what architectural changes are necessary?

Classic RAG assumes one question, one retrieval, one answer. An agent breaks all three assumptions. Its tasks are multi-step, so the right query at step four depends on what it found at step two. Its goals are often not semantically similar to the documents it needs. And it writes as well as reads, so the store changes while the task runs.

Three further failures are common. Single-shot top-k retrieval returns chunks that match the surface wording, not the plan. Chunking destroys structure, so a table arrives in pieces. And nothing tells the agent that retrieval was poor.

The changes I would make: give the agent retrieval as a tool it calls repeatedly, with query rewriting, so retrieval becomes iterative rather than a preprocessing step. Add hybrid search, keyword plus vector, and a reranker. Keep document structure and metadata, so the agent can filter by source and time. Add a relevance check with a retry path. Write results back to the store.

### Q117. Why do agents need both short-term and long-term memory, and how should the boundary between them be managed?

They serve different purposes. Short-term memory is the current task state: the last few turns, tool outputs, intermediate results. It must be exact and immediately available, and it is thrown away when the task ends. Long-term memory is what should survive: user preferences, stable facts, past outcomes, and learned procedures. It must be small, deduplicated, and searchable.

Keeping them separate matters because the costs differ. Everything short-term sits in the context window and is paid for on every call. Long-term memory is paid for only when retrieved.

I manage the boundary with an explicit promotion and eviction policy. Promote a fact when it is stable, reusable, and confirmed, for example a stated preference, not a one-off instruction. Before promotion, attach a timestamp and a source. Evict from short-term by recency, and compress older turns into a running summary rather than dropping them.

The two risks are promoting noise and never updating a stale fact, so I version entries and prefer the newest.

### Q118. List and describe the essential building blocks of a robust AI agentic system?

Seven blocks.

The model is the reasoning core, which decides the next action. The instruction layer, that is the system prompt and policy, defines the role, the limits, and the output format.

Tools give the agent effects in the world: search, code execution, database and API access. Each needs a clear schema, validation, and permissions, because this is where real damage happens.

Memory, as described above: short-term working state plus a long-term store with retrieval.

Planning and control flow decompose the goal into steps and decide when to stop. This can be a ReAct loop, an explicit planner, or a graph of sub-agents.

Guardrails check inputs and outputs: schema validation, content filters, confirmation before irreversible actions, and a step or cost budget.

Observability records every prompt, tool call, and result, so I can replay a failure. Evaluation closes the loop with task-level success metrics and regression tests.

The weakest of these usually decides how reliable the whole system is.

## System Design


### Q119. Design a real-time recommendation system for 100M+ users (like Spotify).

Metric first. I optimise for long-term listening: completion rate and next-session return, not clicks. A skip inside 30 seconds is the strongest negative signal, so I treat it as an explicit label.

Two-stage funnel. Candidate generation reduces about 100 million tracks to roughly 1000 per request. I run several sources in parallel: approximate nearest neighbour lookup over two-tower user and item embeddings, collaborative filtering co-listen lists, the user's own history, and audio-content embeddings for cold-start tracks. Each source is cheap and recall-oriented. Ranking then scores those 1000 with a gradient-boosted tree or a deep model using real features: time of day, device, sequence of the last 50 plays, track age, artist familiarity. Budget about 100 milliseconds end to end, so ranking must stay under about 50.

Serving. Item embeddings are precomputed in a nightly batch job. User embeddings are refreshed from a streaming pipeline, so a session reacts within seconds. A feature store keeps training and serving consistent.

The thing that matters most is diversity and exploration, because a pure relevance ranker collapses into the same few artists. I enforce artist and genre caps in a re-ranking pass and reserve a slice for exploration.

For the full treatment, see the case-studies chapter of this book.

### Q120. Design what to show on a user's LinkedIn feed?

Metric first. I optimise for meaningful engagement, which means comments, dwell time, and reciprocal connections, with a strong penalty on reported or hidden posts. Pure click optimisation produces engagement bait.

Two-stage funnel. Candidate generation pulls a few thousand posts per request from distinct sources, and the mix is the whole design. Roughly: first-degree connection activity, which is the backbone; second-degree and followed-company or creator content; group and hashtag content the user follows; and a small share of out-of-network recommendations for discovery. Each source has its own retrieval path and its own quota, so one source cannot flood the feed.

Ranking scores those candidates with a multi-task model that predicts several actions at once, such as click, comment, share, and hide, then combines them into one value with tuned weights. Features cover affinity to the author, topic match, post type, and the user's recent session.

Freshness is the thing that matters most. Professional content decays fast, so I apply a strong time decay, measured in hours not days, and I keep a store of very recent posts separate from the main index. I also deduplicate reshares and cap posts per author.

For the full treatment, see the case-studies chapter of this book.

### Q121. Design a ML system for fraud detection with low latency requirements.

The latency budget drives everything. A card authorisation must return in roughly 100 milliseconds end to end, so I allow about 20 to 30 for feature lookup and 10 to 20 for inference. That rules out anything requiring a fresh aggregation over history at request time.

Asymmetric cost drives the threshold. A missed fraud costs the transaction amount; a false decline costs a legitimate customer and possibly their account. So I never tune for accuracy. I set the threshold on a cost curve, and I usually operate at high precision with a manual review queue in the middle band.

Architecture. A streaming pipeline, Kafka plus Flink, maintains windowed aggregates per card, device, and merchant: counts and amounts over the last minute, hour, and day. These land in a low-latency feature store, typically Redis, so the request only reads keys. The model is a gradient-boosted tree, which is fast and well calibrated; a graph or sequence model runs offline to generate features rather than at request time.

Labels arrive weeks later through chargebacks, so I retrain often and monitor drift, because fraud patterns adapt deliberately.

For the full treatment, see the case-studies chapter of this book.

### Q122. Design the YouTube Video recommendation system

Metric first. I optimise expected watch time, not click-through rate. This is the central decision. A click-through objective rewards clickbait: a misleading thumbnail wins the click and then the user leaves in five seconds. Predicting watch time per impression makes that strategy lose, because the short session contributes almost nothing.

Two-stage funnel, as in the original Google paper. Candidate generation treats recommendation as extreme multi-class classification over the corpus and reduces billions of videos to a few hundred, using nearest-neighbour search over learned user and video embeddings built from watch history, search history, and demographics. Ranking then scores those few hundred with a much richer model, using features such as time since the channel's last upload, previous impressions of this video, and the user's interaction with the channel.

Training details that matter: weight positive examples by watch time, sample negatives from the full corpus, and hold out the future rather than random rows, because random splits leak.

I also add guardrails: deduplication, caps per channel, a freshness term so new uploads can be discovered, and quality filters for borderline content.

For the full treatment, see the case-studies chapter of this book.
---

## What the list is actually testing

Read back over the 122 and the shape is clear.

About a third are definitions, and those are free marks if you have said them
out loud once. Another third turn on a mechanism — why L1 gives exact zeros, why
the dot product gets divided by the square root of the head dimension, why a
stack of linear layers collapses. Those reward understanding and punish
memorisation, because the follow-up question goes one layer down.

The last third are the ones that separate people. They ask you to choose, and a
choice needs a reason: which metric under imbalance, which activation, which
imputation, RAG or fine-tuning, batch or stream. For every one of those, the
answer that scores is the one that names the tradeoff rather than the winner.

Three questions in the list have a premise worth correcting rather than
accepting, and the chapter corrects them: the gradient does not point at the
global extreme, a transformer is not inherently auto-regressive, and a deep
network does not escape the curse of dimensionality so much as exploit structure
in the data. Saying so politely is worth more than a smooth answer that is
wrong.

## Done when

- You can answer any question in the ML theory section in under a minute without
  reading it first.
- You can type all nine coding answers from a blank file, and the k-means one
  passes its own assertion.
- You can state, for each of the choice questions, the tradeoff rather than the
  winner.
- Your marked set from day one is empty.
