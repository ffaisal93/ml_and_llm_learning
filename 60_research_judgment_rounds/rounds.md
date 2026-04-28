# Research Judgment Rounds

---

## Round 1: One Metric Improved

### Prompt

A new model improves perplexity by 3% but shows no gain on downstream QA.

### Good answer structure

- objective metric and task metric are different
- possible mismatch between training objective and task success
- decoding or calibration may matter
- inspect slices and answer-type breakdown
- do not overclaim downstream improvement

---

## Round 2: One Seed Only

### Prompt

A method beats baseline on one seed but not the others.

### Good answer structure

- do not claim robust improvement
- report mean and variance
- increase number of seeds
- inspect sensitivity to initialization or optimizer noise

---

## Round 3: Strong Gain, Weak Baseline

### Prompt

A paper reports a big gain, but the baseline is outdated and under-tuned.

### Good answer structure

- the result is not yet convincing
- stronger baseline needed
- same data/compute budget needed
- isolate whether the gain is real or due to weak comparison

---

## Round 4: Retrieval Metric Improves, Final System Gets Worse

### Prompt

Recall@10 improved, but final answer accuracy dropped.

### Good answer structure

- retrieved context may be noisier
- ordering and truncation may hurt
- generator may ignore evidence
- retrieval and generation objectives are not identical
- inspect failure stage explicitly

---

## Round 5: Small Reported Improvement

### Prompt

A paper reports a 0.2-point gain on a benchmark.

### Good answer structure

- ask for variance across runs
- ask whether the metric is saturated
- ask whether the gain is consistent across slices
- check whether compute/data changed

---

## Round 6: Bigger Model Wins

### Prompt

A method improves results, but it also uses a much larger model.

### Good answer structure

- capacity confounds method effect
- need matched-size comparison
- need compute-normalized or parameter-normalized evidence

---

## Round 7: Preference Win Rate Improved

### Prompt

Human preference win rate improved, but factuality declined.

### Good answer structure

- preference signal may reward style over truth
- reward misspecification
- evaluation mismatch
- need factuality and robustness checks in parallel

---

## Round 8: Benchmark Leakage Suspicion

### Prompt

Results look unusually strong on one benchmark but not others.

### Good answer structure

- check contamination
- check preprocessing overlap
- inspect dataset construction
- compare transfer to other benchmarks
