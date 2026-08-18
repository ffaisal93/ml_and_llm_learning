# Derivation Memory Skeletons

These are short memory cues, not full answers.

## Logistic Regression

1. `z = Xw + b`
2. `p = sigmoid(z)`
3. BCE loss
4. `dL/dz = p - y`
5. `grad_w = X^T (p - y) / n`

## Softmax + CE

1. write softmax
2. write CE
3. use one-hot target
4. result: `p - y`

## Bernoulli MLE

1. write Bernoulli likelihood
2. take log
3. differentiate w.r.t. `p`
4. solve -> sample mean

## Gaussian MLE

1. write Gaussian log-likelihood
2. derive w.r.t. `mu`
3. derive w.r.t. `sigma^2`
4. note MLE uses `/ n`

## Confidence Interval

1. estimate
2. standard error
3. critical value
4. center +/- margin

## Attention Shapes

1. `Q (n, d_k)`
2. `K (n, d_k)`
3. `QK^T -> (n, n)`
4. multiply by `V (n, d_v)` -> `(n, d_v)`

> **Saying it out loud (how to use this page).** These aren't answers, they're the order of your hand movements — the point is that under pressure you forget structure long before you forget algebra. So practise them the way you'd practise a route: say the step, then write it, then say the next one, out loud, standing up, without looking. If you can recite the five steps of the logistic regression skeleton cold, you can rebuild the whole derivation live even if the sigmoid derivative escapes you for a second, because the skeleton tells you what to reach for next. Two rules that make this work: never write silently, and always say what you're about to do before you do it, so a pause reads as thinking rather than being stuck. The failure mode this prevents is the worst one at a whiteboard — freezing mid-derivation with a marker up and no idea what line comes next.
