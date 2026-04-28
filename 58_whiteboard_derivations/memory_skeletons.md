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
