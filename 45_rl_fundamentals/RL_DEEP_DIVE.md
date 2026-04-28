# Reinforcement Learning Fundamentals — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

RL is the foundation underneath RLHF, agentic systems, and tool-use training. Frontier-lab interviews probe RL not because they want game-playing agents but because RLHF/PPO/GRPO fluency requires understanding the underlying machinery. This deep dive covers what you need.

---

## 1. The MDP framework

A Markov Decision Process is $(S, A, P, R, \gamma)$:
- $S$: state space.
- $A$: action space.
- $P(s'|s, a)$: transition probability.
- $R(s, a)$ (or $R(s, a, s')$): reward function.
- $\gamma \in [0, 1)$: discount factor.

**Markov property**: $P(s_{t+1}|s_t, a_t, s_{t-1}, \ldots) = P(s_{t+1}|s_t, a_t)$. Future depends only on current state and action.

**Policy** $\pi$: distribution over actions given state. Deterministic: $a = \pi(s)$. Stochastic: $\pi(a|s)$.

**Trajectory**: $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$.

**Return** (cumulative discounted reward):

$$
G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}
$$

The agent maximizes $\mathbb{E}_\pi[G_0]$.

---

## 2. Value functions

**State-value** $V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]$ — expected return starting from $s$ following $\pi$.

**Action-value** $Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$ — expected return from $s$ taking $a$ first, then $\pi$.

**Advantage**:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

How much better is action $a$ than the policy's average behavior in state $s$?

### Bellman equations

$V^\pi$ satisfies (one-step decomposition):

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')]
$$

$$
Q^\pi(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]
$$

### Bellman optimality

For optimal policy $\pi^*$:

$$
V^*(s) = \max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^*(s')]
$$

$$
Q^*(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \max_{a'} Q^*(s', a')]
$$

These are fixed-point equations. The Bellman operator $\mathcal{T}^*$ is a contraction → unique solution → value iteration converges.

---

## 3. Dynamic programming methods

When the model is known, you can compute $V^*$ and $Q^*$ exactly.

### Value iteration

Iterate the Bellman optimality operator:

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s, a) [R + \gamma V_k(s')]
$$

Converges geometrically with rate $\gamma$. Optimal policy: $\pi^*(s) = \arg\max_a Q^*(s, a)$.

### Policy iteration

1. **Policy evaluation**: solve $V^\pi = \mathcal{T}^\pi V^\pi$ (linear system).
2. **Policy improvement**: $\pi'(s) = \arg\max_a Q^\pi(s, a)$.
3. Repeat until convergence.

Each step strictly improves (or terminates). Often faster than value iteration in practice.

---

## 4. Model-free methods — when you don't know $P$ and $R$

### Monte Carlo

Run full episodes; average returns to estimate $V^\pi(s)$:

$$
V^\pi(s) \leftarrow V^\pi(s) + \alpha (G_t - V^\pi(s))
$$

Pros: unbiased. Cons: high variance, requires episodic structure.

### Temporal Difference (TD) learning

Bootstrap from current value estimate:

$$
V(s_t) \leftarrow V(s_t) + \alpha [r_t + \gamma V(s_{t+1}) - V(s_t)]
$$

The bracketed quantity is the **TD error** $\delta_t$. TD trades variance for bias.

### Q-learning (off-policy)

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
$$

Update toward the *greedy* next-action value, even if behavior policy was exploratory. Off-policy: learn $Q^*$ while acting $\epsilon$-greedy.

### SARSA (on-policy)

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
$$

Update toward the action actually taken. Learns $Q^\pi$ for the behavior policy.

---

## 5. Function approximation and DQN

For continuous or huge state spaces, use a function approximator $Q_\theta$.

### DQN (Deep Q-Network, Mnih et al. 2015)

Loss:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')}[(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a))^2]
$$

### Tricks that made DQN work
- **Experience replay**: store transitions in a buffer; sample uniformly. Breaks temporal correlations.
- **Target network** $\theta^-$: snapshot of $\theta$ updated infrequently. Prevents the target from chasing itself.
- **Frame stacking + CNN**: handles partial observability of single-frame Atari.

### Improvements
- **Double DQN**: decouple action selection (online net) from evaluation (target net) to reduce overestimation bias.
- **Dueling DQN**: separate value $V(s)$ and advantage $A(s, a)$ heads.
- **Prioritized experience replay**: sample by TD error magnitude.
- **Rainbow**: combines all of these.

---

## 6. Policy gradient methods

Directly parameterize the policy $\pi_\theta(a|s)$ and optimize via gradient ascent on expected return.

### Policy gradient theorem

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a)]
$$

The gradient of the *return* equals the expectation of (gradient of log-probability) × (Q-value).

### REINFORCE

Use Monte Carlo return $G_t$ as an unbiased estimator of $Q$:

$$
\nabla_\theta J \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(a_i|s_i) G_i
$$

Pros: simple, unbiased. Cons: high variance.

### Variance reduction with baselines

$$
\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) (Q^\pi(s, a) - b(s))]
$$

For any baseline $b(s)$ that doesn't depend on $a$. Standard choice: $b(s) = V^\pi(s)$, giving advantage:

$$
\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) A^\pi(s, a)]
$$

### Actor-critic

Train both:
- **Actor**: policy $\pi_\theta$.
- **Critic**: value $V_\phi$ (or $Q_\phi$).

Use the critic's advantage estimate $A^\pi$ in the policy gradient. Reduces variance vs Monte Carlo at cost of some bias.

### A2C / A3C

Advantage Actor-Critic / Asynchronous A3C. Synchronous (A2C) and asynchronous (A3C) variants. Standard before PPO.

---

## 7. Trust-region and PPO

Vanilla policy gradient suffers from **destructive updates**: large step → policy collapses.

### Natural policy gradient

Use the Fisher metric to control update magnitude:

$$
\theta \leftarrow \theta + \alpha F(\theta)^{-1} \nabla J(\theta)
$$

Step size in the *KL geometry*, not the parameter geometry. Computationally expensive (Fisher matrix inversion).

### TRPO (Schulman et al. 2015)

Constrained optimization: maximize the surrogate objective subject to $\mathrm{KL}(\pi_{\mathrm{old}} \| \pi_\theta) \leq \delta$. Solve via conjugate gradient + line search.

### PPO (Schulman et al. 2017)

Replace the constraint with a clipped surrogate. The clean way to write it (and to code it):

```python
r = pi_theta(a|s) / pi_old(a|s)             # importance ratio
surr1 = r * A
surr2 = clip(r, 1 - eps, 1 + eps) * A         # clipped version
loss = -min(surr1, surr2).mean()              # negate for gradient ascent
```

Equivalent formula:

$$
\mathcal{L}^{\mathrm{CLIP}}(\theta) = \mathbb{E}\!\left[\min\!\big(r_t A_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\big)\right]
$$

Standard $\epsilon = 0.2$. When the new policy moves too far in the direction the advantage points, the clip kills the gradient — that's the trust-region effect.

PPO is simpler than TRPO, more stable than vanilla PG, and the workhorse of modern RL — including RLHF.

### GAE (Generalized Advantage Estimation)

A flexible advantage estimator:

$$
A^{\mathrm{GAE}(\lambda)}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}
$$

with TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. $\lambda$ trades bias and variance:
- $\lambda = 0$: pure TD (low variance, high bias).
- $\lambda = 1$: Monte Carlo (high variance, low bias).
- Standard: $\lambda \approx 0.95$.

---

## 8. Exploration vs exploitation

Without exploration, the agent can be stuck on suboptimal policies.

- **$\epsilon$-greedy**: with prob $\epsilon$, random; else greedy.
- **Boltzmann (softmax)**: sample from $\pi(a|s) \propto \exp(Q(s, a)/T)$.
- **UCB**: bonus to less-tried actions: $a = \arg\max [Q(s, a) + c \sqrt{\log t / N(s, a)}]$.
- **Thompson sampling**: maintain posterior over $Q$; sample and act greedily w.r.t. sample.
- **Entropy bonus**: add $\beta H(\pi(\cdot|s))$ to the objective. Used in PPO for LLM alignment.
- **Curiosity / intrinsic motivation**: reward novelty. Useful in sparse-reward tasks.

In LLM RLHF, the KL penalty serves as a regularizer that prevents over-specialization (a form of soft exploration constraint).

---

## 9. RL for LLMs (RLHF connection)

In RLHF:
- **State**: prompt + tokens generated so far.
- **Action**: next token.
- **Reward**: from a learned reward model (or rule-based for verifiable tasks like math).
- **Policy**: the LLM itself, $\pi_\theta(\mathrm{token}|\mathrm{context})$.
- **Reference policy**: $\pi_{\mathrm{ref}}$, the SFT model. KL penalty $\beta \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$ prevents drift.

The PPO objective for RLHF:

$$
\mathcal{L}(\theta) = \mathbb{E}\left[\mathrm{clip}\,\mathrm{surrogate}(\theta) - \beta \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})\right]
$$

GRPO (DeepSeekMath/R1) is a simplification: drops the learned value/critic network. Advantage is computed from group-relative reward normalization (sample $K$ responses per prompt; advantage is $(r_i - \mu_{\mathrm{group}})/\sigma_{\mathrm{group}}$).

```python
def grpo_advantage(rewards):
    """rewards: [B, K] — K sampled responses per prompt. Returns [B, K] advantages."""
    mu = rewards.mean(dim=-1, keepdim=True)
    sigma = rewards.std(dim=-1, keepdim=True) + 1e-8
    return (rewards - mu) / sigma     # group-relative, no critic needed
```

Recent follow-ups (DAPO, Dr. GRPO, 2025) drop the $\sigma$ normalization to reduce length bias.

---

## 10. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Q-learning is on or off policy? | On | Off — uses max over next actions, regardless of behavior |
| SARSA — on or off? | Off | On — uses the action actually taken |
| Why discount? | "Convention" | Stationary fixed-point of Bellman; bounded value when reward is bounded; preference for sooner rewards |
| Why not just use return as Q? | "It's biased" | Monte Carlo $G_t$ is unbiased but high variance; bootstrap reduces variance |
| Why does PPO clip the ratio? | "Why not?" | Prevents destructive policy updates; stable training |
| Advantage = return - baseline. Any baseline works? | Yes | Any baseline that doesn't depend on $a$ doesn't change the gradient's expectation |
| RLHF uses what RL algo? | DQN | Usually PPO; sometimes DPO (which isn't RL); GRPO in DeepSeek-R1 |

---

## 11. Eight most-asked interview questions

1. **State the Bellman equation for $V^\pi$ and explain.** (Recursive expectation; one-step decomposition.)
2. **Q-learning vs SARSA — what's the difference?** (Off-policy max vs on-policy actual action.)
3. **Why does DQN need a target network?** (Stabilize the target; prevent oscillation.)
4. **Derive the policy gradient theorem.** (Log-derivative trick; expectation of $\nabla \log \pi \cdot Q$.)
5. **Why use a baseline in REINFORCE?** (Reduce variance without changing bias.)
6. **What does PPO clip and why?** (Probability ratio; prevent destructive updates.)
7. **GAE — what does $\lambda$ control?** (Bias-variance: 0 = TD, 1 = Monte Carlo.)
8. **In RLHF, what role does the KL penalty play?** (Prevents the policy from drifting too far from SFT/reference; soft constraint.)

---

## 12. Drill plan

- Memorize Bellman equations (V, Q, optimal V, optimal Q).
- Derive policy gradient theorem on paper. 5 minutes.
- For each algorithm (Q-learning, SARSA, REINFORCE, A2C, PPO), recite: update rule, on/off-policy, key properties.
- Trace one episode of Q-learning with $\epsilon$-greedy on a 2-state MDP.
- For RLHF, write the full PPO objective with KL penalty.

---

## 13. Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* — the canonical text.
- Mnih et al. (2015), *Human-level control through deep reinforcement learning* — DQN.
- Schulman et al. (2015), *Trust Region Policy Optimization*.
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
- Schulman et al. (2016), *High-Dimensional Continuous Control Using Generalized Advantage Estimation* — GAE.
- Christiano et al. (2017), *Deep RL from Human Preferences* — RLHF foundation.
