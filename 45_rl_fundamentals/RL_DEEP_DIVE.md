# Reinforcement Learning Fundamentals — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

RL is the foundation underneath RLHF, agentic systems, and tool-use training. Frontier-lab interviews probe RL not because they want game-playing agents but because RLHF/PPO/GRPO fluency requires understanding the underlying machinery. This deep dive covers what you need.

---

## 1. The MDP framework

> **In plain language.** An MDP is the standard container for any decision problem: where you can be, what you can do, how the world responds, what score you get, and how much the future counts. The tuple notation below is just those five things given letters.

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

> **Saying it out loud.** An MDP is how you write down 'I'm in a situation, I act, the world responds, I get a score.' Five pieces — states, actions, transitions, rewards, discount — and the discount is the one people skip that matters most, because it silently sets how far ahead the agent can see. The Markov property is the load-bearing assumption: the current state tells you everything, so history can be discarded. It's frequently false in practice — poker, dialogue, anything partially observed — and the standard workaround is to cram enough history into the state to make it approximately true, which is exactly what a context window does.

---

## 2. Value functions

**State-value** $V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]$ — expected return starting from $s$ following $\pi$.

**Action-value** $Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$ — expected return from $s$ taking $a$ first, then $\pi$.

**Advantage**:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

How much better is action $a$ than the policy's average behavior in state $s$?

> **Saying it out loud.** Three quantities and they're easy to keep straight. $V$ is how good it is to be somewhere, $Q$ is how good it is to be somewhere and do a particular thing first, and advantage is the difference — how much better this action is than your usual behavior here. Advantage is the one that ends up mattering most, because raw $Q$ values carry a big shared offset from simply being in a good state, and that offset is pure noise when you're choosing between actions. Subtracting it is the single largest variance reduction in policy gradient methods, and it's free — no bias introduced.

### Bellman equations

> **In plain language.** The Bellman equations say your estimates have to agree with themselves one step apart: the value of being here should equal the reward you're about to collect plus the discounted value of wherever you end up. The sums below are just averaging that over which action the policy picks and where the world sends you.

$V^\pi$ satisfies (one-step decomposition):

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')]
$$

$$
Q^\pi(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]
$$

> **Saying it out loud.** The Bellman equation is a consistency requirement, not a formula you'd derive from scratch. It says the value of here equals the immediate reward plus the discounted value of next, averaged over what your policy does and where the environment takes you. Everything else in RL is a strategy for enforcing that consistency — dynamic programming solves it exactly when you know the model, TD learning nudges toward it from single samples. If you only remember one thing, remember that the TD error is literally how badly this equation is violated right now.

### Bellman optimality

For optimal policy $\pi^*$:

$$
V^*(s) = \max_a \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V^*(s')]
$$

$$
Q^*(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \max_{a'} Q^*(s', a')]
$$

These are fixed-point equations. The Bellman operator $\mathcal{T}^*$ is a contraction → unique solution → value iteration converges.

> **Saying it out loud.** Same consistency statement with one change: instead of averaging over what your current policy would do, take the best action. That single max turns an evaluation equation into an optimization problem and makes it nonlinear, so you can no longer just solve a linear system — you have to iterate. The reason iterating works is that the operator is a contraction with factor gamma, so each pass shrinks your error geometrically and Banach's theorem hands you a unique fixed point. The same max is also the source of Q-learning's optimism bias, since maximizing over noisy estimates skews high.

---

## 3. Dynamic programming methods

When the model is known, you can compute $V^*$ and $Q^*$ exactly.

### Value iteration

Iterate the Bellman optimality operator:

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s, a) [R + \gamma V_k(s')]
$$

Converges geometrically with rate $\gamma$. Optimal policy: $\pi^*(s) = \arg\max_a Q^*(s, a)$.

> **Saying it out loud.** Sweep every state, set its value to the best one-step lookahead, repeat until nothing moves. Each sweep pushes reward information one step further backward through the state space, so if the goal is 50 steps away it takes about 50 sweeps before the starting state knows anything — that's the cleanest intuition for why sparse distant rewards are hard. Convergence is geometric at rate gamma, which means the number of sweeps you need scales like one over one minus gamma. At gamma 0.999 that's a thousand-fold cost for a long horizon.

### Policy iteration

1. **Policy evaluation**: solve $V^\pi = \mathcal{T}^\pi V^\pi$ (linear system).
2. **Policy improvement**: $\pi'(s) = \arg\max_a Q^\pi(s, a)$.
3. Repeat until convergence.

Each step strictly improves (or terminates). Often faster than value iteration in practice.

> **Saying it out loud.** Two phases alternating. Evaluate the current policy exactly — which is a linear solve, since with the policy fixed there's no max — then improve it by acting greedily on those values. The improvement step provably never makes things worse, which is why it terminates, usually in fewer than ten rounds. The tradeoff against value iteration is rounds versus cost per round: policy iteration needs very few but each is expensive. In practice people run modified policy iteration, where you evaluate only partially before improving, which beats both.

---

## 4. Model-free methods — when you don't know $P$ and $R$

### Monte Carlo

Run full episodes; average returns to estimate $V^\pi(s)$:

$$
V^\pi(s) \leftarrow V^\pi(s) + \alpha (G_t - V^\pi(s))
$$

Pros: unbiased. Cons: high variance, requires episodic structure.

> **Saying it out loud.** Play the episode to the end, look at what you actually got, and move your estimate toward it. It's unbiased because you're using ground truth rather than a guess — no bootstrapping anywhere. The cost is variance: one lucky trajectory can swing your estimate hard, so you need a lot of episodes. It also flatly requires episodes to end, so it's useless for continuing tasks like a running control system. Great mental baseline, rarely the right choice.

### Temporal Difference (TD) learning

Bootstrap from current value estimate:

$$
V(s_t) \leftarrow V(s_t) + \alpha [r_t + \gamma V(s_{t+1}) - V(s_t)]
$$

The bracketed quantity is the **TD error** $\delta_t$. TD trades variance for bias.

> **Saying it out loud.** TD updates as soon as you get a hint instead of waiting for the outcome. The everyday version: you predict a two-hour drive, hit traffic ten minutes in, and immediately revise — you don't wait until you arrive to learn something. The gap between old prediction and revised prediction is the TD error, and that's the learning signal. It's model-free like Monte Carlo but bootstrapped like dynamic programming, so you get online learning with far less variance, at the cost of bias while your estimates are still wrong. Worth mentioning that dopamine neurons appear to encode almost exactly this quantity.

### Q-learning (off-policy)

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
$$

Update toward the *greedy* next-action value, even if behavior policy was exploratory. Off-policy: learn $Q^*$ while acting $\epsilon$-greedy.

> **Saying it out loud.** Q-learning is TD on state-action pairs with a max in the target — you bootstrap off the best action available next, not the one you actually took. That's what makes it off-policy, and it's a big practical deal: you can explore randomly, replay old data, even learn from someone else's trajectories, and still converge to the optimal policy. The price is that maximizing over noisy estimates is systematically optimistic, and that bias compounds through bootstrapping. You see it as value estimates drifting far above any return the agent ever actually achieves, which is why Double Q-learning exists.

### SARSA (on-policy)

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
$$

Update toward the action actually taken. Learns $Q^\pi$ for the behavior policy.

> **Saying it out loud.** SARSA replaces Q-learning's max with the action you actually took next — the name is just the tuple it consumes. That makes it on-policy: it learns the value of the policy you're really running, exploration mistakes and all. The cliff-walking example is the one to tell: Q-learning learns the shortest path hugging the cliff edge because its target assumes optimal play afterward, while SARSA knows epsilon-greedy will occasionally step off and learns a safer route back from the edge. Which one you want depends on whether you'll still be exploring at deployment time.

---

## 5. Function approximation and DQN

For continuous or huge state spaces, use a function approximator $Q_\theta$.

### DQN (Deep Q-Network, Mnih et al. 2015)

Loss:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')}[(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a))^2]
$$

> **Saying it out loud.** DQN is Q-learning with a neural network standing in for the table, and the loss is just squared error against a bootstrapped target. The critical implementation detail is that the target is computed from a separate frozen copy and treated as a constant — you don't backpropagate through it. If you do, you're differentiating both sides of the Bellman equation and training diverges. That plus the replay buffer is the whole story of why this worked in 2015 when the same idea had been unstable for years.

### Tricks that made DQN work
- **Experience replay**: store transitions in a buffer; sample uniformly. Breaks temporal correlations.
- **Target network** $\theta^-$: snapshot of $\theta$ updated infrequently. Prevents the target from chasing itself.
- **Frame stacking + CNN**: handles partial observability of single-frame Atari.

> **Saying it out loud.** Two of these do the real work. Experience replay fixes the fact that consecutive frames are nearly identical, which violates the near-independence SGD assumes and makes the network chase whatever it just saw — plus it lets you reuse each transition many times, which matters when environment interaction is the expensive part. The target network fixes the moving-target problem: freeze a copy for a few thousand steps and the whole thing looks like ordinary supervised regression against fixed labels. Frame stacking is really an admission that a single Atari frame isn't a Markov state, since you can't tell velocity from one image.

### Improvements
- **Double DQN**: decouple action selection (online net) from evaluation (target net) to reduce overestimation bias.
- **Dueling DQN**: separate value $V(s)$ and advantage $A(s, a)$ heads.
- **Prioritized experience replay**: sample by TD error magnitude.
- **Rainbow**: combines all of these.

> **Saying it out loud.** Double DQN is the one I'd lead with because it's a one-line change with a clear story: use the online network to choose the action and the target network to score it, so an action that got lucky in one net is unlikely to get lucky in the other, and the overestimation largely cancels. Dueling splits $Q$ into state value plus action advantage, which helps because in most states no action matters much and you'd rather learn that state's value once. Prioritized replay samples high-error transitions more, with importance weights to correct the resulting bias. Rainbow stacks all of them and roughly doubles median Atari performance over plain DQN — a good example of engineering increments compounding.

---

## 6. Policy gradient methods

Directly parameterize the policy $\pi_\theta(a|s)$ and optimize via gradient ascent on expected return.

> **Saying it out loud.** Policy gradient skips values entirely and adjusts the policy's parameters directly to make good actions more likely. The reason you'd want this is concrete: with continuous actions there's no max to take, so value-based methods stall, and if the optimal behavior is genuinely stochastic — bluffing, say — a greedy policy can't even represent it. The cost is variance, because you're estimating a gradient from sampled trajectories. Everything that follows in this section is a variance-reduction technique layered on the same core idea.

### Policy gradient theorem

> **In plain language.** Underneath the notation this says one thing: make actions that went well more likely, and actions that went badly less likely, in proportion to how well they went. The gradient-of-log-probability term is just the direction that increases an action's probability.

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a)]
$$

The gradient of the *return* equals the expectation of (gradient of log-probability) × (Q-value).

> **Saying it out loud.** In plain terms: take the actions you sampled, and push their probabilities up or down in proportion to how good they turned out. The log-probability gradient is the direction in parameter space that makes an action more likely; $Q$ is the weight saying how hard to push and which way. What makes it a theorem rather than an observation is that it survives a real subtlety — changing the policy also changes which states you visit, and you'd expect a term for that, but it cancels out. That cancellation is why you can estimate the gradient purely from sampled trajectories.

### REINFORCE

Use Monte Carlo return $G_t$ as an unbiased estimator of $Q$:

$$
\nabla_\theta J \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(a_i|s_i) G_i
$$

Pros: simple, unbiased. Cons: high variance.

> **Saying it out loud.** REINFORCE is the naive instantiation: run an episode, take the total reward, scale every action's log-probability gradient by it. Good episode, everything gets reinforced. The problem is credit assignment — an action at step three gets the same credit as one at step three hundred, so variance is enormous and you need thousands of episodes for tasks TD methods solve in dozens. Everything after it in this section exists to fix that.

### Variance reduction with baselines

$$
\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) (Q^\pi(s, a) - b(s))]
$$

For any baseline $b(s)$ that doesn't depend on $a$. Standard choice: $b(s) = V^\pi(s)$, giving advantage:

$$
\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) A^\pi(s, a)]
$$

> **Saying it out loud.** You only care whether an action beat the average, not whether the state happened to be good. Subtracting any function of the state leaves the gradient unbiased — the expected log-probability gradient is zero, so the baseline term vanishes in expectation — while cutting variance dramatically. Concretely, if every action in a state returns roughly 100, weighting by 100 gives you a big noisy shove on everything; subtract it and only the meaningful differences survive. Using $V(s)$ as the baseline is what turns $Q$ into advantage, which is why advantage shows up everywhere downstream.

### Actor-critic

Train both:
- **Actor**: policy $\pi_\theta$.
- **Critic**: value $V_\phi$ (or $Q_\phi$).

Use the critic's advantage estimate $A^\pi$ in the policy gradient. Reduces variance vs Monte Carlo at cost of some bias.

> **Saying it out loud.** The actor picks actions, the critic judges them, and each trains the other — the actor follows the critic's advantage estimates, the critic learns from observed rewards by TD. The gain over REINFORCE is that you no longer wait for the episode to end to know whether an action was good. The risk is that you're now optimizing against the critic's opinions rather than reality, so a bad critic actively misleads the actor and the two can spiral. That coupling is why critic-free methods like GRPO are attractive at LLM scale, where a second full-size network is expensive and its failures are hard to diagnose.

### A2C / A3C

Advantage Actor-Critic / Asynchronous A3C. Synchronous (A2C) and asynchronous (A3C) variants. Standard before PPO.

> **Saying it out loud.** Both run many actors in parallel so the data isn't temporally correlated; the difference is whether they synchronize. A3C lets workers push gradients whenever they finish, which means they're computing against slightly stale parameters. A2C waits for everyone and does one batched update. Synchronous turned out to be simpler and faster on GPUs — one large batch beats many small ones — so A2C is what survived, and then PPO largely replaced both.

---

## 7. Trust-region and PPO

Vanilla policy gradient suffers from **destructive updates**: large step → policy collapses.

### Natural policy gradient

Use the Fisher metric to control update magnitude:

$$
\theta \leftarrow \theta + \alpha F(\theta)^{-1} \nabla J(\theta)
$$

Step size in the *KL geometry*, not the parameter geometry. Computationally expensive (Fisher matrix inversion).

> **Saying it out loud.** The insight is that Euclidean distance in parameter space is the wrong ruler. A tiny weight change can flip a policy from near-deterministic to near-uniform, or barely move it at all, depending on where you are. The natural gradient measures distance in the space of distributions instead, using the Fisher information matrix as the metric, so a fixed step size means a fixed amount of behavioral change. It's the right idea and it's expensive, because you're inverting a matrix the size of your parameter count — which is exactly why TRPO approximates it and PPO abandons it for a clipping heuristic.

### TRPO (Schulman et al. 2015)

Constrained optimization: maximize the surrogate objective subject to $\mathrm{KL}(\pi_{\mathrm{old}} \| \pi_\theta) \leq \delta$. Solve via conjugate gradient + line search.

> **Saying it out loud.** TRPO turns the trust region into an explicit constraint: improve the surrogate objective as much as you like, provided the new policy stays within a fixed KL distance of the old one. It works and it has the guarantees, and it's genuinely painful to implement — conjugate gradients, Fisher-vector products, a backtracking line search. That implementation cost is the entire reason PPO exists, and PPO getting 95 percent of the benefit in twenty lines is why almost nobody runs TRPO today.

### PPO (Schulman et al. 2017)

> **In plain language.** PPO's rule is: you may make a good action more likely, but only up to about 20 percent more likely per update, and past that you stop getting credit. The code below is the whole algorithm — a ratio, a clamp, and a minimum.

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

> **Saying it out loud.** PPO computes how much more likely the new policy makes each action than the old one, and refuses to reward you for pushing that ratio past roughly 20 percent. The clever part is the minimum, which makes it one-sided: you gain nothing from moving too far in the good direction, but if the action was bad you're still fully penalized for having increased its probability. A symmetric clip wouldn't work, and that asymmetry is the thing people miss when they reimplement it. Epsilon 0.2 is the near-universal default, and the honest caveat is that clipping doesn't actually guarantee a bounded KL — which is why serious RLHF code monitors KL separately and early-stops on it.

### GAE (Generalized Advantage Estimation)

A flexible advantage estimator:

$$
A^{\mathrm{GAE}(\lambda)}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}
$$

with TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. $\lambda$ trades bias and variance:
- $\lambda = 0$: pure TD (low variance, high bias).
- $\lambda = 1$: Monte Carlo (high variance, low bias).
- Standard: $\lambda \approx 0.95$.

> **Saying it out loud.** GAE is a dial between two ways of judging an action. One extreme uses a single step of real reward plus your value estimate — low noise, but only as good as your critic. The other uses the whole observed return — honest, but wildly noisy. Lambda exponentially blends every horizon in between, and 0.95 is standard because it keeps most of the variance reduction while limiting the damage from a mediocre critic. Note that gamma and lambda multiply, so the effective horizon is set by their product, and people tune them as a pair rather than independently.

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

> **Saying it out loud.** The tension is that every step spent learning about an option is a step not spent cashing in on the best one you know, and vice versa. Epsilon-greedy is the crude version — explore at random some fraction of the time — and it's still everywhere because it's one line, though it wastes exploration on actions already proven terrible. UCB is the principled version: add a bonus for uncertainty so you explore where you might be wrong, which buys you logarithmic regret in bandits instead of linear. In LLM training the picture changes shape entirely, because the pretrained model is already a decent policy — so the KL penalty against the reference model is doing the exploration control, keeping you from wandering into regions where nothing sensible lives.

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

> **Saying it out loud.** Map the pieces onto tokens and it's straightforward: the state is the prompt plus what you've generated so far, the action is the next token, and the reward arrives only at the end from a learned model scoring the whole response. That's a several-hundred-step episode with a single terminal reward, which is about the hardest credit assignment problem in the book. The KL penalty against the fine-tuned reference model is what keeps it from falling apart, because the reward model is an approximation and any policy optimized hard enough will exploit its cracks rather than genuinely improve — that's reward hacking, and it shows up as verbosity and sycophancy while the reward curve keeps climbing. GRPO's contribution is dropping the critic entirely: sample a group of responses per prompt and use the group mean as the baseline, which saves a full-size value network and removes a notorious source of instability, at the cost of needing 8 to 64 samples per prompt.

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
