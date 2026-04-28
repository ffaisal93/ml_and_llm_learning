# RL Fundamentals — Interview Grill

> 50 questions on MDPs, value functions, Q-learning, policy gradients, PPO. Drill until you can answer 35+ cold.

---

## A. MDPs and value functions

**1. State the components of an MDP.**
$(S, A, P, R, \gamma)$ — states, actions, transitions, reward, discount.

**2. State the Markov property.**
$P(s_{t+1}|s_t, a_t, s_{t-1}, \ldots) = P(s_{t+1}|s_t, a_t)$. Future depends only on current state-action.

**3. Define discounted return.**
$G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$.

**4. Why discount?**
Bounded value when rewards bounded; favors sooner rewards; mathematical convenience (Bellman fixed point unique with $\gamma < 1$).

**5. State-value $V^\pi$ vs action-value $Q^\pi$?**
$V^\pi(s) = \mathbb{E}_\pi[G_t|s_t=s]$. $Q^\pi(s,a) = \mathbb{E}_\pi[G_t|s_t=s, a_t=a]$.

**6. Define advantage.**
$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$. How much better is action $a$ than the policy's average.

**7. Bellman equation for $V^\pi$?**
$V^\pi(s) = \mathbb{E}_a[R + \gamma V^\pi(s')]$ — expectation over policy and dynamics.

**8. Bellman optimality for $V^*$?**
$V^*(s) = \max_a \mathbb{E}[R + \gamma V^*(s')]$. Take max over actions.

**9. Why does value iteration converge?**
**Each iteration shrinks the error by a factor of $\gamma$**, so it converges geometrically. Formally: the Bellman optimality operator is a $\gamma$-contraction in sup-norm — Banach fixed-point theorem then guarantees a unique fixed point and convergence from any start.

---

## B. Dynamic programming

**10. Value iteration update?**
$V_{k+1}(s) = \max_a \mathbb{E}[R + \gamma V_k(s')]$.

**11. Convergence rate of value iteration?**
Geometric, rate $\gamma$.

**12. Policy iteration steps?**
(1) Policy evaluation — solve $V^\pi$ as linear system. (2) Policy improvement — $\pi'(s) = \arg\max_a Q^\pi(s,a)$.

**13. Value vs policy iteration — when each?**
Both find optimal policy. Policy iteration often converges in fewer iterations but each iteration is more expensive (exact policy evaluation).

---

## C. Model-free TD methods

**14. TD(0) update for $V$?**
$V(s_t) \leftarrow V(s_t) + \alpha [r_t + \gamma V(s_{t+1}) - V(s_t)]$.

**15. What's the TD error?**
$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

**16. Q-learning update?**
$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$.

**17. SARSA update?**
$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$. Uses next action actually taken.

**18. Q-learning vs SARSA: on or off-policy?**
Q-learning: off-policy (uses max regardless of behavior). SARSA: on-policy (uses behavior policy's action).

**19. Why might SARSA learn safer policies?**
SARSA accounts for the actual exploration (e.g., $\epsilon$-greedy) → may avoid risky paths. Q-learning learns optimal regardless.

**20. Monte Carlo vs TD — bias and variance?**
MC unbiased high variance (uses full return). TD biased lower variance (uses bootstrap).

---

## D. DQN

**21. DQN loss?**
$\mathcal{L} = \mathbb{E}[(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s,a))^2]$.

**22. Why experience replay?**
Breaks temporal correlation between consecutive samples; allows reuse of data; more iid-like batches for SGD.

**23. Why a target network?**
Stabilizes training. Without it, the target $Q_{\theta^-}$ shifts with each update — chasing your own tail. Update target slowly (every $K$ steps or Polyak average).

**24. Q-learning overestimates — why?**
$\max_a Q$ tends to overestimate due to noise. Sampling errors get amplified by max.

**25. Double DQN fix?**
Use online net to *select* action, target net to *evaluate*: $r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a'))$. Decouples selection and evaluation.

**26. Dueling DQN — what does it split?**
Network outputs $V(s)$ and $A(s, a)$ separately, then $Q(s, a) = V(s) + (A(s, a) - \mathrm{mean}_a A(s, a))$. Better when only some actions matter.

**27. Prioritized replay?**
Sample high-TD-error transitions more often. Importance weights correct the bias.

---

## E. Policy gradient

**28. State the policy gradient theorem.**
$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s, a)]$. **Intuition (the whole point):** push up the log-probability of actions, weighted by how good they were. Good action → push it up; bad action → push it down. That's it.

**29. Log-derivative trick — what is it?**
$\nabla \log p(x;\theta) = \nabla p(x;\theta)/p(x;\theta)$. Lets you write expectation gradient as expectation of (log-prob gradient × value).

**30. REINFORCE estimator?**
$\nabla J \approx \frac{1}{N}\sum_i \nabla \log \pi(a_i|s_i) G_i$ with $G_i$ the empirical return.

**31. Why use a baseline?**
Reduces variance without bias. $\mathbb{E}[\nabla \log \pi \cdot b(s)] = b(s) \mathbb{E}[\nabla \log \pi] = 0$ for any state-only baseline.

**32. What's the optimal baseline?**
$b^*(s) = \mathbb{E}[Q^\pi(s,a) | s] = V^\pi(s)$ minimizes variance of the gradient estimator.

**33. Actor-critic — actor and critic do what?**
Actor: policy $\pi_\theta$. Critic: value function $V_\phi$ (or $Q_\phi$). Critic provides advantage estimates.

**34. A2C vs A3C?**
A2C: synchronous (one update from all parallel actors). A3C: asynchronous (workers update parameters independently).

---

## F. PPO

**35. Why does naive policy gradient fail with large updates?**
Policy can collapse — large step takes you to a region where $\pi$ assigns near-zero probability to actions you're trying to reinforce. Hard to recover.

**36. TRPO constraint?**
Maximize surrogate subject to $\mathrm{KL}(\pi_{\mathrm{old}} \| \pi_\theta) \leq \delta$. Update step in KL geometry.

**37. PPO clipped surrogate?**
$L = \mathbb{E}[\min(r A, \mathrm{clip}(r, 1-\epsilon, 1+\epsilon) A)]$ with $r = \pi_\theta/\pi_{\mathrm{old}}$. Standard $\epsilon = 0.2$.

**38. Why clip ratio $r$ instead of constraining KL?**
Simpler, no Lagrangian. Heuristic but works extremely well in practice.

**39. What's GAE and what does $\lambda$ control?**
**Intuition**: GAE blends short-horizon TD (low variance, bootstrapped from value estimate) and long-horizon Monte Carlo (high variance, true returns). $\lambda$ slides between them — trade bias vs variance.

**Formula**: $A^{\mathrm{GAE}(\lambda)}_t = \sum_{l \geq 0} (\gamma\lambda)^l \delta_{t+l}$ where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. $\lambda=0$ → pure TD; $\lambda=1$ → Monte Carlo. Standard for PPO: $\lambda \approx 0.95$.

**40. Standard $\lambda$ for PPO?**
0.95.

---

## G. Exploration

**41. $\epsilon$-greedy?**
With prob $\epsilon$, random action; else greedy. Simple but widely used.

**42. Boltzmann exploration?**
$\pi(a|s) \propto \exp(Q(s,a)/T)$. $T$ controls exploration; $T \to 0$ greedy, $T \to \infty$ uniform.

**43. UCB principle?**
Optimism in the face of uncertainty. Add bonus to less-tried actions: $a = \arg\max [Q + c\sqrt{\log t/N(s,a)}]$.

**44. Entropy bonus — what does it do?**
Adds $\beta H(\pi(\cdot|s))$ to the loss. Encourages diverse actions; prevents premature collapse to deterministic policy.

**45. Curiosity-driven exploration?**
Reward novelty (unpredicted states). Useful in sparse-reward problems where extrinsic reward signal is rare.

---

## H. RL for LLMs

**46. RLHF state, action, reward?**
State: prompt + generated tokens so far. Action: next token. Reward: from learned reward model at end of sequence (or rule-based for verifiable tasks).

**47. Why KL penalty in RLHF?**
Prevents the policy from drifting too far from the SFT model. Acts as regularization; prevents reward hacking.

**48. PPO objective for RLHF?**
$\mathcal{L} = \mathbb{E}[\mathrm{clip}\,\mathrm{surrogate} - \beta \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})]$.

**49. GRPO simplification over PPO?**
Drops value/critic network. Computes advantage via group-relative reward normalization (sample $K$ responses per prompt, compare rewards within group). Used in DeepSeekMath, DeepSeek-R1.

**50. Reward hacking in RLHF?**
Policy finds high-reward outputs that don't correspond to truly good behavior — exploits reward model errors. Mitigated by KL penalty, robust reward modeling, evaluation on held-out tasks.

---

## Quick fire

**51.** *Q-learning is on/off-policy?* Off.
**52.** *SARSA is on/off-policy?* On.
**53.** *Discount factor $\gamma$ range?* $[0, 1)$.
**54.** *DQN target network update?* Slowly (every $K$ steps or Polyak).
**55.** *Policy gradient log trick?* $\nabla p = p \nabla \log p$.
**56.** *PPO standard $\epsilon$?* 0.2.
**57.** *GAE $\lambda$?* Trade variance vs bias.
**58.** *RLHF main RL algo?* PPO (or GRPO).
**59.** *Bellman optimality is fixed point of?* $\mathcal{T}^*$ operator.
**60.** *DPO is RL?* No — direct preference optimization, no RL loop.

---

## Self-grading

If you can't answer 1-15, you don't know RL basics. If you can't answer 16-35, you'll struggle on RLHF/PPO interview questions. If you can't answer 36-50, frontier-lab interviews on alignment will go past you.

Aim for 40+/60 cold.
