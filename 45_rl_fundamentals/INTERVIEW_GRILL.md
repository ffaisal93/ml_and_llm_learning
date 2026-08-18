# RL Fundamentals — Interview Grill

> 50 questions on MDPs, value functions, Q-learning, policy gradients, PPO. Drill until you can answer 35+ cold.

---

## A. MDPs and value functions

**1. State the components of an MDP.**
$(S, A, P, R, \gamma)$ — states, actions, transitions, reward, discount.

> **Saying it out loud.** An MDP is the standard way to write down a decision problem: the situations you can be in, the things you can do, the rule for how the world responds, the score you get, and how much you care about the future. That last one, the discount factor, is the one people forget to mention and it's the one that does the most quiet work. If you want to sound like you've used this rather than read it, add that the hard part in practice is never the math — it's that real problems don't hand you a clean state, so most of the engineering is deciding what goes into it.

**2. State the Markov property.**
$P(s_{t+1}|s_t, a_t, s_{t-1}, \ldots) = P(s_{t+1}|s_t, a_t)$. Future depends only on current state-action.

> **Saying it out loud.** The Markov property says the present is a sufficient summary of the past — if you know where you are now, how you got there tells you nothing extra. That's what makes the whole framework tractable, because otherwise your value function would have to depend on entire histories. It's an assumption, not a law, and it's routinely false: in poker you can't see the opponents' cards, in dialogue the state is the whole conversation. The standard dodge is to stuff enough history into the state to make it approximately true, which is exactly what a transformer's context window is doing.

**3. Define discounted return.**
$G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$.

> **Saying it out loud.** The return is just the total reward from here on out, with each future reward multiplied by gamma raised to how far away it is. So a reward ten steps out counts for gamma to the tenth. With gamma at 0.99 you're effectively looking about a hundred steps ahead; at 0.9 it's more like ten. That's the practical way to think about it — gamma isn't a philosophical parameter, it's your planning horizon, roughly one over one minus gamma.

**4. Why discount?**
Bounded value when rewards bounded; favors sooner rewards; mathematical convenience (Bellman fixed point unique with $\gamma < 1$).

> **Saying it out loud.** Three reasons, one honest and two convenient. The honest one is that a reward now really is worth more than a reward later — uncertainty, and the episode might end. The convenient ones are that without discounting an infinite-horizon sum can be infinite, and that gamma strictly below one is what makes the Bellman operator a contraction, which is what guarantees a unique solution and convergence. The practical consequence is that gamma silently sets your horizon, and picking 0.9 when the reward comes 200 steps later means your agent literally cannot see it.

**5. State-value $V^\pi$ vs action-value $Q^\pi$?**
$V^\pi(s) = \mathbb{E}_\pi[G_t|s_t=s]$. $Q^\pi(s,a) = \mathbb{E}_\pi[G_t|s_t=s, a_t=a]$.

> **Saying it out loud.** $V$ answers 'how good is it to be here, playing my usual way,' and $Q$ answers 'how good is it to be here and do this specific thing first, then play my usual way.' The reason we bother with $Q$ is that it's directly actionable — you can pick the best action by comparing $Q$ values, whereas with $V$ alone you'd need the transition model to know where each action leads. That's exactly why Q-learning is model-free and value iteration isn't.

**6. Define advantage.**
$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$. How much better is action $a$ than the policy's average.

> **Saying it out loud.** Advantage is how much better this action is than what you'd normally do here — $Q$ minus $V$. The reason it matters is that raw $Q$ values carry a big shared offset from just being in a good state, and that offset is noise from the perspective of choosing an action. Subtracting $V$ cancels it, which is why every modern policy gradient method weights its updates by advantage rather than by return. It's the single biggest variance reduction in the field and it costs you nothing in bias.

**7. Bellman equation for $V^\pi$?**
$V^\pi(s) = \mathbb{E}_a[R + \gamma V^\pi(s')]$ — expectation over policy and dynamics.

> **Saying it out loud.** The Bellman equation is just self-consistency: the value of being here equals the reward you're about to get plus the discounted value of wherever you land. It's not a formula you derive so much as a statement that your estimates should agree with themselves one step apart. Every algorithm in this document is some way of enforcing that agreement — dynamic programming solves it exactly, TD learning nudges toward it from samples.

**8. Bellman optimality for $V^*$?**
$V^*(s) = \max_a \mathbb{E}[R + \gamma V^*(s')]$. Take max over actions.

> **Saying it out loud.** Same self-consistency, with one change: instead of averaging over what your current policy would do, you take the best action. That single max is what turns an evaluation equation into an optimization one, and it's what makes it nonlinear — you can no longer just solve a linear system, you have to iterate. It's also where Q-learning's optimism comes from, since a max over noisy estimates is biased upward.

**9. Why does value iteration converge?**
**Each iteration shrinks the error by a factor of $\gamma$**, so it converges geometrically. Formally: the Bellman optimality operator is a $\gamma$-contraction in sup-norm — Banach fixed-point theorem then guarantees a unique fixed point and convergence from any start.

> **Saying it out loud.** Because every sweep shrinks your error by a factor of gamma, so it decays geometrically no matter where you started. The formal version is that the Bellman optimality operator is a contraction in the max-norm and Banach's theorem does the rest, giving you a unique fixed point. The practical reading is that convergence speed is entirely governed by gamma: at 0.99 you need roughly a hundred times more sweeps to reach the same accuracy as at 0.9, which is the hidden cost of a long planning horizon.

---

## B. Dynamic programming

**10. Value iteration update?**
$V_{k+1}(s) = \max_a \mathbb{E}[R + \gamma V_k(s')]$.

> **Saying it out loud.** You sweep every state and set its value to the best one-step lookahead: the immediate reward plus the discounted value of where you'd land, maximized over actions. Repeat until nothing moves. Effectively each sweep pushes information one step further back from the rewards, so if your goal is 50 steps away it takes about 50 sweeps before the start state knows anything. That's the intuition for why sparse, distant rewards are hard.

**11. Convergence rate of value iteration?**
Geometric, rate $\gamma$.

> **Saying it out loud.** Geometric at rate gamma — every sweep multiplies your remaining error by gamma. So the number of sweeps you need scales like one over one minus gamma, which is why long-horizon problems are expensive even when the state space is small. At gamma 0.999 that factor is a thousand.

**12. Policy iteration steps?**
(1) Policy evaluation — solve $V^\pi$ as linear system. (2) Policy improvement — $\pi'(s) = \arg\max_a Q^\pi(s,a)$.

> **Saying it out loud.** Two phases you alternate. First evaluate: pin down how good your current policy is at every state, which is a linear system because there's no max once the policy is fixed. Then improve: at each state, switch to whichever action looks best under those values. You repeat, and the improvement step is guaranteed never to make things worse — that monotonicity is why it terminates, and typically in under ten rounds.

**13. Value vs policy iteration — when each?**
Both find optimal policy. Policy iteration often converges in fewer iterations but each iteration is more expensive (exact policy evaluation).

> **Saying it out loud.** They converge to the same place; the difference is how much work you do per round. Policy iteration usually needs very few iterations, often under ten, but each one includes a full policy evaluation, which is itself a solve. Value iteration does one cheap sweep per round and needs many more. In practice people use modified policy iteration — evaluate partially, improve, repeat — which is the sweet spot between the two.

---

## C. Model-free TD methods

**14. TD(0) update for $V$?**
$V(s_t) \leftarrow V(s_t) + \alpha [r_t + \gamma V(s_{t+1}) - V(s_t)]$.

> **Saying it out loud.** You move your estimate a little toward what one step of actual experience suggests: the reward you just got plus your estimate of the next state. The gap between that and your current estimate is the error, and alpha controls how far you step toward fixing it. What makes it powerful is that you learn from a single transition without waiting for the episode to end, so it works in continuing tasks where Monte Carlo simply has nothing to average.

**15. What's the TD error?**
$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

> **Saying it out loud.** The TD error is the surprise — what you just observed, plus what you now think the future is worth, minus what you predicted before. Positive means things went better than expected. It's the learning signal for basically every value-based method in RL, and it's also the quantity dopamine neurons appear to encode in the brain, which is one of the more striking coincidences in the field.

**16. Q-learning update?**
$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$.

> **Saying it out loud.** Same shape as TD, but on state-action pairs and with a max in the target: you bootstrap off the best action available in the next state, not the one you actually took. That max is what makes it off-policy — you can behave randomly, or badly, and still converge to the optimal policy. The same max is also its weakness, because maximizing over noisy estimates systematically overestimates, which is what Double Q-learning was built to correct.

**17. SARSA update?**
$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$. Uses next action actually taken.

> **Saying it out loud.** SARSA is Q-learning with the max replaced by the action you actually took next. The name is literally the tuple it uses — state, action, reward, state, action. That one substitution makes it on-policy: it learns the value of the policy you're really running, exploration mistakes included, rather than the value of a hypothetical optimal one.

**18. Q-learning vs SARSA: on or off-policy?**
Q-learning: off-policy (uses max regardless of behavior). SARSA: on-policy (uses behavior policy's action).

> **Saying it out loud.** Q-learning is off-policy because its target uses the max, which is a policy you're not actually following. SARSA is on-policy because its target uses the action your behavior policy actually chose. The consequence that matters is data reuse: off-policy methods can learn from a replay buffer, from old data, even from someone else's demonstrations, while on-policy methods have to throw data away after each update. That's precisely why PPO is so much more sample-hungry than DQN.

**19. Why might SARSA learn safer policies?**
SARSA accounts for the actual exploration (e.g., $\epsilon$-greedy) → may avoid risky paths. Q-learning learns optimal regardless.

> **Saying it out loud.** The cliff-walking example is the one to tell. There's a shortest path running right along the edge of a cliff, and Q-learning learns it because its target assumes you'll act optimally afterward. SARSA's target includes your actual epsilon-greedy behavior, so it knows there's a real chance you'll randomly step off the edge, and it learns a safer route slightly further back. So SARSA optimizes the policy you're actually running, which is the more honest objective if you'll keep exploring in deployment.

**20. Monte Carlo vs TD — bias and variance?**
MC unbiased high variance (uses full return). TD biased lower variance (uses bootstrap).

> **Saying it out loud.** Monte Carlo waits for the real outcome, so it's unbiased but extremely noisy — a single unlucky episode swings the update. TD bootstraps off its own current estimate, so it's biased while those estimates are wrong, but far less noisy and it can update every step. That's the bias-variance tradeoff in the cleanest form you'll ever see it, and TD-lambda is the dial that lets you sit anywhere in between. In practice TD wins on almost every real problem, because variance hurts more than a bias that shrinks as you learn.

---

## D. DQN

**21. DQN loss?**
$\mathcal{L} = \mathbb{E}[(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s,a))^2]$.

> **Saying it out loud.** It's just squared error between your predicted $Q$ and a target built from the reward plus the discounted best next value. The two things to point at are that the target comes from a separate frozen network, and that you treat it as a constant — you don't backpropagate through it. Skip that detail and you're differentiating both sides of the Bellman equation, which is a known way to make training diverge.

**22. Why experience replay?**
Breaks temporal correlation between consecutive samples; allows reuse of data; more iid-like batches for SGD.

> **Saying it out loud.** Two reasons. Consecutive frames in an episode are nearly identical, and SGD assumes roughly independent samples, so training on them in order means your gradients are correlated and the network chases whatever it just saw. And it lets you reuse each transition many times instead of once, which matters enormously when interaction is the expensive part. The typical buffer holds a million transitions, and adding it was one of the two changes that turned Q-learning from unstable to Atari-beating.

**23. Why a target network?**
Stabilizes training. Without it, the target $Q_{\theta^-}$ shifts with each update — chasing your own tail. Update target slowly (every $K$ steps or Polyak average).

> **Saying it out loud.** Because otherwise you're regressing toward a target that moves every time you update, which is chasing your own tail. Freeze a copy of the network and use it to compute targets for a few thousand steps, and suddenly the problem looks like ordinary supervised regression against a fixed label. The tradeoff is staleness — update it too slowly and you're learning from outdated values, too quickly and the instability comes back. Every 10,000 steps was the DQN default; Polyak averaging is the smooth version.

**24. Q-learning overestimates — why?**
$\max_a Q$ tends to overestimate due to noise. Sampling errors get amplified by max.

> **Saying it out loud.** Because the expectation of a max is bigger than the max of expectations. If your $Q$ estimates are noisy but unbiased, taking the max preferentially picks whichever action got lucky, so the target is systematically too high, and that inflated target then feeds back into the next update. The bias compounds across bootstrapping. It shows up in practice as value estimates drifting far above any return the agent ever actually achieves — a good diagnostic to watch.

**25. Double DQN fix?**
Use online net to *select* action, target net to *evaluate*: $r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a'))$. Decouples selection and evaluation.

> **Saying it out loud.** Split the two jobs the max was doing. Use the online network to pick which action looks best, then use the target network to say how good that action actually is. Since the two networks have independent noise, an action that got lucky in one is unlikely to get lucky in the other, so the upward bias largely cancels. It's a one-line change with no extra network, and it measurably improved scores across the Atari suite — one of the best effort-to-payoff ratios in deep RL.

**26. Dueling DQN — what does it split?**
Network outputs $V(s)$ and $A(s, a)$ separately, then $Q(s, a) = V(s) + (A(s, a) - \mathrm{mean}_a A(s, a))$. Better when only some actions matter.

> **Saying it out loud.** It splits $Q$ into 'how good is this state' plus 'how much better is this action than average.' The reason that helps is that in many states no action matters much — you're driving down an empty road — and you'd rather learn the state's value once from every sample than learn it separately inside every action's $Q$. The subtraction of the mean advantage is there to make the decomposition identifiable, since otherwise you could add a constant to $V$ and subtract it from $A$ with no change.

**27. Prioritized replay?**
Sample high-TD-error transitions more often. Importance weights correct the bias.

> **Saying it out loud.** Instead of sampling uniformly from the buffer, sample transitions with large TD error more often, on the theory that those are the ones you still have something to learn from. It speeds up learning noticeably. The catch is that non-uniform sampling biases the expected gradient, so you correct with importance weights — and if you skip that correction you've quietly changed the objective you're optimizing.

---

## E. Policy gradient

> **In plain language.** This section is about learning the policy directly rather than learning values and acting greedily. The central formula looks intimidating, but it's saying something very simple: increase the probability of actions that turned out well, decrease the probability of ones that didn't, in proportion to how well they did.

**28. State the policy gradient theorem.**
$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s, a)]$. **Intuition (the whole point):** push up the log-probability of actions, weighted by how good they were. Good action → push it up; bad action → push it down. That's it.

> **Saying it out loud.** Strip the notation and it says: take the actions you sampled, and push their probabilities up or down in proportion to how good they turned out. The gradient of the log-probability is the direction in parameter space that makes this action more likely, and $Q$ is the weight telling you how hard to push and which way. That's genuinely the whole theorem. The reason it's a theorem rather than an observation is that it holds even though changing the policy changes which states you visit — the state-distribution term drops out, which is the non-obvious part.

**29. Log-derivative trick — what is it?**
$\nabla \log p(x;\theta) = \nabla p(x;\theta)/p(x;\theta)$. Lets you write expectation gradient as expectation of (log-prob gradient × value).

> **Saying it out loud.** It's the algebraic move that makes policy gradients possible. You want the gradient of an expectation, but the thing you're averaging over depends on the parameters, so you can't just push the gradient inside. Rewriting the gradient of $p$ as $p$ times the gradient of log $p$ turns it back into an expectation you can estimate by sampling. It's also called the score function estimator or REINFORCE, and it's the same trick behind variational inference — worth knowing under all three names because interviewers use them interchangeably.

**30. REINFORCE estimator?**
$\nabla J \approx \frac{1}{N}\sum_i \nabla \log \pi(a_i|s_i) G_i$ with $G_i$ the empirical return.

> **Saying it out loud.** REINFORCE is the simplest possible thing: run an episode, see the total reward, and scale every action's log-probability gradient by that number. If the episode went well, everything you did gets reinforced. That's also its problem — an action taken at step three gets credit for a reward at step three hundred, so the variance is enormous and it needs thousands of episodes for problems that TD methods solve in dozens. Every method after it is fundamentally a variance-reduction story.

**31. Why use a baseline?**
Reduces variance without bias. $\mathbb{E}[\nabla \log \pi \cdot b(s)] = b(s) \mathbb{E}[\nabla \log \pi] = 0$ for any state-only baseline.

> **Saying it out loud.** Because you only care about whether an action was better than usual, not whether the state was good. Subtracting any function of the state leaves the gradient unbiased in expectation — the proof is that the expected gradient of the log-probability is zero — but it hugely reduces variance. Concretely, if every action in a state returns about 100, weighting by 100 gives you a big noisy push on everything; subtracting the 100 leaves only the differences, which is the actual signal.

**32. What's the optimal baseline?**
$b^*(s) = \mathbb{E}[Q^\pi(s,a) | s] = V^\pi(s)$ minimizes variance of the gradient estimator.

> **Saying it out loud.** The variance-minimizing baseline is technically a gradient-magnitude-weighted average of $Q$, but in practice everyone uses $V(s)$, which is nearly as good and has an obvious interpretation: $Q$ minus $V$ is the advantage. That's the entire justification for actor-critic — the critic exists to estimate the baseline. So 'use a value function as your baseline' and 'weight by advantage' are the same statement said two ways.

**33. Actor-critic — actor and critic do what?**
Actor: policy $\pi_\theta$. Critic: value function $V_\phi$ (or $Q_\phi$). Critic provides advantage estimates.

> **Saying it out loud.** The actor is the policy that picks actions; the critic is a value function that judges them. The actor gets updated in the direction the critic says is good, and the critic gets updated by ordinary TD learning against observed rewards. The win over REINFORCE is that you no longer wait for the episode to finish to know whether an action was good — the critic gives you an immediate opinion. The risk is that you're now training on the critic's opinions, so a bad critic actively misleads the actor, and the two can destabilize each other.

**34. A2C vs A3C?**
A2C: synchronous (one update from all parallel actors). A3C: asynchronous (workers update parameters independently).

> **Saying it out loud.** Both run many actors in parallel to decorrelate data; the difference is coordination. A3C lets each worker update the shared parameters whenever it's ready, which means workers are computing gradients against slightly stale weights. A2C waits for all of them and does one batched synchronous update. Synchronous turned out to be both simpler and better on GPUs, because you get one big batch instead of many small ones, so A2C is what survived.

---

## F. PPO

**35. Why does naive policy gradient fail with large updates?**
Policy can collapse — large step takes you to a region where $\pi$ assigns near-zero probability to actions you're trying to reinforce. Hard to recover.

> **Saying it out loud.** Because a small step in parameter space can be an enormous step in policy space, and there's no way back. If one update drops the probability of a good action to near zero, you'll basically never sample it again, so you can't learn that it was good — the data you need has disappeared. Supervised learning doesn't have this failure mode because the dataset is fixed; in RL your policy determines your data, so a bad update poisons the well. That's the entire motivation for trust regions and clipping.

**36. TRPO constraint?**
Maximize surrogate subject to $\mathrm{KL}(\pi_{\mathrm{old}} \| \pi_\theta) \leq \delta$. Update step in KL geometry.

> **Saying it out loud.** TRPO says: improve the surrogate objective as much as you can, but don't let the new policy differ from the old one by more than a fixed KL divergence. The key insight is that KL measures distance in the space of distributions, which is what actually matters, rather than Euclidean distance in parameter space, which doesn't. It works well and it's a pain to implement — you need conjugate gradients and Fisher-vector products and a line search. PPO exists because people wanted TRPO's behavior in twenty lines of code.

**37. PPO clipped surrogate?**
$L = \mathbb{E}[\min(r A, \mathrm{clip}(r, 1-\epsilon, 1+\epsilon) A)]$ with $r = \pi_\theta/\pi_{\mathrm{old}}$. Standard $\epsilon = 0.2$.

> **Saying it out loud.** PPO looks at the ratio of new to old action probability and refuses to reward you for pushing it beyond about 20 percent in either direction. Taking the minimum of the clipped and unclipped terms is what makes it one-sided — you get no benefit from moving too far, but if the action was bad you're still fully penalized for having made it more likely. That asymmetry is the actual cleverness, and it's why a naive symmetric clip doesn't work. Epsilon of 0.2 is the near-universal default.

**38. Why clip ratio $r$ instead of constraining KL?**
Simpler, no Lagrangian. Heuristic but works extremely well in practice.

> **Saying it out loud.** Because it gets you most of TRPO's benefit for a fraction of the complexity — no second-order optimization, no Lagrange multipliers, just a min and a clamp that you can implement in a few lines and run with Adam. It's a heuristic, and it doesn't actually guarantee the KL stays bounded, which is why serious RLHF implementations still monitor KL and often add an explicit penalty or early-stop on it. But the empirical record is overwhelming: PPO is the default because it works with almost no tuning.

**39. What's GAE and what does $\lambda$ control?**
**Intuition**: GAE blends short-horizon TD (low variance, bootstrapped from value estimate) and long-horizon Monte Carlo (high variance, true returns). $\lambda$ slides between them — trade bias vs variance.

**Formula**: $A^{\mathrm{GAE}(\lambda)}_t = \sum_{l \geq 0} (\gamma\lambda)^l \delta_{t+l}$ where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. $\lambda=0$ → pure TD; $\lambda=1$ → Monte Carlo. Standard for PPO: $\lambda \approx 0.95$.

> **Saying it out loud.** GAE is a dial between two ways of estimating how good an action was. At one extreme you use a single step of real reward plus your value estimate — low noise, but wrong if your value function is bad. At the other you use the entire observed return — correct, but wildly noisy. Lambda exponentially blends every horizon in between, and 0.95 is the standard setting because it keeps most of the variance reduction while limiting the bias from a mediocre critic. It's the same bias-variance tradeoff as TD-lambda, wearing a different hat.

**40. Standard $\lambda$ for PPO?**
0.95.

> **Saying it out loud.** 0.95, with gamma at 0.99. Those two together set your effective credit-assignment horizon at a few dozen steps. It's worth knowing they interact — the product gamma-lambda is what actually controls the decay in GAE — so dropping lambda to 0.9 shortens your horizon just like lowering gamma does, and people tune them as a pair.

---

## G. Exploration

**41. $\epsilon$-greedy?**
With prob $\epsilon$, random action; else greedy. Simple but widely used.

> **Saying it out loud.** Act greedily most of the time, and with probability epsilon just pick at random. It's the crudest possible exploration and it's still everywhere because it's one line and it works. The weakness is that it explores uniformly — it's as likely to try an action it has already proven terrible as one it's genuinely unsure about. Usually you anneal epsilon from 1.0 down to about 0.05 over training, and if you don't anneal it you keep paying that random-action tax forever.

**42. Boltzmann exploration?**
$\pi(a|s) \propto \exp(Q(s,a)/T)$. $T$ controls exploration; $T \to 0$ greedy, $T \to \infty$ uniform.

> **Saying it out loud.** Instead of a flat random choice, sample actions in proportion to the exponential of their values, scaled by a temperature. High temperature is nearly uniform, low temperature is nearly greedy. It's better than epsilon-greedy in that a clearly bad action rarely gets picked while two near-tied actions both stay in play. The awkward part is that the temperature is in reward units, so a sensible value depends on the scale of your rewards and doesn't transfer between problems. This is exactly the softmax temperature you already know from LLM sampling.

**43. UCB principle?**
Optimism in the face of uncertainty. Add bonus to less-tried actions: $a = \arg\max [Q + c\sqrt{\log t/N(s,a)}]$.

> **Saying it out loud.** Optimism in the face of uncertainty: rate each action by its estimated value plus a bonus that grows when you haven't tried it much. So an untested action looks attractive purely because you're uncertain, and that uncertainty shrinks as you sample it. The appeal is that it's principled rather than random — you explore where you might be wrong, not everywhere. In bandits it gives you logarithmic regret, versus linear for fixed-epsilon greedy, and it's the backbone of the tree search in AlphaGo.

**44. Entropy bonus — what does it do?**
Adds $\beta H(\pi(\cdot|s))$ to the loss. Encourages diverse actions; prevents premature collapse to deterministic policy.

> **Saying it out loud.** You add a term rewarding the policy for staying uncertain, which stops it from collapsing to a deterministic choice before it has really explored. The failure mode it prevents is a policy that finds a mediocre strategy, becomes confident, and then can't sample the alternatives that would have taught it better. The tradeoff is that too much entropy bonus leaves you permanently random and never converging. Typical coefficients are around 0.01, and in RLHF it matters less because the KL penalty against the reference model is already doing similar work.

**45. Curiosity-driven exploration?**
Reward novelty (unpredicted states). Useful in sparse-reward problems where extrinsic reward signal is rare.

> **Saying it out loud.** When the real reward almost never fires, you invent an internal one: reward the agent for visiting states it couldn't predict. That way it has something to optimize during the long stretch before it stumbles onto real reward. It's what got agents through hard exploration games like Montezuma's Revenge. The famous failure mode is the noisy-TV problem — put a screen of static in the environment and the agent will happily watch it forever, because randomness is permanently unpredictable and therefore permanently rewarding.

---

## H. RL for LLMs

**46. RLHF state, action, reward?**
State: prompt + generated tokens so far. Action: next token. Reward: from learned reward model at end of sequence (or rule-based for verifiable tasks).

> **Saying it out loud.** Map it token by token: the state is the prompt plus everything generated so far, the action is the next token, and the reward comes at the end from a learned reward model that scores the whole response. So it's a long episode — hundreds of steps — with a single reward at the finish, which is about the hardest credit-assignment setup there is. That's why RLHF leans so heavily on the KL penalty and a good value function, and why verifiable domains like math and code are easier, since there you can replace the learned reward model with an actual correctness check.

**47. Why KL penalty in RLHF?**
Prevents the policy from drifting too far from the SFT model. Acts as regularization; prevents reward hacking.

> **Saying it out loud.** Because the reward model is a flawed approximation, and any policy optimized hard enough will find its cracks rather than actually getting better. The KL term tethers you to the fine-tuned starting point, so you can improve on what the reward model measures without wandering into weird regions where its scores are meaningless. Drop it and you get textbook reward hacking — outputs that score beautifully and read as gibberish, or that endlessly repeat whatever phrasing the reward model happens to like. The coefficient is a real tradeoff dial: too high and nothing changes, too low and the model degenerates.

**48. PPO objective for RLHF?**
$\mathcal{L} = \mathbb{E}[\mathrm{clip}\,\mathrm{surrogate} - \beta \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})]$.

> **Saying it out loud.** It's the standard clipped surrogate with a KL penalty against the reference model subtracted off. So you have two safety mechanisms doing different jobs: the clip limits how far you move in one update, and the KL term limits how far you drift from the original model overall. People conflate them, and the distinction is worth making — per-step versus cumulative. In practice the KL is usually applied as a per-token penalty folded into the reward rather than as a separate loss term.

**49. GRPO simplification over PPO?**
Drops value/critic network. Computes advantage via group-relative reward normalization (sample $K$ responses per prompt, compare rewards within group). Used in DeepSeekMath, DeepSeek-R1.

> **Saying it out loud.** GRPO throws away the critic entirely. Instead of training a value network to estimate a baseline, you sample a group of responses to the same prompt and use the group's mean reward as the baseline — an answer that beats its siblings gets a positive advantage. That saves you from training and storing a second model the size of the policy, which is a large chunk of memory at scale, and it removes a notorious source of instability since a bad critic misleads the actor. DeepSeekMath introduced it and R1 made it famous. The cost is that you need multiple samples per prompt, typically 8 to 64, so you trade critic compute for generation compute.

**50. Reward hacking in RLHF?**
Policy finds high-reward outputs that don't correspond to truly good behavior — exploits reward model errors. Mitigated by KL penalty, robust reward modeling, evaluation on held-out tasks.

> **Saying it out loud.** Reward hacking is the policy finding outputs that score well without being good — exploiting the reward model's mistakes rather than satisfying the intent behind it. The classic symptoms are verbosity, since reward models tend to like longer answers, and sycophancy, since they like agreement. It's Goodhart's law with a gradient-based optimizer pointed at it. The defenses are the KL penalty, retraining the reward model on the policy's own failures, ensembling reward models, and evaluating on held-out human judgments rather than trusting the reward curve — which, notably, keeps going up the whole time this is happening.

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
