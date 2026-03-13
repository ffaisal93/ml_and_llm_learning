"""
Reinforcement Learning Fundamentals: Complete Implementations
MDP, Multi-Armed Bandit, Q-Learning, Monte Carlo, etc.
Simple, easy-to-understand code
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

# ==================== MULTI-ARMED BANDIT ====================

class MultiArmedBandit:
    """
    Multi-Armed Bandit
    
    SIMPLEST RL PROBLEM:
    - Multiple actions (arms)
    - Each gives random reward
    - Goal: Find best arm
    - Challenge: Exploration vs Exploitation
    
    EASY EXPLANATION:
    You're in a casino with slot machines. Each machine gives
    different amounts of money, but you don't know which is best.
    You have limited plays. Should you explore (try new machines)
    or exploit (play the best one you know)?
    """
    def __init__(self, num_arms: int, true_rewards: Optional[List[float]] = None):
        """
        Args:
            num_arms: Number of slot machines (actions)
            true_rewards: True average reward for each arm (for simulation)
        """
        self.num_arms = num_arms
        if true_rewards is None:
            # Random true rewards (unknown to the agent)
            self.true_rewards = np.random.normal(0, 1, num_arms)
        else:
            self.true_rewards = np.array(true_rewards)
        
        # Agent's estimates (what it has learned)
        self.estimates = np.zeros(num_arms)  # Estimated average reward per arm
        self.counts = np.zeros(num_arms)  # How many times each arm was played
    
    def pull_arm(self, arm: int) -> float:
        """
        Pull an arm and get reward
        
        Args:
            arm: Which arm to pull (0 to num_arms-1)
        Returns:
            Reward (random around true reward)
        """
        # Reward = true reward + noise
        reward = self.true_rewards[arm] + np.random.normal(0, 0.1)
        return reward
    
    def epsilon_greedy(self, epsilon: float = 0.1) -> int:
        """
        Epsilon-Greedy Strategy
        
        EASY EXPLANATION:
        - Most of the time (1-ε): Play the best arm you know (exploit)
        - Sometimes (ε): Try a random arm (explore)
        
        This balances: "Use what you know is good" vs "Try new things"
        
        Args:
            epsilon: Probability of exploring (0.1 = 10% explore, 90% exploit)
        Returns:
            Which arm to pull
        """
        if random.random() < epsilon:
            # Explore: try random arm
            return random.randint(0, self.num_arms - 1)
        else:
            # Exploit: play best arm
            return np.argmax(self.estimates)
    
    def update(self, arm: int, reward: float):
        """
        Update estimates after pulling an arm
        
        EASY EXPLANATION:
        After pulling an arm and getting a reward, update your
        estimate of how good that arm is. Use running average.
        
        Args:
            arm: Which arm was pulled
            reward: Reward received
        """
        self.counts[arm] += 1
        # Running average: new_estimate = old_estimate + (1/n) * (reward - old_estimate)
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]
    
    def ucb(self, c: float = 2.0) -> int:
        """
        Upper Confidence Bound (UCB) Strategy
        
        EASY EXPLANATION:
        Play arms that either:
        - Have high average reward (exploit)
        - Haven't been tried much (explore)
        
        Automatically balances exploration and exploitation!
        
        Args:
            c: Exploration constant (higher = more exploration)
        Returns:
            Which arm to pull
        """
        total_pulls = self.counts.sum()
        if total_pulls == 0:
            return random.randint(0, self.num_arms - 1)
        
        # UCB value = average_reward + c * sqrt(log(total_pulls) / arm_pulls)
        ucb_values = self.estimates + c * np.sqrt(
            np.log(total_pulls + 1) / (self.counts + 1e-10)
        )
        return np.argmax(ucb_values)


# ==================== Q-LEARNING ====================

class QLearning:
    """
    Q-Learning Algorithm
    
    EASY EXPLANATION:
    Q-Learning learns "how good" each action is in each state.
    Q(s, a) = "If I'm in state s and take action a, then do my best
    from there, how much total reward will I get?"
    
    HOW IT WORKS:
    1. Try actions, see what happens
    2. Update Q-values: "If I was in state s, took action a, got
       reward r, and ended up in state s', then Q(s,a) should be
       r + best_I_can_do_from_s'"
    3. Repeat many times
    4. Eventually, Q-values converge to true values
    5. Then just pick action with highest Q-value in each state
    
    KEY PROPERTY: Off-policy
    - Can learn optimal policy while following different policy
    - Can explore randomly but still learn the best way to act
    """
    def __init__(self, num_states: int, num_actions: int,
                 learning_rate: float = 0.1, discount: float = 0.9,
                 epsilon: float = 0.1):
        """
        Args:
            num_states: Number of possible states
            num_actions: Number of possible actions
            learning_rate: How fast to learn (alpha)
            discount: How much to care about future rewards (gamma)
            epsilon: Exploration rate for epsilon-greedy
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        
        # Q-table: Q[state][action] = Q-value
        self.Q = defaultdict(lambda: np.zeros(num_actions))
    
    def choose_action(self, state: int) -> int:
        """
        Choose action using epsilon-greedy
        
        EASY EXPLANATION:
        - Most of the time: Pick best action (highest Q-value)
        - Sometimes: Try random action (explore)
        """
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: best action
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool = False):
        """
        Update Q-value using Q-Learning update rule
        
        UPDATE RULE:
        Q(s, a) ← Q(s, a) + α [r + γ * max Q(s', a') - Q(s, a)]
        
        EASY EXPLANATION:
        - Current Q-value: Q(s, a)
        - What it should be: r + γ * best_from_next_state
        - Update: Move current value towards what it should be
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            done: Whether episode ended
        """
        current_q = self.Q[state][action]
        
        if done:
            # Episode ended, no future rewards
            target = reward
        else:
            # Future rewards: best we can do from next state
            next_max_q = np.max(self.Q[next_state])
            target = reward + self.gamma * next_max_q
        
        # Update: current + learning_rate * (target - current)
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def get_policy(self) -> Dict[int, int]:
        """
        Get optimal policy: in each state, pick action with highest Q-value
        
        Returns:
            Dictionary: state -> best action
        """
        policy = {}
        for state in self.Q.keys():
            policy[state] = np.argmax(self.Q[state])
        return policy


# ==================== MONTE CARLO POLICY EVALUATION ====================

class MonteCarlo:
    """
    Monte Carlo Policy Evaluation
    
    EASY EXPLANATION:
    Monte Carlo = "Try it and see what happens"
    
    HOW IT WORKS:
    1. Follow a policy and play complete episodes
    2. For each state you visited, collect the total reward
       you got from that point until the end
    3. Average these returns - that's your estimate of
       how good each state is
    
    EXAMPLE:
    Play chess game. If you win, all your moves were probably
    good. If you lose, all your moves were probably bad.
    Look back through the game and learn from the outcome.
    
    KEY PROPERTY: Model-free
    - Don't need to know how environment works
    - Just need to be able to play and see results
    """
    def __init__(self, num_states: int, discount: float = 0.9):
        """
        Args:
            num_states: Number of possible states
            discount: Discount factor for future rewards
        """
        self.num_states = num_states
        self.gamma = discount
        
        # Value estimates for each state
        self.V = np.zeros(num_states)
        
        # Store returns for each state (for averaging)
        self.returns = defaultdict(list)
    
    def evaluate_policy(self, episodes: List[List[Tuple[int, float]]]):
        """
        Evaluate policy using Monte Carlo
        
        EASY EXPLANATION:
        For each episode:
        1. Calculate return (total reward) from each state
        2. Add to list of returns for that state
        3. Average all returns for each state
        
        Args:
            episodes: List of episodes, each episode is list of (state, reward) tuples
        """
        for episode in episodes:
            # Calculate returns (total reward from each state to end)
            G = 0  # Return (total reward)
            returns = []
            
            # Go backwards through episode
            for state, reward in reversed(episode):
                G = reward + self.gamma * G  # Discounted return
                returns.append((state, G))
            
            # Store returns for each state
            for state, return_value in returns:
                self.returns[state].append(return_value)
            
            # Update value estimates (average of all returns)
            for state in range(self.num_states):
                if len(self.returns[state]) > 0:
                    self.V[state] = np.mean(self.returns[state])


# ==================== VALUE ITERATION ====================

class ValueIteration:
    """
    Value Iteration Algorithm
    
    EASY EXPLANATION:
    Value Iteration finds "how good" each state is (value function).
    Then you can derive the best policy from values.
    
    HOW IT WORKS:
    1. Start with random values for each state
    2. For each state, update its value:
       "My value = best action I can take = max over actions of
       (immediate_reward + value_of_next_state)"
    3. Repeat until values stop changing
    4. Then: optimal policy = in each state, take action that
       leads to best value
    
    REQUIREMENT: Need to know environment model
    - Transition probabilities: P(s' | s, a)
    - Rewards: R(s, a, s')
    
    This is "planning" - you have a map and plan the route,
    rather than exploring and learning.
    """
    def __init__(self, num_states: int, num_actions: int,
                 transitions: Dict[Tuple[int, int], List[Tuple[int, float, float]]],
                 discount: float = 0.9, threshold: float = 1e-6):
        """
        Args:
            num_states: Number of states
            num_actions: Number of actions
            transitions: Dict mapping (state, action) -> [(next_state, prob, reward), ...]
            discount: Discount factor
            threshold: Convergence threshold
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.transitions = transitions
        self.gamma = discount
        self.threshold = threshold
        
        # Value function: V[state] = value
        self.V = np.zeros(num_states)
    
    def iterate(self) -> int:
        """
        Perform one iteration of value iteration
        
        Returns:
            Maximum change in values
        """
        V_new = np.zeros(self.num_states)
        
        for state in range(self.num_states):
            # For each state, find best action
            best_value = float('-inf')
            
            for action in range(self.num_actions):
                # Calculate expected value for this action
                action_value = 0
                
                if (state, action) in self.transitions:
                    for next_state, prob, reward in self.transitions[(state, action)]:
                        # Expected value = sum over next states of
                        # prob * (reward + discount * value_of_next_state)
                        action_value += prob * (reward + self.gamma * self.V[next_state])
                
                best_value = max(best_value, action_value)
            
            V_new[state] = best_value
        
        # Calculate change
        max_change = np.max(np.abs(V_new - self.V))
        self.V = V_new
        
        return max_change
    
    def solve(self, max_iterations: int = 1000) -> Dict[int, int]:
        """
        Solve MDP using value iteration
        
        Returns:
            Optimal policy: state -> best action
        """
        for iteration in range(max_iterations):
            max_change = self.iterate()
            
            if max_change < self.threshold:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Derive policy from values
        policy = {}
        for state in range(self.num_states):
            best_action = None
            best_value = float('-inf')
            
            for action in range(self.num_actions):
                action_value = 0
                if (state, action) in self.transitions:
                    for next_state, prob, reward in self.transitions[(state, action)]:
                        action_value += prob * (reward + self.gamma * self.V[next_state])
                
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            
            policy[state] = best_action
        
        return policy


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("Reinforcement Learning Fundamentals")
    print("=" * 80)
    
    # ========== Multi-Armed Bandit ==========
    print("\n1. Multi-Armed Bandit")
    print("-" * 80)
    
    num_arms = 5
    bandit = MultiArmedBandit(num_arms)
    
    print(f"True rewards: {bandit.true_rewards}")
    print(f"Best arm: {np.argmax(bandit.true_rewards)}")
    
    # Play 1000 times with epsilon-greedy
    total_reward = 0
    for _ in range(1000):
        arm = bandit.epsilon_greedy(epsilon=0.1)
        reward = bandit.pull_arm(arm)
        bandit.update(arm, reward)
        total_reward += reward
    
    print(f"\nAfter 1000 plays:")
    print(f"Estimates: {bandit.estimates}")
    print(f"Best arm found: {np.argmax(bandit.estimates)}")
    print(f"Total reward: {total_reward:.2f}")
    
    # ========== Q-Learning ==========
    print("\n2. Q-Learning")
    print("-" * 80)
    
    # Simple 3-state, 2-action MDP
    q_learner = QLearning(num_states=3, num_actions=2, learning_rate=0.1)
    
    # Simulate some episodes
    for episode in range(100):
        state = 0
        while state < 2:  # States 0, 1, 2 (2 is terminal)
            action = q_learner.choose_action(state)
            # Simple transition: state -> state+1, reward = 1 if action=0, 0 if action=1
            reward = 1.0 if action == 0 else 0.0
            next_state = min(state + 1, 2)
            done = (next_state == 2)
            
            q_learner.update(state, action, reward, next_state, done)
            state = next_state
    
    print("Q-values:")
    for state in range(3):
        print(f"  State {state}: {dict(q_learner.Q[state])}")
    
    policy = q_learner.get_policy()
    print(f"Policy: {policy}")
    
    # ========== Monte Carlo ==========
    print("\n3. Monte Carlo Policy Evaluation")
    print("-" * 80)
    
    mc = MonteCarlo(num_states=3, discount=0.9)
    
    # Generate some episodes
    episodes = [
        [(0, 0.0), (1, 1.0), (2, 0.0)],  # Episode 1
        [(0, 0.0), (1, 1.0), (2, 0.0)],  # Episode 2
        [(1, 1.0), (2, 0.0)],  # Episode 3 (started at state 1)
    ]
    
    mc.evaluate_policy(episodes)
    print(f"Value estimates: {mc.V}")
    
    # ========== Value Iteration ==========
    print("\n4. Value Iteration")
    print("-" * 80)
    
    # Simple 3-state MDP
    transitions = {
        (0, 0): [(1, 1.0, 1.0)],  # From state 0, action 0: go to state 1, prob=1, reward=1
        (0, 1): [(0, 1.0, 0.0)],  # From state 0, action 1: stay, prob=1, reward=0
        (1, 0): [(2, 1.0, 2.0)],  # From state 1, action 0: go to state 2, prob=1, reward=2
        (1, 1): [(1, 1.0, 0.0)],  # From state 1, action 1: stay, prob=1, reward=0
    }
    
    vi = ValueIteration(num_states=3, num_actions=2, transitions=transitions)
    policy = vi.solve()
    
    print(f"Optimal values: {vi.V}")
    print(f"Optimal policy: {policy}")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
    1. Multi-Armed Bandit: Simplest RL - explore vs exploit
    2. Q-Learning: Learn action values, off-policy, model-free
    3. Monte Carlo: Learn from complete episodes, model-free
    4. Value Iteration: Plan with known model, finds optimal values
    5. All these are building blocks for modern RL (PPO, RLHF, etc.)
    """)

