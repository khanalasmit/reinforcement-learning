# Markov Decision Processes (MDP) and Value Functions in Reinforcement Learning

This week’s notebooks introduce Markov Decision Processes (MDPs), state-value and action-value functions, and demonstrate their use in modeling and solving RL environments using Gymnasium.

## Contents

- **MDP Concepts:**  
  - States, actions, rewards, and transition probabilities  
  - Markov property: future state depends only on current state and action

- **State-Value Functions (V):**  
  - Definition: Expected return starting from a state and following a policy  
  - Computation using Bellman equations  
  - Policy evaluation: Iteratively updating state values under a given policy

- **Action-Value Functions (Q):**  
  - Definition: Expected return starting from a state, taking an action, then following a policy  
  - Q-values provide insight into the desirability of actions within each state  
  - Used for policy improvement by selecting actions with the highest Q-value

- **Policy Representation and Improvement:**  
  - Policies map states to actions  
  - Policy improvement: For each state, choose the action with the highest Q-value  
  - Example:
    ```python
    max_action = max(range(num_actions), key=lambda action: Q[(state, action)])
    improved_policy[state] = max_action
    ```

- **FrozenLake and CliffWalking Examples:**  
  - How to interpret states, actions, transitions, and rewards  
  - Using `env.unwrapped.P` to inspect transition probabilities  
  - Visualizing environments and extracting transition details

## Key Code Snippets

**Policy Evaluation (State-Value Update):**
```python
for state in range(num_states):
    V[state] = sum(policy[state][a] * sum(p * (r + gamma * V[next_s])
        for p, next_s, r, done in env.unwrapped.P[state][a])
        for a in range(num_actions))
```

**Policy Improvement (Greedy w.r.t Q):**
```python
for state in range(num_states):
    max_action = max(range(num_actions), key=lambda action: Q[(state, action)])
    improved_policy[state] = max_action
```

**Q-value Computation:**
```python
def compute_q_value(state, action):
    return sum(p * (r + gamma * V[next_s])
        for p, next_s, r, done in env.unwrapped.P[state][action])
```

## Notes

- The notebooks explain how to interpret the transition dictionary in Gymnasium environments.
- Examples use both FrozenLake and CliffWalking to illustrate MDP, value function, and policy improvement concepts.
- Visualization helps understand the environment’s state space.
- Value iteration and policy iteration are demonstrated with code.

---

More content and examples will be added as the course progresses.