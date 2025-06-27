# Taxi Route Optimization with Reinforcement Learning

This project demonstrates how to use Q-learning, a model-free reinforcement learning algorithm, to optimize taxi routing in the classic OpenAI Gymnasium `Taxi-v3` environment. The agent learns to efficiently pick up and drop off passengers by maximizing its cumulative reward through trial and error.

---

## Overview

- **Environment:** [Taxi-v3](https://www.gymlibrary.dev/environments/toy_text/taxi/)
- **Goal:** Learn an optimal policy for picking up and dropping off passengers at the correct locations with minimal steps and penalties.
- **Approach:** Q-learning with epsilon-greedy exploration.

---

## Key Components

- **Q-table:**  
  A table of shape `(num_states, num_actions)` that stores the expected future rewards for each state-action pair.

- **Epsilon-Greedy Policy:**  
  Balances exploration (random actions) and exploitation (choosing the best-known action). Epsilon decays over time to favor exploitation as learning progresses.

- **Q-learning Update Rule:**  
  Updates the Q-table using the Bellman equation:
  ```python
  q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
- **Demostration:**
![Taxi Agent Behavior](taxi_agent_behavior.gif)