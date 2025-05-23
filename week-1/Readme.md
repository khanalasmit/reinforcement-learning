# Reinforcement Learning Notes

This repository contains notes and code from my study of Reinforcement Learning. Each week’s content is organized in Jupyter notebooks, starting with an introduction to RL concepts and progressing to key framework ideas.

## Structure

- `week-1/1-intro.ipynb` — Introduction to reinforcement learning, key concepts, and applications.
- `week-1/2-navigating_rl_framework.ipynb` — Explains the RL framework, episodic vs. continuous tasks, return, and discounted return with code examples.

## Highlights

### Reinforcement Learning Basics

- **Agent learns through trial and error**
- Receives rewards for good decisions, penalties for bad ones
- **Goal:** Maximize positive feedback over time

### RL vs Other Machine Learning Types

|                | Supervised Learning | Unsupervised Learning | Reinforcement Learning         |
|----------------|--------------------|----------------------|-------------------------------|
| **Data type**  | Labeled data       | Unlabeled data       | No predefined training data   |
| **Objective**  | Predict outcomes   | Find patterns        | Maximize reward from environment |
| **Suitability**| Classification, regression | Clustering, association | Decision-making tasks      |

### RL Applications

- **Robotics:** Walking, object manipulation
- **Finance:** Trading optimization
- **Autonomous Vehicles:** Safety and efficiency

---

### Navigating the RL Framework

- **Episodic tasks:** Segmented into episodes (e.g., chess)
- **Continuous tasks:** Ongoing interaction (e.g., traffic lights)

#### Return

- The sum of all expected rewards over time.

#### Discounted Return

- Immediate rewards are more valuable than future ones.
- Discount factor ($\gamma$) prioritizes present rewards.
- Discounted return formula:  
  $r_1 + \gamma r_2 + \gamma^2 r_3 + ... + \gamma^{n-1} r_n$

#### Example: Discounted Return Calculation

```python
import numpy as np

exp_rewards_strategy_2 = np.array([6, -5, -3, -2])
discount_factor = 0.9

# Compute discounts
discounts_strategy_2 = np.array([discount_factor**i for i in range(len(exp_rewards_strategy_2))])

# Compute the discounted return
discounted_return_strategy_2 = np.sum(exp_rewards_strategy_2 * discounts_strategy_2)

print(f"The discounted return of the second strategy is {discounted_return_strategy_2}")
```