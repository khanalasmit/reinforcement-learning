# Reinforcement Learning Notes

This repository contains notes and code from my study of Reinforcement Learning. Each week’s content is organized in Jupyter notebooks, starting with an introduction to RL concepts and progressing to key framework ideas.

## Structure

- `week-1/1-intro.ipynb` — Introduction to reinforcement learning, key concepts, and applications.
- `week-1/2-navigating_rl_framework.ipynb` — Explains the RL framework, episodic vs. continuous tasks, return, and discounted return with code examples.
- `week-1/3-Interaction_with_Gymnasium_environement.ipynb` — Practical interaction with RL environments using Gymnasium, including CartPole, MountainCar, and FrozenLake.

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

---

## Interacting with Gymnasium Environments

This section demonstrates how to use the [Gymnasium](https://gymnasium.farama.org/) library to interact with classic RL environments.

### Key Gymnasium Environments

- **CartPole:** Agent must balance a pole on a moving cart.
- **MountainCar:** Agent must drive a car up a steep hill.
- **Taxi:** Agent picks up and drops off a passenger.
- **FrozenLake:** Agent navigates a frozen lake to reach a goal.

### Example: FrozenLake-v1

**Action Mapping:**
- `0`: left
- `1`: down
- `2`: right
- `3`: up

**Sample code to reach the goal:**

```python
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', render_mode='rgb_array', is_slippery=False)
state, info = env.reset(seed=42)

actions = [2, 2, 1, 1, 2, 1]  # right, right, down, down, right, down
frames = []
action_names = {0: "left", 1: "down", 2: "right", 3: "up"}

for i, action in enumerate(actions):
    state, reward, terminated, _, _ = env.step(action)
    frames.append(env.render())
    print(f"Step {i+1}: Action = {action_names[action]}")
    if terminated:
        print("Reached goal")
        break

for i, frame in enumerate(frames):
    plt.imshow(frame)
    plt.title(f"Step {i+1}")
    plt.show()
```

### Notes

- The code above demonstrates step-by-step interaction and visualization.
- Similar approaches are used for CartPole and MountainCar environments.
- The `render()` method is used to visualize the environment after each action.

---

More content will be added as the course progresses.