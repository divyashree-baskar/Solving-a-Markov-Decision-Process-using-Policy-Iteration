# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python


import gymnasium as gym
import numpy as np

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=True)
env = env.unwrapped

# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------
def policy_evaluation(env, policy, gamma=0.95, theta=1e-6):
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    while True:
        delta = 0

        for s in range(n_states):
            v = 0

            action = policy[s]

            for prob, next_state, reward, done in env.P[s][action]:
                v += prob * (reward + gamma * V[next_state])

            delta = max(delta, abs(V[s] - v))
            V[s] = v

        if delta < theta:
            break

    return V



# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------
def policy_improvement(env, V, gamma=0.95):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        action_values = np.zeros(n_actions)

        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state]
                )

        policy[s] = np.argmax(action_values)

    return policy
#-------------------------------------------------
# Policy Iteration
# -------------------------------------------------

def policy_iteration(env, gamma=0.95, theta=1e-6):
    n_states = env.observation_space.n

    # Random initial policy
    policy = np.random.randint(env.action_space.n, size=n_states)

    iterations = 0

    while True:
        iterations += 1

        # Policy Evaluation
        V = policy_evaluation(env, policy, gamma, theta)

        # Policy Improvement
        new_policy = policy_improvement(env, V, gamma)

        # Check for convergence
        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return policy, V, iterations

   optimal_policy, optimal_value, iterations = policy_iteration(env)

# Action Symbols
action_names = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

print("Total policy iterations:", iterations)

print("\nOptimal State-Value Function:\n")
print(np.round(optimal_value.reshape(4, 4), 3))

print("\nOptimal Policy:\n")

policy_grid = np.array(
    [action_names[a] for a in optimal_policy]
).reshape(4, 4)

print(policy_grid)


```

## Output

```
Name : DIVYASHREE B
Reg no: 212224040081

Total policy iterations: 5

Optimal State-Value Function:
[[0.180464 0.154751 0.153473 0.132544]
 [0.208961 0.       0.176429 0.      ]
 [0.270453 0.374649 0.403671 0.      ]
 [0.       0.508978 0.723673 0.      ]]


Optimal Policy:
[['←' '↑' '←' '↑']
 ['←' '←' '←' '←']
 ['↑' '↓' '←' '←']
 ['←' '→' '↓' '←']]

```
<img width="536" height="337" alt="image" src="https://github.com/user-attachments/assets/fbbfb68a-95df-462d-b56d-58895d6e0ef8" />

Policy Evaluation Value Function:


<img width="582" height="235" alt="image" src="https://github.com/user-attachments/assets/6d1d53db-2cf3-47c6-8349-a4b06afd114f" />





## Result

```text
The Policy Iteration algorithm successfully found the optimal state-value function and the optimal policy for the FrozenLake-v1 environment.




```
---

## Inference
```text
Policy Iteration converged to the optimal policy after a few iterations by repeatedly evaluating and improving the policy.


```
---

