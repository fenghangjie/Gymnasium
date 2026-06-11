"""Lesson 2: train a small Q-learning agent on FrozenLake."""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np


def choose_action(
    q_table: np.ndarray,
    state: int,
    action_count: int,
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.randrange(action_count)
    return int(np.argmax(q_table[state]))


def train() -> np.ndarray:
    env = gym.make("FrozenLake-v1", is_slippery=False)

    state_count = env.observation_space.n
    action_count = env.action_space.n
    q_table = np.zeros((state_count, action_count))

    alpha = 0.8
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.05
    episodes = 20_000
    training_wins = 0

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False

        while not done:
            action = choose_action(q_table, state, action_count, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next_value = np.max(q_table[next_state])
            old_value = q_table[state, action]
            learned_value = reward + gamma * best_next_value
            q_table[state, action] = old_value + alpha * (learned_value - old_value)

            state = next_state

        training_wins += int(reward > 0)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if episode % 5_000 == 0:
            print(
                f"episode={episode} epsilon={epsilon:.3f} "
                f"training_wins={training_wins}"
            )

    env.close()
    return q_table


def evaluate(q_table: np.ndarray, episodes: int = 100) -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    wins = 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        wins += int(reward > 0)

    env.close()
    print(f"Win rate: {wins}/{episodes}")


def main() -> None:
    q_table = train()
    print("\nLearned Q-table:")
    print(np.round(q_table, 2))
    print()
    evaluate(q_table)


if __name__ == "__main__":
    main()
