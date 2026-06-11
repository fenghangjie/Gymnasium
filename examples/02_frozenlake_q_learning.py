"""Lesson 2: train a small Q-learning agent on FrozenLake."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


SNAPSHOT_EPISODES = [0, 100, 500, 1_000, 2_500, 5_000, 10_000, 20_000]
SNAPSHOT_PATH = Path("visualizations/q_learning_training_snapshots.json")


def choose_action(
    q_table: np.ndarray,
    state: int,
    action_count: int,
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.randrange(action_count)
    return int(np.argmax(q_table[state]))


def make_snapshot(
    episode: int,
    epsilon: float,
    training_wins: int,
    q_table: np.ndarray,
) -> dict[str, Any]:
    return {
        "episode": episode,
        "epsilon": round(epsilon, 4),
        "training_wins": training_wins,
        "q_table": np.round(q_table, 4).tolist(),
    }


def print_q_table_snapshot(snapshot: dict[str, Any]) -> None:
    print(
        f"\nEpisode {snapshot['episode']} | "
        f"epsilon={snapshot['epsilon']:.4f} | "
        f"training_wins={snapshot['training_wins']}"
    )
    print(np.array(snapshot["q_table"]))


def save_snapshots(snapshots: list[dict[str, Any]]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": "FrozenLake-v1",
        "is_slippery": False,
        "actions": ["left", "down", "right", "up"],
        "map": ["SFFF", "FHFH", "FFFH", "HFFG"],
        "snapshots": snapshots,
    }
    SNAPSHOT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved training snapshots to {SNAPSHOT_PATH}")


def train() -> tuple[np.ndarray, list[dict[str, Any]]]:
    random.seed(7)
    np.random.seed(7)

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
    snapshots = [make_snapshot(0, epsilon, training_wins, q_table)]
    print_q_table_snapshot(snapshots[0])

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=episode)
        done = False
        reward = 0.0

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

        if episode in SNAPSHOT_EPISODES:
            snapshot = make_snapshot(episode, epsilon, training_wins, q_table)
            snapshots.append(snapshot)
            print_q_table_snapshot(snapshot)

    env.close()
    return q_table, snapshots


def evaluate(q_table: np.ndarray, episodes: int = 100) -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    wins = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=10_000 + episode)
        done = False
        reward = 0.0

        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        wins += int(reward > 0)

    env.close()
    print(f"Win rate: {wins}/{episodes}")


def main() -> None:
    q_table, snapshots = train()
    save_snapshots(snapshots)
    print("\nFinal learned Q-table:")
    print(np.round(q_table, 2))
    print()
    evaluate(q_table)


if __name__ == "__main__":
    main()
