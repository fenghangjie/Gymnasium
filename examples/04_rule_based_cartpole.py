"""Lesson 4: compare a random policy with a simple rule-based policy."""

from collections.abc import Callable

import gymnasium as gym
import numpy as np


Policy = Callable[[np.ndarray, gym.Env], int]


def random_policy(observation: np.ndarray, env: gym.Env) -> int:
    return int(env.action_space.sample())


def angle_policy(observation: np.ndarray, env: gym.Env) -> int:
    pole_angle = observation[2]
    return 0 if pole_angle < 0 else 1


def run_episode(env: gym.Env, policy: Policy, seed: int) -> float:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0

    while True:
        action = policy(observation, env)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            return total_reward


def evaluate_policy(policy_name: str, policy: Policy, episodes: int = 20) -> None:
    env = gym.make("CartPole-v1")
    rewards = []

    for episode in range(episodes):
        reward = run_episode(env, policy, seed=episode)
        rewards.append(reward)

    env.close()

    print(f"{policy_name}")
    print(f"  rewards: {rewards}")
    print(f"  average reward: {np.mean(rewards):.1f}")
    print(f"  best reward: {np.max(rewards):.1f}")
    print()


def main() -> None:
    evaluate_policy("Random policy", random_policy)
    evaluate_policy("Angle policy", angle_policy)


if __name__ == "__main__":
    main()
