"""Lesson 3: use a custom Gymnasium environment."""

from __future__ import annotations

import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from line_world_env import LineWorldEnv


def main() -> None:
    env = LineWorldEnv(size=7, max_steps=20)
    observation, info = env.reset(seed=7)
    total_reward = 0.0

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Start:", observation, info)
    print(env.render())
    print()

    for step in range(1, 21):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"step={step:02d} action={action} obs={observation} reward={reward}")
        print(env.render())
        print()

        if terminated or truncated:
            print("Finished:", {"terminated": terminated, "truncated": truncated})
            break

    print(f"Total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()

