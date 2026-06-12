"""Lesson 5: train a PPO agent on CartPole with Stable-Baselines3.

PPO is the first deep reinforcement learning algorithm worth learning for
embodied AI. This example keeps the environment simple so you can focus on the
training loop and the meaning of the important PPO parameters.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


MODEL_PATH = PROJECT_ROOT / "models" / "ppo_cartpole"
MODEL_ZIP_PATH = MODEL_PATH.with_suffix(".zip")
DEFAULT_GIF_PATH = PROJECT_ROOT / "visualizations" / "ppo_cartpole.gif"


def train_model() -> PPO:
    env = gym.make("CartPole-v1")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=100,
    )

    print("\nPPO key ideas in this example:")
    print("  Actor: the policy network that chooses left or right.")
    print("  Critic: the value network that estimates how good a state is.")
    print("  Advantage: actual result minus the critic's expectation.")
    print("  clip_range: limits how far the new policy moves from the old one.\n")

    model.learn(total_timesteps=50_000)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    env.close()

    print(f"\nSaved model to: {MODEL_PATH}.zip")
    return model


def evaluate_model(model: PPO, episodes: int = 10) -> None:
    env = Monitor(gym.make("CartPole-v1"))
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=episodes,
        deterministic=True,
    )
    env.close()

    print("\nEvaluation")
    print(f"  episodes: {episodes}")
    print(f"  mean reward: {mean_reward:.1f}")
    print(f"  reward std: {std_reward:.1f}")
    print("  CartPole-v1 is considered solved around an average reward of 475+.")


def run_one_episode(model: PPO, seed: int = 1) -> None:
    env = gym.make("CartPole-v1")
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0

    for step in range(1, 501):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print("\nOne deterministic episode")
            print(f"  steps survived: {step}")
            print(f"  total reward: {total_reward:.1f}")
            break

    env.close()


def watch_model(model: PPO, seed: int = 1) -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    observation, _ = env.reset(seed=seed)

    for _ in range(500):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    env.close()


def save_gif(model: PPO, gif_path: Path, seed: int = 1) -> None:
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    observation, _ = env.reset(seed=seed)
    frames = []

    for _ in range(500):
        frames.append(env.render())
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            frames.append(env.render())
            break

    env.close()
    gif_path.parent.mkdir(exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=30, loop=0)
    print(f"\nSaved animation to: {gif_path}")


def compare_clip_ranges() -> None:
    """Quickly compare small, normal, and large PPO clip ranges.

    This is intentionally short. It is not a rigorous benchmark, just a way to
    feel what the stage-3 idea means: PPO learns from advantage while clipping
    updates so the policy does not change too violently in one update.
    """

    results = []

    for clip_range in [0.05, 0.2, 0.4]:
        env = gym.make("CartPole-v1")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_range,
            verbose=0,
            seed=100,
        )

        model.learn(total_timesteps=15_000)
        rewards = []

        for seed in range(5):
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0

            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    rewards.append(total_reward)
                    break

        env.close()
        results.append((clip_range, np.mean(rewards), np.std(rewards), rewards))

    print("\nClip range mini experiment")
    for clip_range, mean_reward, std_reward, rewards in results:
        rounded_rewards = [round(reward, 1) for reward in rewards]
        print(
            f"  clip_range={clip_range:<4} "
            f"mean={mean_reward:>6.1f} std={std_reward:>5.1f} "
            f"rewards={rounded_rewards}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the saved PPO model instead of training again.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Open a live CartPole animation window after evaluation.",
    )
    parser.add_argument(
        "--gif",
        nargs="?",
        const=str(DEFAULT_GIF_PATH),
        help="Save a CartPole animation GIF. Optionally pass an output path.",
    )
    parser.add_argument(
        "--skip-clip-experiment",
        action="store_true",
        help="Skip the clip_range mini experiment to finish faster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.load and MODEL_ZIP_PATH.exists():
        try:
            model = PPO.load(MODEL_PATH)
            print(f"Loaded model from: {MODEL_ZIP_PATH}")
        except ModuleNotFoundError as error:
            print(f"Could not load the saved model: {error}")
            print("Training a fresh model with the current Python environment.")
            model = train_model()
    else:
        if args.load:
            print(f"No saved model found at {MODEL_ZIP_PATH}; training a new one.")
        model = train_model()

    evaluate_model(model)
    run_one_episode(model)

    if not args.skip_clip_experiment:
        compare_clip_ranges()

    if args.gif:
        save_gif(model, Path(args.gif))

    if args.watch:
        watch_model(model)


if __name__ == "__main__":
    main()
