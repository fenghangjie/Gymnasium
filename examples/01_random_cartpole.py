"""Lesson 1: interact with a Gymnasium environment using a random policy."""

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")

    observation, info = env.reset(seed=42)
    total_reward = 0.0

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation:", observation)
    print("Initial info:", info)
    print()

    for step in range(1, 501):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"step={step:03d} action={action} reward={reward:.1f} "
            f"terminated={terminated} truncated={truncated}"
        )

        if terminated or truncated:
            print(f"\nEpisode finished after {step} steps.")
            print(f"Total reward: {total_reward:.1f}")
            break

    env.close()


if __name__ == "__main__":
    main()

