"""A tiny custom environment for learning Gymnasium's Env API."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces


class LineWorldEnv(gym.Env):
    """Move on a line until the agent reaches the goal on the far right."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = 5, max_steps: int = 15) -> None:
        if size < 2:
            raise ValueError("size must be at least 2")

        self.size = size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(size)
        self.position = 0
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self.position = 0
        self.steps = 0
        return self.position, {"goal": self.size - 1}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1

        if action == 0:
            self.position = max(0, self.position - 1)
        else:
            self.position = min(self.size - 1, self.position + 1)

        terminated = self.position == self.size - 1
        truncated = self.steps >= self.max_steps
        reward = 1.0 if terminated else -0.01

        return self.position, reward, terminated, truncated, {"steps": self.steps}

    def render(self) -> str:
        cells = ["."] * self.size
        cells[self.position] = "A"
        cells[-1] = "G" if self.position != self.size - 1 else "A"
        return "".join(cells)

