"""
Build a fully-wrapped env for a given (alpha, mode, bucket, seed).

The ONLY place that knows how the wrappers stack. Adding a task = add a base-env
branch here; the training/eval loops stay task-agnostic.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from controlab.envs.interface_wrapper import ActionLatency, DisturbanceForce, ObservationNoise
from controlab.envs.randomiser_wrapper import RandomisedDynamicsWrapper


def build_env(
    seed: int, base_env: str, path: str, test_bucket: str, alpha: float, mode: str
) -> gym.Env:
    rng = np.random.default_rng(seed)
    env = gym.make(base_env)
    env = RandomisedDynamicsWrapper(
        env=env, path=path, alpha=alpha, rng=rng, test_bucket=test_bucket, mode=mode
    )
    env = ObservationNoise(env=env, rng=rng)
    env = ActionLatency(env=env, rng=rng)
    env = DisturbanceForce(env=env, rng=rng)
    env.reset(seed=seed)
    return env
