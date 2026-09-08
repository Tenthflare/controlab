"""Interface-family randomization: the sim<->policy boundary, not the plant.

These model the parts of the reality gap MuJoCo does not: noisy/biased sensing,
delayed actuation, and external disturbances. Each reads its per-episode
parameter from `info["sampled_values"]` (set by RandomizedDynamicsWrapper) so all
randomization is drawn in one place and logged together.
"""
from __future__ import annotations
import collections
import gymnasium as gym
import numpy as np


class ObservationNoise(gym.ObservationWrapper):
    """Add Gaussian noise (std=obs_noise_std) and a fixed bias (obs_bias)."""
    def __init__(self, env, rng: np.random.Generator):
        super().__init__(env)
        self.rng = rng
        self.std = 0.0
        self.bias = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.std = info["sampled_values"]["obs_noise_std"]
        self.bias = info["sampled_values"]["obs_bias"]
        return self.observation(obs), info

    def observation(self, obs):
        # applied to every obs (ObservationWrapper wraps reset & step returns)
        return obs + self.bias + self.rng.normal(0.0, self.std, size=obs.shape)

class ActionLatency(gym.Wrapper):
    """Delay actions by L control steps via a FIFO buffer (L from xi)."""
    def __init__(self, env, rng: np.random.Generator):
        super().__init__(env)
        self.rng = rng
        self.buffer: collections.deque = collections.deque()
        self.buffer_length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.buffer_length = int(info["sampled_values"]["action_latency"])
        # prefill with L zero (no-op) actions so early steps are "delayed" too
        zero = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        self.buffer = collections.deque(zero.copy() for _ in range(self.buffer_length))
        return obs, info

    def step(self, action):
        if self.buffer_length == 0:
            return self.env.step(action)
        self.buffer.append(action)          # push newest
        delayed = self.buffer.popleft()     # pop oldest -> actually execute
        return self.env.step(delayed)


class DisturbanceForce(gym.Wrapper):
    """Occasionally apply an external push (magnitude from sampled_value) via xfrc_applied."""
    def __init__(self, env, rng: np.random.Generator,
                 prob: float = 0.01):
        super().__init__(env)
        self.rng = rng
        self.prob = prob
        self.body_id = env.unwrapped.model.body("cart").id
        self.magnitude = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.magnitude = info["sampled_values"]["disturbance_force"]
        return obs, info

    def step(self, action):
        data = self.env.unwrapped.data
        data.xfrc_applied[self.body_id] = 0.0
        if self.magnitude > 0 and self.rng.random() < self.prob:
            force = self.rng.choice([-1, 1]) * self.magnitude
            data.xfrc_applied[self.body_id, 0] = force  # force along x (slider axis)
        return self.env.step(action)