"""
Perturb MuJoCo *model* parameters (the physical family) at each reset.

Nominals are captured once at init so every episode perturbs around the true
baseline, never around an already-perturbed value.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
from typing import Any
import numpy as np

from controlab.randomisation.param_spec import ParamSpec


class RandomisedDynamicsWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        path: str,
        alpha: float,
        rng: np.random.Generator,
        mode: str,
        test_bucket: str | None = None,
    ):
        super().__init__(env)
        self.alpha = alpha
        self.rng = rng
        self.mode = mode  # "train" | "test"
        self.test_bucket = test_bucket
        self.param_spec = ParamSpec()
        self.param_spec.from_yaml(path)

        model = env.unwrapped.model  # type: ignore[attr-defined]
        # resolve once: name -> the (array, index) slot to write
        self.index = {
            "pole_mass": (model.body_mass, model.body("pole").id),
            "cart_mass": (model.body_mass, model.body("cart").id),
            "pole_length": (model.geom_size, (model.geom("cpole").id, 1)),  # 2D idx
            "actuator_gain": (model.actuator_gainprm, (model.actuator("slide").id, 0)),
            "hinge_damping": (model.dof_damping, model.jnt("hinge").dofadr[0]),
            "slider_damping": (model.dof_damping, model.jnt("slider").dofadr[0]),
        }
        self.nominals = {k: float(arr[idx]) for k, (arr, idx) in self.index.items()}

    def apply_values(self, sampled_values: dict[str, float]) -> None:
        """Write sampled physical params into the MuJoCo model arrays."""
        for name, val in sampled_values.items():
            arr, idx = self.index[name]
            arr[idx] = val  # writes into the live model
        model = self.env.unwrapped.model  # type: ignore[attr-defined]
        data = self.env.unwrapped.data  # type: ignore[attr-defined]
        mujoco.mj_setConst(model, data)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None):
        if seed is not None:
            # reseed the shared rng IN PLACE so noise/latency/disturbance replay
            # identical realizations across policies -> strict pairing
            self.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        options = options or {}
        sampled_values = options.pop("sampled_values", None)
        if sampled_values is None:
            if self.mode == "train":
                sampled_values = self.param_spec.sample_train(self.rng, self.alpha, self.nominals)
            else:
                assert self.test_bucket is not None
                sampled_values = self.param_spec.sample_test(
                    self.rng, self.test_bucket, self.nominals
                )
        self.apply_values({k: v for k, v in sampled_values.items() if k in self.index})
        obs, info = self.env.reset(seed=seed, options=options or None)
        info["sampled_values"] = sampled_values  # log the realized dynamics
        return obs, info
