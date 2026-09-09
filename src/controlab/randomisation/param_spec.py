"""
Load the parameter spec and sample values

`family` routes the parameter to a model-edit (physical) or a wrapper (interface).
`scale` = "frac" (delta is a fraction of nominal) or "abs" (delta is absolute).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class ParamDef:
    name: str
    family: str  # "physical" | "interface"
    half_range: float  # delta_i at alpha=1
    scale: str  # "frac" | "abs"
    nominal: float | None  # None => read from model at load time (physical)
    integer: bool = False
    meta: dict | None = None  # routing info (mujoco_field/selector or wrapper)


class ParamSpec:
    """Parsed param_spec.yaml with alpha-aware sampling."""

    def __init__(self):
        self.specs: list[ParamDef] = []
        self.test_buckets: dict = {}

    def from_yaml(self, path: str | Path) -> None:
        raw = yaml.safe_load(Path(path).read_text())
        self.test_buckets = raw["test_buckets"]

        for family in ("physical", "interface"):
            for name, cfg in raw.get(family, {}).items():
                cfg = dict(cfg)  # copy so we can pop
                self.specs.append(
                    ParamDef(
                        name=name,
                        family=family,
                        half_range=float(cfg.pop("half_range")),
                        scale=cfg.pop("scale"),
                        nominal=cfg.pop("nominal", None),
                        integer=cfg.pop("integer", False),
                        meta=cfg,  # leftover = routing info
                    )
                )

    def sample_train(
        self, rng: np.random.Generator, alpha: float, nominals: dict[str, float]
    ) -> dict[str, float]:
        """Sample from U(c - alpha*delta, c + alpha*delta) per parameter.

        `nominals` supplies c_i for params whose nominal is read from the model.
        `scale == "frac"` interprets delta as a fraction of c_i; "abs" as absolute.
        """
        sampled_values: dict[str, float] = {}
        for i in range(len(self.specs)):
            param_name = self.specs[i].name
            if self.specs[i].nominal is None:
                # if None, get from Mujoco model at load time
                nominal_value = nominals[param_name]
            else:
                nominal_value = self.specs[i].nominal
            delta = (
                nominal_value * self.specs[i].half_range
                if self.specs[i].scale == "frac"
                else self.specs[i].half_range
            )
            # alpha=0 returns nominals
            if alpha == 0:
                sample = nominal_value
            else:
                delta *= alpha
                if self.specs[i].name == "obs_bias":
                    sample = rng.uniform(nominal_value - delta, nominal_value + delta)
                else:
                    sample = rng.uniform(max(0.0, nominal_value - delta), nominal_value + delta)
            # Integer params (e.g. latency) are rounded
            sample = int(sample) if self.specs[i].integer else sample
            sampled_values[param_name] = sample

        return sampled_values

    def sample_test(
        self, rng: np.random.Generator, bucket: str, nominals: dict[str, float]
    ) -> dict[str, float]:
        """
        Draw one test values for a bucket (nominal / interpolation / extrapolation).
        """
        sampled_values: dict[str, float] = {}
        for i in range(len(self.specs)):
            param_name = self.specs[i].name
            if self.specs[i].nominal is None:
                # if None, get from Mujoco model at load time
                nominal_value = nominals[param_name]
            else:
                nominal_value = self.specs[i].nominal
            delta = (
                nominal_value * self.specs[i].half_range
                if self.specs[i].scale == "frac"
                else self.specs[i].half_range
            )
            # alpha=0 returns nominals
            if bucket == "nominal":
                sample = nominal_value
            else:
                magnitude = rng.uniform(
                    self.test_buckets[bucket]["lo_mult"] * delta,
                    self.test_buckets[bucket]["hi_mult"] * delta,
                )
                # downward allowed only if it stays valid (>=0); obs_bias is signed
                if self.specs[i].name == "obs_bias" or (nominal_value - magnitude) >= 0:
                    sign = rng.choice([-1, 1])
                else:
                    sign = 1
                sample = nominal_value + sign * magnitude
            sample = int(sample) if self.specs[i].integer else sample
            sampled_values[param_name] = sample

        return sampled_values

    def make_test_set(
        self,
        buckets: list[str],
        n_per_bucket: int,
        nominals: dict[str, float],
        eval_seed: int = 12345,
    ) -> dict[str, list[dict]]:
        """Pre-sample n dynamics per bucket, once, deterministically."""
        eval_rng = np.random.default_rng(eval_seed)
        return {
            bucket: [self.sample_test(eval_rng, bucket, nominals) for _ in range(n_per_bucket)]
            for bucket in buckets
        }
