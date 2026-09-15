"""Train one SAC policy for a single (alpha, seed, mode) config.

`train(config)` is the reusable unit: the sweep orchestrator (main.py) calls it
once per config. It is deliberately free of argument parsing and file discovery
so it can be driven from a loop, a notebook, or a test.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from controlab.envs.env_wrapper import build_env

torch.set_num_threads(1)

# quiet TensorFlow/oneDNN chatter before SB3 imports them
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# repo root = .../controlab  (train.py is src/controlab/train/train.py)
REPO_ROOT = Path(__file__).resolve().parents[3]


def train(config: dict) -> Path:
    """Train a SAC policy from a config dict; return the saved-model path.

    All input/output paths are resolved against the repo root, so the result
    is independent of the current working directory.
    """
    seed = config["seed"]
    label = config.get("label", "")
    prefix = f"sac_{label}_" if label else "sac_"
    run_name = f"{prefix}alpha{config['alpha']}_seed{config['seed']}"
    families = set(config["families"]) if config.get("families") else None

    out_dir = REPO_ROOT / "output"
    log_dir = out_dir / "logs" / run_name
    model_dir = out_dir / "saved_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    param_spec = REPO_ROOT / config["param_spec"]

    env = build_env(
        seed=seed,
        base_env=config["base_env"],
        path=str(param_spec),
        test_bucket=config["test_bucket"],
        alpha=config["alpha"],
        mode=config["mode"],
        families=families
    )

    model = SAC("MlpPolicy", env, seed=seed, verbose=1)
    model.set_logger(configure(str(log_dir), ["csv", "tensorboard"]))
    model.learn(total_timesteps=config["total_timesteps"], log_interval=4)

    save_path = model_dir / run_name
    model.save(str(save_path))
    return save_path
