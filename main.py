"""
Entry point: load one or more configs and train a policy per config.

Usage:
    python main.py                          # default: configs/test_config.json
    python main.py configs/my_run.json      # a specific config
    python main.py configs/a.json configs/b.json   # several, in sequence
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

from controlab.train.train import train

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "randomise_interface_config.json"
RUNS_OUTPUT = REPO_ROOT / "output" / "interface_runs.json"

def load_config(path: Path) -> dict:
    return json.loads(path.read_text())

def config_hash(cfg: dict) -> str:
    """Short stable hash of a run config, for reproducibility."""
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]

def main(argv: list[str]) -> None:
    paths = [Path(a) for a in argv[1:]] or [DEFAULT_CONFIG]
    runs: list[dict] = []
    RUNS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for p in paths:
        if not p.is_absolute():
            p = REPO_ROOT / p
        full_config = load_config(p)

        alphas = list(full_config["alpha"])
        seeds = list(full_config["seed"])
        for alpha, seed in itertools.product(alphas, seeds):
            indiv_config = {
                "base_env": full_config["base_env"],
                "param_spec": full_config["param_spec"],
                "total_timesteps": full_config["total_timesteps"],
                "test_bucket": "nominal",  # unused in train
                "mode": full_config["mode"],
                "alpha": alpha,
                "seed": seed,
                "families": full_config.get("families"),
                "label": full_config.get("label", "")
            }
            print(f"=== Training {p.name}: alpha={alpha} seed={seed} ===")
            saved = train(indiv_config)

            runs.append({
                "alpha": alpha,
                "seed": seed,
                "checkpoint": str(saved.relative_to(REPO_ROOT)),
                "total_timesteps": indiv_config["total_timesteps"],
                "config_hash": config_hash(indiv_config),
            })
            RUNS_OUTPUT.write_text(json.dumps(runs, indent=2))  # incremental save
            print(f"Saved -> {saved}")

if __name__ == "__main__":
    main(sys.argv)