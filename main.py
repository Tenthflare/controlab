"""
Entry point: load one or more configs and train a policy per config.

Usage:
    python main.py                          # default: configs/test_config.json
    python main.py configs/my_run.json      # a specific config
    python main.py configs/a.json configs/b.json   # several, in sequence
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

from controlab.train.train import train

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "test_config.json"

def load_config(path: Path) -> dict:
    return json.loads(path.read_text())

def main(argv: list[str]) -> None:
    paths = [Path(a) for a in argv[1:]] or [DEFAULT_CONFIG]
    for p in paths:
        if not p.is_absolute():
            p = REPO_ROOT / p
        config = load_config(p)
        print(f"=== Training {p.name}: alpha={config['alpha']} seed={config['seed']} ===")
        saved = train(config)
        print(f"Saved -> {saved}")


if __name__ == "__main__":
    main(sys.argv)