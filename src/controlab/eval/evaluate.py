"""
Evaluate every checkpoint on the shared, fixed test set and log tidy rows.

Each policy is evaluated with horizon=1000 timesteps for
5 episodes (seeds) for each set of test param values.
There are 30 param values for each of three buckets (nominal, interpolation, extrapolation)

Primary metric is episodic return per set of param values
Secondary metric is survival rate (fraction of policies reaching the horizon
Output: one parquet with columns
    [alpha, seed, bucket, set_id, episode, return, survived]
which is all downstream stats/figures need. The test set is generated once
so every policy faces identical dynamics (paired).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from stable_baselines3 import SAC

from controlab.envs.env_wrapper import build_env
from controlab.randomisation.param_spec import ParamSpec

REPO_ROOT = Path(__file__).resolve().parents[3]


def evaluate_checkpoint(eval_config: dict) -> None:
    """Roll the frozen policy over the fixed test set; return tidy rows."""
    seed: int = eval_config["seed"]
    base_env: str = eval_config["base_env"]
    param_spec: Path = REPO_ROOT / eval_config["param_spec"]
    run_path: Path = REPO_ROOT / eval_config["run_path"]
    eval_output_path: Path = REPO_ROOT / eval_config["eval_output_path"]
    test_buckets: list[str] = eval_config["test_buckets"]
    n_per_bucket: int = eval_config["n_per_bucket"]
    mode: str = eval_config["mode"]
    horizon: int = eval_config["horizon"]
    num_episode: int = eval_config["num_episode"]

    spec = ParamSpec()
    spec.from_yaml(param_spec)
    env = build_env(
        seed=seed,
        base_env=base_env,
        path=str(param_spec),
        test_bucket="nominal",  # stand-in value
        alpha=0.0,  # stand-in value
        mode=mode,
    )
    nominals = env.get_wrapper_attr("nominals")

    runs_list = json.loads(run_path.read_text())
    test_set = spec.make_test_set(
        buckets=test_buckets, n_per_bucket=n_per_bucket, nominals=nominals
    )
    rows = []
    for run in runs_list:
        model = SAC.load(REPO_ROOT / run["checkpoint"])
        for bucket in test_buckets:
            for set_id, set_values in enumerate(test_set[bucket]):
                for episode in range(num_episode):
                    eval_base_seed = 12345
                    bucket_idx = test_buckets.index(bucket)
                    episode_seed = eval_base_seed + (
                        (bucket_idx * n_per_bucket + set_id) * num_episode + episode
                    )
                    obs, info = env.reset(
                        seed=episode_seed, options={"sampled_values": set_values}
                    )  # inject frozen dynamics
                    ep_reward, steps, done = 0.0, 0, False
                    terminated, truncated = False, False
                    for _ in range(horizon):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        ep_reward += float(reward)
                        steps += 1
                        done = terminated or truncated
                        if done:
                            break
                    rows.append(
                        {
                            "alpha": run["alpha"],
                            "seed": run["seed"],
                            "bucket": bucket,
                            "set_id": set_id,
                            "episode": episode,
                            "episode_reward": ep_reward,
                            "survived": bool(truncated and not terminated),
                        }
                    )

    pd.DataFrame(rows).to_parquet(eval_output_path)


if __name__ == "__main__":
    eval_config_ = json.loads((REPO_ROOT / "configs" / "eval_config.json").read_text())
    evaluate_checkpoint(eval_config_)
