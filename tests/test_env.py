from controlab.envs.env_wrapper import build_env
from controlab.train.train import REPO_ROOT

PARAM_SPEC = REPO_ROOT / "src/controlab/randomisation/inverted_pendulum_param.yaml"


def test_train_env_runs():
    env = build_env(
        seed=1,
        base_env="InvertedPendulum-v5",
        path=str(PARAM_SPEC),
        test_bucket="nominal",
        alpha=1.0,
        mode="train",
    )
    obs, info = env.reset(seed=1)
    sv = info["sampled_values"]
    assert sv["obs_noise_std"] >= 0 and sv["action_latency"] >= 0
    for _ in range(200):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
