
import os

import ray
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from ray import tune
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env
import logging
import coup_v1
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.get_logger().setLevel('ERROR')

ray.logger.setLevel(logging.CRITICAL)


def env_creator(args):
    env = coup_v1.env()
    return env


if __name__ == "__main__":
    ray.init()
    env_name = "coup_v1"

    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    test_env = PettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy(i):
        config = {
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)

    policies = {"player_1": gen_policy(0), "player_2": gen_policy(0)}

    policy_ids = list(policies.keys())

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 10_000},
        checkpoint_freq=10,
        local_dir = os.path.expanduser('~/ray_results/coup_v1'),
        config={
            # Environment specific
            "env": env_name,
            # https://github.com/ray-project/ray/issues/10761
            "no_done_at_end": True,
            # "soft_horizon" : True,
            "num_gpus": 0,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "compress_observations": False,
            "batch_mode": 'truncate_episodes',
            "clip_rewards": False,
            "vf_clip_param": 500.0,
            "entropy_coeff": 0.01,
            # effective batch_size: train_batch_size * num_agents_in_each_environment [5, 10]
            # see https://github.com/ray-project/ray/issues/4628
            "train_batch_size": 128,  # 5000
            "rollout_fragment_length": 50,  # 100
            "sgd_minibatch_size": 100,  # 500
            "vf_share_layers": False,
            },
    )
