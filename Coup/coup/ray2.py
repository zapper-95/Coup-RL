"""Uses Ray's RLlib to train agents to play Leduc Holdem.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import ray
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from ray import tune
from registry import get_algorithm_class
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env
import logging
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from copy import deepcopy
from ray.rllib.utils.typing import ModelConfigDict, TensorType

import coup_v1

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class TorchMaskedActions(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)

        # Assuming 'observation' is a part of your observation space that represents the actual observations
        # and 'action_mask' represents the mask for available actions.
        self.orig_obs_space = obs_space.original_space["observation"]
        
        self.action_embed_model = TorchFC(
            self.orig_obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]}
        )
        
        # Apply mask
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e10, max=torch.finfo(torch.float32).max)
        masked_logits = action_logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self.action_embed_model.value_function()


def train():
    alg_name = "PPO"
    # ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)
    # function that outputs the environment you wish to register.

    def env_creator():
        env = coup_v1.env()
        return env


    num_cpus = 1
    #class_mate = get_algorithm_class(alg_name, True)
    #config = deepcopy(get_algorithm_class(alg_name).get_default_config())

    env_name = "coup_v1"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    print(obs_space)
    act_space = test_env.action_space

    # Initialize PPOConfig
    config = PPOConfig()

    # Configure training parameters and the custom model
    config = config.training(
        gamma=0.9, 
        lr=0.01, 
        kl_coeff=0.3, 
        model={
            "custom_model": "pa_model",  # Use the key you registered your custom model with
            # Add any additional model configuration here if needed
        },
        _enable_learner_api=False
    )

    # Configure resources
    config = config.resources(
        num_gpus=0,
    )

    # Configure rollouts
    config = config.rollouts(
        num_rollout_workers=1
    )

    # Configure multi-agent setup
    # Ensure 'obs_space' and 'act_space' are correctly defined for your environment
    config = config.multi_agent(
        policies={
            "player_1": (None, obs_space, act_space, {}),
            "player_2": (None, obs_space, act_space, {})
        }
        # Additional multi-agent configurations can be added here
    )

    config.rl_module( _enable_rl_module_api=False)
    # Optionally, print out the configuration to verify
    print(config.to_dict())
    ray.init(num_cpus=num_cpus + 1)

    tune.run(
        alg_name,
        name="PPO-coup",
        stop={"timesteps_total": 10000000},
        checkpoint_freq=10,
        config=config,
    )


train()

    # # Get the current working directory
    # current_working_directory = os.getcwd()

    # # Define your relative path
    # relative_path = './checkpoints/'
    # os.makedirs(relative_path, exist_ok=True)

    # # Concatenate them to form an absolute path
    # abs_checkpoint = os.path.join(current_working_directory, relative_path)
    

    # tune.run(
    #     alg_name,
    #     name="PPO",
    #     stop={"timesteps_total": 100},
    #     checkpoint_config={"checkpoint_frequency": 100, "checkpoint_at_end": True},
    #     local_dir=abs_checkpoint,
    #     config=config.to_dict(),
    # )