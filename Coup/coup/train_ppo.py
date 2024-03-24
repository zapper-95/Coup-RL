"""
Code to train a PPO agent on the Coup environment using Ray RLlib.

This code is based on the action masking example from the RLlib repository:
https://github.com/ray-project/ray/blob/master/rllib/examples/action_masking.py#L52

"""

import argparse
import os
import coup_v1
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.examples.env.action_mask_env import ActionMaskEnv
from ray.rllib.examples.rl_module.action_masking_rlm import (
    TorchActionMaskRLM,
    TFActionMaskRLM,
)
from ray.train import CheckpointConfig
from ray import tune
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.env import PettingZooEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.torch_utils import FLOAT_MAX


torch, nn = try_import_torch()
class ActionMaskModel(TorchModelV2, nn.Module):
    """Custom PyTorch model to handle action masking, and processing multi-discrete observations."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.action_space = action_space
        self.obs_space = obs_space

        self.fcnet = TorchFC(obs_space.spaces['observations'], action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        """Forward propogation to get a set of logits corresponding to actions.
        
        To prevent illegal actions, we add a large negative value to the logits of illegal actions.
        """
        # Corrected observation extraction from input_dict
        obs = input_dict["obs"]["observations"]
        action_mask = input_dict["obs"]["action_mask"]
        # Forward pass through the network
        action_logits, _ = self.fcnet({"obs": obs})
        
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)
        
        return action_logits + inf_mask, state

    def value_function(self):
        return self.fcnet.value_function()


def env_creator(render=None):
    if render:
        env = coup_v1.env(render_mode="human")
    else:
        env = coup_v1.env()
    return env


if __name__ == "__main__":

    
    ModelCatalog.register_custom_model("am_model", ActionMaskModel)

    register_env("Coup", lambda config: PettingZooEnv(env_creator()))


    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space


    config = (
        ppo.PPOConfig()
        .multi_agent(
            policies={
                "player_1": (None, obs_space, act_space, {}),
                "player_2": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .training(
            model={"custom_model": "am_model"},
            _enable_learner_api=False,
        )
        .environment(
            # random env with 100 discrete actions and 5x [-1,1] observations
            # some actions are declared invalid and lead to errors
            "Coup",
            env_config={
                "action_space": act_space,
                #"action_space": Discrete(11),
                # This is not going to be the observation space that our RLModule sees.
                # It's only the configuration provided to the environment.
                # The environment will instead create Dict observations with
                # the keys "observations" and "action_mask".
                #"observations_space": MultiDiscrete([5, 5, 2, 2, 14, 6, 6, 14, 11, 11]),
                "observation_space": obs_space["observations"]
                
            },
        )
        # We need to disable preprocessing of observations, because preprocessing
        # would flatten the observation dict of the environment before it is passed to the model.
        .experimental(
            #_enable_new_api_stack=True,
            _disable_preprocessor_api=True,            
        )
        .framework("torch")
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        )
        .rl_module(_enable_rl_module_api=False)

    )
    ray.init()

    os.makedirs("ray_results", exist_ok=True)

    tune.run(
        "PPO",
        name="PPO",
        stop={"training_iteration": 100},
        checkpoint_config= CheckpointConfig(checkpoint_at_end=True),
        config=config.to_dict(),
        local_dir= os.path.normpath(os.path.abspath("./ray_results")),
    )

    print("Finished training.")
