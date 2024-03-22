"""Example showing how to use "action masking" in RLlib.

"Action masking" allows the agent to select actions based on the current
observation. This is useful in many practical scenarios, where different
actions are available in different time steps.
Blog post explaining action masking: https://boring-guy.sh/posts/masking-rl/

RLlib supports action masking, i.e., disallowing these actions based on the
observation, by slightly adjusting the environment and the model as shown in
this example.

Here, the ActionMaskEnv wraps an underlying environment (here, RandomEnv),
defining only a subset of all actions as valid based on the environment's
observations. If an invalid action is selected, the environment raises an error
- this must not happen!

The environment constructs Dict observations, where obs["observations"] holds
the original observations and obs["action_mask"] holds the valid actions.
To avoid selection invalid actions, the ActionMaskModel is used. This model
takes the original observations, computes the logits of the corresponding
actions and then sets the logits of all invalid actions to zero, thus disabling
them. This only works with discrete actions.

---
Run this example with defaults (using Tune and action masking):

  $ python action_masking.py

Then run again without action masking, which will likely lead to errors due to
invalid actions being selected (ValueError "Invalid action sent to env!"):

  $ python action_masking.py --no-masking

Other options for running this example:

  $ python action_masking.py --help
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
class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.action_space = action_space
        self.obs_space = obs_space

        # Adjusted access to the 'observations' subspace for network initialization
        obs_size = obs_space.spaces['observations'].shape[0]
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

# Make sure to register the custom model before using it in the configuration
ModelCatalog.register_custom_model("pa_model", CustomTorchModel)


def env_creator(render=None):
    if render:
        env = coup_v1.env(render_mode="human")
    else:
        env = coup_v1.env()
    return env

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=1, help="Number of iterations to train."
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    if args.framework == "torch":
        rlm_class = TorchActionMaskRLM
    else:
        rlm_class = TFActionMaskRLM


    rlm_spec = SingleAgentRLModuleSpec(module_class=rlm_class)

    ModelCatalog.register_custom_model("pa_model", CustomTorchModel)

    register_env("Coup", lambda config: PettingZooEnv(env_creator()))


    # main part: configure the ActionMaskEnv and ActionMaskModel
    config = (
        ppo.PPOConfig()
        .training(
            model={"custom_model": "pa_model"},
            _enable_learner_api=False,
        )
        .environment(
            # random env with 100 discrete actions and 5x [-1,1] observations
            # some actions are declared invalid and lead to errors
            "Coup",
            env_config={
                "action_space": Discrete(11),
                # This is not going to be the observation space that our RLModule sees.
                # It's only the configuration provided to the environment.
                # The environment will instead create Dict observations with
                # the keys "observations" and "action_mask".
                "observations_space": MultiDiscrete([5, 5, 2, 2, 14, 11, 6, 6, 14, 11]),
                
            },
        )
        # We need to disable preprocessing of observations, because preprocessing
        # would flatten the observation dict of the environment.
        .experimental(
            #_enable_new_api_stack=True,
            _disable_preprocessor_api=True,            
        )
        .framework(args.framework)
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        )
        .rl_module(_enable_rl_module_api=False)

    )

    #algo = config.build()


    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 10_000},
        checkpoint_config= CheckpointConfig(checkpoint_at_end=True),
        config=config.to_dict(),
    )
    # run manual training loop and print results after each iteration
    # for _ in range(args.stop_iters):
    #     result = algo.train()
    #     print(pretty_print(result))

    # manual test loop
    print("Finished training. Running manual test/inference loop.")
    # prepare environment with max 10 steps
    #config["env_config"]["max_episode_len"] = 10





    # env = env_creator(render=True)
    # obs, info = env.reset()
    # done = False
    # # run one iteration until done
    # print(f"Coup with {config['env_config']}")
    # while not done:
    #     obs, reward, termination, truncation, info = env.last()
    #     # strip the "player prefix from the observation"
        # for player_key, player_data in obs.items():
        #     # Assuming 'player_data' is a dictionary with 'observations' and 'action_mask'
        #     observations = player_data['observations']
        #     action_masks = player_data['action_mask']

        # observations = {"observations": observations}
        # action = algo.compute_single_action(observations)
    #     next_obs, reward, done, truncated, _ = env.step(action)
    #     # observations contain original observations and the action mask
    #     # reward is random and irrelevant here and therefore not printed
    #     #print(f"Obs: {obs}, Action: {action}")
    #     obs = next_obs

    # print("Finished successfully without selecting invalid actions.")
    # ray.shutdown()