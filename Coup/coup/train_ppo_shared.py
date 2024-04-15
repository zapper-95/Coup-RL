"""
Code to train a PPO agent on the Coup environment using Ray RLlib.

This code is based on the action masking example from the RLlib repository:
https://github.com/ray-project/ray/blob/master/rllib/examples/action_masking.py#L52
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_eval.py
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_eval.py
"""

import argparse
import os
import coup_v2
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
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.examples.policy.random_policy import RandomPolicy
from gymnasium.spaces import Box
import numpy as np
import random
import tree  # pip install dm_tree
from typing import (
    List,
    Optional,
    Union,
)

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights, TensorStructType, TensorType
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel as ActionMaskModel
from models import ActionMaskCentralizedCritic


torch, nn = try_import_torch()

class RandomPolicyActionMask(RandomPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @override(RandomPolicy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        **kwargs,
    ):
        obs_batch_size = len(tree.flatten(obs_batch)[0])
        return (
            [self.action_space_for_sampling.sample(obs_batch["action_mask"][0]) for i in range(obs_batch_size)],
            [],
            {},
        )    




def eval_policy_vs_random(eval_workers):

    # Set so environment is partially observable
    eval_workers.foreach_env(lambda env: env.env.set_training_mode(False))


    print(f"Evaluating against random:")
    # Run evaluation episodes.
    for _ in range(100):
        # Running one episode per worker.
        eval_workers.foreach_worker(func=lambda w: w.sample(), local_worker=False)

    # Collect and summarize episodes into a metrics dict.
    episodes = collect_episodes(workers=eval_workers, timeout_seconds=99999)
    metrics = summarize_episodes(episodes)

    # remove as always zero, because the game is zero sum
    del metrics["hist_stats"]["episode_reward"]

    policy_winrate = [1 if x > 0 else 0 for x in metrics["hist_stats"][f"policy_policy_reward"]]
    policy_winrate = sum(policy_winrate)/len(policy_winrate)

    metrics["policy_winrate"] = policy_winrate

    print(metrics)
    return metrics


def custom_eval_function(algorithm, eval_workers:WorkerSet):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """

    metrics= eval_policy_vs_random(eval_workers)

    return metrics



# class ActionMaskModel(TorchModelV2, nn.Module):
#     """Custom PyTorch model to handle action masking, and processing multi-discrete observations."""

#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
#         nn.Module.__init__(self)
        
#         print(model_config["fcnet_hiddens"])

#         self.action_space = action_space
#         self.obs_space = obs_space
#         self.fcnet = TorchFC(obs_space.spaces['observations'], action_space, num_outputs, model_config, name)

#     def forward(self, input_dict, state, seq_lens):
#         """Forward propogation to get a set of logits corresponding to actions.
        
#         To prevent illegal actions, we add a large negative value to the logits of illegal actions.
#         """
#         # Corrected observation extraction from input_dict
#         obs = input_dict["obs"]["observations"]
#         action_mask = input_dict["obs"]["action_mask"]
#         # Forward pass through the network
#         action_logits, _ = self.fcnet({"obs": obs})
        
#         inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)
        
#         return action_logits + inf_mask, state

#     def value_function(self):
#         return self.fcnet.value_function()


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Function to allow evaluation evenly"""
    if episode.episode_id % 2:
        return "policy" if agent_id == "player_1" else "random"
    else:
        return "policy" if agent_id == "player_2" else "random"



def env_creator(render=None):
    env = coup_v2.env(k_actions=4)
    env.set_training_mode(True)
    return env


if __name__ == "__main__":

    eval_fn = custom_eval_function
    ModelCatalog.register_custom_model("am_model", ActionMaskCentralizedCritic)

    register_env("Coup", lambda config: PettingZooEnv(env_creator()))


    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space["player_1"]
    act_space = test_env.action_space["player_1"]

    config = (
        ppo.PPOConfig()
        .multi_agent(
            policies={
                "policy": (None, obs_space, act_space, {}),
                "random": (RandomPolicyActionMask, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: "policy"),
            policies_to_train=["policy"],
        )
        .training(
            model={"custom_model": "am_model"},
            train_batch_size = 10_000,
            entropy_coeff=0.001,
            #entropy_coeff = 0.01,
            lr=0.001,
            sgd_minibatch_size=10_000,
        )
        .environment(
            "Coup",
            env_config={
                "action_space": act_space,
                "observation_space": obs_space["observations"]
                
            },
        )
        # We need to disable preprocessing of observations, because preprocessing
        # would flatten the observation dict of the environment before it is passed to the model.
        .experimental(
            _enable_new_api_stack=False,
            _disable_preprocessor_api=True,            
        )
        .framework("torch")
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus = 1 if torch.cuda.is_available() else 0,
            num_cpus_per_worker = 2,
        )
        .evaluation(
            evaluation_num_workers=2,
            # Enable evaluation, once per training iteration.
            evaluation_interval=1,
            #evaluation_duration=100,
            evaluation_config={
                "multiagent": {
                    "policy_mapping_fn": policy_mapping_fn
                    
                }
            },       
            custom_evaluation_function=eval_fn
        )
        .rollouts(num_rollout_workers=3)
    )
    
    ray.init(ignore_reinit_error=True, local_mode=True)

    os.makedirs("ray_results", exist_ok=True)

    tune.run(
        "PPO",
        name="PPO",
        stop={"training_iteration": 20},
        checkpoint_config= CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=10),
        config=config.to_dict(),
        storage_path= os.path.normpath(os.path.abspath("./ray_results")),
    )

    print("Finished training.")


    ray.shutdown()
