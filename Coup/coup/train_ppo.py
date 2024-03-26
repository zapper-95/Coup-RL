"""
Code to train a PPO agent on the Coup environment using Ray RLlib.

This code is based on the action masking example from the RLlib repository:
https://github.com/ray-project/ray/blob/master/rllib/examples/action_masking.py#L52
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_eval.py

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
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        obs_batch_size = len(tree.flatten(obs_batch)[0])
        return (
            [self.action_space_for_sampling.sample(obs_batch["action_mask"][0]) for i in range(obs_batch_size)],
            [],
            {},
        )    

def eval_policy_vs_random(player, eval_workers):
    print(f"Evaluating for {player} against random:")
    eval_workers._policy_class = {
                "multiagent": {
                    "policy_mapping_fn": (lambda agent_id, *args, **kwargs: player if agent_id == player else "random")
                }
            }

    # Run evaluation episodes.
    for _ in range(100):
        # Running one episode per worker.
        eval_workers.foreach_worker(func=lambda w: w.sample(), local_worker=False)

    # Collect and summarize episodes into a metrics dict.
    episodes = collect_episodes(workers=eval_workers, timeout_seconds=99999)
    metrics = summarize_episodes(episodes)

    # remove as always zero, because the game is zero sum
    del metrics["hist_stats"]["episode_reward"]

    winrate = [1 if x > 0 else 0 for x in metrics["hist_stats"]["policy_player_1_reward"]]
    winrate = sum(winrate)/len(winrate)

   
    metrics["win_rate"] = winrate


    strg_rewards = [x for x in metrics["hist_stats"]["policy_player_1_reward"] if x not in [2, 1, -1, 2]]
    metrics["strg_rewards"] = strg_rewards




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

    print("policies:")

    
    # eval_workers.foreach_worker(
    # func=lambda w: w.foreach_policy(
    #     lambda policy, pid: print(pid, policy)
    # )

    #     # func=lambda w: w.foreach_env(
    #     #     lambda env: env.set_corridor_length(4 if w.worker_index == 1 else 7)
    #     # )
    # )
    
    # Evaluate player 1 policy against random
    metrics_1 = eval_policy_vs_random("player_1", eval_workers)
    metrics_2 = eval_policy_vs_random("player_2", eval_workers)


    return {"player_1": metrics_1, "player_2": metrics_2}
    # eval_workers._policy_class = {
    #             "multiagent": {
    #                 "policy_mapping_fn": (lambda agent_id, *args, **kwargs: "player_1" if agent_id == "player_1" else "random")
    #             }
    #         }

    # # Run evaluation episodes.
    # for _ in range(100):
    #     # Running one episode per worker.
    #     eval_workers.foreach_worker(func=lambda w: w.sample(), local_worker=False)

    # # Collect and summarize episodes into a metrics dict.
    # episodes = collect_episodes(workers=eval_workers, timeout_seconds=99999)
    # metrics = summarize_episodes(episodes)

    # # remove as always zero, because the game is zero sum
    # del metrics["hist_stats"]["episode_reward"]

    # winrate = [1 if x > 0 else 0 for x in metrics["hist_stats"]["policy_player_1_reward"]]
    # winrate = sum(winrate)/len(winrate)

   
    # metrics["win_rate"] = winrate


    # strg_rewards = [x for x in metrics["hist_stats"]["policy_player_1_reward"] if x not in [2, 1, -1, 2]]
    # metrics["strg_rewards"] = strg_rewards




    # print(metrics)
    # return metrics


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

    eval_fn = custom_eval_function
    ModelCatalog.register_custom_model("am_model", ActionMaskModel)

    register_env("Coup", lambda config: PettingZooEnv(env_creator()))


    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space["player_1"]
    act_space = test_env.action_space["player_1"]

    config = (
        ppo.PPOConfig()
        .multi_agent(
            policies={
                "player_1": (None, obs_space, act_space, {}),
                "player_2": (None, obs_space, act_space, {}),
                "random": (RandomPolicyActionMask, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .training(
            model={"custom_model": "am_model"},
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
            _enable_new_api_stack=False,
            _disable_preprocessor_api=True,            
        )
        .framework("torch")
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        )
        .evaluation(
            evaluation_num_workers=2,
            # Enable evaluation, once per training iteration.
            evaluation_interval=1,
            evaluation_duration=100,
            evaluation_config={
                "multiagent": {
                    "policy_mapping_fn": (lambda agent_id, *args, **kwargs: "player_1" if agent_id == "player_1" else "random")
                }
            },       
            custom_evaluation_function=eval_fn
        )

    )
    ray.init(ignore_reinit_error=True)

    os.makedirs("ray_results", exist_ok=True)

    tune.run(
        "PPO",
        name="PPO",
        stop={"training_iteration": 10},
        checkpoint_config= CheckpointConfig(checkpoint_at_end=True),
        config=config.to_dict(),
        storage_path= os.path.normpath(os.path.abspath("./ray_results")),
    )

    print("Finished training.")


    ray.shutdown()
