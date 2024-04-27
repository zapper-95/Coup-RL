"""
Code to train a PPO agent on the Coup environment using Ray RLlib.

This code is based on the action masking example from the RLlib repository:
https://github.com/ray-project/ray/blob/master/rllib/examples/action_masking.py#L52
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_eval.py
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_eval.py
https://github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic.py
"""



import numpy as np
from gymnasium.spaces import Discrete
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo.ppo import PPO

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
import scenario_tests

import argparse
import os
import coup_v2
from ray.rllib.algorithms import ppo
from ray.train import CheckpointConfig
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.examples.policy.random_policy import RandomPolicy
import numpy as np
import tree  # pip install dm_tree
from typing import (
    List,
    Optional,
    Union,
)

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights, TensorStructType, TensorType
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel as ActionMaskModel
from models import ActionMaskCentralisedCritic




torch, nn = try_import_torch()

OPPONENT_OBS = "opponent_obs"


class CentralisedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralised_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")):
        assert other_agent_batches is not None
        if policy.config["enable_connectors"]:
            [(_, _, opponent_batch)] = list(other_agent_batches.values())
        else:
            [(_, opponent_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch[SampleBatch.REWARDS], policy.device
                ),
                convert_to_torch_tensor(
                    sample_batch[SampleBatch.CUR_OBS], policy.device
                ),
                convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
            )
            .cpu()
            .detach()
            .numpy()
        )
    else:
        # Policy hasn't been initialized yet, use zeros.
        print(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS]["observations"])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.REWARDS],
        train_batch[SampleBatch.CUR_OBS],
        train_batch[OPPONENT_OBS],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss


class CCPPOTorchPolicy(CentralisedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralisedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return centralised_critic_postprocessing(
            self, sample_batch, other_agent_batches, episode
        )

class CentralisedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
        return CCPPOTorchPolicy


# START OF NORMAL CODE


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


def eval_policy_tests(eval_workers:WorkerSet, metrics):
    """Custom policy tests for evaluation."""
    
    # get the policy
    policy = eval_workers.local_worker().policy_map["policy"]
    env = eval_workers.local_worker().env_creator({}).env


    scen_tests = [
    int(scenario_tests.foreign_aid_test(policy, env)),
    int(scenario_tests.coup_test(policy, env)),
    int(scenario_tests.assassinate_test(policy, env)),
    int(scenario_tests.counter_assassinate_test(policy, env)),
    int(scenario_tests.steal_test(policy, env)),
    int(scenario_tests.counter_assassinate_test_2(policy, env)),
    ]

    metrics["policy_scenario_tests_fa"] = scen_tests[0]
    metrics["policy_scenario_tests_coup"] = scen_tests[1]
    metrics["policy_scenario_tests_assassinate"] = scen_tests[2]
    metrics["policy_scenario_tests_counter_assassinate"] = scen_tests[3]
    metrics["policy_scenario_test_steal"] = scen_tests[4]
    metrics["policy_scenario_counter_assassinate_2"] = scen_tests[5]
    metrics["policy_scenario_tests_total"] = sum(scen_tests)


def custom_eval_function(algorithm, eval_workers:WorkerSet):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """

    metrics = eval_policy_vs_random(eval_workers)
    eval_policy_tests(eval_workers, metrics)
    return metrics





def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Function to allow evaluation evenly"""
    if episode.episode_id % 2:
        return "policy" if agent_id == "player_1" else "random"
    else:
        return "policy" if agent_id == "player_2" else "random"



def env_creator(k_actions=4):
    env = coup_v2.env(k_actions=k_actions)
    return env


if __name__ == "__main__":




    parser = argparse.ArgumentParser(description="Process the system type.")
    parser.add_argument("-t", "--training", default="decentralised",
                        choices=["decentralised", "centralised"],
                        help="Specify the type of training: 'decentralised' or 'centralised'")

    args = parser.parse_args()


    eval_fn = custom_eval_function
    ModelCatalog.register_custom_model("am_model", ActionMaskModel if args.training == "decentralised" else ActionMaskCentralisedCritic)

    register_env("Coup", lambda config: PettingZooEnv(env_creator()))


    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space["player_1"]
    act_space = test_env.action_space["player_1"]

    config = ppo.PPOConfig()
    config.multi_agent(
        policies={
            "policy": (None, obs_space, act_space, {}),
            "random": (RandomPolicyActionMask, obs_space, act_space, {}),
        },
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: "policy"),
        policies_to_train=["policy"],
    )
    config.training(
        model={"custom_model": "am_model"},
        entropy_coeff = 0.001,
        lr=0.001,
        sgd_minibatch_size=2048,
        train_batch_size=20_000
    )
    config.environment(
        "Coup",
        env_config={
            "action_space": act_space,
            "observation_space": obs_space["observations"]
        },
    )
    config.experimental(
        _enable_new_api_stack=False,
        _disable_preprocessor_api=True,
    )
    config.framework("torch")
    config.resources(
        num_gpus=1 if torch.cuda.is_available() else 0,
        num_cpus_per_worker=1,
    )
    config.evaluation(
        evaluation_num_workers=1,
        evaluation_interval=1,
        evaluation_config={
            "multiagent": {
                "policy_mapping_fn": policy_mapping_fn
            }
        },
        custom_evaluation_function=eval_fn
    )
    config.rollouts(num_rollout_workers=2, batch_mode="complete_episodes")
    
    ray.init(ignore_reinit_error=True, local_mode=True)

    os.makedirs("ray_results", exist_ok=True)


    stop = {
        "training_iteration": 20,
    }





    tuner = tune.Tuner(
        PPO if args.training == "decentralised" else CentralisedCritic,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            name=f"PPO_{args.training}",
            stop=stop,
            verbose=1,
            storage_path= os.path.normpath(os.path.abspath("./ray_results")),
            checkpoint_config= CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=1),
            ),

    )

    results = tuner.fit()  

    print("Finished training.")


    ray.shutdown()
