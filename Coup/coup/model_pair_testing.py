"""Scripts to play all policies in a folder pairwise against each and and record the results in
a pandas dataframe. The results should then be displayed as a heatmap.
"""

import coup_v2
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel as ActionMaskModel
from models import ActionMaskCentralisedCritic
import ray
from utils import get_experiment_folders, get_checkpoints_folder
import argparse
import pandas as pd

def play_games(policy_1, policy_2, env, num_games):
    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []
    
    for i in range(num_games):
        env.reset()
        done = False
        while not done:
            actions = {}
            for agent in env.possible_agents:
                if agent == "player_1":
                    actions[agent] = policy_1.compute_single_action(env.get_observation(agent), env.get_info(agent))
                else:
                    actions[agent] = policy_2.compute_single_action(env.get_observation(agent), env.get_info(agent))
            obs, rewards, done, info = env.step(actions)
            for agent in env.possible_agents:
                total_rewards[agent] += rewards[agent]
            if done:
                for agent in env.possible_agents:
                    scores[agent] += 1 if rewards[agent] == 1 else 0
                round_rewards.append(total_rewards.copy())
    print(scores)
    print(round_rewards)
    return scores, round_rewards


scores_df = pd.DataFrame(columns=["model_1", "model_2", "winrate"])
rewards_df = pd.DataFrame(columns=["model_1", "model_2", "avg_rewards"])


num_games = 100
# use arg parsing to get the local directory to the folder containing the models
argparser = argparse.ArgumentParser()
argparser.add_argument("--models_dir", type=str, default="./ray_results/PPO_decentralised", help="Directory containing the models")


model_names = get_experiment_folders(argparser.parse_args().models_dir)

print("model_names")

for model_1 in range(len(model_names)):
    for model_2 in range(len(model_names)):
        
        # dont want to play the games twice
        if model_1 < model_2:
            
            # load both models using the respective model path
            try:
                model_1_path = get_checkpoints_folder(model_names[model_1])
                model_2_path = get_checkpoints_folder(model_names[model_2])

                policy_1 = Algorithm.from_checkpoint(model_1_path).get_policy(policy_id="policy")
                policy_2 = Algorithm.from_checkpoint(model_2_path).get_policy(policy_id="policy")
                
                env = coup_v2.env()

                scores, round_reward = play_games(policy_1, policy_2, env, num_games)
                scores_df = scores_df.append({"model_1": model_names[model_1], "model_2": model_names[model_2], "winrate": scores["player_1"] / num_games}, ignore_index=True)
                rewards_df = rewards_df.append({"model_1": model_names[model_1], "model_2": model_names[model_2], "avg_rewards": round_reward["player_1"] / num_games}, ignore_index=True)
            except:
                print("Error loading models. Skipping.")
                pass
        
        #print(scores_df)