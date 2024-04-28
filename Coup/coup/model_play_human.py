import os
import argparse
import ray
import coup_v2
from ray.rllib.models import ModelCatalog
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel as ActionMaskModel
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from utils import get_experiment_folders, get_sorted_checkpoints, get_checkpoints_folder

def print_action_distribution(policy, action_mask, env, extra_fetches):
    dist_class = policy.dist_class
    action_dist = dist_class(extra_fetches['action_dist_inputs'], policy.model)
    probs = ["{:.2f}".format(value) for value in action_dist.dist.probs.tolist()]
    print(f"Predicted value: {extra_fetches['vf_preds']} \nProbabilities:")
    for i, prob in enumerate(probs):
        if action_mask[i] == 1:
            print(f"{env.get_action_string(i)} : {prob}")

def print_avaliable_actions(action_mask, env):
    print("Available actions:")
    for i in range(len(action_mask)):
        if action_mask[i] == 1:
            print(f"{i} {env.get_action_string(i)}")

def print_new_game():
    print("--------------------")
    print("New game")
    print("--------------------")

def env_creator():
    return coup_v2.env()

# Argument parsing
parser = argparse.ArgumentParser(description="Run a RLlib model with a custom environment")
parser.add_argument("--model_folder", type=str, default="./ray_results/PPO_decentralised/", help="Folder containing model checkpoints")
parser.add_argument("--model_number", type=int, default=-1, help="Number of the model to load")
args = parser.parse_args()

ray.shutdown()
ray.init(local_mode=True)
ModelCatalog.register_custom_model("am_model", ActionMaskModel)
register_env("Coup", lambda _: PettingZooEnv(env_creator()))
main_folder = os.path.abspath(args.model_folder)
policies = []

for model_path in get_experiment_folders(main_folder):
    try:
        path = get_sorted_checkpoints(get_checkpoints_folder(model_path))[-1]
        policies.append(Algorithm.from_checkpoint(path).get_policy(policy_id="policy"))
    except:
        print(f"No checkpoint found for {model_path}")

policy = policies[-1]
env = coup_v2.env(render_mode="human", seed=42)

while True:
    print_new_game()
    env.reset()
    env.render(display_action=False)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action_mask = observation["action_mask"]
        if termination or truncation:
            break
        elif agent == "player_1":
            act, state, extra_fetches = policy.compute_single_action(observation, action_mask)
            print_action_distribution(policy, action_mask, env, extra_fetches)
        else:
            print_avaliable_actions(action_mask, env)
            act = int(input("Enter action: "))
            while act not in range(len(action_mask)) or action_mask[act] == 0:
                print("Invalid action. Try again.")
                act = int(input("Enter action: "))
        env.step(act)
    env.close()
