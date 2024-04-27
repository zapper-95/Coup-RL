import os
os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"
import coup_v2
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel as ActionMaskModel
from models import ActionMaskCentralisedCritic
from utils import get_experiment_folders, get_sorted_checkpoints, get_checkpoints_folder
import argparse





def env_creator(render_mode=None):
    env = coup_v2.env(render_mode=render_mode)
    return env



def print_obs(env, obs):
    print("Card 1: ", env.get_card(obs[0]))
    print("Card 2: ", env.get_card(obs[1]))
    print("Card 1 Alive: ", obs[2])
    print("Card 2 Alive: ", obs[3])
    print("Coins: ", obs[4])
    
    print()
    if obs[5] == 5:
        print(f"Agents Card 1: (Hidden)")
    else:
        print(f"Agents Card 1: {env.get_card(obs[5])}")

    if obs[6] == 5:
        print(f"Agents Card 2: (Hidden)")
    else:
        print(f"Agents Card 2: {env.get_card(obs[6])}")

    print(f"Agents Coins: {obs[7]}")
    print("-------------------")




if __name__ == "__main__":



    parser = argparse.ArgumentParser()


    parser.add_argument("--experiment_folder", type=str, default="./ray_results/PPO_decentralised/test_2", help="Path to the experiment folder")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--render_mode", type=str, default="human", help="Render mode for the environment")
    args = parser.parse_args()

    if not args.model_path:
        main_folder = os.path.abspath(args.experiment_folder)

        model_paths = get_experiment_folders(main_folder)
        print(model_paths)

        # take the latest checkpoint
        checkpoint_path = get_sorted_checkpoints(get_checkpoints_folder(model_paths[-1]))[-1]
    else:
        checkpoint_path = get_sorted_checkpoints(get_checkpoints_folder(args.model_path))[-1]




    render_mode = args.render_mode

    if render_mode:
        env = env_creator(render_mode)
    else:
        env = env_creator()



    # register the correct model depending if it is centralised or decentralised
    if "decentralised" in checkpoint_path:
        ModelCatalog.register_custom_model("am_model", ActionMaskModel)
    else:
        ModelCatalog.register_custom_model("am_model", ActionMaskCentralisedCritic)
    
    register_env("Coup", lambda config: PettingZooEnv(env_creator()))
    #PPO_agent = Algorithm.from_checkpoint(checkpoint_path)
    policy = Algorithm.from_checkpoint(checkpoint_path).get_policy("policy")
 


    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    while True:
        print("----- New Game -----")
        print("Player 1: Human")
        print("Player 2: AI")
        print("-------------------")
        env.reset()

        env.action_space(env.possible_agents[0])
        
        if render_mode:
            env.render(display_action=False)

        rewards = {agent: 0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()



            for a in env.agents:
                rewards[a] += env.rewards[a]   
          
            if termination or truncation:
                winner = max(env.rewards, key=env.rewards.get)
                scores[winner] += 1 # only tracks the largest reward (winner of game)
                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    total_rewards[a] += rewards[a]
                # List of rewards by round, for reference
                round_rewards.append(rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    if not render_mode:
                        print_obs(env, observation)

                    for i in range(len(action_mask)):
                        if action_mask[i] == 1:
                            print(f"Action {i}: {env.get_action_string(i)}")
                    act = int(input())
                else:


                    act, state, extra_fetches = policy.compute_single_action(obs)

                    if not render_mode == None:

                        # get the logits of the action distribution
                        dist_inputs = extra_fetches['action_dist_inputs']

                        # get the action distribution class
                        dist_class = policy.dist_class

                        # create the action distribution using the logits
                        action_dist = dist_class(dist_inputs, policy.model)


                        # print value estimate of the state
                        print(f"Predicted value: {extra_fetches['vf_preds']} \n")

                        # print the action probabilities
                        print("Probabilities:")
                        probs = action_dist.dist.probs
                        probs = ["{:.2f}".format(value) for value in probs.tolist()]
                        
                        for i in range(len(action_mask)):
                            if action_mask[i] == 1:
                                print(f"{env.get_action_string(i)} : {probs[i]}")

            if not render_mode:
                print("-------------------")
                print("ACTION: ", env.get_action_string(act))
                print("-------------------")  
            env.step(act)
    env.close()