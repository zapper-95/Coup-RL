"""
Trains a model to play Coup using PPO, by playing it against itself.
The model is then evaluated against a random agent.


Adapted from pettingzoo tutorial, originally for Connect 4: 
Source: https://pettingzoo.farama.org/tutorials/sb3/connect_four/
Author: Elliot (https://github.com/elliottower)
"""
import glob
import os
import time
import argparse
import random

from ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils
import coup_v1


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        super().step(action)
        return super().last()

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]
        


def mask_fn(env):
    return env.action_mask()


def train_action_mask(env_fn, steps=10_000, seed=0):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    #env = env_fn.env(render_mode="human")
    env = env_fn.env(render_mode=None)
    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning.
    model = MaskablePPO(MaskableActorCriticPolicy, env, ent_coef=0, verbose=1, tensorboard_log="logs/PPO_New")
    model.set_random_seed(seed)

    model_name = f"models/{env.unwrapped.metadata.get('name')}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    model.learn(total_timesteps=steps, progress_bar=True, tb_log_name=model_name.removeprefix("models/"))


    model.save(model_name)

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()
    return model_name 

def get_latest_model(env):
    try:
        # uses creation time on file metadata to find the latest model
        latest_model = max(
            glob.glob(f"models/*_{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("No models found.")
        exit(0)
    return latest_model


def get_best_model(env):
    try:
        # Get all model files for the specific environment
        model_files = glob.glob(f"models/*_{env.metadata['name']}_*.zip")
        
        # Extract win rates and associate them with their file paths
        winrate_file_pairs = []
        for file_path in model_files:
            # Extract the filename from the path and split it by underscores
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            # Assuming the win rate is always the first part of the filename
            win_rate = float(parts[0])
            winrate_file_pairs.append((win_rate, file_path))
        
        # Find the file with the highest win rate
        if not winrate_file_pairs:
            raise ValueError("No models found.")
        
        _, best_model = max(winrate_file_pairs, key=lambda x: x[0])
        
    except ValueError:
        print("No models found.")
        exit(0)
    
    return best_model

def eval_random_vs_trained(env_fn, num_games=100, model_name=None, render_mode=None):
    # Evaluate a trained agent vs a random agent
    render_mode = "human"
    env = env_fn.env(render_mode=render_mode)

    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[1]}."
    )


    if model_name == "latest":
        model_name = get_latest_model(env)
    elif model_name == "best" or model_name == None:
        model_name = get_best_model(env)
    else:
        
        if not model_name.endswith(".zip"):
            model_name += ".zip"
        
        if "models/" not in model_name:
            model_name = f"models/{model_name}"
        
    try:
        print(f"Loading model: {model_name}")
        model = MaskablePPO.load(model_name)
    except ValueError:
        print("Model not found.")
        exit(0)    

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset()
        env.action_space(env.possible_agents[0]).seed(i)
        
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
                print(winner)
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample(action_mask)
                else:
                    act = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )
            env.step(act)
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / num_games
    #print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores

def rename_model(env_fn, winrate):
    env = env_fn.env(render_mode=None)
    try:
        latest_policy = max(
            glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    # extract the directory and the filename
    directory, filename = os.path.split(latest_policy)

    # construct the new filename with win rate
    new_filename = f"{directory}/{winrate}_{filename}"
    
    os.rename(latest_policy, new_filename)

def test_human(env_fn, num_games=1, model_name=None, render_mode=None):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode)

    agent_index = random.randint(0, 1)

    print("Starting game against human.")
    print(f"Human will play as {env.possible_agents[1-agent_index]}.")
    print(f"Trained agent will play as {env.possible_agents[agent_index]}.")


    if model_name == "latest":
        model_name = get_latest_model(env)
    elif model_name == "best" or model_name == None:
        model_name = get_best_model(env)
    else:
        model_name = f"models/{model_name}.zip"
    try:
        print(f"Loading model: {model_name}")
        model = MaskablePPO.load(model_name)
    except ValueError:
        print("Model not found.")
        exit(0)    

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset()
        env.action_space(env.possible_agents[0]).seed(i)
        
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
                if agent == env.possible_agents[1-agent_index]:

                    if not render_mode:
                        print("-------------------")
                        print("OBSERVATIONS")
                        # shallow copy
                        raw_obs = obs["observation"][:]

                        print(f"Card 1: {env.get_card(raw_obs[0])}")
                        print(f"Card 2: {env.get_card(raw_obs[1])}")
                        print(f"Card 1 Alive: {bool(raw_obs[2])}")
                        print(f"Card 2 Alive: {bool(raw_obs[3])}")
                        print(f"Coins: {raw_obs[4]}")
                        print(f"Previous action: {env.get_action_string(raw_obs[5])}")

                        print()
                        if raw_obs[6] == 5:
                            print(f"Agents Card 1: (Hidden)")
                        else:
                            print(f"Agents Card 1: {env.get_card(raw_obs[6])}")

                        if raw_obs[7] == 5:
                            print(f"Agents Card 2: (Hidden)")
                        else:
                            print(f"Agents Card 2: {env.get_card(raw_obs[7])}")

                        print(f"Agents Coins: {raw_obs[8]}")
                        print(f"Agents Previous action: {env.get_action_string(raw_obs[9])}")
                        print("-------------------")
                        print()
                    print("Human's turn")

                    for i in range(len(action_mask)):
                        if action_mask[i] == 1:
                            print(f"Action {i}: {env.get_action_string(i)}")
                    
                    act = input("Enter action: ")
                    while not act.isdigit() or int(act) not in list(range(len(action_mask))) or action_mask[int(act)] == 0:
                        print("Invalid action. Try again.")
                        act = input("Enter action: ") 
                    act = int(act)
                    print()
                else:
                    act = int(
                        # don't want determinism here
                        model.predict(
                            observation, action_masks=action_mask, deterministic=False
                        )[0]
                    )
            env.step(act)
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[agent_index]] / num_games
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores
if __name__ == "__main__":
        # For 100 games, the agent needs to win 58/100 or more to have a p value of 0.05 or less
    # For 1,000 games, the agent needs to win 526/100 or more to have a p value of 0.05 or less
    #eval_action_mask(env_fn, num_games=2, render_mode="human", **env_kwargs)

    
    
    env_fn = coup_v1

    
    # create the parser
    parser = argparse.ArgumentParser(description="SB3 Command Line Interface")

    # create a subparser object
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('-s', '--steps', type=int, default=20480, help='Number of training steps')

    parser_test = subparsers.add_parser('test', help='Test the model')
    parser_test.add_argument('-m', '--model_name', type=str, help='Name of the model to test')


    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate the model against a random agent')
    parser_evaluate.add_argument('-m', '--model_name', type=str, required=False, help='Name of the model to evaluate')



    parser_test_human = subparsers.add_parser('test_human', help='Test the model with human interaction')
    parser_test_human.add_argument('-r', '--real', action='store_true', help='Use real interaction mode')
    parser_test_human.add_argument('-m', '--model_name', type=str, help='Name of the model to test')
    
    # parse the arguments from the command line
    args = parser.parse_args()

    if args.command == 'train':
        model_name = train_action_mask(env_fn, steps=args.steps, seed=0)
        # evaluate 1,000 games against a random agent
        _, _, winrate, _ = eval_random_vs_trained(env_fn, num_games=1_000, render_mode="None", model_name=model_name)
        # rename model to include the winrate
        rename_model(env_fn, winrate=winrate)
    elif args.command == 'test':
        if args.model_name:
            eval_random_vs_trained(env_fn, num_games=1, render_mode="human", model_name=args.model_name)
        else:
            eval_random_vs_trained(env_fn, num_games=1, render_mode="human")
    elif args.command == 'evaluate':
            eval_random_vs_trained(env_fn, num_games=1_000, render_mode=None, model_name=args.model_name)
    elif args.command == 'test_human':   
        if args.real:
            render_mode = None
        else:
            render_mode = "human"

        test_human(env_fn, num_games=1, render_mode=render_mode, model_name=args.model_name)
    else:
        parser.print_help()