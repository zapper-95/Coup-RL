import coup_v2
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from train_ppo import ActionMaskModel
from utils import get_last_agent_path, get_penultimate_agent_path, get_nth_latest_model
import ray
import numpy as np

num_games = 1000


model_1_path = get_nth_latest_model(exp_n=1)
model_2_path = get_nth_latest_model(exp_n=2)
#model_2_path = get_last_agent_path()

def env_creator():
    env = coup_v2.env()
    return env


ModelCatalog.register_custom_model("am_model", ActionMaskModel)
register_env("Coup", lambda config: PettingZooEnv(env_creator()))

policy1 = Algorithm.from_checkpoint(model_1_path).get_policy(policy_id="policy")
policy2 = Algorithm.from_checkpoint(model_2_path).get_policy(policy_id="policy")

obs_space_1 = len(policy1.observation_space["observations"])
obs_space_2 = len(policy2.observation_space["observations"])

# k past actions is equal the max observation space minus that 
k = max(obs_space_1, obs_space_2) + coup_v2.env().k_actions - len(coup_v2.env().observation_space("player_1")["observations"])

env = coup_v2.env(render_mode=None, k_actions=k)
scores = {agent: 0 for agent in env.possible_agents}
total_rewards = {agent: 0 for agent in env.possible_agents}
round_rewards = []



# ensure policy 1's observations are equal to a slice of the environments observations
assert (env.observation_space("player_1")["observations"]
    [:obs_space_1]
    == 
    policy1.observation_space["observations"])


# ensure policy 2's observations are equal to a slice of the environments observations
assert (env.observation_space("player_2")["observations"]
    [:obs_space_2]
    == 
    policy2.observation_space["observations"])




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
            for a in env.possible_agents:
                total_rewards[a] += rewards[a]
            round_rewards.append(rewards)
            break
        else:
            fixed_obs = obs["observations"][:-k]
            if agent == env.possible_agents[0]:
                k1 = k - (len(obs["observations"]) - obs_space_1)
                action_history = obs["observations"][-k1:]

                obs["observations"] = np.concatenate([fixed_obs, action_history])
                act = policy1.compute_single_action(obs)[0]
            else:
                k2 = k - (len(obs["observations"]) - obs_space_2)
                action_history = obs["observations"][-k2:]

                obs["observations"] = np.concatenate([fixed_obs, action_history])
                act = policy2.compute_single_action(obs)[0]
        env.step(act)
env.close()

print(f"Scores: {scores}")
print(f"Total rewards: {total_rewards}")
