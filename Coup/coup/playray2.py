import coup_v2
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from train_ppo import ActionMaskModel


num_games = 10



checkpoint_path = "C:/Users/josep/ray_results\PPO\PPO_Coup_0455a_00000_0_2024-03-22_12-23-40\checkpoint_000000"

def env_creator():
    env = coup_v2.env()
    return env


ModelCatalog.register_custom_model("pa_model", ActionMaskModel)
register_env("Coup", lambda config: PettingZooEnv(env_creator()))
PPO_agent = Algorithm.from_checkpoint(checkpoint_path)



env = coup_v2.env(render_mode="human")
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
            if agent == env.possible_agents[0]:
                act = env.action_space(agent).sample(action_mask)
            else:
                act = PPO_agent.compute_single_action(obs)
        env.step(act)
env.close()