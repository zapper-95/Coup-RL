import coup_v1
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from Coup.coup.train_ppo import ActionMaskModel
from utils import get_last_agent_path


num_games = 1000



checkpoint_path = get_last_agent_path()

def env_creator():
    env = coup_v1.env()
    return env


ModelCatalog.register_custom_model("am_model", ActionMaskModel)
register_env("Coup", lambda config: PettingZooEnv(env_creator()))
PPO_agent = Algorithm.from_checkpoint(checkpoint_path)



env = coup_v1.env(render_mode=None)
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
                act = PPO_agent.compute_single_action(obs, policy_id="player_2")
        env.step(act)
env.close()

print(f"Scores: {scores}")
print(f"Total rewards: {total_rewards}")
