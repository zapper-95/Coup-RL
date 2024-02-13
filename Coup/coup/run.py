import coup_v0
import numpy as np


env = coup_v0.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():

    observation, reward, termination, truncation, info = env.last()
    action_mask = observation["action_mask"]

    if reward != 0:
        print(f"Player 1 reward: {env.rewards['player_1']}")
        print(f"Player 2 reward: {env.rewards['player_2']}")

        
    if termination or truncation:
        break
    else:
        action = env.action_space(agent).sample(action_mask)
        
    env.step(action)
 

env.close()