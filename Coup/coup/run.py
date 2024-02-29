import coup_v1
import numpy as np

while True:
    env = coup_v1.env(render_mode="humans")
    env.reset(seed=42)

    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()
        print(agent)
        print(reward)
        action_mask = observation["action_mask"]
        # if reward != 0:
        #     print(f"Player 1 reward: {env.rewards['player_1']}")
        #     print(f"Player 2 reward: {env.rewards['player_2']}")

            
        if termination or truncation:
            break
        else:
            action = env.action_space(agent).sample(action_mask)
            
        env.step(action)
    

    env.close()
    break