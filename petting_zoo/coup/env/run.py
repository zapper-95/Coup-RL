import coup
import numpy as np

action_mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int8)
challenge_mask = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
diff_mask = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8)

env = coup.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():

    observation, reward, termination, truncation, info = env.last()

    if reward != 0:
        print(f"{agent} Reward {reward}")
        
    if termination or truncation:

        action = None
    else:
        action = env.action_space(agent).sample()
        
    env.step(action)
 

env.close()