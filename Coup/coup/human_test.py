import coup_v1
import numpy as np


env = coup_v1.env(render_mode="human")
env.reset(seed=42)
env.render(display_action=False)
for agent in env.agent_iter():

    observation, reward, termination, truncation, info = env.last()
    action_mask = observation["action_mask"]
    #print(f"Player 1 reward: {env.rewards['player_1']}")
    #print(f"Player 2 reward: {env.rewards['player_2']}")

        
    if termination or truncation:
        break
    else:
        print(f"Player {agent}")
        for i in range(len(action_mask)):
            if action_mask[i] == 1:
                print(f"Action {i}: {env.get_action_string(i)}")
        
        act = input("Enter action: ")
        while not act.isdigit() or int(act) not in list(range(len(action_mask))) or action_mask[int(act)] == 0:
            print("Invalid action. Try again.")
            act = input("Enter action: ") 
        act = int(act)
        print()
    env.step(act)
 

env.close()