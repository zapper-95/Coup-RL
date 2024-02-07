import coup
import numpy as np

action_mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int8)
challenge_mask = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)

env = coup.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    actions = []

    observation, reward, termination, truncation, info = env.last()


    if reward != 0:
        print(f"Reward {reward}")
        
    if termination or truncation:
        fst_action = None
        actions.append(None)

        # iterate to the correct agent for the next block
        env.get_next_agent()

    else:
        fst_action = env.action_space(agent).sample(action_mask)
        actions.append(fst_action)
        env.update_state({agent:env.get_action_string(fst_action)})


        scd_agent = env.get_next_agent()
        scd_action = env.action_space(scd_agent).sample(challenge_mask)

        if env.get_action_string(scd_action) == "challenge":
            actions.append(scd_action) 
            env.update_state({scd_agent:"challenge"})

        elif env.get_action_string(scd_action) == "counteract":

            actions.append(scd_action) 

            env.update_state({scd_agent:"counteract"})

            trd_action = env.action_space(agent).sample(challenge_mask)

            if env.get_action_string(trd_action) == "challenge":
                actions.append(trd_action)
                env.update_state({agent:"challenge"})



        if len(actions) == 2:
            if env.get_action_string(actions[1]) == "challenge":
                if env.action_legal(agent, fst_action):
                    env.loose_card(scd_agent)
                else:
                    env.loose_card(agent)
                    actions.pop(0)
            else:
                if not env.can_counteract(fst_action):
                    actions.pop()
                else:
                    actions.pop(0)
            
        elif len(actions) == 3:
            if env.can_counteract(fst_action):
                if env.counteraction_legal(fst_action, scd_agent):
                    env.loose_card(agent)
                    actions.pop(0)
                else:
                    env.loose_card(scd_agent)





    # reset to the first agent
    env.get_next_agent()
    # iterate through the actions and apply them
    for act in actions:
        env.step(act)
 
    
    # ensure it goes back to the first agent
    if len(actions) == 2:
        env.increment_next_agent()

    env.remove_actions()
    

env.close()