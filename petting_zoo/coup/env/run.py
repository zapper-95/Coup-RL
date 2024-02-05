import coup



env = coup.env(render_mode="human")
env.reset(seed=42)

#

for agent in env.agent_iter():
    actions = []

    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        fst_action = None
        break
    else:
        fst_action = env.action_space(agent).sample()

        actions.append(fst_action)
        env.update_state({agent:env.get_action_string(fst_action)})


        scd_agent = env.get_next_agent()
        scd_action = env.action_space(scd_agent).sample()


        if env.get_action_string(scd_action) == "challenge":
            actions.append(scd_action) 
            env.update_state({scd_agent:"challenge"})

        elif env.get_action_string(scd_action) == "counteract":

            actions.append(scd_action) 

            env.update_state({scd_agent:"counteract"})

            trd_action = env.action_space(agent).sample()

            if env.get_action_string(trd_action) == "challenge":
                actions.append(trd_action)
                env.update_state({agent:"challenge"})

        
 

    #print(len(actions))


    if len(actions) == 0:
        env.step(None)


    # reset to the first agent
    env.get_next_agent()
    # iterate through the actions and apply them
    for act in actions:
        
        #print(f"{env.agent_selection} is taking action {env.get_action_string(act)}")
        #env.increment_next_agent()
        # updates the current agent in the env

        env.step(act)
 
    
    # ensure it goes back to the first agent
    if len(actions) == 2:
        env.increment_next_agent()

    env.remove_actions()
    

env.close()