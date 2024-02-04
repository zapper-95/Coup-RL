import coup
env = coup.env(render_mode="human")
env.reset(seed=42)

#

for agent in env.agent_iter():
    actions = []

    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
        actions.append(action)

        env.update_state(env.get_action_string(action), None)


        scd_agent = env.get_next_agent()
        
        scd_action = env.action_space(scd_agent).sample()


        if env.get_action_string(scd_action) == "challenge":
            actions.append(scd_action) 
            env.update_state(None, "challenge")

        elif env.get_action_string(scd_action) == "counteract":

            actions.append(scd_action) 

            env.update_state(None, "counteract")

            trd_action = env.action_space(agent).sample()

            if env.get_action_string(trd_action) == "challenge":
                actions.append(trd_action)
                #env.update_state()





        # this is where you would insert your policy
    print(len(actions))

    # reset it back to the first agent


    for act in actions:
        
        print(f"Agent {env.agent_selection} is taking action {env.get_action_string(act)}")

        env.step(action)
        env.increment_next_agent()
    
    if len(actions) == 2:
        env.increment_next_agent()

    

env.close()