import coup_v1
import numpy as np
env = coup_v1.env(render_mode="human")
env.reset(seed=42)
while True:
    env.step(int(input()))
    obs_d = {}
    rew_d = {}
    terminated_d = {}
    truncated_d = {}
    info_d = {}

    while env.agents:
        obs, rew, terminated, truncated, info = env.last()
        agent_id = env.agent_selection
        obs_d[agent_id] = obs
        rew_d[agent_id] = rew
        terminated_d[agent_id] = terminated
        truncated_d[agent_id] = truncated
        info_d[agent_id] = info
        if (
            env.terminations[env.agent_selection]
            or env.truncations[env.agent_selection]
        ): 
            env.step(None)
        else:
            break

    all_gone = not env.agents
    terminated_d["__all__"] = all_gone and all(terminated_d.values())
    truncated_d["__all__"] = all_gone and all(truncated_d.values())